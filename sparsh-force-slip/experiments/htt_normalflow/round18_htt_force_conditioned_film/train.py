#!/usr/bin/env python3
"""Round-18 frozen-cache V0/C0/M0 trainer with exact resumability."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("XFORMERS_DISABLED", "1")

import torch
from torch import nn

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

GROUPS = ("V0", "C0", "M0")
SEEDS = (20260914, 20260915, 20260916)
ROLES = ("fit", "selection", "calibration", "validation")
FORMAL_MAX_EPOCHS = 60
FORMAL_PATIENCE = 10
BATCH_SIZE = 256


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_hash(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    os.close(fd)
    try:
        Path(tmp).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_save(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    os.close(fd)
    try:
        torch.save(value, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def module_seed(seed: int, name: str) -> int:
    return int(hashlib.sha256(f"round18:{seed}:{name}".encode()).hexdigest()[:8], 16)


class SlipHead(nn.Module):
    """Nine-step visual model, concatenation model, or force-conditioned FiLM model."""

    def __init__(self, group: str):
        super().__init__()
        if group not in GROUPS:
            raise ValueError(group)
        self.group = group
        self.visual_ln = nn.LayerNorm(192)
        self.gru = nn.GRU(195 if group == "C0" else 192, 128, batch_first=True)
        self.risk = nn.Linear(128, 1)
        if group == "M0":
            self.film_hidden = nn.Linear(3, 14)
            self.film_out = nn.Linear(14, 384)
            nn.init.zeros_(self.film_out.weight)
            nn.init.zeros_(self.film_out.bias)

    def conditioned_visual(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        visual = self.visual_ln(x[..., :192])
        if self.group != "M0":
            zero = visual.new_zeros((*visual.shape[:-1], 192))
            return visual, zero, zero
        gamma, beta = self.film_out(torch.nn.functional.gelu(self.film_hidden(x[..., 192:195]))).chunk(2, dim=-1)
        return (1.0 + gamma) * visual + beta, gamma, beta

    def components(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3 or x.shape[1:] != (9, 195):
            raise ValueError(f"expected [N,9,195], got {tuple(x.shape)}")
        visual, gamma, beta = self.conditioned_visual(x)
        fused = torch.cat((visual, x[..., 192:195]), dim=-1) if self.group == "C0" else visual
        hidden = self.gru(fused)[0][:, -1]
        return {"logit": self.risk(hidden).squeeze(-1), "gamma": gamma, "beta": beta}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.components(x)["logit"]


def init_model(group: str, seed: int) -> SlipHead:
    model = SlipHead(group)
    # Shared names use the same independent random stream. M0's zero output is restored last.
    for name in ("visual_ln", "gru", "risk", "film_hidden"):
        if not hasattr(model, name):
            continue
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(module_seed(seed, name))
            for parameter in getattr(model, name).parameters():
                if parameter.ndim >= 2:
                    nn.init.xavier_uniform_(parameter)
                elif name == "visual_ln" and parameter.shape == (192,):
                    if "weight" in next(k for k, v in getattr(model, name).named_parameters() if v is parameter):
                        nn.init.ones_(parameter)
                    else:
                        nn.init.zeros_(parameter)
                else:
                    nn.init.zeros_(parameter)
    if group == "M0":
        nn.init.zeros_(model.film_out.weight)
        nn.init.zeros_(model.film_out.bias)
    return model


def validate_cache(data: dict, path: Path, fold: int, seed: int) -> None:
    if data.get("schema") != "round17_common_cache_v1" or data.get("fold") != fold or data.get("seed") != seed:
        raise ValueError("cache identity mismatch")
    seen: dict[str, set[str]] = {}
    for role in ROLES:
        item = data["roles"][role]
        if item["x"].shape[1:] != (9, 195) or item["x"].dtype != torch.float32 or not torch.isfinite(item["x"]).all():
            raise ValueError(f"bad {role} input")
        if (item["t"] < 13).any() or not torch.isin(item["stage"], torch.tensor([0, 1, 2])).all():
            raise ValueError(f"bad {role} time/stage")
        groups = set(item["leakage_group"])
        if any(groups & old for old in seen.values()):
            raise ValueError(f"role leakage at {role}")
        seen[role] = groups
    upstream = data["provenance"]["immutable_upstream"]
    for name, record in upstream.items():
        if sha(Path(record["path"])) != record["sha256"]:
            raise ValueError(f"immutable upstream hash mismatch: {name}")
    if "test" in data["roles"]:
        raise ValueError("test role forbidden")
    if not path.is_file():
        raise ValueError("cache missing")


def fit_assets(data: dict) -> tuple[dict[str, torch.Tensor], torch.Tensor, list[dict]]:
    fit = data["roles"]["fit"]
    primary = fit["stage"].eq(0) | fit["stage"].eq(2)
    x = fit["x"][primary].double()
    normalizer = {
        "mean": x.reshape(-1, 195).mean(0).float(),
        "std": x.reshape(-1, 195).std(0, unbiased=False).clamp_min(1e-6).float(),
    }
    n = int(primary.sum())
    weights = torch.zeros(len(primary), dtype=torch.float64)
    records: list[dict] = []
    for cls in (0, 2):
        ids = torch.nonzero(fit["stage"].eq(cls), as_tuple=False).flatten()
        episodes = np.asarray(fit["episode_id"], object)[ids.numpy()]
        unique = sorted(set(episodes.tolist()))
        if not unique:
            raise ValueError(f"fit class {cls} unsupported")
        for episode in unique:
            local = ids[torch.from_numpy(episodes == episode)]
            value = n / (2 * len(unique) * len(local))
            weights[local] = value
            records.append({"class": cls, "episode": episode, "endpoints": len(local), "weight": value, "contribution": value * len(local)})
    if not torch.isclose(weights.sum(), torch.tensor(float(n), dtype=torch.float64), rtol=1e-10, atol=1e-8):
        raise ValueError("weight normalization")
    return normalizer, weights.float(), records


def normalize(x: torch.Tensor, normalizer: dict[str, torch.Tensor]) -> torch.Tensor:
    result = (x - normalizer["mean"]) / normalizer["std"]
    if not torch.isfinite(result).all():
        raise ValueError("nonfinite normalized input")
    return result


def low_fpr_auc(stage: torch.Tensor, probability: torch.Tensor, limit: float = 0.1) -> float:
    labels = (stage == 2).numpy().astype(np.int64)
    scores = probability.numpy().astype(np.float64)
    if set(labels.tolist()) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError("selection lacks two finite classes")
    order = np.argsort(-scores, kind="stable")
    labels = labels[order]
    scores = scores[order]
    ends = np.r_[np.flatnonzero(scores[1:] != scores[:-1]), len(scores) - 1]
    tpr = np.r_[0.0, np.cumsum(labels)[ends] / labels.sum()]
    fpr = np.r_[0.0, np.cumsum(1 - labels)[ends] / (len(labels) - labels.sum())]
    keep = fpr < limit
    xs = np.r_[fpr[keep], limit]
    ys = np.r_[tpr[keep], np.interp(limit, fpr, tpr)]
    return float(np.trapezoid(ys, xs) / limit)


def infer(model: nn.Module, x: torch.Tensor, device: str) -> torch.Tensor:
    model.eval()
    chunks = []
    with torch.inference_mode():
        for ids in torch.arange(len(x)).split(1024):
            chunks.append(torch.sigmoid(model(x[ids].to(device))).cpu())
    return torch.cat(chunks)


def train(args: argparse.Namespace) -> dict:
    if args.group not in GROUPS or args.seed not in SEEDS or args.fold not in range(1, 5):
        raise ValueError("grid identity")
    if args.execute_formal:
        if args.max_epochs != FORMAL_MAX_EPOCHS or args.patience != FORMAL_PATIENCE:
            raise ValueError("formal budget override forbidden")
        if "formal" not in args.output.parts or "smoke" in args.output.parts:
            raise ValueError("formal/smoke path separation")
    else:
        if "formal" in args.output.parts:
            raise ValueError("formal requires --execute-formal")
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    validate_cache(data, args.data, args.fold, args.seed)
    data_sha = sha(args.data)
    normalizer, weights, weight_records = fit_assets(data)
    x = {role: normalize(data["roles"][role]["x"], normalizer) for role in ROLES}
    fit = data["roles"]["fit"]
    primary = torch.nonzero(fit["stage"].eq(0) | fit["stage"].eq(2), as_tuple=False).flatten()
    target = fit["stage"].eq(2).float()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = init_model(args.group, args.seed).to(args.device)
    initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed + 1800)
    protocol = Path(__file__).with_name("PROTOCOL.md")
    identity = {
        "schema": "round18_run_identity_v1",
        "group": args.group,
        "fold": args.fold,
        "seed": args.seed,
        "input_path": str(args.data.resolve()),
        "input_sha256": data_sha,
        "source_sha256": sha(Path(__file__)),
        "protocol_sha256": sha(protocol),
        "normalizer_sha256": hashlib.sha256(normalizer["mean"].numpy().tobytes() + normalizer["std"].numpy().tobytes()).hexdigest(),
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "batch_size": BATCH_SIZE,
        "selector": "internal_selection_pAUC_0_0.1_earliest_strict_max",
        "formal": bool(args.execute_formal),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "config.json", identity)
    atomic_json(args.output / "TRAIN_WEIGHTS.json", {"schema": "round18_class_trial_weights_v1", "fit_static_gross_endpoints": len(primary), "records": weight_records})

    latest = args.output / "latest.pth"
    best_path = args.output / "best.pth"
    start = 0
    best = -float("inf")
    best_epoch = -1
    wait = 0
    history: list[dict] = []
    optimizer_steps = 0
    if latest.exists():
        checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
        if checkpoint["identity"] != identity or checkpoint["initial_state_sha256"] != state_hash(initial):
            raise ValueError("resume identity mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(args.device)
        generator.set_state(checkpoint["generator"])
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        start = checkpoint["epoch"] + 1
        best = checkpoint["best_metric"]
        best_epoch = checkpoint["best_epoch"]
        wait = checkpoint["wait"]
        history = checkpoint["history"]
        optimizer_steps = checkpoint["optimizer_steps"]

    began = time.perf_counter()
    for epoch in range(start, args.max_epochs):
        if wait >= args.patience:
            break
        model.train()
        permutation = primary[torch.randperm(len(primary), generator=generator)]
        losses = []
        for ids in permutation.split(BATCH_SIZE):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x["fit"][ids].to(args.device))
            loss = (nn.functional.binary_cross_entropy_with_logits(logits, target[ids].to(args.device), reduction="none") * weights[ids].to(args.device)).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite loss")
            loss.backward()
            if any(parameter.grad is None or not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                raise FloatingPointError("missing/nonfinite gradient")
            optimizer.step()
            optimizer_steps += 1
            losses.append(float(loss.detach()))
        selection = data["roles"]["selection"]
        mask = selection["stage"].eq(0) | selection["stage"].eq(2)
        probability = infer(model, x["selection"], args.device)
        metric = low_fpr_auc(selection["stage"][mask], probability[mask])
        improved = metric > best
        if improved:
            best = metric
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
        history.append({"epoch": epoch, "fit_loss": float(np.mean(losses)), "selection_pAUC_0_0.1": metric})
        checkpoint = {
            "identity": identity,
            "model": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "generator": generator.get_state(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "normalizer": normalizer,
            "normalizer_fit_role": "fit_static_gross_only",
            "epoch": epoch,
            "best_metric": best,
            "best_epoch": best_epoch,
            "wait": wait,
            "history": history,
            "optimizer_steps": optimizer_steps,
            "initial_state_sha256": state_hash(initial),
        }
        atomic_save(latest, checkpoint)
        if improved:
            atomic_save(best_path, checkpoint)
        print(json.dumps({"epoch": epoch, "loss": history[-1]["fit_loss"], "selection_pAUC": metric, "best_epoch": best_epoch}), flush=True)
        if args.interrupt_after is not None and epoch >= args.interrupt_after:
            result = {"status": "interrupted", "epoch": epoch, "identity": identity}
            atomic_json(args.output / "summary.json", result)
            return result

    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    changes = {name: not torch.equal(initial[name], value) for name, value in best_checkpoint["model"].items()}
    result = {
        "schema": "round18_training_summary_v1",
        "status": "complete",
        "group": args.group,
        "fold": args.fold,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_metric": best,
        "epochs": len(history),
        "optimizer_steps": optimizer_steps,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "parameter_changes_at_best": changes,
        "all_parameters_structurally_active": True,
        "upstream_frozen_by_cached_input_boundary": True,
        "input_unchanged": sha(args.data) == data_sha,
        "initial_state_sha256": state_hash(initial),
        "latest_state_sha256": state_hash(torch.load(latest, map_location="cpu", weights_only=False)["model"]),
        "best_state_sha256": state_hash(best_checkpoint["model"]),
        "wall_seconds_this_invocation": time.perf_counter() - began,
        "identity": identity,
    }
    atomic_json(args.output / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group", choices=GROUPS, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-epochs", type=int, default=FORMAL_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=FORMAL_PATIENCE)
    parser.add_argument("--interrupt-after", type=int)
    parser.add_argument("--execute-formal", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train(args), sort_keys=True))


if __name__ == "__main__":
    main()
