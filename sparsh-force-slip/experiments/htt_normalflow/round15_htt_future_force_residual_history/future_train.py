#!/usr/bin/env python3
"""Round-15 persistence-anchored residual future-force trainer."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

HORIZONS = (1, 5, 10)
GROUPS = ("C_current", "H_history")
SEEDS = (20260914, 20260915, 20260916)


def sha(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def atomic_save(value, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    os.close(fd)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(value, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    os.close(fd)
    try:
        Path(temporary).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def state_hash(state) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class ResidualFutureModel(nn.Module):
    """Identical module for both groups; only selected sequence length differs."""

    def __init__(self, group: str):
        super().__init__()
        if group not in GROUPS:
            raise ValueError(group)
        self.group = group
        self.joint = nn.Sequential(nn.Linear(195, 48), nn.GELU())
        self.gru = nn.GRU(48, 48, batch_first=True)
        self.out = nn.Linear(48, 9)

    def forward(self, normalized_x: torch.Tensor) -> torch.Tensor:
        if normalized_x.ndim != 3 or normalized_x.shape[1:] != (9, 195):
            raise ValueError(f"expected [batch,9,195], got {tuple(normalized_x.shape)}")
        selected = normalized_x[:, -1:, :] if self.group == "C_current" else normalized_x
        encoded = self.joint(selected)
        final = self.gru(encoded)[0][:, -1]
        return self.out(final).reshape(-1, 3, 3)


def init_model(group: str, seed: int) -> ResidualFutureModel:
    torch.manual_seed(seed)
    model = ResidualFutureModel(group)
    for name, parameter in model.named_parameters():
        if name.startswith("out."):
            nn.init.zeros_(parameter)
        elif parameter.ndim >= 2:
            nn.init.xavier_uniform_(parameter)
        else:
            nn.init.zeros_(parameter)
    return model


def assert_role_isolation(roles) -> None:
    required = {"fit", "selection", "calibration", "validation"}
    if set(roles) != required:
        raise ValueError(f"roles must be {sorted(required)}, got {sorted(roles)}")
    seen = {}
    for role, data in roles.items():
        groups = set(data["leakage_group"])
        for other, previous in seen.items():
            overlap = groups & previous
            if overlap:
                raise ValueError(f"leakage overlap {role}/{other}: {sorted(overlap)[:2]}")
        seen[role] = groups


def validate_cache(data) -> None:
    if data.get("schema") != "round14_future_cache_v1":
        raise ValueError("only accepted Round-14 prepared caches are allowed")
    if tuple(data.get("horizons", ())) != HORIZONS:
        raise ValueError("horizon mismatch")
    assert_role_isolation(data["roles"])
    for role, value in data["roles"].items():
        if value["x"].shape[1:] != (9, 195) or value["y"].shape[1:] != (3, 3):
            raise ValueError(f"shape mismatch in {role}")
        if "y_current" not in value or value["y_current"].shape[1:] != (3,):
            raise ValueError(f"current GT missing in {role}")
        if not (torch.isfinite(value["x"]).all() and torch.isfinite(value["y"]).all()):
            raise ValueError(f"nonfinite cache in {role}")


def fit_normalizer(fit):
    x = fit["x"]
    current_prediction = x[:, -1, 192:195]
    residual = fit["y"] - current_prediction[:, None, :]
    return {
        "x_mean": x.mean((0, 1)),
        "x_std": x.std((0, 1), unbiased=False).clamp_min(1e-6),
        # Scale only. There is deliberately no residual mean.
        "residual_scale": residual.std(0, unbiased=False).clamp_min(1e-6),
        "residual_offset": torch.zeros(3, 3, dtype=residual.dtype),
    }


def normalize_x(x, normalizer):
    return (x - normalizer["x_mean"]) / normalizer["x_std"]


def target_residual(role, normalizer):
    current_prediction = role["x"][:, -1, 192:195]
    return (role["y"] - current_prediction[:, None, :]) / normalizer["residual_scale"]


def decode_future(normalized_residual, role, normalizer):
    current_prediction = role["x"][:, -1, 192:195]
    residual_n = normalized_residual * normalizer["residual_scale"]
    return current_prediction[:, None, :] + residual_n


def _normalizer_hash(normalizer) -> str:
    digest = hashlib.sha256()
    for key in ("x_mean", "x_std", "residual_scale", "residual_offset"):
        digest.update(key.encode())
        digest.update(normalizer[key].contiguous().numpy().tobytes())
    return digest.hexdigest()


def train(
    data,
    output: Path | str,
    group: str,
    fold: int,
    seed: int,
    max_epochs: int = 60,
    patience: int = 10,
    interrupt_after: int | None = None,
    device: str = "cpu",
):
    validate_cache(data)
    if group not in GROUPS or fold not in range(1, 5) or seed not in SEEDS:
        raise ValueError("unregistered formal identity")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fit = data["roles"]["fit"]
    selection = data["roles"]["selection"]
    normalizer = fit_normalizer(fit)
    fit_x = normalize_x(fit["x"], normalizer)
    fit_target = target_residual(fit, normalizer)
    selection_x = normalize_x(selection["x"], normalizer)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = init_model(group, seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sampler = torch.Generator().manual_seed(seed + 1500)
    latest = output / "latest.pth"
    best = output / "best.pth"
    protocol = Path(__file__).with_name("PROTOCOL.md")
    protocol_lock = Path(__file__).with_name("PROTOCOL_LOCK.json")
    source_lock = Path(__file__).with_name("SOURCE_LOCK.json")
    input_lock = Path(__file__).with_name("INPUT_LOCK.json")
    identity = {
        "round": 15,
        "group": group,
        "fold": fold,
        "seed": seed,
        "horizons": list(HORIZONS),
        "history_steps": 1 if group == "C_current" else 9,
        "batch_size": 256,
        "max_epochs": max_epochs,
        "patience": patience,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "loss": "smooth_l1_beta_1_scale_only_residual",
        "selection": "earliest_strict_min_native_future_mae",
        "round14_data_sha256": data.get("round14_data_sha256", "synthetic"),
        "normalizer_sha256": _normalizer_hash(normalizer),
        "source_sha256": sha(__file__),
        "protocol_sha256": sha(protocol),
        "protocol_lock_sha256": sha(protocol_lock) if protocol_lock.exists() else "unlocked_smoke",
        "source_lock_sha256": sha(source_lock) if source_lock.exists() else "unlocked_smoke",
        "input_lock_sha256": sha(input_lock) if input_lock.exists() else "unlocked_smoke",
    }

    start = 0
    best_metric = float("inf")
    best_epoch = -1
    wait = 0
    history = []
    if latest.exists():
        checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
        if checkpoint["identity"] != identity:
            raise ValueError("resume identity mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        sampler.set_state(checkpoint["sampler_rng"].cpu())
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if torch.cuda.is_available() and checkpoint.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        start = checkpoint["epoch"] + 1
        best_metric = checkpoint["best_metric"]
        best_epoch = checkpoint["best_epoch"]
        wait = checkpoint["wait"]
        history = checkpoint["history"]

    if wait >= patience or start >= max_epochs:
        result = {
            "status": "complete",
            "best_epoch": best_epoch,
            "best_metric_native_n_mae": best_metric,
            "epochs": len(history),
            "state_sha256": state_hash(torch.load(latest, map_location="cpu", weights_only=False)["model"]),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "identity": identity,
            "resume_without_extra_epoch": True,
        }
        atomic_json(result, output / "summary.json")
        return result

    for epoch in range(start, max_epochs):
        model.train()
        permutation = torch.randperm(len(fit_x), generator=sampler)
        losses = []
        for indices in permutation.split(256):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(fit_x[indices].to(device))
            loss = nn.functional.smooth_l1_loss(
                prediction, fit_target[indices].to(device), beta=1.0
            )
            if not torch.isfinite(loss):
                raise ValueError("nonfinite loss")
            loss.backward()
            if not all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise ValueError("nonfinite gradient")
            optimizer.step()
            losses.append(float(loss))

        model.eval()
        with torch.inference_mode():
            normalized_residual = model(selection_x.to(device)).cpu()
            future = decode_future(normalized_residual, selection, normalizer)
            metric = float((future - selection["y"]).abs().mean())
        improved = metric < best_metric
        if improved:
            best_metric = metric
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
        history.append(
            {"epoch": epoch, "fit_loss": float(np.mean(losses)), "selection_mae": metric}
        )
        checkpoint = {
            "identity": identity,
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "sampler_rng": sampler.get_state(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "normalizer": normalizer,
            "epoch": epoch,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "wait": wait,
            "history": history,
        }
        atomic_save(checkpoint, latest)
        if improved:
            atomic_save(checkpoint, best)
        if interrupt_after is not None and epoch >= interrupt_after:
            result = {
                "status": "interrupted",
                "latest": str(latest),
                "identity": identity,
                "epochs": len(history),
            }
            atomic_json(result, output / "summary.json")
            return result
        if wait >= patience:
            break

    result = {
        "status": "complete",
        "best_epoch": best_epoch,
        "best_metric_native_n_mae": best_metric,
        "epochs": len(history),
        "state_sha256": state_hash(torch.load(latest, map_location="cpu", weights_only=False)["model"]),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "identity": identity,
    }
    atomic_json(result, output / "summary.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group", choices=GROUPS, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--interrupt-after", type=int)
    arguments = parser.parse_args()
    data = torch.load(arguments.data, map_location="cpu", weights_only=False)
    data["round14_data_sha256"] = sha(arguments.data)
    result = train(
        data,
        arguments.output,
        arguments.group,
        arguments.fold,
        arguments.seed,
        arguments.max_epochs,
        arguments.patience,
        arguments.interrupt_after,
        arguments.device,
    )
    print(stable_json(result))


if __name__ == "__main__":
    main()
