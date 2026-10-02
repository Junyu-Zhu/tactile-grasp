#!/usr/bin/env python3
"""Train fresh slip-only adapters from cached complete DINO/I-JEPA tokens."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import phase2_b_multitask as p2  # noqa: E402
from prepare_splits import verify as verify_split_manifest  # noqa: E402


FOLDS = tuple(f"htt_leave_p{i}" for i in range(1, 5))
SEEDS = (20260914, 20260915, 20260916)
PROTOCOL = {
    "optimizer": "AdamW",
    "learning_rate": 1.0e-4,
    "weight_decay": 1.0e-4,
    "batch_size": 128,
    "max_epochs": 30,
    "patience": 7,
    "selection": "earliest strict maximum validation balanced accuracy at threshold 0.5",
    "labels": "static=0, gross=1; incipient excluded from loss/primary metrics",
    "precision": "FP32",
}


def canonical_encoder(name: str) -> str:
    return "mae" if name == "mae_letterbox" else name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(tmp, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, tmp)
    os.replace(tmp, path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_determinism() -> None:
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


class SlipBranch(nn.Module):
    def __init__(self, decoder: nn.Module) -> None:
        super().__init__()
        if not all(hasattr(decoder, name) for name in ("slip_pooler", "slip_trunk", "slip_head")):
            raise TypeError("Round 3 requires the fully decoupled slip branch")
        self.slip_pooler = copy.deepcopy(decoder.slip_pooler)
        self.slip_trunk = copy.deepcopy(decoder.slip_trunk)
        self.slip_head = copy.deepcopy(decoder.slip_head)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        features = self.slip_pooler(tokens).squeeze(1)
        return self.slip_head(self.slip_trunk(features))


class CachedFrames(Dataset):
    def __init__(self, entries: list[dict[str, Any]], fold: str, role: str, primary_only: bool) -> None:
        self.entries = [entry for entry in entries if entry["roles_by_fold"][fold] == role]
        self.index: list[tuple[int, int, int]] = []
        self._tokens: dict[int, np.ndarray] = {}
        for entry_index, entry in enumerate(self.entries):
            labels = np.load(entry["label_path"], mmap_mode="r", allow_pickle=False)
            for frame, stage in enumerate(labels.tolist()):
                if not primary_only or stage in (0, 2):
                    self.index.append((entry_index, frame, int(stage)))
        if not self.index:
            raise ValueError(f"No samples for {fold}/{role}, primary_only={primary_only}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int, str, int, int]:
        entry_index, frame, stage = self.index[item]
        if entry_index not in self._tokens:
            self._tokens[entry_index] = np.load(self.entries[entry_index]["token_path"], mmap_mode="r", allow_pickle=False)
        # Copy removes the read-only memmap warning and decouples worker lifetime.
        tokens = torch.from_numpy(np.array(self._tokens[entry_index][frame], copy=True))
        binary = 1 if stage == 2 else 0
        return tokens, binary, self.entries[entry_index]["episode_id"], frame, stage


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int, workers: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator,
                      num_workers=workers, pin_memory=torch.cuda.is_available(), drop_last=False,
                      persistent_workers=workers > 0)


def balanced_accuracy(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    pred = probabilities >= 0.5
    y = labels.astype(bool)
    tn = int((~y & ~pred).sum())
    fp = int((~y & pred).sum())
    fn = int((y & ~pred).sum())
    tp = int((y & pred).sum())
    tnr = tn / (tn + fp) if tn + fp else float("nan")
    tpr = tp / (tp + fn) if tp + fn else float("nan")
    ba = (tnr + tpr) / 2
    return {"balanced_accuracy": ba, "static_fpr": 1 - tnr, "gross_recall": tpr,
            "confusion_matrix": [[tn, fp], [fn, tp]]}


def evaluate_primary(branch: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    branch.eval()
    labels, probabilities = [], []
    with torch.inference_mode():
        for tokens, binary, _, _, _ in loader:
            logits = branch(tokens.to(device, non_blocking=True).float())
            if not torch.isfinite(logits).all():
                raise RuntimeError("Nonfinite validation logits")
            labels.extend(binary.numpy().tolist())
            probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist())
    return balanced_accuracy(np.asarray(labels), np.asarray(probabilities))


def capture_rng() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def checkpoint_payload(branch: nn.Module, optimizer: torch.optim.Optimizer, epoch: int,
                       history: list[dict[str, Any]], best_epoch: int, best_ba: float,
                       wait: int, config: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    return {"format": "round4_fresh_sparsh_slip_branch_v1", "epoch": epoch, "branch_state": branch.state_dict(),
            "optimizer_state": optimizer.state_dict(), "rng_state": capture_rng(), "history": history,
            "best_epoch": best_epoch, "best_balanced_accuracy": best_ba, "patience_wait": wait,
            "config": config, "provenance": provenance}


def optimizer_tensors(optimizer: torch.optim.Optimizer) -> list[torch.Tensor]:
    tensors = []
    state = optimizer.state_dict()["state"]
    for parameter_index in sorted(state):
        for key in sorted(state[parameter_index]):
            value = state[parameter_index][key]
            if torch.is_tensor(value):
                tensors.append(value.detach().cpu())
    return tensors


def same_next_step_proof(branch: nn.Module, serialized: dict[str, Any],
                         criterion: nn.Module, batch: tuple[Any, ...], device: torch.device) -> dict[str, Any]:
    """Compare continuation from in-memory and serialized-equivalent states."""
    left = copy.deepcopy(branch).to(device).train()
    right = copy.deepcopy(branch).to(device).train()
    right.load_state_dict(serialized["branch_state"], strict=True)
    left_opt = torch.optim.AdamW(left.parameters(), lr=PROTOCOL["learning_rate"], weight_decay=PROTOCOL["weight_decay"])
    right_opt = torch.optim.AdamW(right.parameters(), lr=PROTOCOL["learning_rate"], weight_decay=PROTOCOL["weight_decay"])
    state = copy.deepcopy(serialized["optimizer_state"])
    left_opt.load_state_dict(state)
    right_opt.load_state_dict(copy.deepcopy(state))
    tokens, binary = batch[0].to(device).float(), batch[1].to(device)

    def step(model: nn.Module, opt: torch.optim.Optimizer) -> tuple[torch.Tensor, list[torch.Tensor]]:
        opt.zero_grad(set_to_none=True)
        loss = criterion(model(tokens), binary)
        loss.backward()
        gradients = [p.grad.detach().cpu().clone() for p in model.parameters()]
        opt.step()
        return loss.detach().cpu(), gradients

    loss_left, grad_left = step(left, left_opt)
    loss_right, grad_right = step(right, right_opt)
    left_optimizer_tensors = optimizer_tensors(left_opt)
    right_optimizer_tensors = optimizer_tensors(right_opt)
    proof = {
        "loss_exact": torch.equal(loss_left, loss_right),
        "gradients_exact": all(torch.equal(a, b) for a, b in zip(grad_left, grad_right)),
        "parameters_exact": all(torch.equal(a, b) for a, b in zip(left.state_dict().values(), right.state_dict().values())),
        "optimizer_tensors_exact": len(left_optimizer_tensors) == len(right_optimizer_tensors) and all(
            torch.equal(a, b) for a, b in zip(left_optimizer_tensors, right_optimizer_tensors)),
    }
    proof["pass"] = all(proof.values())
    return proof


def frozen_source_proof(encoder_name: str, provenance: dict[str, Any]) -> dict[str, Any]:
    encoder_name = canonical_encoder(encoder_name)
    torch.manual_seed(42)
    model = p2.FrozenEncoderSharedForceSlip(encoder_name, "decoupled")
    checkpoint = p2.ENCODER_CHECKPOINTS[encoder_name].resolve()
    force_state = {k: v for k, v in model.decoder.state_dict().items() if k.startswith("force_")}
    proof = {
        "source_checkpoint_file_unchanged": sha256_file(checkpoint) == provenance["source_checkpoint_sha256"],
        "encoder_state_unchanged": tensor_state_sha256(model.encoder.state_dict()) == provenance["source_encoder_state_sha256"],
        "force_state_unchanged": tensor_state_sha256(force_state) == provenance["source_force_state_sha256"],
        "encoder_parameters_in_optimizer": provenance["encoder_parameters_in_optimizer"],
        "force_parameters_in_optimizer": provenance["force_parameters_in_optimizer"],
    }
    proof["pass"] = all(proof[key] for key in ("source_checkpoint_file_unchanged", "encoder_state_unchanged", "force_state_unchanged")) and not proof["encoder_parameters_in_optimizer"] and not proof["force_parameters_in_optimizer"]
    return proof


def role_isolation(entries: list[dict[str, Any]], fold: str) -> dict[str, Any]:
    by_role = {role: {entry["episode_id"] for entry in entries if entry["roles_by_fold"][fold] == role}
               for role in ("train", "validation", "calibration", "test")}
    intersections = {f"{a}_{b}": sorted(by_role[a] & by_role[b])
                     for i, a in enumerate(by_role) for b in list(by_role)[i + 1:]}
    result = {"episode_counts": {key: len(value) for key, value in by_role.items()},
              "pairwise_intersections": intersections,
              "pass": all(not value for value in intersections.values())}
    if not result["pass"]:
        raise RuntimeError("Cached episode roles overlap")
    return result


def validate_cache_roles(cache: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(cache["manifest"])
    if sha256_file(manifest_path) != cache["manifest_sha256"]:
        raise RuntimeError("Frozen split manifest hash changed")
    manifest = json.loads(manifest_path.read_text())
    verify_split_manifest(manifest)
    expected = {}
    for fold, split in manifest["splits"].items():
        if fold.startswith("htt_leave_"):
            for role, ids in split.items():
                for episode_id in ids:
                    expected.setdefault(episode_id, {})[fold] = role
    for entry in cache["entries"]:
        if entry["roles_by_fold"] != expected[entry["episode_id"]]:
            raise RuntimeError(f"Cache role metadata mismatch: {entry['episode_id']}")
    return {"schema_version": manifest["schema_version"], "verification": manifest["verification"],
            "cache_roles_match_frozen_manifest": True}


def stratified_smoke_subset(index: list[tuple[int, int, int]], limit: int) -> list[tuple[int, int, int]]:
    halves = max(1, limit // 2)
    static = [row for row in index if row[2] == 0][:halves]
    gross = [row for row in index if row[2] == 2][:halves]
    result = static + gross
    if not static or not gross:
        raise ValueError("Smoke requires both classes")
    random.Random(0).shuffle(result)
    return result


def validate_completed_summary(path: Path, config: dict[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    summary = json.loads(path.read_text())
    if summary.get("status") not in ("complete", "smoke_complete"):
        return None
    if summary.get("config") != config:
        raise RuntimeError("Completed summary has incompatible configuration")
    for key in ("best_checkpoint", "latest_checkpoint"):
        artifact = Path(summary[key])
        expected = summary[f"{key}_sha256"]
        if not artifact.exists() or sha256_file(artifact) != expected:
            raise RuntimeError(f"Completed artifact failed integrity check: {key}")
    for role, artifact_name in summary["predictions"].items():
        artifact = Path(artifact_name)
        if not artifact.exists() or sha256_file(artifact) != summary["prediction_sha256"][role]:
            raise RuntimeError(f"Completed prediction failed integrity check: {role}")
    return summary


def branch_from_source(encoder_name: str, seed: int, device: torch.device) -> tuple[SlipBranch, dict[str, Any]]:
    variant_name = encoder_name
    encoder_name = canonical_encoder(encoder_name)
    torch.manual_seed(42)
    source_model = p2.FrozenEncoderSharedForceSlip(encoder_name, "decoupled")
    checkpoint = p2.ENCODER_CHECKPOINTS[encoder_name].resolve()
    source_encoder_sha = tensor_state_sha256(source_model.encoder.state_dict())
    force_state = {k: v for k, v in source_model.decoder.state_dict().items() if k.startswith("force_")}
    source_force_sha = tensor_state_sha256(force_state)
    seed_all(seed)
    branch = SlipBranch(p2.build_decoder("decoupled"))
    seed_all(seed)
    independent = SlipBranch(p2.build_decoder("decoupled"))
    proof = {
        "independent_fresh_exact": all(torch.equal(branch.state_dict()[k], independent.state_dict()[k]) for k in branch.state_dict()),
        "trainable_parameter_tensors": len(list(branch.parameters())),
        "trainable_parameters": sum(parameter.numel() for parameter in branch.parameters()),
    }
    if not proof["independent_fresh_exact"] or proof["trainable_parameter_tensors"] != 23 or proof["trainable_parameters"] != 7_459_778:
        raise RuntimeError(f"Fresh slip branch initialization/boundary proof failed: {proof}")
    seed_all(seed)  # Equalize subsequent data ordering after construction.
    provenance = {"encoder_variant": variant_name, "encoder": encoder_name, "constructor_seed": 42,
                  "source_checkpoint": str(checkpoint), "source_checkpoint_sha256": sha256_file(checkpoint),
                  "source_load_info": source_model.load_info, "source_encoder_state_sha256": source_encoder_sha,
                  "source_force_state_sha256": source_force_sha,
                  "initial_slip_state_sha256": tensor_state_sha256(branch.state_dict()), "initialization_proof": proof,
                  "trainable_parameter_names": list(dict(branch.named_parameters())),
                  "encoder_parameters_in_optimizer": [], "force_parameters_in_optimizer": [],
                  "source_frozen_state_unchanged_during_initialization": {
                      "encoder": tensor_state_sha256(source_model.encoder.state_dict()) == source_encoder_sha,
                      "force": tensor_state_sha256({k: v for k, v in source_model.decoder.state_dict().items() if k.startswith("force_")}) == source_force_sha,
                  }}
    return branch.to(device), provenance


def write_predictions(branch: nn.Module, cache: dict[str, Any], fold: str, role: str, output: Path,
                      device: torch.device, batch_size: int, workers: int, seed: int,
                      init: str, checkpoint_epoch: int) -> None:
    dataset = CachedFrames(cache["entries"], fold, role, primary_only=False)
    loader = make_loader(dataset, batch_size, False, seed, workers)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + f".tmp.{os.getpid()}")
    fields = ["episode_id", "t", "stage", "p_slip", "p_static", "p_gross", "fold", "seed", "init", "checkpoint_epoch"]
    branch.eval()
    with tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        with torch.inference_mode():
            for tokens, _, episode_ids, frames, stages in loader:
                probs = torch.softmax(branch(tokens.to(device, non_blocking=True).float()), dim=1).cpu().numpy()
                if not np.isfinite(probs).all():
                    raise RuntimeError("Nonfinite prediction probabilities")
                for i, episode_id in enumerate(episode_ids):
                    writer.writerow({"episode_id": episode_id, "t": int(frames[i]), "stage": int(stages[i]),
                                     "p_slip": float(probs[i, 1]), "p_static": float(probs[i, 0]),
                                     "p_gross": float(probs[i, 1]), "fold": fold, "seed": seed,
                                     "init": init, "checkpoint_epoch": checkpoint_epoch})
    os.replace(tmp, output)


def run(args: argparse.Namespace) -> dict[str, Any]:
    configure_determinism()
    device = torch.device(args.device)
    cache_path = args.cache_dir / "cache_manifest.json"
    cache = json.loads(cache_path.read_text())
    if cache.get("status") != "complete" or cache.get("format") != "round4_htt_full_sparsh_tokens_v1":
        raise ValueError("Training requires a complete compatible cache")
    if cache.get("encoder") != args.encoder or cache.get("constructor_seed") != 42:
        raise ValueError("Cache encoder or constructor seed does not match the frozen protocol")
    if args.fold not in FOLDS or args.seed not in SEEDS:
        raise ValueError("Fold/seed outside frozen protocol")
    cache_role_proof = validate_cache_roles(cache)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = {**PROTOCOL, "encoder": args.encoder, "fold": args.fold, "seed": args.seed, "init": "fresh",
              "device": str(device), "workers": args.workers, "batch_size": args.batch_size, "smoke": args.smoke,
              "cache_manifest_sha256": sha256_file(cache_path),
              "source_checkpoint_sha256": sha256_file(p2.ENCODER_CHECKPOINTS[canonical_encoder(args.encoder)].resolve()),
              "training_source_sha256": sha256_file(Path(__file__)),
              "protocol_sha256": sha256_file(HERE / "PROTOCOL.md")}
    if args.smoke:
        config["smoke_max_samples"] = args.smoke_max_samples
        config["max_epochs"] = 2
        config["patience"] = 2
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("Refusing to resume an incompatible run configuration")
    atomic_json(config_path, config)
    completed = output / "training_summary.json"
    previous = validate_completed_summary(completed, config)
    if previous is not None:
        print(json.dumps(previous, indent=2))
        return previous

    branch, provenance = branch_from_source(args.encoder, args.seed, device)
    if cache["checkpoint_sha256"] != provenance["source_checkpoint_sha256"] or cache["encoder_state_sha256"] != provenance["source_encoder_state_sha256"]:
        raise RuntimeError("Cache/source checkpoint encoder provenance mismatch")
    train_data = CachedFrames(cache["entries"], args.fold, "train", primary_only=True)
    val_data = CachedFrames(cache["entries"], args.fold, "validation", primary_only=True)
    isolation = role_isolation(cache["entries"], args.fold)
    if args.smoke:
        train_data.index = stratified_smoke_subset(train_data.index, args.smoke_max_samples)
        val_data.index = stratified_smoke_subset(val_data.index, args.smoke_max_samples)
    stages = np.asarray([stage for _, _, stage in train_data.index])
    counts = np.asarray([(stages == 0).sum(), (stages == 2).sum()], dtype=np.int64)
    if np.any(counts == 0):
        raise ValueError("Both train classes are required")
    weights = len(stages) / (2.0 * counts.astype(np.float64))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(branch.parameters(), lr=PROTOCOL["learning_rate"], weight_decay=PROTOCOL["weight_decay"])
    if set(id(p) for group in optimizer.param_groups for p in group["params"]) != {id(p) for p in branch.parameters()}:
        raise RuntimeError("Optimizer is not exactly the slip branch")
    latest, best = output / "latest.pth", output / "best.pth"
    history: list[dict[str, Any]] = []
    start_epoch, best_epoch, best_ba, wait = 1, 0, -float("inf"), 0
    max_epochs = config["max_epochs"]
    patience = config["patience"]
    if latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if saved["config"] != config or saved["provenance"] != provenance:
            raise RuntimeError("Incompatible resume checkpoint")
        if best.exists():
            saved_best = torch.load(best, map_location="cpu", weights_only=False)
            if saved_best["config"] != config or saved_best["provenance"] != provenance:
                raise RuntimeError("Incompatible best checkpoint")
            if saved_best["epoch"] == saved["epoch"] + 1:
                # An improved best is committed before latest. If interrupted
                # between the two atomic replaces, promote that complete epoch.
                if saved_best["best_epoch"] != saved_best["epoch"] or saved_best["history"][:-1] != saved["history"]:
                    raise RuntimeError("Invalid best-ahead-of-latest recovery state")
                saved = saved_best
                atomic_torch_save(latest, saved)
            elif saved_best["epoch"] != saved["best_epoch"] or saved_best["best_balanced_accuracy"] != saved["best_balanced_accuracy"]:
                raise RuntimeError("Best/latest checkpoint selection mismatch")
        elif saved["best_epoch"] == saved["epoch"]:
            atomic_torch_save(best, saved)
        else:
            raise RuntimeError("Latest checkpoint references a missing earlier best checkpoint")
        branch.load_state_dict(saved["branch_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state"])
        restore_rng(saved["rng_state"])
        history = saved["history"]
        start_epoch, best_epoch, best_ba, wait = saved["epoch"] + 1, saved["best_epoch"], saved["best_balanced_accuracy"], saved["patience_wait"]
        if wait >= patience:
            start_epoch = max_epochs + 1

    initial_sha = provenance["initial_slip_state_sha256"]
    for epoch in range(start_epoch, max_epochs + 1):
        train_loader = make_loader(train_data, args.batch_size, True, args.seed + epoch, args.workers)
        branch.train()
        losses, finite_gradients = [], True
        for tokens, binary, _, _, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = branch(tokens.to(device, non_blocking=True).float())
            loss = criterion(logits, binary.to(device, non_blocking=True))
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            finite_gradients = finite_gradients and all(p.grad is None or torch.isfinite(p.grad).all().item() for p in branch.parameters())
            if not finite_gradients:
                raise RuntimeError("Nonfinite slip gradient")
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_loader = make_loader(val_data, args.batch_size, False, args.seed, args.workers)
        metrics = evaluate_primary(branch, val_loader, device)
        improved = metrics["balanced_accuracy"] > best_ba
        if improved:
            best_ba, best_epoch, wait = metrics["balanced_accuracy"], epoch, 0
        else:
            wait += 1
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "finite_gradients": finite_gradients, **metrics}
        history.append(row)
        payload = checkpoint_payload(branch, optimizer, epoch, history, best_epoch, best_ba, wait, config, provenance)
        if improved:
            atomic_torch_save(best, payload)
        atomic_torch_save(latest, payload)
        atomic_json(output / "history.json", history)
        print(json.dumps(row), flush=True)
        if wait >= patience:
            break

    selected = torch.load(best, map_location="cpu", weights_only=False)
    branch.load_state_dict(selected["branch_state"], strict=True)
    optimizer.load_state_dict(selected["optimizer_state"])
    branch.to(device)
    restored_sha = tensor_state_sha256(branch.state_dict())
    if restored_sha != tensor_state_sha256(selected["branch_state"]):
        raise RuntimeError("Best checkpoint restore mismatch")
    changed = restored_sha != initial_sha
    if not changed:
        raise RuntimeError("Slip branch did not change during training")
    continuation_proof = None
    frozen_proof = None
    if args.smoke:
        proof_loader = make_loader(train_data, min(args.batch_size, 8), False, args.seed, 0)
        continuation_proof = same_next_step_proof(branch, selected, criterion, next(iter(proof_loader)), device)
        if not continuation_proof["pass"]:
            raise RuntimeError(f"Same-next-step checkpoint continuation proof failed: {continuation_proof}")
        frozen_proof = frozen_source_proof(args.encoder, provenance)
        if not frozen_proof["pass"]:
            raise RuntimeError(f"Frozen encoder/force proof failed: {frozen_proof}")
    write_predictions(branch, cache, args.fold, "calibration", output / "predictions" / "calibration.csv",
                      device, args.batch_size, args.workers, args.seed, "fresh", selected["epoch"])
    write_predictions(branch, cache, args.fold, "validation", output / "predictions" / "validation.csv",
                      device, args.batch_size, args.workers, args.seed, "fresh", selected["epoch"])
    prediction_paths = {r: output / "predictions" / f"{r}.csv" for r in ("calibration", "validation")}
    summary = {"status": "smoke_complete" if args.smoke else "complete", "config": config,
               "fold": args.fold, "seed": args.seed,
               "encoder": args.encoder, "init": "fresh", "epochs_completed": len(history), "best_epoch": selected["epoch"],
               "best_balanced_accuracy_at_0_5": selected["best_balanced_accuracy"],
               "class_counts_train_static_gross": counts.tolist(), "class_weights_train_only": weights.tolist(),
               "slip_parameters_changed": changed, "finite_gradients_all_epochs": all(h["finite_gradients"] for h in history),
               "checkpoint_restore_exact": True, "same_next_step_proof": continuation_proof,
               "frozen_encoder_force_proof": frozen_proof,
               "role_isolation": isolation, "cache_role_proof": cache_role_proof,
               "best_checkpoint": str(best), "best_checkpoint_sha256": sha256_file(best),
               "latest_checkpoint": str(latest), "latest_checkpoint_sha256": sha256_file(latest), "manifest_sha256": cache["manifest_sha256"],
               "cache_manifest": str(cache_path), "cache_manifest_sha256": sha256_file(cache_path),
               "provenance": provenance, "predictions": {r: str(path) for r, path in prediction_paths.items()},
               "prediction_sha256": {r: sha256_file(path) for r, path in prediction_paths.items()}}
    atomic_json(completed, summary)
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--encoder", choices=("dino", "ijepa", "mae_letterbox"), required=True)
    parser.add_argument("--fold", choices=FOLDS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=PROTOCOL["batch_size"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-max-samples", type=int, default=256)
    args = parser.parse_args()
    if args.workers < 0 or args.smoke_max_samples < 2 or args.batch_size < 1:
        parser.error("workers must be nonnegative; sizes must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
