#!/usr/bin/env python3
"""Train one frozen-input round-4 future-risk head under the frozen protocol."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


FOLD = "htt_leave_p1"
HORIZON = 8
HISTORY = 4
SEEDS = (20260914, 20260915, 20260916)
EXPECTED_SUPPORT_AUDIT = "79ae344ad3bfd01971a264023b484460c00d4d748e5150c68cd3cc8979d5d596"
CONFIG = {"batch_size": 128, "learning_rate": 1e-3, "weight_decay": 1e-4,
          "max_epochs": 100, "patience": 10, "hidden_size": 128, "dropout": 0.1}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def atomic_torch(path: Path, payload: dict) -> None:
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(payload, temp)
    os.replace(temp, path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("wb") as handle: np.save(handle, value, allow_pickle=False)
    os.replace(temp, path)


def configure(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"] is not None: torch.cuda.set_rng_state_all(state["cuda"])


def raw_steps(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        z = np.asarray(data["z"], np.float32); force = np.asarray(data["force_pred"], np.float32)
        delta = np.asarray(data["delta_force_pred"], np.float32); pslip = np.asarray(data["p_slip"], np.float32)
        labels = np.asarray(data["labels"], np.int8)
    x = np.concatenate([z, force, delta, pslip[:, None]], axis=1).astype(np.float32)
    return x, labels, pslip


def primary_indices(labels: np.ndarray) -> list[tuple[int, int]]:
    gross = np.flatnonzero(labels == 2); onset = int(gross[0]) if len(gross) else None
    stop = len(labels) if onset is None else onset
    result = []
    for t in range(HISTORY - 1, stop):
        if labels[t] != 0 or t + HORIZON >= len(labels): continue
        target = int(onset is not None and t < onset <= t + HORIZON)
        result.append((t, target))
    return result


def load_cache(manifest_path: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or manifest.get("format") != "round4_future_causal_inputs_v1":
        raise ValueError("incompatible future cache")
    episodes = []
    for entry in manifest["entries"]:
        path = Path(entry["path"])
        if sha256(path) != entry["sha256"]: raise ValueError(f"cache hash mismatch: {entry['episode_id']}")
        x, labels, pslip = raw_steps(path)
        episodes.append({**entry, "x": x, "labels": labels, "p_slip": pslip})
    return manifest, episodes


def examples(episodes: list[dict], role: str) -> tuple[np.ndarray, np.ndarray, list[tuple[str, int]]]:
    xs, ys, keys = [], [], []
    for episode in episodes:
        if episode["role"] != role: continue
        for t, target in primary_indices(episode["labels"]):
            xs.append(episode["x"][t - HISTORY + 1:t + 1]); ys.append(target); keys.append((episode["episode_id"], t))
    if not xs: raise ValueError(f"no primary examples for {role}")
    return np.stack(xs), np.asarray(ys, np.float32), keys


class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None: self.x, self.y = x, y
    def __len__(self) -> int: return len(self.y)
    def __getitem__(self, index: int): return torch.from_numpy(self.x[index]), torch.tensor(self.y[index])


class MLPHead(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__(); self.net = nn.Sequential(nn.Flatten(), nn.Linear(HISTORY * features, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


class GRUHead(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__(); self.gru = nn.GRU(features, 128, batch_first=True); self.out = nn.Linear(128, 1)
    def forward(self, x): return self.out(self.gru(x)[0][:, -1]).squeeze(-1)


def make_model(name: str, features: int) -> nn.Module:
    return MLPHead(features) if name == "mlp" else GRUHead(features) if name == "gru" else (_ for _ in ()).throw(ValueError(name))


def loader(x: np.ndarray, y: np.ndarray, batch: int, shuffle: bool, seed: int, workers: int) -> DataLoader:
    return DataLoader(ArrayDataset(x, y), batch_size=batch, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed), num_workers=workers,
                      pin_memory=torch.cuda.is_available(), drop_last=False)


def weighted_loss(model: nn.Module, data: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval(); total = count = 0
    with torch.inference_mode():
        for x, y in data:
            logits = model(x.to(device, non_blocking=True)); target = y.to(device, non_blocking=True)
            loss = criterion(logits, target)
            total += float(loss) * len(y); count += len(y)
    return total / count


def state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode()); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def checkpoint_payload(model, optimizer, epoch, history, best_epoch, best_loss, wait, config,
                       normalization_mean: np.ndarray, normalization_std: np.ndarray) -> dict:
    return {"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(), "epoch": epoch, "history": history,
            "best_epoch": best_epoch, "best_validation_weighted_bce": best_loss, "patience_wait": wait,
            "config": config, "rng_state": rng_state(),
            "normalization": {"mean": normalization_mean.copy(), "std": normalization_std.copy(),
                              "source": "all history steps from eligible train examples only"},
            "provenance": {"cache_manifest_sha256": config["cache_manifest_sha256"],
                           "support_audit_sha256": config["support_audit_sha256"],
                           "protocol_sha256": config["protocol_sha256"]}}


def same_next_step_proof(model_name: str, features: int, payload: dict, x: np.ndarray, y: np.ndarray,
                         pos_weight: float, device: torch.device) -> dict:
    models, optimizers = [], []
    for _ in range(2):
        model = make_model(model_name, features).to(device); model.load_state_dict(copy.deepcopy(payload["model_state"]), strict=True); model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
        optimizer.load_state_dict(copy.deepcopy(payload["optimizer_state"])); models.append(model); optimizers.append(optimizer)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    before = rng_state(); losses = []
    for model, optimizer in zip(models, optimizers):
        restore_rng(before); optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(torch.from_numpy(x).to(device)), torch.from_numpy(y).to(device))
        loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
    return {"loss_bitwise_equal": losses[0] == losses[1],
            "parameters_bitwise_equal": all(torch.equal(a, b) for a, b in zip(models[0].state_dict().values(), models[1].state_dict().values())),
            "pass": losses[0] == losses[1] and state_hash(models[0].state_dict()) == state_hash(models[1].state_dict())}


def write_predictions(model, episodes, mean_, std_, device, output: Path, batch_size: int) -> None:
    fields = ["episode_id", "t", "stage", "first_gross_index", "eligible_primary", "target_first_gross_h8",
              "p_future", "p_slip_current", "p_slip_history4_mean", "role"]
    temp = output.with_name(output.name + f".tmp.{os.getpid()}"); output.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for episode in episodes:
            labels, x = episode["labels"], episode["x"]
            gross = np.flatnonzero(labels == 2); onset = int(gross[0]) if len(gross) else None
            ts = np.arange(HISTORY - 1, len(labels))
            histories = np.stack([x[t-HISTORY+1:t+1] for t in ts])
            normalized = (histories - mean_) / std_
            scores = []
            with torch.inference_mode():
                for start in range(0, len(ts), batch_size):
                    logits = model(torch.from_numpy(normalized[start:start+batch_size]).to(device))
                    scores.extend(torch.sigmoid(logits).cpu().tolist())
            eligible = dict(primary_indices(labels))
            for t, score in zip(ts, scores):
                writer.writerow({"episode_id": episode["episode_id"], "t": int(t), "stage": int(labels[t]),
                                 "first_gross_index": "" if onset is None else onset,
                                 "eligible_primary": int(t in eligible),
                                 "target_first_gross_h8": eligible.get(int(t), ""), "p_future": score,
                                 "p_slip_current": float(episode["p_slip"][t]),
                                 "p_slip_history4_mean": float(np.mean(episode["p_slip"][t-HISTORY+1:t+1])),
                                 "role": episode["role"]})
    os.replace(temp, output)


def run(args: argparse.Namespace) -> dict:
    if args.seed not in SEEDS: raise ValueError("seed outside frozen protocol")
    support_path = args.support_audit.resolve()
    if sha256(support_path) != EXPECTED_SUPPORT_AUDIT: raise ValueError("support audit differs from frozen protocol")
    support = json.loads(support_path.read_text())
    if support["decision"] != {"current_stage": 0, "horizon": 8, "selection": "shortest supported primary horizon from train only", "status": "train", "task": "current_static_to_first_gross"}:
        raise ValueError("frozen support decision mismatch")
    cache_path = args.cache_manifest.resolve(); cache, episodes = load_cache(cache_path)
    train_x, train_y, _ = examples(episodes, "train"); val_x, val_y, _ = examples(episodes, "validation")
    mean_ = train_x.reshape(-1, train_x.shape[-1]).mean(0, keepdims=True).astype(np.float32)
    std_ = train_x.reshape(-1, train_x.shape[-1]).std(0, keepdims=True).astype(np.float32)
    std_[std_ < 1e-6] = 1.0
    train_x = ((train_x - mean_) / std_).astype(np.float32); val_x = ((val_x - mean_) / std_).astype(np.float32)
    counts = [int(np.sum(train_y == 0)), int(np.sum(train_y == 1))]
    if min(counts) == 0: raise ValueError("train target requires both classes")
    pos_weight = counts[0] / counts[1]
    config = {**CONFIG, "model": args.model, "seed": args.seed, "fold": FOLD, "horizon": HORIZON,
              "history": HISTORY, "feature_dimension": train_x.shape[-1], "train_counts_negative_positive": counts,
              "positive_weight_train_only": pos_weight, "cache_manifest": str(cache_path),
              "cache_manifest_sha256": sha256(cache_path), "support_audit_sha256": sha256(support_path),
              "protocol_sha256": sha256(args.protocol.resolve()), "device": args.device, "workers": args.workers,
              "smoke": args.smoke}
    if args.smoke: config.update(max_epochs=2, patience=2)
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config: raise RuntimeError("incompatible existing run")
    atomic_json(config_path, config)
    completed_path = output / "training_summary.json"
    if completed_path.exists():
        completed = json.loads(completed_path.read_text())
        expected_status = "smoke_complete" if args.smoke else "complete"
        if completed.get("status") == expected_status and completed.get("config") == config:
            for key in ("best_checkpoint", "latest_checkpoint"):
                if sha256(Path(completed[key])) != completed[f"{key}_sha256"]: raise RuntimeError(f"completed {key} hash mismatch")
            return completed
    configure(args.seed); device = torch.device(args.device)
    model = make_model(args.model, train_x.shape[-1]).to(device)
    initial_hash = state_hash(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    latest, best = output / "latest.pth", output / "best.pth"
    history = []; start, best_epoch, best_loss, wait = 1, 0, float("inf"), 0
    if latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if saved["config"] != config: raise RuntimeError("incompatible resume checkpoint")
        if not (np.array_equal(saved["normalization"]["mean"], mean_) and np.array_equal(saved["normalization"]["std"], std_)):
            raise RuntimeError("resume checkpoint normalization mismatch")
        model.load_state_dict(saved["model_state"]); optimizer.load_state_dict(saved["optimizer_state"]); restore_rng(saved["rng_state"])
        history = saved["history"]; start = saved["epoch"] + 1; best_epoch = saved["best_epoch"]
        best_loss = saved["best_validation_weighted_bce"]; wait = saved["patience_wait"]
        if wait >= config["patience"]: start = config["max_epochs"] + 1
    for epoch in range(start, config["max_epochs"] + 1):
        model.train(); losses = []
        for x, y in loader(train_x, train_y, config["batch_size"], True, args.seed + epoch, args.workers):
            optimizer.zero_grad(set_to_none=True); logits = model(x.to(device, non_blocking=True)); target = y.to(device, non_blocking=True)
            loss = criterion(logits, target)
            if not torch.isfinite(loss): raise RuntimeError("nonfinite train loss")
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()): raise RuntimeError("nonfinite gradient")
            optimizer.step(); losses.append(float(loss.detach().cpu()))
        validation_loss = weighted_loss(model, loader(val_x, val_y, config["batch_size"], False, args.seed, args.workers), criterion, device)
        improved = validation_loss < best_loss
        if improved: best_loss, best_epoch, wait = validation_loss, epoch, 0
        else: wait += 1
        history.append({"epoch": epoch, "train_weighted_bce": float(np.mean(losses)),
                        "validation_weighted_bce": validation_loss, "finite": True})
        payload = checkpoint_payload(model, optimizer, epoch, history, best_epoch, best_loss, wait, config, mean_, std_)
        atomic_torch(latest, payload)
        if improved: atomic_torch(best, payload)
        atomic_json(output / "history.json", history)
        print(json.dumps(history[-1]), flush=True)
        if wait >= config["patience"]: break
    selected = torch.load(best, map_location="cpu", weights_only=False)
    model.load_state_dict(selected["model_state"], strict=True); model.to(device)
    if state_hash(model.state_dict()) != state_hash(selected["model_state"]): raise RuntimeError("best checkpoint restore mismatch")
    changed = state_hash(model.state_dict()) != initial_hash
    if not changed: raise RuntimeError("model parameters did not change")
    for role in ("train", "calibration", "validation"):
        write_predictions(model, [episode for episode in episodes if episode["role"] == role], mean_, std_, device,
                          output / "predictions" / f"{role}.csv", config["batch_size"])
    continuation = same_next_step_proof(args.model, train_x.shape[-1], selected, train_x[:8], train_y[:8], pos_weight, device) if args.smoke else None
    if continuation is not None and not continuation["pass"]: raise RuntimeError(f"checkpoint continuation mismatch: {continuation}")
    mean_path, std_path = output / "normalization_mean.npy", output / "normalization_std.npy"
    atomic_npy(mean_path, mean_); atomic_npy(std_path, std_)
    summary = {"status": "smoke_complete" if args.smoke else "complete", "model": args.model, "seed": args.seed,
               "best_epoch": selected["epoch"], "best_validation_weighted_bce": selected["best_validation_weighted_bce"],
               "epochs_completed": len(history), "parameters_changed": changed, "config": config,
               "checkpoint_restore_exact": True, "same_next_step_proof": continuation,
               "normalization": {"mean": str(mean_path), "mean_sha256": sha256(mean_path),
                                 "std": str(std_path), "std_sha256": sha256(std_path),
                                 "source": "all history steps from eligible train examples only"},
               "best_checkpoint": str(best), "best_checkpoint_sha256": sha256(best),
               "latest_checkpoint": str(latest), "latest_checkpoint_sha256": sha256(latest),
               "predictions": {}, "target": "current_static first gross in (t,t+8], complete window"}
    for role in ("train", "calibration", "validation"):
        path = output / "predictions" / f"{role}.csv"; summary["predictions"][role] = {"path": str(path), "sha256": sha256(path)}
    atomic_json(output / "training_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True); parser.add_argument("--support-audit", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True); parser.add_argument("--model", choices=("mlp", "gru"), required=True)
    parser.add_argument("--seed", type=int, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None: print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__": main()
