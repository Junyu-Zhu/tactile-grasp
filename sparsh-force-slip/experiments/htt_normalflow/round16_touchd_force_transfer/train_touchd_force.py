#!/usr/bin/env python3
"""Resumable ToucHD domain-local force-private adaptation on complete cached tokens."""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler

from touchd_common import (SOURCE_CHECKPOINT, TARGET, atomic_json, atomic_torch, fresh_adapter,
                           seed_all, sha256, state_sha256, transferable_private_state)


PROTOCOL = {"optimizer": "AdamW", "learning_rate": 1e-4, "weight_decay": 1e-4,
            "batch_size": 128, "max_epochs": 30, "patience": 7,
            "loss": "SmoothL1 beta=1 on fit-standardized released [Fx,Fy,-Fz]",
            "selection": "earliest strict minimum selection mean per-axis native-unit RMSE"}


class ShardedFrames(Dataset):
    def __init__(self, entries):
        self.entries = entries
        self.index = [(shard, row) for shard, entry in enumerate(entries)
                      for row in range(entry["identity"]["samples"])]
        self.tokens, self.targets = {}, {}

    def __len__(self): return len(self.index)

    def __getitem__(self, item):
        shard, row = self.index[item]; entry = self.entries[shard]
        if shard not in self.tokens:
            self.tokens[shard] = np.load(entry["token_path"], mmap_mode="r")
            self.targets[shard] = np.load(entry["target_path"], mmap_mode="r")
        return (torch.from_numpy(np.array(self.tokens[shard][row], copy=True)),
                torch.from_numpy(np.array(self.targets[shard][row], copy=True)))


def make_loader(dataset, batch_size, shuffle, generator):
    sampler = RandomSampler(dataset, generator=generator) if shuffle else None
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False,
                      num_workers=2, pin_memory=True, persistent_workers=True,
                      generator=torch.Generator().manual_seed(0))


def capture(generator: torch.Generator) -> dict:
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
            "generator": generator.get_state()}


def restore(state: dict, generator: torch.Generator) -> None:
    torch.set_rng_state(state["torch"]); np.random.set_state(state["numpy"])
    generator.set_state(state["generator"])


def metric(model, tokens, targets, mean, std, batch, device) -> float:
    values = []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(tokens), batch):
            prediction = model(tokens[offset:offset + batch].to(device).float()) * std + mean
            values.append(prediction.cpu())
    prediction = torch.cat(values)
    return float(torch.sqrt(torch.mean((prediction - targets.float()) ** 2, dim=0)).mean())


def run(args) -> dict:
    seed_all(args.seed)
    sharded = args.cache.suffix == ".json"
    if sharded:
        data = json.loads(args.cache.read_text())
        if data.get("schema") != "round16_touchd_token_cache_v1" or data.get("status") != "complete" or data.get("formal") is not True:
            raise RuntimeError("incompatible formal sharded cache")
        fit_entries = [entry for entry in data["entries"] if entry["identity"]["role"] == "fit"]
        selection_entries = [entry for entry in data["entries"] if entry["identity"]["role"] == "selection"]
        fit_data, selection_data = ShardedFrames(fit_entries), ShardedFrames(selection_entries)
        fit_targets = torch.from_numpy(np.concatenate([np.load(entry["target_path"], mmap_mode="r") for entry in fit_entries])).float()
        fit_idx = selection_idx = tokens = targets = None
        fit_count, selection_count = len(fit_data), len(selection_data)
    else:
        data = torch.load(args.cache, map_location="cpu", weights_only=False)
        if data.get("status") != "complete" or data.get("target") != TARGET:
            raise RuntimeError("incompatible complete-token cache")
        role = [row["role"] for row in data["metadata"]]
        fit_idx = torch.tensor([i for i, value in enumerate(role) if value == "fit"])
        selection_idx = torch.tensor([i for i, value in enumerate(role) if value == "selection"])
        if not len(fit_idx) or not len(selection_idx): raise RuntimeError("fit and selection must both be present")
        tokens = data["tokens"]; targets = data["targets"].float(); fit_targets = targets[fit_idx]
        fit_count, selection_count = len(fit_idx), len(selection_idx)
    mean = fit_targets.mean(0); std = fit_targets.std(0, unbiased=False).clamp_min(1e-6)
    device = torch.device(args.device)
    mean_device, std_device = mean.to(device), std.to(device)
    model = fresh_adapter(args.seed).to(device)
    initial = state_sha256(model.state_dict())
    initial_components = {name: state_sha256(getattr(model, name).state_dict())
                          for name in ("force_pooler", "force_trunk", "force_head")}
    optimizer = torch.optim.AdamW(model.parameters(), lr=PROTOCOL["learning_rate"], weight_decay=PROTOCOL["weight_decay"])
    generator = torch.Generator().manual_seed(args.seed)
    epochs = args.epochs or PROTOCOL["max_epochs"]
    patience = args.patience or PROTOCOL["patience"]
    batch_size = args.batch_size or PROTOCOL["batch_size"]
    config = {**PROTOCOL, "seed": args.seed, "epochs": epochs, "patience": patience, "batch_size": batch_size,
              "smoke": args.smoke, "cache": str(args.cache.resolve()), "cache_sha256": sha256(args.cache),
              "source_checkpoint": str(SOURCE_CHECKPOINT), "source_checkpoint_sha256": sha256(SOURCE_CHECKPOINT),
              "target": TARGET, "fit_samples": fit_count, "selection_samples": selection_count,
              "formal_sharded_cache": sharded}
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("refusing changed frozen config")
    atomic_json(config_path, config)
    fit_loader = make_loader(fit_data, batch_size, True, generator) if sharded else None
    selection_loader = make_loader(selection_data, batch_size, False, torch.Generator().manual_seed(0)) if sharded else None
    latest, best = args.output / "latest.pth", args.output / "best.pth"
    history, best_metric, best_epoch, wait, start = [], float("inf"), 0, 0, 1
    if latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if saved["config"] != config:
            raise RuntimeError("incompatible resume")
        model.load_state_dict(saved["model_state"], strict=True); optimizer.load_state_dict(saved["optimizer_state"])
        restore(saved["rng_state"], generator)
        history, best_metric, best_epoch, wait = saved["history"], saved["best_metric"], saved["best_epoch"], saved["wait"]
        start = saved["epoch"] + 1
    if wait >= patience:
        start = epochs + 1
    for epoch in range(start, epochs + 1):
        began = time.monotonic(); model.train(); losses = []
        if sharded:
            batches = fit_loader
        else:
            order = fit_idx[torch.randperm(len(fit_idx), generator=generator)]
            batches = ((tokens[index], targets[index]) for index in order.split(batch_size))
        for x, y in batches:
            x = x.to(device).float(); y = ((y.to(device).float() - mean_device) / std_device)
            optimizer.zero_grad(set_to_none=True); loss = nn.functional.smooth_l1_loss(model(x), y, beta=1.0)
            if not torch.isfinite(loss): raise RuntimeError("nonfinite loss")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError("missing/nonfinite gradient")
            optimizer.step(); losses.append(float(loss.detach()))
        if sharded:
            actual, predicted = [], []; model.eval()
            with torch.inference_mode():
                for x, y in selection_loader:
                    predicted.append((model(x.to(device).float()) * std_device + mean_device).cpu()); actual.append(y.float())
            value = float(torch.sqrt(torch.mean((torch.cat(predicted) - torch.cat(actual)) ** 2, dim=0)).mean())
        else:
            value = metric(model, tokens[selection_idx], targets[selection_idx], mean.to(device), std.to(device), batch_size, device)
        elapsed = time.monotonic() - began
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "selection_mean_rmse": value}
        history.append(row); improved = value < best_metric
        if improved: best_metric, best_epoch, wait = value, epoch, 0
        else: wait += 1
        payload = {"schema": "round16_touchd_force_v1", "epoch": epoch, "model_state": model.state_dict(),
                   "optimizer_state": optimizer.state_dict(), "rng_state": capture(generator), "history": history,
                   "best_metric": best_metric, "best_epoch": best_epoch, "wait": wait, "config": config,
                   "normalization": {"mean": mean, "std": std}}
        atomic_torch(latest, payload)
        if improved: atomic_torch(best, payload)
        print(json.dumps({**row, "elapsed_seconds": elapsed}), flush=True)
        if args.interrupt_after_epoch == epoch:
            return {"status": "interrupted", "epoch": epoch}
        if wait >= patience: break
    selected = torch.load(best, map_location="cpu", weights_only=False)
    model.load_state_dict(selected["model_state"], strict=True)
    changed = {name: initial_components[name] != state_sha256(getattr(model, name).state_dict()) for name in initial_components}
    if not all(changed.values()): raise RuntimeError(f"unchanged trainable component {changed}")
    result = {"schema": "round16_touchd_force_summary_v1", "status": "complete", **config,
              "initial_state_sha256": initial, "selected_state_sha256": state_sha256(model.state_dict()),
              "component_changed": changed, "finite": all(torch.isfinite(v).all() for v in model.state_dict().values()),
              "best_epoch": selected["best_epoch"], "best_metric": selected["best_metric"], "history": history,
              "normalization": {"mean": mean.tolist(), "std": std.tolist()},
              "transfer_boundary": "force_pooler+force_trunk only; output head discarded before HTT",
              "transferable_private_state": transferable_private_state(model),
              "best_checkpoint": str(best), "best_checkpoint_sha256": sha256(best),
              "latest_checkpoint": str(latest), "latest_checkpoint_sha256": sha256(latest)}
    atomic_torch(args.output / "transferable_private_state.pth", result["transferable_private_state"])
    del result["transferable_private_state"]
    result["transferable_private_state_checkpoint"] = str(args.output / "transferable_private_state.pth")
    result["transferable_private_state_sha256"] = sha256(args.output / "transferable_private_state.pth")
    atomic_json(args.output / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(20260914, 20260915, 20260916), required=True)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int); parser.add_argument("--batch-size", type=int)
    parser.add_argument("--smoke", action="store_true"); parser.add_argument("--interrupt-after-epoch", type=int)
    args = parser.parse_args(); print(json.dumps(run(args), indent=2, default=str))


if __name__ == "__main__": main()
