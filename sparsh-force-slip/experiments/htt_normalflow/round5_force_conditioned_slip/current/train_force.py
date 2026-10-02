#!/usr/bin/env python3
"""Resumable HTT-native force adaptation with a frozen cached MAE encoder."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, Subset

from common import (atomic_json, atomic_torch, capture_rng, configure_determinism,
                    optimizer_tensor_values, parameter_boundary, restore_rng, seed_all, sha256_file, source_bundle,
                    tensor_state_sha256)
from data import ForceFrames, force_train_targets, load_cache, require_cache_audit, role_entries
from metrics import force_metrics, force_target_normalization
from models import ForceAdapter
from sources import load_decoupled_decoder


PROTOCOL = {
    "optimizer": "AdamW", "learning_rate": 1e-4, "weight_decay": 1e-4,
    "batch_size": 128, "max_epochs": 30, "patience": 7,
    "loss": "SmoothL1 on train-standardized native [shear_x,shear_y,normal] N; beta=1",
    "selection": "earliest strict minimum validation mean per-axis native-N RMSE",
    "target": "clip((6d_force-ref_force)[:3],-20,20) N; shear_x,shear_y,normal",
}


def make_loader(dataset, batch_size: int, shuffle: bool, generator: torch.Generator, workers: int):
    sampler = RandomSampler(dataset, generator=generator) if shuffle else None
    # Keep sampler RNG checkpointable and independent of worker base-seed draws.
    worker_generator = torch.Generator().manual_seed(0)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler,
                      generator=worker_generator,
                      num_workers=workers, pin_memory=torch.cuda.is_available(), drop_last=False,
                      persistent_workers=workers > 0)


def evaluate(model: nn.Module, loader: DataLoader, mean: torch.Tensor, std: torch.Tensor,
             device: torch.device) -> dict[str, Any]:
    model.eval()
    targets, predictions = [], []
    with torch.inference_mode():
        for tokens, target, _, _ in loader:
            prediction = model(tokens.to(device).float()) * std + mean
            targets.append(target.numpy())
            predictions.append(prediction.cpu().numpy())
    return force_metrics(np.concatenate(targets), np.concatenate(predictions))


def checkpoint(model, optimizer, generator, epoch, history, best_epoch, best_metric, wait,
               config, normalization, provenance):
    return {
        "format": "round5_htt_native_force_adapter_v1", "epoch": epoch,
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
        "rng_state": capture_rng(generator), "history": history, "best_epoch": best_epoch,
        "best_metric": best_metric, "patience_wait": wait, "config": config,
        "normalization": {key: value.copy() for key, value in normalization.items()},
        "provenance": provenance,
    }


def same_next_step(model, saved, batch, mean, std, device) -> dict[str, bool]:
    left, right = copy.deepcopy(model).to(device).train(), copy.deepcopy(model).to(device).train()
    right.load_state_dict(saved["model_state"], strict=True)
    opts = [torch.optim.AdamW(candidate.parameters(), lr=PROTOCOL["learning_rate"],
                              weight_decay=PROTOCOL["weight_decay"]) for candidate in (left, right)]
    opts[0].load_state_dict(copy.deepcopy(saved["optimizer_state"]))
    opts[1].load_state_dict(copy.deepcopy(saved["optimizer_state"]))
    tokens, target = batch[0].to(device).float(), batch[1].to(device).float()
    outputs = []
    for candidate, optimizer in zip((left, right), opts):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.smooth_l1_loss(candidate(tokens), (target - mean) / std, beta=1.0)
        loss.backward()
        gradients = [parameter.grad.detach().cpu().clone() for parameter in candidate.parameters()]
        optimizer.step()
        outputs.append((loss.detach().cpu(), gradients))
    left_opt, right_opt = optimizer_tensor_values(opts[0]), optimizer_tensor_values(opts[1])
    proof = {
        "loss_exact": torch.equal(outputs[0][0], outputs[1][0]),
        "gradients_exact": all(torch.equal(a, b) for a, b in zip(outputs[0][1], outputs[1][1])),
        "parameters_exact": all(torch.equal(a, b) for a, b in zip(left.state_dict().values(), right.state_dict().values())),
        "optimizer_exact": len(left_opt) == len(right_opt) and all(torch.equal(a, b) for a, b in zip(left_opt, right_opt)),
    }
    proof["pass"] = all(proof.values())
    return proof


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--fold", choices=tuple(f"htt_leave_p{i}" for i in range(1, 5)), required=True)
    parser.add_argument("--seed", type=int, choices=(20260914, 20260915, 20260916), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-unverified-cache", action="store_true")
    parser.add_argument("--interrupt-after-epoch", type=int)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    configure_determinism()
    seed_all(args.seed)
    cache = load_cache(args.cache.resolve())
    cache_audit = require_cache_audit(args.cache_audit, cache, args.allow_unverified_cache and args.smoke)
    train_entries = role_entries(cache, args.fold, "train", "force")
    validation_entries = role_entries(cache, args.fold, "validation", "force")
    train_data = ForceFrames(train_entries)
    validation_data = ForceFrames(validation_entries)
    epochs = 2 if args.smoke else PROTOCOL["max_epochs"]
    patience = 2 if args.smoke else PROTOCOL["patience"]
    if args.smoke:
        train_data = Subset(train_data, range(min(64, len(train_data))))
        validation_data = Subset(validation_data, range(min(64, len(validation_data))))
    normalization = force_target_normalization(force_train_targets(train_entries))
    decoder, source_payload = load_decoupled_decoder(args.source_checkpoint.resolve())
    source_force_state = {name: value.detach().clone() for name, value in decoder.state_dict().items()
                          if name.startswith("force_")}
    model = ForceAdapter(decoder).to(args.device)
    initial_state_hash = tensor_state_sha256(model.state_dict())
    initial_components = {name: tensor_state_sha256(getattr(model, name).state_dict())
                          for name in ("force_pooler", "force_trunk", "force_head")}
    optimizer = torch.optim.AdamW(model.parameters(), lr=PROTOCOL["learning_rate"],
                                  weight_decay=PROTOCOL["weight_decay"])
    criterion = nn.SmoothL1Loss(beta=1.0)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = make_loader(train_data, PROTOCOL["batch_size"], True, generator, args.workers)
    validation_loader = make_loader(validation_data, PROTOCOL["batch_size"], False,
                                    torch.Generator().manual_seed(0), args.workers)
    mean = torch.tensor(normalization["mean"], device=args.device)
    std = torch.tensor(normalization["std"], device=args.device)
    config = {**PROTOCOL, "fold": args.fold, "seed": args.seed, "smoke": args.smoke,
              "workers": args.workers, "cache_manifest_sha256": cache["manifest_sha256"],
              "cache_audit_sha256": sha256_file(args.cache_audit.resolve()) if cache_audit else None,
              "source_checkpoint": str(args.source_checkpoint.resolve()),
              "source_checkpoint_sha256": sha256_file(args.source_checkpoint.resolve())}
    current_dir = Path(__file__).resolve().parent
    config["source_bundle"] = source_bundle([
        Path(__file__), current_dir / "common.py", current_dir / "data.py", current_dir / "metrics.py",
        current_dir / "models.py", current_dir / "sources.py",
        Path(sys.modules[load_decoupled_decoder.__module__].p2.__file__),
    ])
    provenance = {
        "source_epoch": source_payload.get("epoch"), "source_force_state_sha256": tensor_state_sha256(source_force_state),
        "source_code_sha256": sha256_file(Path(__file__)), "model_code_sha256": sha256_file(Path(sys.modules[ForceAdapter.__module__].__file__)),
        "parameter_boundary": parameter_boundary(model), "encoder_in_optimizer": False,
        "slip_in_optimizer": False, "force_output_head_reinitialized": True,
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("Refusing to change a frozen force run config")
    atomic_json(config_path, config)
    latest, best = output / "latest.pth", output / "best.pth"
    history, best_epoch, best_metric, wait, start = [], 0, float("inf"), 0, 1
    if latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if saved.get("format") != "round5_htt_native_force_adapter_v1" or saved["config"] != config:
            raise RuntimeError("Refusing incompatible force checkpoint resume")
        if any(not np.array_equal(saved["normalization"][key], normalization[key]) for key in ("mean", "std")):
            raise RuntimeError("Train-only force normalization changed")
        model.load_state_dict(saved["model_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state"])
        restore_rng(saved["rng_state"], generator)
        history, best_epoch, best_metric, wait = (saved["history"], saved["best_epoch"],
                                                  saved["best_metric"], saved["patience_wait"])
        start = saved["epoch"] + 1
    for epoch in range(start, epochs + 1):
        model.train()
        losses, finite_gradients = [], True
        for tokens, target, _, _ in train_loader:
            tokens, target = tokens.to(args.device).float(), target.to(args.device).float()
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(tokens), (target - mean) / std)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite force loss")
            loss.backward()
            finite_gradients &= all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                                    for parameter in model.parameters())
            if not finite_gradients:
                raise RuntimeError("Missing or nonfinite force gradients")
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = evaluate(model, validation_loader, mean, std, torch.device(args.device))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "finite_gradients": finite_gradients, **metrics}
        history.append(row)
        improved = metrics["mean_rmse_xyz_native_n"] < best_metric
        if improved:
            best_metric, best_epoch, wait = metrics["mean_rmse_xyz_native_n"], epoch, 0
        else:
            wait += 1
        payload = checkpoint(model, optimizer, generator, epoch, history, best_epoch, best_metric,
                             wait, config, normalization, provenance)
        atomic_torch(latest, payload)
        if improved:
            atomic_torch(best, payload)
        print(json.dumps(row), flush=True)
        if args.interrupt_after_epoch == epoch:
            raise RuntimeError(f"intentional interruption after committed epoch {epoch}")
        if wait >= patience:
            break
    selected = torch.load(best, map_location="cpu", weights_only=False)
    model.load_state_dict(selected["model_state"], strict=True)
    restored_exact = tensor_state_sha256(model.state_dict()) == tensor_state_sha256(selected["model_state"])
    final_state_hash = tensor_state_sha256(model.state_dict())
    component_changed = {name: initial_components[name] != tensor_state_sha256(getattr(model, name).state_dict())
                         for name in initial_components}
    if args.smoke and not all(component_changed.values()):
        raise RuntimeError(f"Force smoke found unchanged trainable blocks: {[k for k,v in component_changed.items() if not v]}")
    continuation = same_next_step(model, selected, next(iter(train_loader)), mean, std,
                                  torch.device(args.device)) if args.smoke else None
    summary = {
        "status": "complete", "format": "round5_force_training_summary_v1", **config,
        "epochs_completed": len(history), "best_epoch": selected["best_epoch"],
        "best_mean_rmse_xyz_native_n": selected["best_metric"], "history": history,
        "initial_state_sha256": initial_state_hash, "selected_state_sha256": final_state_hash,
        "force_parameters_changed": initial_state_hash != final_state_hash,
        "trainable_component_changed": component_changed,
        "finite_gradients_all_epochs": all(row["finite_gradients"] for row in history),
        "checkpoint_restore_exact": restored_exact, "normalization": {key: value.tolist() for key, value in normalization.items()},
        "same_next_step_proof": continuation,
        "best_checkpoint": str(best), "best_checkpoint_sha256": sha256_file(best),
        "latest_checkpoint": str(latest), "latest_checkpoint_sha256": sha256_file(latest),
        "provenance": provenance,
        "config_path": str(config_path), "config_sha256": sha256_file(config_path),
    }
    atomic_json(output / "training_summary.json", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
