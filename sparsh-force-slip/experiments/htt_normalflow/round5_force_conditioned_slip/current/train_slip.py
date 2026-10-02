#!/usr/bin/env python3
"""Train resumable V/F-old/F-adapt identity-initialized slip fusion heads."""
from __future__ import annotations

import argparse
import copy
import csv
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
from data import SlipFrames, load_cache, require_cache_audit, role_entries, stratified_smoke_indices
from metrics import apply_clip_standardize, fit_clip_standardize, partial_tpr_auc_0_0p1
from models import ForceConditionedSlip
from sources import load_decoupled_decoder, load_r3_slip_branch


PROTOCOL = {
    "optimizer": "AdamW", "learning_rate": 1e-4, "weight_decay": 1e-4,
    "batch_size": 128, "max_epochs": 30, "patience": 7,
    "loss": "train-only balanced weighted cross entropy, static vs gross",
    "selection": "earliest strict maximum validation partial_tpr_auc_0_0p1",
    "strict_start": 10, "incipient": "excluded from loss/selection; retained in predictions",
    "condition_clip_quantiles": [0.005, 0.995], "condition_hidden_dim": 64,
}


def load_force_predictions(path: Path, cache: dict, variant: str, fold: str, seed: int,
                           allow_smoke_parent: bool) -> tuple[dict[str, np.ndarray], dict]:
    manifest_path = path / "prediction_manifest.json" if path.is_dir() else path
    payload = json.loads(manifest_path.read_text())
    expected_variant = "old" if variant == "F-old" else "adapt"
    if payload.get("status") != "complete" or payload.get("format") != "round5_force_predictions_v1":
        raise RuntimeError("Incomplete force prediction cache")
    if payload.get("variant") != expected_variant or payload.get("fold") != fold or payload.get("seed") != seed:
        raise RuntimeError("Force prediction identity mismatch")
    if payload.get("cache_manifest_sha256") != cache["manifest_sha256"]:
        raise RuntimeError("Force predictions use a different token cache")
    if not payload.get("formal") and not allow_smoke_parent:
        raise RuntimeError("Formal slip training refuses smoke force predictions")
    expected = {entry["episode_id"]: entry for entry in cache["entries"]
                if entry["task"] == "slip" and entry["roles_by_fold"][fold] != "test"}
    rows = {entry["episode_id"]: entry for entry in payload.get("entries", [])}
    if set(rows) != set(expected):
        raise RuntimeError("Force prediction episode set mismatch")
    values = {}
    for episode_id, row in rows.items():
        path = Path(row["prediction_path"])
        if sha256_file(path) != row["prediction_sha256"]:
            raise RuntimeError(f"Force prediction hash mismatch: {episode_id}")
        array = np.load(path, allow_pickle=False)
        if array.dtype != np.float32 or array.shape != (expected[episode_id]["frames"], 3) or not np.isfinite(array).all():
            raise RuntimeError(f"Force prediction schema mismatch: {episode_id}")
        values[episode_id] = array
    payload["manifest_path"] = str(manifest_path.resolve())
    payload["manifest_sha256"] = sha256_file(manifest_path)
    return values, payload


def raw_physical_conditions(forces: dict[str, np.ndarray], epsilon: float) -> dict[str, np.ndarray]:
    result = {}
    for episode_id, force in forces.items():
        previous = force[np.maximum(np.arange(len(force)) - 5, 0)]
        fn, previous_fn = np.abs(force[:, 2]), np.abs(previous[:, 2])
        ft, previous_ft = np.linalg.norm(force[:, :2], axis=1), np.linalg.norm(previous[:, :2], axis=1)
        result[episode_id] = np.stack((fn, ft, ft / (fn + epsilon), fn - previous_fn, ft - previous_ft), axis=1).astype(np.float32)
    return result


def fit_condition_data(forces: dict[str, np.ndarray], train_entries: list[dict]) -> tuple[dict[str, np.ndarray], dict]:
    train_ids = {entry["episode_id"] for entry in train_entries}
    train_fn = np.concatenate([np.abs(forces[episode_id][10:, 2]) for episode_id in train_ids])
    epsilon = max(1e-3, float(np.quantile(train_fn, 0.01)))
    raw = raw_physical_conditions(forces, epsilon)
    train_rows = []
    for entry in train_entries:
        labels = np.load(entry["label_path"], allow_pickle=False)
        valid = np.flatnonzero((np.arange(len(labels)) >= 10) & np.isin(labels, (0, 2)))
        train_rows.append(raw[entry["episode_id"]][valid])
    normalization = fit_clip_standardize(np.concatenate(train_rows))
    normalized = {episode_id: apply_clip_standardize(value, normalization) for episode_id, value in raw.items()}
    normalization = {**normalization, "ratio_epsilon_n": epsilon}
    return normalized, normalization


def loader(dataset, shuffle: bool, generator: torch.Generator, workers: int):
    sampler = RandomSampler(dataset, generator=generator) if shuffle else None
    worker_generator = torch.Generator().manual_seed(0)
    return DataLoader(dataset, batch_size=PROTOCOL["batch_size"], shuffle=False, sampler=sampler,
                      generator=worker_generator,
                      num_workers=workers, pin_memory=torch.cuda.is_available(), drop_last=False,
                      persistent_workers=workers > 0)


def evaluate(model, data_loader, device) -> dict:
    labels, probabilities = [], []
    model.eval()
    with torch.inference_mode():
        for tokens, condition, binary, *_ in data_loader:
            condition = None if model.variant == "V" else condition.to(device).float()
            logits = model(tokens.to(device).float(), condition)
            labels.extend(binary.numpy().tolist())
            probabilities.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy().tolist())
    return {"partial_tpr_auc_0_0p1": partial_tpr_auc_0_0p1(np.asarray(labels), np.asarray(probabilities))}


def same_next_step(model, saved, batch, criterion, device) -> dict[str, bool]:
    left, right = copy.deepcopy(model).to(device).train(), copy.deepcopy(model).to(device).train()
    right.load_state_dict(saved["model_state"], strict=True)
    opts = [torch.optim.AdamW(candidate.trainable_parameters(), lr=PROTOCOL["learning_rate"],
                              weight_decay=PROTOCOL["weight_decay"]) for candidate in (left, right)]
    opts[0].load_state_dict(copy.deepcopy(saved["optimizer_state"]))
    opts[1].load_state_dict(copy.deepcopy(saved["optimizer_state"]))
    tokens, condition, binary = batch[0].to(device).float(), batch[1].to(device).float(), batch[2].to(device)
    outputs = []
    for candidate, optimizer in zip((left, right), opts):
        optimizer.zero_grad(set_to_none=True)
        selected_condition = None if candidate.variant == "V" else condition
        loss = criterion(candidate(tokens, selected_condition), binary)
        loss.backward()
        gradients = [parameter.grad.detach().cpu().clone() for parameter in candidate.parameters() if parameter.requires_grad]
        optimizer.step()
        outputs.append((loss.detach().cpu(), gradients))
    left_opt, right_opt = optimizer_tensor_values(opts[0]), optimizer_tensor_values(opts[1])
    proof = {"loss_exact":torch.equal(outputs[0][0],outputs[1][0]),
             "gradients_exact":all(torch.equal(a,b) for a,b in zip(outputs[0][1],outputs[1][1])),
             "parameters_exact":all(torch.equal(a,b) for a,b in zip(left.state_dict().values(),right.state_dict().values())),
             "optimizer_exact":len(left_opt)==len(right_opt) and all(torch.equal(a,b) for a,b in zip(left_opt,right_opt))}
    proof["pass"]=all(proof.values()); return proof


def write_predictions(model, entries, conditions, role, output, device, workers):
    dataset = SlipFrames(entries, conditions, strict_start=10, primary_only=False)
    data_loader = loader(dataset, False, torch.Generator().manual_seed(0), workers)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("episode_id", "frame", "stage", "role", "probability", "base_probability"))
        model.eval()
        with torch.inference_mode():
            for tokens, condition, _, episode_ids, frames, stages in data_loader:
                condition = None if model.variant == "V" else condition.to(device).float()
                result = model(tokens.to(device).float(), condition, return_aux=True)
                probability = torch.softmax(result["logits"], 1)[:, 1].cpu().numpy()
                base = torch.softmax(result["base_logits"], 1)[:, 1].cpu().numpy()
                writer.writerows(zip(episode_ids, frames.tolist(), stages.tolist(), [role] * len(frames), probability, base))
    temporary.replace(output)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--base-slip-checkpoint", type=Path, required=True)
    parser.add_argument("--variant", choices=("V", "F-old", "F-adapt"), required=True)
    parser.add_argument("--force-predictions", type=Path)
    parser.add_argument("--fold", choices=tuple(f"htt_leave_p{i}" for i in range(1, 5)), required=True)
    parser.add_argument("--seed", type=int, choices=(20260914, 20260915, 20260916), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-smoke-parent", action="store_true")
    parser.add_argument("--allow-unverified-cache", action="store_true")
    parser.add_argument("--interrupt-after-epoch", type=int)
    return parser.parse_args()


def run(args):
    configure_determinism(); seed_all(args.seed)
    cache = load_cache(args.cache.resolve())
    cache_audit = require_cache_audit(args.cache_audit, cache, args.allow_unverified_cache and args.smoke)
    train_entries = role_entries(cache, args.fold, "train", "slip")
    validation_entries = role_entries(cache, args.fold, "validation", "slip")
    calibration_entries = role_entries(cache, args.fold, "calibration", "slip")
    conditions, condition_normalization, condition_manifest = None, None, None
    if args.variant != "V":
        if args.force_predictions is None:
            raise ValueError("Force variants require --force-predictions")
        forces, condition_manifest = load_force_predictions(args.force_predictions.resolve(), cache, args.variant,
                                                             args.fold, args.seed, args.allow_smoke_parent)
        conditions, condition_normalization = fit_condition_data(forces, train_entries)
    decoder, _ = load_decoupled_decoder(args.source_checkpoint.resolve())
    base_branch = load_r3_slip_branch(decoder, args.base_slip_checkpoint.resolve())
    base_payload = torch.load(args.base_slip_checkpoint.resolve(), map_location="cpu", weights_only=False)
    base_config = base_payload["config"]
    if base_config["fold"] != args.fold or base_config["seed"] != args.seed or (base_config.get("smoke") and not args.allow_smoke_parent):
        raise RuntimeError("Base R3-B slip parent is not the matching formal fold/seed")
    model = ForceConditionedSlip(base_branch, args.variant, hidden_dim=64).to(args.device)
    frozen_hash_before = tensor_state_sha256(model.base.state_dict())
    initial_trainable = {name: value.detach().cpu().clone() for name, value in model.named_parameters() if value.requires_grad}
    train_data = SlipFrames(train_entries, conditions)
    validation_data = SlipFrames(validation_entries, conditions)
    if args.smoke:
        train_data = Subset(train_data, stratified_smoke_indices(train_data, 128))
        validation_data = Subset(validation_data, stratified_smoke_indices(validation_data, 128))
    labels = np.array([train_data[index][2] for index in range(len(train_data))])
    counts = np.bincount(labels, minlength=2)
    weights = torch.tensor(len(labels) / (2 * counts), device=args.device, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=PROTOCOL["learning_rate"], weight_decay=PROTOCOL["weight_decay"])
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = loader(train_data, True, generator, args.workers)
    validation_loader = loader(validation_data, False, torch.Generator().manual_seed(0), args.workers)
    config = {**PROTOCOL, "variant": args.variant, "fold": args.fold, "seed": args.seed,
              "smoke": args.smoke, "workers": args.workers, "cache_manifest_sha256": cache["manifest_sha256"],
              "cache_audit_sha256": sha256_file(args.cache_audit.resolve()) if cache_audit else None,
              "source_checkpoint_sha256": sha256_file(args.source_checkpoint.resolve()),
              "base_slip_checkpoint": str(args.base_slip_checkpoint.resolve()),
              "base_slip_checkpoint_sha256": sha256_file(args.base_slip_checkpoint.resolve()),
              "force_prediction_manifest_sha256": condition_manifest["manifest_sha256"] if condition_manifest else None}
    current_dir=Path(__file__).resolve().parent
    config["source_bundle"]=source_bundle([
        Path(__file__),current_dir/"common.py",current_dir/"data.py",current_dir/"metrics.py",
        current_dir/"models.py",current_dir/"sources.py",
        Path(sys.modules[load_decoupled_decoder.__module__].p2.__file__),
    ])
    provenance = {"source_code_sha256": sha256_file(Path(__file__)), "parameter_boundary": parameter_boundary(model),
                  "base_frozen_state_sha256": frozen_hash_before, "force_is_external_frozen_prediction": args.variant != "V",
                  "visual_conditioner_extra_parameters": 965 if args.variant == "V" else 0}
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    config_path=output/"config.json"
    if config_path.exists() and json.loads(config_path.read_text())!=config:
        raise RuntimeError("Refusing to change a frozen slip run config")
    atomic_json(config_path,config)
    latest, best = output / "latest.pth", output / "best.pth"
    history, best_epoch, best_metric, wait, start = [], 0, -float("inf"), 0, 1
    epochs, patience = (3, 3) if args.smoke else (PROTOCOL["max_epochs"], PROTOCOL["patience"])
    if latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if saved.get("format") != "round5_force_conditioned_slip_v1" or saved["config"] != config:
            raise RuntimeError("Refusing incompatible slip checkpoint resume")
        saved_normalization = saved.get("condition_normalization")
        if (saved_normalization is None) != (condition_normalization is None):
            raise RuntimeError("Condition normalization presence changed")
        if condition_normalization is not None:
            for key, value in condition_normalization.items():
                if isinstance(value, np.ndarray):
                    if not np.array_equal(saved_normalization[key], value):
                        raise RuntimeError(f"Condition normalization changed: {key}")
                elif saved_normalization[key] != value:
                    raise RuntimeError(f"Condition normalization changed: {key}")
        model.load_state_dict(saved["model_state"], strict=True); optimizer.load_state_dict(saved["optimizer_state"])
        restore_rng(saved["rng_state"], generator)
        history, best_epoch, best_metric, wait = saved["history"], saved["best_epoch"], saved["best_metric"], saved["patience_wait"]
        start = saved["epoch"] + 1
    for epoch in range(start, epochs + 1):
        model.train(); losses=[]; finite=True
        for tokens, condition, binary, *_ in train_loader:
            condition = None if args.variant == "V" else condition.to(args.device).float()
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(tokens.to(args.device).float(), condition), binary.to(args.device))
            if not torch.isfinite(loss): raise RuntimeError("Nonfinite slip loss")
            loss.backward()
            finite &= all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                          for parameter in model.parameters() if parameter.requires_grad)
            if not finite: raise RuntimeError("Missing or nonfinite slip gradient")
            optimizer.step(); losses.append(float(loss.detach()))
        metrics = evaluate(model, validation_loader, args.device)
        row={"epoch":epoch,"train_loss":float(np.mean(losses)),"finite_gradients":finite,**metrics}; history.append(row)
        improved=metrics["partial_tpr_auc_0_0p1"]>best_metric
        if improved: best_metric,best_epoch,wait=metrics["partial_tpr_auc_0_0p1"],epoch,0
        else: wait+=1
        payload={"format":"round5_force_conditioned_slip_v1","epoch":epoch,"model_state":model.state_dict(),
                 "optimizer_state":optimizer.state_dict(),"rng_state":capture_rng(generator),"history":history,
                 "best_epoch":best_epoch,"best_metric":best_metric,"patience_wait":wait,"config":config,
                 "condition_normalization":copy.deepcopy(condition_normalization),"provenance":provenance}
        atomic_torch(latest,payload)
        if improved: atomic_torch(best,payload)
        print(json.dumps(row),flush=True)
        if args.interrupt_after_epoch==epoch: raise RuntimeError(f"intentional interruption after committed epoch {epoch}")
        if wait>=patience: break
    latest_selected=torch.load(latest,map_location="cpu",weights_only=False)
    latest_state=latest_selected["model_state"]
    trainable_changed={name:not torch.equal(initial_trainable[name],latest_state[name].detach().cpu())
                       for name,value in model.named_parameters() if value.requires_grad}
    if args.smoke and not all(trainable_changed.values()):
        raise RuntimeError(f"Smoke found unchanged trainable tensors after all steps: {[k for k,v in trainable_changed.items() if not v]}")
    selected=torch.load(best,map_location="cpu",weights_only=False); model.load_state_dict(selected["model_state"],strict=True)
    if tensor_state_sha256(model.base.state_dict()) != frozen_hash_before:
        raise RuntimeError("Frozen base slip branch changed")
    continuation=same_next_step(model,selected,next(iter(train_loader)),criterion,args.device) if args.smoke else None
    predictions={}
    for role,entries in (("calibration",calibration_entries),("validation",validation_entries)):
        path=output/f"predictions_{role}.csv"; write_predictions(model,entries,conditions,role,path,args.device,args.workers)
        predictions[role]={"path":str(path),"sha256":sha256_file(path)}
    summary={"status":"complete","format":"round5_slip_training_summary_v1",**config,
             "epochs_completed":len(history),"best_epoch":selected["best_epoch"],
             "best_partial_tpr_auc_0_0p1":selected["best_metric"],"history":history,
             "all_trainable_tensors_changed":all(trainable_changed.values()),"trainable_changed":trainable_changed,
             "finite_gradients_all_epochs":all(row["finite_gradients"] for row in history),
             "frozen_base_unchanged":True,"checkpoint_restore_exact":tensor_state_sha256(model.state_dict())==tensor_state_sha256(selected["model_state"]),
             "same_next_step_proof":continuation,
             "class_counts_train":counts.tolist(),"class_weights_train":weights.cpu().tolist(),
             "condition_normalization":({key:(value.tolist() if isinstance(value,np.ndarray) else value)
                                         for key,value in condition_normalization.items()} if condition_normalization else None),
             "best_checkpoint":str(best),"best_checkpoint_sha256":sha256_file(best),
             "latest_checkpoint":str(latest),"latest_checkpoint_sha256":sha256_file(latest),
             "predictions":predictions,"provenance":provenance,
             "config_path":str(config_path),"config_sha256":sha256_file(config_path)}
    atomic_json(output/"training_summary.json",summary); return summary


if __name__=="__main__": print(json.dumps(run(parse_args()),indent=2))
