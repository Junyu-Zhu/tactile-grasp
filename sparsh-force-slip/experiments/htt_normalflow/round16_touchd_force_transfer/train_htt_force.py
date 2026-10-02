#!/usr/bin/env python3
"""Fresh symmetric H/T_H HTT force adaptation from the common source boundary."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler

from touchd_common import (SOURCE_CHECKPOINT, atomic_json, atomic_torch, fresh_adapter,
                           seed_all, sha256, state_sha256)


PROTOCOL = {"optimizer": "AdamW", "learning_rate": 1e-4, "weight_decay": 1e-4,
            "batch_size": 128, "max_epochs": 30, "patience": 7,
            "loss": "SmoothL1 beta=1 on HTT fit-standardized clip((6d_force-ref_force)[:3],-20,20) N",
            "selection": "earliest strict minimum internal-selection mean per-axis native-N RMSE"}


class Frames(Dataset):
    def __init__(self, entries):
        self.entries = entries; self.index = [(i, t) for i, entry in enumerate(entries) for t in range(5, entry["frames"])]
        self.tokens = {}; self.targets = {}
    def __len__(self): return len(self.index)
    def __getitem__(self, item):
        episode, frame = self.index[item]; entry = self.entries[episode]
        if episode not in self.tokens:
            self.tokens[episode] = np.load(entry["token_path"], mmap_mode="r")
            self.targets[episode] = np.load(entry["force_native_n_path"], mmap_mode="r")
        return (torch.from_numpy(np.array(self.tokens[episode][frame], copy=True)),
                torch.from_numpy(np.array(self.targets[episode][frame], copy=True)))


def capture(generator): return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "generator": generator.get_state()}
def restore(state, generator): torch.set_rng_state(state["torch"]); np.random.set_state(state["numpy"]); generator.set_state(state["generator"])


def loader(dataset, batch, shuffle, generator, workers):
    sampler = RandomSampler(dataset, generator=generator) if shuffle else None
    return DataLoader(dataset, batch_size=batch, sampler=sampler, shuffle=False, num_workers=workers,
                      generator=torch.Generator().manual_seed(0), pin_memory=True, persistent_workers=workers > 0)


def targets(entries): return np.concatenate([np.load(e["force_native_n_path"])[5:] for e in entries]).astype(np.float32)


def evaluate(model, data_loader, mean, std, device):
    actual, predicted = [], []; model.eval()
    with torch.inference_mode():
        for x, y in data_loader:
            predicted.append((model(x.to(device).float()) * std + mean).cpu()); actual.append(y)
    y = torch.cat(actual); p = torch.cat(predicted)
    return float(torch.sqrt(torch.mean((p - y) ** 2, dim=0)).mean())


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--route", choices=("H", "T_H"), required=True); parser.add_argument("--t-private", type=Path)
    parser.add_argument("--fold", choices=tuple(f"htt_leave_p{i}" for i in range(1,5)), required=True)
    parser.add_argument("--seed", type=int, choices=(20260914,20260915,20260916), required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=2); parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int); parser.add_argument("--interrupt-after-epoch", type=int)
    args = parser.parse_args(); seed_all(args.seed)
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("status") != "pass" or manifest["fold"] != args.fold: raise RuntimeError("manifest identity")
    if args.route == "T_H" and args.t_private is None: raise RuntimeError("T_H requires T private state")
    if args.route == "H" and args.t_private is not None: raise RuntimeError("H must not consume ToucHD")
    trunk_state = torch.load(args.t_private, map_location="cpu", weights_only=False) if args.t_private else None
    entries = manifest["entries"]
    fit_entries = [e for e in entries if e["role"] == "fit"]; selection_entries = [e for e in entries if e["role"] == "selection"]
    groups = {role: {e["leakage_group"] for e in entries if e["role"] == role} for role in ("fit","selection","calibration","validation")}
    if any(groups[a] & groups[b] for a in groups for b in groups if a != b): raise RuntimeError("role leakage")
    fit, selection = Frames(fit_entries), Frames(selection_entries)
    norm_targets = targets(fit_entries); mean_np = norm_targets.mean(0); std_np = norm_targets.std(0).clip(1e-6)
    device = torch.device(args.device); model = fresh_adapter(args.seed, trunk_state=trunk_state).to(device)
    reset_head_hash = state_sha256(model.force_head.state_dict()); initial = state_sha256(model.state_dict())
    initial_parts = {name: state_sha256(getattr(model,name).state_dict()) for name in ("force_pooler","force_trunk","force_head")}
    optimizer = torch.optim.AdamW(model.parameters(), lr=PROTOCOL["learning_rate"], weight_decay=PROTOCOL["weight_decay"])
    generator = torch.Generator().manual_seed(args.seed); epochs = args.epochs or PROTOCOL["max_epochs"]
    fit_loader = loader(fit, PROTOCOL["batch_size"], True, generator, args.workers)
    selection_loader = loader(selection, PROTOCOL["batch_size"], False, torch.Generator().manual_seed(0), args.workers)
    mean = torch.tensor(mean_np, device=device); std = torch.tensor(std_np, device=device)
    config = {**PROTOCOL, "route": args.route, "fold": args.fold, "seed": args.seed, "smoke": args.smoke,
              "epochs": epochs, "workers": args.workers, "manifest_path": str(args.manifest.resolve()),
              "manifest_sha256": sha256(args.manifest), "source_checkpoint": str(SOURCE_CHECKPOINT),
              "source_checkpoint_sha256": sha256(SOURCE_CHECKPOINT),
              "t_private": None if args.t_private is None else {"path": str(args.t_private.resolve()), "sha256": sha256(args.t_private)},
              "reset": "same-seed explicit force_head.reset_parameters after route-private pooler/trunk load",
              "transferred": ["source force_pooler","source force_trunk"] if args.route == "H" else ["ToucHD force_pooler","ToucHD force_trunk"]}
    args.output.mkdir(parents=True, exist_ok=True); cp=args.output/"config.json"
    if cp.exists() and json.loads(cp.read_text()) != config: raise RuntimeError("frozen config changed")
    atomic_json(cp,config); latest,best=args.output/"latest.pth",args.output/"best.pth"
    history=[]; best_metric=float("inf"); best_epoch=0; wait=0; start=1
    if latest.exists():
        saved=torch.load(latest,map_location="cpu",weights_only=False)
        if saved["config"] != config: raise RuntimeError("resume identity")
        model.load_state_dict(saved["model_state"]); optimizer.load_state_dict(saved["optimizer_state"]); restore(saved["rng_state"],generator)
        history,best_metric,best_epoch,wait=saved["history"],saved["best_metric"],saved["best_epoch"],saved["wait"]; start=saved["epoch"]+1
    if wait >= PROTOCOL["patience"]: start=epochs+1
    for epoch in range(start,epochs+1):
        began=time.monotonic(); model.train(); losses=[]
        for x,y in fit_loader:
            x=x.to(device).float(); y=((y.to(device)-mean)/std).float(); optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.smooth_l1_loss(model(x),y,beta=1.0)
            if not torch.isfinite(loss): raise RuntimeError("nonfinite loss")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()): raise RuntimeError("gradient")
            optimizer.step(); losses.append(float(loss.detach()))
        metric=evaluate(model,selection_loader,mean,std,device); row={"epoch":epoch,"train_loss":float(np.mean(losses)),"selection_mean_rmse":metric}
        history.append(row); improved=metric<best_metric
        if improved: best_metric,best_epoch,wait=metric,epoch,0
        else: wait+=1
        saved={"format":"round16_htt_force_adapter_v1","epoch":epoch,"model_state":model.state_dict(),"optimizer_state":optimizer.state_dict(),
               "rng_state":capture(generator),"history":history,"best_metric":best_metric,"best_epoch":best_epoch,"wait":wait,
               "config":config,"normalization":{"mean":mean_np,"std":std_np},"reset_head_sha256":reset_head_hash}
        atomic_torch(latest,saved)
        if improved: atomic_torch(best,saved)
        print(json.dumps({**row,"elapsed_seconds":time.monotonic()-began}),flush=True)
        if args.interrupt_after_epoch==epoch: raise RuntimeError("intentional committed interruption")
        if wait>=PROTOCOL["patience"]: break
    chosen=torch.load(best,map_location="cpu",weights_only=False); model.load_state_dict(chosen["model_state"])
    changed={name:initial_parts[name]!=state_sha256(getattr(model,name).state_dict()) for name in initial_parts}
    if not all(changed.values()): raise RuntimeError(f"unchanged component {changed}")
    summary={"status":"complete","schema":"round16_htt_force_summary_v1",**config,"initial_state_sha256":initial,
             "reset_head_sha256":reset_head_hash,"selected_state_sha256":state_sha256(model.state_dict()),"component_changed":changed,
             "best_epoch":chosen["best_epoch"],"best_metric":chosen["best_metric"],"history":history,
             "normalization":{"mean":mean_np.tolist(),"std":std_np.tolist()},"fit_samples":len(fit),"selection_samples":len(selection),
             "best_checkpoint":str(best),"best_checkpoint_sha256":sha256(best),"latest_checkpoint":str(latest),"latest_checkpoint_sha256":sha256(latest),
             "config_path":str(cp),"config_sha256":sha256(cp),"formal_does_not_use_smoke":not args.smoke}
    summary["output_hashes"]={str(best):sha256(best),str(latest):sha256(latest),str(cp):sha256(cp)}
    atomic_json(args.output/"training_summary.json",summary); print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
