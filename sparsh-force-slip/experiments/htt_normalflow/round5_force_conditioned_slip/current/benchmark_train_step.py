#!/usr/bin/env python3
"""Measure representative force-training steps without creating a scientific run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from common import atomic_json, configure_determinism, parameter_boundary, seed_all, sha256_file
from data import ForceFrames, force_train_targets, load_cache, require_cache_audit, role_entries
from metrics import force_target_normalization
from models import ForceAdapter
from sources import load_decoupled_decoder


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--cache",type=Path,required=True); parser.add_argument("--cache-audit",type=Path,required=True)
    parser.add_argument("--source-checkpoint",type=Path,required=True); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--fold",default="htt_leave_p1"); parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--workers",type=int,default=2); parser.add_argument("--warmup",type=int,default=2)
    parser.add_argument("--steps",type=int,default=10); args=parser.parse_args()
    if args.steps<3 or args.warmup<1: parser.error("use at least one warmup and three measured steps")
    configure_determinism(); seed_all(20260914)
    cache=load_cache(args.cache.resolve()); require_cache_audit(args.cache_audit.resolve(),cache,False)
    entries=role_entries(cache,args.fold,"train","force"); dataset=ForceFrames(entries)
    loader=DataLoader(dataset,batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(20260914),
                      num_workers=args.workers,pin_memory=True,persistent_workers=args.workers>0,drop_last=True)
    decoder,_=load_decoupled_decoder(args.source_checkpoint.resolve()); model=ForceAdapter(decoder).to(args.device).train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4); criterion=nn.SmoothL1Loss(beta=1)
    norm=force_target_normalization(force_train_targets(entries)); mean=torch.tensor(norm["mean"],device=args.device);std=torch.tensor(norm["std"],device=args.device)
    iterator=iter(loader); compute_durations=[]; loader_wait_durations=[]; wall_durations=[]; finite=True
    torch.cuda.reset_peak_memory_stats(args.device)
    for index in range(args.warmup+args.steps):
        wall_start=time.perf_counter()
        loader_start=wall_start
        try: tokens,target,*_=next(iterator)
        except StopIteration: iterator=iter(loader);tokens,target,*_=next(iterator)
        loader_wait_ms=(time.perf_counter()-loader_start)*1000
        torch.cuda.synchronize(args.device);compute_start=time.perf_counter()
        optimizer.zero_grad(set_to_none=True);loss=criterion(model(tokens.to(args.device).float()),(target.to(args.device).float()-mean)/std)
        loss.backward();finite &= bool(torch.isfinite(loss)) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters());optimizer.step()
        torch.cuda.synchronize(args.device)
        compute_ms=(time.perf_counter()-compute_start)*1000
        wall_ms=(time.perf_counter()-wall_start)*1000
        if index>=args.warmup:
            compute_durations.append(compute_ms);loader_wait_durations.append(loader_wait_ms);wall_durations.append(wall_ms)
    result={"status":"pass" if finite else "fail","scope":"non-scientific in-memory batch128 force forward+backward+AdamW step",
            "fold":args.fold,"batch_size":128,"warmup_steps":args.warmup,"measured_steps":args.steps,
            "compute_step_ms":compute_durations,"loader_wait_ms":loader_wait_durations,"wall_step_ms":wall_durations,
            "median_compute_step_ms":statistics.median(compute_durations),
            "median_loader_wait_ms":statistics.median(loader_wait_durations),
            "median_wall_step_ms":statistics.median(wall_durations),
            "examples_per_second_wall":128/(statistics.median(wall_durations)/1000),
            "peak_allocated_bytes":torch.cuda.max_memory_allocated(args.device),"finite_loss_and_gradients":finite,
            "parameter_boundary":parameter_boundary(model),"cache_manifest_sha256":cache["manifest_sha256"],
            "cache_audit_sha256":sha256_file(args.cache_audit.resolve()),"source_checkpoint_sha256":sha256_file(args.source_checkpoint.resolve()),
            "gpu_condition":subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,name,memory.used,utilization.gpu","--format=csv"],text=True),
            "source_code_sha256":sha256_file(Path(__file__))}
    atomic_json(args.output.resolve(),result);print(json.dumps({k:v for k,v in result.items() if not isinstance(v,list)},indent=2))
    if not finite: raise SystemExit(1)


if __name__=="__main__":main()
