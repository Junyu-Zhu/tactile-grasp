#!/usr/bin/env python3
"""Export the formal F-adapt fused contact interface for the HTT future assembler."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from common import atomic_json, configure_determinism, sha256_file, sha256_json
from data import load_cache, require_cache_audit
from metrics import apply_clip_standardize
from models import ForceConditionedSlip
from sources import load_decoupled_decoder, load_r3_slip_branch
from train_slip import load_force_predictions, raw_physical_conditions


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--base-slip-checkpoint", type=Path, required=True)
    parser.add_argument("--force-predictions", type=Path, required=True)
    parser.add_argument("--fused-slip-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--allow-smoke-parent", action="store_true")
    parser.add_argument("--allow-unverified-cache", action="store_true")
    return parser.parse_args()


def run(args):
    configure_determinism()
    cache = load_cache(args.cache.resolve())
    cache_audit = require_cache_audit(args.cache_audit, cache, args.allow_unverified_cache and args.allow_smoke_parent)
    fused_payload = torch.load(args.fused_slip_checkpoint.resolve(), map_location="cpu", weights_only=False)
    if fused_payload.get("format") != "round5_force_conditioned_slip_v1":
        raise RuntimeError("Unexpected fused-slip checkpoint")
    config = fused_payload["config"]
    if config["variant"] != "F-adapt" or config["fold"] != "htt_leave_p1" or config["seed"] != 20260914:
        raise RuntimeError("Future interface is preregistered to F-adapt leave-p1 seed 20260914")
    if config["smoke"] and not args.allow_smoke_parent:
        raise RuntimeError("Formal future interface refuses a smoke fused-slip parent")
    forces, force_manifest = load_force_predictions(args.force_predictions.resolve(), cache, "F-adapt",
                                                     config["fold"], config["seed"], args.allow_smoke_parent)
    normalization = fused_payload.get("condition_normalization")
    if normalization is None:
        raise RuntimeError("F-adapt checkpoint lacks embedded train-only condition normalization")
    epsilon = float(normalization["ratio_epsilon_n"])
    raw = raw_physical_conditions(forces, epsilon)
    array_normalization = {key: np.asarray(normalization[key], dtype=np.float32)
                           for key in ("clip_lower", "clip_upper", "mean", "std")}
    conditions = {episode_id: apply_clip_standardize(value, array_normalization)
                  for episode_id, value in raw.items()}
    decoder, _ = load_decoupled_decoder(args.source_checkpoint.resolve())
    base = load_r3_slip_branch(decoder, args.base_slip_checkpoint.resolve())
    model = ForceConditionedSlip(base, "F-adapt", hidden_dim=64)
    model.load_state_dict(fused_payload["model_state"], strict=True)
    model.eval().requires_grad_(False).to(args.device)
    output = args.output.resolve(); episode_dir = output / "episodes"; episode_dir.mkdir(parents=True, exist_ok=True)
    rows=[]
    for entry in sorted((row for row in cache["entries"] if row["episode_id"] in forces),key=lambda row:row["episode_id"]):
        tokens=np.load(entry["token_path"],mmap_mode="r",allow_pickle=False); condition=conditions[entry["episode_id"]]
        fused_chunks=[]; base_chunks=[]
        with torch.inference_mode():
            for start in range(0,len(tokens),args.batch_size):
                end=start+args.batch_size
                result=model(torch.from_numpy(np.array(tokens[start:end],copy=True)).to(args.device).float(),
                             torch.from_numpy(condition[start:end]).to(args.device).float(),return_aux=True)
                fused_chunks.append(torch.softmax(result["logits"],1)[:,1].cpu().numpy())
                base_chunks.append(torch.softmax(result["base_logits"],1)[:,1].cpu().numpy())
        fused=np.concatenate(fused_chunks).astype(np.float32); base_probability=np.concatenate(base_chunks).astype(np.float32)
        path=episode_dir/(entry["episode_id"].replace("/","__")+".npz")
        atomic_npz(path,frame_index=np.arange(entry["frames"],dtype=np.int64),
                   force_pred_n=forces[entry["episode_id"]].astype(np.float32),p_slip=fused,base_p_slip=base_probability)
        rows.append({"episode_id":entry["episode_id"],"roles_by_fold":entry["roles_by_fold"],"frames":entry["frames"],
                     "path":str(path),"sha256":sha256_file(path)})
    formal=not config["smoke"] and force_manifest["formal"]
    manifest={"status":"complete","format":"round5_fadapt_contact_predictions_v1","formal":formal,
              "fold":config["fold"],"seed":config["seed"],"p_slip_semantics":"F-adapt fused probability",
              "base_p_slip_semantics":"frozen matching Round3-B probability; diagnostic only",
              "force_semantics":"matching adapted HTT native shear_x/shear_y/normal N",
              "strict_future_history_start":13,"token_cache_sha256":cache["manifest_sha256"],
              "cache_audit_sha256":sha256_file(args.cache_audit.resolve()) if cache_audit else None,
              "force_prediction_manifest":force_manifest["manifest_path"],
              "force_prediction_manifest_sha256":force_manifest["manifest_sha256"],
              "fused_slip_checkpoint":str(args.fused_slip_checkpoint.resolve()),
              "fused_slip_checkpoint_sha256":sha256_file(args.fused_slip_checkpoint.resolve()),
              "fused_slip_config_sha256":sha256_json(config),
              "base_slip_checkpoint":str(args.base_slip_checkpoint.resolve()),
              "base_slip_checkpoint_sha256":sha256_file(args.base_slip_checkpoint.resolve()),
              "condition_normalization":{key:(value.tolist() if isinstance(value,np.ndarray) else value)
                                         for key,value in normalization.items()},
              "entries":rows,"source_code_sha256":sha256_file(Path(__file__))}
    if not formal and not args.allow_smoke_parent:
        raise RuntimeError("Refusing nonformal upstream manifest")
    atomic_json(output/"prediction_manifest.json",manifest); return manifest


if __name__=="__main__": print(json.dumps(run(parse_args()),indent=2))
