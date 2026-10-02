#!/usr/bin/env python3
"""Create formal, episode-aligned force predictions for slip/future consumers."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from common import atomic_json, configure_determinism, sha256_file, sha256_json, tensor_state_sha256
from data import load_cache, require_cache_audit
from models import ForceAdapter, FrozenOldForce
from sources import load_decoupled_decoder


def validate_adapt_parent(parent: dict, args: argparse.Namespace, cache: dict, cache_audit: dict | None) -> None:
    if parent.get("format") != "round5_htt_native_force_adapter_v1":
        raise RuntimeError("Unexpected adapted-force parent checkpoint")
    config = parent.get("config", {})
    expected_source_sha = sha256_file(args.source_checkpoint.resolve())
    expected_audit_sha = sha256_file(args.cache_audit.resolve()) if cache_audit else None
    checks = {
        "fold": config.get("fold") == args.fold,
        "seed": config.get("seed") == args.seed,
        "cache_manifest": config.get("cache_manifest_sha256") == cache["manifest_sha256"],
        "cache_audit": config.get("cache_audit_sha256") == expected_audit_sha,
        "source_checkpoint": config.get("source_checkpoint_sha256") == expected_source_sha,
        "target_semantics": config.get("target") == "clip((6d_force-ref_force)[:3],-20,20) N; shear_x,shear_y,normal",
        "head_reset": parent.get("provenance", {}).get("force_output_head_reinitialized") is True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Adapted-force parent provenance mismatch: {[key for key,value in checks.items() if not value]}")
    normalization = parent.get("normalization", {})
    for key in ("mean", "std"):
        value = np.asarray(normalization.get(key))
        if value.shape != (3,) or not np.isfinite(value).all() or (key == "std" and np.any(value < 1e-6)):
            raise RuntimeError(f"Invalid adapted-force parent normalization: {key}")


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path)
    parser.add_argument("--source-checkpoint", type=Path, required=True,
                        help="Historical epoch-30 MAE decoupled multitask checkpoint")
    parser.add_argument("--variant", choices=("old", "adapt"), required=True)
    parser.add_argument("--force-checkpoint", type=Path, help="Required for adapt")
    parser.add_argument("--fold", choices=tuple(f"htt_leave_p{i}" for i in range(1, 5)), required=True)
    parser.add_argument("--seed", type=int, choices=(20260914, 20260915, 20260916), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--allow-smoke-parent", action="store_true")
    parser.add_argument("--allow-unverified-cache", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    configure_determinism()
    cache = load_cache(args.cache.resolve())
    cache_audit = require_cache_audit(args.cache_audit, cache, args.allow_unverified_cache and args.allow_smoke_parent)
    decoder, source_payload = load_decoupled_decoder(args.source_checkpoint.resolve())
    parent = None
    if args.variant == "old":
        model = FrozenOldForce(decoder)
        normalization = None
    else:
        if args.force_checkpoint is None:
            raise ValueError("--force-checkpoint is required for adapt")
        parent = torch.load(args.force_checkpoint.resolve(), map_location="cpu", weights_only=False)
        validate_adapt_parent(parent, args, cache, cache_audit)
        config = parent["config"]
        if config["smoke"] and not args.allow_smoke_parent:
            raise RuntimeError("Formal predictions refuse a smoke force parent")
        model = ForceAdapter(decoder)
        model.load_state_dict(parent["model_state"], strict=True)
        normalization = parent["normalization"]
    model.eval().requires_grad_(False).to(args.device)
    output = args.output.resolve()
    episode_dir = output / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in sorted((entry for entry in cache["entries"]
                         if entry["task"] == "slip" and entry["roles_by_fold"][args.fold] != "test"),
                        key=lambda item: item["episode_id"]):
        tokens = np.load(entry["token_path"], mmap_mode="r", allow_pickle=False)
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(tokens), args.batch_size):
                batch = torch.from_numpy(np.array(tokens[start:start + args.batch_size], copy=True)).to(args.device)
                prediction = model(batch.float())
                if normalization is not None:
                    mean = torch.as_tensor(normalization["mean"], device=args.device)
                    std = torch.as_tensor(normalization["std"], device=args.device)
                    prediction = prediction * std + mean
                chunks.append(prediction.float().cpu().numpy())
        values = np.concatenate(chunks).astype(np.float32)
        if values.shape != (entry["frames"], 3) or not np.isfinite(values).all():
            raise RuntimeError(f"Invalid force prediction: {entry['episode_id']}")
        key = entry["episode_id"].replace("/", "__")
        path = episode_dir / f"{key}.force_pred_n.npy"
        atomic_npy(path, values)
        rows.append({"episode_id": entry["episode_id"], "roles_by_fold": entry["roles_by_fold"],
                     "frames": entry["frames"], "prediction_path": str(path),
                     "prediction_sha256": sha256_file(path)})
    formal = parent is None or not parent["config"]["smoke"]
    manifest = {
        "status": "complete", "format": "round5_force_predictions_v1", "variant": args.variant,
        "fold": args.fold, "seed": args.seed, "formal": formal,
        "cache_manifest": cache["manifest_path"], "cache_manifest_sha256": cache["manifest_sha256"],
        "cache_audit_sha256": sha256_file(args.cache_audit.resolve()) if cache_audit else None,
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_checkpoint_sha256": sha256_file(args.source_checkpoint.resolve()),
        "source_epoch": source_payload.get("epoch"), "model_state_sha256": tensor_state_sha256(model.state_dict()),
        "force_checkpoint": str(args.force_checkpoint.resolve()) if args.force_checkpoint else None,
        "force_checkpoint_sha256": sha256_file(args.force_checkpoint.resolve()) if args.force_checkpoint else None,
        "force_parent_config_sha256": sha256_json(parent["config"]) if parent else None,
        "coordinate_statement": ("HTT native shear_x/shear_y/normal N" if args.variant == "adapt" else
                                 "historical predicted axes scaled by [1.5,1.5,2.0] N; signed HTT alignment unverified"),
        "entries": rows, "source_code_sha256": sha256_file(Path(__file__)),
    }
    atomic_json(output / "prediction_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
