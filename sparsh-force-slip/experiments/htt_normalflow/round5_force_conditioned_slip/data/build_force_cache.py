#!/usr/bin/env python3
"""Build resumable full MAE tokens and same-frame native HTT force targets.

This entry point is GPU-capable but does not launch work on import.  It mirrors
the audited Round-3 MAE preprocessing and emits only force-task episodes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXP_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import adapters  # noqa: E402
import phase2_b_multitask as p2  # noqa: E402
from contract import FORCE_AXES, FORCE_UNIT, FOLDS, atomic_json, roles_by_fold, sha256_file  # noqa: E402
from prepare_splits import verify as verify_split  # noqa: E402


FORMAT = "round5_htt_force_full_mae_tokens_v1"


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def atomic_npy(path: Path, value: np.ndarray) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    os.replace(tmp, path)


def key_for(episode_id: str) -> str:
    return hashlib.sha256(episode_id.encode()).hexdigest()[:20]


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


def selected_rows(manifest: dict[str, Any], fold: str | None, role: str | None) -> list[dict[str, Any]]:
    rows = {row["id"]: row for row in manifest["episodes"]}
    if fold is None:
        ids = {
            episode_id
            for fold_name in FOLDS
            for development_role in ("train", "validation", "calibration")
            for episode_id in manifest["splits"][fold_name][development_role]
            if rows[episode_id].get("task") == "force"
        }
    else:
        ids = {episode_id for episode_id in manifest["splits"][fold][role] if rows[episode_id].get("task") == "force"}
    return [rows[episode_id] for episode_id in sorted(ids)]


def build(args: argparse.Namespace) -> dict[str, Any]:
    split = json.loads(args.split_manifest.read_text())
    verify_split(split)
    if split.get("schema_version") != 2:
        raise ValueError("schema-2 split required")
    rows = selected_rows(split, args.fold, args.role)
    selected_total = len(rows)
    if args.shards > 1:
        rows = [row for index, row in enumerate(rows) if index % args.shards == args.shard]
    if args.max_episodes is not None:
        rows = rows[:args.max_episodes]
    if not rows:
        raise ValueError("no force episodes selected")

    configure_determinism()
    device = torch.device(args.device)
    model, checkpoint_payload = p2.load_b_checkpoint(args.checkpoint, device)
    if model.encoder_name != "mae" or model.decoder_variant != "decoupled":
        raise ValueError("selected Round-3 MAE decoupled checkpoint required")
    model.encoder.eval().requires_grad_(False)
    encoder_sha = tensor_state_sha256(model.encoder.state_dict())
    checkpoint_sha = sha256_file(args.checkpoint)
    split_sha = sha256_file(args.split_manifest)
    output = args.output_dir
    episode_dir = output / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)
    code_sha = {path.name: sha256_file(path) for path in (Path(__file__), Path(adapters.__file__), Path(p2.__file__))}

    entries = []
    for index, row in enumerate(rows, 1):
        episode = adapters.load_htt(row["path"])
        if episode.labels is not None or row.get("task") != "force":
            raise ValueError(f"{row['id']}: force cache must not load slip supervision")
        native_force = np.asarray(episode.force[:, :3], dtype=np.float32)
        if native_force.shape != (len(episode.images), 3) or not np.isfinite(native_force).all():
            raise ValueError(f"{row['id']}: invalid native force")
        key = key_for(row["id"])
        token_path = episode_dir / f"{key}.tokens.npy"
        force_path = episode_dir / f"{key}.force_native_n.npy"
        meta_path = episode_dir / f"{key}.json"
        expected = {
            "episode_id": row["id"], "task": "force", "frames": len(episode.images),
            "roles_by_fold": roles_by_fold(split, row["id"]), "source_files": row["source_files"],
            "manifest_sha256": split_sha, "checkpoint_sha256": checkpoint_sha,
            "encoder_state_sha256": encoder_sha, "history": 2, "stride": 5,
            "force_axes": list(FORCE_AXES), "force_unit": FORCE_UNIT,
            "force_source": "6d_force[:, :3] unchanged", "subtract_ref_force": False,
            "dtype": "float32", "code_sha256": code_sha,
        }
        reusable = False
        if token_path.exists() and force_path.exists() and meta_path.exists():
            old = json.loads(meta_path.read_text())
            tokens = np.load(token_path, mmap_mode="r", allow_pickle=False)
            forces = np.load(force_path, mmap_mode="r", allow_pickle=False)
            reusable = all(old.get(k) == v for k, v in expected.items())
            reusable &= tokens.dtype == forces.dtype == np.float32
            reusable &= tokens.ndim == 3 and tokens.shape[0] == len(episode.images)
            reusable &= forces.shape == (len(episode.images), 3)
            if reusable:
                reusable &= old.get("token_sha256") == sha256_file(token_path)
                reusable &= old.get("force_native_n_sha256") == sha256_file(force_path)
        if not reusable:
            if meta_path.exists():
                raise RuntimeError(f"refusing incompatible committed cache: {row['id']}")
            token_path.unlink(missing_ok=True)
            force_path.unlink(missing_ok=True)
            chunks, pending = [], []
            for t in range(len(episode.images)):
                pending.append(adapters.window(episode, t)["inputs"]["image"])
                if len(pending) == args.batch_size or t + 1 == len(episode.images):
                    with torch.inference_mode():
                        tokens = model.encoder(torch.stack(pending).to(device, non_blocking=True))
                    if tokens.ndim != 3 or not torch.isfinite(tokens).all():
                        raise RuntimeError(f"{row['id']}: invalid encoder output")
                    chunks.append(tokens.detach().float().cpu().numpy())
                    pending.clear()
            atomic_npy(token_path, np.concatenate(chunks))
            atomic_npy(force_path, native_force)
            expected["token_sha256"] = sha256_file(token_path)
            expected["force_native_n_sha256"] = sha256_file(force_path)
            atomic_json(meta_path, expected)
        else:
            expected["token_sha256"] = old["token_sha256"]
            expected["force_native_n_sha256"] = old["force_native_n_sha256"]
        entries.append({**expected, "token_path": str(token_path), "force_native_n_path": str(force_path),
                        "token_shape": list(np.load(token_path, mmap_mode="r", allow_pickle=False).shape),
                        "status": "reused" if reusable else "built"})
        print(f"[{index}/{len(rows)}] {row['id']}: {entries[-1]['status']}", flush=True)

    status = "complete" if args.fold is None and args.max_episodes is None and args.shards == 1 else "partial_smoke"
    result = {
        "format": FORMAT, "status": status, "manifest": str(args.split_manifest),
        "manifest_sha256": split_sha, "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha, "checkpoint_epoch": checkpoint_payload.get("epoch"),
        "encoder": "mae", "encoder_state_sha256": encoder_sha,
        "preprocess": "Round-3 adapters.preprocess; current-first images [t,t-5]; no augmentation",
        "expected_unique_force_episodes": selected_total, "unique_force_episodes": len(entries),
        "frames": sum(entry["frames"] for entry in entries), "shards": args.shards, "shard": args.shard,
        "entries": entries,
    }
    name = "cache_manifest.json" if args.shards == 1 else f"cache_manifest.part-{args.shard:03d}-of-{args.shards:03d}.json"
    atomic_json(output / name, result)
    print(json.dumps({"status": status, "episodes": len(entries), "frames": result["frames"],
                      "encoder_state_sha256": encoder_sha}, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--fold", choices=FOLDS)
    parser.add_argument("--role", choices=("train", "validation", "calibration"))
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    args = parser.parse_args()
    if (args.fold is None) != (args.role is None):
        parser.error("--fold and --role must be supplied together")
    if not (1 <= args.batch_size and 1 <= args.shards and 0 <= args.shard < args.shards):
        parser.error("invalid batch/shard arguments")
    build(args)


if __name__ == "__main__":
    main()

