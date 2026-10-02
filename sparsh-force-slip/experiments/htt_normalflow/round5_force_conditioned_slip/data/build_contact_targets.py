#!/usr/bin/env python3
"""Derive official HTT ref-relative, clipped force targets without touching tokens.

The original force-cache run stored raw ``6d_force[:, :3]`` targets.  This
adapter preserves that cache and creates a separate target/cache manifest whose
token files and hashes still point to the original encoder run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from contract import DEVELOPMENT_ROLES, FORCE_AXES, FOLDS, atomic_json, sha256_file


FORMAT = "round5_htt_force_ref_relative_clipped_targets_v1"
CLIP_N = (-20.0, 20.0)
OFFICIAL_SOURCE = "https://github.com/jxbi1010/HTT/blob/2e5b0b10d9e527c188bd78c66db3c500d1bc8877/data/gsmini_force_4probe_50each_dataloader.py#L320-L375"


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    os.replace(tmp, path)


def source_npz(entry: dict[str, Any]) -> Path:
    paths = [Path(path) for path in entry["source_files"] if path.endswith(".npz") and ".labeled." not in path]
    if len(paths) != 1:
        raise ValueError(f"{entry['episode_id']}: expected one force source NPZ, got {paths}")
    return paths[0]


def empty_stats() -> dict[str, Any]:
    return {"frames": 0, "values": 0, "clipped_values": 0,
            "ref_abs_sum_axis_N": [0.0] * 3,
            "unclipped_min_N": [float("inf")] * 3, "unclipped_max_N": [float("-inf")] * 3}


def update_stats(stats: dict[str, Any], values: np.ndarray, reference: np.ndarray) -> None:
    clipped = (values < CLIP_N[0]) | (values > CLIP_N[1])
    stats["frames"] += len(values)
    stats["values"] += values.size
    stats["clipped_values"] += int(clipped.sum())
    stats["ref_abs_sum_axis_N"] = (np.asarray(stats["ref_abs_sum_axis_N"]) + np.abs(reference) * len(values)).tolist()
    stats["unclipped_min_N"] = np.minimum(stats["unclipped_min_N"], values.min(axis=0)).tolist()
    stats["unclipped_max_N"] = np.maximum(stats["unclipped_max_N"], values.max(axis=0)).tolist()


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_manifest = json.loads(args.force_cache_manifest.read_text())
    if source_manifest.get("status") != "complete" or len(source_manifest.get("entries", [])) != 101:
        raise ValueError("requires the complete preserved 101-episode raw force cache")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    stats = {fold: {role: empty_stats() for role in sorted(DEVELOPMENT_ROLES)} for fold in FOLDS}
    transformed_entries = []
    transform_sha = sha256_file(Path(__file__))
    for index, old in enumerate(source_manifest["entries"], 1):
        raw_target_path = Path(old["force_native_n_path"])
        raw_target = np.load(raw_target_path, mmap_mode="r", allow_pickle=False)
        path = source_npz(old)
        with np.load(path, allow_pickle=False) as archive:
            original = np.asarray(archive["6d_force"][:, :3], dtype=np.float32)
            reference = np.asarray(archive["ref_force"][:3], dtype=np.float32)
        if raw_target.shape != original.shape or not np.array_equal(raw_target, original):
            raise ValueError(f"{old['episode_id']}: preserved raw target does not match source")
        unclipped = original - reference[None, :]
        contact = np.clip(unclipped, *CLIP_N).astype(np.float32)
        key = raw_target_path.name.removesuffix(".force_native_n.npy")
        contact_path = output / "episodes" / f"{key}.force_contact_n.npy"
        meta_path = output / "episodes" / f"{key}.target.json"
        atomic_npy(contact_path, contact)
        target_meta = {
            "episode_id": old["episode_id"], "source_force_path": str(path),
            "source_raw_target_path": str(raw_target_path), "source_raw_target_sha256": old["force_native_n_sha256"],
            "target_path": str(contact_path), "target_sha256": sha256_file(contact_path),
            "definition": "clip((6d_force - ref_force)[:, :3], -20 N, 20 N)",
            "axes": list(FORCE_AXES), "unit": "N", "official_source": OFFICIAL_SOURCE,
            "transform_source_sha256": transform_sha,
        }
        atomic_json(meta_path, target_meta)
        entry = dict(old)
        entry.update({
            "force_raw_n_path": str(raw_target_path), "force_raw_n_sha256": old["force_native_n_sha256"],
            "force_native_n_path": str(contact_path), "force_native_n_sha256": target_meta["target_sha256"],
            "force_target_definition": target_meta["definition"], "subtract_ref_force": True,
            "clip_N": list(CLIP_N), "target_transform_source_sha256": transform_sha,
            "target_status": "derived_from_preserved_raw_target",
        })
        transformed_entries.append(entry)
        for fold, role in old["roles_by_fold"].items():
            if role in DEVELOPMENT_ROLES:
                update_stats(stats[fold][role], unclipped, reference)
        print(f"[{index}/101] {old['episode_id']}: target built", flush=True)

    for fold_roles in stats.values():
        for row in fold_roles.values():
            row["clipped_value_fraction"] = row["clipped_values"] / row["values"] if row["values"] else None
            row["mean_abs_ref_correction_N"] = (np.asarray(row.pop("ref_abs_sum_axis_N")) / row["frames"]).tolist()
    result = {
        "format": FORMAT, "status": "complete", "generated_utc": datetime.now(timezone.utc).isoformat(),
        "entries": transformed_entries,
        "source_force_cache_manifest": str(args.force_cache_manifest),
        "source_force_cache_manifest_sha256": sha256_file(args.force_cache_manifest),
        "token_provenance_preserved": {
            key: source_manifest.get(key) for key in
            ("checkpoint", "checkpoint_sha256", "encoder_state_sha256", "preprocess")
        },
        "target_transform": {"definition": "clip((6d_force - ref_force)[:, :3], -20 N, 20 N)",
                             "axes": list(FORCE_AXES), "unit": "N", "official_source": OFFICIAL_SOURCE,
                             "source_sha256": transform_sha},
        "contact_stats_by_fold_development_role": stats,
        "test_role_summary_computed": False,
    }
    atomic_json(output / "cache_manifest.json", result)
    print(json.dumps({"status": "complete", "entries": len(transformed_entries),
                      "stats": stats}, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-cache-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
