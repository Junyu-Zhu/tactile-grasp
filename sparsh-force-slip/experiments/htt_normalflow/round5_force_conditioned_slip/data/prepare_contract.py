#!/usr/bin/env python3
"""Audit Round-5 supervision and assemble the authoritative cache contract."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from contract import (
    DEVELOPMENT_ROLES, FORCE_AXES, FORCE_CONDITIONED_MIN_T, FORCE_INPUT_MIN_T,
    FORCE_UNIT, FORMAT, FOLDS, FUTURE_MIN_T, atomic_json, roles_by_fold,
    sha256_file, validate_contract,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def summarize_counts(split: dict[str, Any]) -> dict[str, Any]:
    rows = {row["id"]: row for row in split["episodes"]}
    out: dict[str, Any] = {}
    for fold in FOLDS:
        out[fold] = {}
        for role in sorted(DEVELOPMENT_ROLES):
            task_rows = [rows[x] for x in split["splits"][fold][role]]
            force_rows = [row for row in task_rows if row.get("task") == "force"]
            slip_rows = [row for row in task_rows if row.get("task") == "slip"]
            out[fold][role] = {
                "force_episodes": len(force_rows),
                "force_frames_t_ge_5": sum(max(0, row["frames"] - FORCE_INPUT_MIN_T) for row in force_rows),
                "force_frames_common_t_ge_10": sum(max(0, row["frames"] - FORCE_CONDITIONED_MIN_T) for row in force_rows),
                "slip_episodes": len(slip_rows),
                "slip_frames_common_t_ge_10": sum(max(0, row["frames"] - FORCE_CONDITIONED_MIN_T) for row in slip_rows),
                "future_frames_t_ge_13_before_target_tail_mask": sum(max(0, row["frames"] - FUTURE_MIN_T) for row in slip_rows),
            }
    return out


def inspect_force(paths: list[Path]) -> list[dict[str, Any]]:
    result = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            force = np.asarray(archive["6d_force"])
            raw = force[:, :3]
            reference = np.asarray(archive["ref_force"][:3])
            unclipped = raw - reference[None, :]
            native = np.clip(unclipped, -20.0, 20.0)
            result.append({
                "path": str(path), "keys": sorted(archive.files),
                "force_shape": list(force.shape), "native_shape": list(native.shape),
                "native_dtype": str(native.dtype), "native_finite": bool(np.isfinite(native).all()),
                "native_min_N": native.min(axis=0).tolist(),
                "native_max_N": native.max(axis=0).tolist(),
                "native_median_abs_N": np.median(np.abs(native), axis=0).tolist(),
                "ref_relative_unclipped_min_N": unclipped.min(axis=0).tolist(),
                "ref_relative_unclipped_max_N": unclipped.max(axis=0).tolist(),
                "clipped_value_count": int(((unclipped < -20.0) | (unclipped > 20.0)).sum()),
                "clipped_value_fraction": float(((unclipped < -20.0) | (unclipped > 20.0)).mean()),
                "ref_force_shape": list(np.asarray(archive["ref_force"]).shape),
                "label_definition": "clip((6d_force - ref_force)[:, :3], -20 N, 20 N)",
            })
    return result


def load_cache_entries(cache_manifest: Path, task: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = read_json(cache_manifest)
    entries = []
    for old in payload["entries"]:
        entry = {
            "episode_id": old["episode_id"], "task": task,
            "roles_by_fold": old["roles_by_fold"], "source_files": old["source_files"],
            "token_path": old["token_path"], "token_sha256": old["token_sha256"],
            "frames": old["frames"], "token_shape": old["token_shape"],
        }
        if task == "slip":
            entry["label_path"] = old["label_path"]
            entry["label_sha256"] = old["label_sha256"]
        else:
            entry["force_native_n_path"] = old["force_native_n_path"]
            entry["force_native_n_sha256"] = old["force_native_n_sha256"]
            for key in ("force_raw_n_path", "force_raw_n_sha256", "force_target_definition",
                        "subtract_ref_force", "clip_N", "target_transform_source_sha256", "target_status"):
                if key in old:
                    entry[key] = old[key]
        entries.append(entry)
    provenance = {key: payload.get(key) for key in ("format", "status", "manifest_sha256", "checkpoint_sha256",
                                                     "encoder_state_sha256", "preprocess")}
    provenance["entries"] = len(entries)
    return provenance, entries


def run(args: argparse.Namespace) -> dict[str, Any]:
    split = read_json(args.split_manifest)
    if split.get("schema_version") != 2:
        raise ValueError("schema-2 split required")
    rows = {row["id"]: row for row in split["episodes"]}
    force_ids = [x for x in split["splits"][args.inspect_fold][args.inspect_role] if rows[x].get("task") == "force"]
    inspected = inspect_force([Path(rows[x]["path"]) for x in force_ids[:args.inspect_first]])
    slip_prov, slip_entries = load_cache_entries(args.slip_cache_manifest, "slip")
    if len(slip_entries) != 101:
        raise ValueError("expected all 101 HTT slip episodes in Round-3 cache")
    entries = list(slip_entries)
    force_prov = None
    if args.force_cache_manifest:
        force_prov, force_entries = load_cache_entries(args.force_cache_manifest, "force")
        entries.extend(force_entries)
    force_complete = bool(force_prov and force_prov["status"] == "complete" and force_prov["entries"] == 101)
    contract = {
        "format": FORMAT,
        "status": "complete" if force_complete else ("partial_force_token_cache" if force_prov else "pending_force_token_cache"),
        "split_manifest": str(args.split_manifest),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "encoder": "sparsh_mae_frozen",
        "slip_cache_provenance": slip_prov,
        "force_cache_provenance": force_prov,
        "force_target_semantics": "clip((6d_force-ref_force)[:3],-20,20) N; shear_x,shear_y,normal",
        "force_target": {"source": "clip((6d_force - ref_force)[:, :3], -20 N, 20 N)",
                         "axes": list(FORCE_AXES), "unit": FORCE_UNIT, "subtract_ref_force": True,
                         "clip_N": [-20.0, 20.0], "alignment": "same episode and same frame only"},
        "history_contract": {"encoder_frames": ["t", "t-5"], "force_training_min_t": FORCE_INPUT_MIN_T,
                             "force_conditioned_slip_min_t": FORCE_CONDITIONED_MIN_T, "future_min_t": FUTURE_MIN_T,
                             "reason": "force(t-5) itself consumes images at t-5 and t-10; future 4-step history adds t-3"},
        "pairing_policy": "independent_episodes_no_force_slip_basename_join",
        "entries": sorted(entries, key=lambda row: (row["task"], row["episode_id"])),
    }
    validate_contract(contract)
    precheck = {
        "status": "pass", "contract_status": contract["status"],
        "split_schema": 2, "split_sha256": contract["split_manifest_sha256"],
        "test_role_arrays_read": 0,
        "inspection_selection": {"fold": args.inspect_fold, "role": args.inspect_role, "first_n": args.inspect_first},
        "inspected_force_archives": inspected,
        "force_semantics": contract["force_target"], "history_contract": contract["history_contract"],
        "old_model_scale_reference": {"force_scale_N": [1.5, 1.5, 2.0],
            "use_for_htt_targets": False,
            "reason": "legacy fixed normalization is far below observed HTT normal-force range; fit normalization on each fold train role only"},
        "counts": summarize_counts(split),
        "cache_reuse": {"slip": "reuse Round-3 complete full-token cache byte-for-byte",
                        "force": "build one new full-token cache with identical frozen MAE/checkpoint/preprocess; do not duplicate slip tensors"},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "contract.json", contract)
    atomic_json(args.output_dir / "PRECHECK.json", precheck)
    print(json.dumps({"status": "pass", "contract_status": contract["status"], "entries": len(entries),
                      "counts": precheck["counts"]}, indent=2))
    return precheck


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--slip-cache-manifest", type=Path, required=True)
    parser.add_argument("--force-cache-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inspect-fold", choices=FOLDS, default="htt_leave_p1")
    parser.add_argument("--inspect-role", choices=sorted(DEVELOPMENT_ROLES), default="train")
    parser.add_argument("--inspect-first", type=int, default=5)
    args = parser.parse_args()
    if args.inspect_first < 1:
        parser.error("--inspect-first must be positive")
    run(args)


if __name__ == "__main__":
    main()
