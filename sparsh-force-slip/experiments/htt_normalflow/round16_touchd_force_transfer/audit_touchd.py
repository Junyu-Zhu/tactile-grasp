#!/usr/bin/env python3
"""Audit the released ToucHD-Force GelSight Mini supervision without extraction."""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import io
import json
import math
import re
import statistics
import zipfile
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(archive: zipfile.ZipFile, suffix: str) -> list[list[str]]:
    names = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(names) != 1:
        raise RuntimeError(f"expected one {suffix}, found {names}")
    text = archive.read(names[0]).decode("utf-8")
    return list(csv.reader(io.StringIO(text)))


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    index = (len(ordered) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - index) + ordered[hi] * (index - lo)


def object_roles() -> dict[str, list[int]]:
    """Pre-fixed all-object 80/20 split, independent of data values and results."""
    ordered = sorted(range(71), key=lambda obj: hashlib.sha256(
        f"round16-touchd-object-role-v1|obj{obj:03d}".encode()).hexdigest())
    selection = set(ordered[:14])
    return {
        "fit": [obj for obj in range(71) if obj not in selection],
        "selection": sorted(selection),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-sha-proof", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    compact_path = root / "all_data_direction.json"
    readme_path = root / "README.md"
    compact = json.loads(compact_path.read_text())
    expected_keys = [f"obj{obj:03d}_speed{speed}" for obj in range(71) for speed in (1, 2)]
    if sorted(compact) != sorted(expected_keys):
        raise RuntimeError("compact JSON does not contain exactly 71 objects x 2 speeds")

    roles = object_roles()
    role_by_object = {obj: role for role, objects in roles.items() for obj in objects}
    stage_counts: Counter[str] = Counter()
    role_samples: Counter[str] = Counter()
    force_min = [math.inf] * 3
    force_max = [-math.inf] * 3
    offsets_ms: list[float] = []
    raw_fixed_changed = 0
    raw_fixed_total = 0
    raw_fixed_max_abs = 0.0
    compact_all_exact = 0
    compact_nearest_raw_exact = 0
    compact_nearest_fixed_exact = 0
    compact_nearest_both_exact = 0
    compact_nearest_neither: list[dict] = []
    compact_axis_source: Counter[str] = Counter()
    missing_images: list[dict] = []
    duplicate_ids: list[dict] = []
    missing_csv: list[dict] = []
    archive_summaries = []

    for key in expected_keys:
        obj = int(key[3:6])
        archive_path = root / f"{key}.zip"
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
            base = f"{key}/gelsight/"
            json_rows = compact[key].get("gelsight")
            if not isinstance(json_rows, list) or not json_rows:
                raise RuntimeError(f"missing gelsight compact rows: {key}")
            ids = [int(row[0]) for row in json_rows]
            if len(ids) != len(set(ids)):
                duplicate_ids.append({"archive": key, "rows": len(ids), "unique": len(set(ids))})
            absent = [image_id for image_id in ids if f"{base}image_{image_id}.png" not in names]
            if f"{base}image_1.png" not in names:
                absent.append(1)
            if absent:
                missing_images.append({"archive": key, "count": len(absent), "examples": absent[:10]})
            try:
                tactile = rows(archive, "gelsighttactile.csv")
                fixed = rows(archive, "gelsightdata_fixed.csv")
                raw = rows(archive, "gelsightdata.csv")
                all_rows = rows(archive, "all_data.csv")
            except RuntimeError as error:
                missing_csv.append({"archive": key, "error": str(error)})
                continue

            all_gelsight = [[int(row[1]), *(float(v) for v in row[2:5])] for row in all_rows if row[0] == "gelsight"]
            compact_numeric = [[int(row[0]), *(float(v) for v in row[1:4])] for row in json_rows]
            if all_gelsight != compact_numeric:
                raise RuntimeError(f"compact/all_data mismatch: {key}")
            compact_all_exact += len(json_rows)

            raw_values = [[float(v) for v in row] for row in raw]
            fixed_values = [[float(v) for v in row] for row in fixed]
            if len(raw_values) != len(fixed_values):
                raise RuntimeError(f"raw/fixed length mismatch: {key}")
            for before, after in zip(raw_values, fixed_values):
                if before[3] != after[3]:
                    raise RuntimeError(f"raw/fixed timestamp mismatch: {key}")
                delta = max(abs(a - b) for a, b in zip(before[:3], after[:3]))
                raw_fixed_total += 1
                raw_fixed_changed += delta > 0
                raw_fixed_max_abs = max(raw_fixed_max_abs, delta)

            fixed_times = [row[3] for row in fixed_values]
            tactile_by_id = {int(row[0]): float(row[1]) for row in tactile}
            for row in json_rows:
                image_id = int(row[0])
                target = [float(v) for v in row[1:4]]
                if image_id not in tactile_by_id:
                    raise RuntimeError(f"compact image id absent from tactile CSV: {key}/{image_id}")
                timestamp = tactile_by_id[image_id]
                insertion = bisect.bisect_left(fixed_times, timestamp)
                candidates = [i for i in (insertion - 1, insertion) if 0 <= i < len(fixed_times)]
                nearest = min(candidates, key=lambda i: abs(fixed_times[i] - timestamp))
                paired_fixed = fixed_values[nearest][:3]
                paired_raw = raw_values[nearest][:3]
                raw_match = paired_raw == target
                fixed_match = paired_fixed == target
                for raw_value, fixed_value, target_value in zip(paired_raw, paired_fixed, target):
                    if target_value == raw_value == fixed_value:
                        compact_axis_source["both"] += 1
                    elif target_value == raw_value:
                        compact_axis_source["raw_only"] += 1
                    elif target_value == fixed_value:
                        compact_axis_source["fixed_only"] += 1
                    else:
                        compact_axis_source["neither"] += 1
                compact_nearest_raw_exact += raw_match
                compact_nearest_fixed_exact += fixed_match
                compact_nearest_both_exact += raw_match and fixed_match
                if not raw_match and not fixed_match:
                    compact_nearest_neither.append({
                        "archive": key, "image_id": image_id, "target": target,
                        "nearest_raw": paired_raw, "nearest_fixed": paired_fixed,
                        "offset_ms": (timestamp - fixed_times[nearest]) * 1000.0,
                    })
                offsets_ms.append((timestamp - fixed_times[nearest]) * 1000.0)
                stage_counts[str(row[4])] += 1
                role_samples[role_by_object[obj]] += 1
                # Published probe convention is [Fx, Fy, -Fz].
                published = [target[0], target[1], -target[2]]
                for axis in range(3):
                    force_min[axis] = min(force_min[axis], published[axis])
                    force_max[axis] = max(force_max[axis], published[axis])
            archive_summaries.append({
                "key": key, "object": obj, "speed": int(key[-1]), "role": role_by_object[obj],
                "eligible_rows": len(json_rows), "background_present": f"{base}image_1.png" in names,
            })

    official = None
    if args.official_repo:
        repo = args.official_repo.resolve()
        dataset_py = repo / "data/downstream_dataset.py"
        engine_py = repo / "train/probe_touchd_engine.py"
        official = {
            "repo": "https://github.com/GeWu-Lab/AnyTouch2",
            "commit": __import__("subprocess").check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
            "files": {str(dataset_py): sha256(dataset_py), str(engine_py): sha256(engine_py)},
            "observed_contract": {
                "supported_probe_sensor_names": ["digit", "gelsight"],
                "gelsight_is_mini": "project paper/README sensor list and released compact key; archive directory is named gelsight",
                "target": "[Fx,Fy,-Fz] divided by fixed gelsight maxima [6.84,9.87,8.52]; normalized z clipped to [0,1]",
                "reference": "image_1.png",
                "preprocess": "image-reference+130/255, clamp [0,1], resize 224x224, fixed RGB normalize",
                "two_frame_gelsight_stride": 3,
                "published_train_objects": [6, 41, 52, 53, 59, 69, 70],
                "published_eval_objects": [18, 22, 61],
                "reported_unit_use": "de-normalized RMSE multiplied by 1000 and labeled mN, hence released force values are used as N",
            },
        }

    audit = {
        "schema": "round16_touchd_gelsight_release_audit_v1",
        "status": "pass" if not (missing_images or duplicate_ids or missing_csv or compact_axis_source["neither"]) else "fail",
        "scope": "released ToucHD-Force gelsight/GelSight Mini stream only; no cross-domain coordinate equivalence claimed",
        "identity": {
            "root": str(root), "readme_sha256": sha256(readme_path),
            "all_data_direction_sha256": sha256(compact_path),
            "parent_full_collection_sha_proof": str(args.parent_sha_proof.resolve()),
            "parent_full_collection_sha_proof_sha256": sha256(args.parent_sha_proof.resolve()),
            "archives_reused": 142, "full_rehash_repeated": False,
        },
        "split": {
            "algorithm": "SHA256(round16-touchd-object-role-v1|objNNN), first 14 of 71 to selection, remaining 57 fit",
            "roles": roles, "all_objects_assigned_once": len(role_by_object) == 71,
            "both_speeds_in_same_role": True, "performance_independent": True,
            "official_10_object_lists_used_for_training": False,
            "reason": "official probe lists cover only 10/71 objects; R16 fixes roles for all released objects",
        },
        "alignment": {
            "eligible_gelsight_rows": sum(role_samples.values()), "role_samples": dict(role_samples),
            "compact_equals_archive_all_data_rows_exact": compact_all_exact,
            "compact_target_equals_nearest_timestamped_raw_row_exact": compact_nearest_raw_exact,
            "compact_target_equals_nearest_timestamped_fixed_row_exact": compact_nearest_fixed_exact,
            "compact_target_equals_both_raw_and_fixed_exact": compact_nearest_both_exact,
            "compact_target_hybrid_vector_rows": len(compact_nearest_neither),
            "compact_axis_source_counts": dict(compact_axis_source),
            "compact_target_matches_neither_nearest_raw_nor_fixed": compact_nearest_neither[:20],
            "offset_ms": {"min": min(offsets_ms), "max": max(offsets_ms),
                          "mean": statistics.fmean(offsets_ms), "p50_abs": percentile([abs(v) for v in offsets_ms], .5),
                          "p95_abs": percentile([abs(v) for v in offsets_ms], .95),
                          "p99_abs": percentile([abs(v) for v in offsets_ms], .99)},
            "missing_images": missing_images, "duplicate_image_ids": duplicate_ids, "missing_csv": missing_csv,
        },
        "released_target": {
            "domain_local_vector": "[Fx,Fy,-Fz] from compact all_data_direction.json, in released force-value units used by official code as N",
            "stage_counts": dict(stage_counts), "published_axis_min": force_min, "published_axis_max": force_max,
            "raw_fixed": {"rows": raw_fixed_total, "changed_rows": raw_fixed_changed,
                          "timestamps_exact": True, "maximum_absolute_value_change": raw_fixed_max_abs},
            "data_fixed_policy": "use the released compact target; audit tests whether it follows nearest raw or fixed rows and does not infer undocumented fixing/tare physics",
        },
        "official": official,
        "must_fix_gates": {
            "released_target_reproducible": compact_all_exact == sum(role_samples.values()),
            "raw_fixed_derivation_fully_explains_each_scalar": compact_axis_source["neither"] == 0,
            "image_target_mapping_complete": not missing_images and not duplicate_ids and not missing_csv,
            "all_object_roles_locked": len(role_by_object) == 71,
            "source_encoder_interface_and_reset_still_require_smoke": True,
        },
        "disclosed_nonblockers": [
            "sensor-to-wrist/world rotation is not established; ToucHD and HTT axis errors must remain domain-local",
            "physical zero/tare and the data_fixed transformation are undocumented",
            "hardware synchronization tolerance is not certified beyond the released nearest-row pairing reproduced here",
            "official 7/3 object lists are a probe-code split, not a complete 71-object split",
        ],
        "archives": archive_summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: audit[k] for k in ("status", "split", "alignment", "must_fix_gates")}, indent=2))
    if audit["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
