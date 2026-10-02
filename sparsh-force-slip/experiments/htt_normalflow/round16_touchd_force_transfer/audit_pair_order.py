#!/usr/bin/env python3
"""Verify the actual compact-row ToucHD temporal pairs used by R16."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import zipfile
from pathlib import Path

from touchd_common import atomic_json, sha256


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] if lo == hi else values[lo] * (hi - position) + values[hi] * (position - lo)


def tactile_timestamps(archive: zipfile.ZipFile) -> dict[int, float]:
    names = [name for name in archive.namelist() if name.endswith("gelsighttactile.csv")]
    if len(names) != 1:
        raise RuntimeError(f"expected one gelsighttactile.csv, found {names}")
    rows = csv.reader(io.StringIO(archive.read(names[0]).decode("utf-8")))
    result = {int(row[0]): float(row[1]) for row in rows}
    if not result:
        raise RuntimeError("empty tactile timestamp table")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compact_path = args.root / "all_data_direction.json"
    compact = json.loads(compact_path.read_text())
    id_gaps: list[int] = []
    elapsed_ms: list[float] = []
    archive_rows = []
    failures = []
    pairs = 0
    for key in sorted(compact):
        rows = compact[key]["gelsight"]
        ids = [int(row[0]) for row in rows]
        strict_ids = all(right > left for left, right in zip(ids, ids[1:]))
        unique_ids = len(ids) == len(set(ids))
        with zipfile.ZipFile(args.root / f"{key}.zip") as archive:
            names = set(archive.namelist())
            timestamps = tactile_timestamps(archive)
            referenced = {ids[i] for i in range(5, len(ids))} | {ids[i - 3] for i in range(5, len(ids))}
            missing_images = sorted(image_id for image_id in referenced
                                    if f"{key}/gelsight/image_{image_id}.png" not in names)
            missing_timestamps = sorted(referenced - timestamps.keys())
            nonpositive_time = 0
            for i in range(5, len(ids)):
                current, previous = ids[i], ids[i - 3]
                if current in timestamps and previous in timestamps:
                    dt = (timestamps[current] - timestamps[previous]) * 1000.0
                    elapsed_ms.append(dt)
                    id_gaps.append(current - previous)
                    nonpositive_time += dt <= 0
                    pairs += 1
            row = {
                "key": key,
                "rows": len(rows),
                "pairs": max(0, len(rows) - 5),
                "strictly_increasing_compact_ids": strict_ids,
                "unique_compact_ids": unique_ids,
                "missing_pair_images": len(missing_images),
                "missing_pair_timestamps": len(missing_timestamps),
                "nonpositive_pair_elapsed": nonpositive_time,
            }
            archive_rows.append(row)
            if not (strict_ids and unique_ids) or missing_images or missing_timestamps or nonpositive_time:
                failures.append({**row, "missing_image_examples": missing_images[:10],
                                 "missing_timestamp_examples": missing_timestamps[:10]})
    result = {
        "schema": "round16_touchd_pair_order_audit_v1",
        "status": "pass" if not failures and pairs == 80783 else "fail",
        "compact_sha256": sha256(compact_path),
        "dependency": "compact row i and compact row i-3; actual image IDs come from those rows",
        "fixed_raw_frame_gap_claimed": False,
        "fixed_physical_duration_claimed": False,
        "archives": len(archive_rows),
        "pairs": pairs,
        "id_gap": {"min": min(id_gaps), "max": max(id_gaps), "mean": statistics.fmean(id_gaps),
                   "p50": percentile(id_gaps, .5), "p95": percentile(id_gaps, .95), "p99": percentile(id_gaps, .99)},
        "elapsed_ms": {"min": min(elapsed_ms), "max": max(elapsed_ms), "mean": statistics.fmean(elapsed_ms),
                       "p50": percentile(elapsed_ms, .5), "p95": percentile(elapsed_ms, .95),
                       "p99": percentile(elapsed_ms, .99)},
        "failures": failures,
        "archive_checks": archive_rows,
    }
    atomic_json(args.output, result)
    print(json.dumps({key: result[key] for key in ("status", "archives", "pairs", "id_gap", "elapsed_ms", "failures")}, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
