#!/usr/bin/env python3
"""Complete the Round-8 paired-CI grid without recomputing valid intervals."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "CI_COMPLETION_PROTOCOL.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError("CI completion row schema drift")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def selected_available_seeds(metrics: list[dict[str, str]], group: str, horizon: int,
                             operating_point: str) -> set[int]:
    return {int(row["seed"]) for row in metrics
            if row.get("method_type") == "neural_operational" and row.get("group") == group
            and int(row.get("horizon", -1)) == horizon and row.get("population") == "primary"
            and row.get("operating_point") == operating_point and row.get("rule_selected") == "True"
            and row.get("threshold_status") == "available"}


def complete(manifest_path: Path, output: Path) -> dict[str, Any]:
    protocol = json.loads(PROTOCOL.read_text())
    if protocol.get("schema") != "round8_ci_completion_protocol_v1" or protocol.get("thresholds_or_models_refit") is not False:
        raise ValueError("CI completion protocol drift")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or manifest.get("schema") != "round8_prediction_manifest_v1":
        raise ValueError("invalid prediction manifest")
    seeds = sorted(int(seed) for seed in manifest["seeds"]); expected_seeds = set(seeds)
    horizons = [int(horizon) for horizon in manifest["horizons"]]
    metrics = read_csv(output / "metrics.csv")
    paired_path = output / "paired_bootstrap_ci.csv"
    summary_path = output / "summary.json"
    original_paired = output / "paired_bootstrap_ci_pre_completion.csv"
    original_summary = output / "summary_pre_ci_completion.json"
    if original_paired.exists() or original_summary.exists():
        raise ValueError("CI completion already applied or stale preservation files exist")
    shutil.copy2(paired_path, original_paired); shutil.copy2(summary_path, original_summary)
    existing_rows = read_csv(paired_path)
    existing = {(row["comparison"], int(row["horizon"]), row["operating_point"], row["metric"]): row
                for row in existing_rows}
    if len(existing) != len(existing_rows):
        raise ValueError("duplicate pre-completion CI row")

    output_rows: list[dict[str, Any]] = []
    unavailable = 0
    empty_interval = {key: "" for key in ("estimate", "ci_low", "ci_high", "valid_replicates",
                                           "requested_replicates", "unit", "paired")}
    for comparison in manifest["comparisons"]:
        candidate, comparator = comparison["candidate"], comparison["comparator"]
        comparison_name = f"{candidate}_vs_{comparator}"
        for horizon in horizons:
            specifications = [(operating_point, metric)
                              for operating_point in protocol["event_operating_points"]
                              for metric in protocol["event_metrics"]]
            specifications += [(protocol["raw_operating_point"], metric) for metric in protocol["raw_metrics"]]
            for operating_point, metric in specifications:
                if operating_point == protocol["raw_operating_point"]:
                    candidate_available = comparator_available = expected_seeds
                else:
                    candidate_available = selected_available_seeds(metrics, candidate, horizon, operating_point)
                    comparator_available = selected_available_seeds(metrics, comparator, horizon, operating_point)
                key = (comparison_name, horizon, operating_point, metric)
                row = existing.pop(key, None)
                complete_grid = candidate_available == expected_seeds and comparator_available == expected_seeds
                evidence = {
                    "status": "available" if complete_grid else "unavailable_incomplete_seed_grid",
                    "candidate_available_seeds": ";".join(map(str, sorted(candidate_available))),
                    "candidate_unavailable_seeds": ";".join(map(str, sorted(expected_seeds - candidate_available))),
                    "comparator_available_seeds": ";".join(map(str, sorted(comparator_available))),
                    "comparator_unavailable_seeds": ";".join(map(str, sorted(expected_seeds - comparator_available))),
                    "expected_seed_count": len(seeds),
                }
                if complete_grid:
                    if row is None:
                        raise ValueError(f"available CI missing from original evaluation {key}")
                    base = {name: row.get(name, "") for name in empty_interval}
                else:
                    if row is not None:
                        raise ValueError(f"incomplete seed CI unexpectedly computed {key}")
                    unavailable += 1; base = empty_interval
                output_rows.append({"comparison": comparison_name, "horizon": horizon,
                                    "operating_point": operating_point, "metric": metric, **base, **evidence})
    if existing:
        raise ValueError(f"undeclared original CI rows {sorted(existing)[:3]}")
    atomic_csv(paired_path, output_rows)

    source_hashes = {str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
                     str(PROTOCOL.resolve()): sha256(PROTOCOL),
                     str(manifest_path.resolve()): sha256(manifest_path),
                     str((output / "metrics.csv").resolve()): sha256(output / "metrics.csv")}
    audit = {"schema": "round8_ci_completion_audit_v1", "status": "pass",
             "rows": len(output_rows), "available_rows": len(output_rows) - unavailable,
             "unavailable_rows": unavailable, "expected_seeds": seeds,
             "pre_completion_paired_sha256": sha256(original_paired),
             "completed_paired_sha256": sha256(paired_path),
             "pre_completion_summary_sha256": sha256(original_summary),
             "source_hashes": source_hashes, "thresholds_or_models_refit": False}
    audit_path = output / "CI_COMPLETION_AUDIT.json"; atomic_json(audit_path, audit)
    summary = json.loads(original_summary.read_text())
    summary.setdefault("source_hashes", {}).update(source_hashes)
    summary["ci_completion"] = {"status": "complete", "audit": str(audit_path.resolve()),
                                "audit_sha256": sha256(audit_path), "available_rows": len(output_rows) - unavailable,
                                "unavailable_rows": unavailable,
                                "manifest_sha256": source_hashes[str(manifest_path.resolve())],
                                "metrics_sha256": source_hashes[str((output / "metrics.csv").resolve())]}
    summary["output_hashes"] = {str(path.relative_to(output)): sha256(path)
                                for path in sorted(output.rglob("*"))
                                if path.is_file() and path.name != "summary.json"}
    atomic_json(summary_path, summary)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = complete(args.manifest, args.output)
    print(json.dumps({"status": result["status"], "rows": result["rows"],
                      "unavailable_rows": result["unavailable_rows"]}, sort_keys=True))


if __name__ == "__main__":
    main()
