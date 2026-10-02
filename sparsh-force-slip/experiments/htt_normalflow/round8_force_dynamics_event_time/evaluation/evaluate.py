#!/usr/bin/env python3
"""Validate Round-8 manifests and produce a bounded pre-formal evaluation audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

try:
    from .schema import load_timeline_csv, validate_cross_run_identity
except ImportError:
    from schema import load_timeline_csv, validate_cross_run_identity

ROLES = ("selection", "calibration", "outer")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "round8_prediction_manifest_v1":
        raise ValueError("manifest schema must be round8_prediction_manifest_v1")
    if manifest.get("status") != "complete":
        raise ValueError("prediction manifest status must be complete")
    training_spec = manifest.get("training_audit")
    if not isinstance(training_spec, dict) or not {"path", "sha256"}.issubset(training_spec):
        raise ValueError("prediction manifest must bind training_audit path and SHA")
    training_path = Path(training_spec["path"]).resolve()
    if sha256(training_path) != training_spec["sha256"]:
        raise ValueError("training audit SHA mismatch")
    training_audit = json.loads(training_path.read_text())
    if training_audit.get("schema") != "round8_training_audit_v1" or training_audit.get("status") != "pass":
        raise ValueError("training audit is not pass")
    artifacts = manifest.get("artifacts", [])
    if not artifacts:
        raise ValueError("manifest has no artifacts")
    declared_groups = manifest.get("groups")
    declared_seeds = manifest.get("seeds")
    if not isinstance(declared_groups, list) or not declared_groups or len(set(declared_groups)) != len(declared_groups):
        raise ValueError("manifest must declare unique non-empty groups")
    if not isinstance(declared_seeds, list) or not declared_seeds or len(set(declared_seeds)) != len(declared_seeds):
        raise ValueError("manifest must declare unique non-empty seeds")
    expected_runs = {(group, int(seed)) for group in declared_groups for seed in declared_seeds}
    audit_runs = training_audit.get("runs", [])
    if training_audit.get("run_count") != len(expected_runs) or len(audit_runs) != len(expected_runs):
        raise ValueError("training audit run count drift")
    if {(row.get("id"),) for row in audit_runs} != {(f"future_{group}_{seed}",) for group, seed in expected_runs}:
        raise ValueError("training audit run identity drift")
    if any(not isinstance(row.get("checks"), dict) or not row["checks"] or not all(row["checks"].values())
           for row in audit_runs):
        raise ValueError("training audit contains failed or empty run checks")
    if not training_audit.get("frozen_inputs") or not all(training_audit["frozen_inputs"].values()):
        raise ValueError("training audit frozen input checks failed")
    if not training_audit.get("initialization") or not all(training_audit["initialization"].values()):
        raise ValueError("training audit initialization checks failed")
    for field in ("source_sha256", "inventory_sha256"):
        value = training_audit.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"training audit missing valid {field}")
    checkpoint_spec = manifest.get("checkpoint_index")
    if not isinstance(checkpoint_spec, dict) or not {"path", "sha256"}.issubset(checkpoint_spec):
        raise ValueError("prediction manifest must bind checkpoint_index path and SHA")
    checkpoint_path = Path(checkpoint_spec["path"]).resolve()
    if sha256(checkpoint_path) != checkpoint_spec["sha256"]:
        raise ValueError("checkpoint index SHA mismatch")
    with checkpoint_path.open(newline="") as handle:
        checkpoint_rows = list(csv.DictReader(handle))
    expected_checkpoint_grid = {(f"future_{group}_{seed}", group, seed, kind)
                                for group, seed in expected_runs for kind in ("best", "latest", "config")}
    actual_checkpoint_grid = {(row.get("id"), row.get("group"), int(row.get("seed", -1)), row.get("kind"))
                              for row in checkpoint_rows}
    if len(checkpoint_rows) != len(expected_checkpoint_grid) or actual_checkpoint_grid != expected_checkpoint_grid:
        raise ValueError("checkpoint index grid drift")
    for row in checkpoint_rows:
        artifact_path = Path(row["path"]).resolve()
        if sha256(artifact_path) != row["sha256"]:
            raise ValueError(f"checkpoint index artifact SHA mismatch {artifact_path}")
    seen = set()
    role_reference = {}
    sources = {str(path.resolve()): sha256(path), str(training_path): training_spec["sha256"],
               str(checkpoint_path): checkpoint_spec["sha256"]}
    run_counts = {}
    run_paths: dict[tuple[str, int], set[Path]] = {}
    for artifact in artifacts:
        key = (artifact["group"], int(artifact["seed"]), artifact["role"])
        if key in seen or artifact["role"] not in ROLES:
            raise ValueError(f"duplicate or invalid artifact {key}")
        seen.add(key)
        timeline_path = Path(artifact["path"]).resolve()
        paths = run_paths.setdefault((key[0], key[1]), set())
        if timeline_path in paths:
            raise ValueError(f"run reuses one timeline across roles {key[:2]}: {timeline_path}")
        paths.add(timeline_path)
        actual_sha = sha256(timeline_path)
        if actual_sha != artifact["sha256"]:
            raise ValueError(f"prediction SHA mismatch {timeline_path}")
        rows = load_timeline_csv(timeline_path, manifest.get("horizons", [1, 3, 5]), head_type=artifact["head_type"])
        upstream_required = {"force_x", "force_y", "force_z", "delta_x", "delta_y", "delta_z", "delta_valid"}
        if any(not upstream_required.issubset(row) for row in rows):
            raise ValueError(f"formal timeline missing frozen-upstream fields {timeline_path}")
        if artifact["role"] in role_reference:
            validate_cross_run_identity(role_reference[artifact["role"]], rows, manifest.get("horizons", [1, 3, 5]))
        else:
            role_reference[artifact["role"]] = rows
        sources[str(timeline_path.resolve())] = actual_sha
        run_counts[f"{artifact['group']}:{artifact['seed']}:{artifact['role']}"] = len(rows)
    actual_runs = {(artifact["group"], int(artifact["seed"])) for artifact in artifacts}
    if actual_runs != expected_runs:
        raise ValueError(f"formal run grid drift missing={sorted(expected_runs-actual_runs)} extra={sorted(actual_runs-expected_runs)}")
    for run in expected_runs:
        if {role for group, seed, role in seen if (group, seed) == run} != set(ROLES):
            raise ValueError(f"run lacks complete role set {run}")
    role_groups = {role: {row["leakage_group"] for row in role_reference[role]} for role in ROLES}
    for index, left in enumerate(ROLES):
        for right in ROLES[index + 1:]:
            overlap = role_groups[left] & role_groups[right]
            if overlap:
                raise ValueError(f"role leakage-group overlap {left}/{right}: {sorted(overlap)[:5]}")
    return {"status": "pass", "schema": manifest["schema"], "test_role_consumed": False,
            "runs": len(expected_runs), "artifacts": len(artifacts), "rows": run_counts, "sources": sources}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = validate_manifest(args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
