#!/usr/bin/env python3
"""Independent P0 rebind after the bounded Round-8 CSV export repair."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def restored_pre_repair_source(current: str) -> str:
    start = current.index("def atomic_predictions(")
    end = current.index("\ndef _group_enabled", start)
    prefix, current, suffix = current[:start], current[start:end], current[end:]
    replacements = (
        ("if not np.allclose(probabilities, expected, rtol=0, atol=1e-6):",
         "if not np.allclose(probabilities, expected, rtol=0, atol=1e-7):"),
        ("    base = _row_base(row)\n", ""),
        ("float(base[index, -1, 768])", "float(_row_base(row)[index, -1, 768])"),
        ("force = base[index, -1, 769:772].float().tolist()",
         "force = _row_base(row)[index, -1, 769:772].float().tolist()"),
    )
    for before, after in replacements:
        if current.count(before) != 1:
            raise ValueError(f"repair source pattern count drift: {before}")
        current = current.replace(before, after, 1)
    return prefix + current + suffix


def export_equivalence(old, new, payload: dict) -> dict:
    old.PROTOCOL = new.PROTOCOL
    source = payload["timelines"]["selection"]
    def first(value, count=7):
        if torch.is_tensor(value) or isinstance(value, np.ndarray) or isinstance(value, list):
            return value[:count]
        return value
    rows = {key: first(value) for key, value in source.items()}
    count = len(rows["t"])
    results = {}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for group in ("B_xyz", "C_hazard"):
            if group == "C_hazard":
                hazards = np.linspace(0.01, 0.45, count * 5, dtype=np.float32).reshape(count, 5)
                probabilities = (1.0 - np.cumprod(1.0 - hazards, axis=1))[:, [0, 2, 4]]
            else:
                hazards = None
                probabilities = np.linspace(0.01, 0.99, count * 3, dtype=np.float32).reshape(count, 3)
            prediction = {"probabilities": probabilities, "hazards": hazards, "states": None}
            old_path, new_path = root / f"{group}_old.csv", root / f"{group}_new.csv"
            old.atomic_predictions(old_path, payload, rows, prediction, group, timeline=True)
            new.atomic_predictions(new_path, payload, rows, prediction, group, timeline=True)
            results[group] = {"rows": count, "byte_equal": old_path.read_bytes() == new_path.read_bytes(),
                              "old_sha256": sha256(old_path), "new_sha256": sha256(new_path)}
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prepare-audit", type=Path, required=True)
    parser.add_argument("--p4-decision", type=Path, required=True)
    parser.add_argument("--smoke-audit", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--pre-repair-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    review_v1 = args.round_root / "reviews/review_p0.py"
    with tempfile.TemporaryDirectory() as directory:
        underlying_path = Path(directory) / "underlying.json"
        command = [sys.executable, str(review_v1),
                   "--round-root", str(args.round_root), "--output-root", str(args.output_root),
                   "--prepared", str(args.prepared), "--prepare-audit", str(args.prepare_audit),
                   "--p4-decision", str(args.p4_decision), "--smoke-audit", str(args.smoke_audit),
                   "--output", str(underlying_path)]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if not underlying_path.is_file():
            raise RuntimeError(f"underlying P0 did not produce audit: {completed.stderr}")
        underlying = json.loads(underlying_path.read_text())

    train_path = args.round_root / "training/train.py"
    train = load_module("r8_repaired_train_review", train_path)
    old = load_module("r8_pre_repair_train_review", args.pre_repair_source)
    payload = torch.load(args.prepared, map_location="cpu", weights_only=False)
    smoke = json.loads(args.smoke_audit.read_text())
    amendment = json.loads(args.amendment.read_text())
    allowed_underlying_failure = {"no_formal_training_started_before_review"}
    checks = {
        "all_original_p0_checks_still_pass_except_expected_post_start_marker":
            set(underlying.get("failures", [])) <= allowed_underlying_failure,
        "repair_scope_exactly_matches_disclosed_source_change":
            restored_pre_repair_source(train_path.read_text()) == args.pre_repair_source.read_text(),
        "repair_amendment_records_zero_accepted_runs":
            amendment.get("status") == "implementation_repair"
            and amendment.get("completed_accepted_runs_before_repair") == 0,
        "new_smoke_passes_all_groups_and_binds_repaired_trainer":
            smoke.get("status") == "pass" and len(smoke.get("runs", [])) == 8
            and all(all(run["checks"].values()) for run in smoke["runs"])
            and smoke.get("trainer_sha256") == sha256(train_path),
        "new_smoke_resume_exact":
            set(smoke.get("actual_interruption_comparisons", {})) == {"C_xyz_delta", "C_hazard", "P4_state"}
            and all(all(values.values()) for values in smoke["actual_interruption_comparisons"].values()),
    }
    equivalence = export_equivalence(old, train, payload)
    checks["pre_post_export_byte_equivalence"] = all(value["byte_equal"] for value in equivalence.values())
    checks["prepared_and_protocol_identity_unchanged"] = (
        train.sha256(args.prepared) == json.loads(args.prepare_audit.read_text())["prepared"]["sha256"]
        and sha256(args.round_root / "training/protocol.json") == smoke.get("training_protocol_sha256")
    )
    result = {
        "schema": "round8_p0_post_start_repair_rebind_v2",
        "status": "pass" if all(checks.values()) else "fail",
        "scope": "independent rebind after bounded CSV export performance/tolerance repair; original P0 v1 preserved",
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
        "disclosure": {
            "formal_training_had_started_before_repair": True,
            "accepted_formal_runs_before_repair": amendment.get("completed_accepted_runs_before_repair"),
            "repair": "cache immutable base tensor once per export; hazard cumulative audit tolerance 1e-7 to 1e-6",
            "scientific_configuration_changed": False,
        },
        "export_equivalence": equivalence,
        "source_hashes": {str(path.resolve()): sha256(path) for path in (
            train_path, args.pre_repair_source, args.round_root / "training/protocol.json", review_v1,
            Path(__file__), args.prepared, args.prepare_audit, args.p4_decision, args.smoke_audit, args.amendment
        )},
        "underlying_p0_failures": underlying.get("failures", []),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    train.atomic_json(args.output, result)
    print(json.dumps({"status": result["status"], "checks": len(checks), "failures": result["failures"]}))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
