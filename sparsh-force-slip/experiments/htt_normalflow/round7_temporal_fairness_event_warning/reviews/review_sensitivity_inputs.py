#!/usr/bin/env python3
"""Independent bounded review of Round-7 force sensitivity inputs."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch

GROUPS = ("B_force", "C_force_delta")
SEEDS = (20260914, 20260915, 20260916)
TOLERANCE = 2e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def prediction_map(path: Path, horizons: list[int]) -> dict[tuple[str, int], np.ndarray]:
    result = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["episode_id"], int(row["t"]))
            if key in result:
                raise ValueError(f"duplicate formal prediction identity: {key}")
            result[key] = np.asarray([float(row[f"p_future_H{h}_raw"]) for h in horizons], dtype=np.float64)
    return result


def unchanged_except(before: torch.Tensor, after: torch.Tensor, columns: set[int]) -> bool:
    keep = sorted(set(range(before.shape[2])) - columns)
    return bool(torch.equal(before[:, :, keep], after[:, :, keep]))


def function_ast(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(functions) != 1:
        raise ValueError(f"expected one {name} in {path}")
    return ast.dump(functions[0], include_attributes=False)


def check_interventions(sensitivity, x: torch.Tensor, group: str) -> list[dict]:
    derived = set(range(769, 772))
    if group == "C_force_delta":
        derived.update(range(772, 775))

    original = x.clone()
    zeroed = sensitivity.intervention(x, group, "zero_fit_mean")
    lagged = sensitivity.intervention(x, group, "causal_lag1")
    checks = [
        {
            "name": "intervention_does_not_mutate_input",
            "pass": bool(torch.equal(x, original)),
        },
        {
            "name": "zero_fit_mean_sets_only_force_derived_normalized_columns_to_zero",
            "pass": bool(torch.count_nonzero(zeroed[:, :, sorted(derived)]).item() == 0)
            and unchanged_except(x, zeroed, derived),
        },
        {
            "name": "causal_lag1_first_force_derived_slot_is_fit_mean_zero",
            "pass": bool(torch.count_nonzero(lagged[:, 0, sorted(derived)]).item() == 0),
        },
        {
            "name": "causal_lag1_force_derived_slots_use_only_previous_history_slot",
            "pass": bool(torch.equal(lagged[:, 1:, sorted(derived)], x[:, :-1, sorted(derived)])),
        },
    ]
    allowed = set(derived)
    if group == "C_force_delta":
        allowed.add(775)
        checks.extend(
            [
                {
                    "name": "causal_lag1_C_validity_first_slot_is_invalid",
                    "pass": bool(torch.count_nonzero(lagged[:, 0, 775]).item() == 0),
                },
                {
                    "name": "causal_lag1_C_validity_uses_only_previous_history_slot",
                    "pass": bool(torch.equal(lagged[:, 1:, 775], x[:, :-1, 775])),
                },
                {
                    "name": "zero_fit_mean_C_retains_auxiliary_validity",
                    "pass": bool(torch.equal(zeroed[:, :, 775], x[:, :, 775])),
                },
            ]
        )
    else:
        checks.extend(
            [
                {
                    "name": "zero_fit_mean_B_retains_shared_validity_metadata",
                    "pass": bool(torch.equal(zeroed[:, :, 775], x[:, :, 775])),
                },
                {
                    "name": "causal_lag1_B_retains_shared_validity_metadata",
                    "pass": bool(torch.equal(lagged[:, :, 775], x[:, :, 775])),
                },
            ]
        )
    checks.append(
        {
            "name": "causal_lag1_changes_no_unapproved_columns",
            "pass": unchanged_except(x, lagged, allowed),
        }
    )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--formal-inventory", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--training-source", type=Path, required=True)
    parser.add_argument("--sensitivity-source", type=Path, required=True)
    parser.add_argument("--verified-sensitivity-source", type=Path, required=True)
    parser.add_argument("--backend-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=32)
    args = parser.parse_args()

    if args.rows != 32:
        raise ValueError("this bounded review is frozen to 32 outer common-population rows")
    train = load_module("round7_train_sensitivity_review", args.training_source.resolve())
    sensitivity = load_module("round7_sensitivity_review", args.sensitivity_source.resolve())
    payload = torch.load(args.prepared, map_location="cpu", weights_only=False)
    prepared_audit = train.validate_prepared(payload, args.prepared)
    horizons = prepared_audit["horizons"]
    outer = payload["timelines"]["outer"]
    common_indices = torch.nonzero(outer["common_mask"].bool(), as_tuple=True)[0]
    if len(common_indices) < args.rows:
        raise ValueError("fewer than 32 common outer rows")
    indices = common_indices[: args.rows]
    identities = [(str(outer["episode_id"][i]), int(outer["t"][i])) for i in indices.tolist()]
    if len(set(identities)) != args.rows:
        raise ValueError("selected outer identities are not unique")

    inventory = json.loads(args.formal_inventory.read_text())
    inventory_runs = {(x["group"], int(x["seed"])): Path(x["output"]) for x in inventory["runs"]}
    backend = json.loads(args.backend_audit.read_text())
    sources = {
        str(path.resolve()): sha256(path)
        for path in (
            args.prepared,
            args.formal_inventory,
            args.training_source,
            args.sensitivity_source,
            args.verified_sensitivity_source,
            args.backend_audit,
            Path(__file__).resolve(),
        )
    }
    replay_checks = []
    intervention_checks = []
    assembled_by_group = {}
    for group in GROUPS:
        full_x = train.assemble_inputs(payload, outer, group)
        x = full_x[indices].contiguous()
        assembled_by_group[group] = x
        group_checks = check_interventions(sensitivity, x, group)
        for check in group_checks:
            intervention_checks.append({"group": group, **check})
        for seed in SEEDS:
            folder = inventory_runs.get((group, seed))
            expected_folder = args.formal_root / f"future_{group}_{seed}"
            if folder is None or folder.resolve() != expected_folder.resolve():
                raise ValueError(f"formal inventory identity mismatch for {group}/{seed}")
            summary_path = folder / "summary.json"
            summary = json.loads(summary_path.read_text())
            if summary.get("status") != "complete" or not summary.get("formal") or summary.get("smoke"):
                raise ValueError(f"non-formal parent {group}/{seed}")
            if summary.get("group") != group or int(summary.get("seed")) != seed:
                raise ValueError(f"parent identity mismatch {group}/{seed}")
            if summary["run_config"]["data_sha256"] != sha256(args.prepared):
                raise ValueError(f"prepared hash mismatch {group}/{seed}")
            best = Path(summary["artifacts"]["best"]["path"])
            if sha256(best) != summary["artifacts"]["best"]["sha256"]:
                raise ValueError(f"best checkpoint hash mismatch {group}/{seed}")
            sources[str(summary_path.resolve())] = sha256(summary_path)
            sources[str(best.resolve())] = sha256(best)
            model = train.MultiWindowGRU(776, 128, len(horizons))
            model.load_state_dict(torch.load(best, map_location="cpu", weights_only=False)["model_state"], strict=True)
            model.eval()
            actual = train.predict(model, x, "cpu", args.rows).astype(np.float64)
            prediction_path = Path(summary["artifacts"]["predictions"]["outer"]["timeline"]["path"])
            sources[str(prediction_path.resolve())] = sha256(prediction_path)
            saved = prediction_map(prediction_path, horizons)
            expected = np.stack([saved[key] for key in identities])
            error = float(np.max(np.abs(actual - expected)))
            replay_checks.append(
                {
                    "group": group,
                    "seed": seed,
                    "rows": args.rows,
                    "horizons": horizons,
                    "max_abs_error": error,
                    "tolerance": TOLERANCE,
                    "pass": error <= TOLERANCE,
                }
            )

    backend_outer = [x for x in backend.get("checks", []) if x.get("role") == "outer" and x.get("group") in GROUPS]
    backend_identity_pass = (
        backend.get("status") == "pass"
        and backend.get("tolerance") == TOLERANCE
        and len(backend_outer) == 6
        and all(int(x.get("rows", -1)) == 32 and x.get("pass") for x in backend_outer)
    )
    checks = {
        "six_formal_models_replayed": len(replay_checks) == 6,
        "same_32_outer_common_identities_for_all_models": len(identities) == 32,
        "independent_cpu_replay_within_frozen_tolerance": all(x["pass"] for x in replay_checks),
        "existing_backend_audit_outer_subset_consistent": backend_identity_pass,
        "intervention_semantics_all_pass": all(x["pass"] for x in intervention_checks),
        "verified_exporter_intervention_is_ast_identical_to_reviewed_implementation": function_ast(
            args.verified_sensitivity_source, "intervention"
        )
        == function_ast(args.sensitivity_source, "intervention"),
    }
    result = {
        "schema": "round7_sensitivity_input_independent_review_v1",
        "status": "pass" if all(checks.values()) else "fail",
        "scope": "six B/C formal models x the same 32 common-population outer rows; CPU only",
        "selected_identity_sha256": hashlib.sha256(json.dumps(identities, separators=(",", ":")).encode()).hexdigest(),
        "selected_first_identity": identities[0],
        "selected_last_identity": identities[-1],
        "checks": checks,
        "replay_checks": replay_checks,
        "maximum_replay_absolute_error": max(x["max_abs_error"] for x in replay_checks),
        "intervention_checks": intervention_checks,
        "semantics": {
            "zero_fit_mean": "only active normalized force-derived feature columns are set to zero; zero is the fit mean; validity is retained",
            "causal_lag1": "force-derived history slot j receives only original slot j-1 and slot 0 receives fit-mean zero; C auxiliary validity follows the same causal shift",
            "boundary": "this audit proves input assembly and intervention causality on a bounded replay subset; it does not establish causal effects or statistical significance",
        },
        "source_hashes": sources,
    }
    atomic_json(args.output, result)
    print(json.dumps({"status": result["status"], "max_abs_error": result["maximum_replay_absolute_error"], "checks": checks}))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
