#!/usr/bin/env python3
"""Independent pre-formal review for Round 8 preparation and training."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prepare-audit", type=Path, required=True)
    parser.add_argument("--p4-decision", type=Path, required=True)
    parser.add_argument("--smoke-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train = load_module("r8_review_train", args.round_root / "training/train.py")
    prepare = load_module("r8_review_prepare", args.round_root / "prepare/prepare.py")
    payload = torch.load(args.prepared, map_location="cpu", weights_only=False)
    prepared_audit = json.loads(args.prepare_audit.read_text())
    p4 = json.loads(args.p4_decision.read_text())
    smoke = json.loads(args.smoke_audit.read_text())
    checks = {}
    checks["prepared_sha_bound"] = train.sha256(args.prepared) == prepared_audit["prepared"]["sha256"] == p4["prepared_sha256"] == smoke["real_prepared_sha256"]
    checks["trainer_accepts_prepared"] = train.validate_prepared(payload, args.prepared)["data_sha256"] == train.sha256(args.prepared)
    prepare_checks = prepared_audit.get("checks", {})
    checks["prepare_audit_complete"] = (
        prepared_audit.get("status") == "complete"
        and prepare_checks.get("test_or_global_heldout_read") is False
        and all(value is True for name, value in prepare_checks.items() if name != "test_or_global_heldout_read")
    )
    checks["p4_support_passes_fixed_gates"] = p4.get("status") == "complete" and p4.get("triggered") is True and p4.get("target_is_physical_ground_truth") is False
    checks["smoke_all_eight_groups_pass"] = smoke.get("status") == "pass" and len(smoke.get("runs", [])) == 8 and all(all(run["checks"].values()) for run in smoke["runs"])
    checks["smoke_resume_exact"] = set(smoke.get("actual_interruption_comparisons", {})) == {"C_xyz_delta", "C_hazard", "P4_state"} and all(
        all(values.values()) for values in smoke.get("actual_interruption_comparisons", {}).values()
    )
    checks["smoke_not_formal_parent"] = smoke.get("smoke_is_not_formal_parent") is True

    role_groups = {role: set(payload["role_groups"][role]) for role in train.ROLES}
    checks["roles_pairwise_disjoint"] = all(not role_groups[left] & role_groups[right]
                                               for index, left in enumerate(train.ROLES) for right in train.ROLES[index + 1:])
    exact_delta = True; event_consistent = True; current_static = True; no_future_input = True
    for role in train.ROLES:
        row = payload["roles"][role]
        force = row["base"][:, :, 769:772]
        expected_delta = torch.zeros_like(row["force_delta_slots"]); expected_delta[:, 5:] = force[:, 5:] - force[:, :4]
        exact_delta &= torch.equal(expected_delta, row["force_delta_slots"])
        occurrence, observed, event_bin = row["event_occurrence"].bool(), row["event_observed_mask"].bool(), row["event_time_bin"].long()
        event_consistent &= not bool((occurrence & ~observed).any()) and not bool((occurrence.sum(1) > 1).any())
        event_consistent &= all(values == sorted(values, reverse=True) for values in observed.tolist())
        for column, horizon in enumerate((1, 3, 5)):
            mask = row["horizon_mask"][:, column].bool()
            expected = ((event_bin > 0) & (event_bin <= horizon)).float()
            known = ((event_bin > 0) & (event_bin <= horizon)) | observed[:, horizon - 1]
            event_consistent &= not bool((row["y"][mask, column] != expected[mask]).any()) and not bool((mask & ~known).any())
        current_static &= not bool((row["current_slip_label"][row["horizon_mask"].any(1)] != 0).any())
        altered = dict(row)
        altered["future_force_target"] = torch.randn_like(row["future_force_target"])
        altered["future_force_observed_mask"] = ~row["future_force_observed_mask"].bool()
        for group in train.json.loads(train.PROTOCOL.read_text())["groups"]:
            no_future_input &= torch.equal(train.assemble_inputs(payload, row, group), train.assemble_inputs(payload, altered, group))
    checks["signed_xyz_lag5_exact"] = exact_delta
    checks["event_masks_labels_and_censor_consistent"] = event_consistent
    checks["eligible_training_endpoints_current_static"] = current_static
    checks["future_force_target_not_model_input"] = no_future_input

    fit = payload["roles"]["fit_train"]
    expected_normalizers = {
        "base": prepare.population_stats(fit["base"].reshape(-1, 772)),
        "force_delta": prepare.population_stats(fit["force_delta_slots"][:, 5:].reshape(-1, 3)),
        "future_force": prepare.population_stats(fit["future_force_target"][fit["future_force_observed_mask"]]),
    }
    checks["normalizers_recompute_from_fit_only"] = payload["normalization"]["fit_role"] == "fit_train" and all(
        all(torch.equal(payload["normalization"][name][field], expected_normalizers[name][field]) for field in ("mean", "std", "variance"))
        for name in expected_normalizers
    )
    p4_direct, direct_hash = train.fixed_initial_state(20260914, "P4_direct")
    p4_state, state_hash = train.fixed_initial_state(20260914, "P4_state")
    checks["p4_same_architecture_and_initialization"] = direct_hash == state_hash and train.nested_equal(p4_direct, p4_state)
    p4_model = train.make_model("P4_state")
    checks["p4_linear_state_and_hidden_bypass_disclosed"] = (
        isinstance(p4_model.state, torch.nn.Linear) and isinstance(p4_model.risk, torch.nn.Linear)
        and p4_model.risk.in_features == 143
        and "代数合并" in (args.round_root / "PROTOCOL.md").read_text()
        and "隐藏直连" in (args.round_root / "PROTOCOL.md").read_text()
    )
    analysis = json.loads((args.round_root / "ANALYSIS_PROTOCOL.json").read_text())
    checks["p4_state_ablation_and_baselines_preregistered"] = (
        "normalized predicted state15 into risk set0" in analysis.get("P4_state_ablation", "")
        and analysis.get("P4_state_baselines") == ["persistence current predictedXYZ", "fit-only ridge linear future predictedXYZ"]
        and "no refit" in analysis.get("P4_state_ablation", "")
    )
    formal_summaries = []
    for path in args.output_root.glob("future_*/summary.json"):
        value = json.loads(path.read_text())
        if value.get("formal"):
            formal_summaries.append(str(path))
    checks["no_formal_training_started_before_review"] = not formal_summaries
    checks["test_role_not_consumed"] = prepared_audit["checks"].get("test_or_global_heldout_read") is False

    source_paths = [args.round_root / name for name in (
        "PROTOCOL.md", "NUMERIC_PROTOCOL.json", "ANALYSIS_PROTOCOL.json", "prepare/prepare.py", "prepare/protocol.json",
        "training/train.py", "training/protocol.json", "run_training_queue.py", "infer_interventions.py"
    )] + [args.prepared, args.prepare_audit, args.p4_decision, args.smoke_audit]
    result = {
        "schema": "round8_p0_independent_review_v1", "status": "pass" if all(checks.values()) else "fail",
        "scope": "preparation/training/protocol review independent of their authors; evaluator self-review excluded",
        "checks": checks, "failures": [name for name, passed in checks.items() if not passed],
        "source_hashes": {str(path.resolve()): train.sha256(path) for path in source_paths},
        "prepared_role_rows": {role: len(payload["roles"][role]["t"]) for role in train.ROLES},
        "p4_limitation": "state=Linear(hidden) and risk=Linear([hidden,state]) can be algebraically merged; hidden bypass remains; P4 tests auxiliary-supervision inductive bias, not world-model validity by structure alone",
        "formal_summaries_seen": formal_summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    train.atomic_json(args.output, result)
    print(json.dumps({"status": result["status"], "checks": len(checks), "failures": result["failures"]}))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
