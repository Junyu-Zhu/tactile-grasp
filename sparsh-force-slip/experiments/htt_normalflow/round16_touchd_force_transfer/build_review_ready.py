import argparse, csv, hashlib, json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
p.add_argument("--code", type=Path, required=True)
a = p.parse_args()
O, C = a.root, a.code

audit = O / "audit"
evaluation = O / "evaluation"
required_json = {
    "touchd_release": audit / "TOUCHD_GELSIGHT_AUDIT.json",
    "pair_order": audit / "PAIR_ORDER_AUDIT.json",
    "touchd_formal": audit / "T_FORMAL_AUDIT.json",
    "htt_formal": audit / "HTT_FORMAL_AUDIT.json",
    "downstream_formal": audit / "DOWNSTREAM_FORMAL_AUDIT.json",
    "slip_compatibility": audit / "SLIP_COMPATIBILITY_AUDIT.json",
    "bootstrap_equivalence": audit / "BOOTSTRAP_EQUIVALENCE.json",
    "bootstrap_summary": evaluation / "bootstrap" / "SUMMARY.json",
    "failure_cases": evaluation / "failure_cases" / "SUMMARY.json",
    "fixed_case": evaluation / "fixed_case_figure" / "SUMMARY.json",
    "key_results": evaluation / "reporting" / "KEY_RESULTS.json",
    "old_force_regression": evaluation / "old_force_regression" / "AUDIT.json",
    "case_evidence": evaluation / "case_evidence" / "AUDIT.json",
    "window_diagnostics": evaluation / "window_diagnostics" / "AUDIT.json",
    "low_fpr_curves": evaluation / "low_fpr_curves" / "SUMMARY.json",
    "future_note_repair": audit / "FUTURE_SUMMARY_NOTE_REPAIR.json",
    "review_repairs": audit / "REVIEW_REPAIRS_AUDIT.json",
}
loaded = {}
for key, path in required_json.items():
    assert path.is_file(), (key, path)
    value = json.loads(path.read_text())
    assert value.get("status") in {"pass", "complete"}, (key, value.get("status"))
    loaded[key] = value

assert loaded["bootstrap_summary"]["rows"] == 8544
assert loaded["bootstrap_summary"]["draws_per_fold"] == 2000
assert loaded["bootstrap_equivalence"]["draws"] == 7
assert loaded["bootstrap_equivalence"]["same_draws_shared_across_three_seeds"]
assert all(loaded["bootstrap_equivalence"]["paired_metric_values_exact"].values())
assert all(loaded["bootstrap_equivalence"]["linear_quantiles_exact"].values())
assert loaded["slip_compatibility"]["V_thresholds_exactly_match_accepted_R13"] == 276
assert loaded["slip_compatibility"]["threshold_rows"] == 828 and loaded["slip_compatibility"]["metric_rows"] == 3348
assert loaded["downstream_formal"]["current_slip_runs"] == 24
assert loaded["downstream_formal"]["future_runs"] == 24
assert loaded["htt_formal"]["runs"] == 24
assert len(loaded["touchd_formal"]["runs"]) == 3

with (evaluation / "bootstrap" / "PAIRED_CI.csv").open(newline="") as f:
    ci = list(csv.DictReader(f))
assert len(ci) == 8544
assert {r["task"] for r in ci} == {"force", "slip", "future"}
assert all(int(r["valid"]) > 0 for r in ci)

checkpoint_entries = []
stage_specs = [
    ("touchd", "summary.json", 3),
    ("htt_force", "training_summary.json", 24),
    ("slip", "summary.json", 24),
    ("future", "summary.json", 24),
]
for stage, summary_name, expected in stage_specs:
    summaries = sorted((O / "formal" / stage).glob(f"*/{summary_name}"))
    assert len(summaries) == expected, (stage, len(summaries))
    for summary_path in summaries:
        value = json.loads(summary_path.read_text())
        assert value["status"] == "complete"
        run = summary_path.parent
        record = {
            "stage": stage,
            "run": run.name,
            "summary": str(summary_path),
            "summary_sha256": sha(summary_path),
            "best": str(run / "best.pth"),
            "best_sha256": sha(run / "best.pth"),
            "latest": str(run / "latest.pth"),
            "latest_sha256": sha(run / "latest.pth"),
        }
        if value.get("best_checkpoint_sha256"):
            assert record["best_sha256"] == value["best_checkpoint_sha256"]
        if value.get("latest_checkpoint_sha256"):
            assert record["latest_sha256"] == value["latest_checkpoint_sha256"]
        checkpoint_entries.append(record)

checkpoint_index = {
    "schema": "round16_checkpoint_index_v1",
    "status": "complete",
    "neural_runs": len(checkpoint_entries),
    "counts": {stage: expected for stage, _, expected in stage_specs},
    "best_and_latest_rehashed": True,
    "entries": checkpoint_entries,
}
atomic(O / "CHECKPOINT_INDEX.json", checkpoint_index)

receipt = {
    "schema": "round16_evaluation_execution_v1",
    "status": "complete_after_local_bootstrap_retry",
    "reason_for_retry": "The first correct metric-outer bootstrap implementation was stopped for excessive repeated aggregation; no official CI was emitted by it.",
    "formal_definition_changed": False,
    "bootstrap": {
        "draws_per_fold": 2000,
        "seed_formula": "2026091601+fold_index-1",
        "paired_ci_rows": 8544,
        "optimization_equivalence_audit": str(audit / "BOOTSTRAP_EQUIVALENCE.json"),
    },
    "completed": {
        "slip_runs": 36,
        "force_runs": 24,
        "future_runs": 24,
        "future_diagnostics_runs": 24,
        "future_range_runs": 24,
        "failure_cases": loaded["failure_cases"]["selected_rows"],
        "fixed_case_figure": True,
        "report": True,
        "bounded_review_repairs": {
            "old_force_readonly_runs": 24,
            "case_curve_plots": 48,
            "raw_tactile_frames": 4,
            "low_fpr_curve_runs": 36,
            "descriptive_window_added_after_results": True,
            "training_or_existing_ci_rerun": False,
        },
    },
    "test_consumed": False,
}
atomic(evaluation / "EXECUTION.json", receipt)

for name in ("COST_SCOPE.json", "OLD_DOMAIN_SCOPE.json", "SCIENTIFIC_CONCLUSION.md", "DELIVERY_REPORT.md", "INDEPENDENT_REVIEW_REQUEST.md", "REVIEW_REVISION.md"):
    source = C / name
    target = audit / name
    target.write_bytes(source.read_bytes())

artifacts = {
    **required_json,
    "paired_ci": evaluation / "bootstrap" / "PAIRED_CI.csv",
    "group_draws": evaluation / "bootstrap" / "GROUP_DRAWS.json",
    "fixed_case_source": evaluation / "fixed_case_figure" / "FIXED_CASE_SOURCE.csv",
    "fixed_case_figure": evaluation / "fixed_case_figure" / "FIXED_CASE_TIMELINE.svg",
    "summary_zh": evaluation / "reporting" / "SUMMARY_ZH.md",
    "execution": evaluation / "EXECUTION.json",
    "checkpoint_index": O / "CHECKPOINT_INDEX.json",
    "cost_scope": audit / "COST_SCOPE.json",
    "old_domain_scope": audit / "OLD_DOMAIN_SCOPE.json",
    "scientific_conclusion": audit / "SCIENTIFIC_CONCLUSION.md",
    "delivery_report": audit / "DELIVERY_REPORT.md",
    "independent_review_request": audit / "INDEPENDENT_REVIEW_REQUEST.md",
    "review_revision": audit / "REVIEW_REVISION.md",
    "old_force_summary_metrics": evaluation / "old_force_regression" / "SUMMARY_METRICS.csv",
    "old_force_per_trial_axis": evaluation / "old_force_regression" / "PER_TRIAL_AXIS.csv",
    "case_raw_index": evaluation / "case_evidence" / "TACTILE_RAW_INDEX.csv",
    "case_raw_contact_sheet": evaluation / "case_evidence" / "TACTILE_RAW_CONTACT_SHEET.png",
    "case_curve_source": evaluation / "case_evidence" / "CASE_CURVE_SOURCE.csv",
    "case_plot_index": evaluation / "case_evidence" / "CASE_PLOT_INDEX.csv",
    "force_trial_axis_diagnostics": evaluation / "window_diagnostics" / "FORCE_TRIAL_AXIS_DIAGNOSTICS.csv",
    "force_window_axis_diagnostics": evaluation / "window_diagnostics" / "FORCE_WINDOW_AXIS_DIAGNOSTICS.csv",
    "future_window_axis_diagnostics": evaluation / "window_diagnostics" / "FUTURE_WINDOW_AXIS_DIAGNOSTICS.csv",
    "low_fpr_curve_csv": evaluation / "low_fpr_curves" / "LOW_FPR_CURVES.csv",
    "low_fpr_curve_svg": evaluation / "low_fpr_curves" / "LOW_FPR_CURVES.svg",
}
index = {
    "schema": "round16_pre_independent_review_index_v1",
    "status": "ready_for_one_independent_review",
    "final_manifest": False,
    "reason_not_final": "Final manifest and three-location SHA freeze wait for independent-review PASS and root metadata freeze.",
    "formal_neural_runs": 75,
    "released_rows": 81493,
    "eligible_pairs": 80783,
    "paired_ci_rows": 8544,
    "test_consumed": False,
    "artifacts": {k: {"path": str(v), "sha256": sha(v), "bytes": v.stat().st_size} for k, v in artifacts.items()},
}
atomic(O / "REVIEW_READY.json", index)
print(json.dumps({"status": "ready", "checkpoints": len(checkpoint_entries), "artifacts": len(artifacts)}))
