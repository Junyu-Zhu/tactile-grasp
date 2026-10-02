#!/usr/bin/env python3
"""Build the fixed Round-8 result tables, figures, checkpoint index, and Chinese report."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "protocol.json"
ROUND_ROOT = HERE.parent


def pyplot():
    # Matplotlib 3.7 imports a deprecated pyparsing alias in this frozen environment.
    # Suppress only that upstream compatibility warning; all project warnings remain errors under strict tests.
    from pyparsing.warnings import PyparsingDeprecationWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PyparsingDeprecationWarning)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as local_pyplot
    return local_pyplot
GROUP_ORDER = (
    "B_xyz", "C_xyz_delta", "C_hazard", "P3_concat_independent",
    "P3_fusion_independent", "P3_fusion_hazard", "P4_direct", "P4_state",
)
SEEDS = (20260914, 20260915, 20260916)
HORIZONS = (1, 3, 5)
GROUP_LABEL = {
    "B_xyz": "B XYZ",
    "C_xyz_delta": "C XYZ+delta",
    "C_hazard": "C hazard",
    "P3_concat_independent": "P3 concat",
    "P3_fusion_independent": "P3 fusion",
    "P3_fusion_hazard": "P3 fusion hazard",
    "P4_direct": "P4 direct",
    "P4_state": "P4 state",
}
KEY_COMPARISONS = (
    ("C_xyz_delta", "B_xyz", "完整XYZ历史下显式力差分"),
    ("C_hazard", "C_xyz_delta", "事件时间头相对独立窗口头"),
    ("P3_fusion_independent", "P3_concat_independent", "时序融合相对容量拼接"),
    ("P3_fusion_independent", "C_xyz_delta", "时序融合相对普通GRU"),
    ("P3_fusion_hazard", "P3_fusion_independent", "融合结构内事件时间头"),
    ("P3_fusion_hazard", "C_hazard", "融合+hazard相对普通GRU+hazard"),
    ("P4_state", "P4_direct", "未来状态辅助监督"),
)
R7_REFERENCE = (
    {"group": "A_visual", "force_layout": "none", "H1_AP": .3932, "H3_AP": .4145, "H5_AP": .3985,
     "H3_trial_FA": .0760, "H3_event_recall": .4196, "H3_mean_lead": 1.601},
    {"group": "B_force", "force_layout": "Fn,Ft,Ft/Fn", "H1_AP": .4381, "H3_AP": .4482, "H5_AP": .4261,
     "H3_trial_FA": .0780, "H3_event_recall": .4412, "H3_mean_lead": 1.588},
    {"group": "C_force_delta", "force_layout": "Fn,Ft,Ft/Fn + signed XYZ delta", "H1_AP": .6241, "H3_AP": .7917, "H5_AP": .7502,
     "H3_trial_FA": .0916, "H3_event_recall": .6392, "H3_mean_lead": 1.939},
    {"group": "D_visual_delta", "force_layout": "none; PCA3 visual delta", "H1_AP": .4035, "H3_AP": .4301, "H5_AP": .4099,
     "H3_trial_FA": .0741, "H3_event_recall": .4176, "H3_mean_lead": 1.619},
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reporting_source_paths() -> tuple[Path, ...]:
    return (Path(__file__).resolve(), (HERE / "REPRODUCE.md").resolve(), PROTOCOL.resolve())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(text)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty table {path}")
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows):
        raise ValueError(f"inconsistent table fields {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def as_bool(value: Any) -> bool:
    return str(value).lower() in {"true", "1"}


def number(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return math.nan
    result = float(value)
    return result if math.isfinite(result) else math.nan


def summarize(values: Iterable[float]) -> tuple[float, float, int]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return math.nan, math.nan, 0
    return float(statistics.mean(finite)), float(statistics.stdev(finite)) if len(finite) > 1 else math.nan, len(finite)


def aggregate_rows(rows: list[dict[str, Any]], keys: tuple[str, ...], metrics: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[key]) for key in keys)].append(row)
    result = []
    for identity, subset in grouped.items():
        item: dict[str, Any] = dict(zip(keys, identity))
        seeds = sorted({str(row.get("seed", "")) for row in subset if str(row.get("seed", ""))})
        item["seeds"] = ";".join(seeds)
        for metric in metrics:
            mean, sample_sd, count = summarize(number(row, metric) for row in subset)
            item[f"{metric}_mean"] = mean
            item[f"{metric}_sample_sd"] = sample_sd
            item[f"{metric}_n"] = count
        result.append(item)
    return result


def require_evaluation(directory: Path) -> dict[str, Path]:
    names = {
        "summary": "summary.json", "metrics": "metrics.csv", "aggregate": "all_seed_summary.csv",
        "paired": "paired_bootstrap_ci.csv", "state": "state_prediction_metrics.csv",
        "interventions": "intervention_metrics.csv", "consistency": "multi_horizon_consistency.csv",
        "failures": "failure_cases.csv", "gate": "fixed_multiplicative_gate_diagnostic.csv",
        "reliability": "reliability.csv", "threshold_transfer": "threshold_transfer.csv",
        "baseline_models": "baseline_models.json", "state_linear_model": "state_linear_baseline.json",
        "state_trials": "state_per_trial.csv", "state_paired": "state_paired_bootstrap_ci.csv",
    }
    paths = {key: directory / name for key, name in names.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing formal evaluation artifacts: " + ", ".join(missing))
    summary = read_json(paths["summary"])
    if summary.get("format") != "round8_formal_evaluation_v1" or summary.get("status") != "complete":
        raise ValueError("formal evaluation is not complete")
    if summary.get("test_role_consumed") is not False:
        raise ValueError("test role boundary violated")
    expected_hashes = summary.get("output_hashes", {})
    actual_files = [path for path in directory.rglob("*") if path.is_file() and path.name != "summary.json"]
    actual_hashes = {str(path.relative_to(directory)): sha256(path) for path in actual_files}
    if actual_hashes != expected_hashes:
        raise ValueError("formal evaluation output hash manifest mismatch")
    return paths


def require_state_diagnostics(directory: Path, prediction_path: Path, evaluation_summary: Path) -> list[dict[str, Any]]:
    audit_path = directory / "STATE_DIAGNOSTIC_AUDIT.json"
    summary_path = directory / "P4_STATE_SEED_STEP_SUMMARY.csv"
    per_trial_path = directory / "P4_STATE_PER_TRIAL_TEMPORAL_DIAGNOSTICS.csv"
    if any(not path.is_file() for path in (audit_path, summary_path, per_trial_path)):
        raise FileNotFoundError("state diagnostic artifacts are incomplete")
    audit = read_json(audit_path)
    if audit.get("schema") != "round8_p4_state_temporal_diagnostic_v1" or audit.get("status") != "complete" or audit.get("test_role_consumed") is not False:
        raise ValueError("state diagnostic audit is not complete")
    sources = audit.get("source_hashes", {})
    if sources.get(str(prediction_path.resolve())) != sha256(prediction_path) or sources.get(str(evaluation_summary.resolve())) != sha256(evaluation_summary):
        raise ValueError("state diagnostic accepted-manifest/evaluation binding drift")
    expected_outputs = {summary_path.name: sha256(summary_path), per_trial_path.name: sha256(per_trial_path)}
    if audit.get("output_hashes") != expected_outputs:
        raise ValueError("state diagnostic output hash mismatch")
    rows = read_csv(summary_path)
    expected = {(population, seed, step) for population in ("primary_common", "full_timeline") for seed in SEEDS for step in range(1, 6)}
    actual = {(row["population"], int(row["seed"]), int(row["step"])) for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError("state diagnostic population/seed/step grid incomplete or duplicated")
    metrics = ("prediction_temporal_variance_xyz_trial_mean", "target_temporal_variance_xyz_trial_mean",
               "persistence_temporal_variance_xyz_trial_mean", "prediction_target_mse_xyz_trial_mean",
               "target_vs_current_mse_xyz_trial_mean", "linear_target_mse_xyz_trial_mean",
               "prediction_vs_current_mse_xyz_trial_mean")
    return sorted(aggregate_rows(rows, ("population", "step"), metrics), key=lambda row: (row["population"], int(row["step"])))


def require_group_horizon_seed_grid(rows: list[dict[str, Any]], label: str) -> None:
    actual = {(str(row["group"]), int(row["horizon"]), int(row["seed"])) for row in rows}
    expected = {(group, horizon, seed) for group in GROUP_ORDER for horizon in HORIZONS for seed in SEEDS}
    if actual != expected or len(rows) != len(expected):
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{label} grid mismatch: missing={missing[:5]} extra={extra[:5]}")


def require_checkpoint_grid(rows: list[dict[str, str]]) -> None:
    actual = {(row.get("group"), int(row.get("seed", -1)), row.get("kind")) for row in rows}
    expected = {(group, seed, kind) for group in GROUP_ORDER for seed in SEEDS for kind in ("best", "latest", "config")}
    if actual != expected or len(rows) != len(expected):
        raise ValueError("checkpoint index does not contain exactly best/latest/config for all 24 runs")


def require_provenance(prediction_path: Path, evaluation_path: Path, training_path: Path,
                       checkpoint_path: Path, evaluation_summary: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    prediction, evaluation = read_json(prediction_path), read_json(evaluation_path)
    for name, manifest in (("prediction", prediction), ("evaluation", evaluation)):
        if manifest.get("schema") != "round8_prediction_manifest_v1" or manifest.get("status") != "complete":
            raise ValueError(f"{name} manifest is not an accepted Round-8 prediction manifest")
        if manifest.get("groups") != list(GROUP_ORDER) or tuple(manifest.get("seeds", [])) != SEEDS:
            raise ValueError(f"{name} manifest grid drift")
    if evaluation.get("training_audit") != prediction.get("training_audit") or evaluation.get("checkpoint_index") != prediction.get("checkpoint_index") or evaluation.get("prepared") != prediction.get("prepared"):
        raise ValueError("evaluation and accepted prediction manifests do not share audit/index/prepared identity")
    if prediction["training_audit"]["sha256"] != sha256(training_path) or Path(prediction["training_audit"]["path"]).resolve() != training_path.resolve():
        raise ValueError("training audit is not the accepted prediction-manifest audit")
    if prediction["checkpoint_index"]["sha256"] != sha256(checkpoint_path) or Path(prediction["checkpoint_index"]["path"]).resolve() != checkpoint_path.resolve():
        raise ValueError("checkpoint index is not the accepted prediction-manifest index")
    eval_sources = evaluation_summary.get("source_hashes", {})
    if eval_sources.get(str(evaluation_path.resolve())) != sha256(evaluation_path):
        raise ValueError("formal evaluation summary is not bound to the supplied evaluation manifest")
    prediction_artifacts = {(a["group"], int(a["seed"]), a["role"], a["path"], a["sha256"]) for a in prediction["artifacts"]}
    evaluation_artifacts = {(a["group"], int(a["seed"]), a["role"], a["path"], a["sha256"]) for a in evaluation["artifacts"]}
    if prediction_artifacts != evaluation_artifacts or len(prediction_artifacts) != 72:
        raise ValueError("evaluation manifest does not preserve the 72 accepted prediction timelines")
    return prediction, evaluation


def require_e2e_identity(items: list[dict[str, Any]], prediction_path: Path, checkpoints: list[dict[str, str]],
                         prepared_sha: str, checkpoint_index_sha: str) -> None:
    expected = {(group, SEEDS[0]) for group in GROUP_ORDER}
    if {(item.get("group"), int(item.get("seed", -1))) for item in items} != expected or len(items) != len(expected):
        raise ValueError("end-to-end inputs must contain every group exactly once at the fixed representative seed")
    best = {(row["group"], int(row["seed"])): row for row in checkpoints if row["kind"] == "best"}
    common = {(item.get("episode"), int(item.get("t", -1)), item.get("source_sha256"), item.get("prepared_sha256")) for item in items}
    if len(common) != 1:
        raise ValueError("end-to-end benchmarks do not share episode/t/source/prepared identity")
    if len({item.get("analysis_protocol_sha256") for item in items}) != 1 or len({item.get("code_sha256") for item in items}) != 1:
        raise ValueError("end-to-end benchmark protocol or code identity drift")
    if any(item.get("analysis_protocol_sha256") != sha256(ROUND_ROOT / "ANALYSIS_PROTOCOL.json")
           or item.get("code_sha256") != sha256(ROUND_ROOT / "benchmark_e2e.py") for item in items):
        raise ValueError("end-to-end benchmark does not match the current locked protocol/source")
    for item in items:
        indexed = best[(item["group"], int(item["seed"]))]
        if (Path(item["checkpoint"]).resolve() != Path(indexed["path"]).resolve()
                or item["checkpoint_sha256"] != indexed["sha256"]):
            raise ValueError("end-to-end checkpoint is not the accepted indexed best checkpoint")
        if item.get("accepted_manifest_sha256") != sha256(prediction_path) or item.get("prepared_sha256") != prepared_sha:
            raise ValueError("end-to-end accepted-manifest/prepared binding drift")
        if item.get("checkpoint_index_sha256") != checkpoint_index_sha:
            raise ValueError("end-to-end checkpoint-index binding drift")
        if item.get("status") != "complete" or item.get("cache_identity_diagnostics", {}).get("pass") is not True:
            raise ValueError("an end-to-end benchmark is incomplete or failed cache parity")
        if item.get("upstream_frozen") is not True or item.get("head_frozen") is not True:
            raise ValueError("an end-to-end benchmark mutated a frozen model")


def raw_table(metrics: list[dict[str, str]]) -> list[dict[str, Any]]:
    selected = [row for row in metrics if row.get("method_type") == "neural_operational"
                and row.get("population") == "primary" and row.get("rule") == "raw"
                and row.get("operating_point") == "fixed_0.5" and row.get("threshold_status") == "available"]
    require_group_horizon_seed_grid(selected, "raw risk")
    table = aggregate_rows(selected, ("group", "horizon"),
                           ("average_precision", "brier", "prevalence", "balanced_accuracy", "macro_f1",
                            "tn", "fp", "fn", "tp"))
    return sorted(table, key=lambda row: (GROUP_ORDER.index(row["group"]), int(row["horizon"])))


def operational_table(metrics: list[dict[str, str]], horizon: int, operating_point: str) -> list[dict[str, Any]]:
    selected = [row for row in metrics if row.get("method_type") == "neural_operational"
                and row.get("population") == "primary" and int(row.get("horizon", -1)) == horizon
                and row.get("operating_point") == operating_point and as_bool(row.get("rule_selected"))]
    expected = {(group, horizon, seed) for group in GROUP_ORDER for seed in SEEDS}
    actual = {(row["group"], int(row["horizon"]), int(row["seed"])) for row in selected}
    if actual != expected or len(selected) != len(expected):
        raise ValueError("operational result grid is incomplete or duplicated")
    names = ("frame_fpr", "frame_recall", "trial_false_alarm_rate", "event_recall", "mean_lead_frames",
             "lead_ge_3_recall", "lead_ge_5_recall", "balanced_accuracy", "macro_f1", "prevalence",
             "tn", "fp", "fn", "tp")
    table = aggregate_rows(selected, ("group", "horizon", "operating_point"), names)
    for item in table:
        subset = [row for row in selected if row["group"] == item["group"]]
        item["never_alarm_seeds"] = sum(as_bool(row.get("never_alarm")) for row in subset)
        available = sorted(int(row["seed"]) for row in subset if row.get("threshold_status") == "available")
        unavailable = sorted(int(row["seed"]) for row in subset if row.get("threshold_status") != "available")
        item["threshold_status"] = "available" if len(available) == len(SEEDS) else "unavailable_incomplete_seed_grid"
        item["available_seeds"] = ";".join(map(str, available)); item["unavailable_seeds"] = ";".join(map(str, unavailable))
    return sorted(table, key=lambda row: GROUP_ORDER.index(row["group"]))


def calibration_table(metrics: list[dict[str, str]]) -> list[dict[str, Any]]:
    selected = [row for row in metrics if row.get("method_type") in {"neural_operational", "neural_shared_platt_raw"}
                and row.get("population") == "primary" and row.get("operating_point") == "fixed_0.5"
                and row.get("threshold_status") == "available"
                and (row.get("method_type") == "neural_shared_platt_raw" or row.get("rule") == "raw")]
    for method in ("neural_operational", "neural_shared_platt_raw"):
        require_group_horizon_seed_grid([row for row in selected if row["method_type"] == method], method)
    return sorted(aggregate_rows(selected, ("method_type", "group", "horizon"), ("average_precision", "brier")),
                  key=lambda row: (GROUP_ORDER.index(row["group"]), int(row["horizon"]), row["method_type"]))


def reliability_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("method") == "shared_monotone_platt_raw_risk":
            grouped[(row["group"], int(row["seed"]), int(row["horizon"]), row["role"])].append(row)
    run_rows = []
    for (group, seed, horizon, role), bins in grouped.items():
        total = sum(int(row["n"]) for row in bins)
        ece = sum(int(row["n"]) * abs(float(row["mean_probability"]) - float(row["positive_fraction"]))
                  for row in bins if int(row["n"]) > 0) / total
        run_rows.append({"group": group, "seed": seed, "horizon": horizon, "role": role, "ece": ece, "n": total})
    expected = {(g, s, h, role) for g in GROUP_ORDER for s in SEEDS for h in HORIZONS for role in ("calibration", "outer")}
    if {(r["group"], r["seed"], r["horizon"], r["role"]) for r in run_rows} != expected:
        raise ValueError("reliability grid is incomplete")
    return sorted(aggregate_rows(run_rows, ("group", "horizon", "role"), ("ece",)),
                  key=lambda row: (GROUP_ORDER.index(row["group"]), int(row["horizon"]), row["role"]))


def baseline_table(metrics: list[dict[str, str]], horizon: int, operating_point: str) -> list[dict[str, Any]]:
    rows = [row for row in metrics if row.get("method_type") == "fit_only_baseline"
            and int(row.get("horizon", -1)) == horizon and row.get("population") == "primary"
            and row.get("operating_point") == operating_point and as_bool(row.get("rule_selected"))
            and row.get("threshold_status") == "available"]
    expected = {"current_p_slip_raw", "history9_p_slip_mean_raw", "history9_p_slip_slope_fit",
                "latest_force_delta_fit", "elapsed_position_fit", "fit_prevalence"}
    if {row["group"] for row in rows} != expected or len(rows) != len(expected):
        raise ValueError("simple risk baseline grid is incomplete or duplicated")
    return [{key: row.get(key, "") for key in ("group", "horizon", "operating_point", "rule", "threshold",
                                                 "prevalence", "average_precision", "brier", "balanced_accuracy",
                                                 "macro_f1", "frame_fpr", "frame_recall", "tn", "fp", "fn", "tp",
                                                 "never_alarm", "trial_false_alarm_rate", "event_recall", "mean_lead_frames")}
            for row in rows]


def consistency_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    selected = [row for row in rows if row.get("role") == "outer"]
    return sorted(aggregate_rows(selected, ("group", "head_type", "method"), ("violation_fraction",)),
                  key=lambda row: (GROUP_ORDER.index(row["group"]), row["method"]))


def paired_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    labels = {f"{candidate}_vs_{comparator}": label for candidate, comparator, label in KEY_COMPARISONS}
    result = []
    for row in rows:
        if row.get("comparison") not in labels:
            continue
        status = row.get("status", "available")
        low, high, estimate = number(row, "ci_low"), number(row, "ci_high"), number(row, "estimate")
        if status != "available":
            direction = "unavailable_incomplete_seed_grid"
        elif math.isfinite(low) and low > 0:
            direction = "candidate_higher"
        elif math.isfinite(high) and high < 0:
            direction = "candidate_lower"
        else:
            direction = "interval_crosses_zero"
        result.append({"comparison": row["comparison"], "meaning_zh": labels[row["comparison"]], "horizon": int(row["horizon"]),
                       "operating_point": row["operating_point"], "metric": row["metric"], "estimate": estimate,
                       "ci_low": low, "ci_high": high, "direction": direction,
                       "status": status,
                       "valid_replicates": int(row["valid_replicates"]) if row.get("valid_replicates") else "",
                       "unit": row.get("unit", ""),
                       "candidate_available_seeds": row.get("candidate_available_seeds", ""),
                       "candidate_unavailable_seeds": row.get("candidate_unavailable_seeds", ""),
                       "comparator_available_seeds": row.get("comparator_available_seeds", ""),
                       "comparator_unavailable_seeds": row.get("comparator_unavailable_seeds", "")})
    expected = {(f"{a}_vs_{b}", 3, point, metric) for a, b, _ in KEY_COMPARISONS
                for point, metric in (("raw_common_population", "average_precision"),
                                      ("raw_common_population", "brier"),
                                      ("trial_FA_0.10", "event_recall"),
                                      ("trial_FA_0.10", "trial_false_alarm_rate"),
                                      ("trial_FA_0.10", "mean_lead_frames"))}
    identities = [(row["comparison"], row["horizon"], row["operating_point"], row["metric"]) for row in result]
    if not expected.issubset(set(identities)) or any(identities.count(key) != 1 for key in expected):
        raise ValueError("paired evidence grid required by the six questions is incomplete or duplicated")
    return result


def state_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    return aggregate_rows(rows, ("group", "method", "step"), ("rmse_xyz", "mae_xyz", "prediction_variance", "target_variance", "variance_ratio"))


def intervention_table(rows: list[dict[str, str]], horizon: int) -> list[dict[str, Any]]:
    subset = [row for row in rows if int(row.get("horizon", -1)) == horizon and row.get("population") == "primary"
              and row.get("operating_point") == "trial_FA_0.10" and row.get("intervention") != "fixed_multiplicative_gate"]
    expected_pairs = {(kind, group) for group in GROUP_ORDER
                      for kind in ("force_input_fit_mean_zero", "force_causal_lag1")}
    expected_pairs.add(("state_input_fit_mean_zero", "P4_state"))
    expected = {(kind, group, seed) for kind, group in expected_pairs for seed in SEEDS}
    actual = {(row["intervention"], row["group"], int(row["seed"])) for row in subset}
    if actual != expected or len(subset) != len(expected):
        raise ValueError("fixed intervention grid is incomplete or duplicated")
    return aggregate_rows(subset, ("intervention", "group", "horizon", "operating_point"),
                          ("mean_score_delta_vs_unperturbed", "trial_false_alarm_rate", "event_recall", "mean_lead_frames"))


def gate_table(rows: list[dict[str, str]], horizon: int, operating_point: str) -> list[dict[str, Any]]:
    selected = [row for row in rows if int(row.get("horizon", -1)) == horizon
                and row.get("population") == "primary" and row.get("operating_point") == operating_point]
    expected = {(group, seed) for group in GROUP_ORDER for seed in SEEDS}
    actual = {(row["group"], int(row["seed"])) for row in selected}
    if actual != expected or len(selected) != len(expected):
        raise ValueError("fixed multiplicative gate grid is incomplete or duplicated")
    metrics = ("delta_raw_average_precision", "delta_frame_recall", "delta_trial_false_alarm_rate",
               "delta_event_recall", "delta_mean_lead_frames")
    return sorted(aggregate_rows(selected, ("group", "horizon", "operating_point"), metrics),
                  key=lambda row: GROUP_ORDER.index(row["group"]))


def e2e_table(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for item in items:
        for mode in ("cold_history", "streaming"):
            blocks = item.get("measurement", {}).get(mode, [])
            samples = [float(value) for block in blocks for value in block.get("samples_ms", [])]
            mean, sample_sd, count = summarize(samples)
            if not samples:
                raise ValueError(f"empty end-to-end latency samples for {item.get('group')}/{mode}")
            result.append({"group": item["group"], "seed": int(item["seed"]), "mode": mode,
                           "latency_ms_mean": mean, "latency_ms_sample_sd": sample_sd,
                           "latency_ms_median": float(np.median(samples)), "latency_ms_p10": float(np.quantile(samples, .1)),
                           "latency_ms_p90": float(np.quantile(samples, .9)), "samples": count,
                           "peak_allocated_bytes_max": max(int(block.get("peak_allocated_bytes", 0)) for block in blocks),
                           "head_parameters": int(item["head_parameters"]),
                           "encoder_parameters": int(item["encoder_parameters"]),
                           "upstream_parameters": int(item["upstream_parameters"])})
    return sorted(result, key=lambda row: (GROUP_ORDER.index(row["group"]), row["mode"]))


def head_only_table(item: dict[str, Any], checkpoints: list[dict[str, str]], prediction_path: Path,
                    checkpoint_path: Path, inventory_path: Path) -> list[dict[str, Any]]:
    if item.get("schema") != "round8_incremental_head_cost_v1" or item.get("status") != "complete":
        raise ValueError("incremental head-cost artifact is incomplete")
    expected_root_files = {"protocol_sha256": ROUND_ROOT / "NUMERIC_PROTOCOL.json",
                           "source_sha256": ROUND_ROOT / "benchmark_head.py"}
    if any(not path.is_file() or item.get(field) != sha256(path) for field, path in expected_root_files.items()):
        raise ValueError("incremental head-cost protocol/source identity drift")
    if (item.get("accepted_manifest_sha256") != sha256(prediction_path)
            or item.get("checkpoint_index_sha256") != sha256(checkpoint_path)
            or item.get("inventory_sha256") != sha256(inventory_path)):
        raise ValueError("incremental head-cost accepted manifest/index/inventory binding drift")
    rows = item.get("rows", [])
    if {(row.get("group"), int(row.get("seed", -1))) for row in rows} != {(g, SEEDS[0]) for g in GROUP_ORDER}:
        raise ValueError("incremental head-cost grid drift")
    best = {(row["group"], int(row["seed"])): row for row in checkpoints if row["kind"] == "best"}
    result = []
    for row in rows:
        if row.get("head_frozen") is not True or row.get("checkpoint_sha256") != best[(row["group"], int(row["seed"]))]["sha256"]:
            raise ValueError("incremental head-cost checkpoint/freeze mismatch")
        samples = [float(value) for block in row["measurement"] for value in block["samples_ms"]]
        mean, sample_sd, count = summarize(samples)
        result.append({"group": row["group"], "seed": int(row["seed"]), "parameters": int(row["parameters"]),
                       "latency_ms_mean": mean, "latency_ms_sample_sd": sample_sd,
                       "latency_ms_median": float(np.median(samples)), "latency_ms_p10": float(np.quantile(samples, .1)),
                       "latency_ms_p90": float(np.quantile(samples, .9)), "samples": count,
                       "peak_allocated_bytes_max": max(int(block["peak_allocated_bytes"]) for block in row["measurement"])})
    return sorted(result, key=lambda row: GROUP_ORDER.index(row["group"]))


def plot_raw(table: list[dict[str, Any]], path: Path) -> None:
    plt = pyplot()
    horizons = (1, 3, 5)
    figure, axis = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(GROUP_ORDER)); width = 0.24
    lookup = {(row["group"], int(row["horizon"])): row for row in table}
    for offset, horizon in enumerate(horizons):
        means = [lookup[(group, horizon)]["average_precision_mean"] for group in GROUP_ORDER]
        errors = [lookup[(group, horizon)]["average_precision_sample_sd"] for group in GROUP_ORDER]
        axis.bar(x + (offset - 1) * width, means, width, yerr=errors, capsize=2, label=f"H{horizon}")
    axis.set_ylabel("Outer raw AP (mean ± sample SD)"); axis.set_ylim(0, 1)
    axis.set_xticks(x, [GROUP_LABEL[group] for group in GROUP_ORDER], rotation=28, ha="right")
    axis.grid(axis="y", alpha=0.25); axis.legend(); figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True); figure.savefig(path, dpi=180); plt.close(figure)


def plot_operational(table: list[dict[str, Any]], path: Path) -> None:
    plt = pyplot()
    figure, axis = plt.subplots(figsize=(7, 6))
    for row in table:
        if not (math.isfinite(row["trial_false_alarm_rate_mean"]) and math.isfinite(row["event_recall_mean"])):
            continue
        axis.errorbar(row["trial_false_alarm_rate_mean"], row["event_recall_mean"],
                      xerr=row["trial_false_alarm_rate_sample_sd"], yerr=row["event_recall_sample_sd"],
                      marker="o", capsize=3, label=GROUP_LABEL[row["group"]])
    axis.set(xlabel="Outer trial-any false alarm", ylabel="Outer new-start event recall", xlim=(0, 1), ylim=(0, 1),
             title="H3, calibration trial-FA 10% operating point")
    axis.grid(alpha=0.25); axis.legend(fontsize=7); figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True); figure.savefig(path, dpi=180); plt.close(figure)


def plot_state(table: list[dict[str, Any]], path: Path) -> None:
    plt = pyplot()
    selected = [row for row in table if row["group"] in {"P4_state", "fit_linear_state_baseline"}
                and row["method"] in {"state_transition", "persistence", "fit_only_ridge_linear"}]
    figure, axis = plt.subplots(figsize=(8, 5))
    for identity in sorted({(row["group"], row["method"]) for row in selected}):
        subset = sorted([row for row in selected if (row["group"], row["method"]) == identity], key=lambda row: int(row["step"]))
        axis.plot([int(row["step"]) for row in subset], [row["rmse_xyz_mean"] for row in subset], marker="o", label="/".join(identity))
    axis.set(xlabel="Future step", ylabel="XYZ RMSE", xticks=[1, 2, 3, 4, 5]); axis.grid(alpha=0.25); axis.legend(fontsize=7)
    figure.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); figure.savefig(path, dpi=180); plt.close(figure)


def fmt(value: Any, digits: int = 4) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "NA" if not math.isfinite(numeric) else f"{numeric:.{digits}f}"


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> list[str]:
    result = ["| " + " | ".join(label for _, label in columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        result.append("| " + " | ".join(fmt(row.get(key, "")) for key, _ in columns) + " |")
    return result


def comparison_sentence(rows: list[dict[str, Any]], comparison: str, horizon: int, operating_point: str, metric: str) -> str:
    matches = [row for row in rows if row["comparison"] == comparison and row["horizon"] == horizon
               and row["operating_point"] == operating_point and row["metric"] == metric]
    if not matches:
        return f"{comparison}：缺少预注册配对区间，不能作方向性结论。"
    row = matches[0]
    metric_label = {"average_precision": f"H{horizon} common-population raw AP", "brier": f"H{horizon} common-population raw Brier",
                    "event_recall": f"H{horizon} calibration trial-FA10%阈值外推outer事件召回",
                    "trial_false_alarm_rate": f"H{horizon} calibration trial-FA10%阈值外推outer trial FA",
                    "mean_lead_frames": f"H{horizon} calibration trial-FA10%阈值外推outer提前帧"}.get(metric, metric)
    if row["direction"] == "unavailable_incomplete_seed_grid":
        return (f"{row['meaning_zh']}（{metric_label}）：该预注册工作点无法在全部三个种子上校准，配对区间不可用"
                f"（候选不可用种子：{row['candidate_unavailable_seeds'] or '无'}；"
                f"对照不可用种子：{row['comparator_unavailable_seeds'] or '无'}）。这不是0效应或never-alarm证据。")
    result = f"{row['meaning_zh']}（{metric_label}）：差值{fmt(row['estimate'])}，95%配对区间[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}]。"
    if row["direction"] == "interval_crosses_zero":
        if horizon == 1 and metric == "mean_lead_frames" and all(number(row, key) == 0 for key in ("estimate", "ci_low", "ci_high")):
            return result + "H1检出后的提前量按固定窗口定义必然相同，差值为0；这不是泛化方向证据。"
        return result + "区间跨0，当前开发证据不足以确认稳定方向。"
    return result + ("候选值稳定更高；是否更好仍取决于该指标方向。" if row["direction"] == "candidate_higher" else "候选值稳定更低；是否更好仍取决于该指标方向。")


def evidence_label(rows: list[dict[str, Any]], comparison: str, horizon: int, operating_point: str,
                   metric: str, higher_is_better: bool) -> str:
    matches = [row for row in rows if row["comparison"] == comparison and row["horizon"] == horizon
               and row["operating_point"] == operating_point and row["metric"] == metric]
    if not matches or matches[0]["direction"] == "interval_crosses_zero":
        return "不支持（区间跨0或缺失）"
    if matches[0]["direction"] == "unavailable_incomplete_seed_grid":
        return "探索性（工作点未在全部种子可达，不能作配对方向结论）"
    favorable = matches[0]["direction"] == ("candidate_higher" if higher_is_better else "candidate_lower")
    return "支持" if favorable else "支持反向结论"


def comparison_block(rows: list[dict[str, Any]], comparison: str) -> list[str]:
    requests = (("raw_common_population", "average_precision"), ("raw_common_population", "brier"),
                ("trial_FA_0.10", "event_recall"), ("trial_FA_0.10", "trial_false_alarm_rate"),
                ("trial_FA_0.10", "mean_lead_frames"))
    return [comparison_sentence(rows, comparison, 3, point, metric) for point, metric in requests]


def state_ci_sentence(rows: list[dict[str, str]], comparison: str, step: int) -> str:
    matches = [row for row in rows if row["comparison"] == comparison and int(row["step"]) == step
               and row["metric"] == "rmse_xyz"]
    if not matches:
        return f"{comparison}/step{step}：缺失配对区间。"
    row = matches[0]; estimate, low, high = (number(row, key) for key in ("estimate", "ci_low", "ci_high"))
    if high < 0:
        verdict = "候选RMSE稳定更低"
    elif low > 0:
        verdict = "候选RMSE稳定更高"
    else:
        verdict = "区间跨0，方向未确认"
    return f"{comparison}/step{step}：ΔRMSE={fmt(estimate)}，95%配对区间[{fmt(low)}, {fmt(high)}]，{verdict}。"


def require_state_ci(rows: list[dict[str, str]]) -> None:
    expected = {(f"P4_state_vs_{comparator}", step, "rmse_xyz")
                for comparator in ("persistence", "fit_only_ridge_linear") for step in range(1, 6)}
    identities = [(row["comparison"], int(row["step"]), row["metric"]) for row in rows]
    if not expected.issubset(set(identities)) or any(identities.count(key) != 1 for key in expected):
        raise ValueError("P4 state paired-RMSE evidence grid is incomplete or duplicated")


def build_report(raw: list[dict[str, Any]], operational: list[dict[str, Any]], paired: list[dict[str, Any]],
                 calibration: list[dict[str, Any]], reliability: list[dict[str, Any]], baselines: list[dict[str, Any]],
                 state: list[dict[str, Any]], state_ci: list[dict[str, str]], interventions: list[dict[str, Any]],
                 consistency: list[dict[str, str]],
                 gate: list[dict[str, Any]], checkpoints: list[dict[str, str]], e2e: list[dict[str, Any]],
                 e2e_results: list[dict[str, Any]], head_only: list[dict[str, Any]],
                 state_diagnostics: list[dict[str, Any]], candidate_windows: list[dict[str, Any]], r7_summary: Path) -> str:
    h3_raw = [row for row in raw if int(row["horizon"]) == 3]
    lines = ["# 第八轮力动力学与事件时间预测综合报告", "",
             "本报告只汇总已验收产物。所有种子均保留，波动为三个种子的样本标准差（ddof=1）。代表性失败案例固定使用H3和seed 20260914，不依据结果挑选种子。", "",
             "## 核心结果", ""]
    lines += markdown_table(h3_raw, [("group", "组"), ("average_precision_mean", "H3 raw AP均值"),
                                     ("average_precision_sample_sd", "样本SD"), ("brier_mean", "Brier均值"),
                                     ("prevalence_mean", "正类自然占比"),
                                     ("balanced_accuracy_mean", "BA@0.5"), ("macro_f1_mean", "Macro-F1@0.5")])
    lines += ["", "H3在calibration的trial-FA 10%约束下原样应用到outer：", ""]
    lines += markdown_table(operational, [("group", "组"), ("trial_false_alarm_rate_mean", "实际trial FA"),
                                          ("event_recall_mean", "事件召回"), ("mean_lead_frames_mean", "检出事件平均提前帧"),
                                          ("lead_ge_3_recall_mean", "提前≥3帧召回"),
                                          ("balanced_accuracy_mean", "BA"), ("macro_f1_mean", "Macro-F1"),
                                          ("tn_mean", "TN"), ("fp_mean", "FP"), ("fn_mean", "FN"), ("tp_mean", "TP"),
                                          ("threshold_status", "三种子可达性"), ("unavailable_seeds", "不可达种子"),
                                          ("never_alarm_seeds", "never-alarm seeds")])
    lines += ["", "工作点若未在全部三个种子上可达，表中对应均值为NA，配对区间标为不可用；不可达不按0效应、区间跨0或never-alarm解释。"]
    unavailable_ci = sum(row["direction"] == "unavailable_incomplete_seed_grid" for row in paired)
    lines += ["", f"预注册配对网格中共有{len(paired)}项，其中{unavailable_ci}项因工作点未在三个种子全部可达而标为不可用；其估计与区间保持NA。", "",
              "H3简单risk基线（fit-only拟合，相同固定工作点）：", ""]
    lines += markdown_table(baselines, [("group", "基线"), ("average_precision", "AP"), ("brier", "Brier"),
                                        ("trial_false_alarm_rate", "trial FA"), ("event_recall", "事件召回"),
                                        ("mean_lead_frames", "提前帧"), ("never_alarm", "never alarm")])
    lines += ["", "## 六个预注册问题", "", "### Q1 显式力差分是否在等历史条件下有效？", ""]
    lines += ["历史R7 A/B/C/D参照如下；其B使用Fn/Ft/Ft比值，C再加有符号XYZ差分，与R8的完整有符号XYZ力历史布局不同，不混入R8主配对区间。", ""]
    lines += markdown_table(list(R7_REFERENCE), [("group", "R7组"), ("force_layout", "力/变化布局"),
                                                   ("H1_AP", "H1 AP"), ("H3_AP", "H3 AP"), ("H5_AP", "H5 AP"),
                                                   ("H3_trial_FA", "H3 trial FA"),
                                                   ("H3_event_recall", "H3事件召回"),
                                                   ("H3_mean_lead", "H3提前帧")])
    lines += [""]
    lines += [f"- {sentence}" for sentence in comparison_block(paired, "C_xyz_delta_vs_B_xyz")]
    lines += ["", "H1/H5跨窗口复核：", ""]
    for horizon in (1, 5):
        for point, metric in (("raw_common_population", "average_precision"), ("raw_common_population", "brier"),
                              ("trial_FA_0.10", "event_recall"), ("trial_FA_0.10", "trial_false_alarm_rate"),
                              ("trial_FA_0.10", "mean_lead_frames")):
            lines.append("- " + comparison_sentence(paired, "C_xyz_delta_vs_B_xyz", horizon, point, metric))
    lines += ["", "结论：显式lag5力差分在等原始图像历史下显著提高H3排序与事件召回并降低Brier；trial FA差异区间跨0，提前帧增加得到配对区间支持。排序改善与固定工作点告警收益分别报告。", "",
              "### Q2 事件时间头是否改善一致性、校准和告警？", ""]
    lines += [f"- {sentence}" for sentence in comparison_block(paired, "C_hazard_vs_C_xyz_delta")]
    h3_cal = [row for row in calibration if int(row["horizon"]) == 3 and row["group"] in {"C_xyz_delta", "C_hazard"}]
    h3_rel = [row for row in reliability if int(row["horizon"]) == 3 and row["role"] == "outer"
              and row["group"] in {"C_xyz_delta", "C_hazard"}]
    lines += ["", "H3 raw/shared正斜率Platt结果：", ""]
    lines += markdown_table(h3_cal, [("group", "组"), ("method_type", "风险"), ("brier_mean", "Brier"),
                                     ("brier_sample_sd", "样本SD"), ("average_precision_mean", "AP")])
    lines += ["", "H3 shared-Platt outer可靠性：", ""]
    lines += markdown_table(h3_rel, [("group", "组"), ("ece_mean", "ECE"), ("ece_sample_sd", "样本SD")])
    lines += ["", "结论：hazard保证累计风险单调，显著降低raw Brier并提高固定工作点事件召回；raw AP、trial FA和提前帧没有稳定差异。shared-Platt后两组Brier/ECE接近，因此不能声称校准全面更优。"]
    lines += ["", "### Q3 时序力条件融合是否超过容量对照？", ""]
    for comparison in ("P3_fusion_independent_vs_P3_concat_independent",
                       "P3_fusion_independent_vs_C_xyz_delta",
                       "P3_fusion_hazard_vs_P3_fusion_independent",
                       "P3_fusion_hazard_vs_C_hazard"):
        lines += [f"- {sentence}" for sentence in comparison_block(paired, comparison)]
    lines += ["", "P3 fusion与concat投影参数只差34，但结构归纳偏置不同；与C_xyz_delta的对照还同时变了前端投影容量，只能作补充对照。", "",
              "容量近似匹配的fusion-vs-concat中，raw AP区间跨0，但事件召回获得支持；fusion相对普通GRU的raw AP/Brier和事件召回均获得支持。因两种参照给出不同强度证据，论文主张应限定为事件级告警收益，不能声称门控融合全面优于容量对照。", "",
              "### Q4 未来状态辅助监督是否提供额外信息？", ""]
    lines += [f"- {sentence}" for sentence in comparison_block(paired, "P4_state_vs_P4_direct")]
    state_focus = [row for row in state if row["group"] in {"P4_state", "fit_linear_state_baseline"}
                   and row["method"] in {"state_transition", "persistence", "fit_only_ridge_linear"}]
    lines += ["", "逐步状态预测（RMSE、方差比均保留全种子）：", ""]
    lines += markdown_table(state_focus, [("group", "组"), ("method", "方法"), ("step", "步"),
                                          ("rmse_xyz_mean", "RMSE"), ("rmse_xyz_sample_sd", "样本SD"),
                                          ("variance_ratio_mean", "预测/目标方差比")])
    lines += [""]
    for comparator in ("persistence", "fit_only_ridge_linear"):
        for step in range(1, 6):
            lines.append("- " + state_ci_sentence(state_ci, f"P4_state_vs_{comparator}", step))
    state_zero = [row for row in interventions if row["intervention"] == "state_input_fit_mean_zero"]
    if state_zero:
        lines += ["", "P4_state固定state-zero干预（不重校准）：", ""]
        lines += markdown_table(state_zero, [("group", "组"), ("event_recall_mean", "事件召回"),
                                             ("mean_score_delta_vs_unperturbed_mean", "平均分数变化"),
                                             ("mean_lead_frames_mean", "提前帧")])
    lines += ["", "P4风险头可将线性state支路代数合并到hidden路径，因此P4_state主要检验辅助监督的归纳偏置；必须结合state-zero与简单状态基线，不能仅用MSE声称世界模型成立。", "",
              "P4_direct的15维内部量没有接受状态监督，只用于容量匹配；它的MSE或方差不作状态预测质量、数值错误或塌缩证据。", "",
              "P4_state逐试次时序诊断（各试次先计算连续方差/误差，再对试次和三种子汇总）：", ""]
    lines += markdown_table(state_diagnostics, [("population", "人口"), ("step", "步"),
                                                  ("prediction_target_mse_xyz_trial_mean_mean", "P4 MSE"),
                                                  ("target_vs_current_mse_xyz_trial_mean_mean", "保持MSE"),
                                                  ("linear_target_mse_xyz_trial_mean_mean", "fit-linear MSE"),
                                                  ("prediction_temporal_variance_xyz_trial_mean_mean", "P4试次内方差"),
                                                  ("persistence_temporal_variance_xyz_trial_mean_mean", "当前力试次内方差"),
                                                  ("target_temporal_variance_xyz_trial_mean_mean", "目标试次内方差")])
    primary_state = [row for row in state_diagnostics if row["population"] == "primary_common"]
    worse_persistence = sum(row["prediction_target_mse_xyz_trial_mean_mean"] > row["target_vs_current_mse_xyz_trial_mean_mean"] for row in primary_state)
    worse_linear = sum(row["prediction_target_mse_xyz_trial_mean_mean"] > row["linear_target_mse_xyz_trial_mean_mean"] for row in primary_state)
    persistence_rmse_worse = sum(number(row, "ci_low") > 0 for row in state_ci
                                 if row["comparison"] == "P4_state_vs_persistence" and row["metric"] == "rmse_xyz")
    p4_operation = next(row for row in operational if row["group"] == "P4_state")
    state_zero_row = next((row for row in interventions if row["intervention"] == "state_input_fit_mean_zero"), None)
    zero_cost = (p4_operation["event_recall_mean"] - state_zero_row["event_recall_mean"]) if state_zero_row else math.nan
    lines += ["", "primary_common对应主要future风险比较的共同稳定端点；full_timeline包含所有state mask有效行，可能包含当前已滑移帧，因此后者不作为稳定阶段预测质量。方差仅作连续数值诊断，不使用事后低方差阈值。",
              f"结论：正式全轨迹表中，P4_state有{persistence_rmse_worse}/5个窗口的RMSE显著差于保持基线；在primary_common上，其逐试次MSE也有{worse_persistence}/5个窗口高于保持基线、{worse_linear}/5个窗口高于fit-only线性基线。因此不支持其状态预测优于简单动力学基线或构成有效状态世界模型。它相对P4_direct的H3 raw AP虽有小幅支持，但H3固定低误报工作点的事件召回、trial FA与提前量区间均跨0；H1事件召回还出现稳定下降：{comparison_sentence(paired, 'P4_state_vs_P4_direct', 1, 'trial_FA_0.10', 'event_recall')} state-zero后H3事件召回从{fmt(p4_operation['event_recall_mean'])}降至{fmt(state_zero_row['event_recall_mean'] if state_zero_row else math.nan)}（约-{fmt(zero_cost)}），只构成弱敏感性证据，hidden bypass与线性可合并限制仍在。", "",
              "### Q5 当前证据支持哪些论文主张？", ""]
    claims = (("C_xyz_delta_vs_B_xyz", "力差分改善raw AP", "raw_common_population", "average_precision", True),
              ("C_hazard_vs_C_xyz_delta", "hazard改善raw Brier", "raw_common_population", "brier", False),
              ("P3_fusion_independent_vs_P3_concat_independent", "时序融合相对容量匹配concat改善raw AP", "raw_common_population", "average_precision", True),
              ("P4_state_vs_P4_direct", "状态辅助改善raw排序AP（仅排序，不代表状态预测或告警改善）", "raw_common_population", "average_precision", True))
    for comparison, claim, point, metric, hib in claims:
        lines.append(f"- {claim}：{evidence_label(paired, comparison, 3, point, metric, hib)}。")
    lines += ["- 时序融合相对普通GRU的raw AP获得支持，但该对照同时改变前端投影容量，只作为补充证据。",
              "- 所有R8结论仍是Sparsh原域重复开发outer上的开发证据，不升格为独立盲测、实物成功率或已验证世界模型。", "",
              "### Q6 哪些缺口仍需新数据？", "",
              "- R7当前HTT slip引用的calibration FPR1%工作点迁移到validation后，V的实际FPR/召回为15.54%/75.60%，F-adapt为11.82%/65.45%；误报下降同时召回下降，且F-adapt相对V的24个折×目标对照仅3个95%区间整体支持正收益。R8的原域future收益不能外推为HTT当前力融合稳健改善。",
              "- R7/R8都没有新的无标记GSmini实物域current-slip开发证据，无法回答域迁移后误报和召回。",
              "- Sparsh原域future的首次slip起点仍是数据集标签语义，缺少独立物理滑移起点和动作/控制输入；因此提前帧数只是被动观察下的数据集指标。",
              "- P4只预测冻结上游的力表征，缺少真实力轨迹与外部干预验证；ToucHD或新的独立事件数据才能检查可传递性。", ""]
    lines += ["P2的C_hazard同时改变了输出参数化和fit目标：从三个带pos-weight的独立BCE改为五步无权删失hazard NLL。因此与C_xyz_delta的差异不能归因成单一网络层收益。", "",
              "R7的B/C对照同时混入了力方向与差分信息；R8的B_xyz/C_xyz_delta在同一完整有符号XYZ历史上增加确定性的lag5变换，才是差分归纳偏置的主要归因对照。", ""]
    consistency_summary = consistency_table(consistency)
    q2_consistency = [row for row in consistency_summary if row["group"] in {"C_xyz_delta", "C_hazard"}]
    lines += ["", "多窗口累计风险单调性（outer违例比例）：", ""]
    lines += markdown_table(q2_consistency, [("group", "组"), ("method", "风险"),
                                              ("violation_fraction_mean", "违例比例"),
                                              ("violation_fraction_sample_sd", "样本SD")])
    hazard = [row for row in consistency if row.get("head_type") == "discrete_hazard" and row.get("role") == "outer" and row.get("method") == "raw_risk"]
    violations = sum(int(row["violations"]) for row in hazard) if hazard else 0
    total = sum(int(row["rows"]) for row in hazard) if hazard else 0
    lines += [f"事件时间头在outer原始累计风险中的单调违例为{violations}/{total}。这项结构保证只说明概率顺序一致，不能单独证明召回、提前量或校准改善。", ""]
    if state:
        p4s = [row for row in state if row["group"] == "P4_state"]
        lines += ["P4状态预测结果已与保持不变和fit-only线性基线分开列出。由于风险头对hidden和线性state的组合可代数合并，本实验主要检验状态辅助监督的归纳偏置；必须结合固定state-ablation，不能仅凭较低MSE宣称世界模型成立。", ""]
    if interventions:
        kinds = "、".join(sorted({row["intervention"] for row in interventions}))
        lines += [f"固定干预（{kinds}）使用原模型的规则与calibration阈值，不重新校准。其中旧乘法门控p_future×p_current直接回答门控是否压制提前告警；告警减少只有在事件召回和提前量代价同时报告时才可解释。", ""]
    if gate:
        lines += ["旧乘法门控在H3固定trial-FA 10%工作点的变化（门控减去原始模型）：", ""]
        lines += markdown_table(gate, [("group", "组"), ("delta_raw_average_precision_mean", "Δraw AP"),
                                       ("delta_trial_false_alarm_rate_mean", "Δtrial FA"),
                                       ("delta_event_recall_mean", "Δ事件召回"),
                                       ("delta_mean_lead_frames_mean", "Δ提前帧")])
        recall_losses = [-row["delta_event_recall_mean"] for row in gate]
        lead_losses = [-row["delta_mean_lead_frames_mean"] for row in gate]
        lines += ["", f"旧门控几乎消除了trial FA，但事件召回损失为{fmt(min(recall_losses),3)}–{fmt(max(recall_losses),3)}，平均提前量损失为{fmt(min(lead_losses),3)}–{fmt(max(lead_losses),3)}帧；这直接支持其压制提前预警，而不是改善future预测。", ""]
    lines += ["## 下一阶段建议（本轮不启动）", "",
              "P3_fusion_hazard是当前最值得带入独立验证的候选，但应继续保留P3_concat_independent与P3_fusion_independent作为容量和目标参数化对照。三个窗口的固定低误报工作点如下：", ""]
    lines += markdown_table(candidate_windows, [("horizon", "窗口"), ("average_precision_mean", "raw AP"),
                                                  ("trial_false_alarm_rate_mean", "实际trial FA"),
                                                  ("event_recall_mean", "事件召回"), ("mean_lead_frames_mean", "平均提前帧"),
                                                  ("lead_ge_5_recall_mean", "提前≥5帧召回")])
    lines += ["", "H5虽达到较高事件召回和约2.77帧平均提前量，但提前至少5帧的召回仅约5.23%，不能把H5窗口长度表述为5帧有效提前量。H1只反映短窗口检出，提前量按定义为1帧。",
              "P4_state暂不作为论文主贡献：状态预测未超过保持与fit-only线性基线，风险排序的小幅收益也没有转化成稳定低误报告警收益。",
              "下一阶段需要独立事件起点与新的GSmini域评价来验证误报、召回和提前量；在此之前保留‘Sparsh原域开发证据’措辞。完整系统约1.01亿上游参数，轻量化主张只限定新增future头。", ""]
    lines += ["## 证据边界", "",
              "- outer是重复使用的开发评价角色，不是独立盲测；相邻帧和三个种子没有被当作独立试次。",
              "- 首次滑移是数据集标签起点，不是独立物理滑移真值；H5标签窗口不等于稳定提前5帧。",
              "- P4目标是冻结上游预测力XYZ表征，不是真实力真值；无动作输入时只能描述被动接触状态预测。",
              f"- 本轮没有新增HTT当前slip融合训练。当前检测的力融合结论继续受第七轮所述限制约束，证据来源：`{r7_summary}`。",
              "- 公开数据上的结果不能换算成无标记GSmini实物成功率。", "",
              "## 代表性案例与图表", "",
              "- 代表性案例固定为H3、seed 20260914；筛选结果见`H3_SEED20260914_REPRESENTATIVE_CASES.csv`，原始全集及图片在评价目录。",
              "- `figures/raw_ap_by_group.png`：所有组的H1/H3/H5 raw AP与样本SD。",
              "- `figures/H3_operational_tradeoff.png`：H3实际试次误报与事件召回。",
              "- `figures/state_prediction_rmse.png`：未来状态预测与简单基线。", "",
              "阈值从calibration迁移到outer的实际FPR/召回/trial-FA/提前量见`THRESHOLD_TRANSFER.csv`；never-alarm种子已在H3运行表中单列，不作检测改善。", "",
              "## 复现与checkpoint", "",
              f"正式科学网格严格为24项配置；共索引{len(checkpoints)}项best/latest/config产物，完整绝对路径见`CHECKPOINT_PATHS.csv`。性能修复前保留的未验收attempt是恢复证据，不计作新增科学配置。"]
    if e2e:
        lines += ["", "## 端到端部署开销", "",
                  "下表固定使用seed 20260914。cold_history计算九次base输出；streaming计算一次新base输出（读取当前图像与驻留的lag-5图像）并复用八个历史base输出。均不包含传感器采集、数据集I/O和权重加载。", ""]
        lines += markdown_table(e2e_results, [("group", "组"), ("mode", "模式"),
                                              ("latency_ms_median", "延迟中位数ms"),
                                              ("latency_ms_p10", "P10"), ("latency_ms_p90", "P90"),
                                              ("head_parameters", "预测头参数")])
        lines += ["", "共享GPU条件、逐次样本、峰值显存及cache/fresh parity保留在原JSON和`E2E_SUMMARY.csv`。"]
        lines += ["", "增量future头成本（已缓存归一化特征）：", ""]
        lines += markdown_table(head_only, [("group", "组"), ("parameters", "future头参数"),
                                             ("latency_ms_median", "头延迟中位数ms"),
                                             ("latency_ms_p90", "P90"),
                                             ("peak_allocated_bytes_max", "resident total peak bytes")])
        lines += ["", "该头部测量排除图像处理、编码器、force/current-slip和特征组装，不能当作部署速度。完整上游约1.01亿参数，因此‘轻量’只修饰新增future头，不能修饰完整系统。"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-audit", type=Path, required=True)
    parser.add_argument("--checkpoint-index", type=Path, required=True)
    parser.add_argument("--formal-inventory", type=Path, required=True)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--e2e", type=Path, nargs="+", required=True)
    parser.add_argument("--head-only", type=Path, required=True)
    parser.add_argument("--state-diagnostics", type=Path, required=True)
    parser.add_argument("--r7-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = read_json(PROTOCOL)
    training = read_json(args.training_audit)
    if training.get("status") != "pass" or int(training.get("run_count", -1)) != 24:
        raise ValueError("24-run training audit has not passed")
    if training.get("inventory_sha256") != sha256(args.formal_inventory):
        raise ValueError("training audit/formal inventory identity mismatch")
    paths = require_evaluation(args.evaluation_dir)
    eval_summary = read_json(paths["summary"])
    if int(eval_summary.get("runs", -1)) != 24:
        raise ValueError("evaluation does not contain 24 runs")
    checkpoints = read_csv(args.checkpoint_index)
    require_checkpoint_grid(checkpoints)
    prediction_manifest, _ = require_provenance(args.prediction_manifest, args.evaluation_manifest,
                                                args.training_audit, args.checkpoint_index, eval_summary)
    if not args.r7_summary.is_file():
        raise FileNotFoundError(args.r7_summary)
    metrics = read_csv(paths["metrics"])
    aggregate = read_csv(paths["aggregate"])
    aggregate_coverage = {(row.get("group"), int(row.get("horizon", -1))) for row in aggregate}
    expected_coverage = {(group, horizon) for group in GROUP_ORDER for horizon in HORIZONS}
    if not expected_coverage.issubset(aggregate_coverage):
        raise ValueError("evaluator all-seed summary lacks a full group/horizon grid")
    raw = raw_table(metrics)
    operational_all = [row for horizon in HORIZONS for row in operational_table(metrics, horizon, protocol["operational_point"])]
    operational = [row for row in operational_all if int(row["horizon"]) == protocol["representative_horizon"]]
    raw_lookup = {(row["group"], int(row["horizon"])): row for row in raw}
    candidate_windows = []
    for row in operational_all:
        if row["group"] == "P3_fusion_hazard":
            candidate_windows.append({**row, "average_precision_mean": raw_lookup[(row["group"], int(row["horizon"]))]["average_precision_mean"]})
    calibration = calibration_table(metrics)
    reliability = reliability_table(read_csv(paths["reliability"]))
    baselines = baseline_table(metrics, protocol["representative_horizon"], protocol["operational_point"])
    if len(raw) != 24 or len(operational) != 8:
        raise ValueError("group/horizon result grid is incomplete")
    paired = paired_table(read_csv(paths["paired"]))
    state = state_table(read_csv(paths["state"]))
    state_ci = read_csv(paths["state_paired"])
    require_state_ci(state_ci)
    interventions = intervention_table(read_csv(paths["interventions"]), protocol["representative_horizon"])
    gate = gate_table(read_csv(paths["gate"]), protocol["representative_horizon"], protocol["operational_point"])
    consistency = read_csv(paths["consistency"])
    consistency_summary = consistency_table(consistency)
    failures = [row for row in read_csv(paths["failures"])
                if int(row.get("horizon", -1)) == protocol["representative_horizon"]
                and int(row.get("seed", -1)) == protocol["representative_seed"]]
    if not failures:
        raise ValueError("fixed H3/seed20260914 representative cases are absent")
    e2e = [read_json(path) for path in args.e2e]
    require_e2e_identity(e2e, args.prediction_manifest, checkpoints, prediction_manifest["prepared"]["sha256"],
                         prediction_manifest["checkpoint_index"]["sha256"])
    e2e_results = e2e_table(e2e)
    head_only_source = read_json(args.head_only)
    head_only = head_only_table(head_only_source, checkpoints, args.prediction_manifest,
                                args.checkpoint_index, args.formal_inventory)
    if head_only_source.get("prepared_sha256") != prediction_manifest["prepared"]["sha256"]:
        raise ValueError("incremental head-cost prepared identity drift")
    state_diagnostics = require_state_diagnostics(args.state_diagnostics, args.prediction_manifest, paths["summary"])
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_csv(args.output / "RAW_RISK_ALL_GROUPS.csv", raw)
    atomic_csv(args.output / "R7_ABCD_HISTORICAL_REFERENCE.csv", list(R7_REFERENCE))
    atomic_csv(args.output / "H3_OPERATIONAL_ALL_GROUPS.csv", operational)
    atomic_csv(args.output / "OPERATIONAL_ALL_HORIZONS.csv", operational_all)
    atomic_csv(args.output / "CALIBRATION_ALL_GROUPS.csv", calibration)
    atomic_csv(args.output / "RELIABILITY_ECE_ALL_GROUPS.csv", reliability)
    atomic_csv(args.output / "MULTIHORIZON_CONSISTENCY_SUMMARY.csv", consistency_summary)
    atomic_csv(args.output / "H3_SIMPLE_RISK_BASELINES.csv", baselines)
    atomic_csv(args.output / "THRESHOLD_TRANSFER.csv", read_csv(paths["threshold_transfer"]))
    atomic_csv(args.output / "PAIRED_KEY_COMPARISONS.csv", paired)
    atomic_csv(args.output / "STATE_PREDICTION_SUMMARY.csv", state)
    atomic_csv(args.output / "STATE_PAIRED_BOOTSTRAP_CI.csv", state_ci)
    atomic_csv(args.output / "H3_INTERVENTION_SUMMARY.csv", interventions)
    atomic_csv(args.output / "H3_FIXED_MULTIPLICATIVE_GATE.csv", gate)
    atomic_csv(args.output / "H3_SEED20260914_REPRESENTATIVE_CASES.csv", failures)
    atomic_csv(args.output / "CHECKPOINT_PATHS.csv", checkpoints)
    atomic_csv(args.output / "E2E_SUMMARY.csv", e2e_results)
    atomic_csv(args.output / "HEAD_ONLY_COST.csv", head_only)
    atomic_csv(args.output / "P4_STATE_TEMPORAL_DIAGNOSTICS.csv", state_diagnostics)
    plot_raw(raw, args.output / "figures" / "raw_ap_by_group.png")
    plot_operational(operational, args.output / "figures" / "H3_operational_tradeoff.png")
    plot_state(state, args.output / "figures" / "state_prediction_rmse.png")
    report = build_report(raw, operational, paired, calibration, reliability, baselines, state, state_ci, interventions,
                          consistency, gate, checkpoints, e2e, e2e_results, head_only, state_diagnostics, candidate_windows,
                          args.r7_summary.resolve())
    atomic_text(args.output / "SUMMARY_ZH.md", report)
    sources = [args.training_audit, args.checkpoint_index, args.formal_inventory,
               args.prediction_manifest, args.evaluation_manifest,
               args.r7_summary, args.head_only, args.state_diagnostics / "STATE_DIAGNOSTIC_AUDIT.json",
               args.state_diagnostics / "P4_STATE_SEED_STEP_SUMMARY.csv",
               args.state_diagnostics / "P4_STATE_PER_TRIAL_TEMPORAL_DIAGNOSTICS.csv",
               *reporting_source_paths(), *paths.values(), *args.e2e]
    manifest = {str(path.resolve()): sha256(path) for path in sources}
    outputs = [path for path in args.output.rglob("*") if path.is_file() and path.name != "REPORTING_AUDIT.json"]
    result = {"schema": "round8_reporting_summary_v1", "status": "complete", "representative_horizon": 3,
              "representative_seed": 20260914, "seed_dispersion": "sample_sd_ddof1", "runs": 24,
              "groups": list(GROUP_ORDER), "source_hashes": manifest,
              "output_hashes": {str(path.resolve()): sha256(path) for path in outputs}, "test_role_consumed": False}
    atomic_json(args.output / "REPORTING_AUDIT.json", result)
    print(json.dumps({"status": "complete", "outputs": len(outputs) + 1, "runs": 24}, ensure_ascii=False))


if __name__ == "__main__":
    main()
