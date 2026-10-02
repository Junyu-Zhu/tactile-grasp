#!/usr/bin/env python3
"""Full Round-8 evaluation using the frozen Round-7 event implementation."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
R7_EVALUATOR = HERE.parent.parent / "round7_temporal_fairness_event_warning" / "evaluation" / "evaluate_r7.py"

try:
    from .core import (average_precision, apply_shared_platt, fit_shared_monotone_platt,
                       fit_state_linear_baseline, paired_cluster_bootstrap, predict_state_linear_baseline)
    from .evaluate import sha256, validate_manifest
    from .schema import load_timeline_csv, validate_cross_run_identity
    from .fast_thresholds import compact_event_curve as fast_compact_event_curve
except ImportError:
    from core import (average_precision, apply_shared_platt, fit_shared_monotone_platt,
                      fit_state_linear_baseline, paired_cluster_bootstrap, predict_state_linear_baseline)
    from evaluate import sha256, validate_manifest
    from schema import load_timeline_csv, validate_cross_run_identity
    from fast_thresholds import compact_event_curve as fast_compact_event_curve


def _load_r7():
    spec = importlib.util.spec_from_file_location("round7_frozen_evaluator", R7_EVALUATOR)
    if spec is None or spec.loader is None:
        raise ImportError(R7_EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.compact_event_curve = fast_compact_event_curve
    return module


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    if not fields:
        raise ValueError(f"empty output {path}")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader(); writer.writerows(rows); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def horizon_rows(r7, rows: list[dict[str, Any]], horizon: int, population: str) -> list[dict[str, Any]]:
    return r7.role_rows(rows, horizon, population)


def append_state_errors(frame_rows: list[dict[str, Any]], trial_rows: list[dict[str, Any]], *,
                        group: str, seed: int | str, step: int, method: str,
                        cohort: list[dict[str, Any]], prediction: np.ndarray, target: np.ndarray) -> None:
    errors = np.mean((prediction - target) ** 2, axis=1)
    records = []
    for row, squared_error in zip(cohort, errors):
        record = {"group": group, "seed": seed, "step": step, "method": method,
                  "episode_id": row["episode_id"], "leakage_group": row["leakage_group"],
                  "t": int(row["t"]), "squared_error_xyz": float(squared_error)}
        frame_rows.append(record); records.append(record)
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_episode[record["episode_id"]].append(record)
    for episode, episode_rows in sorted(by_episode.items()):
        mse = float(np.mean([row["squared_error_xyz"] for row in episode_rows]))
        trial_rows.append({"group": group, "seed": seed, "step": step, "method": method,
                           "episode_id": episode, "leakage_group": episode_rows[0]["leakage_group"],
                           "n": len(episode_rows), "mse_xyz": mse, "rmse_xyz": math.sqrt(mse)})


def load_runs(manifest_path: Path) -> tuple[dict, dict, dict[str, str]]:
    audit = validate_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    horizons = [int(value) for value in manifest.get("horizons", [1, 3, 5])]
    runs: dict[tuple[str, int], dict[str, Any]] = {}
    reference: dict[str, list[dict[str, Any]]] = {}
    state_reference: dict[str, list[dict[str, Any]]] = {}
    for artifact in manifest["artifacts"]:
        key = (artifact["group"], int(artifact["seed"]))
        rows = load_timeline_csv(Path(artifact["path"]), horizons, head_type=artifact["head_type"])
        if artifact["role"] in reference:
            validate_cross_run_identity(reference[artifact["role"]], rows, horizons)
        else:
            reference[artifact["role"]] = rows
        if any("state_steps" in row for row in rows):
            if artifact["role"] in state_reference:
                validate_cross_run_identity(state_reference[artifact["role"]], rows, horizons)
            else:
                state_reference[artifact["role"]] = rows
        run = runs.setdefault(key, {"roles": {}, "head_type": artifact["head_type"]})
        if run["head_type"] != artifact["head_type"]:
            raise ValueError(f"head type drift {key}")
        run["roles"][artifact["role"]] = rows
    return manifest, runs, audit["sources"]


def validate_intervention_grid(intervention: dict[str, Any], all_runs: dict[tuple[str, int], dict[str, Any]]) -> None:
    """Require the complete preregistered intervention grid before evaluation."""
    expected = {
        (group, seed, kind)
        for group, seed in all_runs
        for kind in (("force_input_fit_mean_zero", "force_causal_lag1", "state_input_fit_mean_zero")
                     if group == "P4_state" else ("force_input_fit_mean_zero", "force_causal_lag1"))
    }
    seen: set[tuple[str, int, str]] = set()
    for artifact in intervention.get("artifacts", []):
        key = (artifact.get("group"), int(artifact.get("seed")), artifact.get("intervention"))
        if key in seen:
            raise ValueError(f"duplicate intervention artifact {key}")
        seen.add(key)
        run_key = (key[0], key[1])
        if artifact.get("role") != "outer" or run_key not in all_runs:
            raise ValueError("intervention run/role drift")
        if artifact.get("head_type") != all_runs[run_key]["head_type"]:
            raise ValueError(f"intervention head type drift {key}")
        if key[2] == "state_input_fit_mean_zero" and key[0] != "P4_state":
            raise ValueError(f"state intervention assigned to non-state group {key}")
    if seen != expected:
        raise ValueError(f"intervention grid drift missing={sorted(expected-seen)} extra={sorted(seen-expected)}")


def evaluate(manifest_path: Path, output: Path) -> dict[str, Any]:
    r7 = _load_r7()
    manifest, all_runs, sources = load_runs(manifest_path)
    for path in (HERE / "core.py", HERE / "schema.py", HERE / "fast_thresholds.py", HERE / "THRESHOLD_OPTIMIZATION_AUDIT.json",
                 HERE / "evaluate.py", HERE / "evaluate_formal.py",
                 HERE / "protocol.json", HERE / "PROTOCOL.md"):
        sources[str(path.resolve())] = sha256(path)
    horizons = [int(value) for value in manifest.get("horizons", [1, 3, 5])]
    groups = sorted({group for group, _ in all_runs})
    seeds = sorted({seed for _, seed in all_runs})
    r7.GROUPS = tuple(groups); r7.SEEDS = tuple(seeds)
    for run in all_runs.values():
        for rows in run["roles"].values():
            for horizon in horizons:
                r7.add_rules(rows, f"p_H{horizon}", f"H{horizon}_")

    metrics: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    transfers: list[dict[str, Any]] = []
    tradeoff: list[dict[str, Any]] = []
    reliability: list[dict[str, Any]] = []
    consistency: list[dict[str, Any]] = []
    trials_by_file: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    selected_rules: dict[tuple[str, int, int], str] = {}
    selected_thresholds: dict[tuple[str, int, int, str], float | None] = {}
    shared_platt_models: list[dict[str, Any]] = []
    baseline_models: list[dict[str, Any]] = []
    state_metrics: list[dict[str, Any]] = []
    state_error_rows: list[dict[str, Any]] = []
    state_trial_metrics: list[dict[str, Any]] = []
    state_linear_model: dict[str, Any] | None = None
    intervention_metrics: list[dict[str, Any]] = []
    fixed_gate_metrics: list[dict[str, Any]] = []

    for (group, seed), run in sorted(all_runs.items()):
        for horizon in horizons:
            for population in ("primary", "per_h_max"):
                role_sets = {role: horizon_rows(r7, rows, horizon, population) for role, rows in run["roles"].items()}
                r7._CAL_CACHE.clear()
                selected, evidence = r7.select_rule(role_sets["selection"], f"H{horizon}_", "metric_mask")
                if population == "primary":
                    selected_rules[(group, seed, horizon)] = selected
                for row in evidence:
                    selections.append({"group": group, "seed": seed, "horizon": horizon, "population": population,
                                       "selected": row["rule"] == selected, **row})
                for rule in r7.RULES:
                    score_key = f"H{horizon}_{rule}"
                    for operating_point, kind, target in r7.OPS:
                        threshold, status, _ = r7.select_threshold(role_sets["calibration"], score_key, "metric_mask", kind, target)
                        if population == "primary" and rule == selected:
                            selected_thresholds[(group, seed, horizon, operating_point)] = threshold
                        base = {"method_type": "neural_operational", "group": group, "seed": seed,
                                "head_type": run["head_type"], "horizon": horizon, "population": population,
                                "rule": rule, "rule_selected": rule == selected, "operating_point": operating_point,
                                "threshold_status": status, "threshold": threshold}
                        if threshold is None:
                            metrics.append(base)
                            continue
                        frame = r7.frame_confusion(role_sets["outer"], score_key, threshold, "metric_mask")
                        event, records = r7.event_metrics(role_sets["outer"], score_key, threshold, "metric_mask")
                        metrics.append({**base, **frame, **event})
                        if rule == selected:
                            if population == "primary":
                                cal_frame = r7.frame_confusion(role_sets["calibration"], score_key, threshold, "metric_mask")
                                cal_event = r7.event_metrics(role_sets["calibration"], score_key, threshold, "metric_mask")[0]
                                transfers.append({"group": group, "seed": seed, "horizon": horizon, "rule": rule,
                                                  "operating_point": operating_point, "threshold": threshold,
                                                  "calibration_frame_fpr": cal_frame["frame_fpr"], "outer_frame_fpr": frame["frame_fpr"],
                                                  "calibration_frame_recall": cal_frame["frame_recall"],
                                                  "outer_frame_recall": frame["frame_recall"],
                                                  "calibration_event_recall": cal_event["event_recall"], "outer_event_recall": event["event_recall"],
                                                  "calibration_trial_false_alarm_rate": cal_event["trial_false_alarm_rate"],
                                                  "outer_trial_false_alarm_rate": event["trial_false_alarm_rate"],
                                                  "calibration_mean_lead_frames": cal_event["mean_lead_frames"],
                                                  "outer_mean_lead_frames": event["mean_lead_frames"]})
                            for record in records:
                                trials_by_file[(horizon, group, population)].append({
                                    "group": group, "seed": seed, "horizon": horizon, "population": population,
                                    "rule": rule, "operating_point": operating_point, "threshold": threshold, **record,
                                })
                    if population == "primary":
                        for point in r7.compact_event_curve(role_sets["outer"], score_key, "metric_mask",
                                                            r7.event_tradeoff_thresholds(role_sets["outer"], score_key, "metric_mask")):
                            tradeoff.append({"group": group, "seed": seed, "horizon": horizon, "rule": rule,
                                             "scope": "outer_descriptive_only", **point})

        # Shared monotone calibration is applied to raw risks only. Per-H
        # operational rules remain separate and carry no monotonicity claim.
        calibration_by_horizon = {
            horizon: horizon_rows(r7, run["roles"]["calibration"], horizon, "primary") for horizon in horizons
        }
        shared = fit_shared_monotone_platt(calibration_by_horizon, lambda horizon: f"p_H{horizon}", "metric_mask")
        shared_platt_models.append({"group": group, "seed": seed, **shared})
        for role_name in ("calibration", "outer"):
            raw_rows = [row for row in run["roles"][role_name] if row["common_population"]]
            raw_violations = sum(not all(row[f"p_H{left}"] <= row[f"p_H{right}"] for left, right in zip(horizons, horizons[1:])) for row in raw_rows)
            consistency.append({"group": group, "seed": seed, "head_type": run["head_type"], "role": role_name,
                                "method": "raw_risk", "rows": len(raw_rows), "violations": raw_violations,
                                "violation_fraction": raw_violations / len(raw_rows) if raw_rows else math.nan})
        if shared["status"] == "available":
            for role_name in ("calibration", "outer"):
                for row in run["roles"][role_name]:
                    for horizon in horizons:
                        row[f"p_H{horizon}_shared_platt"] = apply_shared_platt(row[f"p_H{horizon}"], shared)
                common = [row for row in run["roles"][role_name] if row["common_population"]]
                violations = sum(not all(row[f"p_H{left}_shared_platt"] <= row[f"p_H{right}_shared_platt"]
                                         for left, right in zip(horizons, horizons[1:])) for row in common)
                consistency.append({"group": group, "seed": seed, "head_type": run["head_type"], "role": role_name,
                                    "method": "shared_monotone_platt_raw_risk", "rows": len(common), "violations": violations,
                                    "violation_fraction": violations / len(common) if common else math.nan})
            for horizon in horizons:
                role_sets = {role: horizon_rows(r7, rows, horizon, "primary") for role, rows in run["roles"].items()}
                key = f"p_H{horizon}_shared_platt"
                for role_name in ("calibration", "outer"):
                    reliability.extend(r7.reliability_rows(role_sets[role_name], key, "metric_mask", group, seed,
                                                           horizon, role_name, "shared_monotone_platt_raw_risk"))
                r7._CAL_CACHE.clear()
                for operating_point, kind, target in r7.OPS:
                    threshold, status, _ = r7.select_threshold(role_sets["calibration"], key, "metric_mask", kind, target)
                    base = {"method_type": "neural_shared_platt_raw", "group": group, "seed": seed,
                            "head_type": run["head_type"], "horizon": horizon, "population": "primary",
                            "rule": "raw", "rule_selected": False, "operating_point": operating_point,
                            "threshold_status": status, "threshold": threshold}
                    if threshold is None:
                        metrics.append(base)
                    else:
                        metrics.append({**base, **r7.frame_confusion(role_sets["outer"], key, threshold, "metric_mask"),
                                        **r7.event_metrics(role_sets["outer"], key, threshold, "metric_mask")[0]})

        # State-transition diagnostics use de-normalized XYZ values exported by
        # training. Persistence uses the current predicted force when present.
        outer_rows = run["roles"]["outer"]
        for step in range(1, max(horizons) + 1):
            cohort = [row for row in outer_rows if row.get(f"state_mask_step{step}")]
            if not cohort:
                continue
            prediction = np.asarray([row[f"state_prediction_step{step}"] for row in cohort], dtype=float)
            target = np.asarray([row[f"state_target_step{step}"] for row in cohort], dtype=float)
            error = prediction - target
            state_metrics.append({"group": group, "seed": seed, "role": "outer", "step": step,
                                  "method": "state_transition", "n": len(cohort),
                                  "rmse_xyz": float(np.sqrt(np.mean(error ** 2))), "mae_xyz": float(np.mean(np.abs(error))),
                                  "prediction_variance": float(np.mean(np.var(prediction, axis=0))),
                                  "target_variance": float(np.mean(np.var(target, axis=0))),
                                  "variance_ratio": float(np.mean(np.var(prediction, axis=0)) / max(float(np.mean(np.var(target, axis=0))), 1e-12))})
            append_state_errors(state_error_rows, state_trial_metrics, group=group, seed=seed, step=step,
                                method="state_transition", cohort=cohort, prediction=prediction, target=target)
            if all(all(axis in row for axis in ("force_x", "force_y", "force_z")) for row in cohort):
                persistence = np.asarray([[row["force_x"], row["force_y"], row["force_z"]] for row in cohort], dtype=float)
                persistence_error = persistence - target
                state_metrics.append({"group": group, "seed": seed, "role": "outer", "step": step,
                                      "method": "persistence", "n": len(cohort),
                                      "rmse_xyz": float(np.sqrt(np.mean(persistence_error ** 2))),
                                      "mae_xyz": float(np.mean(np.abs(persistence_error))),
                                      "prediction_variance": float(np.mean(np.var(persistence, axis=0))),
                                      "target_variance": float(np.mean(np.var(target, axis=0))),
                                      "variance_ratio": float(np.mean(np.var(persistence, axis=0)) / max(float(np.mean(np.var(target, axis=0))), 1e-12))})
                append_state_errors(state_error_rows, state_trial_metrics, group=group, seed=seed, step=step,
                                    method="persistence", cohort=cohort, prediction=persistence, target=target)

    # Reuse the complete Round-7 fit-only baseline implementation when the
    # prepared payload is bound in the manifest. Signed XYZ in base[:,769:772]
    # is never interpreted as Fn/Ft/ratio; these baselines only read current
    # pSlip and the explicit force_delta_slots field.
    prepared_spec = manifest.get("prepared")
    if prepared_spec:
        prepared_path = Path(prepared_spec["path"])
        if sha256(prepared_path) != prepared_spec["sha256"]:
            raise ValueError("prepared payload SHA mismatch")
        prepared = r7.torch.load(prepared_path, map_location="cpu", weights_only=False)
        sources[str(prepared_path.resolve())] = prepared_spec["sha256"]
        fit = r7.prepared_rows(prepared, "fit_train", False, horizons)
        timeline = {role: r7.prepared_rows(prepared, role, True, horizons) for role in ("selection", "calibration", "outer")}
        reference_group, reference_seed = sorted(all_runs)[0]
        state_linear_model = fit_state_linear_baseline(prepared["roles"]["fit_train"])
        state_outer = prepared["timelines"]["outer"]
        neural_outer = all_runs[sorted(all_runs)[0]]["roles"]["outer"]
        prepared_identity = list(zip((str(value) for value in state_outer["episode_id"]),
                                     (int(value) for value in state_outer["t"].tolist())))
        neural_identity = [(row["episode_id"], int(row["t"])) for row in neural_outer]
        if prepared_identity != neural_identity:
            raise ValueError("prepared state baseline outer identity drift")
        p4_runs = [run for (name, _), run in sorted(all_runs.items()) if name == "P4_state"]
        if p4_runs:
            p4_outer = p4_runs[0]["roles"]["outer"]
            for index, row in enumerate(p4_outer):
                for step in range(1, 6):
                    expected_mask = bool(state_outer["future_force_observed_mask"][index, step - 1])
                    if row.get(f"state_mask_step{step}") != expected_mask:
                        raise ValueError("prepared/P4 state mask drift")
                    if expected_mask and any(not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-6)
                                             for a, b in zip(row[f"state_target_step{step}"], state_outer["future_force_target"][index, step - 1].tolist())):
                        raise ValueError("prepared/P4 state target drift")
        state_linear_prediction = predict_state_linear_baseline(state_linear_model, state_outer)
        state_target = state_outer["future_force_target"].detach().cpu().numpy()
        state_mask = state_outer["future_force_observed_mask"].detach().cpu().numpy().astype(bool)
        for step in range(1, 6):
            valid = state_mask[:, step - 1]
            if not valid.any():
                continue
            prediction = state_linear_prediction[valid, step - 1]
            target_values = state_target[valid, step - 1]
            error = prediction - target_values
            state_metrics.append({"group": "fit_linear_state_baseline", "seed": "", "role": "outer", "step": step,
                                  "method": "fit_only_ridge_linear", "n": int(valid.sum()),
                                  "rmse_xyz": float(np.sqrt(np.mean(error ** 2))), "mae_xyz": float(np.mean(np.abs(error))),
                                  "prediction_variance": float(np.mean(np.var(prediction, axis=0))),
                                  "target_variance": float(np.mean(np.var(target_values, axis=0))),
                                  "variance_ratio": float(np.mean(np.var(prediction, axis=0)) / max(float(np.mean(np.var(target_values, axis=0))), 1e-12))})
            indices = np.flatnonzero(valid).tolist()
            linear_cohort = [{"episode_id": str(state_outer["episode_id"][index]),
                              "leakage_group": str(state_outer["leakage_group"][index]),
                              "t": int(state_outer["t"][index])} for index in indices]
            append_state_errors(state_error_rows, state_trial_metrics, group="fit_linear_state_baseline", seed="",
                                step=step, method="fit_only_ridge_linear", cohort=linear_cohort,
                                prediction=prediction, target=target_values)
        for horizon in horizons:
            fit_h = [row for row in fit if row["common_population"]]
            for name in r7.BASELINES:
                target = np.asarray([row[f"target_H{horizon}"] for row in fit_h], dtype=float)
                if name == "fit_prevalence":
                    model = {"prevalence": float(target.mean())}
                elif name in ("current_p_slip_raw", "history9_p_slip_mean_raw"):
                    model = {"literal_raw": True}
                else:
                    model = r7.fit_logistic(np.asarray([r7.baseline_features(row, name) for row in fit_h]), target)
                baseline_models.append({"name": name, "horizon": horizon,
                                        "parameters": {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in model.items()}})
                role_sets = {}
                for role, base_rows in timeline.items():
                    if name == "fit_prevalence":
                        prediction = np.full(len(base_rows), model["prevalence"])
                    elif model.get("literal_raw"):
                        prediction = np.asarray([r7.baseline_features(row, name)[0] for row in base_rows])
                    else:
                        prediction = r7.predict_logistic(model, np.asarray([r7.baseline_features(row, name) for row in base_rows]))
                    neural_reference = all_runs[(reference_group, reference_seed)]["roles"][role]
                    if len(base_rows) != len(neural_reference):
                        raise ValueError(f"baseline/neural length drift {role}")
                    rows = []
                    for base_row, neural_row, value in zip(base_rows, neural_reference, prediction):
                        if (base_row["episode_id"], base_row["t"]) != (neural_row["episode_id"], neural_row["t"]):
                            raise ValueError(f"baseline/neural identity drift {role}")
                        row = dict(neural_row); row[f"baseline_H{horizon}"] = float(value); rows.append(row)
                    r7.add_rules(rows, f"baseline_H{horizon}", f"baseline_H{horizon}_")
                    role_sets[role] = r7.role_rows(rows, horizon, "primary")
                r7._CAL_CACHE.clear()
                selected, evidence = r7.select_rule(role_sets["selection"], f"baseline_H{horizon}_", "metric_mask")
                for row in evidence:
                    selections.append({"group": name, "seed": "", "horizon": horizon, "population": "primary",
                                       "selected": row["rule"] == selected, **row})
                score_key = f"baseline_H{horizon}_{selected}"
                for operating_point, kind, threshold_target in r7.OPS:
                    threshold, status, _ = r7.select_threshold(role_sets["calibration"], score_key, "metric_mask", kind, threshold_target)
                    base = {"method_type": "fit_only_baseline", "group": name, "seed": "", "head_type": "baseline",
                            "horizon": horizon, "population": "primary", "rule": selected, "rule_selected": True,
                            "operating_point": operating_point, "threshold_status": status, "threshold": threshold}
                    if threshold is None:
                        metrics.append(base)
                    else:
                        metrics.append({**base, **r7.frame_confusion(role_sets["outer"], score_key, threshold, "metric_mask"),
                                        **r7.event_metrics(role_sets["outer"], score_key, threshold, "metric_mask")[0]})

    state_paired_rows: list[dict[str, Any]] = []
    state_groups = sorted({row["group"] for row in state_error_rows if row["method"] == "state_transition"})
    for group in state_groups:
        for step in range(1, max(horizons) + 1):
            candidate = [row for row in state_error_rows if row["group"] == group and row["step"] == step
                         and row["method"] == "state_transition"]
            if not candidate:
                continue
            comparators = {
                "persistence": [row for row in state_error_rows if row["group"] == group and row["step"] == step
                                and row["method"] == "persistence"],
            }
            linear_lookup = {(row["episode_id"], row["t"]): row for row in state_error_rows
                             if row["group"] == "fit_linear_state_baseline" and row["step"] == step}
            if linear_lookup:
                comparators["fit_only_ridge_linear"] = [
                    {**linear_lookup[(row["episode_id"], row["t"])], "seed": row["seed"]} for row in candidate
                ]
            for comparator_name, comparator in comparators.items():
                if not comparator:
                    continue
                for metric in ("mse", "rmse"):
                    def statistic(rows, metric_name=metric):
                        per_seed = []
                        for run_seed in seeds:
                            values = [row["squared_error_xyz"] for row in rows if int(row["seed"]) == run_seed]
                            if not values:
                                raise ValueError("state CI missing seed")
                            mean = float(np.mean(values)); per_seed.append(mean if metric_name == "mse" else math.sqrt(mean))
                        return float(np.mean(per_seed))
                    interval = paired_cluster_bootstrap(candidate, comparator, statistic,
                                                        repetitions=int(manifest.get("bootstrap_repetitions", 200)),
                                                        seed=8300 + step)
                    state_paired_rows.append({"comparison": f"{group}_vs_{comparator_name}", "step": step,
                                              "metric": f"{metric}_xyz", **interval})

    # Fixed interventions are evaluated with the unperturbed rule and threshold.
    intervention_spec = manifest.get("intervention_manifest")
    if intervention_spec:
        intervention_path = Path(intervention_spec["path"])
        if sha256(intervention_path) != intervention_spec["sha256"]:
            raise ValueError("intervention manifest SHA mismatch")
        intervention = json.loads(intervention_path.read_text())
        if intervention.get("schema") != "round8_intervention_manifest_v1" or intervention.get("status") != "complete":
            raise ValueError("invalid intervention manifest")
        if intervention.get("threshold_rule") != "use only unperturbed selection/calibration choices; no refit":
            raise ValueError("intervention threshold rule drift")
        validate_intervention_grid(intervention, all_runs)
        sources[str(intervention_path.resolve())] = sha256(intervention_path)
        for artifact in intervention["artifacts"]:
            group, seed, kind = artifact["group"], int(artifact["seed"]), artifact["intervention"]
            path = Path(artifact["path"])
            if sha256(path) != artifact["sha256"]:
                raise ValueError("intervention artifact SHA mismatch")
            rows = load_timeline_csv(path, horizons, head_type=artifact["head_type"])
            validate_cross_run_identity(all_runs[(group, seed)]["roles"]["outer"], rows, horizons)
            sources[str(path.resolve())] = artifact["sha256"]
            for horizon in horizons:
                r7.add_rules(rows, f"p_H{horizon}", f"H{horizon}_")
                selected = selected_rules[(group, seed, horizon)]
                altered_outer = r7.role_rows(rows, horizon, "primary")
                original_outer = r7.role_rows(all_runs[(group, seed)]["roles"]["outer"], horizon, "primary")
                score_key = f"H{horizon}_{selected}"
                mean_delta = float(np.mean([left[score_key] - right[score_key] for left, right in zip(altered_outer, original_outer)
                                            if left["metric_mask"]]))
                for operating_point, _, _ in r7.OPS:
                    threshold = selected_thresholds[(group, seed, horizon, operating_point)]
                    if threshold is None:
                        continue
                    intervention_metrics.append({"intervention": kind, "group": group, "seed": seed, "horizon": horizon,
                                                 "population": "primary", "rule": selected, "operating_point": operating_point,
                                                 "threshold_source": "unperturbed_calibration", "threshold": threshold,
                                                 "mean_score_delta_vs_unperturbed": mean_delta,
                                                 **r7.frame_confusion(altered_outer, score_key, threshold, "metric_mask"),
                                                 **r7.event_metrics(altered_outer, score_key, threshold, "metric_mask")[0]})

    # Legacy deployment gate diagnostic. The selected per-H rule and every
    # threshold remain those selected from the un-gated model. No gate-specific
    # selection or calibration is permitted.
    gate_protocol = json.loads((HERE / "GATE_DIAGNOSTIC_PROTOCOL.json").read_text())
    if gate_protocol.get("formula") != "p_future_raw * p_slip_current" or gate_protocol.get("refit") is not False:
        raise ValueError("fixed multiplicative gate protocol drift")
    sources[str((HERE / "GATE_DIAGNOSTIC_PROTOCOL.json").resolve())] = sha256(HERE / "GATE_DIAGNOSTIC_PROTOCOL.json")
    for (group, seed), run in sorted(all_runs.items()):
        gated_rows = [dict(row) for row in run["roles"]["outer"]]
        for horizon in horizons:
            for row in gated_rows:
                row[f"gated_p_H{horizon}"] = row[f"p_H{horizon}"] * row["p_slip_current"]
            r7.add_rules(gated_rows, f"gated_p_H{horizon}", f"gated_H{horizon}_")
            selected = selected_rules[(group, seed, horizon)]
            gated_outer = r7.role_rows(gated_rows, horizon, "primary")
            original_outer = r7.role_rows(run["roles"]["outer"], horizon, "primary")
            gated_key, original_key = f"gated_H{horizon}_{selected}", f"H{horizon}_{selected}"
            eligible_gated = [row for row in gated_outer if row["metric_mask"]]
            eligible_original = [row for row in original_outer if row["metric_mask"]]
            gated_ap = average_precision([row["target"] for row in eligible_gated],
                                         [row[f"gated_p_H{horizon}"] for row in eligible_gated])
            original_ap = average_precision([row["target"] for row in eligible_original],
                                            [row[f"p_H{horizon}"] for row in eligible_original])
            for operating_point, _, _ in r7.OPS:
                threshold = selected_thresholds[(group, seed, horizon, operating_point)]
                if threshold is None:
                    continue
                gated_frame = r7.frame_confusion(gated_outer, gated_key, threshold, "metric_mask")
                gated_event = r7.event_metrics(gated_outer, gated_key, threshold, "metric_mask")[0]
                original_frame = r7.frame_confusion(original_outer, original_key, threshold, "metric_mask")
                original_event = r7.event_metrics(original_outer, original_key, threshold, "metric_mask")[0]
                mean_score_delta = float(np.mean([
                    left[gated_key] - right[original_key]
                    for left, right in zip(gated_outer, original_outer) if left["metric_mask"]
                ]))
                fixed_gate_metrics.append({"group": group, "seed": seed, "horizon": horizon,
                                           "operating_point": operating_point, "population": "primary",
                                           "rule": selected, "threshold": threshold,
                                           "rule_source": "unperturbed_selection", "threshold_source": "unperturbed_calibration",
                                           "gate_formula": "p_future_raw * p_slip_current", "refit": False,
                                           "raw_average_precision_gated": gated_ap, "raw_average_precision_unperturbed": original_ap,
                                           "delta_raw_average_precision": gated_ap - original_ap,
                                           **{f"gated_{key}": value for key, value in {**gated_frame, **gated_event}.items()},
                                           **{f"unperturbed_{key}": value for key, value in {**original_frame, **original_event}.items()},
                                           "delta_frame_recall": gated_frame["frame_recall"] - original_frame["frame_recall"],
                                           "delta_trial_false_alarm_rate": gated_event["trial_false_alarm_rate"] - original_event["trial_false_alarm_rate"],
                                           "delta_event_recall": gated_event["event_recall"] - original_event["event_recall"],
                                           "delta_mean_lead_frames": gated_event["mean_lead_frames"] - original_event["mean_lead_frames"]})
                intervention_metrics.append({"intervention": "fixed_multiplicative_gate", "group": group,
                                             "seed": seed, "horizon": horizon, "population": "primary",
                                             "rule": selected, "operating_point": operating_point,
                                             "threshold_source": "unperturbed_calibration", "threshold": threshold,
                                             "mean_score_delta_vs_unperturbed": mean_score_delta,
                                             "raw_average_precision": gated_ap, "raw_average_precision_unperturbed": original_ap,
                                             **gated_frame, **gated_event})

    # Paired leakage-group intervals are restricted to predeclared comparisons.
    paired_rows: list[dict[str, Any]] = []
    comparisons = manifest.get("comparisons", [])
    for comparison in comparisons:
        candidate_name, comparator_name = comparison["candidate"], comparison["comparator"]
        for horizon in horizons:
            for operating_point in ("trial_FA_0.10", "event_recall_0.70"):
                candidate = [row for row in trials_by_file[(horizon, candidate_name, "primary")]
                             if row["operating_point"] == operating_point]
                comparator = [row for row in trials_by_file[(horizon, comparator_name, "primary")]
                              if row["operating_point"] == operating_point]
                if not candidate or not comparator:
                    continue
                for metric in ("event_recall", "trial_false_alarm_rate", "false_alarm_starts_per_trial",
                               "false_alarm_duration_per_trial", "mean_lead_frames"):
                    def statistic(rows, metric_name=metric):
                        values = []
                        for run_seed in seeds:
                            summary = r7.record_summary([row for row in rows if int(row["seed"]) == run_seed])
                            values.append(summary[metric_name])
                        return float(np.mean(values))
                    result = paired_cluster_bootstrap(candidate, comparator, statistic,
                                                      repetitions=int(manifest.get("bootstrap_repetitions", 200)),
                                                      seed=8100 + horizon)
                    paired_rows.append({"comparison": f"{candidate_name}_vs_{comparator_name}", "horizon": horizon,
                                        "operating_point": operating_point, "metric": metric, **result})
            candidate_frames = []; comparator_frames = []
            for run_seed in seeds:
                for method_name, destination in ((candidate_name, candidate_frames), (comparator_name, comparator_frames)):
                    for row in horizon_rows(r7, all_runs[(method_name, run_seed)]["roles"]["outer"], horizon, "primary"):
                        if row["metric_mask"]:
                            destination.append({**row, "seed": run_seed, "score": row[f"p_H{horizon}"]})
            for metric in ("average_precision", "brier"):
                def frame_statistic(rows, metric_name=metric):
                    values = []
                    for run_seed in seeds:
                        subset = [row for row in rows if int(row["seed"]) == run_seed]
                        y = np.asarray([row["target"] for row in subset], dtype=int)
                        p = np.asarray([row["score"] for row in subset], dtype=float)
                        values.append(average_precision(y, p) if metric_name == "average_precision" else float(np.mean((p - y) ** 2)))
                    return float(np.mean(values))
                result = paired_cluster_bootstrap(candidate_frames, comparator_frames, frame_statistic,
                                                  repetitions=int(manifest.get("bootstrap_repetitions", 200)),
                                                  seed=8200 + horizon)
                paired_rows.append({"comparison": f"{candidate_name}_vs_{comparator_name}", "horizon": horizon,
                                    "operating_point": "raw_common_population", "metric": metric, **result})

    aggregate: list[dict[str, Any]] = []
    summary_metrics = ("average_precision", "brier", "frame_fpr", "frame_recall", "trial_false_alarm_rate",
                       "event_recall", "active_overlap_recall", "mean_lead_frames", "lead_ge_3_recall", "lead_ge_5_recall")
    for method_type in ("neural_operational", "neural_shared_platt_raw"):
        for group in groups:
            for horizon in horizons:
                for operating_point, _, _ in r7.OPS:
                    rows = [row for row in metrics if row.get("method_type") == method_type and row.get("group") == group
                            and row.get("horizon") == horizon and row.get("population") == "primary"
                            and row.get("operating_point") == operating_point and row.get("threshold_status") == "available"
                            and (method_type != "neural_operational" or row.get("rule_selected"))]
                    available_seeds = sorted(int(row["seed"]) for row in rows)
                    unavailable_seeds = sorted(set(seeds) - set(available_seeds))
                    values = {
                        f"{metric}_{stat}": float(np.mean([row[metric] for row in rows]))
                        if stat == "mean" else float(np.std([row[metric] for row in rows], ddof=1))
                        if len(rows) > 1 else math.nan
                        for metric in summary_metrics for stat in ("mean", "std")
                    } if rows else {}
                    aggregate.append({"method_type": method_type, "group": group, "horizon": horizon,
                                      "operating_point": operating_point,
                                      "expected_seed_count": len(seeds), "available_seed_count": len(available_seeds),
                                      "unavailable_seed_count": len(unavailable_seeds),
                                      "seeds": ";".join(map(str, available_seeds)),
                                      "unavailable_seeds": ";".join(map(str, unavailable_seeds)),
                                      "std_definition": "sample_sd_ddof1; unavailable for fewer than 2 seeds",
                                      **values})

    output.mkdir(parents=True, exist_ok=True)
    atomic_csv(output / "metrics.csv", metrics)
    atomic_csv(output / "rule_selection.csv", selections)
    atomic_csv(output / "threshold_transfer.csv", transfers)
    atomic_csv(output / "outer_event_tradeoff_descriptive.csv", tradeoff)
    atomic_csv(output / "multi_horizon_consistency.csv", consistency)
    if aggregate:
        atomic_csv(output / "all_seed_summary.csv", aggregate)
    if reliability:
        atomic_csv(output / "reliability.csv", reliability)
    if paired_rows:
        atomic_csv(output / "paired_bootstrap_ci.csv", paired_rows)
    for (horizon, group, population), rows in trials_by_file.items():
        atomic_csv(output / "trials" / f"H{horizon}_{group}_{population}.csv", rows)
    atomic_json(output / "shared_platt_models.json", {"fit_role": "calibration", "models": shared_platt_models})
    atomic_json(output / "baseline_models.json", {"fit_role": "fit_train", "models": baseline_models,
                                                       "status": "complete" if prepared_spec else "not_run_no_prepared_binding"})
    atomic_json(output / "state_linear_baseline.json", state_linear_model if state_linear_model else {
        "status": "not_run_no_prepared_binding", "fit_role": "fit_train"
    })
    if state_metrics:
        atomic_csv(output / "state_prediction_metrics.csv", state_metrics)
    if state_trial_metrics:
        atomic_csv(output / "state_per_trial.csv", state_trial_metrics)
    if state_paired_rows:
        atomic_csv(output / "state_paired_bootstrap_ci.csv", state_paired_rows)
    if intervention_metrics:
        atomic_csv(output / "intervention_metrics.csv", intervention_metrics)
    atomic_csv(output / "fixed_multiplicative_gate_diagnostic.csv", fixed_gate_metrics)

    # Reuse the frozen R7 deterministic failure-case implementation.
    failure_cases = r7.render_failure_cases(output, all_runs, selected_rules, selected_thresholds,
                                             3 if 3 in horizons else horizons[0])
    atomic_csv(output / "failure_cases.csv", failure_cases)
    figure_directory = output / "figures"; figure_directory.mkdir(exist_ok=True)
    figure, axes = r7.plt.subplots(1, len(horizons), figsize=(5 * len(horizons), 4.5), squeeze=False)
    for axis, horizon in zip(axes[0], horizons):
        for group in groups:
            for index, run_seed in enumerate(seeds):
                curve = sorted([row for row in tradeoff if row["horizon"] == horizon and row["group"] == group
                                and row["rule"] == "raw" and row["seed"] == run_seed], key=lambda row: row["trial_false_alarm_rate"])
                axis.plot([row["trial_false_alarm_rate"] for row in curve], [row["event_recall"] for row in curve],
                          label=group if index == 0 else None, alpha=0.9 if index == 0 else 0.35)
        axis.set(title=f"H{horizon} outer descriptive", xlabel="trial-any false alarm", ylabel="new-start event recall",
                 xlim=(0, 1), ylim=(0, 1)); axis.grid(alpha=0.25)
    axes[0, -1].legend(fontsize=7); figure.tight_layout()
    figure.savefig(figure_directory / "event_recall_vs_trial_false_alarm.png", dpi=180); r7.plt.close(figure)
    report_lines = ["# 第八轮统一评价", "", "本报告沿用第七轮完整时间线、逐窗口规则选择和新告警起点事件口径。",
                    "原始风险与共享正斜率校准风险单独检查多窗口一致性；逐窗口运行告警不作概率单调声明。", "",
                    "## 证据边界", "", "- outer 是重复使用的开发评价角色，不是独立盲测。",
                    "- 数据集标签起点不是独立物理滑移真值。", "- calibration 阈值原样应用到 outer，实际误报必须以 outer 数值为准。", ""]
    (output / "SUMMARY_ZH.md").write_text("\n".join(report_lines))
    sources[str(manifest_path.resolve())] = sha256(manifest_path)
    sources[str(R7_EVALUATOR.resolve())] = sha256(R7_EVALUATOR)
    output_hashes = {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*")) if path.is_file() and path.name != "summary.json"
    }
    summary = {"format": "round8_formal_evaluation_v1", "status": "complete",
               "groups": groups, "seeds": seeds, "horizons": horizons, "runs": len(all_runs),
               "metrics": len(metrics), "paired_intervals": len(paired_rows), "all_seed_summaries": len(aggregate),
               "baseline_status": {"risk_simple": "complete" if prepared_spec else "not_run_no_prepared_binding",
                                   "state_persistence": "complete" if any(row["method"] == "persistence" for row in state_metrics) else "not_applicable_or_missing_predicted_force",
                                   "state_fit_only_linear": "complete" if state_linear_model else "not_run_no_prepared_binding"},
               "intervention_status": "complete" if intervention_spec and intervention_metrics else "not_run_no_manifest",
               "fixed_multiplicative_gate_status": "complete",
               "roles": {"selection": "rule only", "calibration": "probability and thresholds", "outer": "evaluation only"},
               "monotonicity_scope": "raw and shared-positive-slope-calibrated raw risks only; per-H operational rules excluded",
               "source_hashes": sources, "output_hashes": output_hashes, "test_role_consumed": False,
               "limitations": ["outer is reused development data, not a blind test",
                               "dataset-label onset is not independent physical slip truth",
                               "per-horizon operational scores have no cross-horizon monotonicity claim"]}
    atomic_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.manifest, args.output)
    print(json.dumps({"status": result["status"], "runs": result["runs"], "metrics": result["metrics"]}, sort_keys=True))


if __name__ == "__main__":
    main()
