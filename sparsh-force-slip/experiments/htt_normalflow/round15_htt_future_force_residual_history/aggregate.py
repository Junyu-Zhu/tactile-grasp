#!/usr/bin/env python3
"""Aggregate Round-15 runs and complete-group paired comparisons."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import future_train as ft

AXES = ("fx", "fy", "fz")


def read_csv(path):
    return list(csv.DictReader(open(path)))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_prediction(root, group, fold, seed, role="validation"):
    return torch.load(root / f"{group}_p{fold}_s{seed}" / f"predictions_{role}.pt", map_location="cpu", weights_only=False)


def assert_aligned(left, right):
    if left["episode_id"] != right["episode_id"] or left["leakage_group"] != right["leakage_group"] or not torch.equal(left["t"], right["t"]):
        raise ValueError("comparison endpoints differ")
    if not torch.equal(left["y"], right["y"]) or not torch.equal(left["y_current"], right["y_current"]):
        raise ValueError("comparison targets differ")


def stratum_array(payload):
    magnitude = (payload["y"][:, 2] - payload["y_current"]).abs().amax(1).numpy()
    return np.where(magnitude <= 0.25, "stable", np.where(magnitude >= 1.0, "changing", "transitional"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--round14-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    diagnostics = []
    temporal = []
    trials = []
    cache = {}
    for group in ft.GROUPS:
        for fold in range(1, 5):
            for seed in ft.SEEDS:
                run = f"{group}_p{fold}_s{seed}"
                for row in read_csv(arguments.evaluation / run / "metrics.csv"):
                    all_metrics.append({"group": group, "fold": fold, "seed": seed, **row})
                payload = load_prediction(arguments.evaluation, group, fold, seed)
                cache[(group, fold, seed)] = payload
                gt = payload["y"].numpy()
                current = payload["x_current_pred"].numpy()
                persistence = payload["predictions"]["predicted_current_persistence"].numpy()
                group_array = np.array(payload["leakage_group"])
                times = payload["t"].numpy()
                for method, tensor in payload["predictions"].items():
                    prediction = tensor.numpy()
                    error = np.abs(prediction - gt).mean((1, 2))
                    diagnostics.append({"group": group, "fold": fold, "seed": seed, "method": method,
                                        "prediction_std": float(prediction.std()), "gt_std": float(gt.std()),
                                        "std_ratio": float(prediction.std() / max(gt.std(), 1e-12)),
                                        "fraction_within_0p05N_of_pred_current": float(np.mean(np.abs(prediction - persistence) < 0.05)),
                                        "abs_error_vs_t_corr": float(np.corrcoef(error, times)[0, 1]) if np.std(error) else 0.0,
                                        "finite": bool(np.isfinite(prediction).all())})
                    for episode in sorted(set(payload["episode_id"])):
                        mask = np.array(payload["episode_id"]) == episode
                        for horizon_index, horizon in enumerate(ft.HORIZONS):
                            for axis_index, axis in enumerate(AXES):
                                prediction_std = float(prediction[mask, horizon_index, axis_index].std())
                                gt_std = float(gt[mask, horizon_index, axis_index].std())
                                copy = float(np.mean(np.abs(prediction[mask, horizon_index, axis_index] - current[mask, axis_index]) < 0.05))
                                temporal.append({"group": group, "fold": fold, "seed": seed, "method": method,
                                                 "episode_id": episode, "horizon": horizon, "axis": axis, "n": int(mask.sum()),
                                                 "prediction_temporal_std_n": prediction_std, "gt_temporal_std_n": gt_std,
                                                 "temporal_std_ratio": prediction_std / gt_std if gt_std >= 1e-6 else "",
                                                 "gt_near_constant_lt_1e_6": gt_std < 1e-6,
                                                 "collapse_flag_pred_le_0p05_gt_ge_0p25": prediction_std <= 0.05 and gt_std >= 0.25,
                                                 "copy_fraction_within_0p05n": copy})
                    for leakage_group in sorted(set(group_array)):
                        mask = group_array == leakage_group
                        trials.append({"group": group, "fold": fold, "seed": seed, "method": method,
                                       "leakage_group": leakage_group, "n": int(mask.sum()),
                                       "future_mae": float(np.abs(prediction[mask] - gt[mask]).mean()),
                                       "change_mae": float(np.abs((prediction[mask] - current[mask, None, :]) - (gt[mask] - payload["y_current"].numpy()[mask, None, :])).mean())})
    write_csv(arguments.output / "all_metrics.csv", all_metrics)
    write_csv(arguments.output / "diagnostics.csv", diagnostics)
    write_csv(arguments.output / "temporal_diagnostics.csv", temporal)
    write_csv(arguments.output / "trial_metrics.csv", trials)

    collapse = []
    for group in ft.GROUPS:
        for method in ("neural", "predicted_current_persistence", "fit_ridge_residual", "gt_current_persistence_ideal_only"):
            selected = [row for row in temporal if row["group"] == group and row["method"] == method]
            variable = [row for row in selected if not row["gt_near_constant_lt_1e_6"]]
            collapse.append({"group": group, "method": method, "episode_axis_horizon_cells": len(selected),
                             "gt_near_constant_cells": sum(row["gt_near_constant_lt_1e_6"] for row in selected),
                             "collapse_flag_cells": sum(row["collapse_flag_pred_le_0p05_gt_ge_0p25"] for row in selected),
                             "collapse_flag_fraction_of_gt_variable": sum(row["collapse_flag_pred_le_0p05_gt_ge_0p25"] for row in variable) / len(variable) if variable else "",
                             "median_temporal_std_ratio_gt_nonconstant": float(np.median([row["temporal_std_ratio"] for row in variable])) if variable else "",
                             "mean_copy_fraction": float(np.mean([row["copy_fraction_within_0p05n"] for row in selected]))})
    write_csv(arguments.output / "temporal_collapse_summary.csv", collapse)

    historical = []
    for group in ("V", "F_concat", "F_dual"):
        for fold in range(1, 5):
            for seed in ft.SEEDS:
                for row in read_csv(arguments.round14_evaluation / f"{group}_p{fold}_s{seed}" / "metrics.csv"):
                    if row["role"] == "validation" and row["stratum"] == "all":
                        historical.append({"group": group, "fold": fold, "seed": seed, **row})
    write_csv(arguments.output / "round14_historical_reference_metrics.csv", historical)

    summary = []
    for group in ft.GROUPS:
        for method in ("neural", "predicted_current_persistence", "fit_ridge_residual", "gt_current_persistence_ideal_only"):
            selected = [row for row in all_metrics if row["group"] == group and row["method"] == method and row["role"] == "validation" and row["stratum"] == "all"]
            for metric in ("future_mae", "future_rmse", "change_mae", "change_rmse"):
                values = np.array([float(row[metric]) for row in selected])
                summary.append({"group": group, "method": method, "metric": metric, "mean": float(values.mean()),
                                "std_across_run_axis_horizon_cells": float(values.std()), "cells": len(values)})
    for group in ("V", "F_concat", "F_dual"):
        selected = [row for row in historical if row["group"] == group and row["method"] == "neural"]
        for metric in ("future_mae", "future_rmse", "change_mae", "change_rmse"):
            values = np.array([float(row[metric]) for row in selected])
            summary.append({"group": f"R14_{group}", "method": "neural_absolute_output", "metric": metric,
                            "mean": float(values.mean()), "std_across_run_axis_horizon_cells": float(values.std()), "cells": len(values)})
    write_csv(arguments.output / "summary.csv", summary)

    def comparison_payload(label, fold, seed):
        if label == "H_history_minus_C_current":
            left, right = cache[("H_history", fold, seed)], cache[("C_current", fold, seed)]
            assert_aligned(left, right)
            return left, left["predictions"]["neural"], right["predictions"]["neural"]
        if label == "H_history_minus_R14_F_concat":
            left = cache[("H_history", fold, seed)]
            right = load_prediction(arguments.round14_evaluation, "F_concat", fold, seed)
            assert_aligned(left, right)
            return left, left["predictions"]["neural"], right["predictions"]["neural"]
        group = "H_history" if label.startswith("H_history") else "C_current"
        left = cache[(group, fold, seed)]
        return left, left["predictions"]["neural"], left["predictions"]["predicted_current_persistence"]

    ci_rows = []
    labels = ("H_history_minus_C_current", "H_history_minus_R14_F_concat", "H_history_minus_persistence", "C_current_minus_persistence")
    for label in labels:
        for fold in range(1, 5):
            seed_payloads = {seed: comparison_payload(label, fold, seed) for seed in ft.SEEDS}
            groups = sorted(set(seed_payloads[ft.SEEDS[0]][0]["leakage_group"]))
            rng = np.random.default_rng(1500000 + 100 * labels.index(label) + fold)
            draws = [rng.choice(groups, len(groups), replace=True) for _ in range(2000)]
            for stratum in ("all", "stable", "transitional", "changing"):
                overall_per_seed_group = {}
                for seed, (payload, left, right) in seed_payloads.items():
                    strata = stratum_array(payload)
                    group_array = np.array(payload["leakage_group"])
                    for leakage_group in groups:
                        mask = group_array == leakage_group
                        if stratum != "all":
                            mask &= strata == stratum
                        if not mask.any():
                            continue
                        gt = payload["y"][mask].numpy()
                        overall_per_seed_group[(seed, leakage_group)] = float(np.abs(left[mask].numpy() - gt).mean() - np.abs(right[mask].numpy() - gt).mean())
                overall_values = []
                for draw in draws:
                    seed_values = []
                    for seed in ft.SEEDS:
                        available = [overall_per_seed_group[(seed, item)] for item in draw if (seed, item) in overall_per_seed_group]
                        if available:
                            seed_values.append(float(np.mean(available)))
                    if seed_values:
                        overall_values.append(float(np.mean(seed_values)))
                if overall_values:
                    low, high = np.quantile(overall_values, [0.025, 0.975])
                    ci_rows.append({"comparison": label, "fold": fold, "stratum": stratum, "horizon": "all", "axis": "all",
                                    "metric": "future_mae_n", "point": float(np.mean(list(overall_per_seed_group.values()))),
                                    "ci_low": float(low), "ci_high": float(high), "bootstrap_draws": 2000,
                                    "unit": "complete_leakage_group_shared_across_seeds", "noninferiority_margin": ""})
                for horizon_index, horizon in enumerate(ft.HORIZONS):
                    for axis_index, axis in enumerate(AXES):
                        per_seed_group = {}
                        for seed, (payload, left, right) in seed_payloads.items():
                            strata = stratum_array(payload)
                            group_array = np.array(payload["leakage_group"])
                            for leakage_group in groups:
                                mask = group_array == leakage_group
                                if stratum != "all":
                                    mask &= strata == stratum
                                if not mask.any():
                                    continue
                                gt = payload["y"][mask, horizon_index, axis_index].numpy()
                                per_seed_group[(seed, leakage_group)] = float(np.abs(left[mask, horizon_index, axis_index].numpy() - gt).mean() - np.abs(right[mask, horizon_index, axis_index].numpy() - gt).mean())
                        values = []
                        for draw in draws:
                            seed_values = []
                            for seed in ft.SEEDS:
                                available = [per_seed_group[(seed, item)] for item in draw if (seed, item) in per_seed_group]
                                if available:
                                    seed_values.append(float(np.mean(available)))
                            if seed_values:
                                values.append(float(np.mean(seed_values)))
                        if not values:
                            continue
                        point = float(np.mean(list(per_seed_group.values())))
                        low, high = np.quantile(values, [0.025, 0.975])
                        ci_rows.append({"comparison": label, "fold": fold, "stratum": stratum, "horizon": horizon, "axis": axis,
                                        "metric": "future_mae_n", "point": point, "ci_low": float(low), "ci_high": float(high),
                                        "bootstrap_draws": 2000, "unit": "complete_leakage_group_shared_across_seeds", "noninferiority_margin": ""})
    write_csv(arguments.output / "paired_group_ci.csv", ci_rows)
    result = {"schema": "round15_future_aggregate_v1", "status": "complete", "runs": 24,
              "metric_rows": len(all_metrics), "trial_rows": len(trials), "diagnostic_rows": len(diagnostics),
              "temporal_diagnostic_rows": len(temporal), "paired_ci_rows": len(ci_rows),
              "all_finite": all(row["finite"] for row in diagnostics),
              "mse_decomposition_max_abs_residual": max(float(row["mse_decomposition_max_abs_residual"]) for row in all_metrics),
              "stable_noninferiority_margin": None,
              "hashes": {path.name: ft.sha(path) for path in arguments.output.iterdir() if path.is_file() and path.name != "SUMMARY.json"}}
    ft.atomic_json(result, arguments.output / "SUMMARY.json")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
