#!/usr/bin/env python3
"""Derive missing preregistered trial metrics and output-range diagnostics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import future_train as ft

AXES = ("fx", "fy", "fz")
METHODS = ("neural", "predicted_current_persistence", "fit_ridge_residual", "gt_current_persistence_ideal_only")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--reporting", type=Path, required=True)
    parser.add_argument("--final-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    episode_rows, range_rows = [], []
    for group in ft.GROUPS:
        for fold in range(1, 5):
            for seed in ft.SEEDS:
                run = f"{group}_p{fold}_s{seed}"
                payload = torch.load(arguments.evaluation / run / "predictions_validation.pt", map_location="cpu", weights_only=False)
                gt = payload["y"].numpy()
                gt_current = payload["y_current"].numpy()
                predicted_current = payload["x_current_pred"].numpy()
                episodes = np.asarray(payload["episode_id"])
                leakage = np.asarray(payload["leakage_group"])
                for method in METHODS:
                    prediction = payload["predictions"][method].numpy()
                    for episode in sorted(set(episodes)):
                        mask = episodes == episode
                        leakage_values = sorted(set(leakage[mask]))
                        if len(leakage_values) != 1:
                            raise ValueError(f"episode crosses leakage groups: {run} {episode}")
                        for horizon_index, horizon in enumerate(ft.HORIZONS):
                            for axis_index, axis in enumerate(AXES):
                                future_error = prediction[mask, horizon_index, axis_index] - gt[mask, horizon_index, axis_index]
                                predicted_change = prediction[mask, horizon_index, axis_index] - predicted_current[mask, axis_index]
                                true_change = gt[mask, horizon_index, axis_index] - gt_current[mask, axis_index]
                                change_error = predicted_change - true_change
                                episode_rows.append({"group": group, "fold": fold, "seed": seed, "method": method,
                                                     "episode_id": episode, "leakage_group": leakage_values[0],
                                                     "horizon": horizon, "axis": axis, "n": int(mask.sum()),
                                                     "future_mae_n": float(np.abs(future_error).mean()),
                                                     "future_rmse_n": float(np.sqrt(np.square(future_error).mean())),
                                                     "change_mae_n": float(np.abs(change_error).mean()),
                                                     "change_rmse_n": float(np.sqrt(np.square(change_error).mean()))})
                    for horizon_index, horizon in enumerate(ft.HORIZONS):
                        for axis_index, axis in enumerate(AXES):
                            target = gt[:, horizon_index, axis_index]
                            output = prediction[:, horizon_index, axis_index]
                            target_neg = np.isclose(target, -20.0, rtol=0.0, atol=1e-6)
                            target_pos = np.isclose(target, 20.0, rtol=0.0, atol=1e-6)
                            output_neg = output < -20.0
                            output_pos = output > 20.0
                            range_rows.append({"group": group, "fold": fold, "seed": seed, "method": method,
                                               "role": "validation", "horizon": horizon, "axis": axis, "n": len(target),
                                               "target_eq_neg20_count": int(target_neg.sum()),
                                               "target_eq_neg20_fraction": float(target_neg.mean()),
                                               "target_eq_pos20_count": int(target_pos.sum()),
                                               "target_eq_pos20_fraction": float(target_pos.mean()),
                                               "target_eq_abs20_count": int((target_neg | target_pos).sum()),
                                               "target_eq_abs20_fraction": float((target_neg | target_pos).mean()),
                                               "prediction_lt_neg20_count": int(output_neg.sum()),
                                               "prediction_lt_neg20_fraction": float(output_neg.mean()),
                                               "prediction_gt_pos20_count": int(output_pos.sum()),
                                               "prediction_gt_pos20_fraction": float(output_pos.mean()),
                                               "prediction_outside_abs20_count": int((output_neg | output_pos).sum()),
                                               "prediction_outside_abs20_fraction": float((output_neg | output_pos).mean()),
                                               "output_clipping_applied": False})
    episode_path = arguments.reporting / "episode_axis_horizon_metrics.csv"
    range_path = arguments.reporting / "saturation_output_range.csv"
    write_csv(episode_path, episode_rows)
    write_csv(range_path, range_rows)
    summary_rows = []
    for group in ft.GROUPS:
        for method in METHODS:
            selected = [row for row in range_rows if row["group"] == group and row["method"] == method]
            total = sum(row["n"] for row in selected)
            summary_rows.append({"group": group, "method": method, "validation_scalar_cells": total,
                                 "target_eq_abs20_count": sum(row["target_eq_abs20_count"] for row in selected),
                                 "target_eq_abs20_fraction": sum(row["target_eq_abs20_count"] for row in selected) / total,
                                 "prediction_outside_abs20_count": sum(row["prediction_outside_abs20_count"] for row in selected),
                                 "prediction_outside_abs20_fraction": sum(row["prediction_outside_abs20_count"] for row in selected) / total,
                                 "output_clipping_applied": False})
    summary_path = arguments.final_report / "saturation_output_range_summary.csv"
    write_csv(summary_path, summary_rows)
    if not all(row["n"] > 0 and all(np.isfinite(row[field]) for field in ("future_mae_n", "future_rmse_n", "change_mae_n", "change_rmse_n")) for row in episode_rows):
        raise ValueError("non-finite or empty episode metric cell")
    metric_rows = list(csv.DictReader((arguments.reporting / "all_metrics.csv").open()))
    reconciliation_differences = []
    for group in ft.GROUPS:
        for fold in range(1, 5):
            for seed in ft.SEEDS:
                for method in METHODS:
                    for horizon in ft.HORIZONS:
                        for axis in AXES:
                            selected = [row for row in episode_rows if row["group"] == group and row["fold"] == fold and row["seed"] == seed and row["method"] == method and row["horizon"] == horizon and row["axis"] == axis]
                            total = sum(row["n"] for row in selected)
                            derived = {
                                "future_mae": sum(row["n"] * row["future_mae_n"] for row in selected) / total,
                                "future_rmse": np.sqrt(sum(row["n"] * row["future_rmse_n"] ** 2 for row in selected) / total),
                                "change_mae": sum(row["n"] * row["change_mae_n"] for row in selected) / total,
                                "change_rmse": np.sqrt(sum(row["n"] * row["change_rmse_n"] ** 2 for row in selected) / total),
                            }
                            reference = next(row for row in metric_rows if row["group"] == group and int(row["fold"]) == fold and int(row["seed"]) == seed and row["method"] == method and row["role"] == "validation" and row["stratum"] == "all" and int(row["horizon"]) == horizon and row["axis"] == axis)
                            reconciliation_differences.extend(abs(float(reference[field]) - float(value)) for field, value in derived.items())
    maximum_reconciliation_difference = max(reconciliation_differences)
    if maximum_reconciliation_difference > 2e-6:
        raise ValueError(f"episode metrics do not reconcile: {maximum_reconciliation_difference}")
    audit = {"schema": "round15_protocol_diagnostics_v1", "status": "pass", "source_role": "validation",
             "runs": 24, "methods_per_run": len(METHODS), "episode_axis_horizon_rows": len(episode_rows),
             "range_rows": len(range_rows), "episode_count_across_runs": len(episode_rows) // (len(METHODS) * len(AXES) * len(ft.HORIZONS)),
             "maximum_reconciliation_difference_vs_all_metrics": maximum_reconciliation_difference,
             "checks": ["episode maps to exactly one leakage group", "all episode metric cells are finite and nonempty", "episode metrics reconcile to existing validation all-stratum metrics", "future and true-change MAE/RMSE", "finite predictions inherited from evaluation audit", "no output clipping", "target boundary is delivered clipped GT and not a pre-clip saturation estimate"],
             "limitations": ["GT values equal to +/-20 N show contact with the delivered target boundary; the pre-delivery unclipped target is unavailable, so physical sensor saturation prevalence cannot be inferred.", "Prediction outside +/-20 N is reported without clipping and is not itself proof of physical saturation."],
             "hashes": {episode_path.name: sha(episode_path), range_path.name: sha(range_path), summary_path.name: sha(summary_path)}}
    ft.atomic_json(audit, arguments.output)
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
