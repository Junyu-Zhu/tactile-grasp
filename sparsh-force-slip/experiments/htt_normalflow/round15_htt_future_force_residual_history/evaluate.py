#!/usr/bin/env python3
"""Evaluate one Round-15 checkpoint with fixed residual-aware references."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import future_train as ft

AXES = ("fx", "fy", "fz")


def design(normalized_x, group):
    selected = normalized_x[:, -1:, :] if group == "C_current" else normalized_x
    return selected.reshape(len(selected), -1).double()


def ridge_fit(x, y, regularization=1e-6):
    x = torch.cat((x, torch.ones(len(x), 1, dtype=x.dtype)), 1)
    y = y.reshape(len(y), -1).double()
    penalty = torch.eye(x.shape[1], dtype=x.dtype) * regularization
    penalty[-1, -1] = 0
    return torch.linalg.solve(x.T @ x + penalty, x.T @ y)


def ridge_predict(x, weights):
    augmented = torch.cat((x, torch.ones(len(x), 1, dtype=x.dtype)), 1)
    return (augmented @ weights).float().reshape(-1, 3, 3)


def metric_rows(role, data, predictions):
    rows = []
    change_magnitude = (data["y"][:, 2] - data["y_current"]).abs().amax(1)
    strata = np.where(change_magnitude.numpy() <= 0.25, "stable", np.where(change_magnitude.numpy() >= 1.0, "changing", "transitional"))
    predicted_current = data["x"][:, -1, 192:195]
    for method, prediction in predictions.items():
        for horizon_index, horizon in enumerate(ft.HORIZONS):
            for axis_index, axis in enumerate(AXES):
                future_error = prediction[:, horizon_index, axis_index] - data["y"][:, horizon_index, axis_index]
                deployed_change = prediction[:, horizon_index, axis_index] - predicted_current[:, axis_index]
                true_change = data["y"][:, horizon_index, axis_index] - data["y_current"][:, axis_index]
                change_error = deployed_change - true_change
                anchor_error = predicted_current[:, axis_index] - data["y_current"][:, axis_index]
                for stratum in ("all", "stable", "transitional", "changing"):
                    mask = np.ones(len(future_error), dtype=bool) if stratum == "all" else strata == stratum
                    if not mask.any():
                        continue
                    fe = future_error.numpy()[mask]
                    ce = change_error.numpy()[mask]
                    tc = true_change.numpy()[mask]
                    dc = deployed_change.numpy()[mask]
                    ae = anchor_error.numpy()[mask]
                    rows.append({
                        "role": role,
                        "method": method,
                        "horizon": horizon,
                        "axis": axis,
                        "stratum": stratum,
                        "n": int(mask.sum()),
                        "future_mae": float(np.abs(fe).mean()),
                        "future_rmse": float(np.sqrt(np.mean(fe ** 2))),
                        "change_mae": float(np.abs(ce).mean()),
                        "change_rmse": float(np.sqrt(np.mean(ce ** 2))),
                        "future_mse": float(np.mean(fe ** 2)),
                        "anchor_error_mse_component": float(np.mean(ae ** 2)),
                        "residual_minus_true_change_mse_component": float(np.mean(ce ** 2)),
                        "twice_cross_component": float(np.mean(2.0 * ae * ce)),
                        "mse_decomposition_max_abs_residual": float(abs(np.mean(fe ** 2) - np.mean(ae ** 2) - np.mean(ce ** 2) - np.mean(2.0 * ae * ce))),
                        "mean_signed_true_change": float(tc.mean()),
                        "mean_abs_true_change": float(np.abs(tc).mean()),
                        "mean_signed_deployed_change": float(dc.mean()),
                        "mean_abs_deployed_change": float(np.abs(dc).mean()),
                    })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--group", choices=ft.GROUPS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    data = torch.load(arguments.data, map_location="cpu", weights_only=False)
    ft.validate_cache(data)
    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["identity"]["group"] != arguments.group or checkpoint["identity"]["round14_data_sha256"] != ft.sha(arguments.data):
        raise ValueError("checkpoint/data identity mismatch")
    model = ft.init_model(arguments.group, checkpoint["identity"]["seed"])
    model.load_state_dict(checkpoint["model"])
    model.eval().to(arguments.device)
    normalizer = checkpoint["normalizer"]
    fit = data["roles"]["fit"]
    fit_x = ft.normalize_x(fit["x"], normalizer)
    fit_target = ft.target_residual(fit, normalizer)
    ridge = ridge_fit(design(fit_x, arguments.group), fit_target)
    rows = []
    hashes = {}
    for role, value in data["roles"].items():
        normalized_x = ft.normalize_x(value["x"], normalizer)
        with torch.inference_mode():
            normalized_residual = model(normalized_x.to(arguments.device)).cpu()
        neural = ft.decode_future(normalized_residual, value, normalizer)
        predicted_current = value["x"][:, -1, 192:195]
        persistence = predicted_current[:, None, :].expand(-1, 3, -1).clone()
        ideal = value["y_current"][:, None, :].expand(-1, 3, -1).clone()
        linear_residual = ridge_predict(design(normalized_x, arguments.group), ridge)
        linear = ft.decode_future(linear_residual, value, normalizer)
        predictions = {
            "neural": neural,
            "predicted_current_persistence": persistence,
            "fit_ridge_residual": linear,
            "gt_current_persistence_ideal_only": ideal,
        }
        rows.extend(metric_rows(role, value, predictions))
        output = arguments.output / f"predictions_{role}.pt"
        ft.atomic_save({
            "episode_id": value["episode_id"],
            "leakage_group": value["leakage_group"],
            "t": value["t"],
            "x_current_pred": predicted_current,
            "y_current": value["y_current"],
            "y": value["y"],
            "predictions": predictions,
        }, output)
        hashes[role] = ft.sha(output)
    with (arguments.output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    receipt = {
        "schema": "round15_future_evaluation_v1",
        "status": "complete",
        "group": arguments.group,
        "checkpoint": str(arguments.checkpoint),
        "checkpoint_sha256": ft.sha(arguments.checkpoint),
        "data_sha256": ft.sha(arguments.data),
        "fit_only_ridge": 1e-6,
        "prediction_hashes": hashes,
        "metrics_sha256": ft.sha(arguments.output / "metrics.csv"),
        "output_clipped": False,
        "test_consumed": False,
        "interpretation": "Residual/change error mixes predicted-current anchor bias correction with true change prediction.",
    }
    ft.atomic_json(receipt, arguments.output / "SUMMARY.json")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
