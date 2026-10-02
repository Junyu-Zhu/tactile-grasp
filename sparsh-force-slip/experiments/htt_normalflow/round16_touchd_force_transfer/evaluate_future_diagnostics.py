#!/usr/bin/env python3
"""Audit future-force error decomposition and registered episode diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from touchd_common import atomic_json, sha256

AXES = ("fx", "fy", "fz")
HORIZONS = (1, 5, 10)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"no rows for {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def values(array) -> np.ndarray:
    return array.detach().cpu().numpy() if isinstance(array, torch.Tensor) else np.asarray(array)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True,
                        help="R14 evaluate.py output directory containing predictions_<role>.pt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    identity_max = 0.0
    input_hashes = {}
    for role in ("fit", "selection", "calibration", "validation"):
        path = args.evaluation / f"predictions_{role}.pt"
        input_hashes[role] = sha256(path)
        data = torch.load(path, map_location="cpu", weights_only=False)
        episodes = np.asarray(data["episode_id"], dtype=str)
        y_current, y = values(data["y_current"]), values(data["y"])
        if y.shape != (len(episodes), 3, 3) or y_current.shape != (len(episodes), 3):
            raise RuntimeError("future prediction shape")
        strata_value = np.max(np.abs(y[:, 2] - y_current), axis=1)
        strata = np.where(strata_value <= .25, "stable",
                          np.where(strata_value >= 1.0, "changing", "transitional"))
        for method, prediction_tensor in data["predictions"].items():
            prediction = values(prediction_tensor)
            for horizon_index, horizon in enumerate(HORIZONS):
                for axis_index, axis in enumerate(AXES):
                    future_error = prediction[:, horizon_index, axis_index] - y[:, horizon_index, axis_index]
                    current_prediction = values(data["predictions"]["predicted_current_persistence"])[:, 0, axis_index]
                    current_error = current_prediction - y_current[:, axis_index]
                    change_error = ((prediction[:, horizon_index, axis_index] - current_prediction)
                                    - (y[:, horizon_index, axis_index] - y_current[:, axis_index]))
                    residual = future_error - (change_error + current_error)
                    identity_max = max(identity_max, float(np.max(np.abs(residual))))
                    for episode in sorted(set(episodes)):
                        episode_mask = episodes == episode
                        for stratum in ("all", "stable", "transitional", "changing"):
                            mask = episode_mask if stratum == "all" else episode_mask & (strata == stratum)
                            if not mask.any():
                                continue
                            fe, ce, de = future_error[mask], current_error[mask], change_error[mask]
                            rows.append({
                                "role": role, "episode_id": episode, "method": method,
                                "horizon": horizon, "axis": axis, "stratum": stratum, "n": int(mask.sum()),
                                "future_mae": float(np.mean(np.abs(fe))), "future_mse": float(np.mean(fe ** 2)),
                                "current_mae": float(np.mean(np.abs(ce))), "current_mse": float(np.mean(ce ** 2)),
                                "change_mae": float(np.mean(np.abs(de))), "change_mse": float(np.mean(de ** 2)),
                                "cross_2_change_current": float(np.mean(2 * de * ce)),
                                "mse_identity_rhs": float(np.mean(de ** 2) + np.mean(ce ** 2) + np.mean(2 * de * ce)),
                                "prediction_min": float(np.min(prediction[mask, horizon_index, axis_index])),
                                "prediction_max": float(np.max(prediction[mask, horizon_index, axis_index])),
                                "prediction_variance": float(np.var(prediction[mask, horizon_index, axis_index])),
                                "target_variance": float(np.var(y[mask, horizon_index, axis_index])),
                                "copy_current_mae": float(np.mean(np.abs(prediction[mask, horizon_index, axis_index] - current_prediction[mask]))),
                            })
    if identity_max > 2e-5:
        raise RuntimeError(f"future error identity residual {identity_max}")
    write_csv(args.output / "trial_diagnostics.csv", rows)
    result = {
        "schema": "round16_future_diagnostics_v1", "status": "complete",
        "strata": {"stable": "h10 max-axis change <=0.25N", "transitional": "0.25N<h10 max-axis change<1N",
                   "changing": "h10 max-axis change >=1N"},
        "mse_identity": "future_mse = change_mse + current_mse + 2*mean(change_error*current_error)",
        "identity_max_abs_sample_residual": identity_max, "input_hashes": input_hashes,
        "trial_diagnostics_sha256": sha256(args.output / "trial_diagnostics.csv"), "test_consumed": False,
    }
    atomic_json(args.output / "SUMMARY.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
