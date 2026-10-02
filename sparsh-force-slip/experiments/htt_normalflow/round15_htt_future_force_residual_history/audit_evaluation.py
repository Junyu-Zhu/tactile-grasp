#!/usr/bin/env python3
"""Audit Round-15 prediction payload identities and metric reconstruction."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import future_train as ft


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    records = []
    maximum_metric_difference = 0.0
    for group in ft.GROUPS:
        for fold in range(1, 5):
            for seed in ft.SEEDS:
                run = f"{group}_p{fold}_s{seed}"
                root = arguments.evaluation / run
                summary = json.loads((root / "SUMMARY.json").read_text())
                assert summary["status"] == "complete" and not summary["test_consumed"] and not summary["output_clipped"]
                payload = torch.load(root / "predictions_validation.pt", map_location="cpu", weights_only=False)
                assert all(torch.isfinite(value).all() for value in payload["predictions"].values())
                persistence = payload["x_current_pred"][:, None, :].expand(-1, 3, -1)
                assert torch.equal(payload["predictions"]["predicted_current_persistence"], persistence)
                rows = list(csv.DictReader(open(root / "metrics.csv")))
                for method, prediction in payload["predictions"].items():
                    for horizon_index, horizon in enumerate(ft.HORIZONS):
                        for axis_index, axis in enumerate(("fx", "fy", "fz")):
                            row = next(item for item in rows if item["role"] == "validation" and item["method"] == method and item["stratum"] == "all" and int(item["horizon"]) == horizon and item["axis"] == axis)
                            recomputed = float(np.abs(prediction[:, horizon_index, axis_index].numpy() - payload["y"][:, horizon_index, axis_index].numpy()).mean())
                            maximum_metric_difference = max(maximum_metric_difference, abs(recomputed - float(row["future_mae"])))
                            assert abs(recomputed - float(row["future_mae"])) < 1e-6
                            assert float(row["mse_decomposition_max_abs_residual"]) < 1e-5
                records.append({"run": run, "status": "accepted", "summary_sha256": ft.sha(root / "SUMMARY.json"),
                                "validation_prediction_sha256": ft.sha(root / "predictions_validation.pt")})
    receipt = {"schema": "round15_evaluation_audit_v1", "status": "pass", "accepted_count": len(records),
               "max_recomputed_metric_difference": maximum_metric_difference, "test_consumed": False,
               "checks": ["all_predictions_finite", "persistence_identity", "future_mae_recomputed", "mse_error_decomposition", "no_output_clip"],
               "runs": records}
    ft.atomic_json(receipt, arguments.output)
    print(json.dumps({"status": "pass", "runs": len(records), "max_difference": maximum_metric_difference}))


if __name__ == "__main__":
    main()
