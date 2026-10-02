#!/usr/bin/env python3
"""Pre-registered Round-14 F_concat history-sensitivity diagnostic."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

import future_train as r15

AXES = ("fx", "fy", "fz")


def load_r14_module(round14_root):
    specification = importlib.util.spec_from_file_location("round14_future_train", round14_root / "future_train.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round14", type=Path, required=True)
    parser.add_argument("--round14-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    r14 = load_r14_module(arguments.round14)
    endpoint_rows = []
    prediction_cache = {}
    for fold in range(1, 5):
        for seed in r15.SEEDS:
            run = f"F_concat_p{fold}_s{seed}"
            data_path = arguments.round14_output / "prepared" / f"p{fold}_s{seed}" / "prepared.pt"
            checkpoint_path = arguments.round14_output / "formal" / run / "best.pth"
            data = torch.load(data_path, map_location="cpu", weights_only=False)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if checkpoint["identity"]["group"] != "F_concat" or checkpoint["identity"]["round14_data_sha256"] != r15.sha(data_path):
                raise ValueError(f"identity mismatch: {run}")
            model = r14.init_model("F_concat", seed)
            model.load_state_dict(checkpoint["model"])
            model.eval().to(arguments.device)
            normalizer = checkpoint["normalizer"]
            value = data["roles"]["validation"]
            normalized_x = (value["x"] - normalizer["x_mean"]) / normalizer["x_std"]
            rng = np.random.default_rng(2026091501 + 100 * fold + seed % 100)
            order = rng.permutation(8)
            if np.array_equal(order, np.arange(8)):
                order = np.roll(order, 1)
            shuffled = normalized_x.clone()
            shuffled[:, :8] = normalized_x[:, order]
            repeated = normalized_x[:, -1:, :].expand(-1, 9, -1).clone()
            with torch.inference_mode():
                base = model(normalized_x.to(arguments.device)).cpu() * normalizer["y_std"] + normalizer["y_mean"]
                shuffle_prediction = model(shuffled.to(arguments.device)).cpu() * normalizer["y_std"] + normalizer["y_mean"]
                repeat_prediction = model(repeated.to(arguments.device)).cpu() * normalizer["y_std"] + normalizer["y_mean"]
            current_prediction = value["x"][:, -1, 192:195]
            persistence = current_prediction[:, None, :].expand(-1, 3, -1)
            h10_magnitude = (value["y"][:, 2] - value["y_current"]).abs().amax(1)
            strata = np.where(h10_magnitude.numpy() <= 0.25, "stable", np.where(h10_magnitude.numpy() >= 1.0, "changing", "transitional"))
            prediction_cache[(fold, seed)] = {
                "episode_id": value["episode_id"], "leakage_group": value["leakage_group"], "t": value["t"],
                "y": value["y"], "y_current": value["y_current"], "predicted_current": current_prediction,
                "base": base, "persistence": persistence, "shuffle": shuffle_prediction, "repeat": repeat_prediction,
                "strata": strata,
            }
            for index in range(len(value["t"])):
                for horizon_index, horizon in enumerate(r15.HORIZONS):
                    for axis_index, axis in enumerate(AXES):
                        gt_future = float(value["y"][index, horizon_index, axis_index])
                        gt_current = float(value["y_current"][index, axis_index])
                        pred_current = float(current_prediction[index, axis_index])
                        pred_future = float(base[index, horizon_index, axis_index])
                        true_change = gt_future - gt_current
                        deployed_change = pred_future - pred_current
                        direction_eligible = abs(true_change) >= 0.10
                        endpoint_rows.append({
                            "fold": fold, "seed": seed, "episode_id": value["episode_id"][index],
                            "leakage_group": value["leakage_group"][index], "t": int(value["t"][index]),
                            "horizon": horizon, "axis": axis, "stratum": strata[index],
                            "gt_current": gt_current, "predicted_current": pred_current,
                            "current_signed_error": pred_current - gt_current,
                            "current_abs_error": abs(pred_current - gt_current),
                            "gt_future": gt_future, "fconcat_future": pred_future,
                            "fconcat_future_abs_error": abs(pred_future - gt_future),
                            "persistence_future_abs_error": abs(pred_current - gt_future),
                            "true_change_signed": true_change, "true_change_abs": abs(true_change),
                            "deployed_change_signed": deployed_change, "deployed_change_abs": abs(deployed_change),
                            "change_abs_error": abs(deployed_change - true_change),
                            "direction_eligible_abs_true_change_ge_0p10": direction_eligible,
                            "direction_agreement": int(np.sign(deployed_change) == np.sign(true_change)) if direction_eligible else "",
                            "shuffle_future": float(shuffle_prediction[index, horizon_index, axis_index]),
                            "shuffle_displacement_abs": abs(float(shuffle_prediction[index, horizon_index, axis_index]) - pred_future),
                            "shuffle_future_abs_error": abs(float(shuffle_prediction[index, horizon_index, axis_index]) - gt_future),
                            "repeat_future": float(repeat_prediction[index, horizon_index, axis_index]),
                            "repeat_displacement_abs": abs(float(repeat_prediction[index, horizon_index, axis_index]) - pred_future),
                            "repeat_future_abs_error": abs(float(repeat_prediction[index, horizon_index, axis_index]) - gt_future),
                        })
    write_csv(arguments.output / "endpoint_diagnostic.csv", endpoint_rows)

    trial_rows = []
    grouped_rows = {}
    for row in endpoint_rows:
        key = (row["fold"], row["seed"], row["leakage_group"], row["horizon"], row["axis"], row["stratum"])
        grouped_rows.setdefault(key, []).append(row)
    numeric = ["current_abs_error", "fconcat_future_abs_error", "persistence_future_abs_error", "true_change_abs", "deployed_change_abs", "change_abs_error", "shuffle_displacement_abs", "shuffle_future_abs_error", "repeat_displacement_abs", "repeat_future_abs_error"]
    for key in sorted(grouped_rows):
        fold, seed, leakage_group, horizon, axis, stratum = key
        selected = grouped_rows[key]
        eligible = [row for row in selected if row["direction_agreement"] != ""]
        trial_rows.append({"fold": fold, "seed": seed, "leakage_group": leakage_group, "horizon": horizon, "axis": axis, "stratum": stratum, "n": len(selected),
                           **{f"mean_{name}": float(np.mean([row[name] for row in selected])) for name in numeric},
                           "direction_eligible_n": len(eligible), "direction_agreement_fraction": float(np.mean([row["direction_agreement"] for row in eligible])) if eligible else ""})
    write_csv(arguments.output / "trial_diagnostic.csv", trial_rows)

    ci_rows = []
    for fold in range(1, 5):
        groups = sorted(set(prediction_cache[(fold, r15.SEEDS[0])]["leakage_group"]))
        rng = np.random.default_rng(150000 + fold)
        draws = [rng.choice(groups, len(groups), replace=True) for _ in range(2000)]
        for horizon_index, horizon in enumerate(r15.HORIZONS):
            for axis_index, axis in enumerate(AXES):
                group_differences = {}
                for seed in r15.SEEDS:
                    payload = prediction_cache[(fold, seed)]
                    group_array = np.array(payload["leakage_group"])
                    changing = payload["strata"] == "changing"
                    for group in groups:
                        mask = (group_array == group) & changing
                        if not mask.any():
                            continue
                        gt = payload["y"][mask, horizon_index, axis_index].numpy()
                        neural = payload["base"][mask, horizon_index, axis_index].numpy()
                        persistence = payload["persistence"][mask, horizon_index, axis_index].numpy()
                        group_differences[(seed, group)] = float(np.mean(np.abs(neural - gt)) - np.mean(np.abs(persistence - gt)))
                values = []
                for draw in draws:
                    seed_values = []
                    for seed in r15.SEEDS:
                        available = [group_differences[(seed, group)] for group in draw if (seed, group) in group_differences]
                        if available:
                            seed_values.append(float(np.mean(available)))
                    if seed_values:
                        values.append(float(np.mean(seed_values)))
                point_values = list(group_differences.values())
                low, high = np.quantile(values, [0.025, 0.975])
                ci_rows.append({"fold": fold, "horizon": horizon, "axis": axis, "stratum": "changing", "comparison": "F_concat_minus_predicted_current_persistence_future_MAE_N", "point_complete_group_mean": float(np.mean(point_values)), "ci_low": float(low), "ci_high": float(high), "bootstrap_draws": 2000, "shared_draws_across_seeds": True})
    write_csv(arguments.output / "changing_group_bootstrap_ci.csv", ci_rows)
    receipt = {
        "schema": "round15_r14_history_diagnostic_v1", "status": "complete", "runs": 12,
        "endpoint_rows": len(endpoint_rows), "trial_rows": len(trial_rows), "direction_threshold_n": 0.10,
        "strata": {"stable_max_axis_h10_le_n": 0.25, "changing_max_axis_h10_ge_n": 1.0},
        "perturbations": {"shuffle_seed": "2026091501+100*fold+seed%100", "shuffle": "joint first-eight visual+force order; current step unchanged", "repeat": "replace first eight by current base step"},
        "bootstrap": {"draws": 2000, "seed": "150000+fold", "unit": "complete leakage group", "shared_across_seeds": True, "folds_separate": True},
        "interpretation": "Perturbations are out-of-distribution sensitivity probes, not causal evidence; diagnostic cannot alter formal training.",
        "hashes": {path.name: r15.sha(path) for path in arguments.output.iterdir() if path.is_file() and path.name != "SUMMARY.json"},
    }
    r15.atomic_json(receipt, arguments.output / "SUMMARY.json")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
