from __future__ import annotations

import csv
import hashlib
import json
import math
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

try:
    from .core import (apply_shared_platt, cumulative_risk_from_hazard, event_metrics,
                       fit_shared_monotone_platt, fit_state_linear_baseline, paired_cluster_bootstrap,
                       predict_state_linear_baseline)
    from .schema import load_timeline_csv, validate_cross_run_identity
    from .evaluate import validate_manifest
    from .evaluate_formal import evaluate as evaluate_formal, validate_intervention_grid
    from .fast_thresholds import compact_event_curve as fast_compact_event_curve
except ImportError:
    from core import (apply_shared_platt, cumulative_risk_from_hazard, event_metrics,
                      fit_shared_monotone_platt, fit_state_linear_baseline, paired_cluster_bootstrap,
                      predict_state_linear_baseline)
    from schema import load_timeline_csv, validate_cross_run_identity
    from evaluate import validate_manifest
    from evaluate_formal import evaluate as evaluate_formal, validate_intervention_grid
    from fast_thresholds import compact_event_curve as fast_compact_event_curve


def passing_audit_bundle(root: Path, groups: list[str], seeds: list[int]) -> tuple[dict, dict]:
    runs = [{"id": f"future_{group}_{seed}", "checks": {"complete": True}}
            for group in groups for seed in seeds]
    audit = root / "training_audit.json"
    audit.write_text(json.dumps({"schema": "round8_training_audit_v1", "status": "pass",
                                 "source_sha256": "1" * 64, "inventory_sha256": "2" * 64,
                                 "run_count": len(runs), "runs": runs,
                                 "frozen_inputs": {"fixture": True}, "initialization": {"fixture": True}}))
    checkpoint_rows = []
    for group in groups:
        for seed in seeds:
            for kind in ("best", "latest", "config"):
                path = root / f"future_{group}_{seed}_{kind}.fixture"
                path.write_text(f"{group}/{seed}/{kind}\n")
                checkpoint_rows.append({"id": f"future_{group}_{seed}", "group": group, "seed": seed,
                                        "kind": kind, "path": str(path),
                                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    checkpoint = root / "CHECKPOINT_INDEX.csv"
    with checkpoint.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(checkpoint_rows[0])); writer.writeheader(); writer.writerows(checkpoint_rows)
    return ({"path": str(audit), "sha256": hashlib.sha256(audit.read_bytes()).hexdigest()},
            {"path": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()})


class EvaluationTests(unittest.TestCase):
    def test_fast_threshold_sweep_exactly_matches_frozen_r7(self):
        import importlib.util
        r7_path = Path(__file__).resolve().parents[2] / "round7_temporal_fairness_event_warning/evaluation/evaluate_r7.py"
        spec = importlib.util.spec_from_file_location("r7_threshold_reference", r7_path)
        reference = importlib.util.module_from_spec(spec)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); spec.loader.exec_module(reference)
        rows = []
        for episode, onset, scores in (("positive", 6, [0.1, 0.4, 0.3, 0.8, 0.2, 0.9, 0.7]),
                                       ("negative", None, [0.3, 0.2, 0.5, 0.4, 0.1, 0.6, 0.2])):
            for t, score in enumerate(scores):
                rows.append({"episode_id": episode, "t": t, "onset": onset, "metric": t <= 5,
                             "target": int(onset is not None and t < onset <= t + 3) if t <= 5 else None,
                             "score": score})
        thresholds = sorted({math.nextafter(1.0, math.inf), *(row["score"] for row in rows)}, reverse=True)
        expected = reference.compact_event_curve(rows, "score", "metric", thresholds)
        actual = fast_compact_event_curve(rows, "score", "metric", thresholds)
        self.assertEqual(len(expected), len(actual))
        for left, right in zip(expected, actual):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                if isinstance(left[key], float) and math.isnan(left[key]):
                    self.assertTrue(math.isnan(right[key]))
                else:
                    self.assertEqual(left[key], right[key])

    def test_hazard_cumulative_risk_is_monotone(self):
        risk = cumulative_risk_from_hazard([0.1, 0.2, 0.05, 0.4, 0.1])
        self.assertTrue(np.all(np.diff(risk) >= 0))
        self.assertAlmostEqual(risk[2], 1 - 0.9 * 0.8 * 0.95)

    def test_shared_positive_slope_calibration_preserves_horizon_order(self):
        rows = {}
        for horizon, scores in {1: [0.1, 0.2, 0.7, 0.8], 3: [0.2, 0.4, 0.8, 0.9], 5: [0.3, 0.5, 0.9, 0.95]}.items():
            rows[horizon] = [{"metric": True, "target": index >= 2, f"p_H{horizon}": score}
                             for index, score in enumerate(scores)]
        model = fit_shared_monotone_platt(rows, lambda horizon: f"p_H{horizon}", "metric")
        self.assertEqual(model["status"], "available")
        for index in range(4):
            calibrated = [apply_shared_platt(rows[h][index][f"p_H{h}"], model) for h in (1, 3, 5)]
            self.assertLessEqual(calibrated[0], calibrated[1])
            self.assertLessEqual(calibrated[1], calibrated[2])

    def test_fit_only_state_linear_baseline_recovers_linear_dynamics(self):
        generator = np.random.default_rng(9)
        history = generator.normal(size=(80, 9, 3)).astype(np.float32)
        target = np.stack([history[:, -1] + step * (history[:, -1] - history[:, -2]) for step in range(1, 6)], axis=1)
        rows = {"signed_force_xyz": history, "future_force_target": target,
                "future_force_observed_mask": np.ones((80, 5), dtype=bool)}
        model = fit_state_linear_baseline(rows)
        prediction = predict_state_linear_baseline(model, rows)
        self.assertEqual(model["fit_role"], "fit_train")
        self.assertLess(float(np.sqrt(np.mean((prediction - target) ** 2))), 1e-2)

    def test_ongoing_alarm_is_not_a_new_event_hit(self):
        rows = [{"episode_id": "e", "leakage_group": "g", "t": t, "onset": 4,
                 "metric": True, "target": int(t >= 2), "score": 0.9} for t in range(4)]
        summary, records = event_metrics(rows, "score", 0.5, "metric")
        self.assertEqual(summary["event_recall"], 0.0)
        self.assertEqual(summary["active_overlap_recall"], 1.0)
        self.assertTrue(records[0]["ongoing_from_before_positive_window"])

    def test_paired_bootstrap_rejects_population_drift(self):
        candidate = [{"episode_id": "e", "leakage_group": "g", "t": 1, "value": 1.0}]
        comparator = [{"episode_id": "other", "leakage_group": "g", "t": 1, "value": 0.0}]
        with self.assertRaisesRegex(ValueError, "identity drift"):
            paired_cluster_bootstrap(candidate, comparator, lambda rows: np.mean([r["value"] for r in rows]))

    def test_paired_bootstrap_resamples_whole_groups(self):
        candidate = []; comparator = []
        for group, value in (("g1", 1.0), ("g2", 3.0)):
            for t in range(2):
                common = {"episode_id": group, "leakage_group": group, "t": t}
                candidate.append({**common, "value": value})
                comparator.append({**common, "value": 0.0})
        result = paired_cluster_bootstrap(candidate, comparator, lambda rows: np.mean([r["value"] for r in rows]), repetitions=20)
        self.assertAlmostEqual(result["estimate"], 2.0)
        self.assertEqual(result["valid_replicates"], 20)

    def test_hazard_schema_rejects_inconsistent_exported_risk(self):
        fields = ["episode_id", "leakage_group", "t", "first_current_slip_t", "current_slip_label",
                  "p_slip_current", "timeline_contiguous", "common_population", "predicted_force_x",
                  "predicted_force_y", "predicted_force_z", "latest_force_delta_x", "latest_force_delta_y",
                  "latest_force_delta_z", "latest_force_delta_valid"]
        for horizon in (1, 3, 5):
            fields += [f"eligible_H{horizon}", f"common_eligible_H{horizon}", f"right_censored_H{horizon}",
                       f"target_future_H{horizon}", f"p_future_H{horizon}_raw"]
        fields += [f"q_future_step{step}_raw" for step in range(1, 6)]
        row = {field: "" for field in fields}
        row.update({"episode_id": "e", "leakage_group": "g", "t": 18, "first_current_slip_t": 21,
                    "current_slip_label": 0, "p_slip_current": 0.1, "timeline_contiguous": False,
                    "common_population": True})
        for horizon in (1, 3, 5):
            row.update({f"eligible_H{horizon}": True, f"common_eligible_H{horizon}": True,
                        f"right_censored_H{horizon}": False, f"target_future_H{horizon}": int(horizon >= 3),
                        f"p_future_H{horizon}_raw": 0.99})
        for step in range(1, 6):
            row[f"q_future_step{step}_raw"] = 0.1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timeline.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerow(row)
            with self.assertRaisesRegex(ValueError, "hazard/cumulative mismatch"):
                load_timeline_csv(path, head_type="discrete_hazard")

    def test_cross_run_identity_detects_label_drift(self):
        base = {"episode_id": "e", "leakage_group": "g", "t": 18, "onset": 21,
                "current_slip_label": 0, "common_population": True}
        for horizon in (1, 3, 5):
            base[f"eligible_H{horizon}"] = True; base[f"target_H{horizon}"] = int(horizon >= 3)
        changed = dict(base); changed["target_H3"] = 0
        with self.assertRaisesRegex(ValueError, "identity drift"):
            validate_cross_run_identity([base], [changed])

    def test_cross_run_identity_detects_state_target_and_mask_drift(self):
        base = {"episode_id": "e", "leakage_group": "g", "t": 18, "onset": 21,
                "current_slip_label": 0, "common_population": True, "state_steps": [1],
                "state_mask_step1": True, "state_target_step1": [1.0, 2.0, 3.0]}
        for horizon in (1, 3, 5):
            base[f"eligible_H{horizon}"] = True; base[f"target_H{horizon}"] = int(horizon >= 3)
        target_changed = dict(base); target_changed["state_target_step1"] = [9.0, 2.0, 3.0]
        with self.assertRaisesRegex(ValueError, "state-target identity drift"):
            validate_cross_run_identity([base], [target_changed])
        mask_changed = dict(base); mask_changed["state_mask_step1"] = False
        with self.assertRaisesRegex(ValueError, "state-mask identity drift"):
            validate_cross_run_identity([base], [mask_changed])

    def test_intervention_grid_rejects_missing_seed_or_extra_state_intervention(self):
        runs = {
            ("B_xyz", 1): {"head_type": "independent_sigmoid"},
            ("P4_state", 1): {"head_type": "independent_sigmoid"},
        }
        def artifact(group, kind):
            return {"group": group, "seed": 1, "intervention": kind, "role": "outer",
                    "head_type": "independent_sigmoid"}
        complete = [artifact(group, kind) for group, kinds in {
            "B_xyz": ("force_input_fit_mean_zero", "force_causal_lag1"),
            "P4_state": ("force_input_fit_mean_zero", "force_causal_lag1", "state_input_fit_mean_zero"),
        }.items() for kind in kinds]
        validate_intervention_grid({"artifacts": complete}, runs)
        with self.assertRaisesRegex(ValueError, "intervention grid drift"):
            validate_intervention_grid({"artifacts": complete[:-1]}, runs)
        with self.assertRaisesRegex(ValueError, "state intervention assigned to non-state group"):
            validate_intervention_grid({"artifacts": complete + [artifact("B_xyz", "state_input_fit_mean_zero")]}, runs)

    def test_future_target_must_match_onset(self):
        fields = ["episode_id", "leakage_group", "t", "first_current_slip_t", "current_slip_label",
                  "p_slip_current", "timeline_contiguous", "common_population"]
        for horizon in (1, 3, 5):
            fields += [f"eligible_H{horizon}", f"common_eligible_H{horizon}", f"right_censored_H{horizon}",
                       f"target_future_H{horizon}", f"p_future_H{horizon}_raw"]
        row = {"episode_id": "e", "leakage_group": "g", "t": 18, "first_current_slip_t": 21,
               "current_slip_label": 0, "p_slip_current": 0.1, "timeline_contiguous": False,
               "common_population": True}
        for horizon in (1, 3, 5):
            row.update({f"eligible_H{horizon}": True, f"common_eligible_H{horizon}": True,
                        f"right_censored_H{horizon}": False, f"target_future_H{horizon}": int(horizon >= 3),
                        f"p_future_H{horizon}_raw": 0.1})
        row["target_future_H1"] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timeline.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerow(row)
            with self.assertRaisesRegex(ValueError, "future target/onset drift"):
                load_timeline_csv(path, head_type="independent_sigmoid")

    def test_manifest_rejects_role_path_reuse_and_leakage_group_overlap(self):
        fields = ["episode_id", "leakage_group", "t", "first_current_slip_t", "current_slip_label",
                  "p_slip_current", "timeline_contiguous", "common_population", "predicted_force_x",
                  "predicted_force_y", "predicted_force_z", "latest_force_delta_x", "latest_force_delta_y",
                  "latest_force_delta_z", "latest_force_delta_valid"]
        for horizon in (1, 3, 5):
            fields += [f"eligible_H{horizon}", f"common_eligible_H{horizon}", f"right_censored_H{horizon}",
                       f"target_future_H{horizon}", f"p_future_H{horizon}_raw"]
        row = {"episode_id": "e", "leakage_group": "shared", "t": 18, "first_current_slip_t": "",
               "current_slip_label": 0, "p_slip_current": 0.1, "timeline_contiguous": False,
               "common_population": True, "predicted_force_x": 0.0, "predicted_force_y": 0.0,
               "predicted_force_z": 1.0, "latest_force_delta_x": 0.0, "latest_force_delta_y": 0.0,
               "latest_force_delta_z": 0.0, "latest_force_delta_valid": True}
        for horizon in (1, 3, 5):
            row.update({f"eligible_H{horizon}": True, f"common_eligible_H{horizon}": True,
                        f"right_censored_H{horizon}": False, f"target_future_H{horizon}": 0,
                        f"p_future_H{horizon}_raw": 0.1})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); paths = []
            for role in ("selection", "calibration", "outer"):
                path = root / f"{role}.csv"
                with path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerow(row)
                paths.append(path)
            def write_manifest(chosen_paths):
                audit, checkpoint = passing_audit_bundle(root, ["B_xyz"], [1])
                artifacts = [{"group": "B_xyz", "seed": 1, "role": role, "head_type": "independent_sigmoid",
                              "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                             for role, path in zip(("selection", "calibration", "outer"), chosen_paths)]
                manifest = root / "manifest.json"
                manifest.write_text(json.dumps({"schema": "round8_prediction_manifest_v1", "status": "complete",
                                                "training_audit": audit,
                                                "checkpoint_index": checkpoint,
                                                "groups": ["B_xyz"],
                                                "seeds": [1], "horizons": [1, 3, 5], "artifacts": artifacts}))
                return manifest
            with self.assertRaisesRegex(ValueError, "reuses one timeline across roles"):
                validate_manifest(write_manifest([paths[0], paths[0], paths[0]]))
            with self.assertRaisesRegex(ValueError, "role leakage-group overlap"):
                validate_manifest(write_manifest(paths))

    def test_manifest_requires_complete_status_and_passing_bound_training_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"schema": "round8_prediction_manifest_v1", "status": "audit_pending"}))
            with self.assertRaisesRegex(ValueError, "status must be complete"):
                validate_manifest(manifest)
            audit = root / "training_audit.json"; audit.write_text(json.dumps({"status": "fail"}))
            manifest.write_text(json.dumps({"schema": "round8_prediction_manifest_v1", "status": "complete",
                                            "training_audit": {"path": str(audit),
                                                               "sha256": hashlib.sha256(audit.read_bytes()).hexdigest()}}))
            with self.assertRaisesRegex(ValueError, "training audit is not pass"):
                validate_manifest(manifest)

    def test_formal_entry_produces_r7_compatible_outputs(self):
        fields = ["episode_id", "leakage_group", "t", "first_current_slip_t", "current_slip_label",
                  "p_slip_current", "timeline_contiguous", "common_population",
                  "force_x", "force_y", "force_z", "dforce_x", "dforce_y", "dforce_z", "latest_force_delta_valid"]
        for horizon in (1, 3, 5):
            fields += [f"eligible_H{horizon}", f"common_eligible_H{horizon}", f"right_censored_H{horizon}",
                       f"target_future_H{horizon}", f"p_future_H{horizon}_raw"]
        fields += [f"q_future_step{step}_raw" for step in range(1, 6)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); artifacts = []
            for group, head_type in (("B_xyz", "independent_sigmoid"), ("hazard", "discrete_hazard"),
                                     ("P4_state", "independent_sigmoid")):
                for role in ("selection", "calibration", "outer"):
                    path = root / f"{group}_{role}.csv"; rows = []
                    group_fields = list(fields)
                    if group == "P4_state":
                        aliases = {"force_x": "predicted_force_x", "force_y": "predicted_force_y", "force_z": "predicted_force_z"}
                        group_fields = [aliases.get(name, name) for name in group_fields]
                        for step in range(1, 6):
                            for axis in "xyz":
                                group_fields += [f"predicted_state_force_tplus{step}_{axis}", f"target_state_force_tplus{step}_{axis}"]
                            group_fields += [f"state_target_mask_tplus{step}"]
                    for episode_base, onset in (("positive", 6), ("negative", None)):
                        episode = f"{role}_{episode_base}"
                        for t in range(11):
                            eligible = t <= 5
                            row = {"episode_id": episode, "leakage_group": episode, "t": t,
                                   "first_current_slip_t": "" if onset is None else onset,
                                   "current_slip_label": int(onset is not None and t >= onset),
                                   "p_slip_current": 0.1, "timeline_contiguous": t > 0,
                                   "common_population": eligible, "force_x": 0.1 * t, "force_y": 0.0,
                                   "force_z": 1.0, "dforce_x": 0.1, "dforce_y": 0.0, "dforce_z": 0.0,
                                   "latest_force_delta_valid": True}
                            hazards = [0.05, 0.1, 0.2, 0.4, 0.6] if episode == "positive" and t >= 1 else [0.02] * 5
                            cumulative = cumulative_risk_from_hazard(hazards)
                            for step, value in enumerate(hazards, 1):
                                row[f"q_future_step{step}_raw"] = value if head_type == "discrete_hazard" else ""
                            for horizon in (1, 3, 5):
                                target = int(onset is not None and t < onset <= t + horizon) if eligible else ""
                                probability = float(cumulative[horizon - 1]) if head_type == "discrete_hazard" else (0.8 if target == 1 else 0.1)
                                row.update({f"eligible_H{horizon}": eligible, f"common_eligible_H{horizon}": eligible,
                                            f"right_censored_H{horizon}": not eligible, f"target_future_H{horizon}": target,
                                            f"p_future_H{horizon}_raw": probability})
                            if group == "P4_state":
                                row["predicted_force_x"] = row.pop("force_x"); row["predicted_force_y"] = row.pop("force_y"); row["predicted_force_z"] = row.pop("force_z")
                                for step in range(1, 6):
                                    for axis, value in zip("xyz", (0.1 * (t + step), 0.0, 1.0)):
                                        row[f"predicted_state_force_tplus{step}_{axis}"] = value
                                        row[f"target_state_force_tplus{step}_{axis}"] = value
                                    row[f"state_target_mask_tplus{step}"] = True
                            rows.append(row)
                    with path.open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=group_fields); writer.writeheader(); writer.writerows(rows)
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    artifacts.append({"group": group, "seed": 20260914, "role": role, "head_type": head_type,
                                      "path": str(path), "sha256": digest})
            manifest = root / "manifest.json"
            audit, checkpoint = passing_audit_bundle(root, ["B_xyz", "hazard", "P4_state"], [20260914])
            manifest.write_text(json.dumps({"schema": "round8_prediction_manifest_v1", "status": "complete",
                                            "training_audit": audit, "checkpoint_index": checkpoint,
                                            "horizons": [1, 3, 5],
                                            "groups": ["B_xyz", "hazard", "P4_state"], "seeds": [20260914],
                                            "bootstrap_repetitions": 5,
                                            "comparisons": [{"candidate": "hazard", "comparator": "B_xyz"}],
                                            "artifacts": artifacts}))
            output = root / "evaluation"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                summary = evaluate_formal(manifest, output)
            self.assertEqual(summary["status"], "complete")
            self.assertTrue((output / "metrics.csv").is_file())
            self.assertTrue((output / "paired_bootstrap_ci.csv").is_file())
            self.assertTrue((output / "failure_cases.csv").is_file())
            self.assertTrue((output / "fixed_multiplicative_gate_diagnostic.csv").is_file())
            with (output / "intervention_metrics.csv").open() as handle:
                self.assertIn("fixed_multiplicative_gate", {row["intervention"] for row in csv.DictReader(handle)})
            with (output / "state_prediction_metrics.csv").open() as handle:
                methods = {row["method"] for row in csv.DictReader(handle)}
            self.assertIn("persistence", methods)
            self.assertTrue((output / "state_per_trial.csv").is_file())
            self.assertTrue((output / "state_paired_bootstrap_ci.csv").is_file())
            self.assertEqual(summary["baseline_status"]["state_persistence"], "complete")
            self.assertIn("metrics.csv", summary["output_hashes"])


if __name__ == "__main__":
    unittest.main()
