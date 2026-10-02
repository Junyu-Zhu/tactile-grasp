from __future__ import annotations

import importlib.util
import csv
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("r8_report", HERE / "report.py")
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


class ReportingTests(unittest.TestCase):
    @staticmethod
    def paired_grid():
        return [{"comparison": f"{a}_vs_{b}", "horizon": "3", "operating_point": point,
                 "metric": metric, "estimate": "0.02", "ci_low": "-0.01", "ci_high": "0.04",
                 "valid_replicates": "200", "unit": "leakage_group"}
                for a, b, _ in M.KEY_COMPARISONS
                for point, metric in (("raw_common_population", "average_precision"),
                                      ("raw_common_population", "brier"),
                                      ("trial_FA_0.10", "event_recall"),
                                      ("trial_FA_0.10", "trial_false_alarm_rate"),
                                      ("trial_FA_0.10", "mean_lead_frames"))]

    def test_sample_standard_deviation_uses_ddof_one(self):
        mean, sample_sd, count = M.summarize([1.0, 2.0, 3.0])
        self.assertEqual(mean, 2.0)
        self.assertEqual(sample_sd, 1.0)
        self.assertEqual(count, 3)

    def test_reporting_audit_source_set_includes_entry_and_reproduce(self):
        sources = set(M.reporting_source_paths())
        self.assertIn((M.HERE / "report.py").resolve(), sources)
        self.assertIn((M.HERE / "REPRODUCE.md").resolve(), sources)
        self.assertTrue(all(path.is_file() for path in sources))

    def test_raw_table_keeps_all_three_seeds(self):
        rows = []
        for group in M.GROUP_ORDER:
            for horizon in (1, 3, 5):
                for offset, seed in enumerate((20260914, 20260915, 20260916)):
                    rows.append({"method_type": "neural_operational", "population": "primary", "rule": "raw",
                                 "operating_point": "fixed_0.5", "threshold_status": "available", "group": group,
                                 "horizon": str(horizon), "seed": str(seed), "average_precision": str(0.1 + offset * 0.1), "brier": "0.2"})
        table = M.raw_table(rows)
        self.assertEqual(len(table), 24)
        self.assertTrue(all(row["average_precision_n"] == 3 for row in table))
        self.assertTrue(all(math.isclose(row["average_precision_sample_sd"], 0.1) for row in table))

    def test_paired_interval_direction_does_not_call_crossing_zero_supported(self):
        source = self.paired_grid()
        result = M.paired_table(source)
        self.assertEqual(result[0]["direction"], "interval_crosses_zero")
        sentence = M.comparison_sentence(result, "C_xyz_delta_vs_B_xyz", 3, "raw_common_population", "average_precision")
        self.assertIn("证据不足", sentence)

    def test_paired_grid_rejects_one_missing_row(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            M.paired_table(self.paired_grid()[:-1])

    def test_paired_unavailable_is_not_zero_or_direction_evidence(self):
        source = self.paired_grid()
        source[0].update({"status": "unavailable_incomplete_seed_grid", "estimate": "", "ci_low": "", "ci_high": "",
                          "valid_replicates": "", "unit": "", "candidate_unavailable_seeds": "20260914",
                          "comparator_unavailable_seeds": ""})
        result = M.paired_table(source)
        row = next(item for item in result if item["direction"] == "unavailable_incomplete_seed_grid")
        self.assertTrue(math.isnan(row["estimate"]))
        self.assertIn("不可用", M.comparison_sentence(result, row["comparison"], row["horizon"], row["operating_point"], row["metric"]))

    def test_operational_table_requires_selected_rule(self):
        rows = []
        for group in M.GROUP_ORDER:
            for seed in (20260914, 20260915, 20260916):
                base = {"method_type": "neural_operational", "population": "primary", "horizon": "3",
                        "operating_point": "trial_FA_0.10", "threshold_status": "available", "group": group,
                        "seed": str(seed), "frame_fpr": "0.01", "frame_recall": "0.5",
                        "trial_false_alarm_rate": "0.1", "event_recall": "0.6", "mean_lead_frames": "2",
                        "lead_ge_3_recall": "0.3", "lead_ge_5_recall": "0.0"}
                rows.append({**base, "rule_selected": "True", "rule": "raw"})
                rows.append({**base, "rule_selected": "False", "rule": "confirm2", "event_recall": "0.1"})
        table = M.operational_table(rows, 3, "trial_FA_0.10")
        self.assertEqual(len(table), 8)
        self.assertTrue(all(math.isclose(row["event_recall_mean"], 0.6) for row in table))

    def test_operational_table_preserves_unavailable_seed_grid(self):
        rows = []
        for group in M.GROUP_ORDER:
            for seed in M.SEEDS:
                unavailable = group == "B_xyz"
                rows.append({"method_type": "neural_operational", "population": "primary", "horizon": "3",
                             "operating_point": "trial_FA_0.10", "threshold_status": "unavailable" if unavailable else "available",
                             "group": group, "seed": str(seed), "rule_selected": "True", "rule": "raw",
                             "frame_fpr": "" if unavailable else "0.01", "frame_recall": "" if unavailable else "0.5",
                             "trial_false_alarm_rate": "" if unavailable else "0.1", "event_recall": "" if unavailable else "0.6",
                             "mean_lead_frames": "" if unavailable else "2"})
        table = M.operational_table(rows, 3, "trial_FA_0.10")
        b = next(row for row in table if row["group"] == "B_xyz")
        self.assertEqual(b["threshold_status"], "unavailable_incomplete_seed_grid")
        self.assertTrue(math.isnan(b["event_recall_mean"]))

    def test_raw_table_rejects_duplicate_seed_rows(self):
        rows = []
        for group in M.GROUP_ORDER:
            for horizon in M.HORIZONS:
                for seed in M.SEEDS:
                    rows.append({"method_type": "neural_operational", "population": "primary", "rule": "raw",
                                 "operating_point": "fixed_0.5", "threshold_status": "available", "group": group,
                                 "horizon": str(horizon), "seed": str(seed), "average_precision": "0.5", "brier": "0.2"})
        rows.append(dict(rows[0]))
        with self.assertRaisesRegex(ValueError, "grid mismatch"):
            M.raw_table(rows)

    def test_checkpoint_grid_is_exact(self):
        rows = [{"group": group, "seed": str(seed), "kind": kind}
                for group in M.GROUP_ORDER for seed in M.SEEDS for kind in ("best", "latest", "config")]
        M.require_checkpoint_grid(rows)
        with self.assertRaisesRegex(ValueError, "exactly"):
            M.require_checkpoint_grid(rows[:-1])

    def test_e2e_table_preserves_both_deployment_modes(self):
        item = {"group": "B_xyz", "seed": 20260914, "head_parameters": 10, "encoder_parameters": 20,
                "upstream_parameters": 30, "measurement": {
                    "cold_history": [{"samples_ms": [9.0, 11.0], "peak_allocated_bytes": 100}],
                    "streaming": [{"samples_ms": [1.0, 3.0], "peak_allocated_bytes": 50}]}}
        rows = M.e2e_table([item])
        self.assertEqual({row["mode"] for row in rows}, {"cold_history", "streaming"})
        streaming = next(row for row in rows if row["mode"] == "streaming")
        self.assertEqual(streaming["latency_ms_mean"], 2.0)
        self.assertEqual(streaming["latency_ms_sample_sd"], math.sqrt(2.0))

    def test_gate_table_keeps_all_seeds_and_deltas(self):
        rows = [{"group": group, "seed": str(seed), "horizon": "3", "population": "primary",
                 "operating_point": "trial_FA_0.10", "delta_raw_average_precision": "-0.1",
                 "delta_frame_recall": "-0.2", "delta_trial_false_alarm_rate": "-0.3",
                 "delta_event_recall": "-0.4", "delta_mean_lead_frames": "-0.5"}
                for group in M.GROUP_ORDER for seed in M.SEEDS]
        table = M.gate_table(rows, 3, "trial_FA_0.10")
        self.assertEqual(len(table), 8)
        self.assertTrue(all(row["delta_event_recall_n"] == 3 for row in table))

    def test_head_only_table_binds_checkpoint_and_freeze(self):
        checkpoints = [{"group": group, "seed": str(M.SEEDS[0]), "kind": "best", "sha256": group}
                       for group in M.GROUP_ORDER]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction, index, inventory = (root / name for name in ("prediction.json", "index.csv", "inventory.json"))
            for path in (prediction, index, inventory):
                path.write_text(path.name)
            artifact = {"schema": "round8_incremental_head_cost_v1", "status": "complete",
                        "accepted_manifest_sha256": M.sha256(prediction), "checkpoint_index_sha256": M.sha256(index),
                        "inventory_sha256": M.sha256(inventory),
                        "protocol_sha256": M.sha256(M.ROUND_ROOT / "NUMERIC_PROTOCOL.json"),
                        "source_sha256": M.sha256(M.ROUND_ROOT / "benchmark_head.py"), "rows": [
                {"group": group, "seed": M.SEEDS[0], "parameters": 10, "checkpoint_sha256": group,
                 "head_frozen": True, "measurement": [{"samples_ms": [1.0, 3.0], "peak_allocated_bytes": 20}]}
                for group in M.GROUP_ORDER]}
            result = M.head_only_table(artifact, checkpoints, prediction, index, inventory)
            self.assertEqual(len(result), 8)
            self.assertTrue(all(row["latency_ms_mean"] == 2.0 for row in result))

    def test_state_ci_rejects_one_missing_row(self):
        rows = [{"comparison": f"P4_state_vs_{comparator}", "step": str(step), "metric": "rmse_xyz"}
                for comparator in ("persistence", "fit_only_ridge_linear") for step in range(1, 6)]
        M.require_state_ci(rows)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            M.require_state_ci(rows[:-1])

    def test_baseline_grid_rejects_one_missing_row(self):
        names = ("current_p_slip_raw", "history9_p_slip_mean_raw", "history9_p_slip_slope_fit",
                 "latest_force_delta_fit", "elapsed_position_fit", "fit_prevalence")
        rows = [{"method_type": "fit_only_baseline", "population": "primary", "group": name, "horizon": "3",
                 "operating_point": "trial_FA_0.10", "rule_selected": "True", "threshold_status": "available"}
                for name in names]
        self.assertEqual(len(M.baseline_table(rows, 3, "trial_FA_0.10")), 6)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            M.baseline_table(rows[:-1], 3, "trial_FA_0.10")

    def test_intervention_grid_rejects_one_missing_row(self):
        rows = [{"intervention": kind, "group": group, "seed": str(seed), "horizon": "3",
                 "population": "primary", "operating_point": "trial_FA_0.10",
                 "mean_score_delta_vs_unperturbed": "0", "trial_false_alarm_rate": "0.1",
                 "event_recall": "0.5", "mean_lead_frames": "2"}
                for kind, group in ({(kind, group) for group in M.GROUP_ORDER
                                     for kind in ("force_input_fit_mean_zero", "force_causal_lag1")}
                                    | {("state_input_fit_mean_zero", "P4_state")}) for seed in M.SEEDS]
        self.assertEqual(len(M.intervention_table(rows, 3)), 17)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            M.intervention_table(rows[:-1], 3)

    def test_state_diagnostic_requires_two_population_seed_step_grid_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); prediction = root / "prediction.json"; evaluation = root / "summary.json"
            prediction.write_text("prediction"); evaluation.write_text("evaluation")
            fields = ("prediction_temporal_variance_xyz_trial_mean", "target_temporal_variance_xyz_trial_mean",
                      "persistence_temporal_variance_xyz_trial_mean", "prediction_target_mse_xyz_trial_mean",
                      "target_vs_current_mse_xyz_trial_mean", "linear_target_mse_xyz_trial_mean",
                      "prediction_vs_current_mse_xyz_trial_mean")
            rows = [{"population": population, "seed": seed, "step": step, **{field: 1.0 for field in fields}}
                    for population in ("primary_common", "full_timeline") for seed in M.SEEDS for step in range(1, 6)]
            summary = root / "P4_STATE_SEED_STEP_SUMMARY.csv"
            with summary.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            per_trial = root / "P4_STATE_PER_TRIAL_TEMPORAL_DIAGNOSTICS.csv"; per_trial.write_text("x\n1\n")
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            audit = {"schema": "round8_p4_state_temporal_diagnostic_v1", "status": "complete", "test_role_consumed": False,
                     "source_hashes": {str(prediction.resolve()): digest(prediction), str(evaluation.resolve()): digest(evaluation)},
                     "output_hashes": {summary.name: digest(summary), per_trial.name: digest(per_trial)}}
            (root / "STATE_DIAGNOSTIC_AUDIT.json").write_text(json.dumps(audit))
            self.assertEqual(len(M.require_state_diagnostics(root, prediction, evaluation)), 10)
            rows.pop()
            with summary.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            audit["output_hashes"][summary.name] = digest(summary); (root / "STATE_DIAGNOSTIC_AUDIT.json").write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError, "grid incomplete"):
                M.require_state_diagnostics(root, prediction, evaluation)


if __name__ == "__main__":
    unittest.main()
