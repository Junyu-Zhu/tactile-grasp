from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from evaluate import future_monotonicity, load_cache, main, require_complete_cache
from render_report import render_reports
from metrics import (
    average_precision,
    binary_metrics,
    event_metrics,
    future_event_support,
    rising_edges,
    select_balanced_threshold,
    select_fpr_threshold,
)


class MetricTests(unittest.TestCase):
    def test_binary_metrics_and_tied_average_precision(self):
        y = np.array([0, 1, 0, 1])
        score = np.array([0.1, 0.9, 0.2, 0.8])
        row = binary_metrics(y, score, 0.5)
        self.assertEqual(row["confusion"], {"tn": 2, "fp": 0, "fn": 0, "tp": 2})
        self.assertEqual(row["macro_f1"], 1.0)
        self.assertAlmostEqual(average_precision(np.array([1, 0]), np.array([0.5, 0.5])), 0.5)
        one_class = binary_metrics(np.zeros(3), np.zeros(3), 0.5)
        self.assertIsNone(one_class["macro_f1"])
        self.assertEqual(one_class["negative"]["recall"], 1.0)
        self.assertEqual(one_class["brier"], 0.0)
        self.assertIsNone(one_class["positive"])

    def test_fixed_threshold_selection_and_never_alarm_candidate(self):
        y = np.array([0, 0, 1, 1])
        score = np.array([0.1, 0.2, 0.8, 0.9])
        self.assertEqual(select_balanced_threshold(y, score)["threshold"], 0.8)
        chosen = select_fpr_threshold(y, score, max_fpr=0.01)
        self.assertEqual(chosen["threshold"], 0.8)
        self.assertEqual(select_fpr_threshold(np.zeros(4), score)["threshold"], None)

    def test_rising_edge_cooldown(self):
        active = np.array([0, 1, 0, 1, 0, 0, 0, 1], dtype=bool)
        np.testing.assert_array_equal(rising_edges(active, cooldown=5), np.array([1, 7]))

    def test_future_monotonicity(self):
        episodes = [{"id": "x", "p_future": np.array([[0.1, 0.2, 0.3], [0.4, 0.3, 0.5], [np.nan] * 3])}]
        row = future_monotonicity(episodes)
        self.assertEqual(row["valid_frames"], 2)
        self.assertEqual(row["violating_frames"], 1)
        self.assertEqual(row["violation_fraction"], 0.5)

    def test_metric_entry_rejects_partial_cache(self):
        with self.assertRaisesRegex(ValueError, "status must be complete"):
            require_complete_cache({"status": "running", "expected_episodes": 1, "episodes": [{}]})
        with self.assertRaisesRegex(ValueError, "inventory"):
            require_complete_cache({"status": "complete", "expected_episodes": 2, "episodes": [{}]})

    def test_event_hit_late_miss_and_empty(self):
        labels = np.array([0, 0, 0, 0, 2, 2])
        hit = {"id": "hit", "labels": labels, "score": np.array([0, 0, 0, 1, 0, 0])}
        late = {"id": "late", "labels": labels, "score": np.array([0, 0, 0, 0, 1, 0])}
        miss = {"id": "miss", "labels": labels, "score": np.zeros(6)}
        row = event_metrics([hit, late, miss], "score", 0.5, horizon=1, population=0)
        self.assertEqual((row["hits"], row["late"], row["misses"]), (1, 1, 1))
        self.assertEqual(row["lead_steps"]["median"], 1.0)
        empty = event_metrics([], "score", 0.5, horizon=1, population=0)
        self.assertIsNone(empty["event_recall"])

    def test_event_without_population_frame_in_horizon_is_not_denominator(self):
        episode = {"id": "incipient_only", "labels": np.array([0, 1, 2, 2]), "score": np.ones(4)}
        row = event_metrics([episode], "score", 0.5, horizon=1, population=0)
        self.assertEqual(row["episodes_with_first_gross_onset"], 1)
        self.assertEqual(row["support_events"], 0)
        self.assertEqual(row["unsupported_events_no_population_frame_in_horizon"], 1)
        self.assertIsNone(row["event_recall"])

    def test_too_early_persistent_alarm_is_sequence_false_alarm(self):
        episode = {"id": "early", "labels": np.array([0, 0, 1, 1, 2, 2]), "score": np.array([0, 1, 1, 1, 0, 0])}
        row = event_metrics([episode], "score", 0.5, horizon=1, population=1)
        self.assertEqual(row["hits"], 0)
        self.assertEqual(row["misses"], 1)
        self.assertEqual(row["population_conditioned_false_alarm_edges"], 0)
        self.assertEqual(row["sequence_false_alarm_edges_before_warning_window"], 1)

    def test_recovery_alarm_outside_late_window_is_miss(self):
        episode = {"id": "far_late", "labels": np.array([0, 0, 0, 2, 2, 0, 0, 0]), "score": np.array([0, 0, 0, 0, 0, 0, 1, 0])}
        row = event_metrics([episode], "score", 0.5, horizon=1, population=0)
        self.assertEqual(row["late"], 0)
        self.assertEqual(row["misses"], 1)

    def test_near_tail_onset_gets_no_truncated_horizon_credit(self):
        episode = {"id": "tail", "labels": np.array([0, 0, 0, 0, 2]), "score": np.array([0, 0, 0, 1, 1])}
        row = event_metrics([episode], "score", 0.5, horizon=3, population=0)
        # Only t=1 has a complete three-step target ending at the onset; the
        # alarm starts at truncated t=3 and must not receive retrospective credit.
        self.assertEqual(row["support_events"], 1)
        self.assertEqual(row["hits"], 0)
        self.assertEqual(row["misses"], 1)
        self.assertEqual(row["lead_steps"]["count"], 0)
        self.assertIsNone(row["lead_steps"]["max"])

        incipient_only_truncated = {
            "id": "tail_incipient", "labels": np.array([0, 0, 0, 1, 2]), "score": np.ones(5)
        }
        support = future_event_support([incipient_only_truncated], horizon=3, population=1, score_key="score")
        self.assertEqual(support["support_events"], 0)
        self.assertEqual(support["unsupported_events_no_population_frame_in_horizon"], 1)

        no_gross = {"id": "no_gross_tail", "labels": np.zeros(5, dtype=int), "score": np.array([0, 0, 0, 1, 1])}
        no_gross_row = event_metrics([no_gross], "score", 0.5, horizon=3, population=0)
        self.assertEqual(no_gross_row["population_conditioned_negative_frames"], 2)
        self.assertEqual(no_gross_row["population_conditioned_false_alarm_edges"], 0)
        self.assertEqual(no_gross_row["sequence_false_alarm_edges_before_warning_window"], 0)

        no_complete = {"id": "tail0", "labels": np.array([2]), "score": np.ones(1)}
        row = event_metrics([no_complete], "score", 0.5, horizon=3, population=0)
        self.assertEqual(row["support_events"], 0)
        self.assertIsNone(row["event_recall"])


class EndToEndTests(unittest.TestCase):
    @staticmethod
    def _npz(path: Path, labels: np.ndarray, normalflow: bool = False):
        n = len(labels)
        p_slip = np.linspace(0.01, 0.99, n, dtype=np.float32)
        p_future = np.stack([p_slip, p_slip, p_slip], axis=1)
        pose = np.repeat(np.eye(4, dtype=np.float32)[None], n, axis=0)
        pose[:, 0, 3] = np.arange(n)
        np.savez(
            path,
            z=np.arange(n * 768, dtype=np.float32).reshape(n, 768) / 1000,
            force_pred=np.zeros((n, 3), np.float32),
            p_slip=p_slip,
            p_future=p_future,
            labels=labels,
            pose=pose if normalflow else np.full((n, 4, 4), np.nan, np.float32),
            time=np.arange(n, dtype=np.float64) if normalflow else np.full(n, np.nan),
            frame_index=np.arange(n),
            image_delta_l1=np.r_[np.nan, np.ones(n - 1)],
        )

    def test_full_runner_never_reads_test_partitions(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            labels_event = np.array([0] * 10 + [1, 1] + [2] * 8, np.int8)
            labels_none = np.array([0] * 10 + [1] * 10, np.int8)
            rows = []
            for episode_id, labels, domain, group in (
                ("cal_event", labels_event, "htt", "p2"),
                ("cal_none", labels_none, "htt", "p2"),
                ("val_event", labels_event, "htt", "p3"),
                ("val_none", labels_none, "htt", "p3"),
                ("val_force", np.full(20, -1, np.int8), "htt", "p3"),
                ("nf_train", np.full(8, -1, np.int8), "normalflow", "obj1"),
                ("nf_val", np.full(8, -1, np.int8), "normalflow", "obj2"),
            ):
                path = root / f"{episode_id}.npz"
                self._npz(path, labels, normalflow=domain == "normalflow")
                rows.append({"id": episode_id, "domain": domain, "group": group, "frames": len(labels), "cache_path": str(path), "leakage_group": episode_id})
            split_path = root / "splits.json"
            split = {
                "schema_version": 2,
                "splits": {
                    "htt_leave_p1": {
                        "train": ["uncached_train"],
                        "calibration": ["cal_event", "cal_none"],
                        "validation": ["val_event", "val_none", "val_force"],
                        "test": ["uncached_test"],
                    },
                    "normalflow_objects": {"train": ["nf_train"], "validation": ["nf_val"], "test": ["uncached_nf_test"]},
                },
            }
            split_path.write_text(json.dumps(split), encoding="utf-8")
            index_path = root / "index.json"
            index = {
                "schema_version": 1,
                "status": "complete",
                "expected_episodes": len(rows),
                "horizons": [1, 3, 5],
                "split_manifest_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
                "checkpoint_sha256": "synthetic",
                "preprocessing_fingerprint": "synthetic",
                "episodes": rows,
            }
            index_path.write_text(json.dumps(index), encoding="utf-8")
            _, loaded = load_cache(index_path)
            self.assertEqual(len(loaded), 7)
            output = root / "out"
            import sys
            old_argv = sys.argv
            try:
                sys.argv = ["evaluate.py", "--index", str(index_path), "--splits", str(split_path), "--output", str(output)]
                main()
            finally:
                sys.argv = old_argv
            self.assertTrue((output / "COMPLETE.json").exists())
            provenance = json.loads((output / "provenance.json").read_text())
            self.assertFalse(provenance["test_partitions_read"])
            self.assertIn("train", json.loads((output / "normalflow_diagnostics.json").read_text()))
            future = json.loads((output / "future_warning.json").read_text())
            static_h1 = future["htt_leave_p1"]["H1"]["current_static"]["future_head"]
            self.assertIsNone(static_h1["first_gross_frame_metrics"])
            any_metrics = static_h1["future_any_changed_semantics_diagnostic"]["metrics_at_unselected_0_5"]
            self.assertGreater(any_metrics["support"]["positive"], 0)
            self.assertIsNotNone(any_metrics["average_precision"])
            self.assertFalse(static_h1["descriptive_event_metrics_at_unselected_0_5"]["matched_fpr_operating_point"])
            subgroups = static_h1["subgroups_at_unselected_0_5"]
            self.assertNotIn("val_force", subgroups["by_episode"])
            aggregate_support = static_h1["descriptive_probability_and_one_class_metrics_at_unselected_0_5"]["support"]["total"]
            episode_support = sum(row["first_gross_frame_metrics"]["support"]["total"] for row in subgroups["by_episode"].values())
            self.assertEqual(aggregate_support, episode_support)
            self.assertEqual(subgroups["by_probe"]["p3"]["first_gross_frame_metrics"]["support"]["total"], aggregate_support)
            report_output = root / "report"
            render_reports(output, index_path, split_path, report_output)
            report = (report_output / "REPORT.md").read_text()
            self.assertIn("Future warning", report)
            self.assertIn("ground-truth normalized delta-force", report)
            self.assertIn("translation is in metres", report)
            self.assertIn("future-sensor coordinates into the time-t sensor frame", report)
            self.assertIn("normalflow_experiment/blob/9d1c1dd", report)
            self.assertIn("MoCap-to-sensor extrinsic/calibration chain remains independently unverified", report)
            self.assertIn("validation operating points are reported as observed and are not necessarily matched", report)
            train_table = report.index("| train |")
            validation_table = report.index("| validation |", train_table)
            self.assertNotIn("adjacent image delta", report[train_table:validation_table])
            self.assertIn("[current_detection.json](../metrics/current_detection.json)", report)
            failures = (report_output / "failure_cases.csv").read_text()
            self.assertNotIn("uncached_test", failures)
            self.assertNotIn("nf_train", failures)
            self.assertIn("val_event", failures)
            self.assertIn(",B0,", failures)
            self.assertIn(",B0_cal,", failures)
            future_failures = (report_output / "future_failure_cases.csv").read_text()
            self.assertIn("matched_fold_calibration", future_failures)
            self.assertIn("val_event", future_failures)
            detection_comparison = (report_output / "detection_comparison.csv").read_text()
            self.assertIn("negative_precision", detection_comparison)
            self.assertIn("positive_recall", detection_comparison)
            future_comparison = (report_output / "future_comparison.csv").read_text()
            self.assertIn("event_lead_median_steps", future_comparison)
            self.assertIn("sequence_false_edge_rate_per_negative_frame", future_comparison)
            self.assertIn("calibration_constrained_threshold", future_comparison)
            self.assertIn("validation FPR is reported independently and need not match", future_comparison)


if __name__ == "__main__":
    unittest.main()
