from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluate import (
    average_precision,
    binary_metrics,
    bootstrap_by_episode,
    main,
    select_calibration_threshold,
    validate_role,
)


class MetricTests(unittest.TestCase):
    def test_exact_threshold_and_never_alarm_tie_break(self):
        selected = select_calibration_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
        self.assertEqual(selected["threshold"], 0.8)
        self.assertFalse(selected["never_alarm"])
        tied = select_calibration_threshold(np.array([0, 1]), np.array([0.5, 0.5]))
        self.assertGreater(tied["threshold"], 1.0)
        self.assertTrue(tied["never_alarm"])
        self.assertEqual(tied["static_fpr"], 0.0)

    def test_metrics_tied_ap_and_one_class(self):
        row = binary_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), 0.8)
        self.assertEqual(row["confusion"], {"tn": 2, "fp": 0, "fn": 0, "tp": 2})
        self.assertEqual(row["balanced_accuracy"], 1.0)
        self.assertAlmostEqual(average_precision(np.array([1, 0]), np.array([0.5, 0.5])), 0.5)
        one = binary_metrics(np.zeros(3), np.zeros(3), 0.5)
        self.assertFalse(one["evaluable"])
        self.assertIsNone(one["balanced_accuracy"])

    def test_episode_bootstrap_is_reproducible(self):
        y = np.array([0, 1, 0, 1, 0, 1])
        score = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
        episode = np.array(["a", "a", "b", "b", "c", "c"])
        first = bootstrap_by_episode(y, score, episode, 0.5, reps=20, seed=7)
        second = bootstrap_by_episode(y, score, episode, 0.5, reps=20, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["requested_replicates"], 20)

    def test_role_validation_rejects_missing_frame_and_wrong_label(self):
        split = {"calibration": ["e"]}
        metadata = {"e": {"task": "slip", "frames": 2}}
        authority = {"e": np.array([0, 2], dtype=np.int8)}
        with self.assertRaisesRegex(ValueError, "incomplete/noncontiguous frames"):
            validate_role(
                [{"episode_id": "e", "t": 0, "stage": 0}], split, "calibration", Path("x.csv"), metadata, authority
            )
        with self.assertRaisesRegex(ValueError, "stage label mismatch"):
            validate_role(
                [{"episode_id": "e", "t": 0, "stage": 0}, {"episode_id": "e", "t": 1, "stage": 1}],
                split, "calibration", Path("x.csv"), metadata, authority,
            )

    def test_role_validation_rejects_missing_episode(self):
        split = {"validation": ["e1", "e2"]}
        metadata = {"e1": {"task": "slip"}, "e2": {"task": "slip"}}
        with self.assertRaisesRegex(ValueError, "incomplete labeled role"):
            validate_role(
                [{"episode_id": "e1", "t": 0, "stage": 0}], split, "validation", Path("x.csv"), metadata
            )


class EndToEndTests(unittest.TestCase):
    @staticmethod
    def write_predictions(path: Path, fold: str, seed: int, offset: float = 0.0) -> None:
        fields = ["episode_id", "t", "stage", "p_slip", "p_static", "p_gross", "fold", "seed", "init", "checkpoint_epoch"]
        episodes = {
            "cal_static": [(0, 0.1 + offset), (0, 0.2 + offset)],
            "cal_gross": [(2, 0.8 + offset), (2, 0.9 + offset)],
            "val_static": [(0, 0.1 + offset), (0, 0.3 + offset)],
            "val_gross": [(2, 0.7 + offset), (2, 0.9 + offset)],
            "val_incipient": [(1, 0.4 + offset), (1, 0.6 + offset)],
        }
        selected = episodes if "all" in path.name else {
            key: value for key, value in episodes.items()
            if key.startswith("cal_") == ("calibration" in path.name)
        }
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for episode_id, rows in selected.items():
                for t, (stage, score) in enumerate(rows):
                    writer.writerow({"episode_id": episode_id, "t": t, "stage": stage, "p_slip": score,
                                     "p_static": 1-score, "p_gross": score, "fold": fold, "seed": seed,
                                     "init": "fresh", "checkpoint_epoch": 2})

    def test_partial_runner_outputs_reports_without_reading_test(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fold = "htt_leave_p1"
            episodes = [
                {"id": "cal_static", "task": "slip", "leakage_group": "cal_static"},
                {"id": "cal_gross", "task": "slip", "leakage_group": "cal_gross"},
                {"id": "val_static", "task": "slip", "leakage_group": "val_static"},
                {"id": "val_gross", "task": "slip", "leakage_group": "val_gross"},
                {"id": "val_incipient", "task": "slip", "leakage_group": "val_incipient"},
                {"id": "never_read_test", "task": "slip", "leakage_group": "never_read_test"},
            ]
            split = {"schema_version": 2, "episodes": episodes, "splits": {fold: {
                "train": [], "calibration": ["cal_static", "cal_gross"],
                "validation": ["val_static", "val_gross", "val_incipient"], "test": ["never_read_test"]}}}
            split_path = root / "splits.json"
            split_path.write_text(json.dumps(split), encoding="utf-8")
            cal, val = root / "predictions_calibration.csv", root / "predictions_validation.csv"
            self.write_predictions(cal, fold, 20260914)
            self.write_predictions(val, fold, 20260914)
            manifest = root / "runs.json"
            manifest.write_text(json.dumps({"runs": [{"variant": "B", "fold": fold, "seed": 20260914,
                "predictions_calibration": cal.name, "predictions_validation": val.name}]}), encoding="utf-8")
            output = root / "evaluation"
            old_argv = sys.argv
            try:
                sys.argv = ["evaluate.py", "--splits", str(split_path), "--runs-manifest", str(manifest),
                            "--output", str(output), "--allow-partial", "--allow-noncanonical-splits",
                            "--bootstrap-reps", "20"]
                main()
            finally:
                sys.argv = old_argv
            complete = json.loads((output / "COMPLETE.json").read_text())
            evaluation = json.loads((output / "evaluation.json").read_text())
            self.assertEqual(complete["status"], "complete")
            self.assertFalse(evaluation["test_partitions_read"])
            self.assertEqual(evaluation["runs"][0]["validation"]["fixed_0.5"]["balanced_accuracy"], 1.0)
            self.assertEqual(evaluation["runs"][0]["validation_incipient_diagnostic"]["count"], 2)
            self.assertTrue((output / "REPORT_ZH.md").exists())
            self.assertTrue((output / "plots" / "comparison_calibrated.svg").exists())

    def test_role_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fold = "htt_leave_p1"
            split = {"schema_version": 2, "episodes": [], "splits": {fold: {
                "train": [], "calibration": ["allowed"], "validation": ["val"], "test": ["forbidden"]}}}
            split_path = root / "splits.json"
            split_path.write_text(json.dumps(split), encoding="utf-8")
            fields = ["episode_id", "t", "stage", "p_slip", "fold", "seed", "init"]
            paths = []
            for name, episode in (("calibration", "forbidden"), ("validation", "val")):
                path = root / f"predictions_{name}.csv"
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
                    writer.writerow({"episode_id": episode, "t": 0, "stage": 0, "p_slip": 0.1,
                                     "fold": fold, "seed": 20260914, "init": "fresh"})
                paths.append(path)
            manifest = root / "runs.json"
            manifest.write_text(json.dumps({"runs": [{"variant": "B", "fold": fold, "seed": 20260914,
                "predictions_calibration": str(paths[0]), "predictions_validation": str(paths[1])}]}), encoding="utf-8")
            old_argv = sys.argv
            try:
                sys.argv = ["evaluate.py", "--splits", str(split_path), "--runs-manifest", str(manifest),
                            "--output", str(root / "out"), "--allow-partial", "--allow-noncanonical-splits"]
                with self.assertRaisesRegex(ValueError, "out-of-role"):
                    main()
            finally:
                sys.argv = old_argv


if __name__ == "__main__":
    unittest.main()
