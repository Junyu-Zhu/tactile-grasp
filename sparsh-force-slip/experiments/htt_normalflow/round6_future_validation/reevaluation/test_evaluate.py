#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("round6_reevaluation", HERE / "evaluate.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


class ReevaluationTests(unittest.TestCase):
    def test_calibration_tie_prefers_lower_fpr_then_higher_threshold(self):
        rows = [
            {"target": 0, "score": 0.1},
            {"target": 0, "score": 0.2},
            {"target": 1, "score": 0.8},
            {"target": 1, "score": 0.9},
        ]
        self.assertEqual(ev.calibrate(rows, "score"), 0.8)

    def test_fast_calibration_matches_bruteforce_on_discrete_scores(self):
        rng = np.random.default_rng(20260915)
        for _ in range(20):
            y = np.r_[np.zeros(20, dtype=int), np.ones(20, dtype=int)]
            rng.shuffle(y)
            s = rng.choice([0.1, 0.2, 0.5, 0.8, 0.9], size=len(y))
            rows = [{"target": int(a), "score": float(b)} for a, b in zip(y, s)]
            candidates = sorted(set(map(float, s))) + [np.nextafter(1.0, np.inf)]
            brute = max(((t, ev.confusion(y, s, t)) for t in candidates), key=lambda x: (x[1]["balanced_accuracy"], -x[1]["fpr"], x[0]))[0]
            self.assertEqual(ev.calibrate(rows, "score"), brute)

    def test_metrics_reports_single_class_unavailable(self):
        result = ev.metrics([{"target": 0, "score": 0.2}], "score", 0.5)
        self.assertEqual(result["status"], "unavailable")

    def test_position_uses_train_fixed_scale(self):
        train = [
            {"t": 1, "target": 0}, {"t": 2, "target": 0},
            {"t": 9, "target": 1}, {"t": 10, "target": 1},
        ]
        a = ev.position_fit(train, [{"t": 20, "target": 0}])[0]
        b = ev.position_fit(train, [{"t": 1000, "target": 0}])[0]
        self.assertAlmostEqual(float(a), float(b), places=12)

    def test_simple_scores_cover_fit_rows_and_targets(self):
        fit = [
            {"t": 1, "target": 0, "history": [0.1] * 4, "force_change_magnitude": 0.1},
            {"t": 9, "target": 1, "history": [0.2, 0.3, 0.4, 0.5], "force_change_magnitude": 0.9},
        ]
        outer = [{"t": 4, "target": 0, "history": [0.2] * 4, "force_change_magnitude": 0.2}]
        ev.add_simple_scores(fit, outer)
        self.assertTrue(all("current_slip" in row for row in fit + outer))
        self.assertEqual(fit[1]["history_trend"], 0.8)

    def test_source_rows_uses_raw_frame_identity_for_current_state(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            metadata = [{"dataset": "d_train", "trajectory": "0", "sample": i} for i in range(16)]
            current = torch.zeros(16)
            current[0] = 1  # catches the old cache-index/raw-index bug
            raw = {
                "metadata": metadata,
                "force_pred_n": torch.arange(48, dtype=torch.float32).reshape(16, 3),
                "slip_probs": torch.stack((torch.ones(16), torch.linspace(0, 1, 16)), 1),
                "current_slip": current,
            }
            cache = {
                "metadata": [metadata[10]],
                "horizons": [1, 3, 5],
                "future_slip": torch.tensor([[1, 1, 1]], dtype=torch.float32),
            }
            raw_path, cache_path = td / "raw.pt", td / "cache.pt"
            torch.save(raw, raw_path); torch.save(cache, cache_path)
            support = {("source/d_train/0", 10): {"role": "outer", "target": 1, "onset": 11}}
            _, rows = ev.source_rows(cache_path, raw_path, 1, support)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["raw_index"], 10)
            self.assertEqual(rows[0]["current_slip_label"], 0)
            np.testing.assert_allclose(rows[0]["history"], [7/15, 8/15, 9/15, 10/15], rtol=0, atol=1e-6)

    def test_support_endpoint_roles_and_counts(self):
        manifest = {
            "status": "complete",
            "horizons": {"1": {"roles": {
                "fit_train": {"summary": {"positive_endpoints": 1, "negative_endpoints": 1}, "trials": [
                    {"episode_id": "source/d/0", "first_current_slip_t": 5, "positive_endpoints": [4], "negative_endpoints": [3]}
                ]},
                "selection": {"summary": {"positive_endpoints": 0, "negative_endpoints": 0}, "trials": []},
                "calibration": {"summary": {"positive_endpoints": 0, "negative_endpoints": 0}, "trials": []},
                "outer": {"summary": {"positive_endpoints": 0, "negative_endpoints": 0}, "trials": []},
            }}},
        }
        endpoints, onsets = ev.support_endpoints(manifest, "source", 1)
        self.assertEqual(endpoints[("source/d/0", 4)]["target"], 1)
        self.assertEqual(endpoints[("source/d/0", 3)]["role"], "fit_train")
        self.assertEqual(onsets["source/d/0"], 5)

    def test_event_late_is_diagnostic_not_detected(self):
        rows = [
            {"episode_id": "e", "cluster": "g", "t": 10, "onset": 12, "eligible": True, "target": 1, "score": 0.1},
            {"episode_id": "e", "cluster": "g", "t": 12, "onset": 12, "eligible": False, "target": None, "score": 0.9},
        ]
        result = ev.event_records(rows, "score", 0.5, 8)[0]
        self.assertEqual(result["status"], "late")
        self.assertFalse(result["event_detected"])
        self.assertIsNone(result["lead_frames"])

    def test_alarm_runs_reset_across_missing_times(self):
        self.assertEqual(ev.runs_count([True, True, True], [1, 2, 4]), 2)

    def test_cluster_bootstrap_keeps_complete_groups(self):
        rows = [
            {"cluster": "a", "target": 0, "score": 0.1}, {"cluster": "a", "target": 1, "score": 0.8},
            {"cluster": "b", "target": 0, "score": 0.2}, {"cluster": "b", "target": 1, "score": 0.9},
        ]
        result = ev.cluster_bootstrap(rows, "score", 0.5, reps=20)
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["average_precision"]["valid_replicates"], 20)


if __name__ == "__main__":
    unittest.main(verbosity=2)
