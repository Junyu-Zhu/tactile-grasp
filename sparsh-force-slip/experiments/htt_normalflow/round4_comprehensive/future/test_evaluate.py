from __future__ import annotations

import unittest

import numpy as np

from evaluate import average_precision, calibrate, event_metrics


class FutureEvaluateTests(unittest.TestCase):
    def test_max_ba_tie_prefers_lower_fpr_then_larger_threshold(self):
        y = np.array([0, 0, 1, 1], dtype=np.int8)
        scores = np.array([0.1, 0.4, 0.4, 0.9])
        chosen = calibrate(y, scores)["max_balanced_accuracy"]
        self.assertEqual(chosen["threshold"], 0.9)
        self.assertEqual(chosen["calibration"]["fpr"], 0.0)

    def test_fpr_cap_can_choose_explicit_never_alarm(self):
        y = np.array([0, 1], dtype=np.int8)
        scores = np.array([0.9, 0.2])
        chosen = calibrate(y, scores)["fpr_1pct"]
        self.assertGreater(chosen["threshold"], 1.0)
        self.assertEqual(chosen["calibration"]["tp"], 0)

    def test_never_alarm_stays_never_on_larger_validation_score(self):
        y = np.array([0, 1], dtype=np.int8)
        threshold = calibrate(y, np.array([0.9, 0.2]))["fpr_1pct"]["threshold"]
        self.assertFalse(np.any(np.array([0.95, 1.0]) >= threshold))

    def test_average_precision_respects_score_ties(self):
        y = np.array([1, 0, 1], dtype=np.int8)
        scores = np.array([0.5, 0.5, 0.2])
        self.assertAlmostEqual(average_precision(y, scores), (0.5 * 0.5) + (0.5 * 2 / 3))

    def test_event_hit_uses_only_pre_onset_positive_window_and_late_is_separate(self):
        rows = []
        for t, stage, eligible, target, score in [(3, 0, 1, 1, .2), (4, 0, 1, 1, .8),
                                                   (5, 1, 0, None, .1), (6, 2, 0, None, .9)]:
            rows.append({"episode_id": "e1", "t": t, "stage": stage, "first_gross_index": 6,
                         "eligible_primary": eligible, "target_first_gross_h8": target, "p_future": score})
        hit = event_metrics(rows, "p_future", .7)
        self.assertEqual((hit["hits"], hit["lead_frames_mean_hits"], hit["late_detections_among_misses"]), (1, 2.0, 0))
        late = event_metrics(rows, "p_future", .85)
        self.assertEqual((late["misses"], late["late_detections_among_misses"]), (1, 1))


if __name__ == "__main__":
    unittest.main()
