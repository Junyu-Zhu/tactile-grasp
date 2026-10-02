from __future__ import annotations

import unittest

import numpy as np

from audit import alarm_state, gross_events, independently_score, independently_select_rule, independently_select_threshold


class IntegrationAuditTests(unittest.TestCase):
    def test_alarm_state_confirmation_and_release(self):
        alarm, starts = alarm_state(np.array([.6, .6, .45, .39, .6, .6]), .5, 2, .8)
        np.testing.assert_array_equal(alarm, [False, True, True, False, False, True])
        np.testing.assert_array_equal(starts, [False, True, False, False, False, True])

    def test_episode_boundaries_reset_confirmation(self):
        rows = [{"episode": "a", "t": 0, "stage": 0, "score": .6},
                {"episode": "b", "t": 0, "stage": 2, "score": .6}]
        metrics, trials = independently_score(rows, .5, 2, 1.0)
        self.assertEqual((metrics["tp"], metrics["fp"]), (0, 0))
        self.assertEqual(len(trials), 2)

    def test_events_frame_zero_and_delay(self):
        rows = [{"episode": "a", "t": t, "stage": stage, "score": score}
                for t, (stage, score) in enumerate([(2, .1), (2, .8), (0, .1), (2, .9)])]
        metrics, _ = independently_score(rows, .5)
        self.assertEqual(gross_events(np.array([2, 2, 0, 2])), [(0, 2), (3, 4)])
        self.assertEqual(metrics["segments_beginning_at_frame_zero"], 1)
        self.assertEqual((metrics["gross_events"], metrics["gross_events_detected"]), (2, 2))
        self.assertEqual(metrics["mean_detection_delay_frames"], .5)

    def test_threshold_selection_uses_explicit_never_sentinel(self):
        rows = [{"episode": "a", "t": 0, "stage": 0, "score": .9},
                {"episode": "a", "t": 1, "stage": 2, "score": .2}]
        self.assertGreater(independently_select_threshold(rows, "fpr_0.01"), 1.0)

    def test_rule_selection_uses_causal_confirmation(self):
        rows = [{"episode": "a", "t": t, "stage": stage, "score": score}
                for t, (stage, score) in enumerate([(0, .6), (0, .1), (2, .6), (2, .6)])]
        threshold, k, ratio = independently_select_rule(rows, .5, 0.0)
        self.assertEqual((threshold, k, ratio), (.5, 2, 1.0))


if __name__ == "__main__":
    unittest.main()
