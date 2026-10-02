from __future__ import annotations

import unittest

import numpy as np

from support_audit import HISTORY, HORIZONS, PRIMARY_RULE, PROXY_RULE, aggregate, choose_task, support_for_episode


class SupportAuditTests(unittest.TestCase):
    def test_complete_window_and_first_gross_boundary(self):
        labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2], dtype=np.int8)
        static = support_for_episode(labels, 3, 0)
        self.assertEqual(static["eligible_frames"], 3)
        self.assertEqual(static["positive_frames"], 1)
        self.assertEqual(static["negative_frames"], 2)
        incipient = support_for_episode(labels, 1, 1)
        self.assertEqual((incipient["positive_frames"], incipient["negative_frames"]), (1, 1))

    def test_incomplete_causal_history_is_excluded(self):
        labels = np.array([0, 0, 0, 0, 0, 2, 2], dtype=np.int8)
        row = support_for_episode(labels, 3, 0)
        self.assertEqual(HISTORY, 4)
        self.assertEqual((row["eligible_frames"], row["positive_frames"], row["negative_frames"]), (1, 1, 0))

    def test_post_gross_recovery_is_excluded(self):
        labels = np.array([0, 0, 0, 0, 1, 2, 2, 0, 0], dtype=np.int8)
        row = support_for_episode(labels, 1, 0)
        self.assertEqual(row["eligible_frames"], 1)
        self.assertEqual((row["positive_frames"], row["negative_frames"]), (0, 1))

    def test_no_gross_episode_contributes_negatives(self):
        labels = np.array([0, 0, 0, 0, 0, 1, 1], dtype=np.int8)
        static = support_for_episode(labels, 1, 0)
        incipient = support_for_episode(labels, 1, 1)
        self.assertEqual((static["positive_frames"], static["negative_frames"]), (0, 2))
        self.assertEqual((incipient["positive_frames"], incipient["negative_frames"]), (0, 1))

    def test_primary_precedes_proxy_and_shortest_supported_wins(self):
        unsupported_primary = {str(h): {"supported": False} for h in HORIZONS}
        proxy = {str(h): {"supported": h >= 3} for h in HORIZONS}
        choice = choose_task(unsupported_primary, proxy)
        self.assertEqual((choice["task"], choice["horizon"]), ("current_incipient_to_first_gross_proxy", 3))
        primary = {str(h): {"supported": h >= 8} for h in HORIZONS}
        choice = choose_task(primary, proxy)
        self.assertEqual((choice["task"], choice["horizon"]), ("current_static_to_first_gross", 8))

    def test_aggregate_counts_episode_support(self):
        episodes = []
        for index in range(10):
            episodes.append({"static": {"1": {"eligible_frames": 15, "positive_frames": 2 if index < 5 else 0,
                                                   "negative_frames": 13 if index < 5 else 15,
                                                   "positive_episode": index < 5, "negative_episode": True}}})
        result = aggregate(episodes, "static", 1, PRIMARY_RULE)
        self.assertEqual(result["positive_frames"], 10)
        self.assertEqual(result["negative_episodes"], 10)
        self.assertFalse(result["supported"])


if __name__ == "__main__":
    unittest.main()
