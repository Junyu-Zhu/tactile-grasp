from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch

MODULE_PATH = Path(__file__).with_name("prepare.py")
SPEC = importlib.util.spec_from_file_location("round8_prepare", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.episode = "source/d/e"
        self.lookup = {}
        self.labels = {}
        for episode in (self.episode, "source/d/e2", "source/d/e3"):
            for t in range(40):
                self.lookup[(episode, t)] = torch.tensor([float(t), 2.0 * t, -float(t)])
                self.labels[(episode, t)] = 0
        self.labels[(self.episode, 21)] = 1
        self.labels[("source/d/e3", 20)] = 1

    def test_event_mask_stops_at_first_event_and_never_fills_unknown_negative(self):
        result = MODULE.event_arrays(
            [self.episode, "source/d/e2", "source/d/e3"],
            torch.tensor([18, 39, 20]),
            torch.tensor([0, 0, 1]),
            [21, None, 19],
            torch.tensor([[False, False, False], [True, True, True], [False, False, False]]),
            self.lookup,
            self.labels,
        )
        self.assertEqual(result["event_observed_mask"][0].tolist(), [True, True, True, False, False])
        self.assertEqual(result["event_occurrence"][0].tolist(), [False, False, True, False, False])
        self.assertEqual(int(result["event_time_bin"][0]), 3)
        self.assertEqual(result["event_observed_mask"][1].tolist(), [False] * 5)
        self.assertEqual(int(result["event_time_bin"][2]), 0)

    def test_signed_history_and_lag5_slots_are_exact(self):
        base = torch.zeros((1, 9, 772))
        expected_force = torch.stack([self.lookup[(self.episode, t)] for t in range(10, 19)])
        old_delta = torch.zeros((1, 9, 3))
        old_delta[:, 5:] = expected_force[5:] - expected_force[:4]
        rows = {
            "base": base,
            "force_delta_slots": old_delta,
            "visual_delta_slots": torch.zeros_like(old_delta),
            "episode_id": [self.episode],
            "leakage_group": [self.episode],
            "t": torch.tensor([18]),
            "current_slip_label": torch.tensor([0]),
            "first_current_slip_t": [21],
            "horizon_mask": torch.tensor([[True, True, True]]),
            "y": torch.tensor([[0.0, 1.0, 1.0]]),
            "common_mask": torch.tensor([True]),
            "right_censored": torch.tensor([[False, False, False]]),
            "timeline_contiguous": [False],
            "identity_sha256": "ignored",
        }
        result = MODULE.transform_rows(rows, self.lookup, self.labels)
        self.assertTrue(torch.equal(result["base"][0, :, 769:772], expected_force))
        self.assertTrue(torch.equal(result["force_delta_slots"][0, :5], torch.zeros(5, 3)))
        self.assertTrue(torch.equal(result["shared_aux_valid"][0, :, 0], torch.tensor([False, False, False, False, False, True, True, True, True])))
        self.assertTrue(torch.equal(result["future_force_target"][0, 2], self.lookup[(self.episode, 21)]))

    def test_population_stats_uses_population_variance(self):
        result = MODULE.population_stats(torch.tensor([[0.0, 2.0], [2.0, 4.0]]))
        self.assertTrue(torch.equal(result["mean"], torch.tensor([1.0, 3.0])))
        self.assertTrue(torch.equal(result["variance"], torch.tensor([1.0, 1.0])))


if __name__ == "__main__":
    unittest.main()
