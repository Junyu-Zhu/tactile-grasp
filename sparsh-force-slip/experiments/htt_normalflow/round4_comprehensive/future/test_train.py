from __future__ import annotations

import unittest

import numpy as np
import torch

from train import HISTORY, GRUHead, MLPHead, checkpoint_payload, primary_indices


class FutureTrainTests(unittest.TestCase):
    def test_primary_indices_respect_history_complete_window_and_onset(self):
        labels = np.array([0] * 12 + [1] * 5 + [2] * 8, dtype=np.int8)
        rows = primary_indices(labels)
        self.assertTrue(all(t >= HISTORY - 1 and t + 8 < len(labels) and labels[t] == 0 for t, _ in rows))
        self.assertEqual([t for t, y in rows if y], list(range(9, 12)))
        self.assertTrue(all(t < 17 for t, _ in rows))

    def test_no_event_has_only_negatives_and_no_episode_crossing(self):
        labels = np.zeros(15, dtype=np.int8)
        rows = primary_indices(labels)
        self.assertEqual(rows, [(t, 0) for t in range(3, 7)])

    def test_models_consume_same_history_shape(self):
        x = torch.zeros(2, 4, 9)
        self.assertEqual(MLPHead(9)(x).shape, (2,))
        self.assertEqual(GRUHead(9)(x).shape, (2,))

    def test_checkpoint_contains_normalization_and_provenance(self):
        model = MLPHead(2)
        optimizer = torch.optim.AdamW(model.parameters())
        config = {"cache_manifest_sha256": "cache", "support_audit_sha256": "audit", "protocol_sha256": "protocol"}
        mean, std = np.zeros((1, 2), np.float32), np.ones((1, 2), np.float32)
        payload = checkpoint_payload(model, optimizer, 1, [], 1, 0.5, 0, config, mean, std)
        self.assertTrue(np.array_equal(payload["normalization"]["mean"], mean))
        self.assertEqual(payload["provenance"]["protocol_sha256"], "protocol")


if __name__ == "__main__": unittest.main()
