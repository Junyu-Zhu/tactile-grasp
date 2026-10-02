from __future__ import annotations

import unittest
import numpy as np
try:
    from sklearn.metrics import roc_curve
except ImportError:
    roc_curve = None

from metrics import (apply_clip_standardize, fit_clip_standardize, force_metrics,
                     force_target_normalization, partial_tpr_auc_0_0p1)


class MetricTests(unittest.TestCase):
    def test_partial_auc_perfect_and_reversed(self):
        labels = np.array([0, 0, 1, 1])
        self.assertEqual(partial_tpr_auc_0_0p1(labels, np.array([0.1, 0.2, 0.8, 0.9])), 1.0)
        self.assertEqual(partial_tpr_auc_0_0p1(labels, np.array([0.9, 0.8, 0.2, 0.1])), 0.0)

    def test_partial_auc_ties_grouped(self):
        labels = np.array([0, 1, 0, 1])
        a = partial_tpr_auc_0_0p1(labels, np.ones(4))
        b = partial_tpr_auc_0_0p1(labels[::-1], np.ones(4))
        self.assertEqual(a, b)

    def test_partial_auc_preserves_vertical_boundary(self):
        # At FPR=.1 the ROC rises vertically. Only the lower point contributes
        # to the preceding horizontal interval; the vertical edge has zero area.
        labels = np.array([1, 0, 1, 1] + [0] * 9)
        scores = np.arange(len(labels), 0, -1)
        self.assertAlmostEqual(partial_tpr_auc_0_0p1(labels, scores), 1 / 3)

    @unittest.skipIf(roc_curve is None, "optional sklearn reference unavailable")
    def test_partial_auc_matches_sklearn_curve_random_discrete(self):
        generator = np.random.default_rng(20260915)
        for _ in range(20):
            labels = np.array([0] * 30 + [1] * 20)
            generator.shuffle(labels)
            scores = generator.integers(0, 8, size=len(labels)).astype(float)
            fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
            end = int(np.searchsorted(fpr, 0.1, side="right"))
            x, y = fpr[:end], tpr[:end]
            if x[-1] < 0.1:
                boundary = np.interp(0.1, fpr, tpr)
                x, y = np.r_[x, 0.1], np.r_[y, boundary]
            expected = np.sum(np.diff(x) * (y[:-1] + y[1:]) * 0.5) / 0.1
            self.assertAlmostEqual(partial_tpr_auc_0_0p1(labels, scores), expected)

    def test_force_metrics_and_normalization(self):
        target = np.array([[0., 1., 2.], [2., 3., 4.]])
        norm = force_target_normalization(target)
        self.assertTrue(np.allclose(norm["mean"], [1, 2, 3]))
        result = force_metrics(target, target + 1)
        self.assertEqual(result["rmse_xyz_native_n"], [1., 1., 1.])

    def test_condition_clipping_uses_frozen_values(self):
        train = np.arange(100, dtype=np.float32)[:, None] * np.ones((1, 5), np.float32)
        normalization = fit_clip_standardize(train)
        transformed = apply_clip_standardize(np.array([[-100] * 5, [1000] * 5]), normalization)
        self.assertTrue(np.isfinite(transformed).all())
        self.assertTrue(np.all(transformed[0] < transformed[1]))


if __name__ == "__main__":
    unittest.main()
