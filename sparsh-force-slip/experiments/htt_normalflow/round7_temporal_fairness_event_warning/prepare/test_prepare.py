import importlib.util
from pathlib import Path
import unittest

import torch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("r7_prepare", HERE / "prepare.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrepareUnitTests(unittest.TestCase):
    def test_pca_is_deterministically_oriented(self):
        generator = torch.Generator().manual_seed(7)
        x = torch.randn(30, 768, generator=generator)
        result = MODULE.fit_pca(x, [("e", i) for i in range(len(x))])
        self.assertGreaterEqual(result["rank"], 3)
        for row in result["components"]:
            pivot = int(torch.argmax(torch.abs(row)))
            self.assertGreater(float(row[pivot]), 0)

    def test_stats_uses_positive_floor(self):
        result = MODULE.stats(torch.ones(5, 3))
        self.assertTrue(torch.equal(result["mean"], torch.ones(3)))
        self.assertTrue(bool((result["std"] == 1e-6).all()))

    def test_active_aux_positions_do_not_extend_raw_history(self):
        t = 30
        union = {u for s in range(t - 3, t + 1) for r in (s, s - 5) for u in (r, r - 5)}
        self.assertEqual(min(union), t - 13)
        self.assertEqual(max(union), t)

    def test_union_subset_removes_unsupervised_rows(self):
        row = {
            "base": torch.zeros(2, 9, 772),
            "horizon_mask": torch.tensor([[True, False, False], [False, False, False]]),
            "y": torch.zeros(2, 3),
            "episode_id": ["a", "b"],
            "t": torch.tensor([18, 19]),
            "identity_sha256": "old",
        }
        result = MODULE.subset_row(row, row["horizon_mask"].any(1))
        self.assertEqual(result["episode_id"], ["a"])
        self.assertEqual(tuple(result["base"].shape), (1, 9, 772))


if __name__ == "__main__":
    unittest.main()
