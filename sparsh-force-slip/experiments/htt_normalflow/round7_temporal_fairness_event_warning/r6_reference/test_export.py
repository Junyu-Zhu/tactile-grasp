import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("r7_r6_reference", HERE / "export.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class ReferenceTests(unittest.TestCase):
    def row(self):
        return {
            "base": torch.arange(2 * 9 * 772, dtype=torch.float32).reshape(2, 9, 772),
            "force_delta_slots": torch.ones(2, 9, 3),
            "horizon_mask": torch.tensor([[True, True, True], [False, False, False]]),
            "common_mask": torch.tensor([True, False]),
            "right_censored": torch.tensor([[False, False, False], [True, True, True]]),
            "y": torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]),
            "episode_id": ["a", "b"], "leakage_group": ["a", "b"], "t": torch.tensor([18, 19]),
            "first_current_slip_t": [19, None], "current_slip_label": torch.tensor([0, 0]),
            "timeline_contiguous": [False, False],
        }

    def test_group_dimensions_and_exact_last_four(self):
        row = self.row(); mean = torch.zeros(775); std = torch.ones(775)
        for group, dim in M.GROUP_DIMS.items():
            x = M.r7_x(row, group, mean, std)
            self.assertEqual(tuple(x.shape), (2, 4, dim))
            self.assertTrue(torch.equal(x[:, :, :769], row["base"][:, -4:, :769]))
        self.assertTrue(torch.equal(M.r7_x(row, "C_force_delta", mean, std)[:, :, 772:], row["force_delta_slots"][:, -4:]))

    def test_output_is_h1_only_and_keeps_censoring(self):
        header, rows = M.output_rows(self.row(), np.array([0.2, 0.3]))
        self.assertIn("p_future_H1_raw", header)
        self.assertNotIn("p_future_H3_raw", header)
        self.assertEqual(rows[1][header.index("right_censored_H1")], True)
        self.assertEqual(rows[1][header.index("target_future_H1")], "")

    def test_atomic_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.csv"
            M.atomic_csv(path, ["a"], [[1]])
            self.assertEqual(path.read_text().splitlines(), ["a", "1"])


if __name__ == "__main__":
    unittest.main()
