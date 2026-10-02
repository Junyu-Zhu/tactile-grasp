import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("ap_bootstrap.py")
SPEC = importlib.util.spec_from_file_location("r7_ap_bootstrap", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TestMetrics(unittest.TestCase):
    def test_average_precision_aggregates_ties(self):
        target = np.array([1, 0, 1, 0])
        score = np.array([0.5, 0.5, 0.2, 0.1])
        self.assertAlmostEqual(MODULE.average_precision(target, score), 7 / 12)

    def test_brier_and_prevalence(self):
        result = MODULE.metrics(np.array([0, 1]), np.array([0.25, 0.75]))
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["positives"], 1)
        self.assertAlmostEqual(result["prevalence"], 0.5)
        self.assertAlmostEqual(result["brier"], 0.0625)

    def test_whole_group_sampling_preserves_clusters_and_multiplicity(self):
        groups = np.array(["x", "x", "y", "z", "z", "z"], dtype=object)
        names, indices = MODULE.group_indices(groups)
        sampled = MODULE.sampled_indices(np.array([2, 0, 2]), names, indices)
        self.assertEqual(sampled.tolist(), [3, 4, 5, 0, 1, 3, 4, 5])

    def test_atomic_text_replaces_complete_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.json"
            MODULE.atomic_text(path, "first\n")
            MODULE.atomic_text(path, "second\n")
            self.assertEqual(path.read_text(), "second\n")
            self.assertEqual(list(Path(directory).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
