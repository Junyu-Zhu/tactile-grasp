import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("r7_audit", HERE / "audit.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AuditUnitTests(unittest.TestCase):
    def test_exact_history_raw_union(self):
        t = 30
        history = range(t - 8, t + 1)
        base = sorted({u for s in history for u in (s - 5, s)})
        aux = sorted({u for s in range(t - 3, t + 1) for r in (s, s - 5) for u in (r - 5, r)})
        self.assertEqual(base, list(range(t - 13, t + 1)))
        self.assertEqual(min(aux), t - 13)
        self.assertEqual(max(aux), t)

    def test_future_any_differs_after_first_onset(self):
        labels = [0, 1, 0, 1, 0]
        t, horizon, first_onset = 2, 1, 1
        future_any = int(sum(labels[t + 1:t + horizon + 1]) > 0)
        first_onset_target = int(t < first_onset <= t + horizon)
        self.assertEqual(future_any, 1)
        self.assertEqual(first_onset_target, 0)

    def test_trial_counts_distinct_groups(self):
        trials = [
            {"episode_id": "a", "leakage_group": "a", "positive_endpoints": [1, 2], "negative_endpoints": []},
            {"episode_id": "b", "leakage_group": "b", "positive_endpoints": [], "negative_endpoints": [3]},
        ]
        result = MODULE.trial_summary(trials)
        self.assertEqual(result["positive_event_trials"], 1)
        self.assertEqual(result["negative_window_trials"], 1)
        self.assertEqual(result["positive_endpoints"], 2)


if __name__ == "__main__":
    unittest.main()
