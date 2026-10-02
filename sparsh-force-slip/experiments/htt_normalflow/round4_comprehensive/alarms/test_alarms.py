import importlib.util
from pathlib import Path
import csv
import json
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


PATH = Path(__file__).with_name("evaluate_alarms.py")
SPEC = importlib.util.spec_from_file_location("alarms", PATH)
A = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(A)


def rows(stages, scores, episode="e"):
    return [{"episode": episode, "t": i, "stage": y, "score": s} for i, (y, s) in enumerate(zip(stages, scores))]


class AlarmTests(unittest.TestCase):
    def test_budget_exact_and_never_alarm(self):
        data = rows([0, 0, 2, 2], [.1, .2, .8, .9])
        chosen = A.choose_budget(data, 0.0)
        self.assertEqual(chosen["gross_recall"], 1.0)
        self.assertEqual(chosen["static_fpr"], 0.0)
        never = A.raw_metrics(data, np.nextafter(1.0, np.inf))
        self.assertTrue(never["never_alarm"]); self.assertEqual(never["gross_recall"], 0.0)

    def test_hysteresis_and_incipient_advances_state(self):
        data = rows([0, 1, 2, 2, 0], [.8, .9, .7, .3, .1])
        metrics, trials = A.sequential_metrics(data, .75, 2, .8)
        self.assertEqual(metrics["gross_events_detected"], 1)
        self.assertEqual(metrics["mean_detection_delay_frames"], 0.0)
        self.assertEqual(trials[0]["false_alarm_starts"], 0)

    def test_event_tail_and_episode_reset(self):
        data = rows([0, 2, 2], [.1, .8, .8], "a") + rows([2, 2], [.8, .8], "b")
        metrics, _ = A.sequential_metrics(data, .5, 1, 1.0)
        self.assertEqual(metrics["gross_events"], 2)
        self.assertEqual(metrics["gross_events_detected"], 2)
        self.assertEqual(metrics["segments_beginning_at_frame_zero"], 1)

    def test_max_ba_tie_prefers_lower_fpr_then_threshold(self):
        data = rows([0, 0, 2, 2], [.1, .4, .6, .9])
        chosen = A.choose_max_ba(data)
        self.assertEqual(chosen["threshold"], .6)

    def test_discovery_excludes_conflicting_smoke_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); formal = root / "enc/runs/dino/htt_leave_p1/20260914"
            smoke = root / "enc/smoke/dino"; empty_r3 = root / "r3"
            for run, status, is_smoke in ((formal, "complete", False), (smoke, "smoke_complete", True)):
                (run / "predictions").mkdir(parents=True)
                with (run / "predictions/calibration.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=("episode_id", "t", "stage", "p_slip", "fold", "seed"))
                    writer.writeheader(); writer.writerow({"episode_id":"e","t":0,"stage":0,"p_slip":.1,"fold":"htt_leave_p1","seed":20260914})
                (run / "training_summary.json").write_text(json.dumps({"status":status,"config":{"smoke":is_smoke}}))
            found = A.discover(SimpleNamespace(r3_root=empty_r3, encoder_root=root / "enc"))
            self.assertEqual(found, [("dino", "htt_leave_p1", 20260914, formal)])


if __name__ == "__main__": unittest.main()
