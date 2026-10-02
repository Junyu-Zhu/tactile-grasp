from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

try:
    from .complete_unavailable_ci import complete
except ImportError:
    from complete_unavailable_ci import complete


class CICompletionTests(unittest.TestCase):
    def test_complete_grid_preserves_valid_rows_and_marks_partial_seed_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); output = root / "evaluation"; output.mkdir()
            groups, seeds = ["candidate", "comparator"], [1, 2]
            metrics = []
            for group in groups:
                for seed in seeds:
                    for operating_point in ("trial_FA_0.10", "event_recall_0.70"):
                        available = not (group == "candidate" and seed == 2 and operating_point == "event_recall_0.70")
                        metrics.append({"method_type": "neural_operational", "group": group, "seed": seed,
                                        "horizon": 1, "population": "primary", "operating_point": operating_point,
                                        "rule_selected": True, "threshold_status": "available" if available else "unavailable_calibration_constraint"})
            with (output / "metrics.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
            event_metrics = ["event_recall", "trial_false_alarm_rate", "false_alarm_starts_per_trial",
                             "false_alarm_duration_per_trial", "mean_lead_frames"]
            paired = [{"comparison": "candidate_vs_comparator", "horizon": 1, "operating_point": operating_point,
                       "metric": metric, "estimate": "0.125", "ci_low": "0.1", "ci_high": "0.2",
                       "valid_replicates": "200", "requested_replicates": "200", "unit": "leakage_group", "paired": "True"}
                      for operating_point, names in (("trial_FA_0.10", event_metrics),
                                                     ("raw_common_population", ["average_precision", "brier"]))
                      for metric in names]
            with (output / "paired_bootstrap_ci.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(paired[0])); writer.writeheader(); writer.writerows(paired)
            (output / "summary.json").write_text(json.dumps({"status": "complete", "source_hashes": {}, "output_hashes": {}}))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"schema": "round8_prediction_manifest_v1", "status": "complete",
                                            "groups": groups, "seeds": seeds, "horizons": [1],
                                            "comparisons": [{"candidate": "candidate", "comparator": "comparator"}]}))
            audit = complete(manifest, output)
            self.assertEqual(audit["rows"], 12); self.assertEqual(audit["unavailable_rows"], 5)
            with (output / "paired_bootstrap_ci.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            available = [row for row in rows if row["status"] == "available"]
            unavailable = [row for row in rows if row["status"] != "available"]
            self.assertEqual({row["estimate"] for row in available}, {"0.125"})
            self.assertEqual({row["estimate"] for row in unavailable}, {""})
            self.assertEqual({row["candidate_unavailable_seeds"] for row in unavailable}, {"2"})
            self.assertTrue((output / "paired_bootstrap_ci_pre_completion.csv").is_file())
            self.assertTrue((output / "summary_pre_ci_completion.json").is_file())
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["ci_completion"]["status"], "complete")
            self.assertIn("CI_COMPLETION_AUDIT.json", summary["output_hashes"])


if __name__ == "__main__":
    unittest.main()
