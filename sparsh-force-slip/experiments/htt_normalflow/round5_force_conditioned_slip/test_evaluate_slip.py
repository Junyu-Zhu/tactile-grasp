import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import evaluate_slip as e


class MetricTests(unittest.TestCase):
    def rows(self, scores, stages, episode="x"):
        return [dict(episode=episode, t=i + 10, stage=stage, score=score)
                for i, (score, stage) in enumerate(zip(scores, stages))]

    def test_perfect(self):
        self.assertAlmostEqual(e.partial_auc(self.rows([0.1, 0.2, 0.8, 0.9], [0, 0, 2, 2])), 1.0)

    def test_tied(self):
        self.assertAlmostEqual(e.partial_auc(self.rows([0.5] * 4, [0, 0, 2, 2])), 0.05)

    def test_reverse(self):
        self.assertAlmostEqual(e.partial_auc(self.rows([0.9, 0.8, 0.2, 0.1], [0, 0, 2, 2])), 0.0)

    def test_boundary(self):
        metrics, _ = e.metrics(self.rows([0.9, 0.9, 0.1], [2, 2, 0]), 0.5, 1, 1.0, {"x": "g"})
        self.assertEqual(metrics["segments_beginning_at_evaluation_boundary"], 1)
        self.assertEqual(metrics["mean_detection_delay_frames"], 0)

    def test_never(self):
        metrics, _ = e.metrics(self.rows([0.9, 0.9, 0.1], [2, 2, 0]), 1.000001, 1, 1.0, {"x": "g"})
        self.assertTrue(metrics["never_alarm"])
        self.assertEqual(metrics["gross_recall"], 0)

    def test_trial_cluster_pauc_bootstrap_is_deterministic(self):
        rows = self.rows([0.1, 0.2, 0.8, 0.9], [0, 0, 2, 2], "x")
        rows += self.rows([0.05, 0.15, 0.85, 0.95], [0, 0, 2, 2], "y")
        groups = {"x": "gx", "y": "gy"}
        first = e.bootstrap_partial_auc(rows, groups, 7, repetitions=20)
        second = e.bootstrap_partial_auc(rows, groups, 7, repetitions=20)
        self.assertEqual(first, second)
        self.assertEqual((first["lower"], first["upper"]), (1.0, 1.0))
        self.assertEqual(first["bootstrap_unit"], "complete schema-2 leakage_group")


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fold = "htt_leave_p1"
        self.manifest_path = self.root / "splits.json"
        self.manifest = {
            "schema_version": 2,
            "episodes": [
                {"id": "cal", "task": "slip", "frames": 12, "leakage_group": "gcal"},
                {"id": "val", "task": "slip", "frames": 12, "leakage_group": "gval"},
            ],
            "splits": {self.fold: {"train": [], "calibration": ["cal"], "validation": ["val"], "test": []}},
        }
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.labels = {}
        entries = []
        for episode_id, role in (("cal", "calibration"), ("val", "validation")):
            label_path = self.root / f"{episode_id}.labels.npy"
            array = np.array([0] * 10 + [0, 2], dtype=np.int64)
            np.save(label_path, array)
            self.labels[episode_id] = array
            entries.append({
                "episode_id": episode_id,
                "task": "slip",
                "frames": 12,
                "roles_by_fold": {self.fold: role},
                "label_path": str(label_path),
                "label_sha256": e.r4.sha256(label_path),
            })
        self.contract_path = self.root / "contract.json"
        self.contract_path.write_text(json.dumps({
            "status": "complete",
            "format": e.CONTRACT_FORMAT,
            "force_target_semantics": e.FORCE_TARGET_SEMANTICS,
            "split_manifest_sha256": e.r4.sha256(self.manifest_path),
            "entries": entries,
        }))
        self.audit_path = self.root / "audit.json"
        self.audit_path.write_text(json.dumps({
            "status": "pass",
            "format": e.CACHE_AUDIT_FORMAT,
            "cache_manifest_sha256": e.r4.sha256(self.contract_path),
            "entries": len(entries),
            "files_verified": 2 * len(entries),
        }))
        self.calibration = self.root / "calibration.csv"
        self.validation = self.root / "validation.csv"
        self.write_prediction(self.calibration, "cal", "calibration")
        self.write_prediction(self.validation, "val", "validation")
        self.checkpoint = self.root / "best.pth"
        self.checkpoint.write_bytes(b"checkpoint")

    def tearDown(self):
        self.temporary.cleanup()

    def write_prediction(self, path, episode, role, override=None):
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("episode_id", "frame", "stage", "role", "probability"))
            for frame in (10, 11):
                stage = int(self.labels[episode][frame]) if override is None else override
                writer.writerow((episode, frame, stage, role, 0.1 if frame == 10 else 0.9))

    def current_summary(self, smoke=False, formal=None):
        payload = {
            "status": "complete",
            "format": e.CURRENT_SUMMARY_FORMAT,
            "variant": "V",
            "fold": self.fold,
            "seed": 20260914,
            "smoke": smoke,
            "predictions": {
                "calibration": {"path": str(self.calibration), "sha256": e.r4.sha256(self.calibration)},
                "validation": {"path": str(self.validation), "sha256": e.r4.sha256(self.validation)},
            },
            "best_checkpoint": str(self.checkpoint),
            "best_checkpoint_sha256": e.r4.sha256(self.checkpoint),
        }
        if formal is not None:
            payload["formal"] = formal
        path = self.root / "training_summary.json"
        path.write_text(json.dumps(payload))
        return path

    def test_contract_labels_are_authoritative(self):
        labels, _ = e.load_authoritative_labels(
            self.contract_path, self.audit_path, self.manifest, self.manifest_path, self.fold
        )
        self.write_prediction(self.calibration, "cal", "calibration", override=2)
        with self.assertRaisesRegex(ValueError, "authoritative label"):
            e.read_rows(self.calibration, self.manifest, self.fold, "calibration", labels)

    def test_supplied_pre_boundary_rows_are_also_checked(self):
        labels, _ = e.load_authoritative_labels(
            self.contract_path, self.audit_path, self.manifest, self.manifest_path, self.fold
        )
        with self.calibration.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("episode_id", "frame", "stage", "role", "probability"))
            writer.writerow(("cal", 0, 2, "calibration", 0.5))
            writer.writerow(("cal", 10, 0, "calibration", 0.1))
            writer.writerow(("cal", 11, 2, "calibration", 0.9))
        with self.assertRaisesRegex(ValueError, "authoritative label"):
            e.read_rows(self.calibration, self.manifest, self.fold, "calibration", labels)

    def test_contract_requires_matching_passed_full_hash_audit(self):
        bad = json.loads(self.audit_path.read_text())
        bad["cache_manifest_sha256"] = "0" * 64
        self.audit_path.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, "does not certify"):
            e.load_authoritative_labels(
                self.contract_path, self.audit_path, self.manifest, self.manifest_path, self.fold
            )

    def test_current_training_chain_accepts_only_formal_identity(self):
        path = self.current_summary(smoke=False)
        result = e.validate_training_chain(
            path, self.calibration, self.validation, self.fold, 20260914, "V", False
        )
        self.assertEqual(result["source_kind"], "round5_force_conditioned_slip")
        with self.assertRaisesRegex(ValueError, "identity is not formal"):
            e.validate_training_chain(
                self.current_summary(smoke=True), self.calibration, self.validation,
                self.fold, 20260914, "V", False,
            )
        with self.assertRaisesRegex(ValueError, "not a completed formal run"):
            e.validate_training_chain(
                self.current_summary(smoke=False, formal=False), self.calibration, self.validation,
                self.fold, 20260914, "V", False,
            )

    def test_historical_baseline_has_explicit_identity(self):
        summary = {
            "status": "complete",
            "fold": self.fold,
            "seed": 20260914,
            "init": "fresh",
            "config": {"init": "fresh", "smoke": False},
            "role_isolation": {"pass": True},
            "predictions": {"calibration": str(self.calibration), "validation": str(self.validation)},
            "prediction_sha256": {
                "calibration": e.r4.sha256(self.calibration),
                "validation": e.r4.sha256(self.validation),
            },
            "best_checkpoint": str(self.checkpoint),
            "best_checkpoint_sha256": e.r4.sha256(self.checkpoint),
        }
        path = self.root / "historical.json"
        path.write_text(json.dumps(summary))
        result = e.validate_training_chain(
            path, self.calibration, self.validation, self.fold, 20260914, "mae-r3-b", True
        )
        self.assertEqual(result["source_kind"], "round3_mae_fresh_b_historical_baseline")
        with self.assertRaisesRegex(ValueError, "historical baseline"):
            e.validate_training_chain(
                path, self.calibration, self.validation, self.fold, 20260914, "V", True
            )

    def test_trials_are_hashed_before_complete_commit(self):
        output = self.root / "evaluation"
        committed = e.write_outputs(output, {"model_id": "V"}, [{"episode": "val", "score": 1.0}])
        disk = json.loads((output / "metrics.json").read_text())
        self.assertEqual(disk, committed)
        self.assertTrue(disk["formal"])
        self.assertEqual(disk["status"], "complete")
        self.assertEqual(disk["artifacts"]["trials_csv_sha256"], e.r4.sha256(output / "trials.csv"))
        self.assertFalse(list(output.glob("*.tmp.*")))


if __name__ == "__main__":
    unittest.main()
