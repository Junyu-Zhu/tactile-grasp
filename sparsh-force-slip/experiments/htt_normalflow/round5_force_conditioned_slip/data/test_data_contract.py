from pathlib import Path
import tempfile
import unittest

import numpy as np

from contract import FORCE_TARGET_SEMANTICS, FORMAT, eligible_indices, select_entries, validate_contract
from adapter import CachedFrames, fit_force_standardizer


def entry(tmp_path: Path, task: str, episode_id: str):
    target_key = "force_native_n_path" if task == "force" else "label_path"
    return {
        "episode_id": episode_id, "task": task, "frames": 20,
        "roles_by_fold": {"htt_leave_p1": "train", "htt_leave_p2": "validation",
                          "htt_leave_p3": "calibration", "htt_leave_p4": "test"},
        "source_files": {"source.npz": "0" * 64}, "token_path": str(tmp_path / "tokens.npy"),
        "token_sha256": "1" * 64, target_key: str(tmp_path / "target.npy"),
    }


def payload(entries, pairing="independent_episodes_no_force_slip_basename_join"):
    return {"format": FORMAT, "force_target_semantics": FORCE_TARGET_SEMANTICS,
            "pairing_policy": pairing, "entries": entries}


class ContractTests(unittest.TestCase):
    def test_history_bounds_are_strict(self):
        self.assertEqual(list(eligible_indices(12, "force")), list(range(5, 12)))
        self.assertEqual(list(eligible_indices(12, "slip")), [10, 11])
        self.assertEqual(list(eligible_indices(15, "future")), [13, 14])

    def test_contract_rejects_basename_pairing_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            contract = payload([entry(Path(directory), "force", "force/a")], "join_by_basename")
            with self.assertRaisesRegex(ValueError, "pairing policy"):
                validate_contract(contract)

    def test_select_never_exposes_test_role(self):
        with tempfile.TemporaryDirectory() as directory:
            contract = payload([entry(Path(directory), "force", "force/a")])
            self.assertEqual(len(select_entries(contract, "htt_leave_p1", "train", "force")), 1)
            with self.assertRaisesRegex(ValueError, "development roles"):
                select_entries(contract, "htt_leave_p4", "test", "force")

    def test_file_validation_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            row = entry(Path(directory), "force", "force/a")
            np.save(row["token_path"], np.zeros((20, 3, 4), np.float32))
            np.save(row["force_native_n_path"], np.zeros((20, 3), np.float32))
            contract = payload([row])
            validate_contract(contract, require_files=True)

    def test_train_only_standardizer_and_cached_view(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = entry(root, "force", "force/a")
            values = np.arange(60, dtype=np.float32).reshape(20, 3)
            np.save(row["token_path"], np.zeros((20, 3, 4), np.float32))
            np.save(row["force_native_n_path"], values)
            contract = payload([row])
            stats = fit_force_standardizer(contract, "htt_leave_p1")
            expected = values[5:]
            np.testing.assert_allclose(stats["mean_native_N"], expected.mean(0))
            view = CachedFrames(contract, "htt_leave_p1", "train", "force", 5, stats)
            self.assertEqual(len(view), 15)
            np.testing.assert_allclose(view[0]["target"], (values[5] - expected.mean(0)) / expected.std(0))


if __name__ == "__main__":
    unittest.main()
