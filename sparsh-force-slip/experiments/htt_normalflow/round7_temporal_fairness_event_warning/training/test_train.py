from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("r7_training", HERE / "train.py")
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


def make_row(generator: torch.Generator, name: str, n: int, *, timeline: bool) -> dict:
    base = torch.randn(n, 9, 772, generator=generator)
    force_delta = torch.zeros(n, 9, 3)
    visual_delta = torch.zeros(n, 9, 3)
    force_delta[:, 5:] = torch.randn(n, 4, 3, generator=generator)
    visual_delta[:, 5:] = torch.randn(n, 4, 3, generator=generator)
    y = torch.zeros(n, 3)
    for hindex in range(3):
        y[:, hindex] = torch.tensor(([0.0, 1.0] * ((n + 1) // 2))[:n]).roll(hindex)
    mask = torch.ones(n, 3, dtype=torch.bool)
    row = {
        "base": base,
        "force_delta_slots": force_delta,
        "visual_delta_slots": visual_delta,
        "y": y,
        "horizon_mask": mask,
        "common_mask": mask.all(1),
        "episode_id": [f"{name}-episode-{i}" for i in range(n)],
        "leakage_group": [f"{name}-group-{i}" for i in range(n)],
        "t": torch.arange(n) + 18,
        "first_current_slip_t": [50] * n,
        "current_slip_label": torch.zeros(n, dtype=torch.int64),
    }
    if timeline:
        row["right_censored"] = torch.zeros(n, 3, dtype=torch.bool)
        row["timeline_contiguous"] = [True] * n
    return row


def make_payload() -> dict:
    generator = torch.Generator().manual_seed(77)
    roles = {role: make_row(generator, role, 16 if role == "fit_train" else 8, timeline=False) for role in M.ROLES}
    timelines = {role: make_row(generator, role, 8, timeline=True) for role in M.PREDICTION_ROLES}
    return {
        "schema": "round7_temporal_fairness_prepared_v1",
        "status": "complete",
        "formal": True,
        "horizons": [1, 3, 5],
        "history": 9,
        "D_triggered": True,
        "role_groups": {role: [f"{role}-group-{i}" for i in range(16 if role == "fit_train" else 8)] for role in M.ROLES},
        "roles": roles,
        "timelines": timelines,
        "normalization": {
            "fit_role": "fit_train",
            "base": {"mean": torch.zeros(772), "std": torch.ones(772)},
            "force_delta": {"mean": torch.zeros(3), "std": torch.ones(3)},
            "visual_delta": {"mean": torch.zeros(3), "std": torch.ones(3)},
        },
        "provenance": {"numeric_protocol": {"path": str(M.ROOT_NUMERIC), "sha256": M.sha256(M.ROOT_NUMERIC)}},
    }


class Round7TrainingTests(unittest.TestCase):
    def test_same_seed_initialization_is_complete_and_identical(self):
        left, left_hash = M.fixed_initial_state(20260914, 776, 128, 3)
        right, right_hash = M.fixed_initial_state(20260914, 776, 128, 3)
        self.assertEqual(left_hash, right_hash)
        self.assertTrue(M.nested_equal(left, right))
        self.assertEqual(tuple(left["gru.weight_ih_l0"].shape), (384, 776))
        self.assertEqual(tuple(left["risk.weight"].shape), (3, 128))

    def test_group_assembly_uses_equal_shape_and_zero_inactive_after_normalization(self):
        payload = make_payload()
        row = payload["roles"]["fit_train"]
        for group in ("A_visual", "B_force", "C_force_delta", "D_visual_delta"):
            x = M.assemble_inputs(payload, row, group)
            self.assertEqual(tuple(x.shape), (16, 9, 776))
            inactive = sorted(set(range(776)) - M.active_input_columns(group))
            if inactive:
                self.assertEqual(int(torch.count_nonzero(x[:, :, inactive])), 0)
            self.assertEqual(int(torch.count_nonzero(x[:, :5, 775])), 0)
            self.assertEqual(int(torch.count_nonzero(x[:, 5:, 775])), 16 * 4)

    def test_masked_loss_ignores_ineligible_target(self):
        logits = torch.zeros(3, 2)
        target = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        mask = torch.tensor([[1, 1], [1, 0], [0, 1]], dtype=torch.bool)
        first = M.masked_horizon_loss(logits, target, mask, None)
        target[1, 1] = 99
        target[2, 0] = 99
        second = M.masked_horizon_loss(logits, target, mask, None)
        self.assertTrue(torch.equal(first, second))

    def test_masked_loss_accepts_a_minibatch_with_one_empty_horizon(self):
        logits = torch.zeros(3, 3)
        target = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        mask = torch.tensor([[1, 0, 1], [1, 0, 1], [1, 0, 1]], dtype=torch.bool)
        loss = M.masked_horizon_loss(logits, target, mask, None)
        self.assertTrue(torch.isfinite(loss))

    def test_validate_rejects_auxiliary_information_before_last_four(self):
        payload = make_payload()
        payload["roles"]["fit_train"]["force_delta_slots"][0, 0, 0] = 1
        with self.assertRaisesRegex(ValueError, "outside the fixed last-four"):
            M.validate_prepared(payload)

    def test_timeline_may_include_declared_group_without_eligible_training_row(self):
        payload = make_payload()
        payload["role_groups"]["selection"].append("selection-censored-only")
        payload["timelines"]["selection"]["leakage_group"][-1] = "selection-censored-only"
        M.validate_prepared(payload)

    def test_interrupted_resume_is_exact_at_fixed_smoke_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "prepared.pt"
            torch.save(make_payload(), data_path)
            continuous, resumed = root / "continuous", root / "resumed"
            M.train(data_path, "A_visual", 20260914, continuous, "cpu", True, False, False, None)
            interrupted = M.train(data_path, "A_visual", 20260914, resumed, "cpu", True, False, False, 1)
            self.assertEqual(interrupted["status"], "interrupted")
            M.train(data_path, "A_visual", 20260914, resumed, "cpu", True, False, True, None)
            left = torch.load(continuous / "latest.pth", map_location="cpu", weights_only=False)
            right = torch.load(resumed / "latest.pth", map_location="cpu", weights_only=False)
            for key in ("model_state", "optimizer_state", "history", "best_selection_loss", "best_epoch", "best_model_state", "stale", "run_config", "run_identity_sha256", "audit"):
                self.assertTrue(M.nested_equal(left[key], right[key]), key)

    def test_prediction_export_contains_all_horizon_and_censor_fields(self):
        payload = make_payload()
        row = payload["timelines"]["outer"]
        probabilities = np.full((len(row["y"]), 3), 0.25, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            artifact = M.atomic_predictions(path, row, probabilities, [1, 3, 5], timeline=True)
            text = path.read_text().splitlines()
            self.assertEqual(artifact["rows"], len(row["y"]))
            self.assertIn("right_censored_H5", text[0])
            self.assertIn("common_eligible_H3", text[0])
            self.assertIn("p_future_H1_raw", text[0])

    def test_formal_requires_explicit_mode(self):
        with self.assertRaises(ValueError):
            M.train(Path("missing.pt"), "A_visual", 20260914, Path("out"), "cpu", False, False, False, None)


if __name__ == "__main__":
    unittest.main()
