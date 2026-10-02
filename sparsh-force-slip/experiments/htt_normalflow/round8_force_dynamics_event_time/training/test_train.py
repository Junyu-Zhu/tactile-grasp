from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("r8_training", HERE / "train.py")
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


def make_row(generator: torch.Generator, role: str, n: int, *, timeline: bool) -> dict:
    base = torch.randn(n, 9, 772, generator=generator)
    delta = torch.zeros(n, 9, 3)
    delta[:, 5:] = torch.randn(n, 4, 3, generator=generator)
    valid = torch.zeros(n, 9, 1, dtype=torch.bool)
    valid[:, 5:] = True
    event_bin = torch.tensor(([0, 1, 2, 4] * ((n + 3) // 4))[:n], dtype=torch.int64)
    occurrence = torch.zeros(n, 5, dtype=torch.bool)
    occurrence[event_bin > 0, event_bin[event_bin > 0] - 1] = True
    observed = torch.ones(n, 5, dtype=torch.bool)
    y = torch.stack([((event_bin > 0) & (event_bin <= horizon)).float() for horizon in (1, 3, 5)], dim=1)
    mask = torch.ones(n, 3, dtype=torch.bool)
    row = {
        "base": base,
        "force_delta_slots": delta,
        "shared_aux_valid": valid,
        "y": y,
        "horizon_mask": mask,
        "common_mask": mask.all(1),
        "event_time_bin": event_bin,
        "event_occurrence": occurrence,
        "event_observed_mask": observed,
        "future_force_target": torch.randn(n, 5, 3, generator=generator),
        "future_force_observed_mask": torch.ones(n, 5, dtype=torch.bool),
        "episode_id": [f"{role}-episode-{index}" for index in range(n)],
        "leakage_group": [f"{role}-group-{index}" for index in range(n)],
        "t": torch.arange(n) + 18,
        "first_current_slip_t": [50] * n,
        "current_slip_label": torch.zeros(n, dtype=torch.int64),
    }
    if timeline:
        row["right_censored"] = torch.zeros(n, 3, dtype=torch.bool)
        row["timeline_contiguous"] = [True] * n
    return row


def make_payload() -> dict:
    generator = torch.Generator().manual_seed(91)
    counts = {"fit_train": 16, "selection": 8, "calibration": 8, "outer": 8}
    roles = {role: make_row(generator, role, count, timeline=False) for role, count in counts.items()}
    timelines = {role: make_row(generator, role, counts[role], timeline=True) for role in M.PREDICTION_ROLES}
    return {
        "schema": "round8_force_dynamics_prepared_v1",
        "status": "complete",
        "formal": True,
        "horizons": [1, 3, 5],
        "history": 9,
        "conditions": {"P3_triggered": True, "P4_triggered": True},
        "role_groups": {role: [f"{role}-group-{index}" for index in range(count)] for role, count in counts.items()},
        "roles": roles,
        "timelines": timelines,
        "normalization": {
            "fit_role": "fit_train",
            "base": {"mean": torch.zeros(772), "std": torch.ones(772)},
            "force_delta": {"mean": torch.zeros(3), "std": torch.ones(3)},
            "future_force": {"mean": torch.zeros(3), "std": torch.ones(3)},
        },
    }


class Round8TrainingTests(unittest.TestCase):
    def test_p1_same_seed_initialization_is_exact(self):
        left, left_hash = M.fixed_initial_state(20260914, "B_xyz")
        right, right_hash = M.fixed_initial_state(20260914, "C_xyz_delta")
        self.assertEqual(left_hash, right_hash)
        self.assertTrue(M.nested_equal(left, right))

    def test_hazard_groups_share_encoder_initialization(self):
        independent, _ = M.fixed_initial_state(20260914, "C_xyz_delta")
        hazard, _ = M.fixed_initial_state(20260914, "C_hazard")
        for name in ("gru.weight_ih_l0", "gru.weight_hh_l0", "gru.bias_ih_l0", "gru.bias_hh_l0"):
            self.assertTrue(torch.equal(independent[name], hazard[name]))

    def test_input_layout_and_inactive_delta(self):
        payload = make_payload()
        row = payload["roles"]["fit_train"]
        b = M.assemble_inputs(payload, row, "B_xyz")
        c = M.assemble_inputs(payload, row, "C_xyz_delta")
        self.assertEqual(tuple(b.shape), (16, 9, 776))
        self.assertEqual(int(b[:, :, 772:775].count_nonzero()), 0)
        self.assertGreater(int(c[:, 5:, 772:775].count_nonzero()), 0)
        self.assertTrue(torch.equal(b[:, :, :772], c[:, :, :772]))
        self.assertTrue(torch.equal(b[:, :, 775], c[:, :, 775]))

    def test_hazard_loss_obeys_censor_and_cumulative_identity(self):
        logits = torch.zeros(2, 5)
        occurrence = torch.tensor([[0, 1, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.bool)
        observed = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]], dtype=torch.bool)
        loss = M.censored_hazard_nll(logits, occurrence, observed)
        self.assertAlmostEqual(float(loss), float(np.log(2)), places=6)
        cumulative, hazards = M.hazard_probabilities(logits, [1, 3, 5])
        self.assertTrue(torch.all(cumulative[:, 0] <= cumulative[:, 1]))
        self.assertTrue(torch.all(cumulative[:, 1] <= cumulative[:, 2]))
        self.assertTrue(torch.allclose(cumulative[:, 2], 1 - torch.prod(1 - hazards, dim=1)))

    def test_p3_capacity_match_and_causal_per_step_shapes(self):
        concat = M.make_model("P3_concat_independent")
        fusion = M.make_model("P3_fusion_independent")
        concat_n = sum(parameter.numel() for parameter in concat.parameters())
        fusion_n = sum(parameter.numel() for parameter in fusion.parameters())
        self.assertLess(abs(concat_n - fusion_n) / concat_n, 0.0002)
        x = torch.randn(4, 9, 776)
        self.assertEqual(tuple(concat(x)["risk_logits"].shape), (4, 3))
        self.assertEqual(tuple(fusion(x)["risk_logits"].shape), (4, 3))

    def test_p4_models_are_same_capacity_and_initialization(self):
        direct, direct_hash = M.fixed_initial_state(20260914, "P4_direct")
        state, state_hash = M.fixed_initial_state(20260914, "P4_state")
        self.assertEqual(direct_hash, state_hash)
        self.assertTrue(M.nested_equal(direct, state))
        output = M.make_model("P4_state")(torch.randn(2, 9, 776))
        self.assertEqual(tuple(output["state_prediction"].shape), (2, 5, 3))

    def test_validation_rejects_future_leak_and_role_leakage(self):
        payload = make_payload()
        M.validate_prepared(payload)
        payload["roles"]["fit_train"]["event_observed_mask"][0] = torch.tensor([1, 0, 1, 0, 0])
        with self.assertRaisesRegex(ValueError, "contiguous prefix"):
            M.validate_prepared(payload)

    def test_timeline_schema_contains_hazard_and_state_fields(self):
        payload = make_payload()
        row = payload["timelines"]["outer"]
        model = M.make_model("C_hazard")
        prediction = M.infer(model, M.assemble_inputs(payload, row, "C_hazard"), "C_hazard", "cpu", 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timeline.csv"
            spec = M.atomic_predictions(path, payload, row, prediction, "C_hazard", timeline=True)
            with path.open() as stream:
                records = list(csv.DictReader(stream))
            self.assertEqual(spec["schema"], "round8_future_timeline_v1")
            self.assertIn("q_future_step5_raw", records[0])
            q = np.asarray([[float(record[f"q_future_step{step}_raw"]) for step in range(1, 6)] for record in records])
            p5 = np.asarray([float(record["p_future_H5_raw"]) for record in records])
            self.assertTrue(np.allclose(p5, 1 - np.prod(1 - q, axis=1), atol=1e-6))

    def test_prediction_export_validates_base_once_not_per_row(self):
        payload = make_payload()
        row = payload["timelines"]["outer"]
        model = M.make_model("C_hazard")
        prediction = M.infer(model, M.assemble_inputs(payload, row, "C_hazard"), "C_hazard", "cpu", 4)
        original = M._row_base
        calls = []
        def counted(value):
            calls.append(1)
            return original(value)
        M._row_base = counted
        try:
            with tempfile.TemporaryDirectory() as directory:
                M.atomic_predictions(Path(directory) / "timeline.csv", payload, row, prediction, "C_hazard", timeline=True)
        finally:
            M._row_base = original
        self.assertEqual(len(calls), 1)


    def test_exact_interrupted_resume_for_smoke(self):
        payload = make_payload()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "prepared.pt"
            torch.save(payload, data)
            continuous, resumed = root / "continuous", root / "resumed"
            M.train(data, "C_hazard", 20260914, continuous, "cpu", True, False, False, None)
            interrupted = M.train(data, "C_hazard", 20260914, resumed, "cpu", True, False, False, 1)
            self.assertEqual(interrupted["status"], "interrupted")
            M.train(data, "C_hazard", 20260914, resumed, "cpu", True, False, True, None)
            left = torch.load(continuous / "latest.pth", map_location="cpu", weights_only=False)
            right = torch.load(resumed / "latest.pth", map_location="cpu", weights_only=False)
            for key in ("model_state", "optimizer_state", "history", "best_selection_loss", "best_epoch", "best_model_state", "stale", "run_config", "run_identity_sha256", "audit"):
                self.assertTrue(M.nested_equal(left[key], right[key]), key)


if __name__ == "__main__":
    unittest.main()
