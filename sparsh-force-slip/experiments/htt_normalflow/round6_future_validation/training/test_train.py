from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("r6_train", HERE / "train.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def nested_equal(a, b):
    if torch.is_tensor(a): return torch.equal(a, b)
    if isinstance(a, dict): return a.keys() == b.keys() and all(nested_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return len(a) == len(b) and all(nested_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray): return np.array_equal(a, b)
    return a == b


class Round6TrainingTests(unittest.TestCase):
    def test_prefix_initialization_shares_every_common_value(self):
        dims = {"A_visual": 769, "B_force": 772, "C_force_delta": 775}
        c, _ = m.canonical_initial_state("C_force_delta", 20260914, dims, 128)
        for group in ("A_visual", "B_force"):
            state, evidence = m.canonical_initial_state(group, 20260914, dims, 128)
            d = dims[group]
            self.assertTrue(torch.equal(state["gru.weight_ih_l0"], c["gru.weight_ih_l0"][:, :d]))
            for key in ("gru.weight_hh_l0", "gru.bias_ih_l0", "gru.bias_hh_l0", "risk.weight", "risk.bias"):
                self.assertTrue(torch.equal(state[key], c[key]))
            self.assertEqual(evidence["input_columns"], [0, d])

    def test_calibration_includes_global_never_alarm(self):
        y = np.array([0, 0, 1, 1]); score = np.ones(4)
        row = m.calibrate(y, score, "fpr", .01)
        self.assertGreater(row["threshold"], 1.0); self.assertTrue(row["observed_no_alarm"])

    def test_interrupted_resume_matches_continuous_fixed_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); data_path = root / "prepared.pt"
            g = torch.Generator().manual_seed(5); roles = {}
            for role, n in (("fit_train", 16), ("selection", 8), ("calibration", 8), ("outer", 8)):
                x = torch.randn(n, 4, 775, generator=g).half(); y = torch.tensor(([0., 1.] * ((n+1)//2))[:n])
                roles[role] = {"x_C": x, "y": y, "episode_id": [f"{role}-{i}" for i in range(n)], "leakage_group": [f"{role}-{i}" for i in range(n)], "first_current_slip_t": [20]*n, "t": torch.arange(n)+13}
            payload = {"schema":"round6_conditional_gru_v1_prepared_v1","status":"complete","formal":True,"feature_layout_C":{"z":[0,768]},"group_input_dims":{"A_visual":769,"B_force":772,"C_force_delta":775},"normalization_C":{"mean":torch.zeros(775),"std":torch.ones(775)},"roles":roles,"provenance":{"protocol":{"sha256":m.sha256(m.PROTOCOL)}}}
            torch.save(payload, data_path)
            continuous = root/"continuous"; resumed = root/"resumed"
            m.train(data_path,"A_visual",20260914,continuous,"cpu",True,False,False,None)
            interrupted = m.train(data_path,"A_visual",20260914,resumed,"cpu",True,False,False,1)
            self.assertEqual(interrupted["status"],"interrupted")
            m.train(data_path,"A_visual",20260914,resumed,"cpu",True,False,True,None)
            left=torch.load(continuous/"latest.pth",map_location="cpu",weights_only=False); right=torch.load(resumed/"latest.pth",map_location="cpu",weights_only=False)
            for key in ("model_state","optimizer_state","history","best_selection_loss","best_epoch","best_model_state","stale","run_config","run_identity_sha256"):
                self.assertTrue(nested_equal(left[key],right[key]),key)

    def test_formal_requires_explicit_execute_flag(self):
        with self.assertRaises(ValueError):
            m.train(Path("missing.pt"),"A_visual",20260914,Path("out"),"cpu",False,False,False,None)


if __name__ == "__main__": unittest.main()
