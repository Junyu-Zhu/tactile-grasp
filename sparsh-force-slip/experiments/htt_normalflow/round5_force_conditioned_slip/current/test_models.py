from __future__ import annotations

import sys
from pathlib import Path
import os
import unittest

os.environ.setdefault("XFORMERS_DISABLED", "1")
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "scripts"))
import phase2_b_multitask as p2

from models import ForceAdapter, ForceConditionedSlip, physical_condition


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(42)
        cls.decoder = p2.DecoupledForceSlipDecoder()

    def test_force_warm_start_and_reset_head(self) -> None:
        model = ForceAdapter(self.decoder)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(
            model.force_pooler.state_dict().values(), self.decoder.force_pooler.state_dict().values())))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(
            model.force_trunk.state_dict().values(), self.decoder.force_trunk.state_dict().values())))
        self.assertFalse(torch.equal(model.force_head.weight, self.decoder.force_head.weight))
        output = model(torch.randn(2, 481, 768))
        self.assertEqual(tuple(output.shape), (2, 3))
        self.assertTrue(torch.isfinite(output).all())

    def test_identity_initialization_and_frozen_base(self) -> None:
        for variant in ("V", "F-old", "F-adapt"):
            with self.subTest(variant=variant):
                model = ForceConditionedSlip(self.decoder, variant)
                tokens = torch.randn(3, 481, 768)
                condition = None if variant == "V" else torch.randn(3, 5)
                result = model(tokens, condition, return_aux=True)
                self.assertTrue(torch.equal(result["logits"], result["base_logits"]))
                self.assertFalse(any(parameter.requires_grad for parameter in model.base.parameters()))
                self.assertTrue(torch.equal(result["gamma"], torch.zeros_like(result["gamma"])))
                self.assertTrue(torch.equal(result["beta"], torch.zeros_like(result["beta"])))

    def test_all_trainable_tensors_change_after_multiple_steps(self) -> None:
        for variant in ("V", "F-old", "F-adapt"):
            with self.subTest(variant=variant):
                torch.manual_seed(7)
                model = ForceConditionedSlip(self.decoder, variant)
                before = {name: value.detach().clone() for name, value in model.named_parameters()
                          if value.requires_grad}
                optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-2)
                tokens = torch.randn(8, 481, 768)
                condition = None if variant == "V" else torch.randn(8, 5)
                target = torch.tensor([0, 1] * 4)
                for _ in range(3):
                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.cross_entropy(model(tokens, condition), target)
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    optimizer.step()
                unchanged = [name for name, value in model.named_parameters()
                             if value.requires_grad and torch.equal(before[name], value.detach())]
                self.assertEqual(unchanged, [])

    def test_physical_condition(self) -> None:
        now = torch.tensor([[3.0, 4.0, -2.0]])
        previous = torch.tensor([[0.0, 0.0, 1.0]])
        result = physical_condition(now, previous, 0.5)
        expected = torch.tensor([[2.0, 5.0, 2.0, 1.0, 5.0]])
        self.assertTrue(torch.allclose(result, expected))


if __name__ == "__main__":
    unittest.main()
