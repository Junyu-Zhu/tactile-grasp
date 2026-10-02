#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import adapters  # noqa: E402
import cache  # noqa: E402
import training  # noqa: E402


class EncoderRound4Tests(unittest.TestCase):
    def test_fresh_branch_boundary_and_reconstruction(self) -> None:
        branch, provenance = training.branch_from_source("dino", 20260914, torch.device("cpu"))
        self.assertEqual(len(list(branch.parameters())), 23)
        self.assertEqual(sum(parameter.numel() for parameter in branch.parameters()), 7_459_778)
        self.assertTrue(provenance["initialization_proof"]["independent_fresh_exact"])
        self.assertFalse(provenance["encoder_parameters_in_optimizer"])
        self.assertFalse(provenance["force_parameters_in_optimizer"])

    def test_letterbox_is_full_six_channel_input_with_neutral_padding(self) -> None:
        manifest = json.loads(Path(
            "/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round1/splits.json"
        ).read_text())
        row = next(
            item for item in manifest["episodes"]
            if item["domain"] == "htt" and any(path.endswith(".labeled.npz") for path in item["source_files"])
        )
        episode = adapters.load_htt(row["path"])
        value = cache.encoder_input(episode, 0, "mae_letterbox")
        self.assertEqual(tuple(value.shape), (6, 320, 240))
        neutral = torch.tensor(127.0 / 255.0)
        self.assertTrue(torch.equal(value[:, :40], torch.full_like(value[:, :40], neutral)))
        self.assertTrue(torch.equal(value[:, 280:], torch.full_like(value[:, 280:], neutral)))


if __name__ == "__main__":
    unittest.main()
