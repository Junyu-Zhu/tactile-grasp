"""Load only the historical decoder branches needed by Round 5."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import sys

os.environ.setdefault("XFORMERS_DISABLED", "1")
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import phase2_b_multitask as p2


def load_decoupled_decoder(checkpoint: Path) -> tuple[torch.nn.Module, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("train_config", {})
    if config.get("encoder") != "mae" or config.get("decoder_variant") != "decoupled":
        raise RuntimeError("Expected the selected MAE decoupled historical checkpoint")
    state = payload.get("model_state", {})
    decoder_state = {name.removeprefix("decoder."): value for name, value in state.items()
                     if name.startswith("decoder.")}
    decoder = p2.build_decoder("decoupled")
    decoder.load_state_dict(decoder_state, strict=True)
    return decoder, payload


def load_r3_slip_branch(source_decoder: torch.nn.Module, checkpoint: Path) -> torch.nn.Module:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("format") != "round3_mae_slip_branch_v1" or payload.get("config", {}).get("init") != "fresh":
        raise RuntimeError("Expected a formal Round 3-B fresh slip checkpoint")
    branch = copy.deepcopy(source_decoder)
    keep = ("slip_pooler", "slip_trunk", "slip_head")
    for name in tuple(branch._modules):
        if name not in keep:
            del branch._modules[name]
    branch.load_state_dict(payload["branch_state"], strict=True)
    return branch
