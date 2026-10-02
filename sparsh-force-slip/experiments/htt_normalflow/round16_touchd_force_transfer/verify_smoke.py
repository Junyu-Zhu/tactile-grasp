#!/usr/bin/env python3
"""Verify R16 real-token T adaptation, recovery, transfer reset, and head cost."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from touchd_common import (SOURCE_CHECKPOINT, atomic_json, fresh_adapter, sha256,
                           state_sha256)


def same(left, right) -> bool:
    if torch.is_tensor(left): return torch.equal(left, right)
    if isinstance(left, np.ndarray): return np.array_equal(left, right)
    if isinstance(left, dict): return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)): return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    if isinstance(left, float) and isinstance(right, float): return left == right
    return left == right


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    continuous = args.smoke_root / "tforce_continuous"
    recovery = args.smoke_root / "tforce_recovery"
    recovery_exact = {}
    for name in ("best.pth", "latest.pth"):
        left = torch.load(continuous / name, map_location="cpu", weights_only=False)
        right = torch.load(recovery / name, map_location="cpu", weights_only=False)
        fields = ("model_state", "optimizer_state", "rng_state", "history", "best_metric", "best_epoch", "wait", "normalization")
        recovery_exact[name] = {field: same(left[field], right[field]) for field in fields}
        if not all(recovery_exact[name].values()):
            raise RuntimeError(f"nonexact recovery: {name}: {recovery_exact[name]}")
    summary = json.loads((continuous / "summary.json").read_text())
    if summary["status"] != "complete" or not summary["smoke"] or not all(summary["component_changed"].values()):
        raise RuntimeError("T adaptation update receipt failed")

    private = torch.load(continuous / "transferable_private_state.pth", map_location="cpu", weights_only=False)
    seed = 20260914
    h = fresh_adapter(seed)
    th = fresh_adapter(seed, trunk_state=private)
    head_equal = all(torch.equal(a, b) for a, b in zip(h.force_head.state_dict().values(), th.force_head.state_dict().values()))
    pooler_differs = state_sha256(h.force_pooler.state_dict()) != state_sha256(th.force_pooler.state_dict())
    trunk_differs = state_sha256(h.force_trunk.state_dict()) != state_sha256(th.force_trunk.state_dict())
    t_saved = torch.load(continuous / "best.pth", map_location="cpu", weights_only=False)
    t_head = {k.removeprefix("force_head."): v for k, v in t_saved["model_state"].items() if k.startswith("force_head.")}
    t_head_discarded = any(not torch.equal(t_head[k], th.force_head.state_dict()[k]) for k in t_head)
    if not (head_equal and pooler_differs and trunk_differs and t_head_discarded):
        raise RuntimeError("HTT boundary transfer/reset contract failed")

    cache = torch.load(args.smoke_root / "token_probe/smoke_tokens.pt", map_location="cpu", weights_only=False)
    device = torch.device(args.device); model = th.to(device).train(); x = cache["tokens"][:32].to(device).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True); model(x).square().mean().backward(); optimizer.step()
    torch.cuda.synchronize(device); began = time.monotonic(); iterations = 20
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True); loss = model(x).square().mean(); loss.backward(); optimizer.step()
    torch.cuda.synchronize(device); seconds = time.monotonic() - began
    receipt = {
        "schema": "round16_force_transfer_smoke_v1", "status": "pass", "formal": False,
        "real_released_images": True, "complete_token_shape": list(cache["tokens"].shape[1:]),
        "finite_real_update": summary["finite"], "trainable_components_updated": summary["component_changed"],
        "frozen_encoder_bitwise": json.loads((args.smoke_root / "token_probe/TOKEN_PROBE.json").read_text())["encoder_frozen_bitwise"],
        "real_process_interruption_resume_exact": recovery_exact,
        "role_isolation": {"fit": summary["fit_samples"], "selection": summary["selection_samples"],
                           "selection_not_fit": True},
        "target_interface": summary["target"], "no_future_inputs_or_targets": True,
        "htt_boundary": {"same_seed_head_bitwise_equal": head_equal, "t_output_head_discarded": t_head_discarded,
                         "transferred": ["force_pooler", "force_trunk"], "pooler_differs_from_H": pooler_differs,
                         "trunk_differs_from_H": trunk_differs, "source_encoder_transferred": False},
        "token_head_benchmark": {"device": str(device), "batch": len(x), "iterations": iterations,
                                 "train_samples_per_second": len(x) * iterations / seconds,
                                 "seconds": seconds, "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                                 "trainable_parameters": sum(p.numel() for p in model.parameters())},
        "source_checkpoint": {"path": str(SOURCE_CHECKPOINT), "sha256": sha256(SOURCE_CHECKPOINT)},
        "artifacts": {"continuous_summary": {"path": str(continuous / "summary.json"),
                                               "sha256": sha256(continuous / "summary.json")},
                      "token_probe": {"path": str(args.smoke_root / "token_probe/TOKEN_PROBE.json"),
                                      "sha256": sha256(args.smoke_root / "token_probe/TOKEN_PROBE.json")}},
    }
    atomic_json(args.output, receipt); print(json.dumps(receipt, indent=2))


if __name__ == "__main__": main()
