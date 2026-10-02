#!/usr/bin/env python3
"""Exact CPU same-next-step proof from a formal NormalFlow checkpoint."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import torch

import run_normalflow as nf


def optimizer_tensors(opt):
    values = []
    for index in sorted(opt.state_dict()["state"]):
        for key in sorted(opt.state_dict()["state"][index]):
            value = opt.state_dict()["state"][index][key]
            if torch.is_tensor(value): values.append(value)
    return values


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True); parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(); torch.use_deterministic_algorithms(True); torch.set_num_threads(1)
    episodes, _ = nf.load_cache(args.cache, audit_hashes=True); train = nf.build_arrays(episodes, nf.TRAIN_OBJECTS)
    with np.load(args.output_root / "train_only_standardizers.npz") as z:
        stats = {k: {s: z[f"{k}_{s}"] for s in ("mean", "std")} for k in ("z", "delta", "motion")}
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    models, opts = [], []
    for _ in range(2):
        model = nf.DynamicsModel(hidden=payload["config"]["hidden"], auxiliary_motion=payload["variant"] == "D")
        model.load_state_dict(copy.deepcopy(payload["model"]), strict=True); model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=payload["config"]["lr"], weight_decay=payload["config"]["weight_decay"])
        opt.load_state_dict(copy.deepcopy(payload["optimizer"])); models.append(model); opts.append(opt)
    dataset = nf.WindowDataset(train, stats); x, delta, motion, _ = next(iter(torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)))
    st = nf.tensor_stats(stats, "cpu"); losses, gradients = [], []
    for model, opt in zip(models, opts):
        opt.zero_grad(set_to_none=True); pred_delta, pred_motion = model(x)
        feature_loss = torch.mean((nf.reconstruct_raw(x, pred_delta, st) - nf.reconstruct_raw(x, delta, st)) ** 2)
        motion_loss = torch.mean((pred_motion - motion) ** 2) if pred_motion is not None else feature_loss.new_zeros(())
        loss = feature_loss + payload["config"]["motion_weight"] * motion_loss
        loss.backward(); gradients.append([p.grad.detach().clone() for p in model.parameters()]); losses.append(loss.detach().clone()); opt.step()
    proof = {"status": "pass", "device": "cpu", "checkpoint": str(args.checkpoint),
             "loss_exact": torch.equal(losses[0], losses[1]),
             "gradients_exact": all(torch.equal(a, b) for a, b in zip(*gradients)),
             "parameters_exact": all(torch.equal(a, b) for a, b in zip(models[0].state_dict().values(), models[1].state_dict().values())),
             "optimizer_exact": all(torch.equal(a, b) for a, b in zip(optimizer_tensors(opts[0]), optimizer_tensors(opts[1]))),
             "optimizer_tensor_count_equal": len(optimizer_tensors(opts[0])) == len(optimizer_tensors(opts[1])),
             "future_inputs_used": False, "input_requires_grad": bool(x.requires_grad)}
    proof["status"] = "pass" if all(proof[k] for k in ("loss_exact", "gradients_exact", "parameters_exact", "optimizer_exact", "optimizer_tensor_count_equal")) and not proof["input_requires_grad"] else "fail"
    target = args.output or (args.output_root / "same_next_step_proof.json"); tmp = target.with_name(target.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(proof, indent=2) + "\n"); os.replace(tmp, target)
    print(json.dumps(proof, indent=2)); raise SystemExit(proof["status"] != "pass")


if __name__ == "__main__": main()
