#!/usr/bin/env python3
"""Actual interrupted-vs-uninterrupted proof for the Round 4 DINO trainer."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

import training


def exact(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(exact(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(exact(a, b) for a, b in zip(left, right))
    return left == right


def args_for(cache_dir: Path, output: Path) -> SimpleNamespace:
    return SimpleNamespace(
        cache_dir=cache_dir,
        encoder="dino",
        fold="htt_leave_p1",
        seed=20260914,
        output_dir=output,
        device="cpu",
        workers=0,
        batch_size=8,
        smoke=True,
        smoke_max_samples=16,
    )


def placeholder_predictions(branch, cache, fold, role, output, *unused) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("episode_id,t,stage,p_slip,p_static,p_gross,fold,seed,init,checkpoint_epoch\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    original_prediction_writer = training.write_predictions
    original_saver = training.atomic_torch_save
    training.write_predictions = placeholder_predictions
    try:
        with tempfile.TemporaryDirectory(prefix="r4_encoder_recovery_") as temp_name:
            temp = Path(temp_name)
            uninterrupted = temp / "uninterrupted"
            resumed = temp / "resumed"
            training.run(args_for(args.cache_dir, uninterrupted))

            interrupted = {"raised": False}

            def interrupt_after_first_latest(path: Path, value: Any) -> None:
                original_saver(path, value)
                if path.name == "latest.pth" and value.get("epoch") == 1 and not interrupted["raised"]:
                    interrupted["raised"] = True
                    raise RuntimeError("intentional interruption after committed epoch-1 latest")

            training.atomic_torch_save = interrupt_after_first_latest
            caught = False
            try:
                training.run(args_for(args.cache_dir, resumed))
            except RuntimeError as error:
                caught = str(error) == "intentional interruption after committed epoch-1 latest"
            finally:
                training.atomic_torch_save = original_saver
            if not caught or not interrupted["raised"]:
                raise RuntimeError("Intentional interruption was not observed")
            training.run(args_for(args.cache_dir, resumed))

            full = torch.load(uninterrupted / "latest.pth", map_location="cpu", weights_only=False)
            recovered = torch.load(resumed / "latest.pth", map_location="cpu", weights_only=False)
            checks = {
                "epoch_equal": full["epoch"] == recovered["epoch"] == 2,
                "branch_state_exact": exact(full["branch_state"], recovered["branch_state"]),
                "optimizer_state_exact": exact(full["optimizer_state"], recovered["optimizer_state"]),
                "history_exact": exact(full["history"], recovered["history"]),
                "best_selection_exact": exact(
                    (full["best_epoch"], full["best_balanced_accuracy"], full["patience_wait"]),
                    (recovered["best_epoch"], recovered["best_balanced_accuracy"], recovered["patience_wait"]),
                ),
                "rng_state_exact": exact(full["rng_state"], recovered["rng_state"]),
                "configuration_exact": exact(full["config"], recovered["config"]),
                "provenance_exact": exact(full["provenance"], recovered["provenance"]),
            }
            remnants = [str(path) for path in temp.rglob("*.tmp.*")]
            payload = {
                "status": "pass" if all(checks.values()) and not remnants else "fail",
                "comparison": "uninterrupted 2 epochs vs interruption after committed epoch 1 then resumed to epoch 2",
                "encoder": "dino",
                "device": "cpu",
                "smoke_samples_per_role_max": 16,
                "checks": checks,
                "temporary_files_remaining_before_cleanup": remnants,
                "training_source": str(Path(training.__file__).resolve()),
                "training_source_sha256": training.sha256_file(Path(training.__file__).resolve()),
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            tmp = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
            os.replace(tmp, args.output)
            print(json.dumps(payload, indent=2))
            if payload["status"] != "pass":
                raise SystemExit(1)
    finally:
        training.write_predictions = original_prediction_writer
        training.atomic_torch_save = original_saver


if __name__ == "__main__":
    main()
