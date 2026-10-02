#!/usr/bin/env python3
"""Prove interrupted smoke continuation equals uninterrupted training exactly."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import torch

from common import atomic_json, sha256_file


def exact(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return torch.equal(left, right)
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(exact(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(exact(a, b) for a, b in zip(left, right))
    return left == right


def invoke(trainer: Path, base: list[str], output: Path, interrupt: bool) -> subprocess.CompletedProcess:
    command = [sys.executable, str(trainer), *base, "--output", str(output), "--smoke"]
    if interrupt:
        command += ["--interrupt-after-epoch", "1"]
    environment = os.environ.copy(); environment["XFORMERS_DISABLED"] = "1"
    return subprocess.run(command, env=environment, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", type=Path, required=True, choices=(Path("train_force.py"), Path("train_slip.py")))
    parser.add_argument("--base-args-json", type=Path, required=True,
                        help="JSON array of trainer args excluding --output/--smoke/interruption")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trainer = (Path(__file__).resolve().parent / args.trainer).resolve()
    base = json.loads(args.base_args_json.read_text())
    if not isinstance(base, list) or not all(isinstance(item, str) for item in base):
        raise ValueError("base args must be a JSON string array")
    if any(item in ("--output", "--smoke", "--interrupt-after-epoch") for item in base):
        raise ValueError("base args contain proof-owned arguments")
    root = args.output.resolve(); reference, resumed = root / "reference", root / "interrupted_resumed"
    first = invoke(trainer, base, reference, False)
    interrupted = invoke(trainer, base, resumed, True)
    if first.returncode != 0 or interrupted.returncode == 0 or not (resumed / "latest.pth").is_file():
        raise RuntimeError("Failed to establish reference and committed interrupted checkpoint")
    continuation = invoke(trainer, base, resumed, False)
    if continuation.returncode != 0:
        raise RuntimeError("Interrupted run did not resume successfully")
    ref_latest = torch.load(reference / "latest.pth", map_location="cpu", weights_only=False)
    resumed_latest = torch.load(resumed / "latest.pth", map_location="cpu", weights_only=False)
    ref_best = torch.load(reference / "best.pth", map_location="cpu", weights_only=False)
    resumed_best = torch.load(resumed / "best.pth", map_location="cpu", weights_only=False)
    keys = ("model_state", "optimizer_state", "history", "best_epoch", "best_metric", "patience_wait",
            "normalization", "condition_normalization", "rng_state")
    latest_exact = {key: exact(ref_latest.get(key), resumed_latest.get(key)) for key in keys}
    best_exact = {key: exact(ref_best.get(key), resumed_best.get(key)) for key in keys}
    reference_summary = json.loads((reference / "training_summary.json").read_text())
    resumed_summary = json.loads((resumed / "training_summary.json").read_text())
    prediction_exact = {}
    for role in ("calibration", "validation"):
        left, right = reference / f"predictions_{role}.csv", resumed / f"predictions_{role}.csv"
        if left.exists() or right.exists():
            prediction_exact[role] = left.is_file() and right.is_file() and sha256_file(left) == sha256_file(right)
    proof = {
        "status": "pass" if all((*latest_exact.values(), *best_exact.values(), *prediction_exact.values())) else "fail",
        "trainer": str(trainer), "trainer_sha256": sha256_file(trainer),
        "interrupted_returncode": interrupted.returncode, "resumed_from_epoch": 1,
        "latest_exact": latest_exact, "best_exact": best_exact, "prediction_exact": prediction_exact,
        "history_summary_exact": reference_summary["history"] == resumed_summary["history"],
        "reference_log_tail": first.stdout[-2000:], "interrupted_log_tail": interrupted.stdout[-2000:],
        "continuation_log_tail": continuation.stdout[-2000:],
    }
    if not proof["history_summary_exact"]:
        proof["status"] = "fail"
    atomic_json(root / "recovery_proof.json", proof)
    print(json.dumps(proof, indent=2))
    if proof["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
