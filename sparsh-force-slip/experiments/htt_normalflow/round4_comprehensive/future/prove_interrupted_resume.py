#!/usr/bin/env python3
"""Create an actual stop/resume equivalence proof for the future trainer."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch

from train import sha256, state_hash


def command(args, output: Path) -> list[str]:
    return [sys.executable, str(Path(__file__).with_name("train.py").resolve()),
            "--cache-manifest", str(args.cache_manifest.resolve()), "--support-audit", str(args.support_audit.resolve()),
            "--protocol", str(args.protocol.resolve()), "--model", "mlp", "--seed", "20260914",
            "--output", str(output.resolve()), "--device", "cpu", "--workers", "0"]


def run_to_completion(cmd: list[str], log: Path, env: dict[str, str]) -> None:
    with log.open("a", encoding="utf-8") as handle:
        subprocess.run(cmd, check=True, stdout=handle, stderr=subprocess.STDOUT, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--support-audit", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve(); interrupted, reference = root / "interrupted", root / "reference"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    env = dict(os.environ); env.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", XFORMERS_DISABLED="1")
    cmd = command(args, interrupted)
    first_log = root / "interrupted_first_process.log"
    with first_log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        assert process.stdout is not None
        stopped_epoch = None
        for line in process.stdout:
            handle.write(line); handle.flush()
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("epoch") == 1:
                stopped_epoch = 1; process.terminate(); break
        process.wait(timeout=30)
    latest = torch.load(interrupted / "latest.pth", map_location="cpu", weights_only=False)
    if stopped_epoch != 1 or latest["epoch"] != 1 or (interrupted / "training_summary.json").exists():
        raise RuntimeError("process was not interrupted after durable epoch-1 checkpoint")
    run_to_completion(cmd, root / "interrupted_resume.log", env)
    run_to_completion(command(args, reference), root / "reference.log", env)
    resumed_summary = json.loads((interrupted / "training_summary.json").read_text())
    reference_summary = json.loads((reference / "training_summary.json").read_text())
    resumed_best = torch.load(interrupted / "best.pth", map_location="cpu", weights_only=False)
    reference_best = torch.load(reference / "best.pth", map_location="cpu", weights_only=False)
    predictions_equal = all(sha256(interrupted / "predictions" / f"{role}.csv") ==
                            sha256(reference / "predictions" / f"{role}.csv")
                            for role in ("train", "calibration", "validation"))
    proof = {"status": "complete", "actual_process_terminated": True, "durable_interrupted_epoch": latest["epoch"],
             "history_equal": resumed_summary["epochs_completed"] == reference_summary["epochs_completed"] and
                              json.loads((interrupted / "history.json").read_text()) == json.loads((reference / "history.json").read_text()),
             "best_epoch_equal": resumed_summary["best_epoch"] == reference_summary["best_epoch"],
             "best_model_state_equal": state_hash(resumed_best["model_state"]) == state_hash(reference_best["model_state"]),
             "predictions_byte_equal": predictions_equal,
             "interrupted_output": str(interrupted), "reference_output": str(reference)}
    proof["pass"] = all(proof[key] for key in ("actual_process_terminated", "history_equal", "best_epoch_equal",
                                                "best_model_state_equal", "predictions_byte_equal"))
    (root / "proof.json").write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    if not proof["pass"]:
        raise RuntimeError(proof)
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
