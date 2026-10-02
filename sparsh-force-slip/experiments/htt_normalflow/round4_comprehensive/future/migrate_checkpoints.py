#!/usr/bin/env python3
"""Add frozen normalization/provenance to completed R4 checkpoints without retraining."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import torch

from train import SEEDS, atomic_json, atomic_torch, sha256


MODELS = ("mlp", "gru")


def exactly_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(exactly_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(exactly_equal(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original-training-source-sha256", required=True)
    parser.add_argument("--repaired-training-source-sha256", required=True)
    args = parser.parse_args()
    output = args.output.resolve(); archive = output / "original_artifacts"
    if output.exists():
        raise FileExistsError(f"migration output already exists: {output}")
    records = []
    for model in MODELS:
        for seed in SEEDS:
            run = args.runs_root.resolve() / model / f"seed_{seed}"
            summary_path, config_path = run / "training_summary.json", run / "config.json"
            summary = json.loads(summary_path.read_text()); config = json.loads(config_path.read_text())
            if summary["status"] != "complete" or summary["model"] != model or summary["seed"] != seed or summary["config"] != config:
                raise ValueError(f"incompatible completed run: {run}")
            mean_path, std_path = Path(summary["normalization"]["mean"]), Path(summary["normalization"]["std"])
            if sha256(mean_path) != summary["normalization"]["mean_sha256"] or sha256(std_path) != summary["normalization"]["std_sha256"]:
                raise ValueError(f"normalization sidecar hash mismatch: {run}")
            mean = np.load(mean_path, allow_pickle=False); std = np.load(std_path, allow_pickle=False)
            destination = archive / model / f"seed_{seed}"; destination.mkdir(parents=True)
            original_hashes = {}
            for name in ("best.pth", "latest.pth", "training_summary.json", "config.json"):
                source = run / name; shutil.copy2(source, destination / name); original_hashes[name] = sha256(source)
            prediction_hashes = {role: sha256(Path(info["path"])) for role, info in summary["predictions"].items()}
            checkpoint_records = {}
            for name in ("best.pth", "latest.pth"):
                path = run / name; before = torch.load(path, map_location="cpu", weights_only=False)
                if "normalization" in before or "provenance" in before:
                    raise ValueError(f"checkpoint already migrated: {path}")
                protected = {key: copy.deepcopy(before[key]) for key in before}
                after = copy.deepcopy(before)
                after["normalization"] = {"mean": mean.copy(), "std": std.copy(),
                                          "source": "all history steps from eligible train examples only"}
                after["provenance"] = {"cache_manifest_sha256": config["cache_manifest_sha256"],
                                       "support_audit_sha256": config["support_audit_sha256"],
                                       "protocol_sha256": config["protocol_sha256"]}
                atomic_torch(path, after)
                reloaded = torch.load(path, map_location="cpu", weights_only=False)
                unchanged = all(exactly_equal(protected[key], reloaded[key]) for key in protected)
                normalization_exact = np.array_equal(reloaded["normalization"]["mean"], mean) and np.array_equal(reloaded["normalization"]["std"], std)
                if not unchanged or not normalization_exact:
                    raise RuntimeError(f"checkpoint migration altered protected state: {path}")
                checkpoint_records[name] = {"original_sha256": original_hashes[name], "migrated_sha256": sha256(path),
                                            "model_optimizer_rng_history_config_bitwise_equal": unchanged,
                                            "normalization_sidecars_exact": normalization_exact}
            summary["best_checkpoint_sha256"] = sha256(run / "best.pth")
            summary["latest_checkpoint_sha256"] = sha256(run / "latest.pth")
            summary["checkpoint_metadata_migration"] = {
                "reason": "embed already-frozen hashed train-only normalization and provenance required by PROTOCOL; no retraining",
                "original_training_source_sha256": args.original_training_source_sha256,
                "repaired_training_source_sha256": args.repaired_training_source_sha256,
                "original_summary_sha256": original_hashes["training_summary.json"],
                "original_artifact_archive": str(destination), "checkpoints": checkpoint_records}
            atomic_json(summary_path, summary)
            if any(sha256(Path(info["path"])) != prediction_hashes[role] for role, info in summary["predictions"].items()):
                raise RuntimeError(f"prediction changed during migration: {run}")
            records.append({"model": model, "seed": seed, "run": str(run), "archive": str(destination),
                            "original_hashes": original_hashes, "migrated_summary_sha256": sha256(summary_path),
                            "prediction_hashes_unchanged": True, "checkpoints": checkpoint_records})
    report = {"status": "complete", "runs": records, "run_count": len(records),
              "training_math_changed": False, "retraining_performed": False,
              "original_training_source_sha256": args.original_training_source_sha256,
              "repaired_training_source_sha256": args.repaired_training_source_sha256}
    atomic_json(output / "migration_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
