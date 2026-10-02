#!/usr/bin/env python3
"""Final artifact audit for the six-run R4 future experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train import SEEDS, atomic_json, sha256


MODELS = ("mlp", "gru")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    root, protocol = args.root.resolve(), args.protocol.resolve()
    checks = []

    def add(name: str, passed: bool, detail=None) -> None:
        checks.append({"name": name, "pass": bool(passed), "detail": detail})

    expected = {(model, seed) for model in MODELS for seed in SEEDS}
    found = {(path.parent.name, int(path.name.removeprefix("seed_")))
             for path in (root / "runs").glob("*/seed_*") if path.is_dir()}
    add("exact_six_run_identities", found == expected, sorted(found))
    protocol_sha = sha256(protocol)
    for model, seed in sorted(expected):
        run = root / "runs" / model / f"seed_{seed}"
        summary = json.loads((run / "training_summary.json").read_text())
        add(f"{model}_{seed}_complete_identity", summary.get("status") == "complete" and
            summary.get("model") == model and summary.get("seed") == seed)
        add(f"{model}_{seed}_protocol", summary["config"].get("protocol_sha256") == protocol_sha and
            summary["config"].get("horizon") == 8 and summary["config"].get("history") == 4 and
            summary["config"].get("train_counts_negative_positive") == [332, 84])
        mean = np.load(summary["normalization"]["mean"], allow_pickle=False)
        std = np.load(summary["normalization"]["std"], allow_pickle=False)
        for name, summary_key in (("best.pth", "best_checkpoint_sha256"), ("latest.pth", "latest_checkpoint_sha256")):
            path = run / name; payload = torch.load(path, map_location="cpu", weights_only=False)
            add(f"{model}_{seed}_{name}_hash", sha256(path) == summary[summary_key])
            add(f"{model}_{seed}_{name}_normalization", np.array_equal(payload["normalization"]["mean"], mean) and
                np.array_equal(payload["normalization"]["std"], std))
            add(f"{model}_{seed}_{name}_provenance", payload["provenance"] == {
                "cache_manifest_sha256": summary["config"]["cache_manifest_sha256"],
                "support_audit_sha256": summary["config"]["support_audit_sha256"],
                "protocol_sha256": summary["config"]["protocol_sha256"]})
        for role, info in summary["predictions"].items():
            add(f"{model}_{seed}_{role}_predictions", sha256(Path(info["path"])) == info["sha256"])
        migration = summary.get("checkpoint_metadata_migration", {})
        add(f"{model}_{seed}_migration_trace", bool(migration.get("original_artifact_archive")) and
            Path(migration.get("original_artifact_archive", "")).is_dir())
    support = json.loads((root / "support_audit" / "support_audit.json").read_text())
    add("train_only_support_gate", support.get("decision", {}).get("status") == "train" and
        support["decision"].get("horizon") == 8 and support["primary_current_static"]["8"] == {
            "eligible_frames": 416, "negative_episodes": 24, "negative_frames": 332,
            "positive_episodes": 31, "positive_frames": 84,
            "requirements": {"negative_episodes": 10, "negative_frames": 100, "positive_episodes": 5, "positive_frames": 20},
            "supported": True})
    cache = json.loads((root / "cache" / "cache_manifest.json").read_text())
    add("causal_cache", cache.get("status") == "complete" and len(cache.get("entries", [])) == 75 and
        cache.get("ground_truth_force_used_as_input") is False and cache.get("future_frames_used_as_input") is False and
        cache.get("prediction_compatibility", {}).get("all_within_atol_1e-6_rtol_1e-5") is True)
    migration = json.loads((root / "checkpoint_payload_migration" / "migration_report.json").read_text())
    add("checkpoint_migration", migration.get("status") == "complete" and migration.get("run_count") == 6 and
        migration.get("training_math_changed") is False and migration.get("retraining_performed") is False and
        all(row["prediction_hashes_unchanged"] and all(item["model_optimizer_rng_history_config_bitwise_equal"]
            for item in row["checkpoints"].values()) for row in migration["runs"]))
    proof = json.loads((root / "interrupted_resume_proof" / "proof.json").read_text())
    add("actual_interrupted_resume", proof.get("pass") is True and proof.get("actual_process_terminated") is True)
    evaluation = json.loads((root / "evaluation" / "evaluation.json").read_text())
    add("formal_evaluation", evaluation.get("status") == "complete" and len(evaluation.get("runs", [])) == 6 and
        len(evaluation.get("results", [])) == 32)
    source_hashes = {path.name: sha256(path) for path in args.source_dir.resolve().glob("*.py")}
    result = {"status": "pass" if all(row["pass"] for row in checks) else "fail",
              "checks": checks, "passed": sum(row["pass"] for row in checks),
              "failed": sum(not row["pass"] for row in checks), "source_hashes": source_hashes,
              "protocol_sha256": protocol_sha}
    atomic_json(root / "final_audit.json", result)
    print(json.dumps({"status": result["status"], "passed": result["passed"], "failed": result["failed"]}, indent=2))
    if result["status"] != "pass":
        raise RuntimeError([row for row in checks if not row["pass"]])


if __name__ == "__main__":
    main()
