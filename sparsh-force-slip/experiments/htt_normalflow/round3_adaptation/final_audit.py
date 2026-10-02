#!/usr/bin/env python3
"""Read-only acceptance audit for the 24 formal Round-3 B/C training runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
os.environ.setdefault("XFORMERS_DISABLED", "1")

import phase2_b_multitask as p2  # noqa: E402


FOLDS = tuple(f"htt_leave_p{i}" for i in range(1, 5))
SEEDS = (20260914, 20260915, 20260916)
VARIANTS = {"B": "fresh", "C": "old"}
EXPECTED_RUNS = tuple((variant, fold, seed) for variant in VARIANTS for fold in FOLDS for seed in SEEDS)
EXPECTED_CHECKPOINT_SHA = "850e4a74e6d9bd60e8ed22efc8b70d343c5d21bb5f60b3053b0a800674dffcb4"
EXPECTED_ENCODER_SHA = "60ab1af933e4c9a6791e8090554ded0f2229f09c5303a2a2a97ba66922c1eea8"
EXPECTED_PROTOCOL = {
    "optimizer": "AdamW",
    "learning_rate": 1.0e-4,
    "weight_decay": 1.0e-4,
    "batch_size": 128,
    "max_epochs": 30,
    "patience": 7,
    "selection": "earliest strict maximum validation balanced accuracy at threshold 0.5",
    "labels": "static=0, gross=1; incipient excluded from loss/primary metrics",
    "precision": "FP32",
    "smoke": False,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "pass": bool(passed), "detail": detail})


def same_number(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isfinite(float(left)) and math.isfinite(float(right)) and abs(float(left) - float(right)) <= tolerance


def run_path(root: Path, variant: str, fold: str, seed: int) -> Path:
    probe = fold.removeprefix("htt_leave_")
    return root / "runs" / variant / f"fold_{probe}" / f"seed_{seed}"


def load_source_state(checkpoint: Path) -> dict[str, Any]:
    model, payload = p2.load_b_checkpoint(checkpoint, torch.device("cpu"))
    encoder = tensor_state_sha256(model.encoder.state_dict())
    force = tensor_state_sha256({
        key: value for key, value in model.decoder.state_dict().items() if key.startswith("force_")
    })
    # SlipBranch state keys retain slip_pooler/slip_trunk/slip_head, so compute
    # its exact reference through the same wrapper locally without importing
    # the mutable Round-3 trainer.
    old_slip = {
        key: value for key, value in model.decoder.state_dict().items() if key.startswith("slip_")
    }
    return {
        "file_sha256": sha256_file(checkpoint),
        "format": payload.get("format"),
        "epoch": payload.get("epoch"),
        "encoder_state_sha256": encoder,
        "force_state_sha256": force,
        "old_slip_decoder_state_sha256": tensor_state_sha256(old_slip),
    }


def expected_slip_keys() -> set[str]:
    decoder = p2.build_decoder("decoupled")
    return {
        key for key in decoder.state_dict()
        if key.startswith(("slip_pooler.", "slip_trunk.", "slip_head."))
    }


def audit_run(
    root: Path,
    variant: str,
    fold: str,
    seed: int,
    audit: dict[str, Any],
    source: dict[str, Any],
    slip_keys: set[str],
    expected_artifacts: dict[str, str],
) -> dict[str, Any]:
    directory = run_path(root, variant, fold, seed)
    identity = f"{variant}/{fold}/{seed}"
    required = {
        "config": directory / "config.json",
        "summary": directory / "training_summary.json",
        "history": directory / "history.json",
        "latest": directory / "latest.pth",
        "best": directory / "best.pth",
        "calibration": directory / "predictions" / "calibration.csv",
        "validation": directory / "predictions" / "validation.csv",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        return {"identity": identity, "status": "incomplete", "directory": str(directory), "missing": missing}

    checks: list[dict[str, Any]] = []
    try:
        config = json.loads(required["config"].read_text())
        summary = json.loads(required["summary"].read_text())
        history = json.loads(required["history"].read_text())
        latest = torch.load(required["latest"], map_location="cpu", weights_only=False)
        best = torch.load(required["best"], map_location="cpu", weights_only=False)

        add(checks, "summary_complete", summary.get("status") == "complete", summary.get("status"))
        add(checks, "identity", (
            config.get("fold") == fold and config.get("seed") == seed
            and config.get("init") == VARIANTS[variant]
            and summary.get("fold") == fold and summary.get("seed") == seed
            and summary.get("init") == VARIANTS[variant]
        ), {"config": {key: config.get(key) for key in ("fold", "seed", "init")}})
        add(checks, "formal_protocol", all(config.get(key) == value for key, value in EXPECTED_PROTOCOL.items()), {
            key: config.get(key) for key in EXPECTED_PROTOCOL
        })
        add(checks, "frozen_artifact_hashes", all(
            config.get(key) == value for key, value in expected_artifacts.items()
        ), {"expected": expected_artifacts, "actual": {key: config.get(key) for key in expected_artifacts}})
        add(checks, "summary_config_exact", summary.get("config") == config)
        add(checks, "checkpoint_configs_exact", latest.get("config") == config and best.get("config") == config)

        provenance = summary.get("provenance", {})
        add(checks, "checkpoint_provenance_exact", latest.get("provenance") == provenance and best.get("provenance") == provenance)
        add(checks, "source_checkpoint_unchanged", (
            provenance.get("source_checkpoint_sha256") == source["file_sha256"] == EXPECTED_CHECKPOINT_SHA
        ), provenance.get("source_checkpoint_sha256"))
        add(checks, "source_encoder_unchanged", (
            provenance.get("source_encoder_state_sha256") == source["encoder_state_sha256"] == EXPECTED_ENCODER_SHA
        ), provenance.get("source_encoder_state_sha256"))
        add(checks, "source_force_unchanged", provenance.get("source_force_state_sha256") == source["force_state_sha256"])
        add(checks, "source_old_slip_reference_exact", (
            provenance.get("old_slip_state_sha256") == source["old_slip_decoder_state_sha256"]
        ))
        add(checks, "optimizer_excludes_frozen_parts", (
            provenance.get("encoder_parameters_in_optimizer") == []
            and provenance.get("force_parameters_in_optimizer") == []
        ))
        initialization = provenance.get("initialization_proof", {})
        if variant == "B":
            add(checks, "fresh_initialization", (
                initialization.get("independent_fresh_exact") is True
                and len(initialization.get("fresh_keys_differing_from_old", [])) > 0
                and provenance.get("initial_slip_state_sha256") != provenance.get("old_slip_state_sha256")
            ), initialization)
        else:
            add(checks, "old_initialization", (
                initialization.get("old_branch_exact") is True
                and provenance.get("initial_slip_state_sha256") == provenance.get("old_slip_state_sha256")
            ), initialization)

        add(checks, "history_nonempty_contiguous", (
            isinstance(history, list) and bool(history)
            and [row.get("epoch") for row in history] == list(range(1, len(history) + 1))
        ), {"rows": len(history) if isinstance(history, list) else None})
        if history:
            scores = [float(row["balanced_accuracy"]) for row in history]
            maximum = max(scores)
            earliest_best = scores.index(maximum) + 1
            add(checks, "best_is_earliest_maximum", (
                best.get("epoch") == earliest_best
                and best.get("best_epoch") == earliest_best
                and same_number(best.get("best_balanced_accuracy"), maximum)
                and summary.get("best_epoch") == earliest_best
                and same_number(summary.get("best_balanced_accuracy_at_0_5"), maximum)
            ), {"earliest_best": earliest_best, "maximum": maximum, "saved_best": best.get("epoch")})
            add(checks, "latest_matches_full_history", (
                latest.get("epoch") == len(history)
                and latest.get("history") == history
                and latest.get("best_epoch") == earliest_best
                and same_number(latest.get("best_balanced_accuracy"), maximum)
            ))
            add(checks, "best_history_is_prefix", best.get("history") == history[:best.get("epoch", 0)])
            stopped_correctly = len(history) == EXPECTED_PROTOCOL["max_epochs"] or (
                len(history) < EXPECTED_PROTOCOL["max_epochs"]
                and latest.get("patience_wait", -1) >= EXPECTED_PROTOCOL["patience"]
            )
            add(checks, "stopping_rule", stopped_correctly, {
                "epochs": len(history), "patience_wait": latest.get("patience_wait")
            })
            add(checks, "finite_history", all(
                row.get("finite_gradients") is True
                and math.isfinite(float(row.get("train_loss", float("nan"))))
                and math.isfinite(float(row.get("balanced_accuracy", float("nan"))))
                for row in history
            ))

        add(checks, "summary_history_length", summary.get("epochs_completed") == len(history))
        add(checks, "parameters_changed_and_finite", (
            summary.get("slip_parameters_changed") is True
            and summary.get("finite_gradients_all_epochs") is True
        ))
        add(checks, "artifact_hashes", (
            sha256_file(required["best"]) == summary.get("best_checkpoint_sha256")
            and sha256_file(required["latest"]) == summary.get("latest_checkpoint_sha256")
            and sha256_file(required["calibration"]) == summary.get("prediction_sha256", {}).get("calibration")
            and sha256_file(required["validation"]) == summary.get("prediction_sha256", {}).get("validation")
        ))
        add(checks, "branch_state_keys", set(best.get("branch_state", {})) == slip_keys and len(slip_keys) == 23, {
            "expected": len(slip_keys), "actual": len(best.get("branch_state", {}))
        })
        add(checks, "latest_branch_state_keys", set(latest.get("branch_state", {})) == slip_keys)
        add(checks, "branch_states_finite", all(
            torch.isfinite(value).all().item()
            for checkpoint_payload in (best, latest)
            for value in checkpoint_payload["branch_state"].values()
        ))

        expected_support = audit["manifest"]["folds"][fold]["development_label_support"]["train"]
        expected_counts = [expected_support["primary_static_frames"], expected_support["primary_gross_frames"]]
        actual_counts = summary.get("class_counts_train_static_gross")
        add(checks, "train_class_counts_match_audit", actual_counts == expected_counts, {
            "expected": expected_counts, "actual": actual_counts
        })
        expected_weights = [sum(expected_counts) / (2.0 * count) for count in expected_counts]
        actual_weights = summary.get("class_weights_train_only", [])
        add(checks, "train_only_class_weights", (
            len(actual_weights) == 2
            and all(same_number(actual, expected) for actual, expected in zip(actual_weights, expected_weights))
        ), {"expected": expected_weights, "actual": actual_weights})
        add(checks, "role_isolation", summary.get("role_isolation", {}).get("pass") is True)
        add(checks, "cache_roles_match_manifest", summary.get("cache_role_proof", {}).get("cache_roles_match_frozen_manifest") is True)

        return {
            "identity": identity,
            "directory": str(directory),
            "status": "pass" if all(item["pass"] for item in checks) else "fail",
            "checks": checks,
        }
    except Exception as error:  # Preserve all other run results in one audit artifact.
        return {"identity": identity, "directory": str(directory), "status": "fail", "error": repr(error), "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text())
    source = load_source_state(args.checkpoint)
    slip_keys = expected_slip_keys()
    expected_artifacts = {
        "training_source_sha256": sha256_file(HERE / "training.py"),
        "protocol_sha256": sha256_file(HERE / "PROTOCOL.md"),
        "cache_manifest_sha256": sha256_file(args.root / "cache" / "cache_manifest.json"),
        "source_checkpoint_sha256": source["file_sha256"],
    }
    runs = [audit_run(args.root, variant, fold, seed, audit, source, slip_keys, expected_artifacts)
            for variant, fold, seed in EXPECTED_RUNS]
    incomplete = [run["identity"] for run in runs if run["status"] == "incomplete"]
    failed = [run["identity"] for run in runs if run["status"] == "fail"]

    invariant_configs = []
    for run in runs:
        if run["status"] == "incomplete":
            continue
        config_path = Path(run["directory"]) / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            invariant_configs.append({key: config.get(key) for key in (
                "training_source_sha256", "protocol_sha256", "cache_manifest_sha256",
                "source_checkpoint_sha256", "batch_size", "workers", "device",
            )})
    invariant_exact = bool(invariant_configs) and all(item == invariant_configs[0] for item in invariant_configs)
    if len(runs) == len(EXPECTED_RUNS) and not incomplete and not failed and invariant_exact:
        status = "pass"
    elif incomplete and not failed:
        status = "incomplete"
    else:
        status = "fail"
    report = {
        "format": "round3_final_training_acceptance_v1",
        "status": status,
        "expected_runs": len(EXPECTED_RUNS),
        "complete_filesets": len(runs) - len(incomplete),
        "passed_runs": sum(run["status"] == "pass" for run in runs),
        "incomplete_runs": incomplete,
        "failed_runs": failed,
        "cross_run_invariant_config_exact": invariant_exact,
        "cross_run_invariant_configs": invariant_configs,
        "source": source,
        "expected_artifact_hashes": expected_artifacts,
        "audit": str(args.audit),
        "audit_status": audit.get("status"),
        "runs": runs,
    }
    if audit.get("status") != "pass" or not invariant_exact and not incomplete:
        report["status"] = "fail"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({
        "status": report["status"], "passed_runs": report["passed_runs"],
        "incomplete_runs": len(incomplete), "failed_runs": len(failed), "output": str(args.output),
    }, indent=2))
    raise SystemExit(0 if report["status"] == "pass" else 2 if report["status"] == "incomplete" else 1)


if __name__ == "__main__":
    main()
