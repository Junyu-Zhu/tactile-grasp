#!/usr/bin/env python3
"""Verify the complete Round 4 encoder adaptation artifact set."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


FOLDS = tuple(f"htt_leave_p{i}" for i in range(1, 5))
SEEDS = (20260914, 20260915, 20260916)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--encoders", nargs="+", choices=("dino", "ijepa", "mae_letterbox"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    errors: list[str] = []
    runs: list[dict[str, Any]] = []
    caches: dict[str, Any] = {}
    for encoder in args.encoders:
        cache_path = args.output_root / "cache" / encoder / "cache_manifest.json"
        if not cache_path.exists():
            errors.append(f"missing cache manifest: {encoder}")
            continue
        cache = json.loads(cache_path.read_text())
        cache_ok = (
            cache.get("status") == "complete"
            and cache.get("format") == "round4_htt_full_sparsh_tokens_v1"
            and cache.get("encoder") == encoder
            and cache.get("constructor_seed") == 42
            and cache.get("unique_labeled_episodes") == cache.get("expected_unique_labeled_episodes") == 101
        )
        caches[encoder] = {
            "status": cache.get("status"),
            "episodes": cache.get("unique_labeled_episodes"),
            "frames": cache.get("frames"),
            "encoder_state_sha256": cache.get("encoder_state_sha256"),
            "manifest_sha256": sha256_file(cache_path),
            "pass": cache_ok,
        }
        if not cache_ok:
            errors.append(f"incompatible cache manifest: {encoder}")
        for fold in FOLDS:
            for seed in SEEDS:
                summary_path = args.output_root / "runs" / encoder / fold / str(seed) / "training_summary.json"
                if not summary_path.exists():
                    errors.append(f"missing run: {encoder}/{fold}/{seed}")
                    continue
                summary = json.loads(summary_path.read_text())
                row_errors = []
                if summary.get("status") != "complete":
                    row_errors.append("status")
                if summary.get("encoder") != encoder or summary.get("fold") != fold or summary.get("seed") != seed:
                    row_errors.append("identity")
                proof = summary.get("provenance", {}).get("initialization_proof", {})
                if proof.get("trainable_parameter_tensors") != 23 or proof.get("trainable_parameters") != 7_459_778:
                    row_errors.append("trainable_boundary")
                if not summary.get("slip_parameters_changed") or not summary.get("finite_gradients_all_epochs"):
                    row_errors.append("training_proof")
                if not summary.get("checkpoint_restore_exact") or not summary.get("role_isolation", {}).get("pass"):
                    row_errors.append("restore_or_isolation")
                for key in ("best_checkpoint", "latest_checkpoint"):
                    artifact = Path(summary.get(key, ""))
                    if not artifact.is_file() or sha256_file(artifact) != summary.get(f"{key}_sha256"):
                        row_errors.append(key)
                for role in ("calibration", "validation"):
                    artifact = Path(summary.get("predictions", {}).get(role, ""))
                    if not artifact.is_file() or sha256_file(artifact) != summary.get("prediction_sha256", {}).get(role):
                        row_errors.append(f"predictions_{role}")
                if row_errors:
                    errors.append(f"invalid run {encoder}/{fold}/{seed}: {','.join(row_errors)}")
                runs.append({
                    "encoder": encoder,
                    "fold": fold,
                    "seed": seed,
                    "best_epoch": summary.get("best_epoch"),
                    "best_balanced_accuracy_at_0_5": summary.get("best_balanced_accuracy_at_0_5"),
                    "pass": not row_errors,
                })
    payload = {
        "status": "pass" if not errors else "fail",
        "expected_runs": len(args.encoders) * len(FOLDS) * len(SEEDS),
        "verified_runs": len(runs),
        "caches": caches,
        "runs": runs,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, args.output)
    print(json.dumps(payload, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
