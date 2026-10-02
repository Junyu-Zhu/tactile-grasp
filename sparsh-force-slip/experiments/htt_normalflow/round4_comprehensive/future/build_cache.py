#!/usr/bin/env python3
"""Combine frozen round-2 state/force with the predetermined round-3 slip head."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("XFORMERS_DISABLED", "1")
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parents[1]
REPO_ROOT = EXP_ROOT.parents[1]
ROUND3 = EXP_ROOT / "round3_adaptation"
sys.path.insert(0, str(ROUND3))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import training as r3_training  # noqa: E402
import phase2_b_multitask as p2  # noqa: E402


FOLD = "htt_leave_p1"
EXPECTED_R2_INDEX = "dd2cc134d6ee2399148ba4bcb72bd2fcce7abe667c077e07f33d6978c7e48f72"
EXPECTED_R3_CACHE = "83daa4c0be5fdf45cbee650f3609b02c55c491d7cc94d2df35e026ec6c90670f"
EXPECTED_R3_HEAD = "fce6eeb2d9f89e12df07ad40421e1e1f0cdbb971f54efb2c68545a47251f6b59"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def configure_determinism() -> None:
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def load_reference_predictions(run_dir: Path) -> dict[tuple[str, int], float]:
    result = {}
    for role in ("calibration", "validation"):
        path = run_dir / "predictions" / f"{role}.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row["episode_id"], int(row["t"]))
                if key in result:
                    raise ValueError(f"duplicate round-3 reference prediction {key}")
                result[key] = float(row["p_slip"])
    return result


def build(args: argparse.Namespace) -> dict:
    r2_index_path, r3_cache_path, r3_head_path = args.r2_index.resolve(), args.r3_cache_manifest.resolve(), args.r3_head.resolve()
    if sha256(r2_index_path) != EXPECTED_R2_INDEX or sha256(r3_cache_path) != EXPECTED_R3_CACHE or sha256(r3_head_path) != EXPECTED_R3_HEAD:
        raise ValueError("upstream artifact identity differs from frozen protocol")
    r2_index = json.loads(r2_index_path.read_text())
    r3_cache = json.loads(r3_cache_path.read_text())
    if r2_index.get("status") != "complete" or r3_cache.get("status") != "complete":
        raise ValueError("upstream cache incomplete")
    source_model, _ = p2.load_b_checkpoint(Path(r3_cache["checkpoint"]), torch.device("cpu"))
    branch = r3_training.SlipBranch(source_model.decoder)
    head_payload = torch.load(r3_head_path, map_location="cpu", weights_only=False)
    branch.load_state_dict(head_payload["branch_state"], strict=True)
    device = torch.device(args.device)
    branch.eval().requires_grad_(False).to(device)
    reference = load_reference_predictions(args.r3_run_dir.resolve())
    r2_rows = {row["id"]: row for row in r2_index["episodes"] if row.get("domain") == "htt" and row.get("task") == "slip"}
    selected = [entry for entry in r3_cache["entries"] if entry["roles_by_fold"][FOLD] in {"train", "calibration", "validation"}]
    output = args.output.resolve(); episodes_dir = output / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    entries, comparisons = [], []
    configure_determinism()
    for entry in selected:
        episode_id = entry["episode_id"]
        r2 = r2_rows.get(episode_id)
        if r2 is None:
            raise ValueError(f"round-2 cache lacks {episode_id}")
        if sha256(Path(entry["token_path"])) != entry["token_sha256"] or sha256(Path(entry["label_path"])) != entry["label_sha256"]:
            raise ValueError(f"round-3 cache hash mismatch: {episode_id}")
        if sha256(Path(r2["cache_path"])) != r2["cache_sha256"]:
            raise ValueError(f"round-2 cache hash mismatch: {episode_id}")
        tokens = np.load(entry["token_path"], mmap_mode="r", allow_pickle=False)
        labels = np.load(entry["label_path"], allow_pickle=False).astype(np.int8)
        scores = []
        with torch.inference_mode():
            for start in range(0, len(tokens), args.batch_size):
                batch = torch.from_numpy(np.array(tokens[start:start + args.batch_size], copy=True)).to(device)
                scores.append(torch.softmax(branch(batch), dim=1)[:, 1].float().cpu().numpy())
        p_slip = np.concatenate(scores).astype(np.float32)
        with np.load(r2["cache_path"], allow_pickle=False) as archive:
            z = np.asarray(archive["z"], dtype=np.float32)
            force = np.asarray(archive["force_pred"], dtype=np.float32)
            r2_labels = np.asarray(archive["labels"], dtype=np.int8)
        if len(z) != len(labels) or force.shape != (len(labels), 3) or not np.array_equal(labels, r2_labels):
            raise ValueError(f"cross-cache shape/label mismatch: {episode_id}")
        previous = np.maximum(np.arange(len(labels)) - 5, 0)
        delta = force - force[previous]
        if not all(np.isfinite(array).all() for array in (z, force, delta, p_slip)):
            raise ValueError(f"nonfinite future-cache input: {episode_id}")
        role = entry["roles_by_fold"][FOLD]
        if role in {"calibration", "validation"}:
            expected = np.asarray([reference[(episode_id, t)] for t in range(len(labels))])
            difference = np.abs(expected - p_slip)
            comparison = {"episode_id": episode_id, "role": role, "frames": len(labels),
                          "max_abs": float(difference.max()), "mean_abs": float(difference.mean()),
                          "within_atol_1e-6_rtol_1e-5": bool(np.allclose(expected, p_slip, atol=1e-6, rtol=1e-5))}
            if not comparison["within_atol_1e-6_rtol_1e-5"]:
                raise ValueError(f"adapted prediction mismatch: {comparison}")
            comparisons.append(comparison)
        key = hashlib.sha256(episode_id.encode()).hexdigest()[:20]
        destination = episodes_dir / f"{key}.npz"
        temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, z=z, force_pred=force, delta_force_pred=delta.astype(np.float32),
                                p_slip=p_slip, labels=labels, frame_index=np.arange(len(labels), dtype=np.int32))
        os.replace(temporary, destination)
        entries.append({"episode_id": episode_id, "role": role, "frames": len(labels), "path": str(destination),
                        "sha256": sha256(destination), "r2_cache_sha256": r2["cache_sha256"],
                        "r3_token_sha256": entry["token_sha256"], "r3_label_sha256": entry["label_sha256"]})
        print(json.dumps({"completed": len(entries), "total": len(selected), "episode": episode_id, "role": role}), flush=True)
    result = {
        "format": "round4_future_causal_inputs_v1", "status": "complete", "fold": FOLD,
        "feature_schema": ["z[768]", "force_pred[3]", "delta_force_pred_t_minus_max_t5[3]", "adapted_p_slip[1]"],
        "history": 4, "r2_index": str(r2_index_path), "r2_index_sha256": sha256(r2_index_path),
        "r3_cache_manifest": str(r3_cache_path), "r3_cache_manifest_sha256": sha256(r3_cache_path),
        "r3_head": str(r3_head_path), "r3_head_sha256": sha256(r3_head_path),
        "r3_head_identity": {"variant": "B", "fold": FOLD, "seed": 20260914, "best_epoch": head_payload["epoch"]},
        "ground_truth_force_used_as_input": False, "future_frames_used_as_input": False,
        "entries": entries, "prediction_compatibility": {
            "reference": str(args.r3_run_dir.resolve()), "episodes": len(comparisons),
            "frames": sum(row["frames"] for row in comparisons),
            "maximum_absolute_difference": max(row["max_abs"] for row in comparisons),
            "all_within_atol_1e-6_rtol_1e-5": all(row["within_atol_1e-6_rtol_1e-5"] for row in comparisons),
        },
    }
    atomic_json(output / "cache_manifest.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r2-index", type=Path, required=True)
    parser.add_argument("--r3-cache-manifest", type=Path, required=True)
    parser.add_argument("--r3-head", type=Path, required=True)
    parser.add_argument("--r3-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    result = build(parse_args())
    print(json.dumps({"status": result["status"], "episodes": len(result["entries"]),
                      "prediction_compatibility": result["prediction_compatibility"]}, indent=2))


if __name__ == "__main__":
    main()
