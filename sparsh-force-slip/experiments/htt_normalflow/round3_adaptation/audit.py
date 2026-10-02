#!/usr/bin/env python3
"""Read-only provenance, model-boundary, split and baseline audit for Round 3."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
REPO_ROOT = EXP_ROOT.parents[1]
SPARSH_REPO = Path("/home/zjy/document/sparsh")
for item in (SPARSH_REPO, EXP_ROOT, REPO_ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

os.environ.setdefault("XFORMERS_DISABLED", "1")

import phase2_b_multitask as p2  # noqa: E402


DEFAULT_MANIFEST = Path("/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round1/splits.json")
DEFAULT_CHECKPOINT = Path(
    "/vla1/zjy/sparsh_runs/force_slip_phase2/"
    "phase_lambda_decoupled_mae_lam010_20260528_000000/"
    "mae_decoupled_multitask/checkpoints/epoch-0030.pth"
)
DEFAULT_PRETRAINED = Path("/vla1/zjy/sparsh_models/sparsh-mae-base/mae_vitbase.ckpt")
DEFAULT_OLD_CACHE = Path("/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round2/cache/index.json")
EXPECTED_PRETRAINED_SHA = "83ae4a8fa6a7bbd9702b14586feafdba63252e83afad26c3e5832e0ad39446ad"
EXPECTED_SELECTED_SHA = "850e4a74e6d9bd60e8ed22efc8b70d343c5d21bb5f60b3053b0a800674dffcb4"
EXPECTED_MANIFEST_SHA = "bd01e50b06cbea8c9eae8fc1ba448a1b0a5b53110471ae3d12e48ea1313242d1"


def file_sha256(path: Path) -> str:
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


def compare_states(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> dict[str, Any]:
    left_keys, right_keys = set(left), set(right)
    mismatches = []
    for key in sorted(left_keys & right_keys):
        a, b = left[key].detach().cpu(), right[key].detach().cpu()
        if not torch.equal(a, b):
            mismatches.append({
                "key": key,
                "shape": list(a.shape),
                "max_abs_difference": float((a.float() - b.float()).abs().max()),
            })
    return {
        "left_tensors": len(left),
        "right_tensors": len(right),
        "missing_from_right": sorted(left_keys - right_keys),
        "missing_from_left": sorted(right_keys - left_keys),
        "mismatches": mismatches,
        "exact": left_keys == right_keys and not mismatches,
    }


def audit_encoder(checkpoint: Path, pretrained: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = payload["train_config"]
    seed = int(cfg["seed"])

    selected, loaded_payload = p2.load_b_checkpoint(checkpoint, torch.device("cpu"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    fresh = p2.FrozenEncoderSharedForceSlip("mae", decoder_variant="decoupled")

    selected_state = selected.encoder.state_dict()
    fresh_state = fresh.encoder.state_dict()
    full = compare_states(selected_state, fresh_state)

    raw = torch.load(pretrained, map_location="cpu", weights_only=False)["model"]
    prefix = fresh.load_info["source_prefix"] + "."
    mapped = {key[len(prefix):]: value for key, value in raw.items() if key.startswith(prefix)}
    selected_loaded = {key: selected_state[key] for key in mapped if key in selected_state}
    source = compare_states(mapped, selected_loaded)
    seeded_only = sorted(set(fresh_state) - set(mapped))

    return {
        "selected_checkpoint": str(checkpoint),
        "selected_checkpoint_sha256": file_sha256(checkpoint),
        "selected_expected_sha256": EXPECTED_SELECTED_SHA,
        "selected_format": loaded_payload.get("format"),
        "selected_epoch": loaded_payload.get("epoch"),
        "selected_train_config": {
            "encoder": cfg.get("encoder"),
            "decoder_variant": cfg.get("decoder_variant"),
            "seed": seed,
        },
        "canonical_pretrained": str(pretrained),
        "canonical_pretrained_sha256": file_sha256(pretrained),
        "canonical_expected_sha256": EXPECTED_PRETRAINED_SHA,
        "fresh_load_info": fresh.load_info,
        "direct_source_loaded_tensor_comparison": source,
        "constructor_seeded_tensors_not_in_pretrained": seeded_only,
        "full_selected_vs_fresh_constructor_comparison": full,
        "selected_encoder_state_sha256": tensor_state_sha256(selected_state),
        "fresh_encoder_state_sha256": tensor_state_sha256(fresh_state),
        "status": "pass" if (
            file_sha256(checkpoint) == EXPECTED_SELECTED_SHA
            and file_sha256(pretrained) == EXPECTED_PRETRAINED_SHA
            and cfg.get("encoder") == "mae"
            and cfg.get("decoder_variant") == "decoupled"
            and source["exact"]
            and full["exact"]
        ) else "fail",
    }


def audit_boundary(checkpoint: Path) -> dict[str, Any]:
    model, _ = p2.load_b_checkpoint(checkpoint, torch.device("cpu"))
    state = model.state_dict()
    slip_prefixes = ("decoder.slip_pooler.", "decoder.slip_trunk.", "decoder.slip_head.")
    force_prefixes = ("decoder.force_pooler.", "decoder.force_trunk.", "decoder.force_head.")
    slip = sorted(name for name in state if name.startswith(slip_prefixes))
    force = sorted(name for name in state if name.startswith(force_prefixes))
    other_decoder = sorted(
        name for name in state
        if name.startswith("decoder.") and name not in set(slip) | set(force)
    )
    selected_parameter_names = sorted(
        name for name, _ in model.named_parameters() if name.startswith(slip_prefixes)
    )
    parameter_map = dict(model.named_parameters())
    return {
        "decoder_class": type(model.decoder).__name__,
        "decoder_variant": model.decoder_variant,
        "slip_state_tensors": len(slip),
        "slip_parameter_tensors": len(selected_parameter_names),
        "slip_parameters": sum(parameter_map[name].numel() for name in selected_parameter_names),
        "force_state_tensors": len(force),
        "force_parameters": sum(
            parameter.numel() for name, parameter in model.named_parameters() if name.startswith(force_prefixes)
        ),
        "other_decoder_state": other_decoder,
        "optimizer_selection_prefixes": list(slip_prefixes),
        "optimizer_selection_names": selected_parameter_names,
        "selection_contains_encoder": any(name.startswith("encoder.") for name in selected_parameter_names),
        "selection_contains_force": any(name.startswith(force_prefixes) for name in selected_parameter_names),
        "encoder_requires_grad_count": sum(p.requires_grad for p in model.encoder.parameters()),
        "status": "pass" if (
            isinstance(model.decoder, p2.DecoupledForceSlipDecoder)
            and slip and force and not other_decoder
            and not any(name.startswith("encoder.") for name in selected_parameter_names)
            and not any(name.startswith(force_prefixes) for name in selected_parameter_names)
            and not any(p.requires_grad for p in model.encoder.parameters())
        ) else "fail",
    }


def label_file(row: dict[str, Any]) -> Path | None:
    matches = [Path(path) for path in row["source_files"] if path.endswith(".labeled.npz")]
    if len(matches) > 1:
        raise ValueError(f"Multiple label files for {row['id']}")
    return matches[0] if matches else None


def audit_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    rows = {row["id"]: row for row in manifest["episodes"]}
    folds: dict[str, Any] = {}
    all_pass = manifest.get("schema_version") == 2
    for fold, split in manifest["splits"].items():
        if not fold.startswith("htt_leave_"):
            continue
        role_sets = {role: set(ids) for role, ids in split.items()}
        pairwise: dict[str, Any] = {}
        roles = sorted(role_sets)
        for index, left in enumerate(roles):
            for right in roles[index + 1:]:
                left_rows = [rows[item] for item in role_sets[left]]
                right_rows = [rows[item] for item in role_sets[right]]
                overlaps = {
                    "episode_ids": sorted(role_sets[left] & role_sets[right]),
                    "leakage_groups": sorted(
                        {row["leakage_group"] for row in left_rows}
                        & {row["leakage_group"] for row in right_rows}
                    ),
                    "physical_hashes": sorted(
                        {row["physical_hash"] for row in left_rows}
                        & {row["physical_hash"] for row in right_rows}
                    ),
                    "source_paths": sorted(
                        {path for row in left_rows for path in row["source_files"]}
                        & {path for row in right_rows for path in row["source_files"]}
                    ),
                }
                overlaps["disjoint"] = not any(overlaps.values())
                pairwise[f"{left}__{right}"] = overlaps
                all_pass &= overlaps["disjoint"]

        support = {}
        for role in ("train", "validation", "calibration"):
            counts: Counter[int] = Counter()
            labeled_episodes = 0
            unlabeled_episodes = 0
            for episode_id in split[role]:
                row = rows[episode_id]
                label_path = label_file(row)
                if label_path is None:
                    unlabeled_episodes += 1
                    continue
                labeled_episodes += 1
                with np.load(label_path, allow_pickle=False) as data:
                    labels = data["sliding_labels_bracket"].astype(np.int64)
                counts.update(map(int, labels.tolist()))
            role_support = {
                "episodes": len(split[role]),
                "labeled_episodes": labeled_episodes,
                "unlabeled_episodes_excluded_from_slip_supervision": unlabeled_episodes,
                "stage_frame_counts": {str(stage): counts[stage] for stage in (0, 1, 2)},
                "primary_static_frames": counts[0],
                "excluded_incipient_frames": counts[1],
                "primary_gross_frames": counts[2],
                "both_primary_classes_present": counts[0] > 0 and counts[2] > 0,
            }
            support[role] = role_support
            all_pass &= role_support["both_primary_classes_present"]

        heldout = fold.removeprefix("htt_leave_")
        groups = {role: sorted({rows[item]["group"] for item in ids}) for role, ids in split.items()}
        group_policy_pass = groups["test"] == [heldout] and heldout not in set(
            groups["train"] + groups["validation"] + groups["calibration"]
        )
        all_pass &= group_policy_pass
        folds[fold] = {
            "role_episode_counts": {role: len(ids) for role, ids in split.items()},
            "groups_by_role": groups,
            "heldout_group_policy_pass": group_policy_pass,
            "pairwise_role_isolation": pairwise,
            "development_label_support": support,
            "test_labels_opened": False,
        }
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "expected_manifest_sha256": EXPECTED_MANIFEST_SHA,
        "schema_version": manifest.get("schema_version"),
        "folds": folds,
        "cross_fold_overlap_policy": "expected development-fold reuse; never pool folds as independent samples",
        "status": "pass" if all_pass and file_sha256(manifest_path) == EXPECTED_MANIFEST_SHA else "fail",
    }


def audit_baseline(old_cache_path: Path, manifest_path: Path, checkpoint: Path) -> dict[str, Any]:
    cache = json.loads(old_cache_path.read_text())
    episode_entries = cache.get("episodes", [])
    roles = cache.get("role_policy", {})
    checks = {
        "checkpoint_sha_matches_selected": cache.get("checkpoint_sha256") == file_sha256(checkpoint),
        "manifest_sha_matches_round3": cache.get("split_manifest_sha256") == file_sha256(manifest_path),
        "preprocessing_matches_protocol": cache.get("preprocessing")
        == "legacy load_sample_from_buf; raw reference; resize320x240; RGB; current,t-5; no NF background",
        "float32": cache.get("precision") == "float32 TF32 off",
        "complete": cache.get("status") == "complete",
    }
    return {
        "old_cache_index": str(old_cache_path),
        "old_cache_index_sha256": file_sha256(old_cache_path),
        "recorded_checkpoint_sha256": cache.get("checkpoint_sha256"),
        "recorded_manifest_sha256": cache.get("split_manifest_sha256"),
        "preprocessing": cache.get("preprocessing"),
        "role_policy": roles,
        "cached_episode_entries": len(episode_entries),
        "checks": checks,
        "reuse_scope": (
            "A logits may be reused for train/validation/calibration where present; "
            "Round 3 must recompute calibration thresholds under its frozen rule and must not read test-role results."
        ),
        "status": "pass" if all(checks.values()) else "fail",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--old-cache", type=Path, default=DEFAULT_OLD_CACHE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = {
        "format": "round3_mae_slip_adaptation_audit_v1",
        "runtime": {"python": sys.executable, "torch": torch.__version__, "device": "cpu"},
        "encoder_provenance": audit_encoder(args.checkpoint, args.pretrained),
        "model_boundary": audit_boundary(args.checkpoint),
        "manifest": audit_manifest(args.manifest),
        "old_baseline_compatibility": audit_baseline(args.old_cache, args.manifest, args.checkpoint),
    }
    report["status"] = "pass" if all(
        report[key]["status"] == "pass"
        for key in ("encoder_provenance", "model_boundary", "manifest", "old_baseline_compatibility")
    ) else "fail"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "encoder": report["encoder_provenance"]["status"],
        "boundary": report["model_boundary"]["status"],
        "manifest": report["manifest"]["status"],
        "baseline": report["old_baseline_compatibility"]["status"],
        "output": str(args.output),
    }, indent=2))
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
