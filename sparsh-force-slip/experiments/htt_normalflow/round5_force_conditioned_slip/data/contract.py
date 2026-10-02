"""Pure validation and indexing helpers for the Round-5 HTT cache contract."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


FORMAT = "round5_htt_mae_tokens_targets_v1"
FOLDS = tuple(f"htt_leave_p{i}" for i in range(1, 5))
ROLES = ("train", "validation", "calibration", "test")
DEVELOPMENT_ROLES = frozenset(("train", "validation", "calibration"))
FORCE_AXES = ("shear_x", "shear_y", "normal")
FORCE_UNIT = "N"
FORCE_TARGET_SEMANTICS = "clip((6d_force-ref_force)[:3],-20,20) N; shear_x,shear_y,normal"
FORCE_INPUT_MIN_T = 5
FORCE_CONDITIONED_MIN_T = 10
FUTURE_MIN_T = 13


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def roles_by_fold(split_manifest: dict[str, Any], episode_id: str) -> dict[str, str]:
    roles: dict[str, str] = {}
    for fold in FOLDS:
        split = split_manifest["splits"][fold]
        matched = [role for role in ROLES if episode_id in split[role]]
        if len(matched) != 1:
            raise ValueError(f"{episode_id}: expected one role in {fold}, got {matched}")
        roles[fold] = matched[0]
    return roles


def eligible_indices(frames: int, task: str) -> range:
    if task == "force":
        start = FORCE_INPUT_MIN_T
    elif task == "slip":
        start = FORCE_CONDITIONED_MIN_T
    elif task == "future":
        start = FUTURE_MIN_T
    else:
        raise ValueError(f"unknown task {task!r}")
    return range(start, frames)


def validate_entry(entry: dict[str, Any], *, require_files: bool = False) -> None:
    required = {"episode_id", "task", "roles_by_fold", "source_files", "token_path", "token_sha256", "frames"}
    missing = required - entry.keys()
    if missing:
        raise ValueError(f"{entry.get('episode_id')}: missing {sorted(missing)}")
    if entry["task"] not in ("force", "slip"):
        raise ValueError("task must be force or slip")
    if set(entry["roles_by_fold"]) != set(FOLDS):
        raise ValueError(f"{entry['episode_id']}: incomplete fold roles")
    if any(role not in ROLES for role in entry["roles_by_fold"].values()):
        raise ValueError(f"{entry['episode_id']}: invalid role")
    target_key = "force_native_n_path" if entry["task"] == "force" else "label_path"
    if not entry.get(target_key):
        raise ValueError(f"{entry['episode_id']}: missing {target_key}")
    if require_files:
        tokens = np.load(entry["token_path"], mmap_mode="r", allow_pickle=False)
        target = np.load(entry[target_key], mmap_mode="r", allow_pickle=False)
        if tokens.shape[0] != entry["frames"] or tokens.ndim != 3 or tokens.dtype != np.float32:
            raise ValueError(f"{entry['episode_id']}: bad tokens")
        expected = (entry["frames"], 3) if entry["task"] == "force" else (entry["frames"],)
        if target.shape != expected or not np.isfinite(target).all():
            raise ValueError(f"{entry['episode_id']}: bad target")


def validate_contract(payload: dict[str, Any], *, require_files: bool = False) -> None:
    if payload.get("format") != FORMAT:
        raise ValueError(f"unexpected format {payload.get('format')!r}")
    if payload.get("force_target_semantics") != FORCE_TARGET_SEMANTICS:
        raise ValueError("missing or incompatible force_target_semantics")
    entries = payload.get("entries", [])
    ids = [entry["episode_id"] for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate episode IDs")
    for entry in entries:
        validate_entry(entry, require_files=require_files)
    # Similar basenames across separate force/slip acquisitions are never joins.
    if payload.get("pairing_policy") != "independent_episodes_no_force_slip_basename_join":
        raise ValueError("unsafe or missing pairing policy")


def select_entries(payload: dict[str, Any], fold: str, role: str, task: str) -> list[dict[str, Any]]:
    if fold not in FOLDS or role not in DEVELOPMENT_ROLES:
        raise ValueError("only frozen development roles are selectable")
    validate_contract(payload)
    return [entry for entry in payload["entries"] if entry["task"] == task and entry["roles_by_fold"][fold] == role]
