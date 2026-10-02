"""Strict role-bound readers for the frozen Round 5 cache contract."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from common import sha256_file


FOLDS = tuple(f"htt_leave_p{i}" for i in range(1, 5))
ROLES = ("train", "validation", "calibration")
FORCE_TARGET_SEMANTICS = "clip((6d_force-ref_force)[:3],-20,20) N; shear_x,shear_y,normal"


def load_cache(path: Path, verify_hashes: bool = False) -> dict[str, Any]:
    manifest_path = path / "cache_manifest.json" if path.is_dir() else path
    payload = json.loads(manifest_path.read_text())
    if payload.get("status") != "complete" or payload.get("format") != "round5_htt_mae_tokens_targets_v1":
        raise RuntimeError("Expected a complete Round 5 MAE token/target cache")
    if payload.get("force_target_semantics") != FORCE_TARGET_SEMANTICS:
        raise RuntimeError("Force target semantics are absent or incompatible")
    entries = payload.get("entries", [])
    identities = [entry.get("episode_id") for entry in entries]
    if not entries or len(set(identities)) != len(entries):
        raise RuntimeError("Cache episode identities are empty or duplicated")
    for entry in entries:
        if entry.get("task") not in ("force", "slip"):
            raise RuntimeError("Every entry must declare force or slip task")
        if set(entry.get("roles_by_fold", {})) != set(FOLDS):
            raise RuntimeError(f"Incomplete roles for {entry.get('episode_id')}")
        if any(role not in (*ROLES, "test") for role in entry["roles_by_fold"].values()):
            raise RuntimeError("Unknown data role")
        token_path = Path(entry["token_path"])
        if not token_path.is_file() or (verify_hashes and sha256_file(token_path) != entry["token_sha256"]):
            raise RuntimeError(f"Token artifact mismatch: {entry.get('episode_id')}")
        tokens = np.load(token_path, mmap_mode="r", allow_pickle=False)
        if tokens.dtype != np.float32 or tokens.ndim != 3 or len(tokens) != entry["frames"]:
            raise RuntimeError(f"Token schema mismatch: {entry.get('episode_id')}")
        if entry["task"] == "force":
            target_path = Path(entry["force_native_n_path"])
            if not target_path.is_file() or (verify_hashes and sha256_file(target_path) != entry["force_native_n_sha256"]):
                raise RuntimeError(f"Force target mismatch: {entry.get('episode_id')}")
            target = np.load(target_path, mmap_mode="r", allow_pickle=False)
            if target.dtype != np.float32 or target.shape != (entry["frames"], 3) or not np.isfinite(target).all():
                raise RuntimeError(f"Force target schema mismatch: {entry.get('episode_id')}")
        else:
            label_path = Path(entry["label_path"])
            if not label_path.is_file() or (verify_hashes and sha256_file(label_path) != entry["label_sha256"]):
                raise RuntimeError(f"Slip labels mismatch: {entry.get('episode_id')}")
            labels = np.load(label_path, mmap_mode="r", allow_pickle=False)
            if labels.shape != (entry["frames"],) or not np.isin(labels, (0, 1, 2)).all():
                raise RuntimeError(f"Slip label schema mismatch: {entry.get('episode_id')}")
    payload["manifest_path"] = str(manifest_path.resolve())
    payload["manifest_sha256"] = sha256_file(manifest_path)
    return payload


def require_cache_audit(path: Path | None, cache: dict[str, Any], allow_unverified: bool) -> dict[str, Any] | None:
    if path is None:
        if allow_unverified:
            return None
        raise RuntimeError("A matching full cache audit is required")
    audit = json.loads(path.read_text())
    if audit.get("status") != "pass" or audit.get("cache_manifest_sha256") != cache["manifest_sha256"]:
        raise RuntimeError("Cache audit is absent, failed, or belongs to another manifest")
    return audit


def role_entries(cache: dict[str, Any], fold: str, role: str, task: str) -> list[dict[str, Any]]:
    if fold not in FOLDS or role not in ROLES or task not in ("force", "slip"):
        raise ValueError("Invalid fold/role/task")
    entries = [entry for entry in cache["entries"]
               if entry["task"] == task and entry["roles_by_fold"][fold] == role]
    if not entries:
        raise RuntimeError(f"No entries for {fold}/{role}/{task}")
    return entries


class ForceFrames(Dataset):
    def __init__(self, entries: list[dict[str, Any]], min_frame: int = 5) -> None:
        self.entries = entries
        self.index = [(episode, frame) for episode, entry in enumerate(entries)
                      for frame in range(min_frame, entry["frames"])]
        self._tokens: dict[int, np.ndarray] = {}
        self._targets: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int):
        episode, frame = self.index[item]
        entry = self.entries[episode]
        if episode not in self._tokens:
            self._tokens[episode] = np.load(entry["token_path"], mmap_mode="r", allow_pickle=False)
            self._targets[episode] = np.load(entry["force_native_n_path"], mmap_mode="r", allow_pickle=False)
        return (torch.from_numpy(np.array(self._tokens[episode][frame], copy=True)),
                torch.from_numpy(np.array(self._targets[episode][frame], copy=True)),
                entry["episode_id"], frame)


class SlipFrames(Dataset):
    """Primary static/gross frames with strict token and force-delta history."""
    def __init__(self, entries: list[dict[str, Any]], conditions: dict[str, np.ndarray] | None = None,
                 strict_start: int = 10, primary_only: bool = True) -> None:
        self.entries = entries
        self.conditions = conditions
        self.index = []
        self._tokens: dict[int, np.ndarray] = {}
        for episode, entry in enumerate(entries):
            labels = np.load(entry["label_path"], mmap_mode="r", allow_pickle=False)
            for frame in range(strict_start, len(labels)):
                if not primary_only or labels[frame] in (0, 2):
                    self.index.append((episode, frame, int(labels[frame] == 2), int(labels[frame])))
        if not self.index:
            raise RuntimeError("No primary slip frames after strict t>=10 history filter")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int):
        episode, frame, binary, stage = self.index[item]
        entry = self.entries[episode]
        if episode not in self._tokens:
            self._tokens[episode] = np.load(entry["token_path"], mmap_mode="r", allow_pickle=False)
        condition = np.empty(0, np.float32) if self.conditions is None else self.conditions[entry["episode_id"]][frame]
        return (torch.from_numpy(np.array(self._tokens[episode][frame], copy=True)),
                torch.from_numpy(np.array(condition, copy=True)), binary,
                entry["episode_id"], frame, stage)


def force_train_targets(entries: list[dict[str, Any]]) -> np.ndarray:
    return np.concatenate([np.load(entry["force_native_n_path"], allow_pickle=False)[5:] for entry in entries]).astype(np.float32)


def stratified_smoke_indices(dataset: SlipFrames, maximum: int = 128) -> list[int]:
    """Deterministically cover both classes and multiple episodes for smoke only."""
    buckets: dict[tuple[int, int], list[int]] = {}
    for index, (episode, _, binary, _) in enumerate(dataset.index):
        buckets.setdefault((episode, binary), []).append(index)
    selected, offsets = [], {key: 0 for key in buckets}
    while len(selected) < min(maximum, len(dataset)):
        changed = False
        for key in sorted(buckets):
            offset = offsets[key]
            if offset < len(buckets[key]) and len(selected) < maximum:
                selected.append(buckets[key][offset]); offsets[key] += 1; changed = True
        if not changed:
            break
    classes = {dataset.index[index][2] for index in selected}
    episodes = {dataset.index[index][0] for index in selected}
    if classes != {0, 1} or len(episodes) < 2:
        raise RuntimeError("Smoke subset cannot cover both classes across at least two episodes")
    return selected
