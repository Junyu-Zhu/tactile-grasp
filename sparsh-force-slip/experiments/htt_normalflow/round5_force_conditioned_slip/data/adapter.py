#!/usr/bin/env python3
"""Role-safe cached-frame adapter and train-only force normalization CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from contract import FORCE_INPUT_MIN_T, atomic_json, select_entries, validate_contract


def load_contract(path: Path, require_files: bool = False) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    validate_contract(payload, require_files=require_files)
    return payload


def iter_force_arrays(payload: dict[str, Any], fold: str, role: str) -> Iterator[np.ndarray]:
    for entry in select_entries(payload, fold, role, "force"):
        values = np.load(entry["force_native_n_path"], mmap_mode="r", allow_pickle=False)
        yield np.asarray(values[FORCE_INPUT_MIN_T:], dtype=np.float64)


def fit_force_standardizer(payload: dict[str, Any], fold: str) -> dict[str, Any]:
    """Fit native-axis normalization only from the named fold's train role."""
    arrays = list(iter_force_arrays(payload, fold, "train"))
    if not arrays:
        raise ValueError(f"{fold}: no force train arrays")
    count = sum(len(array) for array in arrays)
    total = sum((array.sum(axis=0) for array in arrays), np.zeros(3, dtype=np.float64))
    mean = total / count
    square_total = sum(((array - mean) ** 2).sum(axis=0) for array in arrays)
    std = np.sqrt(square_total / count)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError(f"{fold}: invalid force standardizer")
    return {
        "format": "round5_force_train_standardizer_v1", "fold": fold, "role": "train",
        "min_frame_inclusive": FORCE_INPUT_MIN_T, "count": count,
        "mean_native_N": mean.tolist(), "std_native_N": std.tolist(),
        "axes": ["shear_x", "shear_y", "normal"], "ddof": 0,
        "test_or_validation_used": False,
    }


class CachedFrames:
    """Small dependency-free view; training code can wrap it as a torch Dataset."""

    def __init__(self, payload: dict[str, Any], fold: str, role: str, task: str,
                 min_frame: int, standardizer: dict[str, Any] | None = None) -> None:
        self.entries = select_entries(payload, fold, role, task)
        self.task = task
        self.min_frame = min_frame
        self.standardizer = standardizer
        self.index = [(entry_index, frame) for entry_index, entry in enumerate(self.entries)
                      for frame in range(min_frame, entry["frames"])]
        self._tokens: dict[int, np.ndarray] = {}
        self._targets: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry_index, frame = self.index[index]
        entry = self.entries[entry_index]
        if entry_index not in self._tokens:
            self._tokens[entry_index] = np.load(entry["token_path"], mmap_mode="r", allow_pickle=False)
            target_key = "force_native_n_path" if self.task == "force" else "label_path"
            self._targets[entry_index] = np.load(entry[target_key], mmap_mode="r", allow_pickle=False)
        target = np.array(self._targets[entry_index][frame], copy=True)
        if self.task == "force" and self.standardizer is not None:
            target = (target - np.asarray(self.standardizer["mean_native_N"])) / np.asarray(self.standardizer["std_native_N"])
        return {"tokens": np.array(self._tokens[entry_index][frame], copy=True), "target": target,
                "episode_id": entry["episode_id"], "frame": frame, "role": entry["roles_by_fold"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = load_contract(args.contract, require_files=True)
    result = fit_force_standardizer(payload, args.fold)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

