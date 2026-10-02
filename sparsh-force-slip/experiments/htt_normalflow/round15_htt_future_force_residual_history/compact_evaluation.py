#!/usr/bin/env python3
"""Compact prediction tensor views without changing logical values."""
import argparse
import json
from pathlib import Path

import torch

import future_train as ft


def compact(value):
    if torch.is_tensor(value):
        return value.detach().clone().contiguous()
    if isinstance(value, dict):
        return {key: compact(item) for key, item in value.items()}
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return tuple(value)
    return value


def equal(left, right):
    if torch.is_tensor(left):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return left == right


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    records = []
    for group in ft.GROUPS:
        for fold in range(1, 5):
            for seed in ft.SEEDS:
                run = f"{group}_p{fold}_s{seed}"
                root = arguments.evaluation / run
                summary_path = root / "SUMMARY.json"
                summary = json.loads(summary_path.read_text())
                role_hashes = {}
                for role in ("fit", "selection", "calibration", "validation"):
                    path = root / f"predictions_{role}.pt"
                    before_bytes = path.stat().st_size
                    before = torch.load(path, map_location="cpu", weights_only=False)
                    replacement = compact(before)
                    ft.atomic_save(replacement, path)
                    after = torch.load(path, map_location="cpu", weights_only=False)
                    if not equal(before, after):
                        raise ValueError(f"logical mismatch after compaction: {run}/{role}")
                    role_hashes[role] = ft.sha(path)
                    records.append({"run": run, "role": role, "before_bytes": before_bytes,
                                    "after_bytes": path.stat().st_size, "logical_values_equal": True,
                                    "sha256": role_hashes[role]})
                summary["prediction_hashes"] = role_hashes
                summary["storage_compaction"] = "post-evaluation clone-contiguous repair; logical values reloaded and exactly equal"
                ft.atomic_json(summary, summary_path)
    receipt = {"schema": "round15_evaluation_compaction_v1", "status": "pass", "files": len(records),
               "before_bytes": sum(record["before_bytes"] for record in records),
               "after_bytes": sum(record["after_bytes"] for record in records),
               "bytes_removed": sum(record["before_bytes"] - record["after_bytes"] for record in records),
               "all_logical_values_equal": all(record["logical_values_equal"] for record in records), "records": records}
    ft.atomic_json(receipt, arguments.output)
    print(json.dumps({key: receipt[key] for key in ("status", "files", "before_bytes", "after_bytes", "bytes_removed")}))


if __name__ == "__main__":
    main()
