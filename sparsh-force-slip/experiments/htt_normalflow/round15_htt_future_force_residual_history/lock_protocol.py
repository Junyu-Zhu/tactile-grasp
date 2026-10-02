#!/usr/bin/env python3
"""Freeze Round-15 protocol, executable sources, and inherited formal inputs."""
import argparse
import datetime
import json
from pathlib import Path

import torch

import future_train as ft


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round14-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parent
    created = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    protocol_files = ["USER_SCOPE.md", "PROTOCOL.md", "PARENT_IDENTITY_CHECK.json"]
    missing = [name for name in protocol_files if not (root / name).exists()]
    if missing:
        raise SystemExit(f"missing protocol inputs: {missing}")
    protocol = {
        "schema": "round15_protocol_lock_v1", "status": "locked", "created_at": created,
        "files": {name: ft.sha(root / name) for name in protocol_files},
        "formal_grid": {"groups": list(ft.GROUPS), "folds": [1, 2, 3, 4], "seeds": list(ft.SEEDS), "runs": 24},
        "horizons_frames": list(ft.HORIZONS), "target_clip_n": [-20, 20], "output_clip": False,
        "history_steps": {"C_current": 1, "H_history": 9},
        "normalization": "same nine-step fit-only x mean/std for both groups; fit-only scale-only residual std; zero residual offset",
        "initialization": "all body matrices Xavier, all body biases zero, output weight+bias exactly zero",
        "epoch_minus_one_checkpoint_eligible": False,
        "selection": "earliest strict post-training-epoch minimum mean native-N future MAE on selection only",
        "optimization": {"loss": "SmoothL1 beta=1 normalized residual", "optimizer": "AdamW", "lr": 0.001, "weight_decay": 0.0001, "batch": 256, "max_epochs": 60, "patience": 10},
        "diagnostic": {"direction_exclusion_abs_true_change_lt_n": 0.10, "stable_h10_max_axis_le_n": 0.25, "changing_h10_max_axis_ge_n": 1.0, "shuffle_seed": "2026091501+100*fold+seed%100", "bootstrap_draws": 2000, "bootstrap_seed": "150000+fold"},
        "test_role": "forbidden",
    }
    ft.atomic_json(protocol, arguments.output / "PROTOCOL_LOCK.json")

    source_names = ["future_train.py", "run_all.py", "smoke.py", "evaluate.py", "evaluate_all.py", "diagnose_r14_history.py", "lock_protocol.py"]
    source = {"schema": "round15_source_lock_v1", "status": "locked", "created_at": created,
              "files": {name: ft.sha(root / name) for name in source_names}}
    ft.atomic_json(source, arguments.output / "SOURCE_LOCK.json")

    inputs = []
    upstream = {}
    for fold in range(1, 5):
        for seed in ft.SEEDS:
            prepared = arguments.round14_output / "prepared" / f"p{fold}_s{seed}" / "prepared.pt"
            checkpoint = arguments.round14_output / "formal" / f"F_concat_p{fold}_s{seed}" / "best.pth"
            if not prepared.exists() or not checkpoint.exists():
                raise SystemExit(f"missing inherited input p{fold} s{seed}")
            data = torch.load(prepared, map_location="cpu", weights_only=False)
            ft.validate_cache(data)
            checkpoint_object = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if checkpoint_object["identity"]["round14_data_sha256"] != ft.sha(prepared):
                raise ValueError("Round-14 checkpoint/prepared mismatch")
            inputs.append({"fold": fold, "seed": seed, "prepared": str(prepared), "prepared_sha256": ft.sha(prepared),
                           "r14_fconcat_checkpoint": str(checkpoint), "r14_fconcat_checkpoint_sha256": ft.sha(checkpoint)})
            for name, identity in data["provenance"]["immutable_upstream"].items():
                upstream.setdefault(name, set()).add((identity["path"], identity["sha256"]))
    input_lock = {"schema": "round15_input_lock_v1", "status": "locked", "created_at": created,
                  "round14_output": str(arguments.round14_output), "runs": inputs,
                  "immutable_upstream_identities": {name: [{"path": path, "sha256": digest} for path, digest in sorted(values)] for name, values in upstream.items()},
                  "reuse_statement": "Accepted Round-14 prepared caches and F_concat checkpoints are reused without regeneration; test absent."}
    ft.atomic_json(input_lock, arguments.output / "INPUT_LOCK.json")
    print(json.dumps({"status": "locked", "protocol": ft.sha(arguments.output / "PROTOCOL_LOCK.json"), "source": ft.sha(arguments.output / "SOURCE_LOCK.json"), "input": ft.sha(arguments.output / "INPUT_LOCK.json")}))


if __name__ == "__main__":
    main()
