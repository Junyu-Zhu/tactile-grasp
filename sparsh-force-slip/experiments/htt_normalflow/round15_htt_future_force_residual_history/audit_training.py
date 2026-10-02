#!/usr/bin/env python3
"""Audit all formal Round-15 checkpoints and shared normalization identities."""
import argparse
import json
from pathlib import Path

import torch

import future_train as ft


def tensor_hash(tensor):
    import hashlib
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    records = []
    pair_norms = {}
    for group in ft.GROUPS:
        for fold in range(1, 5):
            for seed in ft.SEEDS:
                run = f"{group}_p{fold}_s{seed}"
                run_root = arguments.formal / run
                summary = json.loads((run_root / "summary.json").read_text())
                best = torch.load(run_root / "best.pth", map_location="cpu", weights_only=False)
                latest = torch.load(run_root / "latest.pth", map_location="cpu", weights_only=False)
                data_path = arguments.data_root / f"p{fold}_s{seed}" / "prepared.pt"
                data = torch.load(data_path, map_location="cpu", weights_only=False)
                ft.validate_cache(data)
                identity = best["identity"]
                assert summary["status"] == "complete" and identity["group"] == group and identity["fold"] == fold and identity["seed"] == seed
                assert identity["round14_data_sha256"] == ft.sha(data_path)
                assert identity["source_sha256"] == ft.sha(Path(__file__).with_name("future_train.py"))
                assert identity["protocol_sha256"] == ft.sha(Path(__file__).with_name("PROTOCOL.md"))
                assert identity["protocol_lock_sha256"] == ft.sha(Path(__file__).with_name("PROTOCOL_LOCK.json"))
                assert identity["source_lock_sha256"] == ft.sha(Path(__file__).with_name("SOURCE_LOCK.json"))
                assert identity["input_lock_sha256"] == ft.sha(Path(__file__).with_name("INPUT_LOCK.json"))
                assert identity["history_steps"] == (1 if group == "C_current" else 9)
                assert torch.count_nonzero(best["normalizer"]["residual_offset"]) == 0
                metrics = [row["selection_mae"] for row in best["history"]]
                minimum = min(metrics)
                earliest = metrics.index(minimum)
                assert best["best_epoch"] == earliest and best["epoch"] == earliest
                assert abs(best["best_metric"] - minimum) < 1e-12
                assert latest["best_epoch"] == earliest and summary["best_epoch"] == earliest
                key = (fold, seed)
                normalized = best["normalizer"]
                fingerprint = {name: tensor_hash(normalized[name]) for name in ("x_mean", "x_std", "residual_scale", "residual_offset")}
                if key in pair_norms:
                    assert pair_norms[key] == fingerprint
                else:
                    pair_norms[key] = fingerprint
                records.append({"run": run, "status": "accepted", "epochs": len(latest["history"]), "best_epoch": earliest,
                                "best_metric_native_n_mae": minimum, "parameters": summary["parameters"],
                                "best_sha256": ft.sha(run_root / "best.pth"), "latest_sha256": ft.sha(run_root / "latest.pth"),
                                "data_sha256": ft.sha(data_path), "normalizer": fingerprint})
    assert len(records) == 24 and all(record["parameters"] == 23961 for record in records)
    receipt = {"schema": "round15_training_audit_v1", "status": "pass", "accepted_count": len(records),
               "normalizer_pair_identity_checks": len(pair_norms), "test_consumed": False, "smoke_checkpoint_used": False,
               "checks": ["fixed_grid", "formal_input_hash", "all_lock_hashes", "earliest_strict_selection", "best_latest_consistency", "scale_only_zero_offset", "C_H_same_foldseed_normalizer", "identical_parameter_count"],
               "runs": records}
    ft.atomic_json(receipt, arguments.output)
    print(json.dumps({"status": "pass", "runs": len(records)}))


if __name__ == "__main__":
    main()
