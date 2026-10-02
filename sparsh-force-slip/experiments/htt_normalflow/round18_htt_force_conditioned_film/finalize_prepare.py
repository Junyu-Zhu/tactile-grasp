#!/usr/bin/env python3
"""Compact Q0 smoke evidence into the code/report directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import train as r18


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-smoke", type=Path, required=True)
    parser.add_argument("--cpu-smoke", type=Path, required=True)
    parser.add_argument("--local", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    gpu = json.loads(args.gpu_smoke.read_text())
    cpu = json.loads(args.cpu_smoke.read_text())
    if gpu.get("status") != "pass" or cpu.get("status") != "pass":
        raise ValueError("smoke not accepted")
    if gpu.get("CUDA_VISIBLE_DEVICES") != "1" or gpu.get("device") != "cuda:0":
        raise ValueError("GPU1 isolation mismatch")
    r18.atomic_json(args.local / "SMOKE_GPU1.json", gpu)
    r18.atomic_json(args.local / "SMOKE_CPU.json", cpu)
    recovery = {
        "schema": "round18_recovery_audit_v1",
        "status": "pass",
        "real_subprocess": True,
        "interrupted_then_resumed_matches_uninterrupted": gpu["real_subprocess_interrupted_resume_exact"],
        "continuous_latest_state_sha256": gpu["continuous_latest_state_sha256"],
        "resumed_latest_state_sha256": gpu["resumed_latest_state_sha256"],
        "input_unchanged": gpu["input_unchanged"],
        "upstream_hashes_unchanged": gpu["upstream_hashes_unchanged"],
    }
    r18.atomic_json(args.local / "RECOVERY_AUDIT.json", recovery)
    budget_path = args.local / "BUDGET.json"
    budget = json.loads(budget_path.read_text())
    budget["actual_cached_head_smoke"] = gpu["actual_budget"]
    budget["gpu"] = gpu["gpu_name"]
    budget["physical_gpu"] = 1
    budget["peak_cuda_allocated_bytes_unit_process"] = gpu["peak_cuda_allocated_bytes_this_process"]
    budget["projected_upper_training_gpu_hours_from_subprocess_wall"] = sum(item["one_full_epoch_subprocess_seconds"] * 60 * 12 for item in gpu["actual_budget"].values()) / 3600
    budget["projection_boundary"] = "60-epoch upper bound from one-epoch process wall; early stopping may reduce it; evaluation/end-to-end image path excluded"
    budget["reserved_post_training_wall_hours"] = {"unified_evaluation_and_diagnostics": 4.0, "report_and_figures": 2.0, "independent_review_and_author_repairs": 4.0, "hash_sync_and_final_readonly_acceptance": 1.0, "contingency": 2.0}
    budget["reserved_post_training_total_wall_hours"] = 13.0
    budget["training_wall_upper_hours_at_three_jobs"] = budget["projected_upper_training_gpu_hours_from_subprocess_wall"] / 3.0
    budget["queue_failure_budget"] = {"generic_attempts_per_run": 2, "oom_attempts_per_run": 3, "oom_action": "decrease only Round-18 parallelism; persistent one-job OOM stops for a group-wide protocol revision", "scientific_or_identity_error": "terminal on first detection"}
    r18.atomic_json(budget_path, budget)
    r18.atomic_json(args.local / "DISPATCH_READY.json", {
        "schema": "round18_dispatch_ready_v1",
        "status": "pass_pending_root_authorization",
        "registered_runs": 36,
        "formal_dispatched": False,
        "gpu1_single_process_smoke": "pass",
        "cpu_smoke": "pass",
        "recovery": "pass",
        "protocol_sha256": r18.sha(args.local / "PROTOCOL.md"),
        "required_root_control": str(args.local / "ROOT_CONTROL.json"),
    })
    print(json.dumps({"status": "Q0_ready", "gpu_smoke": "pass", "cpu_smoke": "pass", "recovery": "pass", "formal_dispatched": False}, indent=2))


if __name__ == "__main__":
    main()
