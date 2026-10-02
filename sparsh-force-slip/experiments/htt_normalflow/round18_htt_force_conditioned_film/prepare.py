#!/usr/bin/env python3
"""Build Round-18 identity/support locks and the undispatched 36-run inventory."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import torch

import train as r18


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r17-prepared", type=Path, required=True)
    parser.add_argument("--remote-output", type=Path, required=True)
    parser.add_argument("--local-output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()
    caches = []
    support = []
    upstream = {}
    for fold in range(1, 5):
        for seed in r18.SEEDS:
            path = args.r17_prepared / f"p{fold}_s{seed}" / "prepared.pt"
            data = torch.load(path, map_location="cpu", weights_only=False)
            r18.validate_cache(data, path, fold, seed)
            record = {"fold": fold, "seed": seed, "path": str(path), "sha256": r18.sha(path), "shape": list(data["roles"]["fit"]["x"].shape)}
            caches.append(record)
            for role in r18.ROLES:
                item = data["roles"][role]
                support.append({
                    "fold": fold,
                    "seed": seed,
                    "role": role,
                    "endpoints": len(item["stage"]),
                    "episodes": len(set(item["episode_id"])),
                    "leakage_groups": len(set(item["leakage_group"])),
                    "static_frames": int(item["stage"].eq(0).sum()),
                    "incipient_frames": int(item["stage"].eq(1).sum()),
                    "gross_frames": int(item["stage"].eq(2).sum()),
                    "static_episodes": len(set(e for e, s in zip(item["episode_id"], item["stage"].tolist()) if s == 0)),
                    "incipient_episodes": len(set(e for e, s in zip(item["episode_id"], item["stage"].tolist()) if s == 1)),
                    "gross_episodes": len(set(e for e, s in zip(item["episode_id"], item["stage"].tolist()) if s == 2)),
                    "incipient_binary_supervision": False,
                })
            for name, source in data["provenance"]["immutable_upstream"].items():
                upstream[(fold, seed, name)] = {"fold": fold, "seed": seed, "name": name, **source}

    source_dir = Path(__file__).resolve().parent
    source_hashes = {path.name: r18.sha(path) for path in sorted(source_dir.glob("*.py"))}
    source_hashes["PROTOCOL.md"] = r18.sha(source_dir / "PROTOCOL.md")
    model_counts = {group: sum(parameter.numel() for parameter in r18.init_model(group, r18.SEEDS[0]).parameters()) for group in r18.GROUPS}
    capacity = {group: (count / model_counts["V0"] - 1.0) * 100.0 for group, count in model_counts.items()}
    if max(abs(value) for value in capacity.values()) > 5.0:
        raise ValueError(capacity)
    runs = []
    for group in r18.GROUPS:
        for fold in range(1, 5):
            for seed in r18.SEEDS:
                cache = next(item for item in caches if item["fold"] == fold and item["seed"] == seed)
                runs.append({
                    "run_id": f"{group}/p{fold}_s{seed}",
                    "group": group,
                    "fold": fold,
                    "seed": seed,
                    "status": "registered_not_dispatched",
                    "input_path": cache["path"],
                    "input_sha256": cache["sha256"],
                    "output": str(args.remote_output / "formal" / group / f"p{fold}_s{seed}"),
                    "gpu": None,
                })
    r11_source = source_dir.parent / "round11_stable_negative_force_residual" / "training" / "train.py"
    r12_protocol = source_dir.parent / "round12_class_preserving_trial_balance" / "PROTOCOL.md"
    identity = {
        "schema": "round18_identity_lock_v1",
        "created_at": now,
        "status": "pass",
        "r17_root_verification_reused": {
            "FINAL_STATUS_sha256": "436a7ee2a55abe84e0fa53c601727f445b6bc6d379a231d73b4481f323f124c0",
            "DELIVERY_MANIFEST_sha256": "3305e8471fc8c454804375f59fb37e07da01810cb0b899e007e383bc858c133c",
            "FINAL_SYNC_PROOF_sha256": "36f584b2e4042aa5e22be13949485925aef1c3bbaf145d4556d6c7be29d88a36",
            "verified_by": "root_agent_fresh_local_and_remote_check",
        },
        "caches": caches,
        "immutable_upstream": list(upstream.values()),
        "raw_image_dependency": "nine base steps; accepted upstream union t-13..t",
        "test_consumed": False,
        "force_source": "accepted Round-10 fold/seed predicted current force; same-image, no GT force input",
        "source_hashes": source_hashes,
    }
    duplicate = {
        "schema": "round18_r11_duplicate_check_v1",
        "status": "not_equivalent",
        "r11_source": str(r11_source),
        "r11_source_sha256": r18.sha(r11_source),
        "r12_protocol": str(r12_protocol),
        "r12_protocol_sha256": r18.sha(r12_protocol),
        "r11_formula": "logit = visual_logit + 2*tanh(MLP(concat(final_visual_hidden, final_force_hidden)))",
        "r11_injection": "bounded scalar post-GRU logit residual",
        "round18_formula": "vprime_t=(1+gamma(fhat_t))*LN(v_t)+beta(fhat_t)",
        "round18_injection": "per-time-step 192-D pre-GRU feature modulation",
        "round18_initialization": "gamma/beta output layer exactly zero; exact V0 identity",
        "r12_relation": "inherits V/F ordinary temporal heads; no feature FiLM",
    }
    budget = {
        "schema": "round18_budget_v1",
        "parameter_counts": model_counts,
        "percent_vs_V0": capacity,
        "within_5_percent": True,
        "invalid_or_padding_parameters": 0,
        "trainable_scope": "new LayerNorm/GRU/risk and M0 FiLM only",
        "upstream_model_parameters": "unknown",
        "end_to_end_gpu_memory_and_latency": "unknown until separate full-image benchmark; cached-head smoke is not end-to-end",
        "actual_cached_head_smoke": "pending SMOKE.json",
        "formal_run_count": 36,
        "concurrency_limit": 3,
        "one_run_per_gpu": True,
        "oom_policy": "reduce Round-18 concurrency before any fair group-wide batch revision; never touch other jobs",
    }
    r18.atomic_json(args.local_output / "IDENTITY_LOCK.json", identity)
    r18.atomic_json(args.local_output / "SUPPORT_AUDIT.json", {"schema": "round18_support_audit_v1", "status": "pass", "records": support, "incipient": "descriptive_only_not_binary_supervision", "test_consumed": False})
    r18.atomic_json(args.local_output / "R11_NON_EQUIVALENCE.json", duplicate)
    r18.atomic_json(args.local_output / "BUDGET.json", budget)
    r18.atomic_json(args.local_output / "RUN_INVENTORY.json", {"schema": "round18_run_inventory_v1", "registered": len(runs), "dispatched": 0, "runs": runs})
    r18.atomic_json(args.local_output / "PROTOCOL_LOCK.json", {"schema": "round18_protocol_lock_v1", "status": "locked", "protocol_sha256": r18.sha(source_dir / "PROTOCOL.md"), "source_hashes": source_hashes, "grid": "V0/C0/M0 x p1..p4 x 20260914/15/16", "formal_authorized": False, "test_forbidden": True})
    r18.atomic_json(args.local_output / "DISPATCH_READY.json", {"schema": "round18_dispatch_ready_v1", "status": "blocked_pending_smoke_and_root_authorization", "registered_runs": len(runs), "formal_dispatched": False, "required_root_control": str(args.local_output / "ROOT_CONTROL.json")})
    print(json.dumps({"status": "prepared", "caches": len(caches), "runs": len(runs), "parameter_counts": model_counts, "capacity_percent_vs_V0": capacity}, indent=2))


if __name__ == "__main__":
    main()

