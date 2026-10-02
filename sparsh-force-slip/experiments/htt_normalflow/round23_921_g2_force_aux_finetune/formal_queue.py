#!/usr/bin/env python3
"""Independent G2 release queue using the immutable Round22 E3 implementation."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent / "round22_921_g1_joint_frozen"


def load_queue_module():
    spec = importlib.util.spec_from_file_location("round23_frozen_queue_core", CODE_ROOT / "formal_queue.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


Q = load_queue_module()
REQUIRED_AUTH_TRUE = (
    "g1_accepted",
    "gpu_uuid_health_rechecked",
    "remaining_budget_rechecked",
    "budget_pass",
    "e3_formal_authorized",
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def valid_existing_state(state, inventory, inventory_sha, authorization_sha):
    expected = {(r["run"], r["output"]) for r in inventory["runs"]}
    actual = {(r.get("run"), r.get("output")) for r in state.get("runs", [])}
    return (
        state.get("schema") == "round23_g2_queue_v1"
        and state.get("inventory_sha256") == inventory_sha
        and state.get("authorization_sha256") == authorization_sha
        and len(state.get("runs", [])) == 48
        and len(actual) == 48
        and actual == expected
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--run-inventory", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--gpu", nargs="+", default=["0", "1", "2"])
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    auth = json.loads(args.authorization.read_text())
    missing_auth = [key for key in REQUIRED_AUTH_TRUE if auth.get(key) is not True]
    inventory, inventory_issues = Q.load_inventory(args.run_inventory, CODE_ROOT / "train_e3.py")
    if args.execute and (missing_auth or inventory_issues):
        raise SystemExit(f"G2 dispatch blocked: authorization={missing_auth}, inventory={inventory_issues}")

    devices = Q.gpu_inventory(args.gpu)
    owner = Q.acquire(args.lock)
    inventory_sha = sha(args.run_inventory)
    authorization_sha = sha(args.authorization)
    if not args.state.exists():
        runs = []
        for item in inventory["runs"]:
            expected = {
                "group": item["group"],
                "fold": item["fold"],
                "seed": item["seed"],
                "data_sha256": item["data_sha256"],
                "prefix_index_sha256": item["prefix_index_sha256"],
                "source_sha256": item["source_sha256"],
                "visual_checkpoint_sha256": item["visual_checkpoint_sha256"],
                "protocol_sha256": item["protocol_sha256"],
                "source_sha256_code": item["train_source_sha256"],
                "authorization_sha256": authorization_sha,
                "formal": True,
            }
            runs.append({**item, "expected_identity": expected, "status": "registered_not_dispatched", "attempts": 0, "pid": None})
        Q.atomic(args.state, {
            "schema": "round23_g2_queue_v1",
            "created_at": Q.now().isoformat(),
            "stop_dispatch": Q.STOP.isoformat(),
            "concurrency_limit": len(devices),
            "quarantined_devices": [],
            "events": [],
            "inventory": str(args.run_inventory),
            "inventory_sha256": inventory_sha,
            "authorization": str(args.authorization),
            "authorization_sha256": authorization_sha,
            "immutable_code_root": str(CODE_ROOT),
            "train_source_sha256": sha(CODE_ROOT / "train_e3.py"),
            "queue_core_sha256": sha(CODE_ROOT / "formal_queue.py"),
            "runs": runs,
        })
    else:
        state = json.loads(args.state.read_text())
        if not valid_existing_state(state, inventory, inventory_sha, authorization_sha):
            raise SystemExit("existing Round23 queue state does not match exact inventory and authorization")

    receipt = {
        "schema": "round23_g2_queue_machine_v1",
        "status": "prepared_not_dispatched" if not inventory_issues else "inventory_blocked",
        "g2_ready": not missing_auth and not inventory_issues,
        "runs": 48,
        "authorization_missing_true": missing_auth,
        "authorization_sha256": authorization_sha,
        "inventory_issues": inventory_issues,
        "inventory_sha256": inventory_sha,
        "gpus": devices,
        "stop_dispatch": Q.STOP.isoformat(),
        "train_source": str(CODE_ROOT / "train_e3.py"),
        "train_source_sha256": sha(CODE_ROOT / "train_e3.py"),
        "queue_core_sha256": sha(CODE_ROOT / "formal_queue.py"),
        "release_queue_sha256": sha(__file__),
    }
    if not args.execute:
        print(json.dumps(receipt, sort_keys=True))
        return

    def build(run, _uuid):
        log = HERE / "logs" / f"formal_{run['group']}_p{run['fold']}_s{run['seed']}.log"
        command = [
            args.python, str(CODE_ROOT / "train_e3.py"),
            "--data", run["data"], "--prefix-index", run["prefix_index"],
            "--source", run["source"], "--visual-checkpoint", run["visual_checkpoint"],
            "--output", run["output"], "--group", run["group"],
            "--fold", str(run["fold"]), "--seed", str(run["seed"]),
            "--formal", "--authorization", str(args.authorization), "--protocol", run["protocol"],
        ]
        return command, log

    state = Q.run_queue(
        args.state, args.lock, [x["uuid"] for x in devices], Q.STOP,
        Q.accepted_e3,
        lambda run: Q.pid_matches(run.get("pid"), ("train_e3.py", run["output"], "--formal")),
        build,
        lambda uuid: {"CUDA_VISIBLE_DEVICES": uuid, "XFORMERS_DISABLED": "1"},
        launch_discover=lambda run: Q.discover_pids(("train_e3.py", run["output"], "--formal")),
        owner=owner,
    )
    print(json.dumps({status: sum(r["status"] == status for r in state["runs"]) for status in sorted({r["status"] for r in state["runs"]})}))


if __name__ == "__main__":
    main()
