#!/usr/bin/env python3
"""Audit final Round-5 contract without rehashing large immutable token files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from contract import FOLDS, atomic_json, sha256_file, validate_contract


def keyed(payload):
    return {entry["episode_id"]: entry for entry in payload["entries"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--raw-force-cache", type=Path, required=True)
    parser.add_argument("--contact-force-cache", type=Path, required=True)
    parser.add_argument("--slip-cache", type=Path, required=True)
    parser.add_argument("--standardizer-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    raw = json.loads(args.raw_force_cache.read_text())
    contact = json.loads(args.contact_force_cache.read_text())
    slip = json.loads(args.slip_cache.read_text())
    validate_contract(contract, require_files=True)
    final_force = keyed({"entries": [e for e in contract["entries"] if e["task"] == "force"]})
    final_slip = keyed({"entries": [e for e in contract["entries"] if e["task"] == "slip"]})
    raw_force, contact_force, old_slip = keyed(raw), keyed(contact), keyed(slip)
    errors = []
    if set(final_force) != set(raw_force) or set(final_force) != set(contact_force) or len(final_force) != 101:
        errors.append("force episode coverage mismatch")
    if set(final_slip) != set(old_slip) or len(final_slip) != 101:
        errors.append("slip episode coverage mismatch")
    for episode_id, entry in final_force.items():
        source, transformed = raw_force[episode_id], contact_force[episode_id]
        if (entry["token_path"], entry["token_sha256"]) != (source["token_path"], source["token_sha256"]):
            errors.append(f"{episode_id}: force token provenance changed")
        if entry["force_native_n_path"] == entry["force_raw_n_path"]:
            errors.append(f"{episode_id}: contact target overwrote raw target")
        if sha256_file(entry["force_native_n_path"]) != entry["force_native_n_sha256"]:
            errors.append(f"{episode_id}: contact target hash mismatch")
        if entry["force_native_n_sha256"] != transformed["force_native_n_sha256"]:
            errors.append(f"{episode_id}: transformed target provenance mismatch")
    for episode_id, entry in final_slip.items():
        source = old_slip[episode_id]
        if (entry["token_path"], entry["token_sha256"], entry["label_path"], entry["label_sha256"]) != (
                source["token_path"], source["token_sha256"], source["label_path"], source["label_sha256"]):
            errors.append(f"{episode_id}: slip cache provenance changed")
    normalizers = {}
    for fold in FOLDS:
        path = args.standardizer_dir / f"{fold}.force_standardizer.json"
        row = json.loads(path.read_text())
        normalizers[fold] = {"sha256": sha256_file(path), "count": row["count"],
                             "train_only": not row["test_or_validation_used"]}
        if row["fold"] != fold or row["role"] != "train" or row["test_or_validation_used"]:
            errors.append(f"{fold}: unsafe standardizer")
    result = {
        "status": "pass" if not errors else "fail", "errors": errors,
        "contract_sha256": sha256_file(args.contract), "entries": len(contract["entries"]),
        "force_entries": len(final_force), "slip_entries": len(final_slip),
        "large_token_bytes_rehashed": False,
        "token_integrity_evidence": "path+recorded SHA equality to preserved complete source manifests; target arrays and hashes checked",
        "raw_force_cache_preserved": True, "test_role_selected": False,
        "standardizers": normalizers,
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

