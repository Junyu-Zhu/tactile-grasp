#!/usr/bin/env python3
"""Build the immutable Round14 payload manifest.

Three files are intentionally outside the payload hash set: the manifest itself,
FINAL_STATUS.json, and FINAL_SYNC_PROOF.json. They are verified independently by
the sync proof, which avoids a circular hash dependency.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXCLUDED = {"DELIVERY_MANIFEST.json", "FINAL_STATUS.json", "FINAL_SYNC_PROOF.json"}
CODE_MIRROR = "/home/zjy/document/tactile-grasp/sparsh-force-slip/experiments/htt_normalflow/round14_htt_future_force_dual"
OUTPUT_MIRROR = "/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round14_htt_future_force_dual/delivery"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    files = []
    for path in sorted(item for item in ROOT.rglob("*") if item.is_file()):
        relative = path.relative_to(ROOT).as_posix()
        if relative in EXCLUDED or "/__pycache__/" in f"/{relative}/":
            continue
        files.append(
            {
                "relative_path": relative,
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "code_mirror": f"{CODE_MIRROR}/{relative}",
                "output_delivery_mirror": f"{OUTPUT_MIRROR}/{relative}",
            }
        )
    object_ = {
        "schema": "round14_delivery_manifest_v2",
        "created_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        "status": "complete",
        "payload_file_count": len(files),
        "payload_total_bytes": sum(item["bytes"] for item in files),
        "payload_files": files,
        "self_reference_exclusions": {
            "paths": sorted(EXCLUDED),
            "reason": "Manifest cannot hash itself; status refers to final sync; proof records and verifies both plus itself separately.",
            "verification": "FINAL_SYNC_PROOF.json contains local and both-mirror hashes for FINAL_STATUS.json and DELIVERY_MANIFEST.json; the proof is copied last and then independently rehashed on both mirrors.",
        },
        "remote_large_artifacts": {
            "root": "/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round14_htt_future_force_dual",
            "exact_index": "REMOTE_ARTIFACT_INDEX.json",
            "formal_best_checkpoints": 36,
            "formal_latest_checkpoints": 36,
            "prepared_caches": 12,
            "evaluation_runs": 36,
        },
        "scope_completed": ["P0", "P1", "P2", "applicable_P5"],
        "scope_excluded": ["P3", "P4", "test", "new_force_training", "new_current_slip_training"],
        "test_role_consumed": False,
    }
    output = ROOT / "DELIVERY_MANIFEST.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(object_, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"status": "complete", "payload_files": len(files), "sha256": sha256(output)}))


if __name__ == "__main__":
    main()
