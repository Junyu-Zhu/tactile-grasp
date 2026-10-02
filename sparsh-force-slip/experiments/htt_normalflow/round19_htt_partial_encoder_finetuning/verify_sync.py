#!/usr/bin/env python3
"""Compare every locally delivered R19 file with its authoritative server copy."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

LOCAL = Path(__file__).resolve().parent
SERVER_CODE = Path("/home/zjy/document/tactile-grasp/sparsh-force-slip/experiments/htt_normalflow/round19_htt_partial_encoder_finetuning")
SERVER_RESULTS = Path("/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round19_htt_partial_encoder_finetuning/results")
OUTPUT = LOCAL / "SYNC_PROOF.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] == "--server-manifest":
        paths = json.load(sys.stdin)
        root = Path("/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round19_htt_partial_encoder_finetuning/results")
        remote = {}
        for rel in paths:
            path = root / rel[len("results/"):] if rel.startswith("results/") else LOCAL / rel
            remote[rel] = sha(path) if path.is_file() else None
        print(json.dumps(remote, sort_keys=True))
        return
    paths = sorted(
        str(path.relative_to(LOCAL))
        for path in LOCAL.rglob("*")
        if path.is_file()
        and path != OUTPUT
        and "__pycache__" not in path.parts
        and not path.name.endswith(".lock")
    )
    process = subprocess.run(
        ["ssh", "zjy-4090", "/home/zjy/miniconda3/envs/sparsh/bin/python", str(SERVER_CODE / "verify_sync.py"), "--server-manifest"],
        input=json.dumps(paths), text=True, capture_output=True, check=True,
    )
    remote = json.loads(process.stdout)
    rows = []
    for rel in paths:
        local_hash = sha(LOCAL / rel)
        rows.append({"relative_path": rel, "local_sha256": local_hash, "server_sha256": remote.get(rel), "match": local_hash == remote.get(rel)})
    result = {
        "schema": "round19_per_file_sync_proof_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "local_root": str(LOCAL),
        "server_code_root": str(SERVER_CODE),
        "server_results_root": str(SERVER_RESULTS),
        "scope": "Every locally delivered R19 file, excluding this self-referential proof, pycache, and runtime locks. Large checkpoints, prefix cache, and prediction NPZ remain server-only and are indexed by checkpoint/prediction manifests.",
        "files": len(rows),
        "matches": sum(row["match"] for row in rows),
        "status": "pass" if all(row["match"] for row in rows) else "fail",
        "rows": rows,
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "files": result["files"], "matches": result["matches"], "mismatches": [r["relative_path"] for r in rows if not r["match"]]}, ensure_ascii=False))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
