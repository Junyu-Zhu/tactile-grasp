#!/usr/bin/env python3
"""Build the deterministic Round-15 compact-delivery manifest."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

CONTROL_FILES = ("DELIVERY_MANIFEST.json", "FINAL_STATUS.json", "FINAL_SYNC_PROOF.json")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    output = arguments.output or root / "DELIVERY_MANIFEST.json"
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in CONTROL_FILES or "__pycache__" in path.parts:
            continue
        files.append({"relative_path": relative, "bytes": path.stat().st_size, "sha256": sha(path)})
    receipt = {"schema": "round15_delivery_manifest_v1", "status": "complete",
               "created_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
               "local_root": str(root), "file_count": len(files), "total_bytes": sum(row["bytes"] for row in files),
               "files": files, "self_referential_control_files_excluded_from_file_table": list(CONTROL_FILES),
               "large_artifacts": "Formal checkpoints, prediction payloads, and endpoint-level R14 diagnostic remain under the server output root and are indexed by REMOTE_ARTIFACT_INDEX.json."}
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "files": len(files), "bytes": receipt["total_bytes"]}))


if __name__ == "__main__":
    main()
