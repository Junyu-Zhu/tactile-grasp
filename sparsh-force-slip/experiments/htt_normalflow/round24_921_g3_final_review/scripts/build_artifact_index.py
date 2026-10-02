#!/usr/bin/env python3
"""Index Round24 author artifacts without hashing large upstream model files."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
EXCLUDE = {
    "ARTIFACT_INDEX.json",
    "SYNC_PROOF.json",
    "INDEPENDENT_RECHECK.json",
    "INDEPENDENT_RECHECK.md",
    "ROOT_G3_ACCEPTANCE.json",
    "ROOT_G3_SYNC_PROOF.json",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    entries = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel in EXCLUDE or "__pycache__" in path.parts or rel.endswith(".pyc"):
            continue
        entries.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha256(path)})
    index = {
        "schema": "round24_g3_artifact_index_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "author_complete_pending_independent_review",
        "files": entries,
        "file_count": len(entries),
        "excluded_future_or_self_referential": sorted(EXCLUDE),
        "upstream_large_checkpoints": "not copied or rehashed; accepted G1/G2 root audits and indexes reused",
        "test_consumed": False,
    }
    (ROOT / "ARTIFACT_INDEX.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"file_count": len(entries)}, sort_keys=True))


if __name__ == "__main__":
    main()
