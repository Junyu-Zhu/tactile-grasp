#!/usr/bin/env python3
"""Perform the expensive Round 5 cache hash audit once before job fan-out."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import atomic_json, sha256_file
from data import load_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cache = load_cache(args.cache.resolve(), verify_hashes=True)
    payload = {"status": "pass", "format": "round5_cache_full_hash_audit_v1",
               "cache_manifest": cache["manifest_path"],
               "cache_manifest_sha256": cache["manifest_sha256"],
               "entries": len(cache["entries"]), "files_verified": len(cache["entries"]) * 2,
               "source_code_sha256": sha256_file(Path(__file__))}
    atomic_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
