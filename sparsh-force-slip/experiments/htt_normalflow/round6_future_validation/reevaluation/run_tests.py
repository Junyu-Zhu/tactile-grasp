#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.discover(str(HERE), pattern="test_evaluate.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    payload = {
        "status": "pass" if result.wasSuccessful() else "fail",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "source_hashes": {str(path.resolve()): sha(path) for path in (HERE / "evaluate.py", HERE / "test_evaluate.py", Path(__file__))},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, args.output)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
