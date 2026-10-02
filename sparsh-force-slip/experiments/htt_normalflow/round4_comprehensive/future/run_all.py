#!/usr/bin/env python3
"""Run the six frozen round-4 future heads serially and resume safely."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


MODELS = ("mlp", "gru")
SEEDS = (20260914, 20260915, 20260916)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--support-audit", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    script = Path(__file__).with_name("train.py").resolve()
    env = dict(os.environ)
    env.setdefault("XFORMERS_DISABLED", "1")
    for model in MODELS:
        for seed in SEEDS:
            output = args.root.resolve() / model / f"seed_{seed}"
            output.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(script), "--cache-manifest", str(args.cache_manifest.resolve()),
                       "--support-audit", str(args.support_audit.resolve()), "--protocol", str(args.protocol.resolve()),
                       "--model", model, "--seed", str(seed), "--output", str(output),
                       "--device", args.device, "--workers", str(args.workers)]
            with (output / "console.log").open("a", encoding="utf-8") as log:
                log.write("COMMAND " + json.dumps(command) + "\n"); log.flush()
                subprocess.run(command, check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
            summary = json.loads((output / "training_summary.json").read_text())
            if summary.get("status") != "complete":
                raise RuntimeError(f"run did not complete: {model}/{seed}")
            print(json.dumps({"model": model, "seed": seed, "status": "complete"}), flush=True)


if __name__ == "__main__":
    main()
