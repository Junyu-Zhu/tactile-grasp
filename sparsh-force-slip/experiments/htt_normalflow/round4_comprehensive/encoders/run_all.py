#!/usr/bin/env python3
"""Sequential, resumable scheduler for one Round 4 encoder on one assigned GPU."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
FOLDS = tuple(f"htt_leave_p{i}" for i in range(1, 5))
SEEDS = (20260914, 20260915, 20260916)


def atomic_json(path: Path, payload: Any) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def command(args: argparse.Namespace, fold: str, seed: int, output: Path, smoke: bool) -> list[str]:
    cmd = [
        sys.executable, str(HERE / "training.py"),
        "--cache-dir", str(args.cache_dir.resolve()),
        "--encoder", args.encoder,
        "--fold", fold,
        "--seed", str(seed),
        "--output-dir", str(output.resolve()),
        "--device", args.device,
        "--workers", str(args.workers),
        "--batch-size", str(args.batch_size),
    ]
    if smoke:
        cmd.extend(["--smoke", "--smoke-max-samples", str(args.smoke_max_samples)])
    return cmd


def run(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    jobs = [
        {
            "kind": "smoke",
            "fold": FOLDS[0],
            "seed": SEEDS[0],
            "output": root / "smoke" / args.encoder,
            "smoke": True,
        },
        *[
            {
                "kind": "formal",
                "fold": fold,
                "seed": seed,
                "output": root / "runs" / args.encoder / fold / str(seed),
                "smoke": False,
            }
            for fold in FOLDS for seed in SEEDS
        ],
    ]
    plan_path = root / f"plan_{args.encoder}.json"
    frozen_plan = {
        "encoder": args.encoder,
        "cache_dir": str(args.cache_dir.resolve()),
        "device": args.device,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "jobs": [{**job, "output": str(job["output"])} for job in jobs],
    }
    if plan_path.exists() and json.loads(plan_path.read_text()) != frozen_plan:
        raise RuntimeError("Refusing to change an existing encoder schedule")
    atomic_json(plan_path, frozen_plan)
    if args.plan_only:
        print(json.dumps(frozen_plan, indent=2))
        return

    env = os.environ.copy()
    env["XFORMERS_DISABLED"] = "1"
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    verify_log = logs / f"verify_cache_{args.encoder}.log"
    verify_cmd = [sys.executable, str(HERE / "cache.py"), "verify", "--output-dir", str(args.cache_dir.resolve())]
    with verify_log.open("a") as stream:
        stream.write("COMMAND " + " ".join(verify_cmd) + "\n")
        stream.flush()
        verified = subprocess.run(verify_cmd, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if verified.returncode:
        raise RuntimeError(f"Cache verification failed; see {verify_log}")
    state: list[dict[str, Any]] = []
    for job in jobs:
        # Always invoke training.py: it is the authority that checks the full
        # configuration plus every checkpoint/prediction hash before reuse.
        cmd = command(args, job["fold"], job["seed"], job["output"], job["smoke"])
        log = logs / f"train_{args.encoder}_{job['kind']}_{job['fold']}_{job['seed']}.log"
        with log.open("a") as stream:
            stream.write("COMMAND " + " ".join(cmd) + "\n")
            stream.flush()
            result = subprocess.run(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT)
        status = "complete" if result.returncode == 0 else "failed"
        state.append({"fold": job["fold"], "seed": job["seed"], "kind": job["kind"],
                      "status": status, "returncode": result.returncode, "log": str(log)})
        atomic_json(root / f"state_{args.encoder}.json", state)
        if result.returncode:
            raise SystemExit(result.returncode)
    print(json.dumps({"encoder": args.encoder, "status": "complete", "jobs": len(jobs)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=("dino", "ijepa", "mae_letterbox"), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--smoke-max-samples", type=int, default=256)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if min(args.workers, args.batch_size, args.smoke_max_samples) < 0 or args.batch_size < 1 or args.smoke_max_samples < 2:
        parser.error("workers must be nonnegative and sample sizes positive")
    return args


if __name__ == "__main__":
    run(parse_args())
