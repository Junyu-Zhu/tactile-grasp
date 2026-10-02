#!/usr/bin/env python3
"""Root-dispatched resumable runner for the fixed 24-run Round-15 grid."""
import argparse
import datetime
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import future_train as ft

DISPATCH_CUTOFF = datetime.datetime.fromisoformat("2026-09-23T01:55:53+08:00")


def assert_dispatch_open():
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
    if now >= DISPATCH_CUTOFF:
        raise SystemExit(f"new training dispatch forbidden at {now.isoformat()}; cutoff={DISPATCH_CUTOFF.isoformat()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--devices", default="cuda:0")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--only", choices=ft.GROUPS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--execute-formal", action="store_true")
    arguments = parser.parse_args()
    if not arguments.smoke and not arguments.execute_formal:
        raise SystemExit("formal execution requires --execute-formal")
    assert_dispatch_open()
    if arguments.execute_formal and (
        arguments.output_root.name != "formal" or "smoke" in str(arguments.output_root).lower()
    ):
        raise SystemExit("formal output-root identity rejected")
    devices = tuple(item.strip() for item in arguments.devices.split(",") if item.strip())
    if not devices or arguments.max_parallel != len(devices) or len(devices) > 3:
        raise SystemExit("device list must equal max-parallel and cannot exceed 3")
    groups = (arguments.only,) if arguments.only else ft.GROUPS
    jobs = []
    for group in groups:
        for fold in range(1, 5):
            for seed in ft.SEEDS:
                run = f"{group}_p{fold}_s{seed}"
                jobs.append(
                    (
                        run,
                        arguments.data_root / f"p{fold}_s{seed}" / "prepared.pt",
                        arguments.output_root / run,
                        group,
                        fold,
                        seed,
                    )
                )

    def execute(job, device):
        run, data, output, group, fold, seed = job
        if "smoke" in str(data).lower():
            return {"run": run, "device": device, "returncode": 2, "stderr_tail": "smoke input rejected"}
        if (output / "summary.json").exists() and (output / "best.pth").exists() and (output / "latest.pth").exists():
            old = json.loads((output / "summary.json").read_text())
            if old.get("status") == "complete":
                return {"run": run, "device": device, "returncode": 0, "reused": True, "stdout_tail": "accepted complete", "stderr_tail": ""}
        command = [
            sys.executable,
            str(Path(__file__).with_name("future_train.py")),
            "--data", str(data),
            "--output", str(output),
            "--group", group,
            "--fold", str(fold),
            "--seed", str(seed),
            "--device", device,
        ]
        if arguments.smoke:
            command.extend(["--max-epochs", "2", "--patience", "99"])
        completed = subprocess.run(command, text=True, capture_output=True)
        return {
            "run": run,
            "device": device,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-1000:],
            "stderr_tail": completed.stderr[-1000:],
        }

    records = []
    for start in range(0, len(jobs), len(devices)):
        assert_dispatch_open()
        batch = jobs[start : start + len(devices)]
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            batch_records = list(executor.map(lambda pair: execute(*pair), zip(batch, devices)))
        records.extend(batch_records)
        ft.atomic_json({"status": "running", "records": records}, arguments.output_root / "RUN_STATE.json")
        if any(record["returncode"] for record in batch_records):
            ft.atomic_json({"status": "failed", "records": records}, arguments.output_root / "RUN_STATE.json")
            raise SystemExit(1)
    ft.atomic_json({"status": "complete", "records": records}, arguments.output_root / "RUN_STATE.json")
    print(json.dumps({"status": "complete", "runs": len(records)}))


if __name__ == "__main__":
    main()
