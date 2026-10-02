#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import future_train as ft


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2")
    arguments = parser.parse_args()
    devices = tuple(item.strip() for item in arguments.devices.split(",") if item.strip())
    if not devices or arguments.max_parallel != len(devices):
        raise SystemExit("devices must equal max-parallel")
    jobs = [(group, fold, seed) for group in ft.GROUPS for fold in range(1, 5) for seed in ft.SEEDS]

    def one(pair):
        (group, fold, seed), device = pair
        run = f"{group}_p{fold}_s{seed}"
        command = [sys.executable, str(Path(__file__).with_name("evaluate.py")),
                   "--data", str(arguments.data_root / f"p{fold}_s{seed}" / "prepared.pt"),
                   "--checkpoint", str(arguments.formal / run / "best.pth"),
                   "--group", group, "--output", str(arguments.output / run), "--device", device]
        completed = subprocess.run(command, text=True, capture_output=True)
        return {"run": run, "device": device, "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-500:], "stderr_tail": completed.stderr[-1000:]}

    records = []
    for start in range(0, len(jobs), len(devices)):
        batch = jobs[start:start + len(devices)]
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            records.extend(executor.map(one, zip(batch, devices)))
        ft.atomic_json({"status": "running", "records": records}, arguments.output / "EVALUATION_STATE.json")
        if any(record["returncode"] for record in records[-len(batch):]):
            ft.atomic_json({"status": "failed", "records": records}, arguments.output / "EVALUATION_STATE.json")
            raise SystemExit(1)
    ft.atomic_json({"status": "complete", "records": records}, arguments.output / "EVALUATION_STATE.json")
    print(json.dumps({"status": "complete", "runs": len(records)}))


if __name__ == "__main__":
    main()
