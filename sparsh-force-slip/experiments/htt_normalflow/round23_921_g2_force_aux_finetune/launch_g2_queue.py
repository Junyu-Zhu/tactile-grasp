#!/usr/bin/env python3
"""Detach the independently versioned Round23 G2 queue."""
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYTHON = "/home/zjy/miniconda3/envs/sparsh/bin/python"
RUN_ROOT = "/vla1/zjy/sparsh_runs/force_slip_htt_normalflow"
SOURCE_ROOT = "/vla1/zjy/sparsh_runs/force_slip_phase2"
REPO = HERE.parents[2]


def command():
    return [
        PYTHON, str(HERE / "formal_queue.py"),
        "--authorization", str(HERE / "G2_FORMAL_AUTHORIZATION.json"),
        "--run-inventory", str(HERE / "G2_RUN_INVENTORY.json"),
        "--state", str(HERE / "G2_QUEUE_STATE.json"),
        "--lock", f"{RUN_ROOT}/round23_921_g2_force_aux_finetune/g2_queue.lock",
        "--python", PYTHON, "--repo", str(REPO), "--run-root", RUN_ROOT,
        "--source-root", SOURCE_ROOT, "--gpu", "0", "1", "2", "--execute",
    ]


def main():
    logs = HERE / "logs"
    logs.mkdir(exist_ok=True)
    pid_path = logs / "g2_queue.pid"
    log_path = logs / "g2_queue.log"
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text().strip())
            os.kill(pid, 0)
        except (ValueError, ProcessLookupError, PermissionError):
            pass
        else:
            raise SystemExit(f"G2 queue already alive pid={pid}")
    env = os.environ.copy()
    env.update(XFORMERS_DISABLED="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    with log_path.open("ab", buffering=0) as log:
        proc = subprocess.Popen(command(), cwd=HERE, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    pid_path.write_text(f"{proc.pid}\n")
    print(json.dumps({"pid": proc.pid, "pid_file": str(pid_path), "log": str(log_path), "command": command()}))


if __name__ == "__main__":
    main()
