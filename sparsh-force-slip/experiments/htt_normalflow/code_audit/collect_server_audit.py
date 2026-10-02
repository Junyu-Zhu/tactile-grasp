#!/usr/bin/env python3
"""Capture a read-only, reproducible server/code/model audit for round 1."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


MAIN_REPO = Path("/home/zjy/document/tactile-grasp")
SPARSH_REPO = Path("/home/zjy/document/sparsh")
WORKSPACE = MAIN_REPO / "sparsh-force-slip"
SELECTED_STAGE1 = Path(
    "/vla1/zjy/sparsh_runs/force_slip_phase2/"
    "phase_lambda_decoupled_mae_lam010_20260528_000000/"
    "mae_decoupled_multitask/checkpoints/epoch-0030.pth"
)
MAE_ENCODER = Path("/vla1/zjy/sparsh_models/sparsh-mae-base/mae_vitbase.ckpt")
SELECTED_FUTURE = Path(
    "/vla1/zjy/sparsh_runs/force_slip_phase4/"
    "phase_lambda_best_future_features_20260528_000000/heads/"
    "phase_lambda_best_full_20260528_000000/checkpoints/best.pth"
)


def run(args: list[str], cwd: Path | None = None, timeout: int = 30) -> dict[str, Any]:
    try:
        result = subprocess.run(
            args, cwd=cwd, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=timeout, check=False,
        )
        return {
            "command": args,
            "returncode": result.returncode,
            "stdout": result.stdout.rstrip(),
            "stderr": result.stderr.rstrip(),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": args,
            "returncode": None,
            "stdout": (exc.stdout or "").rstrip() if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "").rstrip() if isinstance(exc.stderr, str) else "",
            "timed_out": True,
        }


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        record.update({"bytes": path.stat().st_size, "sha256": sha256(path)})
    return record


def git_record(repo: Path, output_dir: Path) -> dict[str, Any]:
    commands = {
        "status": ["git", "status", "--short", "--branch"],
        "head": ["git", "rev-parse", "HEAD"],
        "head_meta": ["git", "log", "-1", "--format=%H%n%cI%n%s"],
        "branches": ["git", "branch", "-vv", "--no-color"],
        "remotes": ["git", "remote", "-v"],
        "diff_stat": ["git", "diff", "--stat"],
        "diff_name_status": ["git", "diff", "--name-status"],
        "untracked": ["git", "ls-files", "--others", "--exclude-standard"],
    }
    results = {name: run(cmd, repo) for name, cmd in commands.items()}
    diff = run(["git", "diff", "--no-ext-diff", "--no-color"], repo)
    (output_dir / f"{repo.name}_tracked_diff.patch").write_text(
        diff["stdout"] + ("\n" if diff["stdout"] else ""), encoding="utf-8"
    )
    untracked = [line for line in results["untracked"]["stdout"].splitlines() if line]
    results["untracked_file_records"] = [file_record(repo / item) for item in untracked]
    return results


def package_versions(names: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def python_torch_probe(path: Path) -> dict[str, Any]:
    probe = (
        "import json,sys,torch; "
        "print(json.dumps({'python':sys.version.split()[0],"
        "'torch':torch.__version__,'torch_cuda':torch.version.cuda,"
        "'cuda_available':torch.cuda.is_available(),"
        "'compiled_arches':torch.cuda.get_arch_list(),"
        "'gpu_capability':list(torch.cuda.get_device_capability(0)) "
        "if torch.cuda.is_available() else None}))"
    )
    record = run([str(path), "-c", probe], timeout=30)
    record["path"] = str(path)
    return record


def render_markdown(payload: dict[str, Any]) -> str:
    main = payload["repositories"]["tactile-grasp"]
    sparsh = payload["repositories"]["sparsh"]
    gpu = payload["gpu"]["query"]["stdout"] or "nvidia-smi query failed"
    lines = [
        "# Round 1 code and environment audit", "",
        f"- generated: `{payload['generated_at']}`",
        f"- host: `{payload['host']['hostname']}`",
        f"- Python: `{payload['python']['executable']}` ({payload['python']['version']})",
        "- Scope: read-only repository/model inspection plus separately logged compatibility forward pass.",
        "",
        "## Repository state", "",
        f"- tactile-grasp HEAD: `{main['head']['stdout']}`; status: `{main['status']['stdout']}`",
        f"- Sparsh HEAD: `{sparsh['head']['stdout']}`; status follows:", "",
        "```text", sparsh["status"]["stdout"], "```", "",
        "Tracked Sparsh changes are preserved verbatim in `sparsh_tracked_diff.patch`; untracked files are recorded by path, size, and SHA-256 in `audit.json`.",
        "",
        "## Remote reachability", "",
    ]
    for name, result in payload["remote_reachability"].items():
        state = "timeout" if result["timed_out"] else f"exit {result['returncode']}"
        first = result["stdout"].splitlines()[0] if result["stdout"] else "no refs returned"
        lines.append(f"- {name}: `{state}`; `{first}`")
    lines += ["", "## GPU snapshot", "", "```text", gpu, "```", "", "## Selected model artifacts", ""]
    for name, record in payload["model_artifacts"].items():
        lines.append(f"- {name}: `{record['path']}`; exists={record['exists']}; sha256=`{record.get('sha256', 'n/a')}`")
    lines += [
        "", "## Reproduction-critical interfaces", "",
        "- Current force/slip Stage-I: `scripts/phase2_b_multitask.py`; `load_b_checkpoint()` reconstructs the frozen encoder and decoupled force/slip decoder from checkpoint metadata.",
        "- Selected current model: MAE, two causal frames, stride 5, channel concatenation to 6 channels, resize `(320, 240)`.",
        "- Shared image utility: `tactile_ssl.data.digit.utils.load_sample_from_buf(image, reference)` followed by `get_resize_transform((320, 240))`.",
        "- Current outputs: normalized three-axis force from `tanh` and two slip logits. Newton conversion uses `[1.5, 1.5, 2.0]` only for the original force convention; HTT 6-D axes/units remain an adapter-level unknown.",
        "- Future/world-model training: `scripts/phase3_2_world_model.py`, `scripts/phase4_paper_experiments.py`, and the later `phase_future_head_*.py` scripts.",
        "- Real-data evaluation: `scripts/evaluate_real_force_slip_model_test.py`. Its current path constructs the same two-frame MAE input causally and declares the selected Stage-I and Stage-II checkpoints.",
        "", "## Evidence boundaries", "",
        "- Evidence: repository status, hashes, package versions, remote probes, and GPU state were captured directly by this run.",
        "- Inference: passing a compatibility forward pass proves tensor/path compatibility only; it does not establish calibration or accuracy on HTT.",
        "- Unknown: HTT ATI force coordinate signs/units and cross-domain force scaling are not established by the Stage-I checkpoint.",
        "", "## GPU runtime compatibility", "",
        "- The Sparsh environment is the reproducible legacy environment, but its torch wheel must contain the GPU compute capability before CUDA execution is valid.",
        "- `python_environments` in `audit.json` records the compiled CUDA architecture list for the Sparsh, base, and TacEx Python installations. A compatible torch wheel in another environment is evidence of an available runtime component, not proof that the full Sparsh dependency stack works there.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    entrypoints = [
        WORKSPACE / "scripts/phase2_b_multitask.py",
        WORKSPACE / "scripts/phase3_2_world_model.py",
        WORKSPACE / "scripts/phase4_paper_experiments.py",
        WORKSPACE / "scripts/phase_future_head_medium.py",
        WORKSPACE / "scripts/evaluate_real_force_slip_model_test.py",
        WORKSPACE / "runbooks/launch_phase3_2_world_model_tmux.sh",
    ]
    payload: dict[str, Any] = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "kernel": platform.release(),
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version.replace("\n", " "),
            "packages": package_versions([
                "torch", "torchvision", "numpy", "opencv-python", "hydra-core",
                "omegaconf", "pytorch-lightning", "wandb", "Pillow",
            ]),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "python_environments": {
            "base": python_torch_probe(Path("/home/zjy/miniconda3/bin/python")),
            "sparsh": python_torch_probe(Path("/home/zjy/miniconda3/envs/sparsh/bin/python")),
            "tacex": python_torch_probe(Path("/home/zjy/miniconda3/envs/tacex/bin/python")),
        },
        "repositories": {
            "tactile-grasp": git_record(MAIN_REPO, args.output_dir),
            "sparsh": git_record(SPARSH_REPO, args.output_dir),
        },
        "remote_reachability": {
            "tactile-grasp origin SSH": run([
                "git", "ls-remote", "git@github.com:Junyu-Zhu/tactile-grasp.git",
                "HEAD", "refs/heads/main", "refs/heads/sparsh-force-slip",
            ], timeout=25),
            "Sparsh configured HTTPS": run([
                "git", "ls-remote", "https://github.com/facebookresearch/sparsh.git",
                "HEAD", "refs/heads/main",
            ], timeout=25),
            "Sparsh alternate SSH read-only probe": run([
                "git", "ls-remote", "git@github.com:facebookresearch/sparsh.git",
                "HEAD", "refs/heads/main",
            ], timeout=25),
        },
        "gpu": {
            "query": run([
                "nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version",
                "--format=csv,noheader",
            ]),
            "compute_processes": run([
                "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader",
            ]),
        },
        "model_artifacts": {
            "selected_stage1_force_slip": file_record(SELECTED_STAGE1),
            "mae_encoder": file_record(MAE_ENCODER),
            "selected_historical_future_head": file_record(SELECTED_FUTURE),
        },
        "entrypoints": {str(path): file_record(path) for path in entrypoints},
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "AUDIT.md").write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "audit": str(args.output_dir / "audit.json"),
        "report": str(args.output_dir / "AUDIT.md"),
        "stage1_sha256": payload["model_artifacts"]["selected_stage1_force_slip"].get("sha256"),
    }, indent=2))


if __name__ == "__main__":
    main()
