#!/usr/bin/env python3
"""Verify the frozen legacy model in one environment without optimizer updates."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


MAIN_REPO = Path("/home/zjy/document/tactile-grasp")
SPARSH_REPO = Path("/home/zjy/document/sparsh")
EXPERIMENT_ROOT = MAIN_REPO / "sparsh-force-slip/experiments/htt_normalflow"
SCRIPT_DIR = MAIN_REPO / "sparsh-force-slip/scripts"
DEFAULT_MANIFEST = Path(
    "/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round1/splits.json"
)
DEFAULT_CHECKPOINT = Path(
    "/vla1/zjy/sparsh_runs/force_slip_phase2/"
    "phase_lambda_decoupled_mae_lam010_20260528_000000/"
    "mae_decoupled_multitask/checkpoints/epoch-0030.pth"
)

for item in (SPARSH_REPO, EXPERIMENT_ROOT, SCRIPT_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

os.environ.setdefault("XFORMERS_DISABLED", "1")

import adapters  # noqa: E402
import phase2_b_multitask as p2  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str | None]:
    names = (
        "torch", "torchvision", "lightning", "pytorch-lightning", "torchmetrics",
        "numpy", "opencv-python", "hydra-core", "omegaconf", "einops",
        "scikit-learn", "wandb", "xformers",
    )
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def gpu_snapshot() -> dict:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, text=True, capture_output=True, timeout=15, check=False)
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader"],
        text=True, capture_output=True, timeout=15, check=False,
    )
    return {
        "inventory_command": command,
        "inventory_returncode": result.returncode,
        "inventory": result.stdout.strip().splitlines(),
        "compute_processes": processes.stdout.strip().splitlines(),
    }


def load_samples(manifest_path: Path, fold: str, partition: str) -> tuple[torch.Tensor, dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = {row["id"]: row for row in manifest["episodes"]}
    episode_id = next(
        item for item in manifest["splits"][fold][partition]
        if rows[item]["domain"] == "htt" and rows[item]["frames"] >= 7
    )
    row = rows[episode_id]
    for source, expected in row["source_files"].items():
        actual = sha256(Path(source))
        if actual != expected:
            raise ValueError(f"Source hash changed for {source}: {actual} != {expected}")
    episode = adapters.load_htt(row["path"])
    times = (5, 6)
    samples = [adapters.window(episode, t, history=2, stride=5) for t in times]
    batch = adapters.collate(samples)
    image = batch["inputs"]["image"].contiguous()
    if tuple(image.shape) != (2, 6, 320, 240):
        raise ValueError(f"Unexpected input shape: {tuple(image.shape)}")
    if image.dtype != torch.float32 or not torch.isfinite(image).all():
        raise ValueError("Expected finite FP32 input")
    return image, {
        "fold": fold,
        "partition": partition,
        "episode_id": episode_id,
        "path": row["path"],
        "source_hashes": row["source_files"],
        "times": list(times),
        "causal_indices": [sample["metadata"]["indices"] for sample in samples],
        "input_shape": list(image.shape),
        "input_min": float(image.min()),
        "input_max": float(image.max()),
    }


def flattened_outputs(output: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat((output["force"], output["slip"]), dim=1).detach().cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tolerances", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--cpu-reference", type=Path)
    parser.add_argument("--fold", default="htt_leave_p1")
    parser.add_argument("--partition", default="calibration")
    args = parser.parse_args()

    tolerance = json.loads(args.tolerances.read_text(encoding="utf-8"))
    seed = int(tolerance["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.device.startswith("cuda"):
        if args.cpu_reference is None:
            raise ValueError("CUDA verification requires --cpu-reference")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.manual_seed_all(seed)

    image, sample = load_samples(args.manifest, args.fold, args.partition)
    device = torch.device(args.device)
    model, payload = p2.load_b_checkpoint(args.checkpoint, device)
    if model.encoder_name != "mae" or model.decoder_variant != "decoupled":
        raise ValueError("Expected frozen MAE decoupled epoch-30 checkpoint")
    if any(parameter.requires_grad for parameter in model.encoder.parameters()):
        raise ValueError("Frozen encoder unexpectedly has trainable parameters")

    model.eval()
    model.zero_grad(set_to_none=True)
    with torch.inference_mode():
        start = time.perf_counter()
        first_raw = model(image.to(device))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        first_ms = (time.perf_counter() - start) * 1000
        second_raw = model(image.to(device))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    first = flattened_outputs(first_raw)
    second = flattened_outputs(second_raw)
    repeat_equal = torch.equal(first, second)

    # Exercise the actual legacy trainable heads only. Encoder parameters are frozen,
    # and no optimizer is constructed or stepped.
    model.zero_grad(set_to_none=True)
    train_output = model(image.to(device))
    force_target = torch.zeros_like(train_output["force"])
    slip_target = torch.tensor([0, 1], dtype=torch.long, device=device)
    loss = F.smooth_l1_loss(train_output["force"], force_target) + F.cross_entropy(
        train_output["slip"], slip_target
    )
    loss.backward()
    head_grads = [
        parameter.grad.detach().cpu()
        for name, parameter in model.named_parameters()
        if not name.startswith("encoder.") and parameter.requires_grad and parameter.grad is not None
    ]
    backward = {
        "loss": float(loss.detach().cpu()),
        "head_gradient_tensors": len(head_grads),
        "all_head_gradients_finite": bool(head_grads and all(torch.isfinite(g).all() for g in head_grads)),
        "encoder_gradient_tensors": sum(
            parameter.grad is not None for parameter in model.encoder.parameters()
        ),
        "optimizer_constructed": False,
        "optimizer_step": False,
    }

    comparison = {"status": "not_requested"}
    reference_device = None
    if args.cpu_reference is not None:
        reference = json.loads(args.cpu_reference.read_text(encoding="utf-8"))
        reference_device = reference["cuda"]["selected_device"]
        if reference_device != "cpu":
            raise ValueError(f"Reference must be CPU, got {reference_device!r}")
        cpu = torch.tensor(reference["outputs"]["force_and_slip_logits"], dtype=torch.float32)
        difference = (first - cpu).abs()
        comparison = {
            "status": "pass" if torch.allclose(
                first, cpu, atol=float(tolerance["atol"]), rtol=float(tolerance["rtol"])
            ) else "fail",
            "cpu_reference": str(args.cpu_reference),
            "candidate_device": str(device),
            "reference_device": reference_device,
            "atol": tolerance["atol"],
            "rtol": tolerance["rtol"],
            "max_abs_difference": float(difference.max()),
            "mean_abs_difference": float(difference.mean()),
        }

    cuda_info = {
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "compiled_arches": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
        "selected_device": str(device),
    }
    if device.type == "cuda":
        cuda_info.update({
            "device_name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
        })

    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "pass" if repeat_equal and backward["all_head_gradients_finite"] and comparison["status"] != "fail" else "fail",
        "scope": "environment compatibility; no optimizer update and no accuracy/calibration claim",
        "host": {"hostname": platform.node(), "platform": platform.platform()},
        "python": {"executable": sys.executable, "version": sys.version, "packages": package_versions()},
        "cuda": cuda_info,
        "gpu_snapshot": gpu_snapshot(),
        "determinism": {
            "precision": tolerance["precision"],
            "seed": seed,
            "repeat_rule": tolerance["repeat_rule"],
            "repeat_equal": repeat_equal,
        },
        "tolerances": {"path": str(args.tolerances), "sha256": sha256(args.tolerances), **tolerance},
        "sample": sample,
        "artifacts": {
            "manifest": str(args.manifest),
            "manifest_sha256": sha256(args.manifest),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
            "checkpoint_epoch": payload.get("epoch"),
            "checkpoint_format": payload.get("format"),
            "adapter": str(Path(adapters.__file__).resolve()),
            "adapter_sha256": sha256(Path(adapters.__file__).resolve()),
            "phase2_loader": str(Path(p2.__file__).resolve()),
            "phase2_loader_sha256": sha256(Path(p2.__file__).resolve()),
        },
        "model": {
            "encoder": model.encoder_name,
            "decoder_variant": model.decoder_variant,
            "encoder_frozen": True,
            "first_forward_ms": first_ms,
        },
        "outputs": {"force_and_slip_logits": first.tolist()},
        "repeat_outputs": {"force_and_slip_logits": second.tolist()},
        "backward": backward,
        "cpu_cross_environment_comparison": comparison if device.type == "cpu" else {
            "status": "not_run", "reason": "candidate device is CUDA"
        },
        "cpu_gpu_comparison": comparison if device.type == "cuda" else {
            "status": "not_run", "reason": "all GPUs had active compute processes"
        },
        "limits": [
            "GPU pass requires an actual CUDA run and cannot be inferred from compiled architecture support.",
            "HTT force axes, signs, and units remain unmapped to the old three-axis output.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
