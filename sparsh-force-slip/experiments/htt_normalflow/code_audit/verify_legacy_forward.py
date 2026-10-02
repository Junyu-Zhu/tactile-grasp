#!/usr/bin/env python3
"""Run the selected frozen MAE force/slip model on a causal HTT mini-batch."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


MAIN_REPO = Path("/home/zjy/document/tactile-grasp")
SPARSH_REPO = Path("/home/zjy/document/sparsh")
EXPERIMENT_ROOT = MAIN_REPO / "sparsh-force-slip/experiments/htt_normalflow"
SCRIPT_DIR = MAIN_REPO / "sparsh-force-slip/scripts"
DEFAULT_HTT = Path("/vla1/zjy/tactile_dataset/HTT-dataset/slip/gsmini/processed")
DEFAULT_CHECKPOINT = Path(
    "/vla1/zjy/sparsh_runs/force_slip_phase2/"
    "phase_lambda_decoupled_mae_lam010_20260528_000000/"
    "mae_decoupled_multitask/checkpoints/epoch-0030.pth"
)

for item in (SPARSH_REPO, EXPERIMENT_ROOT, SCRIPT_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

# CPU fallback must use the repository's standard PyTorch attention path.
# This is a smoke test, so deterministic compatibility matters more than the
# optional xFormers acceleration path.
os.environ.setdefault("XFORMERS_DISABLED", "1")

import adapters  # noqa: E402
import phase2_b_multitask as p2  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--htt-root", type=Path, default=DEFAULT_HTT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda:0"))
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    device_reason = "explicit request"
    requested_device = args.device
    if args.device == "auto":
        args.device = "cpu"
        device_reason = "CUDA unavailable"
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(0)
            capability_name = f"sm_{capability[0]}{capability[1]}"
            compiled_arches = torch.cuda.get_arch_list()
            if capability_name in compiled_arches:
                args.device = "cuda:0"
                device_reason = f"{capability_name} present in torch compiled arches"
            else:
                device_reason = (
                    f"CPU fallback: GPU capability {capability_name} absent from "
                    f"torch compiled arches {compiled_arches}"
                )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    files = sorted(args.htt_root.glob("p*_sliding/*.npz"))
    if not files:
        raise FileNotFoundError(f"No HTT GSmini slip episodes under {args.htt_root}")
    episode = adapters.load_htt(files[0])
    # Select real causal histories rather than padded episode-start frames.
    times = [5 + i for i in range(args.batch_size)]
    if times[-1] >= len(episode.images):
        raise ValueError("Episode is too short for requested batch")
    samples = [adapters.window(episode, t, history=2, stride=5) for t in times]
    batch = adapters.collate(samples)
    x = batch["inputs"]["image"]
    if tuple(x.shape[1:]) != (6, 320, 240):
        raise RuntimeError(f"Unexpected MAE input shape {tuple(x.shape)}")
    if not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise RuntimeError("Preprocessed input is not finite in [0,1]")

    device = torch.device(args.device)
    model, checkpoint_payload = p2.load_b_checkpoint(args.checkpoint, device)
    model.eval()
    if model.encoder_name != "mae" or model.decoder_variant != "decoupled":
        raise RuntimeError(
            f"Selected checkpoint reconstructed {model.encoder_name}/{model.decoder_variant}, "
            "expected mae/decoupled"
        )
    if any(parameter.requires_grad for parameter in model.encoder.parameters()):
        raise RuntimeError("Encoder is not frozen")
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    start = time.perf_counter()
    with torch.inference_mode():
        output = model(x.to(device))
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    force = output["force"].detach().cpu()
    slip_logits = output["slip"].detach().cpu()
    slip_probability = torch.softmax(slip_logits, dim=1)[:, 1]
    if tuple(force.shape) != (args.batch_size, 3) or tuple(slip_logits.shape) != (args.batch_size, 2):
        raise RuntimeError("Legacy output shapes do not match the force/slip contract")
    if not torch.isfinite(force).all() or not torch.isfinite(slip_logits).all():
        raise RuntimeError("Legacy model produced non-finite outputs")

    result = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "pass",
        "scope": "tensor/path compatibility only; no HTT accuracy or calibration claim",
        "python": sys.executable,
        "device": str(device),
        "device_request": requested_device,
        "device_selection_reason": device_reason,
        "episode": episode.episode_id,
        "source_file": str(files[0]),
        "adapter_file": str(Path(adapters.__file__).resolve()),
        "adapter_sha256": sha256(Path(adapters.__file__).resolve()),
        "batch_times": times,
        "causal_indices": [sample["metadata"]["indices"] for sample in samples],
        "reference": {
            "dtype": str(episode.reference.dtype),
            "shape": list(episode.reference.shape),
            "min": float(episode.reference.min()),
            "max": float(episode.reference.max()),
            "raw_0_255_compatible": bool(
                np.isfinite(episode.reference).all()
                and episode.reference.min() >= 0 and episode.reference.max() <= 255
            ),
        },
        "raw_images": {
            "dtype": str(episode.images.dtype),
            "shape": list(episode.images.shape),
            "min": int(episode.images.min()),
            "max": int(episode.images.max()),
        },
        "preprocess": {
            "interface": "load_sample_from_buf(image, reference) then get_resize_transform((320, 240))",
            "background_subtraction": True,
            "input_shape": list(x.shape),
            "dtype": str(x.dtype),
            "min": float(x.min()),
            "max": float(x.max()),
            "history": 2,
            "stride_steps": 5,
            "time_unit": "steps; HTT has no verified timestamps",
        },
        "model": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
            "checkpoint_epoch": checkpoint_payload.get("epoch"),
            "checkpoint_format": checkpoint_payload.get("format"),
            "encoder": model.encoder_name,
            "decoder_variant": model.decoder_variant,
            "encoder_frozen": True,
            "elapsed_batch_ms_single_measurement": elapsed_ms,
        },
        "outputs": {
            "force_normalized": force.tolist(),
            "slip_logits": slip_logits.tolist(),
            "slip_probability": slip_probability.tolist(),
        },
        "limits": [
            "HTT raw 6-D ATI axes, signs, and units are not mapped to the old 3-D force target here.",
            "The old checkpoint was trained on another GSmini data distribution; finite predictions do not imply transfer accuracy.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
