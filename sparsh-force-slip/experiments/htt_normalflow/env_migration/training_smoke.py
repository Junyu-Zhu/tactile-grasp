#!/usr/bin/env python3
"""One-step legacy force-slip training smoke test for a migrated environment.

This is an environment/plumbing check, not a training experiment. It loads the
unchanged epoch-30 checkpoint, updates only its legacy decoder for one synthetic
target batch, and verifies a temporary decoder/optimizer checkpoint round trip.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

# Required by deterministic CUDA matrix multiplication. Set it before importing
# torch or creating any CUDA context.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F


MAIN_REPO = Path("/home/zjy/document/tactile-grasp")
SPARSH_REPO = Path("/home/zjy/document/sparsh")
EXPERIMENT_ROOT = MAIN_REPO / "sparsh-force-slip/experiments/htt_normalflow"
SCRIPT_DIR = MAIN_REPO / "sparsh-force-slip/scripts"
ROUND2_ENV_DIR = EXPERIMENT_ROOT / "round2_env"
DEFAULT_MANIFEST = Path(
    "/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round1/splits.json"
)
DEFAULT_CHECKPOINT = Path(
    "/vla1/zjy/sparsh_runs/force_slip_phase2/"
    "phase_lambda_decoupled_mae_lam010_20260528_000000/"
    "mae_decoupled_multitask/checkpoints/epoch-0030.pth"
)
DEFAULT_TOLERANCES = ROUND2_ENV_DIR / "tolerances.json"

for item in (SPARSH_REPO, EXPERIMENT_ROOT, SCRIPT_DIR, ROUND2_ENV_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

os.environ.setdefault("XFORMERS_DISABLED", "1")

import adapters  # noqa: E402
import phase2_b_multitask as p2  # noqa: E402
from verify_environment import (  # noqa: E402
    flattened_outputs,
    load_samples,
    package_versions,
    sha256,
)


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    """Hash names, tensor metadata, and exact bytes without retaining a clone."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def nested_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.device == right.device and torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            nested_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def tensor_pairs_difference(left: Any, right: Any, path: str = "root") -> dict:
    """Summarize exact tensor mismatches in matching nested state structures."""
    pairs: list[tuple[str, torch.Tensor, torch.Tensor]] = []

    def visit(a: Any, b: Any, current: str) -> None:
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            pairs.append((current, a, b))
        elif isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
            for key in a:
                visit(a[key], b[key], f"{current}.{key}")
        elif isinstance(a, (list, tuple)) and isinstance(b, type(a)) and len(a) == len(b):
            for index, (item_a, item_b) in enumerate(zip(a, b)):
                visit(item_a, item_b, f"{current}[{index}]")

    visit(left, right, path)
    mismatches = []
    maximum = 0.0
    for name, a, b in pairs:
        same_device = a.device == b.device
        same_shape = a.shape == b.shape
        equal = same_device and same_shape and torch.equal(a, b)
        if not equal:
            difference = None
            if same_shape and a.is_floating_point() and b.is_floating_point():
                difference = float((a.detach().cpu() - b.detach().cpu()).abs().max())
                maximum = max(maximum, difference)
            mismatches.append(
                {
                    "path": name,
                    "left_device": str(a.device),
                    "right_device": str(b.device),
                    "max_abs_difference": difference,
                }
            )
    return {
        "tensor_pairs": len(pairs),
        "mismatched_tensor_pairs": len(mismatches),
        "max_abs_difference": maximum,
        "first_mismatches": mismatches[:10],
    }


def verify_manifest_sources(row: dict) -> None:
    for source, expected in row["source_files"].items():
        actual = sha256(Path(source))
        if actual != expected:
            raise ValueError(f"Source hash changed for {source}: {actual} != {expected}")


def normalflow_train_smoke(
    manifest_path: Path, model: torch.nn.Module, device: torch.device
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episode_ids = manifest["splits"]["normalflow_objects"]["train"]
    if not episode_ids:
        raise ValueError("NormalFlow train partition is empty")
    rows = {row["id"]: row for row in manifest["episodes"]}
    episode_id = episode_ids[0]
    row = rows[episode_id]
    if row["domain"] != "normalflow" or episode_id in manifest["splits"]["normalflow_objects"]["test"]:
        raise ValueError("NormalFlow smoke sample is not an allowed train episode")
    verify_manifest_sources(row)
    episode = adapters.load_normalflow(row["path"])
    t = min(5, len(episode.images) - 1)
    sample = adapters.window(episode, t, history=2, stride=5)
    image = sample["inputs"]["image"].unsqueeze(0).contiguous().to(device)
    model.eval()
    with torch.inference_mode():
        output = model(image)
    values = flattened_outputs(output)
    return {
        "status": "pass" if torch.isfinite(values).all() else "fail",
        "partition": "train",
        "episode_id": episode_id,
        "path": row["path"],
        "source_hashes": row["source_files"],
        "frame": t,
        "causal_indices": sample["metadata"]["indices"],
        "input_shape": list(image.shape),
        "output_shape": list(values.shape),
        "all_outputs_finite": bool(torch.isfinite(values).all()),
        "reference_source": episode.metadata["reference_source"],
        "scope": "adapter and finite frozen forward only; no slip label or motion target",
    }


def markdown_report(report: dict) -> str:
    update = report["one_step_update"]
    resume = report["temporary_checkpoint_round_trip"]
    nf = report["normalflow_train_forward"]
    lines = [
        "# Migrated environment training smoke",
        "",
        f"- Overall status: **{report['status']}**",
        f"- Python: `{report['python']['executable']}`",
        f"- Device: `{report['device']['selected']}`",
        f"- Checkpoint SHA256 preserved: `{report['artifacts']['checkpoint_unchanged']}`",
        f"- Initial repeated forward bitwise equal: `{report['initial_repeat']['repeat_equal']}`",
        f"- Synthetic one-step loss: `{update['loss']:.8f}`",
        f"- Decoder tensors changed: `{update['changed_parameter_tensors']}`",
        f"- Encoder exact hash unchanged: `{update['encoder_state_unchanged']}`",
        f"- Decoder/optimizer checkpoint round trip: `{resume['status']}`",
        f"- Identical optimizer continuation: `{resume['identical_continuation_step']['status']}`",
        f"- NormalFlow train adapter forward: `{nf['status']}`",
        "",
        "The optimization targets are synthetic (zero normalized force and labels [0, 1]). ",
        "This result validates environment and checkpoint plumbing only; it is not an accuracy, convergence, or new-dataset training result.",
        "",
        "## Remaining full-entrypoint checks",
        "",
    ]
    lines.extend(f"- {item}" for item in report["limits"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tolerances", type=Path, default=DEFAULT_TOLERANCES)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--cpu-reference", type=Path)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--check-normalflow", action="store_true")
    args = parser.parse_args()

    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    result_path = args.output_dir / "training_smoke.json"
    report_path = args.output_dir / "TRAINING_SMOKE_REPORT.md"
    if result_path.exists() or report_path.exists():
        raise FileExistsError(f"Refusing to overwrite results under {args.output_dir}")

    tolerance = json.loads(args.tolerances.read_text(encoding="utf-8"))
    seed = int(tolerance["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.device == "cuda:0":
        if args.cpu_reference is None:
            raise ValueError("CUDA smoke requires --cpu-reference")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.manual_seed_all(seed)
        # Fused SDPA backward can differ by a few FP32 ulps across otherwise
        # identical calls. Math SDPA is used here because continuation is judged
        # by exact state equality, not by a relaxed numerical tolerance.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    checkpoint_hash_before = sha256(args.checkpoint)
    image, sample = load_samples(args.manifest, "htt_leave_p1", "calibration")
    device = torch.device(args.device)
    model, payload = p2.load_b_checkpoint(args.checkpoint, device)
    if model.encoder_name != "mae" or model.decoder_variant != "decoupled":
        raise ValueError("Expected frozen MAE decoupled epoch-30 checkpoint")
    if any(parameter.requires_grad for parameter in model.encoder.parameters()):
        raise ValueError("Encoder unexpectedly trainable")

    image = image.to(device)
    model.eval()
    with torch.inference_mode():
        initial = flattened_outputs(model(image))
        initial_repeat = flattened_outputs(model(image))
    if not torch.isfinite(initial).all():
        raise ValueError("Initial output is nonfinite")
    initial_repeat_equal = torch.equal(initial, initial_repeat)

    reference_comparison = {"status": "not_requested"}
    if args.cpu_reference is not None:
        reference = json.loads(args.cpu_reference.read_text(encoding="utf-8"))
        cpu = torch.tensor(reference["outputs"]["force_and_slip_logits"], dtype=torch.float32)
        difference = (initial - cpu).abs()
        passed = torch.allclose(
            initial, cpu, atol=float(tolerance["atol"]), rtol=float(tolerance["rtol"])
        )
        reference_comparison = {
            "status": "pass" if passed else "fail",
            "cpu_reference": str(args.cpu_reference),
            "atol": tolerance["atol"],
            "rtol": tolerance["rtol"],
            "max_abs_difference": float(difference.max()),
            "mean_abs_difference": float(difference.mean()),
        }

    encoder_hash_before = tensor_state_sha256(model.encoder.state_dict())
    decoder_before = {
        name: value.detach().cpu().clone() for name, value in model.decoder.state_dict().items()
    }
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=args.learning_rate)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(image)
    force_target = torch.zeros_like(output["force"])
    slip_target = torch.tensor([0, 1], dtype=torch.long, device=device)
    force_loss = F.smooth_l1_loss(output["force"], force_target)
    slip_loss = F.cross_entropy(output["slip"], slip_target)
    loss = force_loss + slip_loss
    loss.backward()
    decoder_grads = [
        parameter.grad for parameter in model.decoder.parameters() if parameter.grad is not None
    ]
    gradients_finite = bool(
        decoder_grads and all(torch.isfinite(gradient).all() for gradient in decoder_grads)
    )
    encoder_gradient_tensors = sum(
        parameter.grad is not None for parameter in model.encoder.parameters()
    )
    optimizer.step()

    decoder_after = model.decoder.state_dict()
    changed = []
    max_change = 0.0
    for name, value in decoder_after.items():
        difference = (value.detach().cpu() - decoder_before[name]).abs()
        if torch.any(difference != 0):
            changed.append(name)
            max_change = max(max_change, float(difference.max()))
    encoder_hash_after = tensor_state_sha256(model.encoder.state_dict())
    model.eval()
    with torch.inference_mode():
        updated = flattened_outputs(model(image))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_round_trip: dict[str, Any]
    with tempfile.TemporaryDirectory(prefix=".training_smoke_", dir=args.output_dir) as temporary:
        temporary_path = Path(temporary) / "decoder_optimizer.pt"
        saved = {
            "decoder_variant": model.decoder_variant,
            "decoder_state": model.decoder.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "synthetic_target": {
                "force": "zeros_like normalized 3-axis force",
                "slip_class": [0, 1],
            },
        }
        torch.save(saved, temporary_path)
        temporary_hash = sha256(temporary_path)
        # Load portable checkpoint tensors on CPU first. Optimizer.load_state_dict
        # then moves parameter-shaped states to the parameter device while keeping
        # Adam's non-capturable scalar step counter on CPU, matching live Adam.
        loaded = torch.load(temporary_path, map_location="cpu", weights_only=False)
        restored_decoder = p2.build_decoder(loaded["decoder_variant"]).to(device)
        restored_decoder.load_state_dict(loaded["decoder_state"], strict=True)
        restored_optimizer = torch.optim.Adam(restored_decoder.parameters(), lr=args.learning_rate)
        restored_optimizer.load_state_dict(loaded["optimizer_state"])
        model.encoder.eval()
        restored_decoder.eval()
        with torch.no_grad():
            latent = model.encoder(image)
            restored_raw = restored_decoder(latent)
        restored = flattened_outputs(restored_raw)
        output_equal = torch.equal(updated, restored)
        optimizer_equal_after_load = nested_equal(
            optimizer.state_dict(), restored_optimizer.state_dict()
        )

        # Prove that the restored optimizer can continue the same trajectory,
        # rather than only proving that its serialized fields can be read.
        model.decoder.train()
        restored_decoder.train()
        optimizer.zero_grad(set_to_none=True)
        restored_optimizer.zero_grad(set_to_none=True)
        continued_raw = model.decoder(latent)
        restored_continued_raw = restored_decoder(latent)
        continued_loss = F.smooth_l1_loss(
            continued_raw["force"], torch.zeros_like(continued_raw["force"])
        ) + F.cross_entropy(continued_raw["slip"], slip_target)
        restored_continued_loss = F.smooth_l1_loss(
            restored_continued_raw["force"],
            torch.zeros_like(restored_continued_raw["force"]),
        ) + F.cross_entropy(restored_continued_raw["slip"], slip_target)
        continued_loss.backward()
        restored_continued_loss.backward()
        gradient_states = (
            {name: parameter.grad for name, parameter in model.decoder.named_parameters()},
            {name: parameter.grad for name, parameter in restored_decoder.named_parameters()},
        )
        gradients_equal = all(
            left.grad is not None
            and right.grad is not None
            and torch.equal(left.grad, right.grad)
            for left, right in zip(
                model.decoder.parameters(), restored_decoder.parameters()
            )
        )
        optimizer.step()
        restored_optimizer.step()
        gradient_difference = tensor_pairs_difference(
            gradient_states[0], gradient_states[1], "decoder_gradients"
        )
        decoder_difference = tensor_pairs_difference(
            model.decoder.state_dict(), restored_decoder.state_dict(), "decoder_state"
        )
        optimizer_difference = tensor_pairs_difference(
            optimizer.state_dict(), restored_optimizer.state_dict(), "optimizer_state"
        )
        decoder_equal_after_continuation = nested_equal(
            model.decoder.state_dict(), restored_decoder.state_dict()
        )
        optimizer_equal_after_continuation = nested_equal(
            optimizer.state_dict(), restored_optimizer.state_dict()
        )
        model.decoder.eval()
        restored_decoder.eval()
        with torch.inference_mode():
            continued_output = flattened_outputs(model.decoder(latent))
            restored_continued_output = flattened_outputs(restored_decoder(latent))
        continuation_output_equal = torch.equal(
            continued_output, restored_continued_output
        )
        continuation_output_difference = float(
            (continued_output - restored_continued_output).abs().max()
        )
        continuation_equal = bool(
            torch.equal(continued_loss.detach(), restored_continued_loss.detach())
            and gradients_equal
            and decoder_equal_after_continuation
            and optimizer_equal_after_continuation
            and continuation_output_equal
        )
        checkpoint_round_trip = {
            "status": (
                "pass"
                if output_equal
                and optimizer_equal_after_load
                and continuation_equal
                else "fail"
            ),
            "temporary_checkpoint_sha256": temporary_hash,
            "temporary_checkpoint_removed_after_verification": True,
            "load_map_location": "cpu_then_state_dict_device_migration",
            "strict_decoder_state_load": True,
            "output_bitwise_equal": output_equal,
            "optimizer_state_equal": optimizer_equal_after_load,
            "optimizer_state_equal_after_load": optimizer_equal_after_load,
            "identical_continuation_step": {
                "status": "pass" if continuation_equal else "fail",
                "synthetic_target": "zero normalized force plus slip classes [0, 1]",
                "loss_bitwise_equal": torch.equal(
                    continued_loss.detach(), restored_continued_loss.detach()
                ),
                "decoder_gradients_bitwise_equal": gradients_equal,
                "decoder_gradient_difference": gradient_difference,
                "decoder_parameters_bitwise_equal": decoder_equal_after_continuation,
                "decoder_parameter_difference": decoder_difference,
                "optimizer_state_bitwise_equal": optimizer_equal_after_continuation,
                "optimizer_state_difference": optimizer_difference,
                "output_bitwise_equal": continuation_output_equal,
                "output_max_abs_difference": continuation_output_difference,
            },
        }

    if args.check_normalflow:
        nf_result = normalflow_train_smoke(args.manifest, model, device)
    else:
        nf_result = {
            "status": "not_run",
            "reason": "pass --check-normalflow to read one fixed NormalFlow train episode",
        }

    checkpoint_hash_after = sha256(args.checkpoint)
    update_ok = bool(
        torch.isfinite(loss)
        and gradients_finite
        and encoder_gradient_tensors == 0
        and encoder_hash_before == encoder_hash_after
        and changed
        and torch.isfinite(updated).all()
    )
    statuses = [
        update_ok,
        checkpoint_round_trip["status"] == "pass",
        reference_comparison["status"] != "fail",
        nf_result["status"] != "fail",
        checkpoint_hash_before == checkpoint_hash_after,
        initial_repeat_equal,
    ]
    report = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "pass" if all(statuses) else "fail",
        "scope": (
            "environment migration smoke only; one synthetic optimizer step on the legacy "
            "decoder; no accuracy, convergence, calibration, or new-method claim"
        ),
        "host": {"hostname": platform.node(), "platform": platform.platform()},
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "packages": package_versions(),
        },
        "device": {
            "selected": str(device),
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "compiled_arches": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
            **(
                {
                    "name": torch.cuda.get_device_name(device),
                    "capability": list(torch.cuda.get_device_capability(device)),
                }
                if device.type == "cuda"
                else {}
            ),
        },
        "determinism": {
            "algorithms": "torch.use_deterministic_algorithms(True)",
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "tf32": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cuda_sdpa": "math_only" if device.type == "cuda" else "not_applicable",
            "repeat_rule": tolerance["repeat_rule"],
        },
        "tolerances": {
            "path": str(args.tolerances),
            "sha256": sha256(args.tolerances),
            **tolerance,
        },
        "artifacts": {
            "manifest": str(args.manifest),
            "manifest_sha256": sha256(args.manifest),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256_before": checkpoint_hash_before,
            "checkpoint_sha256_after": checkpoint_hash_after,
            "checkpoint_unchanged": checkpoint_hash_before == checkpoint_hash_after,
            "checkpoint_epoch": payload.get("epoch"),
            "checkpoint_format": payload.get("format"),
            "adapter": str(Path(adapters.__file__).resolve()),
            "adapter_sha256": sha256(Path(adapters.__file__).resolve()),
            "phase2_loader": str(Path(p2.__file__).resolve()),
            "phase2_loader_sha256": sha256(Path(p2.__file__).resolve()),
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256(Path(__file__).resolve()),
        },
        "htt_sample": sample,
        "initial_output": initial.tolist(),
        "initial_repeat_output": initial_repeat.tolist(),
        "initial_repeat": {
            "rule": tolerance["repeat_rule"],
            "repeat_equal": initial_repeat_equal,
            "status": "pass" if initial_repeat_equal else "fail",
        },
        "updated_output": updated.tolist(),
        "cpu_reference_comparison": reference_comparison,
        "one_step_update": {
            "status": "pass" if update_ok else "fail",
            "optimizer": "torch.optim.Adam",
            "learning_rate": args.learning_rate,
            "target_scope": "synthetic plumbing target; zero force plus slip classes [0, 1]",
            "loss": float(loss.detach().cpu()),
            "force_loss": float(force_loss.detach().cpu()),
            "slip_loss": float(slip_loss.detach().cpu()),
            "all_decoder_gradients_finite": gradients_finite,
            "decoder_gradient_tensors": len(decoder_grads),
            "encoder_gradient_tensors": encoder_gradient_tensors,
            "encoder_state_sha256_before": encoder_hash_before,
            "encoder_state_sha256_after": encoder_hash_after,
            "encoder_state_unchanged": encoder_hash_before == encoder_hash_after,
            "changed_parameter_tensors": len(changed),
            "changed_parameter_names": changed,
            "max_abs_parameter_change": max_change,
            "updated_outputs_finite": bool(torch.isfinite(updated).all()),
        },
        "temporary_checkpoint_round_trip": checkpoint_round_trip,
        "normalflow_train_forward": nf_result,
        "limits": [
            "Synthetic targets do not validate learning quality or dataset semantics.",
            "The full Phase-2 CLI, real GSmini training DataLoader workers, logging, validation, and scheduler are not exercised.",
            "NormalFlow checks adapter compatibility and finite forward only; it has no slip labels here.",
            "A CPU pass cannot establish RTX 6000D execution; CUDA requires a real run and a fixed CPU reference.",
        ],
    }
    result_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
