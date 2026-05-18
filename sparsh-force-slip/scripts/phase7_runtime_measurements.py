#!/usr/bin/env python3
"""Measure Phase7 runtime/model-size evidence on cached/evaluation paths."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import phase2_b_multitask as p2
import phase3_2_world_model as wm

REPO = Path("/home/zjy/document/tactile-grasp")
WORKSPACE = REPO / "sparsh-force-slip"
OUT = WORKSPACE / "reports/phase7/step7_runtime_model_size"
FUTURE_CKPT = Path("/vla1/zjy/sparsh_runs/force_slip_phase6/phase6_1_friction_features_20260518_0130/heads/phase6_1_full_plus_q_seed42_20260518_0130/checkpoints/best.pth")
MODEL_CKPTS = {
    "naive_shared_mae": Path("/vla1/zjy/sparsh_runs/force_slip_phase2/phase2_b_gsmini_20260513_163448/mae_shared_multitask/checkpoints/best_f1.pth"),
    "decoupled_mae": Path("/vla1/zjy/sparsh_runs/force_slip_phase2/phase3_1_decoupled_gsmini_20260516_154730/mae_decoupled_multitask/checkpoints/epoch-0030.pth"),
}


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def fmt(x: Any, digits: int = 6) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def gpu_snapshot() -> dict[str, Any]:
    cmd = ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader"]
    try:
        rows = subprocess.check_output(cmd, text=True).strip().splitlines()
    except Exception as exc:
        rows = [f"ERROR: {exc}"]
    return {"captured_at": now(), "rows": rows}


def count_params(path: str | Path, exclude_encoder: bool = False) -> int:
    d = torch.load(path, map_location="cpu", weights_only=False)
    sd = d.get("state_dict") or d.get("model_state") or d.get("model") or d
    total = 0
    for k, v in sd.items():
        if exclude_encoder and str(k).startswith("encoder."):
            continue
        if hasattr(v, "numel"):
            total += int(v.numel())
    return total


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_block(fn, device: torch.device, iterations: int = 20, warmup: int = 5) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        sync(device)
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        sync(device)
    return (time.perf_counter() - start) / iterations * 1000.0


def measure_current_model(name: str, ckpt: Path, device: torch.device) -> dict[str, Any]:
    model, payload = p2.load_b_checkpoint(ckpt, device)
    model.eval()
    loader = p2.make_loader(["flat_batch_1_val"], 0, 64, 0, shuffle=False, drop_last=False, encoder="mae")
    batch = next(iter(loader))
    x = batch["image"].to(device)
    core = p2.unwrap_model(model)
    with torch.no_grad():
        z = core.encoder(x)
    encoder_ms = time_block(lambda: core.encoder(x), device)
    decoder_ms = time_block(lambda: core.decoder(z.detach()), device)
    total_ms = time_block(lambda: core(x), device)
    params_total = sum(p.numel() for p in core.parameters())
    params_trainable = sum(p.numel() for p in core.parameters() if p.requires_grad)
    return {
        "model": name,
        "measurement_type": "measured_encoder_decoder_total",
        "checkpoint": str(ckpt),
        "device": str(device),
        "batch_size": int(x.shape[0]),
        "iterations": 20,
        "encoder_feature_extraction_ms_per_batch": encoder_ms,
        "force_slip_heads_ms_per_batch": decoder_ms,
        "total_ms_per_batch": total_ms,
        "encoder_feature_extraction_ms_per_sample": encoder_ms / int(x.shape[0]),
        "force_slip_heads_ms_per_sample": decoder_ms / int(x.shape[0]),
        "total_ms_per_sample": total_ms / int(x.shape[0]),
        "trainable_params": int(params_trainable),
        "total_params_loaded": int(params_total),
        "checkpoint_epoch": payload.get("epoch"),
    }


def measure_future_head(device: torch.device) -> dict[str, Any]:
    ck = torch.load(FUTURE_CKPT, map_location="cpu", weights_only=False)
    model = wm.MLP(ck["input_dim"], ck["hidden_dim"], len(ck["horizons"]), ck["dropout"]).to(device).eval()
    model.load_state_dict(ck["model_state"])
    x = torch.randn(512, ck["input_dim"], device=device)
    ms = time_block(lambda: model(x), device, iterations=100, warmup=10)
    return {
        "model": "future_head_full_plus_q",
        "measurement_type": "measured_cached_feature_head",
        "checkpoint": str(FUTURE_CKPT),
        "device": str(device),
        "batch_size": 512,
        "iterations": 100,
        "ms_per_batch": ms,
        "ms_per_sample": ms / 512,
        "trainable_params": count_params(FUTURE_CKPT),
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase7 Step7 Runtime / Model Size / Inference Cost",
        "",
        "## Parameter counts",
        "",
        "| model | parameters |",
        "|---|---:|",
    ]
    for r in payload["parameter_rows"]:
        lines.append(f"| {r['model']} | {r['params']} |")
    lines += [
        "",
        "## Measured current force/slip inference latency",
        "",
        "| model | device | batch | encoder feature extraction ms/batch | force/slip heads ms/batch | total ms/batch | total ms/sample | trainable params |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in payload["current_model_latency"]:
        lines.append(
            f"| {r['model']} | {r['device']} | {r['batch_size']} | {fmt(r['encoder_feature_extraction_ms_per_batch'])} | {fmt(r['force_slip_heads_ms_per_batch'])} | {fmt(r['total_ms_per_batch'])} | {fmt(r['total_ms_per_sample'])} | {r['trainable_params']} |"
        )
    fh = payload["future_head_latency"]
    lines += [
        "",
        "## Future head cached-feature latency",
        f"- device: `{fh['device']}`",
        f"- batch_size: `{fh['batch_size']}`",
        f"- ms_per_batch: `{fmt(fh['ms_per_batch'])}`",
        f"- ms_per_sample: `{fmt(fh['ms_per_sample'])}`",
        f"- trainable_params: `{fh['trainable_params']}`",
        "",
        "## Separate probing deployment estimate",
        "",
    ]
    est = payload["separate_probing_estimate"]
    lines += [
        "Separate force/slip probing checkpoints are independent downstream tasks. For a naive real-time deployment without encoder-feature reuse, the estimate is two frozen-encoder passes plus head cost; with feature reuse, it reduces to one encoder pass plus two lightweight heads.",
        "",
        f"- two-pass total estimate: `{fmt(est['two_encoder_pass_total_ms_per_batch'])}` ms/batch64",
        f"- shared-encoder lower-bound estimate: `{fmt(est['shared_encoder_total_ms_per_batch'])}` ms/batch64",
        "",
        "## GPU memory snapshot",
        "",
    ]
    for row in payload["gpu_memory_snapshot"]["rows"]:
        lines.append(f"- `{row}`")
    lines += ["", "## Notes"] + [f"- {n}" for n in payload["notes"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parameter_rows = [
        {"model": "separate_force_head", "params": count_params("/vla1/zjy/sparsh_runs/experiments/2026.05.12_04-39_phase1_gsmini_20260512_043331_mae_force_gsmini_20260512_043652/checkpoints/epoch-0051.pth")},
        {"model": "separate_slip_head", "params": count_params("/vla1/zjy/sparsh_runs/experiments/2026.05.13_01-21_phase1_gsmini_20260512_043331_mae_slip_allsource_diag_gsmini_20260513_012000/checkpoints/epoch-0051.pth")},
        {"model": "naive_shared_total", "params": count_params(MODEL_CKPTS["naive_shared_mae"])},
        {"model": "naive_shared_downstream_excl_encoder", "params": count_params(MODEL_CKPTS["naive_shared_mae"], True)},
        {"model": "decoupled_total", "params": count_params(MODEL_CKPTS["decoupled_mae"])},
        {"model": "decoupled_downstream_excl_encoder", "params": count_params(MODEL_CKPTS["decoupled_mae"], True)},
        {"model": "future_head_full_plus_q", "params": count_params(FUTURE_CKPT)},
    ]
    latency_rows = [measure_current_model(name, ckpt, device) for name, ckpt in MODEL_CKPTS.items()]
    future = measure_future_head(device)
    base = next(r for r in latency_rows if r["model"] == "naive_shared_mae")
    separate_estimate = {
        "basis": "naive_shared_mae measured encoder/head batch64 latency",
        "two_encoder_pass_total_ms_per_batch": 2 * base["encoder_feature_extraction_ms_per_batch"] + base["force_slip_heads_ms_per_batch"],
        "shared_encoder_total_ms_per_batch": base["encoder_feature_extraction_ms_per_batch"] + 2 * base["force_slip_heads_ms_per_batch"],
    }
    payload = {
        "generated_at": now(),
        "phase": "phase7_step7_runtime_model_size",
        "raw_data_modified": False,
        "parameter_rows": parameter_rows,
        "current_model_latency": latency_rows,
        "future_head_latency": future,
        "separate_probing_estimate": separate_estimate,
        "gpu_memory_snapshot": gpu_snapshot(),
        "notes": [
            "Encoder feature extraction time is measured as the frozen Sparsh MAE encoder forward on a validation batch; no training data were modified.",
            "Force/slip head time is measured by running the decoder on the cached encoder tokens from the same batch.",
            "Separate probing latency is reported as a deployment estimate because the force and slip probes are independent checkpoints rather than one combined callable model.",
            "Measurements were taken on a shared server, so absolute latency can vary with concurrent jobs; relative head/encoder scale is the intended evidence.",
        ],
    }
    write_json(OUT / "step7_runtime_model_size.json", payload)
    (OUT / "step7_runtime_model_size.md").write_text(render(payload), encoding="utf-8")
    print(OUT / "step7_runtime_model_size.md")


if __name__ == "__main__":
    main()
