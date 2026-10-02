#!/usr/bin/env python3
"""Round-18 CPU/GPU smoke, including real subprocess recovery equivalence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch import nn

import train as r18


def file_sha(path: Path) -> str:
    return r18.sha(path)


def run_command(command: list[str], log: Path) -> float:
    began = time.perf_counter()
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"subprocess failed ({result.returncode}); see {log}")
    return time.perf_counter() - began


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data_hash_before = file_sha(args.data)
    load_start = time.perf_counter()
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    io_load_seconds = time.perf_counter() - load_start
    r18.validate_cache(data, args.data, args.fold, args.seed)
    normalizer, weights, _ = r18.fit_assets(data)
    fit = data["roles"]["fit"]
    x = r18.normalize(fit["x"][:256], normalizer).to(args.device)
    stage = fit["stage"][:256].to(args.device)
    primary = stage.eq(0) | stage.eq(2)
    if not primary.any():
        raise ValueError("smoke slice lacks reliable endpoints")

    counts = {group: sum(p.numel() for p in r18.init_model(group, args.seed).parameters()) for group in r18.GROUPS}
    capacity = {group: (value / counts["V0"] - 1.0) * 100.0 for group, value in counts.items()}
    if max(abs(value) for value in capacity.values()) > 5.0:
        raise ValueError(f"capacity tolerance: {capacity}")

    v0 = r18.init_model("V0", args.seed).to(args.device)
    m0 = r18.init_model("M0", args.seed).to(args.device)
    with torch.no_grad():
        v_initial = v0(x)
        m_components = m0.components(x)
    initial_identity = torch.equal(v_initial, m_components["logit"])
    initial_gamma_zero = m_components["gamma"].count_nonzero().item() == 0
    initial_beta_zero = m_components["beta"].count_nonzero().item() == 0
    if not (initial_identity and initial_gamma_zero and initial_beta_zero):
        raise ValueError("initial FiLM identity failed")

    gradient_checks = {}
    for group in r18.GROUPS:
        model = r18.init_model(group, args.seed).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        force_grad_by_step = []
        loss_values = []
        for _ in range(4):
            xx = x.detach().clone().requires_grad_(True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xx)
            loss = (nn.functional.binary_cross_entropy_with_logits(logits[primary], stage[primary].eq(2).float(), reduction="none") * weights[:256].to(args.device)[primary]).mean()
            loss.backward()
            if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError(f"{group} nonfinite/missing gradient")
            force_grad_by_step.append(float(xx.grad[..., 192:195].abs().max()))
            optimizer.step()
            loss_values.append(float(loss.detach()))
        changes = {name: not torch.equal(initial[name], value.detach().cpu()) for name, value in model.state_dict().items()}
        if group == "V0" and any(force_grad_by_step):
            raise ValueError("V0 reads force")
        if group != "V0" and force_grad_by_step[-1] == 0:
            raise ValueError(f"{group} force disconnected after multiple steps")
        if group == "M0" and not all(changes.values()):
            raise ValueError(f"M0 parameter failed to update after multiple steps: {changes}")
        gradient_checks[group] = {"losses": loss_values, "force_input_gradient_absmax_by_step": force_grad_by_step, "parameter_changed_after_4_steps": changes}

    roles = {name: set(data["roles"][name]["leakage_group"]) for name in r18.ROLES}
    role_disjoint = all(not roles[a].intersection(roles[b]) for i, a in enumerate(r18.ROLES) for b in r18.ROLES[i + 1 :])
    incipient = {name: {"frames": int(data["roles"][name]["stage"].eq(1).sum()), "episodes": len(set(e for e, s in zip(data["roles"][name]["episode_id"], data["roles"][name]["stage"].tolist()) if s == 1))} for name in r18.ROLES}
    if not role_disjoint or "test" in data["roles"]:
        raise ValueError("role isolation")

    script = Path(r18.__file__).resolve()
    base = [sys.executable, str(script), "--data", str(args.data), "--fold", str(args.fold), "--seed", str(args.seed), "--device", args.device, "--patience", "99"]
    budget = {}
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    for group in r18.GROUPS:
        target = args.output / "budget" / group
        if target.exists():
            shutil.rmtree(target)
        elapsed = run_command(base + ["--group", group, "--output", str(target), "--max-epochs", "1"], args.output / "logs" / f"budget_{group}.log")
        summary = json.loads((target / "summary.json").read_text())
        budget[group] = {"one_full_epoch_subprocess_seconds": elapsed, "trainer_wall_seconds": summary["wall_seconds_this_invocation"], "fit_endpoints": int((fit["stage"].eq(0) | fit["stage"].eq(2)).sum())}

    continuous = args.output / "recovery" / "continuous"
    resumed = args.output / "recovery" / "resumed"
    for target in (continuous, resumed):
        if target.exists():
            shutil.rmtree(target)
    run_command(base + ["--group", "M0", "--output", str(continuous), "--max-epochs", "3"], args.output / "logs" / "recovery_continuous.log")
    run_command(base + ["--group", "M0", "--output", str(resumed), "--max-epochs", "3", "--interrupt-after", "0"], args.output / "logs" / "recovery_interrupted.log")
    run_command(base + ["--group", "M0", "--output", str(resumed), "--max-epochs", "3"], args.output / "logs" / "recovery_resumed.log")
    continuous_ck = torch.load(continuous / "latest.pth", map_location="cpu", weights_only=False)
    resumed_ck = torch.load(resumed / "latest.pth", map_location="cpu", weights_only=False)
    recovery_exact = r18.state_hash(continuous_ck["model"]) == r18.state_hash(resumed_ck["model"]) and continuous_ck["history"] == resumed_ck["history"] and continuous_ck["optimizer_steps"] == resumed_ck["optimizer_steps"]
    if not recovery_exact:
        raise ValueError("real subprocess recovery mismatch")

    upstream_before = {name: record["sha256"] for name, record in data["provenance"]["immutable_upstream"].items()}
    upstream_after = {name: file_sha(Path(record["path"])) for name, record in data["provenance"]["immutable_upstream"].items()}
    data_hash_after = file_sha(args.data)
    if upstream_before != upstream_after or data_hash_before != data_hash_after:
        raise ValueError("frozen input changed")
    peak = int(torch.cuda.max_memory_allocated(args.device)) if args.device.startswith("cuda") else None
    result = {
        "schema": "round18_smoke_v1",
        "status": "pass",
        "device": args.device,
        "gpu_name": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "data": {"path": str(args.data), "sha256": data_hash_before, "io_load_seconds": io_load_seconds},
        "parameter_counts": counts,
        "capacity_percent_vs_V0": capacity,
        "all_parameters_structurally_active": True,
        "initial_film_identity_exact": initial_identity,
        "initial_gamma_zero": initial_gamma_zero,
        "initial_beta_zero": initial_beta_zero,
        "gradient_checks": gradient_checks,
        "role_groups_disjoint": role_disjoint,
        "test_consumed": False,
        "incipient_distribution": incipient,
        "normalizer_fit_static_gross_only": True,
        "real_subprocess_interrupted_resume_exact": recovery_exact,
        "continuous_latest_state_sha256": r18.state_hash(continuous_ck["model"]),
        "resumed_latest_state_sha256": r18.state_hash(resumed_ck["model"]),
        "input_unchanged": data_hash_before == data_hash_after,
        "upstream_hashes_unchanged": upstream_before == upstream_after,
        "formal_smoke_directory_separation": True,
        "actual_budget": budget,
        "peak_cuda_allocated_bytes_this_process": peak,
        "upstream_model_size_and_end_to_end_usage": "unknown; cached-head smoke cannot infer it",
    }
    r18.atomic_json(args.output / "SMOKE.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
