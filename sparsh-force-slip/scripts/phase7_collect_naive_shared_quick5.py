#!/usr/bin/env python3
"""Collect Phase7 quick 5-epoch naive-shared multi-seed metrics.

This supplemental stability check reads checkpoint-embedded validation metrics and
updates the Phase7 Step5 report without modifying any raw tactile dataset.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path("/home/zjy/document/tactile-grasp")
WORKSPACE = REPO / "sparsh-force-slip"
OUT = WORKSPACE / "reports/phase7/step5_multiseed_stability"

QUICK_CHECKPOINTS = [
    {
        "seed": 42,
        "run_id": "phase2_b_gsmini_20260513_163448",
        "checkpoint": "/vla1/zjy/sparsh_runs/force_slip_phase2/phase2_b_gsmini_20260513_163448/mae_shared_multitask/checkpoints/epoch-0005.pth",
        "source": "existing full run epoch-0005 checkpoint",
    },
    {
        "seed": 43,
        "run_id": "phase7_naive_shared_quick_seed43_20260519",
        "checkpoint": "/vla1/zjy/sparsh_runs/force_slip_phase2/phase7_naive_shared_quick_seed43_20260519/mae_shared_multitask/checkpoints/epoch-0005.pth",
        "source": "Phase7 quick supplemental run",
    },
    {
        "seed": 44,
        "run_id": "phase7_naive_shared_quick_seed44_20260519",
        "checkpoint": "/vla1/zjy/sparsh_runs/force_slip_phase2/phase7_naive_shared_quick_seed44_20260519/mae_shared_multitask/checkpoints/epoch-0005.pth",
        "source": "Phase7 quick supplemental run",
    },
]


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def fmt(x: Any, digits: int = 4) -> str:
    if x is None:
        return "n/a"
    try:
        v = float(x)
    except Exception:
        return str(x)
    if math.isnan(v) or math.isinf(v):
        return "n/a"
    return f"{v:.{digits}f}"


def stat(vals: list[float]) -> dict[str, Any]:
    arr = np.asarray(vals, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "n": int(arr.size)}


def best_worst(rows: list[dict[str, Any]], key: str, lower: bool) -> dict[str, Any]:
    valid = [r for r in rows if r.get(key) is not None]
    best = min(valid, key=lambda r: r[key]) if lower else max(valid, key=lambda r: r[key])
    worst = max(valid, key=lambda r: r[key]) if lower else min(valid, key=lambda r: r[key])
    return {
        "best_seed": best["seed"],
        "best_value": float(best[key]),
        "worst_seed": worst["seed"],
        "worst_value": float(worst[key]),
        "lower_is_better": lower,
    }


def load_row(spec: dict[str, Any]) -> dict[str, Any]:
    ckpt = Path(spec["checkpoint"])
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    metrics = payload.get("metrics") or {}
    cfg = payload.get("train_config") or {}
    if not metrics:
        raise RuntimeError(f"checkpoint has no validation metrics: {ckpt}")
    return {
        "seed": int(spec["seed"]),
        "run_id": spec["run_id"],
        "source": spec["source"],
        "checkpoint": str(ckpt),
        "epoch": int(payload.get("epoch", cfg.get("max_epochs", 5))),
        "max_epochs": int(cfg.get("max_epochs", 5)),
        "encoder": cfg.get("encoder", "mae"),
        "decoder_variant": cfg.get("decoder_variant", "shared"),
        "force_rmse_mean_N": float(metrics.get("force_rmse_mean_N")),
        "slip_f1": float(metrics.get("slip_f1")),
        "slip_accuracy": float(metrics.get("slip_accuracy")),
        "slip_balanced_accuracy": metrics.get("slip_balanced_accuracy"),
        "force_rmse_xyz_N": metrics.get("force_rmse_xyz_N"),
        "force_derived_rmse_N": metrics.get("force_derived_rmse_N"),
        "global_step": metrics.get("global_step"),
        "raw_metrics": metrics,
    }


def render_main(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase7 Step5 Main Results Multi-Seed Stability",
        "",
        "| condition | source | force RMSE mean±std | slip F1 mean±std | H3 F1 mean±std | H5 F1 mean±std | H5 AUPRC mean±std | note |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in payload["rows"]:
        fr = r.get("force_rmse_mean_N") or {}
        sf = r.get("slip_f1") or {}
        fut = r.get("future") or {}
        lines.append(
            f"| {r['condition']} | {r.get('source','')} | "
            f"{fmt(fr.get('mean'))}±{fmt(fr.get('std'))} (n={fr.get('n','')}) | "
            f"{fmt(sf.get('mean'))}±{fmt(sf.get('std'))} (n={sf.get('n','')}) | "
            f"{fmt((fut.get('H3_f1') or {}).get('mean'))}±{fmt((fut.get('H3_f1') or {}).get('std'))} | "
            f"{fmt((fut.get('H5_f1') or {}).get('mean'))}±{fmt((fut.get('H5_f1') or {}).get('std'))} | "
            f"{fmt((fut.get('H5_auprc') or {}).get('mean'))}±{fmt((fut.get('H5_auprc') or {}).get('std'))} | "
            f"{r.get('gap') or r.get('note','')} |"
        )
    lines += [
        "",
        "## Naive shared quick 5-epoch supplemental seeds",
        "",
        "This supplemental row closes the naive-shared 3-seed stability gap with matched 5-epoch checkpoints. It is useful for seed sensitivity, but the 5-epoch statistics are not directly comparable to the full 51-epoch single-seed reference row.",
        "",
        "| seed | run_id | epoch | force RMSE | slip F1 | slip accuracy | checkpoint |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for r in payload.get("naive_shared_quick5_details", []):
        lines.append(
            f"| {r['seed']} | {r['run_id']} | {r['epoch']} | {fmt(r['force_rmse_mean_N'])} | {fmt(r['slip_f1'])} | {fmt(r['slip_accuracy'])} | `{r['checkpoint']}` |"
        )
    lines += ["", "## Best / worst seeds", "", "| metric | best seed | best value | worst seed | worst value | lower is better |", "|---|---|---:|---|---:|---|"]
    for metric, bw in payload.get("best_worst_seed_summary", {}).items():
        lines.append(
            f"| {metric} | {bw['best_seed']} | {fmt(bw['best_value'])} | {bw['worst_seed']} | {fmt(bw['worst_value'])} | {bw['lower_is_better']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [load_row(spec) for spec in QUICK_CHECKPOINTS]
    summary = {
        "condition": "naive_shared_multitask_quick5_supplemental",
        "source": "Phase7 matched 5-epoch checkpoints, seeds 42/43/44",
        "force_rmse_mean_N": stat([r["force_rmse_mean_N"] for r in rows]),
        "slip_f1": stat([r["slip_f1"] for r in rows]),
        "slip_accuracy": stat([r["slip_accuracy"] for r in rows]),
        "note": "3-seed quick stability check; not a full 51-epoch replacement for the original seed42 reference.",
    }
    quick_payload = {
        "generated_at": now(),
        "phase": "phase7_step5_naive_shared_quick5_multiseed",
        "raw_data_modified": False,
        "summary": summary,
        "rows": rows,
        "best_worst": {
            "naive_shared_quick5_force_rmse": best_worst(rows, "force_rmse_mean_N", True),
            "naive_shared_quick5_slip_f1": best_worst(rows, "slip_f1", False),
        },
    }
    write_json(OUT / "naive_shared_quick5_multiseed.json", quick_payload)
    quick_md = [
        "# Naive Shared Quick 5-Epoch Multi-Seed Stability",
        "",
        "- raw_data_modified: `False`",
        "- purpose: supplement Phase7 Step5 with 3 seeds for the naive shared decoder.",
        "- caveat: matched 5-epoch checkpoints measure seed sensitivity but are not directly comparable to the full 51-epoch seed42 reference.",
        "",
        "| seed | run_id | epoch | force RMSE | slip F1 | slip accuracy | checkpoint |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        quick_md.append(f"| {r['seed']} | {r['run_id']} | {r['epoch']} | {fmt(r['force_rmse_mean_N'])} | {fmt(r['slip_f1'])} | {fmt(r['slip_accuracy'])} | `{r['checkpoint']}` |")
    quick_md += [
        "",
        "## Summary",
        "",
        f"- force RMSE: {fmt(summary['force_rmse_mean_N']['mean'])}±{fmt(summary['force_rmse_mean_N']['std'])} (n=3)",
        f"- slip F1: {fmt(summary['slip_f1']['mean'])}±{fmt(summary['slip_f1']['std'])} (n=3)",
    ]
    (OUT / "naive_shared_quick5_multiseed.md").write_text("\n".join(quick_md) + "\n", encoding="utf-8")

    main_path = OUT / "step5_multiseed_stability.json"
    main_payload = json.loads(main_path.read_text(encoding="utf-8"))
    main_payload["generated_at"] = now()
    main_payload["naive_shared_quick5_details"] = rows
    main_payload["naive_shared_quick5_source"] = str(OUT / "naive_shared_quick5_multiseed.json")
    main_payload["rows"] = [r for r in main_payload.get("rows", []) if r.get("condition") != summary["condition"]]
    main_payload["rows"].append(summary)
    bw = main_payload.setdefault("best_worst_seed_summary", {})
    bw.update(quick_payload["best_worst"])
    write_json(main_path, main_payload)
    (OUT / "step5_multiseed_stability.md").write_text(render_main(main_payload), encoding="utf-8")
    print(OUT / "step5_multiseed_stability.md")


if __name__ == "__main__":
    main()
