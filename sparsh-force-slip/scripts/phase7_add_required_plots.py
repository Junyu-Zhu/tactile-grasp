#!/usr/bin/env python3
"""Add Phase7 required CDF/error-distribution plots from existing reports."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/home/zjy/document/tactile-grasp")
WORKSPACE = REPO / "sparsh-force-slip"
PHASE7 = WORKSPACE / "reports/phase7"
STEP3 = PHASE7 / "step3_early_warning_threshold_sweep"
STEP4 = PHASE7 / "step4_force_axis_decomposition"

STEP4_SOURCES = [
    ("separate_baseline", WORKSPACE / "reports/phase3/phase3_1_20260516_154730/eval_cache/a_mae_allsource_val.json"),
    ("naive_shared_multitask", WORKSPACE / "reports/phase2/phase2_b_gsmini_20260513_163448/eval_cache/b_mae_allsource_val.json"),
    ("partially_shared_lambda_0.25", WORKSPACE / "reports/phase2/phase2_b_ps_lam025_gsmini_20260514_063052/eval_cache/b_mae_partially_shared_allsource_val.json"),
    ("consistency_decoder", WORKSPACE / "reports/phase2/phase2_c_consistency_gsmini_20260514_161845/eval_cache/c_mae_consistency_phase2_c_consistency_gsmini_20260514_161845_allsource_val.json"),
    ("decoupled_multitask", WORKSPACE / "reports/phase3/phase3_1_20260516_154730/eval_cache/decoupled_mae_phase3_1_decoupled_gsmini_20260516_154730_allsource_val.json"),
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def geometry(dataset: str) -> str:
    if dataset.startswith("flat"):
        return "flat"
    if dataset.startswith("sharp"):
        return "sharp"
    if dataset.startswith("sphere"):
        return "sphere"
    return "unknown"


def add_step3_cdf() -> None:
    p = STEP3 / "step3_early_warning_threshold_sweep.json"
    data = read_json(p)
    rows = data.get("rows", [])
    fig_paths = []
    # Combined CDF plot for all thresholds with available lead times.
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = False
    for row in rows:
        leads = np.asarray(row.get("lead_times") or [], dtype=float)
        if leads.size == 0:
            continue
        xs = np.sort(leads)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.step(xs, ys, where="post", label=f"thr={row['threshold']}")
        plotted = True
    if plotted:
        ax.set_xlabel("lead time to slip onset (steps)")
        ax.set_ylabel("CDF")
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = STEP3 / "step3_lead_time_cdf_all_thresholds.png"
        fig.savefig(out, dpi=160)
        fig_paths.append(str(out))
    plt.close(fig)
    existing = set(data.get("figures", []))
    for candidate in [
        STEP3 / "step3_threshold_early_warning_recall.png",
        STEP3 / "step3_threshold_false_alarm_rate.png",
        STEP3 / "step3_threshold_lead_time_to_slip_onset_steps_mean.png",
        STEP3 / "step3_lead_time_hist_threshold_0p5.png",
    ]:
        if candidate.exists():
            existing.add(str(candidate))
    existing.update(fig_paths)
    data["figures"] = sorted(existing)
    write_json(p, data)
    md = STEP3 / "step3_early_warning_threshold_sweep.md"
    text = md.read_text(encoding="utf-8")
    if "## Figures" not in text:
        text = text.rstrip() + "\n\n## Figures\n\n" + "\n".join(f"- `{x}`" for x in data["figures"]) + "\n"
    elif "step3_lead_time_cdf_all_thresholds.png" not in text:
        text = text.rstrip() + f"\n- `{STEP3 / 'step3_lead_time_cdf_all_thresholds.png'}`\n"
    md.write_text(text, encoding="utf-8")


def add_step4_distribution() -> None:
    rows: list[dict[str, Any]] = []
    for condition, path in STEP4_SOURCES:
        if not path.exists():
            continue
        data = read_json(path)
        for dataset, metrics in data.get("per_dataset", {}).items():
            xyz = metrics.get("force_rmse_xyz_N") or [None, None, None]
            drv = metrics.get("force_derived_rmse_N") or {}
            rows.append({
                "condition": condition,
                "dataset": dataset,
                "geometry": geometry(dataset),
                "force_rmse_mean_N": metrics.get("force_rmse_mean_N"),
                "Fx_RMSE": xyz[0],
                "Fy_RMSE": xyz[1],
                "Fz_RMSE": xyz[2],
                "Fn_RMSE": drv.get("Fn"),
                "Ft_RMSE": drv.get("Ft"),
                "Fmag_RMSE": drv.get("Fmag"),
                "slip_f1": metrics.get("slip_f1"),
            })
    csv_path = STEP4 / "step4_force_error_distribution_by_dataset.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    labels = []
    values = []
    for condition, _ in STEP4_SOURCES:
        vals = [float(r["force_rmse_mean_N"]) for r in rows if r["condition"] == condition and r["force_rmse_mean_N"] is not None]
        if vals:
            labels.append(condition)
            values.append(vals)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.boxplot(values, labels=labels, showmeans=True)
    ax.set_ylabel("per-dataset mean force RMSE (N)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    box_path = STEP4 / "step4_force_error_distribution_by_dataset.png"
    fig.savefig(box_path, dpi=160)
    plt.close(fig)

    # Geometry-level grouped bars for optional contact-geometry analysis.
    geoms = ["flat", "sharp", "sphere"]
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(geoms))
    width = 0.15
    for i, (condition, _) in enumerate(STEP4_SOURCES):
        means = []
        for g in geoms:
            vals = [float(r["force_rmse_mean_N"]) for r in rows if r["condition"] == condition and r["geometry"] == g and r["force_rmse_mean_N"] is not None]
            means.append(float(np.mean(vals)) if vals else 0.0)
        ax.bar(x + (i - 2) * width, means, width, label=condition)
    ax.set_xticks(x, geoms)
    ax.set_ylabel("mean force RMSE (N)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    geom_path = STEP4 / "step4_force_error_by_contact_geometry.png"
    fig.savefig(geom_path, dpi=160)
    plt.close(fig)

    p = STEP4 / "step4_force_axis_decomposition.json"
    data = read_json(p)
    data["error_distribution"] = {
        "source": "per_dataset metrics from existing Phase2/3 evaluation caches",
        "csv": str(csv_path),
        "figures": [str(box_path), str(geom_path)],
        "n_rows": len(rows),
    }
    write_json(p, data)
    md = STEP4 / "step4_force_axis_decomposition.md"
    text = md.read_text(encoding="utf-8")
    block = (
        "\n## Error distribution across validation datasets\n\n"
        f"- CSV: `{csv_path}`\n"
        f"- Boxplot: `{box_path}`\n"
        f"- Contact-geometry grouped plot: `{geom_path}`\n"
    )
    if "## Error distribution across validation datasets" not in text:
        text = text.rstrip() + block
    md.write_text(text, encoding="utf-8")


def main() -> None:
    add_step3_cdf()
    add_step4_distribution()
    print(STEP3 / "step3_lead_time_cdf_all_thresholds.png")
    print(STEP4 / "step4_force_error_distribution_by_dataset.png")


if __name__ == "__main__":
    main()
