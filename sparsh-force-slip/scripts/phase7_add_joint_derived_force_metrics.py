#!/usr/bin/env python3
"""Fill derived Fn/Ft/Fmag metrics for Phase7 Step4 joint lightweight row."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import phase5_friction_stability as p5
import phase6_paper_supplements as p6

REPO = Path("/home/zjy/document/tactile-grasp")
WORKSPACE = REPO / "sparsh-force-slip"
STEP4 = WORKSPACE / "reports/phase7/step4_force_axis_decomposition"
JOINT_REPORT = WORKSPACE / "reports/phase6/phase6_3_frozen_vs_joint_ablation/phase6_3_joint_lightweight_seed42_20260518_0154_report.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def derived(f: np.ndarray) -> dict[str, np.ndarray]:
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    return {
        "Fn": np.abs(fz),
        "Ft": np.sqrt(fx * fx + fy * fy),
        "Fmag": np.sqrt(fx * fx + fy * fy + fz * fz),
    }


def derived_rmse(force_gt: np.ndarray, force_pred: np.ndarray) -> dict[str, float]:
    gt = derived(force_gt)
    pred = derived(force_pred)
    return {k: float(np.sqrt(np.mean((pred[k] - gt[k]) ** 2))) for k in ["Fn", "Ft", "Fmag"]}


def geometry(dataset: str) -> str:
    if dataset.startswith("flat"):
        return "flat"
    if dataset.startswith("sharp"):
        return "sharp"
    if dataset.startswith("sphere"):
        return "sphere"
    return "unknown"


def fmt(x: Any) -> str:
    if x is None:
        return "n/a"
    return f"{float(x):.4f}"


def load_joint_predictions() -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    rep = read_json(JOINT_REPORT)
    ck = torch.load(rep["checkpoint"], map_location="cpu", weights_only=False)
    payload = p6.load_feature(p6.PHASE6_RUN_ROOT, rep["feature_run_id"], "val")
    indices = p5.all_indices(payload)
    ds = p6.JointDataset(payload, indices, tuple(int(h) for h in ck["horizons"]))
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = p6.JointForceSlipFutureHead(
        ck["input_dim"], ck["hidden_dim"], len(ck["horizons"]), ck["dropout"],
        ck["thresholds"]["q_tau"], ck["thresholds"]["q_alpha"], ck["thresholds"]["ratio_p99"]
    ).to(dev)
    model.load_state_dict(ck["model_state"])
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            preds.append(model(batch["x"].to(dev))["force"].cpu().numpy())
    force_pred = np.concatenate(preds, axis=0)
    force_gt = payload["force_gt_n"].numpy()[indices]
    meta = [payload["metadata"][int(i)] for i in indices]
    return force_gt, force_pred, meta


def recompute_distribution(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return [r for r in rows if r["condition"] != "joint_lightweight_force_slip_future"]


def main() -> None:
    force_gt, force_pred, meta = load_joint_predictions()
    drv = derived_rmse(force_gt, force_pred)
    step4_json = STEP4 / "step4_force_axis_decomposition.json"
    data = read_json(step4_json)
    for row in data["rows"]:
        if row["condition"] == "joint_lightweight_force_slip_future":
            row["Fn_RMSE"] = drv["Fn"]
            row["Ft_RMSE"] = drv["Ft"]
            row["Fmag_RMSE"] = drv["Fmag"]
            row["derived_metric_source"] = str(JOINT_REPORT)
    # Per-dataset distribution rows for joint head.
    csv_path = STEP4 / "step4_force_error_distribution_by_dataset.csv"
    dist_rows = recompute_distribution(csv_path)
    datasets = sorted({m["dataset"] for m in meta})
    for ds in datasets:
        idx = np.array([i for i, m in enumerate(meta) if m["dataset"] == ds], dtype=int)
        err = force_pred[idx] - force_gt[idx]
        rmse_xyz = np.sqrt(np.mean(err * err, axis=0))
        d_rmse = derived_rmse(force_gt[idx], force_pred[idx])
        dist_rows.append({
            "condition": "joint_lightweight_force_slip_future",
            "dataset": ds,
            "geometry": geometry(ds),
            "force_rmse_mean_N": float(np.mean(rmse_xyz)),
            "Fx_RMSE": float(rmse_xyz[0]),
            "Fy_RMSE": float(rmse_xyz[1]),
            "Fz_RMSE": float(rmse_xyz[2]),
            "Fn_RMSE": d_rmse["Fn"],
            "Ft_RMSE": d_rmse["Ft"],
            "Fmag_RMSE": d_rmse["Fmag"],
            "slip_f1": "",
        })
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["condition", "dataset", "geometry", "force_rmse_mean_N", "Fx_RMSE", "Fy_RMSE", "Fz_RMSE", "Fn_RMSE", "Ft_RMSE", "Fmag_RMSE", "slip_f1"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dist_rows)
    # Re-render Step4 table.
    lines = ["# Phase7 Step4 Force Per-Axis / Physical Decomposition", "", "| condition | Fx | Fy | Fz | Fn | Ft | Fmag | mean | slip F1 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in data["rows"]:
        lines.append(f"| {r['condition']} | {fmt(r['Fx_RMSE'])} | {fmt(r['Fy_RMSE'])} | {fmt(r['Fz_RMSE'])} | {fmt(r['Fn_RMSE'])} | {fmt(r['Ft_RMSE'])} | {fmt(r['Fmag_RMSE'])} | {fmt(r['force_RMSE_mean'])} | {fmt(r['slip_F1'])} |")
    data.setdefault("error_distribution", {})["includes_joint_lightweight"] = True
    data["error_distribution"]["csv"] = str(csv_path)
    write_json(step4_json, data)
    # Replot distribution from refreshed CSV.
    conditions = []
    values = []
    for cond in [r["condition"] for r in data["rows"]]:
        vals = [float(r["force_rmse_mean_N"]) for r in dist_rows if r["condition"] == cond and str(r["force_rmse_mean_N"]) not in ("", "None")]
        if vals:
            conditions.append(cond)
            values.append(vals)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.boxplot(values, labels=conditions, showmeans=True)
    ax.set_ylabel("per-dataset mean force RMSE (N)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    box_path = STEP4 / "step4_force_error_distribution_by_dataset.png"
    fig.savefig(box_path, dpi=160)
    plt.close(fig)
    geoms = ["flat", "sharp", "sphere"]
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(geoms))
    width = 0.12
    for i, cond in enumerate(conditions):
        means = []
        for g in geoms:
            vals = [float(r["force_rmse_mean_N"]) for r in dist_rows if r["condition"] == cond and r["geometry"] == g and str(r["force_rmse_mean_N"]) not in ("", "None")]
            means.append(float(np.mean(vals)) if vals else 0.0)
        ax.bar(x + (i - (len(conditions)-1)/2) * width, means, width, label=cond)
    ax.set_xticks(x, geoms)
    ax.set_ylabel("mean force RMSE (N)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    geom_path = STEP4 / "step4_force_error_by_contact_geometry.png"
    fig.savefig(geom_path, dpi=160)
    plt.close(fig)
    lines += [
        "",
        "## Error distribution across validation datasets",
        "",
        f"- CSV: `{csv_path}`",
        f"- Boxplot: `{box_path}`",
        f"- Contact-geometry grouped plot: `{geom_path}`",
        "- Joint lightweight Fn/Ft/Fmag metrics were recomputed from its saved checkpoint and Phase6 cached validation features.",
    ]
    (STEP4 / "step4_force_axis_decomposition.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(STEP4 / "step4_force_axis_decomposition.md")


if __name__ == "__main__":
    main()
