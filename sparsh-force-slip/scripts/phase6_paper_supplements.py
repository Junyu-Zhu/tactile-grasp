#!/usr/bin/env python3
"""Phase6 paper-completeness supplements for force/slip future stability.

Reads existing Phase4/Phase5 derived caches/checkpoints and writes only Phase6
reports, derived feature copies, lightweight-head checkpoints, and figures.
Raw /vla1/zjy datasets are never modified.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import phase2_b_multitask as p2  # noqa: E402
import phase3_2_world_model as wm  # noqa: E402
import phase5_friction_stability as p5  # noqa: E402
import wandb  # noqa: E402

REPO = Path("/home/zjy/document/tactile-grasp")
WORKSPACE = REPO / "sparsh-force-slip"
PHASE4_RUN_ROOT = Path("/vla1/zjy/sparsh_runs/force_slip_phase4")
PHASE5_RUN_ROOT = Path("/vla1/zjy/sparsh_runs/force_slip_phase5")
PHASE6_RUN_ROOT = Path("/vla1/zjy/sparsh_runs/force_slip_phase6")
PHASE6_REPORT_ROOT = WORKSPACE / "reports/phase6"

PHASE5_FRICTION_RUN = "phase5_2_friction_features_20260518_0035"
PHASE4_DECOUPLED_RUN = "phase4_3_reuse_p3_features_20260517_171500"
PHASE4_SEPARATE_RUN = "phase4_2_separate_features_20260517_171500"
DEFAULT_HORIZONS = (1, 3, 5)
BASE_FULL = list(p5.BASE_FULL)
STATIC_FORCE_SLIP = list(p5.STATIC_FORCE_SLIP)
FULL_PLUS_Q = list(p5.FRICTION_MODES["full_plus_q"])
EPS = 1.0e-6


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=p2.json_default), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "n/a"
    try:
        x = float(v)
    except Exception:
        return str(v)
    if math.isnan(x) or math.isinf(x):
        return "n/a"
    return f"{x:.{digits}f}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_feature(root: Path, run_id: str, split: str) -> dict[str, Any]:
    return torch.load(root / run_id / "features" / f"{split}_features.pt", map_location="cpu", weights_only=False)


def ensure_phase6_feature_copy(src_root: Path, src_run_id: str, dst_run_id: str) -> dict[str, Any]:
    dst_dir = PHASE6_RUN_ROOT / dst_run_id / "features"
    dst_manifest = dst_dir / "feature_manifest.json"
    if dst_manifest.exists():
        return read_json(dst_manifest)
    src_dir = src_root / src_run_id / "features"
    if not src_dir.exists():
        raise FileNotFoundError(src_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in ["train_features.pt", "val_features.pt"]:
        shutil.copy2(src_dir / name, dst_dir / name)
    src_manifest = read_json(src_dir / "feature_manifest.json")
    manifest = dict(src_manifest)
    manifest.update({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": dst_run_id,
        "phase": "phase6_feature_copy",
        "source_root": str(src_root),
        "source_run_id": src_run_id,
        "raw_data_modified": False,
    })
    # Rewrite train/val paths to the Phase6 copy.
    for split in ["train", "val"]:
        if split in manifest:
            manifest[split] = dict(manifest[split])
            manifest[split]["path"] = str(dst_dir / f"{split}_features.pt")
            manifest[split]["cached"] = True
    write_json(dst_manifest, manifest)
    return manifest


def metric_get(d: dict[str, Any], path: str) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def mean_std(vals: list[Any]) -> dict[str, Any]:
    xs = [float(v) for v in vals if v is not None]
    if not xs:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(np.mean(xs)), "std": float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0, "n": len(xs)}


def train_or_load_head(
    root: Path,
    run_id: str,
    condition: str,
    aux_names: list[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    horizons: tuple[int, ...],
    report_dir: Path,
    experiment_name: str,
    seed: int,
    max_epochs: int,
    batch_size: int,
    lr: float,
    hidden_dim: int,
    dropout: float,
    wandb_mode: str,
    thresholds: dict[str, Any] | None,
) -> dict[str, Any]:
    out_json = report_dir / f"{experiment_name}_report.json"
    if out_json.exists():
        return read_json(out_json)
    old_wandb_name = os.environ.get("WANDB_NAME")
    os.environ["WANDB_NAME"] = experiment_name
    try:
        return p5.train_future_head(root, run_id, experiment_name, aux_names, train_idx, val_idx, horizons, max_epochs, batch_size, lr, hidden_dim, dropout, seed, wandb_mode, report_dir, thresholds, condition)
    finally:
        if old_wandb_name is None:
            os.environ.pop("WANDB_NAME", None)
        else:
            os.environ["WANDB_NAME"] = old_wandb_name


def render_multiseed(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase6-1 Multi-seed Stability", "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- feature_run_id: `{payload['feature_run_id']}`",
        f"- seeds: `{payload['seeds']}`",
        "- raw_data_modified: `False`", "",
        "## Per-seed metrics", "",
        "| condition | seed | H1 F1 | H1 AUPRC | H1 ECE | H1 high recall | H1 lead mean | H3 F1 | H3 AUPRC | H5 F1 | H5 AUPRC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        best = row.get("best_val") or {}
        h1, h3, h5 = best.get("H1", {}), best.get("H3", {}), best.get("H5", {})
        lines.append(f"| {row['condition']} | {row['seed']} | {fmt(h1.get('future_slip_f1'))} | {fmt(h1.get('future_slip_auprc'))} | {fmt(h1.get('stability_calibration_error'))} | {fmt(h1.get('high_risk_model_recall_on_future_slip'))} | {fmt(h1.get('lead_time_to_slip_onset_steps_mean'))} | {fmt(h3.get('future_slip_f1'))} | {fmt(h3.get('future_slip_auprc'))} | {fmt(h5.get('future_slip_f1'))} | {fmt(h5.get('future_slip_auprc'))} |")
    lines += ["", "## Mean ± std", "", "| condition | metric | mean | std | n |", "|---|---|---:|---:|---:|"]
    for cond, stats in payload["metric_summary"].items():
        for metric, stat in stats.items():
            lines.append(f"| {cond} | {metric} | {fmt(stat['mean'])} | {fmt(stat['std'])} | {stat['n']} |")
    lines += ["", "## Interpretation", "", payload["interpretation"]]
    return "\n".join(lines) + "\n"


def command_phase6_1(args: argparse.Namespace) -> None:
    stamp = args.stamp or now_stamp()
    report_dir = PHASE6_REPORT_ROOT / "phase6_1_multiseed_stability"
    report_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.feature_run_id or f"phase6_1_friction_features_{stamp}"
    manifest = ensure_phase6_feature_copy(PHASE5_RUN_ROOT, PHASE5_FRICTION_RUN, run_id)
    train = load_feature(PHASE6_RUN_ROOT, run_id, "train")
    thresholds = train.get("friction_thresholds_pred_train") or manifest.get("friction_thresholds_pred_train") or p5.threshold_from_train(train, "pred")
    train_idx = p5.all_indices(train)
    val_idx = p5.all_indices(load_feature(PHASE6_RUN_ROOT, run_id, "val"))
    conditions = {
        "full_dynamics_baseline": BASE_FULL,
        "full_plus_q": FULL_PLUS_Q,
    }
    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        for condition, aux in conditions.items():
            exp = f"phase6_1_{condition}_seed{seed}_{stamp}"
            row = train_or_load_head(PHASE6_RUN_ROOT, run_id, condition, aux, train_idx, val_idx, tuple(args.horizons), report_dir, exp, seed, args.max_epochs, args.train_batch_size, args.lr, args.hidden_dim, args.dropout, args.wandb_mode, thresholds)
            row["seed"] = seed
            rows.append(row)
    metric_paths = {
        "H1_f1": "best_val.H1.future_slip_f1",
        "H1_auroc": "best_val.H1.future_slip_auroc",
        "H1_auprc": "best_val.H1.future_slip_auprc",
        "H1_ece": "best_val.H1.stability_calibration_error",
        "H1_high_recall": "best_val.H1.high_risk_model_recall_on_future_slip",
        "H1_lead_mean": "best_val.H1.lead_time_to_slip_onset_steps_mean",
        "H3_f1": "best_val.H3.future_slip_f1",
        "H3_auprc": "best_val.H3.future_slip_auprc",
        "H3_ece": "best_val.H3.stability_calibration_error",
        "H5_f1": "best_val.H5.future_slip_f1",
        "H5_auprc": "best_val.H5.future_slip_auprc",
        "H5_ece": "best_val.H5.stability_calibration_error",
    }
    summary: dict[str, Any] = {}
    for cond in conditions:
        cr = [r for r in rows if r["condition"] == cond]
        summary[cond] = {m: mean_std([metric_get(r, path) for r in cr]) for m, path in metric_paths.items()}
    base_h3 = summary["full_dynamics_baseline"]["H3_f1"]["mean"]
    q_h3 = summary["full_plus_q"]["H3_f1"]["mean"]
    interp = f"Across seeds {args.seeds}, full+q H3 F1 mean={fmt(q_h3)} versus baseline={fmt(base_h3)}; the result tests whether the Phase5 friction gain is initialization-stable."
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "phase": "phase6_1_multiseed_stability",
        "stamp": stamp,
        "feature_run_id": run_id,
        "feature_manifest": str(PHASE6_RUN_ROOT / run_id / "features/feature_manifest.json"),
        "seeds": list(args.seeds),
        "conditions": list(conditions),
        "rows": rows,
        "metric_summary": summary,
        "interpretation": interp,
        "raw_data_modified": False,
    }
    json_path = report_dir / "phase6_1_multiseed_stability_report.json"
    md_path = report_dir / "phase6_1_multiseed_stability_report.md"
    payload["json_path"] = str(json_path); payload["md_path"] = str(md_path)
    write_json(json_path, payload)
    md_path.write_text(render_multiseed(payload), encoding="utf-8")
    print(md_path)


def render_comparison(title: str, payload: dict[str, Any]) -> str:
    lines = [
        f"# {title}", "", f"- generated_at: `{payload['generated_at']}`", "- raw_data_modified: `False`", "",
        "## Results", "",
        "| condition | train n | val/test n | H1 F1 | H1 AUROC | H1 AUPRC | H1 ECE | H1 high recall | H1 lead mean | H3 F1 | H3 AUPRC | H5 F1 | H5 AUPRC | current force RMSE | current slip F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["conditions"]:
        best = row.get("best_val") or {}
        h1, h3, h5 = best.get("H1", {}), best.get("H3", {}), best.get("H5", {})
        cur = row.get("current_metrics_val_subset", {})
        lines.append(f"| {row['condition']} | {row.get('train_count')} | {row.get('val_count')} | {fmt(h1.get('future_slip_f1'))} | {fmt(h1.get('future_slip_auroc'))} | {fmt(h1.get('future_slip_auprc'))} | {fmt(h1.get('stability_calibration_error'))} | {fmt(h1.get('high_risk_model_recall_on_future_slip'))} | {fmt(h1.get('lead_time_to_slip_onset_steps_mean'))} | {fmt(h3.get('future_slip_f1'))} | {fmt(h3.get('future_slip_auprc'))} | {fmt(h5.get('future_slip_f1'))} | {fmt(h5.get('future_slip_auprc'))} | {fmt(cur.get('force_rmse_mean_N'))} | {fmt(cur.get('slip_f1'))} |")
    lines += ["", "## Interpretation", "", payload.get("interpretation", "")]
    if payload.get("notes"):
        lines += ["", "## Notes", ""] + [f"- {n}" for n in payload["notes"]]
    return "\n".join(lines) + "\n"


def command_phase6_2(args: argparse.Namespace) -> None:
    stamp = args.stamp or now_stamp()
    report_dir = PHASE6_REPORT_ROOT / "phase6_2_heldout_sharp_generalization"
    report_dir.mkdir(parents=True, exist_ok=True)
    friction_run_id = args.feature_run_id or f"phase6_2_friction_features_{stamp}"
    ensure_phase6_feature_copy(PHASE5_RUN_ROOT, PHASE5_FRICTION_RUN, friction_run_id)
    train_fric = load_feature(PHASE6_RUN_ROOT, friction_run_id, "train")
    thresholds = train_fric.get("friction_thresholds_pred_train") or p5.threshold_from_train(train_fric, "pred")
    specs = [
        ("separate_late_fusion", PHASE4_RUN_ROOT, PHASE4_SEPARATE_RUN, STATIC_FORCE_SLIP),
        ("decoupled_static", PHASE4_RUN_ROOT, PHASE4_DECOUPLED_RUN, STATIC_FORCE_SLIP),
        ("decoupled_dynamics", PHASE4_RUN_ROOT, PHASE4_DECOUPLED_RUN, BASE_FULL),
        ("decoupled_dynamics_friction", PHASE6_RUN_ROOT, friction_run_id, FULL_PLUS_Q),
    ]
    rows = []
    for condition, root, run_id, aux in specs:
        train = load_feature(root, run_id, "train")
        val = load_feature(root, run_id, "val")
        train_idx = p5.subset_indices_by_groups(train, ["flat", "sphere"])
        val_idx = p5.subset_indices_by_groups(val, ["sharp"])
        exp = f"phase6_2_{condition}_{stamp}"
        row = train_or_load_head(root, run_id, condition, aux, train_idx, val_idx, tuple(args.horizons), report_dir, exp, args.seed, args.max_epochs, args.train_batch_size, args.lr, args.hidden_dim, args.dropout, args.wandb_mode, thresholds if "decoupled" in condition else None)
        row["heldout_split"] = {"train_groups": ["flat", "sphere"], "test_groups": ["sharp"]}
        rows.append(row)
    best = max(rows, key=lambda r: metric_get(r, "best_val.H5.future_slip_auprc") or -1)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "phase": "phase6_2_heldout_sharp_generalization",
        "stamp": stamp,
        "heldout_split": {"train_groups": ["flat", "sphere"], "test_groups": ["sharp"]},
        "conditions": rows,
        "interpretation": f"Best H5 AUPRC on held-out sharp is `{best['condition']}`. This complements Phase5's flat+sharp→sphere stress test by holding out a localized sharp contact geometry.",
        "notes": ["Only lightweight heads are trained; SPARSH encoder and force/slip estimators are reused from Phase4/5.", "The split uses train samples from flat+sphere and validation samples from sharp."],
        "raw_data_modified": False,
    }
    json_path = report_dir / "phase6_2_heldout_sharp_generalization_report.json"
    md_path = report_dir / "phase6_2_heldout_sharp_generalization_report.md"
    payload["json_path"] = str(json_path); payload["md_path"] = str(md_path)
    write_json(json_path, payload)
    md_path.write_text(render_comparison("Phase6-2 Held-out Sharp Generalization", payload), encoding="utf-8")
    print(md_path)


class JointDataset(Dataset):
    def __init__(self, payload: dict[str, Any], indices: np.ndarray, horizons: tuple[int, ...]):
        self.payload = payload
        self.indices = np.asarray(indices, dtype=np.int64)
        self.z = payload["z"].float()
        schema = list(payload["aux_schema"])
        # Use causal delta-force if available; otherwise only z. This remains a lightweight joint-head ablation over frozen features.
        self.delta_idx = [schema.index(n) for n in ["dFx_causal_N", "dFy_causal_N", "dFz_causal_N"] if n in schema]
        self.aux = payload["aux"].float()
        all_h = tuple(int(h) for h in payload["horizons"])
        self.hidx = [all_h.index(int(h)) for h in horizons]
        self.future = payload["future_slip"].float()
        self.force = payload["force_gt_n"].float()
        self.slip = payload["current_slip"].float()
    def __len__(self) -> int:
        return int(len(self.indices))
    @property
    def input_dim(self) -> int:
        return int(self.z.shape[1] + len(self.delta_idx))
    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        idx = int(self.indices[i])
        parts = [self.z[idx]]
        if self.delta_idx:
            parts.append(self.aux[idx, self.delta_idx])
        return {"x": torch.cat(parts), "force": self.force[idx], "slip": self.slip[idx], "future_slip": self.future[idx, self.hidx]}


class JointForceSlipFutureHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, horizons: int, dropout: float, tau: float, alpha: float, r_max: float):
        super().__init__()
        self.tau = float(tau); self.alpha = float(alpha); self.r_max = float(r_max)
        self.trunk = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.force_head = nn.Linear(hidden_dim, 3)
        self.slip_head = nn.Linear(hidden_dim, 1)
        self.future_head = nn.Sequential(nn.Linear(hidden_dim + 6, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, horizons))
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(x)
        force = self.force_head(h)
        slip_logit = self.slip_head(h).squeeze(-1)
        ft = torch.sqrt(force[:,0].square() + force[:,1].square() + EPS)
        fn = torch.abs(force[:,2])
        r = torch.clamp(ft / (fn + EPS), 0.0, self.r_max)
        q = torch.sigmoid(self.alpha * (r - self.tau))
        p_slip = torch.sigmoid(slip_logit)
        future_in = torch.cat([h, force, r[:,None], q[:,None], p_slip[:,None]], dim=1)
        stable_logits = self.future_head(future_in)
        return {"force": force, "slip_logit": slip_logit, "stable_logits": stable_logits, "ratio": r, "q": q}


def current_force_slip_metrics(force_gt: np.ndarray, force_pred: np.ndarray, slip_gt: np.ndarray, slip_prob: np.ndarray) -> dict[str, Any]:
    err = force_pred - force_gt
    rmse_xyz = np.sqrt(np.mean(np.square(err), axis=0))
    mae_xyz = np.mean(np.abs(err), axis=0)
    pred = (slip_prob >= 0.5).astype(int)
    out = {
        "n_samples": int(len(slip_gt)),
        "force_rmse_xyz_N": [float(x) for x in rmse_xyz],
        "force_rmse_mean_N": float(np.mean(rmse_xyz)),
        "force_mae_xyz_N": [float(x) for x in mae_xyz],
        "force_mae_mean_N": float(np.mean(mae_xyz)),
        "slip_accuracy": float(accuracy_score(slip_gt, pred)),
        "slip_precision": float(precision_score(slip_gt, pred, zero_division=0)),
        "slip_recall": float(recall_score(slip_gt, pred, zero_division=0)),
        "slip_f1": float(f1_score(slip_gt, pred, zero_division=0)),
    }
    if len(np.unique(slip_gt)) > 1:
        out["slip_auroc"] = float(roc_auc_score(slip_gt, slip_prob))
        out["slip_auprc"] = float(average_precision_score(slip_gt, slip_prob))
    else:
        out["slip_auroc"] = None; out["slip_auprc"] = None
    return out


def eval_joint(model: nn.Module, loader: DataLoader, payload: dict[str, Any], indices: np.ndarray, horizons: tuple[int, ...]) -> tuple[dict[str, Any], dict[str, Any]]:
    dev = device(); model.eval()
    force_preds=[]; slip_probs=[]; stable_probs=[]
    with torch.no_grad():
        for batch in loader:
            out = model(batch["x"].to(dev))
            force_preds.append(out["force"].cpu().numpy())
            slip_probs.append(torch.sigmoid(out["slip_logit"]).cpu().numpy())
            stable_probs.append(torch.sigmoid(out["stable_logits"]).cpu().numpy())
    force_pred=np.concatenate(force_preds); slip_prob=np.concatenate(slip_probs); stable_prob=np.concatenate(stable_probs)
    force_gt=payload["force_gt_n"].numpy()[indices]
    slip_gt=payload["current_slip"].numpy()[indices].astype(int)
    current=current_force_slip_metrics(force_gt, force_pred, slip_gt, slip_prob)
    future=payload["future_slip"].numpy()[indices]
    all_h=tuple(int(h) for h in payload["horizons"]); hidx=[all_h.index(int(h)) for h in horizons]
    future=future[:,hidx]
    fmetrics={"n_samples": int(len(indices)), "horizons": list(horizons)}
    meta=[payload["metadata"][int(i)] for i in indices]
    for j,h in enumerate(horizons):
        y_stable=1-future[:,j].astype(int); p_stable=stable_prob[:,j]
        y_slip=1-y_stable; p_inst=1-p_stable
        row={**p5.binary_metrics(y_slip, p_inst, "future_slip"), **p5.binary_metrics(y_stable, p_stable, "future_stability"), "stability_calibration_error": p5.calibration_error(y_stable, p_stable)}
        if h==1:
            row.update(wm.lead_time_to_slip(meta, slip_gt, p_inst))
        fmetrics[f"H{h}"]=row
    return current, fmetrics


def train_or_load_joint(report_dir: Path, root: Path, run_id: str, seed: int, horizons: tuple[int, ...], max_epochs: int, batch_size: int, lr: float, hidden_dim: int, dropout: float, wandb_mode: str, thresholds: dict[str, Any], stamp: str) -> dict[str, Any]:
    exp=f"phase6_3_joint_lightweight_seed{seed}_{stamp}"
    out_json=report_dir / f"{exp}_report.json"
    if out_json.exists():
        return read_json(out_json)
    set_seed(seed)
    train=load_feature(root, run_id, "train"); val=load_feature(root, run_id, "val")
    train_idx=p5.all_indices(train); val_idx=p5.all_indices(val)
    tr=JointDataset(train, train_idx, horizons); va=JointDataset(val, val_idx, horizons)
    tr_loader=DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=0)
    va_loader=DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=0)
    dev=device()
    model=JointForceSlipFutureHead(tr.input_dim, hidden_dim, len(horizons), dropout, thresholds["q_tau"], thresholds["q_alpha"], thresholds["ratio_p99"]).to(dev)
    opt=torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    head_dir=PHASE6_RUN_ROOT / run_id / "heads" / exp
    ckpt_dir=head_dir / "checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_MODE"]=wandb_mode
    wb=None
    if wandb_mode != "disabled":
        os.environ["WANDB_NAME"]=exp
        wb=wandb.init(project="sparsh-finetune-tactile-grasp", entity="junyuzhuzjy-zhejiang-university", dir=str(head_dir), id=exp, name=exp, group="phase6_frozen_vs_joint", tags=["phase6","joint","force-slip-future"], notes="Lightweight joint force/slip/future head over frozen cached SPARSH features; not full backbone finetuning.", config={"seed":seed,"run_id":run_id,"raw_data_modified":False})
    best_score=-math.inf; best=None; hist=[]; start=time.time()
    for epoch in range(1,max_epochs+1):
        model.train(); losses=[]; fls=[]; sls=[]; futs=[]
        for batch in tr_loader:
            out=model(batch["x"].to(dev))
            force_t=batch["force"].to(dev)
            slip_t=batch["slip"].to(dev)
            stable_t=1.0-batch["future_slip"].to(dev)
            fl=F.smooth_l1_loss(out["force"], force_t)
            sl=F.binary_cross_entropy_with_logits(out["slip_logit"], slip_t)
            fu=F.binary_cross_entropy_with_logits(out["stable_logits"], stable_t)
            loss=args_force_weight()*fl + args_slip_weight()*sl + args_future_weight()*fu
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),10.0); opt.step()
            losses.append(float(loss.detach().cpu())); fls.append(float(fl.detach().cpu())); sls.append(float(sl.detach().cpu())); futs.append(float(fu.detach().cpu()))
        current,future=eval_joint(model, va_loader, val, val_idx, horizons)
        score=np.mean([future[f"H{h}"].get("future_slip_auprc") or 0.0 for h in horizons])
        rec={"epoch":epoch,"train_loss":float(np.mean(losses)),"force_loss":float(np.mean(fls)),"slip_loss":float(np.mean(sls)),"future_loss":float(np.mean(futs)),"current_metrics_val":current,"future_metrics_val":future,"selection_score_mean_auprc":float(score)}
        hist.append(rec)
        if wb is not None:
            log={"epoch":epoch,"train/loss":rec["train_loss"],"train/force_loss":rec["force_loss"],"train/slip_loss":rec["slip_loss"],"train/future_loss":rec["future_loss"],"val/force_rmse_mean_N":current["force_rmse_mean_N"],"val/slip_f1":current["slip_f1"],"val/selection_score":float(score)}
            for h in horizons:
                hr=future[f"H{h}"]; log[f"val/H{h}_f1"]=hr.get("future_slip_f1"); log[f"val/H{h}_auprc"]=hr.get("future_slip_auprc"); log[f"val/H{h}_ece"]=hr.get("stability_calibration_error")
            wb.log({k:v for k,v in log.items() if v is not None}, step=epoch)
        if score>best_score:
            best_score=float(score); best=rec
            torch.save({"model_state":model.state_dict(),"seed":seed,"horizons":list(horizons),"input_dim":tr.input_dim,"hidden_dim":hidden_dim,"dropout":dropout,"thresholds":thresholds,"epoch":epoch,"metrics":rec}, ckpt_dir/"best.pth")
        write_json(head_dir/"history.json", hist)
    summary={"generated_at":datetime.now().isoformat(timespec="seconds"),"phase":"phase6_3_joint_lightweight","condition":"joint_lightweight_force_slip_future","experiment_name":exp,"feature_run_id":run_id,"seed":seed,"best_epoch":best["epoch"] if best else None,"best_selection_score_mean_auprc":best_score,"current_metrics_val":best["current_metrics_val"] if best else None,"best_val":best["future_metrics_val"] if best else None,"checkpoint":str(ckpt_dir/"best.pth"),"wall_time_seconds":time.time()-start,"raw_data_modified":False,"notes":"Lightweight joint head over frozen cached features; this is a proxy for joint force/slip/future supervision, not full SPARSH backbone finetuning."}
    write_json(out_json, summary)
    if wb is not None:
        def _timeout(_s,_f): raise TimeoutError("wandb.finish timed out")
        old=signal.signal(signal.SIGALRM,_timeout); signal.alarm(30)
        try: wb.finish()
        except Exception as exc: summary["wandb_finish_status"]=str(exc)
        finally: signal.alarm(0); signal.signal(signal.SIGALRM,old); write_json(out_json, summary)
    return summary


def args_force_weight() -> float: return 5.0
def args_slip_weight() -> float: return 1.0
def args_future_weight() -> float: return 1.0


def render_phase6_3(payload: dict[str, Any]) -> str:
    lines=["# Phase6-3 Frozen vs Joint Training Ablation", "", f"- generated_at: `{payload['generated_at']}`", "- raw_data_modified: `False`", "", "## Comparison", "", "| condition | force RMSE | slip F1 | H1 F1 | H1 AUPRC | H1 ECE | H3 F1 | H3 AUPRC | H5 F1 | H5 AUPRC | checkpoint |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in payload["rows"]:
        cur=r.get("current_metrics_val") or r.get("current_metrics_val_subset") or {}
        best=r.get("best_val") or {}
        h1,h3,h5=best.get("H1",{}),best.get("H3",{}),best.get("H5",{})
        lines.append(f"| {r['condition']} | {fmt(cur.get('force_rmse_mean_N'))} | {fmt(cur.get('slip_f1'))} | {fmt(h1.get('future_slip_f1'))} | {fmt(h1.get('future_slip_auprc'))} | {fmt(h1.get('stability_calibration_error'))} | {fmt(h3.get('future_slip_f1'))} | {fmt(h3.get('future_slip_auprc'))} | {fmt(h5.get('future_slip_f1'))} | {fmt(h5.get('future_slip_auprc'))} | `{r.get('checkpoint')}` |")
    lines += ["", "## Interpretation", "", payload["interpretation"], "", "## Notes", ""] + [f"- {n}" for n in payload.get("notes", [])]
    return "\n".join(lines)+"\n"


def command_phase6_3(args: argparse.Namespace) -> None:
    stamp=args.stamp or now_stamp()
    report_dir=PHASE6_REPORT_ROOT/"phase6_3_frozen_vs_joint_ablation"; report_dir.mkdir(parents=True,exist_ok=True)
    run_id=args.feature_run_id or f"phase6_3_friction_features_{stamp}"
    ensure_phase6_feature_copy(PHASE5_RUN_ROOT, PHASE5_FRICTION_RUN, run_id)
    train=load_feature(PHASE6_RUN_ROOT, run_id,"train"); val=load_feature(PHASE6_RUN_ROOT, run_id,"val")
    thresholds=train.get("friction_thresholds_pred_train") or p5.threshold_from_train(train,"pred")
    train_idx=p5.all_indices(train); val_idx=p5.all_indices(val)
    frozen=train_or_load_head(PHASE6_RUN_ROOT, run_id, "frozen_two_stage_full_plus_q", FULL_PLUS_Q, train_idx, val_idx, tuple(args.horizons), report_dir, f"phase6_3_frozen_full_plus_q_seed{args.seed}_{stamp}", args.seed, args.max_epochs, args.train_batch_size, args.lr, args.hidden_dim, args.dropout, args.wandb_mode, thresholds)
    joint=train_or_load_joint(report_dir, PHASE6_RUN_ROOT, run_id, args.seed, tuple(args.horizons), args.max_epochs, args.train_batch_size, args.lr, args.hidden_dim, args.dropout, args.wandb_mode, thresholds, stamp)
    rows=[dict(frozen, condition="frozen_two_stage_full_plus_q"), joint]
    f_rmse=metric_get(frozen,"current_metrics_val_subset.force_rmse_mean_N"); j_rmse=metric_get(joint,"current_metrics_val.force_rmse_mean_N")
    interp=f"The frozen two-stage route keeps the Phase4 decoupled current-task metrics (force RMSE={fmt(f_rmse)}) while the lightweight joint head reaches force RMSE={fmt(j_rmse)}. Use this as design-justification evidence rather than a full end-to-end finetuning claim."
    payload={"generated_at":datetime.now().isoformat(timespec="seconds"),"phase":"phase6_3_frozen_vs_joint_ablation","stamp":stamp,"feature_run_id":run_id,"rows":rows,"interpretation":interp,"notes":["Joint training is a lightweight frozen-feature proxy that jointly supervises force, slip, and future stability heads.","The ablation tests whether coupling all losses immediately is preferable to preserving a trained perception module before dynamics learning."],"raw_data_modified":False}
    json_path=report_dir/"phase6_3_frozen_vs_joint_ablation_report.json"; md_path=report_dir/"phase6_3_frozen_vs_joint_ablation_report.md"
    payload["json_path"]=str(json_path); payload["md_path"]=str(md_path)
    write_json(json_path,payload); md_path.write_text(render_phase6_3(payload),encoding="utf-8"); print(md_path)


def load_model_for_head(row: dict[str, Any], root: Path | None = None, run_id: str | None = None):
    ckpt=torch.load(row["checkpoint"], map_location="cpu", weights_only=False)
    aux_names=ckpt["aux_names"]
    root=Path(row.get("feature_root", root or PHASE6_RUN_ROOT))
    run_id=row.get("feature_run_id", run_id)
    payload=torch.load(root/run_id/"features/val_features.pt", map_location="cpu", weights_only=False)
    indices=p5.all_indices(payload)
    ds=p5.FeatureHeadDataset(payload, indices, aux_names, tuple(ckpt["horizons"]))
    model=wm.MLP(int(ds.z.shape[1]+len(aux_names)), int(ckpt.get("hidden_dim",512)), len(ckpt["horizons"]), float(ckpt.get("dropout",0.1)))
    model.load_state_dict(ckpt["model_state"]); model.to(device())
    stable=p5.model_predictions(model, DataLoader(ds,batch_size=2048,shuffle=False,num_workers=0), device())
    return payload, indices, stable, ckpt


def trajectory_records(payload: dict[str, Any], indices: np.ndarray, stable_prob: np.ndarray, threshold: float=0.5) -> list[dict[str, Any]]:
    meta=[payload["metadata"][int(i)] for i in indices]
    current=payload["current_slip"].numpy()[indices].astype(int)
    p_inst=1.0-stable_prob[:,0]
    groups: dict[tuple[str,str], list[tuple[int,int,int,float]]] = {}
    for local,(m,idx,slip,prob) in enumerate(zip(meta, indices, current, p_inst)):
        key=(str(m["dataset"]), str(m["trajectory"]))
        groups.setdefault(key,[]).append((int(m["sample"]), int(idx), int(slip), float(prob)))
    out=[]
    for (dataset,traj), rows in groups.items():
        rows=sorted(rows)
        slips=[s for s,_,sl,_ in rows if sl==1]
        onset=min(slips) if slips else None
        pre_preds=[s for s,_,_,prob in rows if onset is not None and s<onset and prob>=threshold]
        any_preds=[s for s,_,_,prob in rows if prob>=threshold]
        rec={"dataset":dataset,"trajectory":traj,"n":len(rows),"onset":onset,"max_prob":max(prob for *_,prob in rows),"has_prediction":bool(any_preds)}
        if onset is not None:
            if pre_preds:
                rec.update({"status":"detected_pre","lead":onset-min(pre_preds),"first_pred":min(pre_preds)})
            elif any_preds:
                rec.update({"status":"late","lead":0,"first_pred":min(any_preds)})
            else:
                rec.update({"status":"missed","lead":None,"first_pred":None})
        else:
            rec.update({"status":"false_alarm" if any_preds else "true_negative", "lead":None, "first_pred":min(any_preds) if any_preds else None})
        out.append(rec)
    return out


def early_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    slip=[r for r in records if r["onset"] is not None]
    detected=[r for r in slip if r["status"]=="detected_pre"]
    late=[r for r in slip if r["status"]=="late"]
    missed=[r for r in slip if r["status"]=="missed"]
    false=[r for r in records if r["status"]=="false_alarm"]
    leads=[r["lead"] for r in detected if r.get("lead") is not None]
    return {"trajectory_count":len(records),"slip_trajectories":len(slip),"detected_pre_slip_trajectories":len(detected),"late_warning_trajectories":len(late),"missed_slip_trajectories":len(missed),"false_alarm_trajectories":len(false),"early_warning_recall":len(detected)/max(1,len(slip)),"lead_time_to_slip_onset_steps_mean":float(np.mean(leads)) if leads else None,"lead_time_to_slip_onset_steps_median":float(np.median(leads)) if leads else None}


def plot_case(payload: dict[str, Any], stable_prob: np.ndarray, thresholds: dict[str, Any], case: dict[str, Any], out_dir: Path, prefix: str) -> str | None:
    if plt is None: return None
    rows=[(i,int(m["sample"])) for i,m in enumerate(payload["metadata"]) if str(m["dataset"])==case["dataset"] and str(m["trajectory"])==str(case["trajectory"])]
    if not rows: return None
    rows=sorted(rows,key=lambda x:x[1]); idx=np.array([i for i,_ in rows]); x=np.array([s for _,s in rows])
    current=payload["current_slip"].numpy()[idx].astype(int); p_slip=payload["slip_probs"].numpy()[idx,1]; p_inst=1.0-stable_prob[idx]
    comp=p5.force_components(payload,"pred"); fn,ft,fmag=comp["Fn"][idx],comp["Ft"][idx],comp["Fmag"][idx]
    r=np.clip(comp["ratio"][idx],0,thresholds["ratio_p99"]); q=p5.sigmoid_np(thresholds["q_alpha"]*(r-thresholds["q_tau"]))
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    axes[0].plot(x,current,label="GT current slip",color="black",lw=1.5); axes[0].plot(x,p_slip,label="current p_slip",color="tab:blue")
    for j,h in enumerate([1,3,5][:p_inst.shape[1]]): axes[0].plot(x,p_inst[:,j],label=f"future instability H{h}")
    if case.get("onset") is not None: axes[0].axvline(case["onset"],color="red",ls="--",label="slip onset")
    if case.get("first_pred") is not None: axes[0].axvline(case["first_pred"],color="green",ls=":",label="first warning")
    axes[0].set_ylim(-0.05,1.05); axes[0].legend(fontsize=8,loc="upper right"); axes[0].set_ylabel("prob/label")
    axes[1].plot(x,r,label="clipped Ft/Fn",color="tab:green"); axes[1].plot(x,q,label="friction q",color="tab:brown"); axes[1].axhline(thresholds["ratio_p80"],color="gray",ls="--",label="train p80"); axes[1].legend(fontsize=8); axes[1].set_ylabel("friction")
    axes[2].plot(x,fn,label="Fn"); axes[2].plot(x,ft,label="Ft"); axes[2].plot(x,fmag,label="Fmag"); axes[2].legend(fontsize=8); axes[2].set_ylabel("force N"); axes[2].set_xlabel("sample")
    fig.suptitle(f"{case['label']} | {case['dataset']} | traj={case['trajectory']}")
    out_dir.mkdir(parents=True,exist_ok=True)
    name=f"{prefix}_{case['label']}_{case['dataset']}_{case['trajectory']}.png".replace('/','_')[:220]
    path=out_dir/name; plt.tight_layout(); plt.savefig(path,dpi=170); plt.close(fig); return str(path)


def command_phase6_4(args: argparse.Namespace) -> None:
    report_dir=PHASE6_REPORT_ROOT/"phase6_4_early_warning_analysis"; fig_dir=report_dir/"figures"; report_dir.mkdir(parents=True,exist_ok=True)
    p61=read_json(PHASE6_REPORT_ROOT/"phase6_1_multiseed_stability/phase6_1_multiseed_stability_report.json")
    # Use the best full+q seed by H3 AUPRC.
    candidates=[r for r in p61["rows"] if r["condition"]=="full_plus_q"]
    best=max(candidates,key=lambda r: metric_get(r,"best_val.H3.future_slip_auprc") or -1)
    payload, indices, stable, ckpt=load_model_for_head(best, PHASE6_RUN_ROOT, p61["feature_run_id"])
    thresholds=ckpt.get("thresholds") or p61.get("thresholds_pred_train") or p5.threshold_from_train(payload,"pred")
    recs=trajectory_records(payload, indices, stable)
    summ=early_summary(recs)
    def pick(status, label, key=None):
        xs=[r for r in recs if r["status"]==status]
        if not xs: return None
        if status=="detected_pre": r=max(xs,key=lambda r:r.get("lead") or 0)
        elif status=="late": r=max(xs,key=lambda r:r.get("max_prob") or 0)
        elif status=="missed": r=max(xs,key=lambda r:r.get("max_prob") or 0)
        else: r=max(xs,key=lambda r:r.get("max_prob") or 0)
        out=dict(r); out["label"]=label; return out
    cases=[]
    for status,label in [("detected_pre","success_early_warning"),("late","late_warning")]:
        c=pick(status,label)
        if c: cases.append(c)
    c=pick("missed","missed_warning")
    if c is None:
        slip_recs=[r for r in recs if r["onset"] is not None]
        if slip_recs:
            # No actual miss at threshold 0.5: plot the weakest slip trajectory as an absence check.
            c=dict(min(slip_recs, key=lambda r: r.get("max_prob") or 0.0))
            c["label"]="missed_warning_absence_check"
            c["case_note"]="No true missed-warning trajectory at threshold 0.5; this is the lowest-risk slip trajectory."
    if c: cases.append(c)
    c=pick("false_alarm","false_alarm")
    if c is None:
        nonslip=[r for r in recs if r["onset"] is None]
        if nonslip:
            # No actual false alarm at threshold 0.5: plot the highest-risk no-slip trajectory as an absence check.
            c=dict(max(nonslip, key=lambda r: r.get("max_prob") or 0.0))
            c["label"]="false_alarm_absence_check"
            c["case_note"]="No true false-alarm trajectory at threshold 0.5; this is the highest-risk no-slip trajectory."
    if c: cases.append(c)
    # Held-out sharp case from Phase6-2 friction model, if available.
    p62=read_json(PHASE6_REPORT_ROOT/"phase6_2_heldout_sharp_generalization/phase6_2_heldout_sharp_generalization_report.json")
    fr=[r for r in p62["conditions"] if r["condition"]=="decoupled_dynamics_friction"]
    held_case=None
    if fr:
        hp, hi, hs, hc=load_model_for_head(fr[0])
        hrecs=trajectory_records(hp, hi, hs)
        sharp=[r for r in hrecs if r["dataset"].startswith("sharp")]
        if sharp:
            h=max(sharp,key=lambda r:r.get("max_prob") or 0); held_case=dict(h); held_case["label"]="heldout_contact_case"
            held_case["figure"]=plot_case(hp, hs, hc.get("thresholds") or thresholds, held_case, fig_dir, "heldout")
            cases.append(held_case)
    figures=[]
    for c in cases:
        if c.get("figure"): figures.append(c["figure"]); continue
        path=plot_case(payload, stable, thresholds, c, fig_dir, "main")
        c["figure"]=path
        if path: figures.append(path)
    payload_out={"generated_at":datetime.now().isoformat(timespec="seconds"),"phase":"phase6_4_early_warning_analysis","selected_condition":best["condition"],"selected_seed":best.get("seed"),"checkpoint":best["checkpoint"],"early_warning_summary":summ,"cases":cases,"figures":figures,"raw_data_modified":False,"notes":["Warnings use H1 future instability probability with threshold 0.5.","Figures overlay GT slip onset, current slip probability, H1/H3/H5 future instability, friction q/FtFn, and predicted force components."]}
    json_path=report_dir/"phase6_4_early_warning_analysis_report.json"; md_path=report_dir/"phase6_4_early_warning_analysis_report.md"
    payload_out["json_path"]=str(json_path); payload_out["md_path"]=str(md_path); write_json(json_path,payload_out)
    lines=["# Phase6-4 Early-warning Analysis", "", f"- generated_at: `{payload_out['generated_at']}`", f"- selected_condition: `{best['condition']}` seed `{best.get('seed')}`", f"- checkpoint: `{best['checkpoint']}`", "- raw_data_modified: `False`", "", "## Early-warning summary", "", "| metric | value |", "|---|---:|"]
    for k,v in summ.items(): lines.append(f"| {k} | {fmt(v) if isinstance(v,float) or v is None else v} |")
    lines += ["", "## Cases", "", "| label | status | dataset | trajectory | onset | first warning | note | figure |", "|---|---|---|---|---:|---:|---|---|"]
    for c in cases: lines.append(f"| {c['label']} | {c['status']} | {c['dataset']} | {c['trajectory']} | {c.get('onset')} | {c.get('first_pred')} | {c.get('case_note','')} | `{c.get('figure')}` |")
    md_path.write_text("\n".join(lines)+"\n",encoding="utf-8"); print(md_path)


def command_summary(args: argparse.Namespace) -> None:
    root=PHASE6_REPORT_ROOT
    p61=read_json(root/"phase6_1_multiseed_stability/phase6_1_multiseed_stability_report.json")
    p62=read_json(root/"phase6_2_heldout_sharp_generalization/phase6_2_heldout_sharp_generalization_report.json")
    p63=read_json(root/"phase6_3_frozen_vs_joint_ablation/phase6_3_frozen_vs_joint_ablation_report.json")
    p64=read_json(root/"phase6_4_early_warning_analysis/phase6_4_early_warning_analysis_report.json")
    payload={"generated_at":datetime.now().isoformat(timespec="seconds"),"phase":"phase6_summary","phase_reports":{"phase6_1":p61["md_path"],"phase6_2":p62["md_path"],"phase6_3":p63["md_path"],"phase6_4":p64["md_path"]},"recommended_paper_method":"Two-stage MAE decoupled force/slip perception + friction-aware tactile dynamics head (full+q), with multi-seed stability and two contact-geometry holdout stress tests.","main_table_candidates":["Phase6-1 multi-seed full dynamics vs full+q mean±std","Phase6-2 flat+sphere→sharp heldout comparison","Phase6-3 frozen two-stage vs lightweight joint-head ablation"],"supplement_candidates":["Phase6-4 early-warning/failure-case trajectory figures","Per-seed W&B logs and checkpoint paths"],"safe_claims":["full+q is evaluated across three random seeds while keeping data split and cached features fixed.","friction-aware dynamics is stress-tested on both sphere-heldout and sharp-heldout contact geometry splits.","a lightweight joint-head proxy is included to justify the two-stage design without claiming full end-to-end world-model finetuning."],"avoid_overclaiming":["Do not claim real-robot grasp success or full physical world modeling.","Do not describe tau/q as a measured friction coefficient; it is train-derived empirical conditioning.","Do not present the lightweight joint ablation as exhaustive architecture search."],"remaining_risks":["All experiments remain on derived GSmini datasets, not live robot trials.","Phase6-3 joint training is a frozen-feature proxy, not a full SPARSH backbone finetune."],"raw_data_modified":False}
    json_path=root/"phase6_summary.json"; md_path=root/"phase6_summary.md"; payload["json_path"]=str(json_path); payload["md_path"]=str(md_path); write_json(json_path,payload)
    lines=["# Phase6 Summary", "", f"- generated_at: `{payload['generated_at']}`", "- raw_data_modified: `False`", "", "## Reports", ""]
    for k,v in payload["phase_reports"].items(): lines.append(f"- {k}: `{v}`")
    for title,key in [("Recommended paper method","recommended_paper_method"),("Main table candidates","main_table_candidates"),("Supplement candidates","supplement_candidates"),("Safe claims","safe_claims"),("Avoid overclaiming","avoid_overclaiming"),("Remaining risks","remaining_risks")]:
        lines += ["", f"## {title}", ""]
        val=payload[key]
        if isinstance(val,str): lines.append(val)
        else:
            for x in val: lines.append(f"- {x}")
    md_path.write_text("\n".join(lines)+"\n",encoding="utf-8"); print(md_path)


def build_parser() -> argparse.ArgumentParser:
    ap=argparse.ArgumentParser(description=__doc__); sub=ap.add_subparsers(dest="command",required=True)
    def common(p):
        p.add_argument("--stamp", default=None); p.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS)); p.add_argument("--max-epochs", type=int, default=40); p.add_argument("--train-batch-size", type=int, default=1024); p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--hidden-dim", type=int, default=512); p.add_argument("--dropout", type=float, default=0.1); p.add_argument("--wandb-mode", choices=["online","offline","disabled"], default=os.environ.get("WANDB_MODE","online"))
    p=sub.add_parser("phase6-1"); common(p); p.add_argument("--feature-run-id", default=None); p.add_argument("--seeds", nargs="+", type=int, default=[42,43,44]); p.set_defaults(func=command_phase6_1)
    p=sub.add_parser("phase6-2"); common(p); p.add_argument("--feature-run-id", default=None); p.add_argument("--seed", type=int, default=42); p.set_defaults(func=command_phase6_2)
    p=sub.add_parser("phase6-3"); common(p); p.add_argument("--feature-run-id", default=None); p.add_argument("--seed", type=int, default=42); p.set_defaults(func=command_phase6_3)
    p=sub.add_parser("phase6-4"); p.set_defaults(func=command_phase6_4)
    p=sub.add_parser("summary"); p.set_defaults(func=command_summary)
    return ap


def main() -> None:
    args=build_parser().parse_args(); args.func(args)

if __name__ == "__main__":
    main()
