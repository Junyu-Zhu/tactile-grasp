#!/usr/bin/env python3
"""Phase5 friction-aware tactile dynamics experiments.

This script reads Phase4 derived feature caches and writes only Phase5 derived
reports, figures, checkpoints, and caches. It never mutates raw datasets under
/vla1/zjy.
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
from dataclasses import dataclass
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
except Exception:  # pragma: no cover - figure fallback is recorded in reports.
    plt = None

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import phase2_b_multitask as p2  # noqa: E402
import phase3_2_world_model as wm  # noqa: E402
import wandb  # noqa: E402

REPO = Path("/home/zjy/document/tactile-grasp")
WORKSPACE = REPO / "sparsh-force-slip"
PHASE4_RUN_ROOT = Path("/vla1/zjy/sparsh_runs/force_slip_phase4")
PHASE5_RUN_ROOT = Path("/vla1/zjy/sparsh_runs/force_slip_phase5")
PHASE5_REPORT_ROOT = WORKSPACE / "reports/phase5"
DEFAULT_DECOUPLED_RUN = "phase4_3_reuse_p3_features_20260517_171500"
DEFAULT_SEPARATE_RUN = "phase4_2_separate_features_20260517_171500"
DEFAULT_HORIZONS = (1, 3, 5)
EPS = 1.0e-6

BASE_FULL = ["Fn_pred_N", "Ft_pred_N", "Ft_over_Fn_pred", "p_slip_current", "dFx_causal_N", "dFy_causal_N", "dFz_causal_N"]
STATIC_FORCE_SLIP = ["Fn_pred_N", "Ft_pred_N", "Ft_over_Fn_pred", "p_slip_current"]
FRICTION_MODES = {
    "full_dynamics_baseline": BASE_FULL,
    "full_plus_r": BASE_FULL + ["friction_r_clipped_pred"],
    "full_plus_r_dr": BASE_FULL + ["friction_r_clipped_pred", "friction_dr_clipped_pred"],
    "full_plus_q": BASE_FULL + ["friction_q_pred"],
    "full_plus_q_dq": BASE_FULL + ["friction_q_pred", "friction_dq_pred"],
}


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=p2.json_default), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def feature_path(run_root: Path, run_id: str, split: str) -> Path:
    return run_root / run_id / "features" / f"{split}_features.pt"


def manifest_path(run_root: Path, run_id: str) -> Path:
    return run_root / run_id / "features" / "feature_manifest.json"


def load_feature(run_root: Path, run_id: str, split: str) -> dict[str, Any]:
    path = feature_path(run_root, run_id, split)
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def group_name(meta: dict[str, Any]) -> str:
    ds = str(meta.get("dataset", "unknown"))
    if ds.startswith("flat"):
        return "flat"
    if ds.startswith("sharp"):
        return "sharp"
    if ds.startswith("sphere"):
        return "sphere"
    return ds.split("/")[0].split("_")[0] or "unknown"


def force_components(payload: dict[str, Any], source: str) -> dict[str, np.ndarray]:
    force_key = "force_gt_n" if source == "gt" else "force_pred_n"
    f = payload[force_key].float().numpy()
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    fn = np.abs(fz)
    ft = np.sqrt(np.square(fx) + np.square(fy))
    fmag = np.sqrt(np.square(fx) + np.square(fy) + np.square(fz))
    ratio = ft / (fn + EPS)
    return {"Fx": fx, "Fy": fy, "Fz": fz, "Fn": fn, "Ft": ft, "Fmag": fmag, "ratio": ratio}


def threshold_from_train(train_payload: dict[str, Any], source: str) -> dict[str, Any]:
    comp = force_components(train_payload, source)
    fn = comp["Fn"]
    ratio = comp["ratio"]
    fn_threshold = float(max(EPS, np.quantile(fn[np.isfinite(fn)], 0.10)))
    valid = np.isfinite(ratio) & (fn > fn_threshold)
    ref = ratio[valid]
    p20, p50, p80, p95, p99 = [float(np.quantile(ref, q)) for q in (0.20, 0.50, 0.80, 0.95, 0.99)]
    alpha = float(8.0 / max(p80 - p20, EPS))
    return {
        "source": source,
        "fn_valid_threshold_p10_N": fn_threshold,
        "ratio_p20": p20,
        "ratio_p50": p50,
        "ratio_p80": p80,
        "ratio_p95": p95,
        "ratio_p99": p99,
        "q_tau": p80,
        "q_alpha": alpha,
        "n_train": int(len(ratio)),
        "n_train_valid_contact": int(valid.sum()),
        "valid_contact_rule": "Fn > train Fn p10 for the same force source",
    }


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.std(x) <= 0 or np.std(y) <= 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(a: np.ndarray) -> np.ndarray:
    # Average ranks with stable handling for ties, implemented to avoid scipy dependency.
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and a[order[j]] == a[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3:
        return None
    return pearson(rankdata(x), rankdata(y))


def safe_rate(num: int | float, den: int | float) -> float | None:
    return None if den == 0 else float(num) / float(den)


def diagnostic_rows_for_source(train_payload: dict[str, Any], val_payload: dict[str, Any], source: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    th = threshold_from_train(train_payload, source)
    comp = force_components(val_payload, source)
    ratio = comp["ratio"]
    fn = comp["Fn"]
    valid = np.isfinite(ratio) & (fn > th["fn_valid_threshold_p10_N"])
    labels = val_payload["current_slip"].numpy().astype(int)
    p_slip = val_payload["slip_probs"].float().numpy()[:, 1]
    pred_slip = p_slip >= 0.5
    high = valid & (ratio >= th["ratio_p80"])
    low = valid & (ratio <= th["ratio_p20"])
    q = sigmoid_np(th["q_alpha"] * (np.clip(ratio, 0, th["ratio_p99"]) - th["q_tau"]))
    groups = ["overall", "flat", "sharp", "sphere"]
    rows: list[dict[str, Any]] = []
    meta_groups = np.array([group_name(m) for m in val_payload["metadata"]])
    for group in groups:
        gm = np.ones(len(labels), dtype=bool) if group == "overall" else (meta_groups == group)
        mask = valid & gm
        hm = high & gm
        lm = low & gm
        slip_pos = mask & (labels == 1)
        no_slip = mask & (labels == 0)
        high_slip_pos = hm & (labels == 1)
        low_no_slip = lm & (labels == 0)
        contradiction = ((hm & (p_slip < 0.5)) | (lm & (p_slip >= 0.5)))
        row = {
            "force_source": source,
            "group": group,
            "n_samples": int(gm.sum()),
            "n_valid_contact": int(mask.sum()),
            "valid_contact_ratio": safe_rate(int(mask.sum()), int(gm.sum())),
            "slip_positive_count": int((gm & (labels == 1)).sum()),
            "slip_positive_ratio_valid": safe_rate(int(slip_pos.sum()), int(mask.sum())),
            "ratio_mean_valid": float(np.mean(ratio[mask])) if mask.any() else None,
            "ratio_median_valid": float(np.median(ratio[mask])) if mask.any() else None,
            "ratio_p80_threshold_train": th["ratio_p80"],
            "ratio_p20_threshold_train": th["ratio_p20"],
            "q_tau_train": th["q_tau"],
            "q_alpha_train": th["q_alpha"],
            "q_mean_valid": float(np.mean(q[mask])) if mask.any() else None,
            "high_ratio_count": int(hm.sum()),
            "high_ratio_slip_rate": safe_rate(int((hm & (labels == 1)).sum()), int(hm.sum())),
            "high_ratio_slip_recall_dataset": safe_rate(int((hm & (labels == 1)).sum()), int(slip_pos.sum())),
            "high_ratio_model_recall_on_slip": safe_rate(int((high_slip_pos & pred_slip).sum()), int(high_slip_pos.sum())),
            "low_ratio_count": int(lm.sum()),
            "low_ratio_slip_rate": safe_rate(int((lm & (labels == 1)).sum()), int(lm.sum())),
            "low_ratio_model_false_alarm_on_no_slip": safe_rate(int((low_no_slip & pred_slip).sum()), int(low_no_slip.sum())),
            "p_slip_ratio_pearson_valid": pearson(ratio[mask], p_slip[mask]) if mask.sum() > 2 else None,
            "p_slip_ratio_spearman_valid": spearman(ratio[mask], p_slip[mask]) if mask.sum() > 2 else None,
            "contradiction_rate_valid": safe_rate(int((contradiction & gm).sum()), int(mask.sum())),
            "contradiction_count": int((contradiction & gm).sum()),
        }
        rows.append(row)
    return th, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_phase5_1(val_payload: dict[str, Any], thresholds: dict[str, Any], out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if plt is None:
        return []
    fig_paths: list[str] = []
    groups = np.array([group_name(m) for m in val_payload["metadata"]])
    p_slip = val_payload["slip_probs"].float().numpy()[:, 1]
    for source, th in thresholds.items():
        comp = force_components(val_payload, source)
        ratio = comp["ratio"]
        valid = np.isfinite(ratio) & (comp["Fn"] > th["fn_valid_threshold_p10_N"])
        r_clip = np.clip(ratio, 0, th["ratio_p99"])
        plt.figure(figsize=(8, 5))
        for g in ["flat", "sharp", "sphere"]:
            m = valid & (groups == g)
            if m.any():
                plt.hist(r_clip[m], bins=50, alpha=0.45, density=True, label=g)
        plt.axvline(th["ratio_p20"], color="gray", linestyle="--", label="train p20")
        plt.axvline(th["ratio_p80"], color="black", linestyle="--", label="train p80/tau")
        plt.xlabel("clipped Ft/Fn")
        plt.ylabel("density")
        plt.title(f"Phase5-1 friction ratio distribution ({source})")
        plt.legend()
        path = out_dir / f"phase5_1_ratio_hist_{source}.png"
        plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()
        fig_paths.append(str(path))

        # Binned relationship between ratio and model slip probability.
        m = valid
        xs = r_clip[m]
        ys = p_slip[m]
        if len(xs) > 10:
            bins = np.quantile(xs, np.linspace(0, 1, 21))
            bins = np.unique(bins)
            mids, means = [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                bm = (xs >= lo) & (xs <= hi)
                if bm.any():
                    mids.append(float((lo + hi) / 2.0)); means.append(float(np.mean(ys[bm])))
            plt.figure(figsize=(7, 4.5))
            plt.plot(mids, means, marker="o")
            plt.axvline(th["ratio_p80"], color="black", linestyle="--", label="train p80/tau")
            plt.xlabel("clipped Ft/Fn bin midpoint")
            plt.ylabel("mean current p_slip")
            plt.title(f"p_slip vs friction ratio ({source})")
            plt.legend()
            path = out_dir / f"phase5_1_p_slip_vs_ratio_{source}.png"
            plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()
            fig_paths.append(str(path))
    return fig_paths


class FeatureHeadDataset(Dataset):
    def __init__(self, payload: dict[str, Any], indices: np.ndarray, aux_names: list[str], horizons: tuple[int, ...]):
        self.payload = payload
        self.indices = np.asarray(indices, dtype=np.int64)
        self.z = payload["z"].float()
        self.aux = payload["aux"].float()
        schema = list(payload["aux_schema"])
        missing = [n for n in aux_names if n not in schema]
        if missing:
            raise ValueError(f"Missing aux columns {missing}; schema={schema}")
        self.aux_idx = [schema.index(n) for n in aux_names]
        all_h = tuple(int(x) for x in payload["horizons"])
        self.h_idx = [all_h.index(int(h)) for h in horizons]
        self.future_slip = payload["future_slip"].float()

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        idx = int(self.indices[i])
        aux = self.aux[idx, self.aux_idx] if self.aux_idx else torch.empty(0, dtype=torch.float32)
        return {
            "x": torch.cat([self.z[idx], aux], dim=0),
            "future_slip": self.future_slip[idx, self.h_idx],
        }


def binary_metrics(y_true: np.ndarray, prob: np.ndarray, prefix: str) -> dict[str, Any]:
    y_true = y_true.astype(int)
    pred = (prob >= 0.5).astype(int)
    out: dict[str, Any] = {
        f"{prefix}_positive_count": int(y_true.sum()),
        f"{prefix}_positive_ratio": float(y_true.mean()) if len(y_true) else 0.0,
        f"{prefix}_f1": float(f1_score(y_true, pred, zero_division=0)) if len(y_true) else None,
        f"{prefix}_accuracy": float(accuracy_score(y_true, pred)) if len(y_true) else None,
        f"{prefix}_precision": float(precision_score(y_true, pred, zero_division=0)) if len(y_true) else None,
        f"{prefix}_recall": float(recall_score(y_true, pred, zero_division=0)) if len(y_true) else None,
    }
    if len(np.unique(y_true)) > 1:
        out[f"{prefix}_auroc"] = float(roc_auc_score(y_true, prob))
        out[f"{prefix}_auprc"] = float(average_precision_score(y_true, prob))
    else:
        out[f"{prefix}_auroc"] = None
        out[f"{prefix}_auprc"] = None
    return out


def calibration_error(y_true_stable: np.ndarray, p_stable: np.ndarray, bins: int = 10) -> float:
    return wm.calibration_error(y_true_stable.astype(int), p_stable, bins=bins)


def risk_arrays(payload: dict[str, Any], thresholds: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
    comp = force_components(payload, "pred")
    ratio = comp["ratio"]
    if thresholds is None:
        fn_th = np.quantile(comp["Fn"][np.isfinite(comp["Fn"])], 0.10)
        ref = ratio[(comp["Fn"] > fn_th) & np.isfinite(ratio)]
        p20, p80, p99 = np.quantile(ref, [0.2, 0.8, 0.99])
        alpha = 8.0 / max(float(p80 - p20), EPS)
        thresholds = {"ratio_p20": float(p20), "ratio_p80": float(p80), "ratio_p99": float(p99), "q_tau": float(p80), "q_alpha": alpha, "fn_valid_threshold_p10_N": float(fn_th)}
    r_clip = np.clip(ratio, 0, thresholds["ratio_p99"])
    q = sigmoid_np(thresholds["q_alpha"] * (r_clip - thresholds["q_tau"]))
    return {"ratio": ratio, "r_clipped": r_clip, "q": q, "valid_contact": comp["Fn"] > thresholds["fn_valid_threshold_p10_N"]}


def trajectory_delta(values: np.ndarray, metadata: list[dict[str, Any]]) -> np.ndarray:
    delta = np.zeros_like(values, dtype=np.float32)
    groups: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for i, meta in enumerate(metadata):
        key = (str(meta.get("dataset")), str(meta.get("trajectory")))
        groups.setdefault(key, []).append((int(meta.get("sample", i)), i))
    for rows in groups.values():
        rows = sorted(rows)
        prev_i: int | None = None
        for _sample, i in rows:
            if prev_i is None:
                delta[i] = 0.0
            else:
                delta[i] = float(values[i] - values[prev_i])
            prev_i = i
    return delta


def augment_payload(payload: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    aux = payload["aux"].float()
    schema = list(payload["aux_schema"])
    ra = risk_arrays(payload, thresholds)
    dr = trajectory_delta(ra["r_clipped"].astype(np.float32), payload["metadata"])
    dq = trajectory_delta(ra["q"].astype(np.float32), payload["metadata"])
    dr_clip = np.clip(dr, -thresholds["ratio_p99"], thresholds["ratio_p99"])
    add_schema = ["friction_r_clipped_pred", "friction_dr_clipped_pred", "friction_q_pred", "friction_dq_pred"]
    add = torch.tensor(np.stack([ra["r_clipped"], dr_clip, ra["q"], dq], axis=1), dtype=torch.float32)
    for name in add_schema:
        if name in schema:
            raise ValueError(f"Augmented schema already contains {name}")
    out["aux"] = torch.cat([aux, add], dim=1)
    out["aux_schema"] = schema + add_schema
    out["friction_thresholds_pred_train"] = thresholds
    out["created_at"] = datetime.now().isoformat(timespec="seconds")
    return out


def save_augmented_features(src_run_id: str, out_run_id: str, thresholds: dict[str, Any]) -> dict[str, Any]:
    run_dir = PHASE5_RUN_ROOT / out_run_id
    feat_dir = run_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    infos = {}
    for split in ["train", "val"]:
        src = load_feature(PHASE4_RUN_ROOT, src_run_id, split)
        aug = augment_payload(src, thresholds)
        path = feat_dir / f"{split}_features.pt"
        torch.save(aug, path)
        infos[split] = {"split": split, "path": str(path), "n_samples": int(aug["z"].shape[0]), "cached": False}
    src_manifest = read_json(manifest_path(PHASE4_RUN_ROOT, src_run_id))
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": out_run_id,
        "source_phase4_run_id": src_run_id,
        "feature_kind": "decoupled_joint_friction_augmented",
        "encoder": src_manifest.get("encoder", "mae"),
        "checkpoint": src_manifest.get("checkpoint"),
        "horizons": src_manifest.get("horizons", list(DEFAULT_HORIZONS)),
        "feature_schema": {"latent": src_manifest.get("feature_schema", {}).get("latent"), "aux": torch.load(infos["train"]["path"], map_location="cpu", weights_only=False)["aux_schema"]},
        "train": infos["train"],
        "val": infos["val"],
        "friction_thresholds_pred_train": thresholds,
        "raw_data_modified": False,
        "derived_from": str(PHASE4_RUN_ROOT / src_run_id),
    }
    write_json(feat_dir / "feature_manifest.json", manifest)
    return manifest


def subset_indices_by_groups(payload: dict[str, Any], include: Iterable[str]) -> np.ndarray:
    include_set = set(include)
    groups = np.array([group_name(m) for m in payload["metadata"]])
    return np.flatnonzero(np.isin(groups, list(include_set))).astype(np.int64)


def all_indices(payload: dict[str, Any]) -> np.ndarray:
    return np.arange(int(payload["z"].shape[0]), dtype=np.int64)


def model_predictions(model: nn.Module, loader: DataLoader, dev: torch.device) -> np.ndarray:
    model.eval()
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["x"].to(dev))
            stable = torch.sigmoid(logits).detach().cpu().numpy()
            probs.append(stable)
    return np.concatenate(probs, axis=0) if probs else np.zeros((0, 0), dtype=float)


def evaluate_predictions(payload: dict[str, Any], indices: np.ndarray, stable_prob: np.ndarray, horizons: tuple[int, ...], thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
    future = payload["future_slip"].numpy()[indices]
    all_h = tuple(int(h) for h in payload["horizons"])
    h_idx = [all_h.index(int(h)) for h in horizons]
    future = future[:, h_idx]
    current_slip = payload["current_slip"].numpy()[indices].astype(int)
    metadata = [payload["metadata"][int(i)] for i in indices]
    ra = risk_arrays(payload, thresholds)
    ratio = ra["ratio"][indices]
    valid_contact = ra["valid_contact"][indices]
    th = thresholds or {}
    high = valid_contact & (ratio >= th.get("ratio_p80", np.nanquantile(ratio, 0.8)))
    low = valid_contact & (ratio <= th.get("ratio_p20", np.nanquantile(ratio, 0.2)))
    out: dict[str, Any] = {"n_samples": int(len(indices)), "horizons": list(horizons)}
    for j, h in enumerate(horizons):
        y_stable = 1 - future[:, j].astype(int)
        p_stable = stable_prob[:, j]
        y_slip = 1 - y_stable
        p_inst = 1.0 - p_stable
        pred_inst = p_inst >= 0.5
        high_pos = high & (y_slip == 1)
        low_neg = low & (y_slip == 0)
        row = {
            **binary_metrics(y_slip, p_inst, "future_slip"),
            **binary_metrics(y_stable, p_stable, "future_stability"),
            "stability_calibration_error": calibration_error(y_stable, p_stable),
            "high_risk_count": int(high.sum()),
            "low_risk_count": int(low.sum()),
            "high_risk_future_slip_rate": safe_rate(int((high & (y_slip == 1)).sum()), int(high.sum())),
            "high_risk_model_recall_on_future_slip": safe_rate(int((high_pos & pred_inst).sum()), int(high_pos.sum())),
            "low_risk_model_false_alarm_on_future_stable": safe_rate(int((low_neg & pred_inst).sum()), int(low_neg.sum())),
        }
        if h == 1:
            row.update(wm.lead_time_to_slip(metadata, current_slip, p_inst))
        out[f"H{h}"] = row
    return out


def current_metrics_for_subset(payload: dict[str, Any], indices: np.ndarray) -> dict[str, Any]:
    sub = {
        "force_gt_n": payload["force_gt_n"].numpy()[indices],
        "force_pred_n": payload["force_pred_n"].numpy()[indices],
        "label_gt": payload["current_slip"].numpy()[indices].astype(int),
        "slip_probs": payload["slip_probs"].numpy()[indices],
    }
    ref = p2.collect_force_ratio_reference(slip_horizon=0)
    return p2.arrays_to_summary(sub, ref)


def train_future_head(
    run_root: Path,
    feature_run_id: str,
    experiment_name: str,
    aux_names: list[str],
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    horizons: tuple[int, ...],
    max_epochs: int,
    batch_size: int,
    lr: float,
    hidden_dim: int,
    dropout: float,
    seed: int,
    wandb_mode: str,
    report_dir: Path,
    thresholds: dict[str, Any] | None,
    condition: str,
) -> dict[str, Any]:
    set_seed(seed)
    run_dir = run_root / feature_run_id
    train_payload = torch.load(run_dir / "features/train_features.pt", map_location="cpu", weights_only=False)
    val_payload = torch.load(run_dir / "features/val_features.pt", map_location="cpu", weights_only=False)
    train_ds = FeatureHeadDataset(train_payload, train_indices, aux_names, horizons)
    val_ds = FeatureHeadDataset(val_payload, val_indices, aux_names, horizons)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    dev = device()
    input_dim = int(train_ds.z.shape[1] + len(aux_names))
    model = wm.MLP(input_dim, hidden_dim, len(horizons), dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.0e-4)
    head_dir = run_dir / "heads" / experiment_name
    ckpt_dir = head_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_MODE"] = wandb_mode
    wb = None
    if wandb_mode != "disabled":
        wb_base = os.environ.get("WANDB_NAME")
        wb_name = f"{wb_base}_{condition}" if wb_base else experiment_name
        wb = wandb.init(
            project="sparsh-finetune-tactile-grasp",
            entity="junyuzhuzjy-zhejiang-university",
            dir=str(head_dir),
            id=wb_name,
            name=wb_name,
            group="phase5_friction_stability",
            tags=["sparsh", "tactile_grasp", "force-slip", "phase5", condition],
            notes="Phase5 friction-aware future stability head; target is future slip/instability, not grasp success.",
            config={
                "feature_run_id": feature_run_id,
                "condition": condition,
                "input_schema": ["z_t_mean_pooled"] + aux_names,
                "horizons": list(horizons),
                "seed": seed,
                "train_count": int(len(train_indices)),
                "val_count": int(len(val_indices)),
                "raw_data_modified": False,
            },
        )
    start = time.time()
    best_score = -math.inf
    best_record: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            logits = model(batch["x"].to(dev))
            target = 1.0 - batch["future_slip"].to(dev)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        stable_prob = model_predictions(model, val_loader, dev)
        metrics = evaluate_predictions(val_payload, val_indices, stable_prob, horizons, thresholds)
        scores = []
        for h in horizons:
            v = metrics[f"H{h}"].get("future_slip_auprc")
            if v is not None:
                scores.append(float(v))
        score = float(np.mean(scores)) if scores else float(metrics[f"H{horizons[0]}"].get("future_slip_f1") or 0.0)
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val": metrics, "selection_score": score}
        history.append(rec)
        if wb is not None:
            log = {"epoch": epoch, "train/loss": rec["train_loss"], "val/selection_score": score}
            for h in horizons:
                row = metrics[f"H{h}"]
                log[f"val/H{h}_future_slip_f1"] = row.get("future_slip_f1")
                log[f"val/H{h}_future_slip_auprc"] = row.get("future_slip_auprc")
                log[f"val/H{h}_future_slip_auroc"] = row.get("future_slip_auroc")
                log[f"val/H{h}_ece"] = row.get("stability_calibration_error")
                log[f"val/H{h}_high_risk_recall"] = row.get("high_risk_model_recall_on_future_slip")
            wb.log({k: v for k, v in log.items() if v is not None}, step=epoch)
        if score > best_score:
            best_score = score
            best_record = rec
            torch.save({
                "feature_run_id": feature_run_id,
                "condition": condition,
                "experiment_name": experiment_name,
                "aux_names": aux_names,
                "model_state": model.state_dict(),
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "horizons": list(horizons),
                "epoch": epoch,
                "metrics": metrics,
                "thresholds": thresholds,
                "seed": seed,
            }, ckpt_dir / "best.pth")
        write_json(head_dir / "history.json", history)
    manifest = read_json(run_dir / "features/feature_manifest.json") if (run_dir / "features/feature_manifest.json").exists() else {}
    val_payload = torch.load(run_dir / "features/val_features.pt", map_location="cpu", weights_only=False)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "phase": "phase5_future_head",
        "condition": condition,
        "feature_run_id": feature_run_id,
        "feature_root": str(run_root),
        "experiment_name": experiment_name,
        "input_schema": ["z_t_mean_pooled"] + aux_names,
        "train_count": int(len(train_indices)),
        "val_count": int(len(val_indices)),
        "horizons": list(horizons),
        "best_epoch": best_record["epoch"] if best_record else None,
        "best_selection_score_mean_auprc": best_score,
        "best_val": best_record["val"] if best_record else None,
        "current_metrics_val_subset": current_metrics_for_subset(val_payload, val_indices),
        "feature_manifest": str(run_dir / "features/feature_manifest.json"),
        "checkpoint": str(ckpt_dir / "best.pth"),
        "wall_time_seconds": time.time() - start,
        "wandb_mode": wandb_mode,
        "raw_data_modified": False,
        "manifest_feature_kind": manifest.get("feature_kind"),
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / f"{experiment_name}_report.json", summary)
    (report_dir / f"{experiment_name}_report.md").write_text(render_head_report(summary), encoding="utf-8")
    if wb is not None:
        def _wandb_timeout(_signum, _frame):
            raise TimeoutError("wandb.finish timed out")
        old_handler = signal.signal(signal.SIGALRM, _wandb_timeout)
        signal.alarm(30)
        try:
            wb.finish()
            summary["wandb_finish_status"] = "finished"
        except Exception as exc:
            summary["wandb_finish_status"] = f"best_effort_timeout_or_error: {exc}"
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            write_json(report_dir / f"{experiment_name}_report.json", summary)
    return summary


def render_head_report(s: dict[str, Any]) -> str:
    lines = [
        f"# Phase5 Head Report: {s['condition']}", "",
        f"- generated_at: `{s['generated_at']}`",
        f"- feature_run_id: `{s['feature_run_id']}`",
        f"- experiment_name: `{s['experiment_name']}`",
        f"- input_schema: `{s['input_schema']}`",
        f"- train_count / val_count: `{s['train_count']}` / `{s['val_count']}`",
        f"- best_epoch: `{s['best_epoch']}`",
        f"- selection_score_mean_auprc: `{fmt(s['best_selection_score_mean_auprc'])}`",
        f"- raw_data_modified: `{s['raw_data_modified']}`",
        "", "| horizon | F1 | AUROC | AUPRC | ECE | high-risk recall | low-risk false alarm | lead mean |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    best = s.get("best_val") or {}
    for h in best.get("horizons", []):
        row = best.get(f"H{h}", {})
        lines.append(f"| {h} | {fmt(row.get('future_slip_f1'))} | {fmt(row.get('future_slip_auroc'))} | {fmt(row.get('future_slip_auprc'))} | {fmt(row.get('stability_calibration_error'))} | {fmt(row.get('high_risk_model_recall_on_future_slip'))} | {fmt(row.get('low_risk_model_false_alarm_on_future_stable'))} | {fmt(row.get('lead_time_to_slip_onset_steps_mean'))} |")
    cur = s.get("current_metrics_val_subset", {})
    lines += ["", "## Current-task subset metrics", "", f"- force_rmse_mean_N: `{fmt(cur.get('force_rmse_mean_N'))}`", f"- slip_f1: `{fmt(cur.get('slip_f1'))}`", "", "## Artifacts", "", f"- checkpoint: `{s['checkpoint']}`", f"- feature_manifest: `{s['feature_manifest']}`"]
    return "\n".join(lines) + "\n"

def render_phase5_1(payload: dict[str, Any]) -> str:
    lines = ["# Phase5-1 Friction-aware Diagnostic", "", f"- generated_at: `{payload['generated_at']}`", f"- source_feature_run_id: `{payload['source_feature_run_id']}`", "- raw_data_modified: `False`", "", "## Thresholds frozen from train split", ""]
    for src, th in payload["thresholds"].items():
        lines += [f"### {src}", "", f"- valid contact: `{th['valid_contact_rule']}` = `{fmt(th['fn_valid_threshold_p10_N'], 6)} N`", f"- Ft/Fn p20/p50/p80/p99: `{fmt(th['ratio_p20'])}` / `{fmt(th['ratio_p50'])}` / `{fmt(th['ratio_p80'])}` / `{fmt(th['ratio_p99'])}`", f"- q = sigmoid(alpha * (clipped(Ft/Fn)-tau)), tau=`{fmt(th['q_tau'])}`, alpha=`{fmt(th['q_alpha'])}`", ""]
    lines += ["## Diagnostic table", "", "| source | group | valid n | slip+ valid | high ratio n | high slip rate | high recall(dataset) | high model recall | low false alarm | p_slip~ratio Spearman | contradiction |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in payload["rows"]:
        lines.append(f"| {r['force_source']} | {r['group']} | {r['n_valid_contact']} | {fmt(r.get('slip_positive_ratio_valid'))} | {r['high_ratio_count']} | {fmt(r.get('high_ratio_slip_rate'))} | {fmt(r.get('high_ratio_slip_recall_dataset'))} | {fmt(r.get('high_ratio_model_recall_on_slip'))} | {fmt(r.get('low_ratio_model_false_alarm_on_no_slip'))} | {fmt(r.get('p_slip_ratio_spearman_valid'))} | {fmt(r.get('contradiction_rate_valid'))} |")
    lines += ["", "## Interpretation", "", payload["interpretation"], "", "## Artifacts", ""]
    for pp in payload.get("figures", []):
        lines.append(f"- figure: `{pp}`")
    lines.append(f"- CSV: `{payload['csv_path']}`")
    return "\n".join(lines) + "\n"

def phase5_1(args: argparse.Namespace) -> None:
    report_dir = PHASE5_REPORT_ROOT / "phase5_1_friction_diagnostic"
    fig_dir = report_dir / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)
    train = load_feature(PHASE4_RUN_ROOT, args.decoupled_run_id, "train")
    val = load_feature(PHASE4_RUN_ROOT, args.decoupled_run_id, "val")
    thresholds: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for src in ["gt", "pred"]:
        th, src_rows = diagnostic_rows_for_source(train, val, src)
        thresholds[src] = th
        rows.extend(src_rows)
    csv_path = report_dir / "phase5_1_friction_diagnostic_rows.csv"
    write_csv(csv_path, rows)
    figures = plot_phase5_1(val, thresholds, fig_dir)
    pred_overall = next(r for r in rows if r["force_source"] == "pred" and r["group"] == "overall")
    gt_overall = next(r for r in rows if r["force_source"] == "gt" and r["group"] == "overall")
    interp = (
        f"Predicted-force Ft/Fn has Spearman correlation {fmt(pred_overall.get('p_slip_ratio_spearman_valid'))} with current p_slip; "
        f"ground-truth-force Ft/Fn has Spearman correlation {fmt(gt_overall.get('p_slip_ratio_spearman_valid'))}. "
        "The proxy is therefore treated as a physical diagnostic/conditioning cue rather than a hard friction law."
    )
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "phase": "phase5_1_friction_diagnostic",
        "source_feature_run_id": args.decoupled_run_id,
        "source_feature_root": str(PHASE4_RUN_ROOT / args.decoupled_run_id),
        "threshold_source": "train split only",
        "thresholds": thresholds,
        "rows": rows,
        "figures": figures,
        "csv_path": str(csv_path),
        "interpretation": interp,
        "raw_data_modified": False,
    }
    json_path = report_dir / "phase5_1_friction_diagnostic_report.json"
    md_path = report_dir / "phase5_1_friction_diagnostic_report.md"
    payload["json_path"] = str(json_path); payload["md_path"] = str(md_path)
    write_json(json_path, payload)
    md_path.write_text(render_phase5_1(payload), encoding="utf-8")
    print(md_path)


def render_comparison(title: str, payload: dict[str, Any]) -> str:
    lines = [f"# {title}", "", f"- generated_at: `{payload['generated_at']}`", f"- raw_data_modified: `{payload['raw_data_modified']}`", "", "## Results", "", "| condition | train n | val/test n | H1 F1 | H1 AUROC | H1 AUPRC | H1 ECE | H1 high recall | H3 F1 | H3 AUPRC | H5 F1 | H5 AUPRC | current force RMSE | current slip F1 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for ss in payload["conditions"]:
        best = ss.get("best_val") or {}
        h1, h3, h5 = best.get("H1", {}), best.get("H3", {}), best.get("H5", {})
        cur = ss.get("current_metrics_val_subset", {})
        lines.append(f"| {ss['condition']} | {ss.get('train_count')} | {ss.get('val_count')} | {fmt(h1.get('future_slip_f1'))} | {fmt(h1.get('future_slip_auroc'))} | {fmt(h1.get('future_slip_auprc'))} | {fmt(h1.get('stability_calibration_error'))} | {fmt(h1.get('high_risk_model_recall_on_future_slip'))} | {fmt(h3.get('future_slip_f1'))} | {fmt(h3.get('future_slip_auprc'))} | {fmt(h5.get('future_slip_f1'))} | {fmt(h5.get('future_slip_auprc'))} | {fmt(cur.get('force_rmse_mean_N'))} | {fmt(cur.get('slip_f1'))} |")
    lines += ["", "## Recommendation", "", payload.get("recommendation", ""), "", "## Notes", ""]
    for note in payload.get("notes", []):
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"

def best_by_metric(conditions: list[dict[str, Any]], metric_path: tuple[str, ...]) -> dict[str, Any] | None:
    best = None; best_val = -math.inf
    for c in conditions:
        cur: Any = c
        for k in metric_path:
            cur = cur.get(k, {}) if isinstance(cur, dict) else {}
        try:
            v = float(cur)
        except Exception:
            continue
        if v > best_val:
            best_val = v; best = c
    return best


def phase5_2(args: argparse.Namespace) -> None:
    stamp = args.stamp or now_stamp()
    report_dir = PHASE5_REPORT_ROOT / "phase5_2_friction_future_head"
    report_dir.mkdir(parents=True, exist_ok=True)
    train = load_feature(PHASE4_RUN_ROOT, args.decoupled_run_id, "train")
    thresholds = threshold_from_train(train, "pred")
    out_run_id = args.feature_run_id or f"phase5_2_friction_features_{stamp}"
    if not manifest_path(PHASE5_RUN_ROOT, out_run_id).exists():
        save_augmented_features(args.decoupled_run_id, out_run_id, thresholds)
    train_aug = load_feature(PHASE5_RUN_ROOT, out_run_id, "train")
    val_aug = load_feature(PHASE5_RUN_ROOT, out_run_id, "val")
    train_idx = all_indices(train_aug); val_idx = all_indices(val_aug)
    conditions = []
    for condition, aux_names in FRICTION_MODES.items():
        exp = f"phase5_2_{condition}_{stamp}"
        s = train_future_head(PHASE5_RUN_ROOT, out_run_id, exp, aux_names, train_idx, val_idx, tuple(args.horizons), args.max_epochs, args.train_batch_size, args.lr, args.hidden_dim, args.dropout, args.seed, args.wandb_mode, report_dir, thresholds, condition)
        conditions.append(s)
    best_h3 = best_by_metric(conditions, ("best_val", "H3", "future_slip_auprc"))
    baseline = next(c for c in conditions if c["condition"] == "full_dynamics_baseline")
    rec = f"Recommended condition by H3 AUPRC: `{best_h3['condition'] if best_h3 else 'n/a'}`. Baseline H3 F1={fmt((baseline.get('best_val') or {}).get('H3', {}).get('future_slip_f1'))}; best H3 F1={fmt(((best_h3 or {}).get('best_val') or {}).get('H3', {}).get('future_slip_f1'))}."
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "phase": "phase5_2_friction_future_head",
        "stamp": stamp,
        "source_phase4_run_id": args.decoupled_run_id,
        "phase5_feature_run_id": out_run_id,
        "thresholds_pred_train": thresholds,
        "conditions": conditions,
        "recommendation": rec,
        "notes": [
            "Phase4 full dynamics already included a raw Ft/Fn column; Phase5 adds train-clipped r, temporal delta-r, calibrated q, and delta-q to test whether friction-aware conditioning adds value beyond the existing ratio.",
            "Selection uses mean AUPRC across H1/H3/H5 while the report highlights H3/H5 because longer-horizon warning is the paper-facing target.",
        ],
        "raw_data_modified": False,
    }
    json_path = report_dir / "phase5_2_friction_future_head_report.json"
    md_path = report_dir / "phase5_2_friction_future_head_report.md"
    payload["json_path"] = str(json_path); payload["md_path"] = str(md_path)
    write_json(json_path, payload)
    md_path.write_text(render_comparison("Phase5-2 Friction Proxy Future Stability Head", payload), encoding="utf-8")
    print(md_path)


def phase5_3(args: argparse.Namespace) -> None:
    stamp = args.stamp or now_stamp()
    report_dir = PHASE5_REPORT_ROOT / "phase5_3_heldout_generalization"
    report_dir.mkdir(parents=True, exist_ok=True)
    # Ensure decoupled friction features exist/reuse best known phase5-2 feature run.
    train_dec = load_feature(PHASE4_RUN_ROOT, args.decoupled_run_id, "train")
    thresholds = threshold_from_train(train_dec, "pred")
    friction_run_id = args.friction_feature_run_id or f"phase5_3_friction_features_{stamp}"
    if not manifest_path(PHASE5_RUN_ROOT, friction_run_id).exists():
        save_augmented_features(args.decoupled_run_id, friction_run_id, thresholds)
    specs = [
        ("separate_late_fusion", PHASE4_RUN_ROOT, args.separate_run_id, STATIC_FORCE_SLIP),
        ("decoupled_static", PHASE4_RUN_ROOT, args.decoupled_run_id, STATIC_FORCE_SLIP),
        ("decoupled_dynamics", PHASE4_RUN_ROOT, args.decoupled_run_id, BASE_FULL),
        ("decoupled_dynamics_friction", PHASE5_RUN_ROOT, friction_run_id, FRICTION_MODES[args.friction_mode]),
    ]
    conditions = []
    split_desc = {"train_groups": args.train_groups, "test_groups": args.test_groups}
    for condition, root, run_id, aux_names in specs:
        train_payload = load_feature(root, run_id, "train")
        val_payload = load_feature(root, run_id, "val")
        train_idx = subset_indices_by_groups(train_payload, args.train_groups)
        val_idx = subset_indices_by_groups(val_payload, args.test_groups)
        exp = f"phase5_3_{condition}_{stamp}"
        s = train_future_head(root, run_id, exp, aux_names, train_idx, val_idx, tuple(args.horizons), args.max_epochs, args.train_batch_size, args.lr, args.hidden_dim, args.dropout, args.seed, args.wandb_mode, report_dir, thresholds if "friction" in condition or "decoupled" in condition else None, condition)
        s["heldout_split"] = split_desc
        conditions.append(s)
    best_h5 = best_by_metric(conditions, ("best_val", "H5", "future_slip_auprc"))
    rec = f"Held-out split train={args.train_groups}, test={args.test_groups}. Best H5 AUPRC condition: `{best_h5['condition'] if best_h5 else 'n/a'}`. Interpret this as contact-geometry stress testing, not final robot deployment evidence."
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "phase": "phase5_3_heldout_generalization",
        "stamp": stamp,
        "heldout_split": split_desc,
        "conditions": conditions,
        "recommendation": rec,
        "notes": ["Training uses only train split samples from the selected train groups; evaluation uses val split samples from held-out groups, avoiding test-threshold tuning.", "No raw datasets are modified; generated heads/checkpoints are under /vla1/zjy/sparsh_runs/force_slip_phase5 when friction features are used and Phase4 feature roots otherwise."],
        "raw_data_modified": False,
    }
    json_path = report_dir / "phase5_3_heldout_generalization_report.json"
    md_path = report_dir / "phase5_3_heldout_generalization_report.md"
    payload["json_path"] = str(json_path); payload["md_path"] = str(md_path)
    write_json(json_path, payload)
    md_path.write_text(render_comparison("Phase5-3 Held-out Contact Generalization", payload), encoding="utf-8")
    print(md_path)


def load_best_phase5_2() -> dict[str, Any]:
    path = PHASE5_REPORT_ROOT / "phase5_2_friction_future_head" / "phase5_2_friction_future_head_report.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return read_json(path)


def choose_visual_cases(payload: dict[str, Any], stable_prob: np.ndarray, thresholds: dict[str, Any], max_cases: int = 5) -> list[dict[str, Any]]:
    meta = payload["metadata"]
    current_slip = payload["current_slip"].numpy().astype(int)
    future = payload["future_slip"].numpy()[:, 0].astype(int)
    p_inst = 1.0 - stable_prob[:, 0]
    p_slip = payload["slip_probs"].numpy()[:, 1]
    ra = risk_arrays(payload, thresholds)
    r = ra["r_clipped"]
    groups = np.array([group_name(m) for m in meta])
    cases: list[tuple[str, int]] = []
    # success early warning: future slip positive, current not slipping, high predicted future risk.
    idxs = np.flatnonzero((future == 1) & (current_slip == 0) & (p_inst >= 0.5))
    if len(idxs): cases.append(("success_early_warning", int(idxs[np.argmax(p_inst[idxs])])))
    for g, label in [("sharp", "sharp_contact_case"), ("sphere", "sphere_contact_case")]:
        idxs = np.flatnonzero((groups == g) & (future == 1))
        if len(idxs): cases.append((label, int(idxs[np.argmax(p_inst[idxs])])))
    idxs = np.flatnonzero((current_slip == 0) & (r >= thresholds["ratio_p80"]))
    if len(idxs):
        cases.append(("high_force_no_current_slip", int(idxs[np.argmax(r[idxs])])))
    else:
        # In this split the train-derived high Ft/Fn region may be almost all slip;
        # still include a force-high/no-current-slip counterexample by Fmag.
        fmag = force_components(payload, "pred")["Fmag"]
        idxs = np.flatnonzero(current_slip == 0)
        if len(idxs):
            cases.append(("high_force_no_current_slip", int(idxs[np.argmax(fmag[idxs])])))
    idxs = np.flatnonzero((current_slip == 1) & (r <= thresholds["ratio_p20"]))
    if len(idxs): cases.append(("slip_with_low_friction_ratio_failure", int(idxs[0])))
    # Deduplicate by trajectory and keep requested labels.
    seen = set(); out = []
    for label, i in cases:
        key = (meta[i]["dataset"], meta[i]["trajectory"])
        if (label, key) in seen:
            continue
        seen.add((label, key))
        out.append({"label": label, "index": i, "dataset": meta[i]["dataset"], "trajectory": meta[i]["trajectory"], "sample": int(meta[i]["sample"])})
        if len(out) >= max_cases:
            break
    return out


def plot_trajectory_case(payload: dict[str, Any], stable_prob: np.ndarray, thresholds: dict[str, Any], case: dict[str, Any], out_dir: Path) -> str | None:
    if plt is None:
        return None
    meta = payload["metadata"]
    key = (case["dataset"], case["trajectory"])
    rows = [(i, int(m["sample"])) for i, m in enumerate(meta) if m["dataset"] == key[0] and str(m["trajectory"]) == str(key[1])]
    if not rows:
        return None
    rows = sorted(rows, key=lambda x: x[1])
    idx = np.array([i for i, _s in rows], dtype=int)
    x = np.array([s for _i, s in rows], dtype=int)
    current_slip = payload["current_slip"].numpy()[idx]
    p_slip = payload["slip_probs"].numpy()[idx, 1]
    p_inst = 1.0 - stable_prob[idx]
    comp = force_components(payload, "pred")
    fn, ft, fmag = comp["Fn"][idx], comp["Ft"][idx], comp["Fmag"][idx]
    r = np.clip(comp["ratio"][idx], 0, thresholds["ratio_p99"])
    q = sigmoid_np(thresholds["q_alpha"] * (r - thresholds["q_tau"]))
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(x, current_slip, label="GT current slip", color="black", linewidth=1.5)
    axes[0].plot(x, p_slip, label="current p_slip", color="tab:blue")
    axes[0].plot(x, p_inst[:, 0], label="future instability H1", color="tab:red")
    if p_inst.shape[1] > 1:
        axes[0].plot(x, p_inst[:, 1], label="future instability H3", color="tab:orange", alpha=0.8)
    if p_inst.shape[1] > 2:
        axes[0].plot(x, p_inst[:, 2], label="future instability H5", color="tab:purple", alpha=0.8)
    axes[0].set_ylim(-0.05, 1.05); axes[0].legend(loc="upper right", fontsize=8); axes[0].set_ylabel("prob/label")
    axes[1].plot(x, r, label="clipped Ft/Fn", color="tab:green")
    axes[1].plot(x, q, label="friction q", color="tab:brown")
    axes[1].axhline(thresholds["ratio_p80"], linestyle="--", color="gray", label="train p80")
    axes[1].legend(loc="upper right", fontsize=8); axes[1].set_ylabel("friction")
    axes[2].plot(x, fn, label="Fn pred N"); axes[2].plot(x, ft, label="Ft pred N"); axes[2].plot(x, fmag, label="Fmag pred N")
    axes[2].legend(loc="upper right", fontsize=8); axes[2].set_ylabel("force N"); axes[2].set_xlabel("sample index")
    fig.suptitle(f"{case['label']} | {case['dataset']} | traj={case['trajectory']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = f"{case['label']}_{str(case['dataset']).replace('/', '_')}_{str(case['trajectory']).replace('/', '_')}.png"[:220]
    path = out_dir / safe
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close(fig)
    return str(path)


def phase5_4(args: argparse.Namespace) -> None:
    report_dir = PHASE5_REPORT_ROOT / "phase5_4_visualization_failure_cases"
    fig_dir = report_dir / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)
    p52 = load_best_phase5_2()
    # Pick recommended/best condition by H3 AUPRC, falling back to final condition.
    best = best_by_metric(p52["conditions"], ("best_val", "H3", "future_slip_auprc")) or p52["conditions"][-1]
    ckpt = torch.load(best["checkpoint"], map_location="cpu", weights_only=False)
    run_root = Path(best["feature_root"])
    run_id = best["feature_run_id"]
    payload = torch.load(run_root / run_id / "features/val_features.pt", map_location="cpu", weights_only=False)
    thresholds = ckpt.get("thresholds") or p52.get("thresholds_pred_train") or threshold_from_train(torch.load(run_root / run_id / "features/train_features.pt", map_location="cpu", weights_only=False), "pred")
    aux_names = ckpt["aux_names"]
    indices = all_indices(payload)
    ds = FeatureHeadDataset(payload, indices, aux_names, tuple(ckpt["horizons"]))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = wm.MLP(int(ds.z.shape[1] + len(aux_names)), int(ckpt.get("hidden_dim", args.hidden_dim)), len(ckpt["horizons"]), float(ckpt.get("dropout", args.dropout)))
    model.load_state_dict(ckpt["model_state"])
    model.to(device())
    stable_prob = model_predictions(model, loader, device())
    cases = choose_visual_cases(payload, stable_prob, thresholds)
    figures = []
    for case in cases:
        path = plot_trajectory_case(payload, stable_prob, thresholds, case, fig_dir)
        case["figure"] = path
        if path:
            figures.append(path)
    payload_out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "phase": "phase5_4_visualization_failure_cases",
        "selected_condition": best["condition"],
        "checkpoint": best["checkpoint"],
        "feature_run_id": run_id,
        "cases": cases,
        "figures": figures,
        "notes": ["Trajectory figures show current slip probability, future instability probabilities, friction ratio/risk, and predicted force components on the same temporal axis.", "Failure cases are intentionally included to avoid overclaiming a hard friction law; the proxy is used as an interpretable stability cue."],
        "raw_data_modified": False,
    }
    json_path = report_dir / "phase5_4_visualization_failure_cases_report.json"
    md_path = report_dir / "phase5_4_visualization_failure_cases_report.md"
    payload_out["json_path"] = str(json_path); payload_out["md_path"] = str(md_path)
    write_json(json_path, payload_out)
    lines = ["# Phase5-4 Visualization and Failure Cases", "", f"- generated_at: `{payload_out['generated_at']}`", f"- selected_condition: `{best['condition']}`", f"- checkpoint: `{best['checkpoint']}`", "- raw_data_modified: `False`", "", "## Cases", "", "| label | dataset | trajectory | sample | figure |", "|---|---|---|---:|---|"]
    for c in cases:
        lines.append(f"| {c['label']} | {c['dataset']} | {c['trajectory']} | {c['sample']} | `{c.get('figure')}` |")
    lines += ["", "## Interpretation", "", "These plots translate Phase5 metrics into paper-facing examples: successful early warnings, difficult contact geometries, and counterexamples where friction ratio alone is insufficient."]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path)


def phase5_summary(args: argparse.Namespace) -> None:
    report_dir = PHASE5_REPORT_ROOT
    p51 = read_json(report_dir / "phase5_1_friction_diagnostic/phase5_1_friction_diagnostic_report.json")
    p52 = read_json(report_dir / "phase5_2_friction_future_head/phase5_2_friction_future_head_report.json")
    p53 = read_json(report_dir / "phase5_3_heldout_generalization/phase5_3_heldout_generalization_report.json")
    p54 = read_json(report_dir / "phase5_4_visualization_failure_cases/phase5_4_visualization_failure_cases_report.json")
    best52 = best_by_metric(p52["conditions"], ("best_val", "H3", "future_slip_auprc")) or p52["conditions"][-1]
    best53 = best_by_metric(p53["conditions"], ("best_val", "H5", "future_slip_auprc")) or p53["conditions"][-1]
    pred_overall = next(r for r in p51["rows"] if r["force_source"] == "pred" and r["group"] == "overall")
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "phase": "phase5_summary",
        "phase_reports": {"phase5_1": p51["md_path"], "phase5_2": p52["md_path"], "phase5_3": p53["md_path"], "phase5_4": p54["md_path"]},
        "recommended_paper_method": f"MAE decoupled multitask + dynamics future head with friction-aware conditioning ({best52['condition']}) if reporting Phase5-2, plus held-out stress test best={best53['condition']}.",
        "safe_claims": [
            f"Predicted-force Ft/Fn is an interpretable stability cue with p_slip Spearman={fmt(pred_overall.get('p_slip_ratio_spearman_valid'))} on the Phase4 validation split.",
            "Friction-aware inputs can be evaluated without modifying raw data and without turning the proxy into a hard loss that may harm force regression.",
            "Held-out contact-geometry evaluation provides a stronger stress test than the original all-source validation split.",
        ],
        "avoid_overclaiming": ["Do not call this a full physical world model or a measured friction-cone estimator.", "Do not claim a hard universal Ft/Fn threshold; tau is empirical and train-derived.", "If a friction ablation does not beat full dynamics on all metrics, present it as interpretability/robustness evidence rather than a universal performance gain."],
        "remaining_risks": ["Validation remains derived from GSmini force-estimation data; robot grasp-success labels are not introduced.", "Held-out contact split may be harder than the original split and should be framed as stress testing."],
        "raw_data_modified": False,
    }
    json_path = report_dir / "phase5_summary.json"
    md_path = report_dir / "phase5_summary.md"
    payload["json_path"] = str(json_path); payload["md_path"] = str(md_path)
    write_json(json_path, payload)
    lines = ["# Phase5 Summary", "", f"- generated_at: `{payload['generated_at']}`", "- raw_data_modified: `False`", "", "## Reports", ""]
    for k, v in payload["phase_reports"].items(): lines.append(f"- {k}: `{v}`")
    lines += ["", "## Recommended paper method", "", payload["recommended_paper_method"], "", "## Safe claims", ""]
    for x in payload["safe_claims"]: lines.append(f"- {x}")
    lines += ["", "## Avoid overclaiming", ""]
    for x in payload["avoid_overclaiming"]: lines.append(f"- {x}")
    lines += ["", "## Remaining risks", ""]
    for x in payload["remaining_risks"]: lines.append(f"- {x}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("phase5-1")
    p.add_argument("--decoupled-run-id", default=DEFAULT_DECOUPLED_RUN)
    p.set_defaults(func=phase5_1)
    p = sub.add_parser("phase5-2")
    p.add_argument("--decoupled-run-id", default=DEFAULT_DECOUPLED_RUN)
    p.add_argument("--feature-run-id", default=None)
    p.add_argument("--stamp", default=None)
    p.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    p.add_argument("--max-epochs", type=int, default=40)
    p.add_argument("--train-batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=os.environ.get("WANDB_MODE", "online"))
    p.set_defaults(func=phase5_2)
    p = sub.add_parser("phase5-3")
    p.add_argument("--decoupled-run-id", default=DEFAULT_DECOUPLED_RUN)
    p.add_argument("--separate-run-id", default=DEFAULT_SEPARATE_RUN)
    p.add_argument("--friction-feature-run-id", default=None)
    p.add_argument("--friction-mode", choices=sorted(FRICTION_MODES), default="full_plus_q_dq")
    p.add_argument("--train-groups", nargs="+", default=["flat", "sharp"])
    p.add_argument("--test-groups", nargs="+", default=["sphere"])
    p.add_argument("--stamp", default=None)
    p.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    p.add_argument("--max-epochs", type=int, default=40)
    p.add_argument("--train-batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=os.environ.get("WANDB_MODE", "online"))
    p.set_defaults(func=phase5_3)
    p = sub.add_parser("phase5-4")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.set_defaults(func=phase5_4)
    p = sub.add_parser("summary")
    p.set_defaults(func=phase5_summary)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
