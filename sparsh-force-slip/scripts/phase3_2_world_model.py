#!/usr/bin/env python3
"""Phase3-2 lightweight tactile world-model heads.

The script keeps the Sparsh encoder and the selected Phase3-1 decoupled
force/slip decoder frozen. It precomputes train/val latent features from the
Phase1 derived dataset, then trains lightweight heads for:

1. stability proxy: future slip-free probability / instability risk;
2. one-step latent predictor: z_t -> stopgrad(z_{t+1}) plus future stability;
3. multi-horizon future stability: H=1/3/5 slip-free probabilities.

It never modifies raw data; cached features and reports are written under
/vla1/zjy and sparsh-force-slip respectively.
"""
from __future__ import annotations

import argparse
import json
import math
import os
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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import phase2_b_multitask as p2  # noqa: E402
import wandb  # noqa: E402

REPO = Path("/home/zjy/document/tactile-grasp")
WORKSPACE = REPO / "sparsh-force-slip"
PHASE3_RUN_ROOT = Path("/vla1/zjy/sparsh_runs/force_slip_phase3")
PHASE3_REPORT_ROOT = WORKSPACE / "reports/phase3"
DEFAULT_HORIZONS = (1, 3, 5)
AUX_SCHEMA = ["Fn_pred_N", "Ft_pred_N", "Ft_over_Fn_pred", "p_slip_current", "dFx_causal_N", "dFy_causal_N", "dFz_causal_N"]


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=p2.json_default), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def pool_latent(z: torch.Tensor) -> torch.Tensor:
    if z.ndim == 3:
        return z.mean(dim=1)
    return z.flatten(start_dim=1)


class FutureIndexedDataset(Dataset):
    """Wrap VisionForceSlipDataset with valid future labels and next image."""

    def __init__(self, dataset_name: str, encoder: str, horizons: Iterable[int]) -> None:
        self.dataset_name = dataset_name
        self.encoder = encoder
        self.horizons = tuple(int(h) for h in horizons)
        self.max_h = max(self.horizons)
        self.ds = p2.make_dataset(dataset_name, slip_horizon=0, encoder=encoder)
        self.valid_indices: list[int] = []
        for idx, meta in self.ds.idx2traj.items():
            traj = meta["trajectory"]
            sample = int(meta["sample"])
            labels = self.ds.trajectories[traj]["slip_label"]
            if sample + self.max_h < len(labels) and sample + 1 < len(self.ds.traj2idx[traj]):
                self.valid_indices.append(int(idx))

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, i: int) -> dict[str, Any]:
        idx = self.valid_indices[i]
        meta = self.ds.idx2traj[idx]
        traj = meta["trajectory"]
        sample = int(meta["sample"])
        labels = self.ds.trajectories[traj]["slip_label"].astype(int)
        item = self.ds[idx]
        next_idx = self.ds.traj2idx[traj][sample + 1]
        next_item = self.ds[next_idx]
        future_slip = []
        for h in self.horizons:
            window = labels[sample + 1 : sample + h + 1]
            future_slip.append(1 if int(window.sum()) > 0 else 0)
        item["next_image"] = next_item["image"]
        item["future_slip"] = torch.tensor(future_slip, dtype=torch.float32)
        item["sample_index"] = torch.tensor(sample, dtype=torch.long)
        item["trajectory_key"] = str(traj)
        item["dataset_name"] = self.dataset_name
        return item


class CachedFeatureDataset(Dataset):
    def __init__(self, payload: dict[str, Any], stage: str, horizons: tuple[int, ...]) -> None:
        self.z = payload["z"].float()
        self.aux = payload["aux"].float()
        self.z_next = payload["z_next"].float()
        self.future_slip = payload["future_slip"].float()
        all_h = tuple(int(h) for h in payload["horizons"])
        self.indices = [all_h.index(int(h)) for h in horizons]
        self.stage = stage

    def __len__(self) -> int:
        return int(self.z.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x = torch.cat([self.z[idx], self.aux[idx]], dim=0)
        return {
            "x": x,
            "z_next": self.z_next[idx],
            "future_slip": self.future_slip[idx, self.indices],
        }


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LatentPredictor(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, hidden_dim: int, horizons: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.latent_head = nn.Linear(hidden_dim, latent_dim)
        self.stability_head = nn.Linear(hidden_dim, horizons)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(x)
        return {"z_next": self.latent_head(h), "stability_logits": self.stability_head(h)}


@dataclass
class StageSpec:
    stage: str
    horizons: tuple[int, ...]
    latent_loss_weight: float = 0.0


def stage_spec(stage: str, all_horizons: tuple[int, ...]) -> StageSpec:
    if stage == "proxy":
        return StageSpec(stage=stage, horizons=(1,), latent_loss_weight=0.0)
    if stage == "latent":
        return StageSpec(stage=stage, horizons=(1,), latent_loss_weight=0.25)
    if stage == "multihorizon":
        return StageSpec(stage=stage, horizons=all_horizons, latent_loss_weight=0.0)
    raise ValueError(f"Unknown stage {stage!r}")


def split_feature_path(run_dir: Path, split: str) -> Path:
    return run_dir / "features" / f"{split}_features.pt"


def manifest_path(run_dir: Path) -> Path:
    return run_dir / "features" / "feature_manifest.json"


def make_future_loader(names: list[str], encoder: str, horizons: tuple[int, ...], batch_size: int, num_workers: int) -> DataLoader:
    datasets = [FutureIndexedDataset(name, encoder, horizons) for name in names]
    concat = torch.utils.data.ConcatDataset(datasets)
    return DataLoader(
        concat,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def precompute_split(
    split: str,
    names: list[str],
    encoder: str,
    checkpoint: Path,
    run_dir: Path,
    horizons: tuple[int, ...],
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    out_path = split_feature_path(run_dir, split)
    if out_path.exists():
        payload = torch.load(out_path, map_location="cpu", weights_only=False)
        return {"split": split, "path": str(out_path), "n_samples": int(payload["z"].shape[0]), "cached": True}
    device = get_device()
    model, ckpt_payload = p2.load_b_checkpoint(checkpoint, device)
    model.eval()
    loader = make_future_loader(names, encoder, horizons, batch_size, num_workers)
    chunks: dict[str, list[Any]] = {
        "z": [],
        "z_next": [],
        "aux": [],
        "future_slip": [],
        "current_slip": [],
        "force_gt_n": [],
        "force_pred_n": [],
        "slip_probs": [],
        "metadata": [],
    }
    for batch in tqdm(loader, desc=f"precompute:{split}"):
        x = batch["image"].to(device, non_blocking=True)
        x_next = batch["next_image"].to(device, non_blocking=True)
        z_tokens = model.encoder(x)
        z_next_tokens = model.encoder(x_next)
        out = model.decoder(z_tokens)
        z = pool_latent(z_tokens).float()
        z_next = pool_latent(z_next_tokens).float()
        force_scale = batch["force_scale"].to(device, non_blocking=True)
        delta_scale = batch["delta_force_scale"].to(device, non_blocking=True)
        force_pred_n = out["force"] * force_scale
        force_gt_n = batch["force"].to(device, non_blocking=True) * force_scale
        delta_force_n = batch["delta_force"].to(device, non_blocking=True) * delta_scale
        ft = torch.sqrt(torch.square(force_pred_n[:, 0]) + torch.square(force_pred_n[:, 1]))
        fn = torch.abs(force_pred_n[:, 2])
        ratio = ft / (fn + 1.0e-6)
        p_slip = F.softmax(out["slip"], dim=1)[:, 1]
        aux = torch.cat([fn[:, None], ft[:, None], ratio[:, None], p_slip[:, None], delta_force_n], dim=1)
        slip_probs = F.softmax(out["slip"], dim=1)
        chunks["z"].append(z.cpu().to(torch.float16))
        chunks["z_next"].append(z_next.cpu().to(torch.float16))
        chunks["aux"].append(aux.cpu().float())
        chunks["future_slip"].append(batch["future_slip"].cpu().float())
        chunks["current_slip"].append(batch["slip_label"].cpu().long())
        chunks["force_gt_n"].append(force_gt_n.cpu().float())
        chunks["force_pred_n"].append(force_pred_n.cpu().float())
        chunks["slip_probs"].append(slip_probs.cpu().float())
        datasets = batch["dataset_name"]
        trajs = batch["trajectory_key"]
        samples = batch["sample_index"].cpu().tolist()
        for dataset_name, traj, sample in zip(datasets, trajs, samples):
            chunks["metadata"].append({"dataset": dataset_name, "trajectory": str(traj), "sample": int(sample)})
    payload = {
        "split": split,
        "encoder": encoder,
        "checkpoint": str(checkpoint),
        "horizons": list(horizons),
        "aux_schema": AUX_SCHEMA,
        "z": torch.cat(chunks["z"], dim=0),
        "z_next": torch.cat(chunks["z_next"], dim=0),
        "aux": torch.cat(chunks["aux"], dim=0),
        "future_slip": torch.cat(chunks["future_slip"], dim=0),
        "current_slip": torch.cat(chunks["current_slip"], dim=0),
        "force_gt_n": torch.cat(chunks["force_gt_n"], dim=0),
        "force_pred_n": torch.cat(chunks["force_pred_n"], dim=0),
        "slip_probs": torch.cat(chunks["slip_probs"], dim=0),
        "metadata": chunks["metadata"],
        "encoder_checkpoint_payload_epoch": ckpt_payload.get("epoch"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    return {"split": split, "path": str(out_path), "n_samples": int(payload["z"].shape[0]), "cached": False}


def tensor_arrays_for_summary(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "force_gt_n": payload["force_gt_n"].numpy(),
        "force_pred_n": payload["force_pred_n"].numpy(),
        "label_gt": payload["current_slip"].numpy().astype(int),
        "slip_probs": payload["slip_probs"].numpy(),
    }


def ensure_features(
    run_id: str,
    encoder: str,
    checkpoint: Path,
    horizons: tuple[int, ...],
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    run_dir = PHASE3_RUN_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    train_info = precompute_split("train", p2.TRAIN_DATASETS, encoder, checkpoint, run_dir, horizons, batch_size, num_workers)
    val_info = precompute_split("val", p2.VAL_DATASETS, encoder, checkpoint, run_dir, horizons, batch_size, num_workers)
    val_payload = torch.load(split_feature_path(run_dir, "val"), map_location="cpu", weights_only=False)
    reference = p2.collect_force_ratio_reference(slip_horizon=0)
    base_current_metrics = p2.arrays_to_summary(tensor_arrays_for_summary(val_payload), reference)
    manifest = {
        "run_id": run_id,
        "encoder": encoder,
        "checkpoint": str(checkpoint),
        "horizons": list(horizons),
        "feature_schema": {"latent": "mean pooled frozen Sparsh encoder tokens z_t", "aux": AUX_SCHEMA},
        "train": train_info,
        "val": val_info,
        "base_current_metrics_val": base_current_metrics,
        "raw_data_modified": False,
        "derived_dataset": str(p2.DERIVED_ROOT),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(manifest_path(run_dir), manifest)
    return manifest


def binary_metrics(y_true: np.ndarray, prob: np.ndarray, prefix: str) -> dict[str, Any]:
    y_true = y_true.astype(int)
    pred = (prob >= 0.5).astype(int)
    out: dict[str, Any] = {
        f"{prefix}_positive_count": int(y_true.sum()),
        f"{prefix}_positive_ratio": float(y_true.mean()) if len(y_true) else 0.0,
        f"{prefix}_f1": float(f1_score(y_true, pred, zero_division=0)),
        f"{prefix}_accuracy": float(accuracy_score(y_true, pred)),
        f"{prefix}_precision": float(precision_score(y_true, pred, zero_division=0)),
        f"{prefix}_recall": float(recall_score(y_true, pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        out[f"{prefix}_auroc"] = float(roc_auc_score(y_true, prob))
        out[f"{prefix}_auprc"] = float(average_precision_score(y_true, prob))
    else:
        out[f"{prefix}_auroc"] = None
        out[f"{prefix}_auprc"] = None
    return out


def calibration_error(y_true_stable: np.ndarray, p_stable: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    total = len(y_true_stable)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        if i == bins - 1:
            mask = (p_stable >= lo) & (p_stable <= hi)
        else:
            mask = (p_stable >= lo) & (p_stable < hi)
        if not mask.any():
            continue
        conf = float(p_stable[mask].mean())
        acc = float(y_true_stable[mask].mean())
        ece += float(mask.sum()) / max(1, total) * abs(conf - acc)
    return float(ece)


def lead_time_to_slip(metadata: list[dict[str, Any]], current_slip: np.ndarray, p_instability: np.ndarray) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[tuple[int, int, float]]] = {}
    for meta, slip, prob in zip(metadata, current_slip.astype(int), p_instability):
        key = (meta["dataset"], str(meta["trajectory"]))
        groups.setdefault(key, []).append((int(meta["sample"]), int(slip), float(prob)))
    leads: list[int] = []
    missed = 0
    for rows in groups.values():
        rows = sorted(rows)
        onset_samples = [sample for sample, slip, _ in rows if slip == 1]
        if not onset_samples:
            continue
        onset = min(onset_samples)
        preds = [sample for sample, _, prob in rows if sample < onset and prob >= 0.5]
        if preds:
            leads.append(onset - min(preds))
        else:
            missed += 1
    return {
        "lead_time_to_slip_onset_steps_mean": float(np.mean(leads)) if leads else None,
        "lead_time_to_slip_onset_steps_median": float(np.median(leads)) if leads else None,
        "lead_time_detected_trajectories": int(len(leads)),
        "lead_time_missed_trajectories": int(missed),
    }


def make_stage_model(stage: str, input_dim: int, latent_dim: int, horizons_count: int, hidden_dim: int, dropout: float) -> nn.Module:
    if stage == "latent":
        return LatentPredictor(input_dim, latent_dim, hidden_dim, horizons_count, dropout)
    return MLP(input_dim, hidden_dim, horizons_count, dropout)


def model_logits(stage: str, model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    out = model(x)
    if isinstance(out, dict):
        return out["stability_logits"], out.get("z_next")
    return out, None


def evaluate_stage(
    stage: str,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    horizons: tuple[int, ...],
    full_val_payload: dict[str, Any],
    full_horizons: tuple[int, ...],
) -> dict[str, Any]:
    model.eval()
    probs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    latent_losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            future_slip = batch["future_slip"].to(device)
            stability_target = 1.0 - future_slip
            logits, z_pred = model_logits(stage, model, x)
            p_stable = torch.sigmoid(logits)
            probs.append(p_stable.cpu())
            targets.append(stability_target.cpu())
            if z_pred is not None:
                z_next = batch["z_next"].to(device)
                latent_losses.append(float(F.mse_loss(z_pred, z_next).detach().cpu()))
    p_stable_np = torch.cat(probs, dim=0).numpy()
    y_stable_np = torch.cat(targets, dim=0).numpy().astype(int)
    metrics: dict[str, Any] = {"n_samples": int(y_stable_np.shape[0]), "horizons": list(horizons)}
    for i, horizon in enumerate(horizons):
        y_stable = y_stable_np[:, i]
        stable_prob = p_stable_np[:, i]
        y_slip = 1 - y_stable
        inst_prob = 1.0 - stable_prob
        metrics[f"H{horizon}"] = {
            **binary_metrics(y_slip, inst_prob, "future_slip"),
            **binary_metrics(y_stable, stable_prob, "future_stability"),
            "stability_calibration_error": calibration_error(y_stable, stable_prob),
        }
        if horizon == 1:
            metrics[f"H{horizon}"].update(
                lead_time_to_slip(full_val_payload["metadata"], full_val_payload["current_slip"].numpy(), inst_prob)
            )
    if latent_losses:
        metrics["latent_mse"] = float(np.mean(latent_losses))
    return metrics


def train_stage(
    run_id: str,
    stage: str,
    encoder: str,
    checkpoint: Path,
    horizons: tuple[int, ...],
    max_epochs: int,
    batch_size: int,
    lr: float,
    hidden_dim: int,
    dropout: float,
    latent_loss_weight: float | None,
    wandb_mode: str,
    report_dir: Path,
) -> dict[str, Any]:
    run_dir = PHASE3_RUN_ROOT / run_id
    spec = stage_spec(stage, horizons)
    if latent_loss_weight is not None:
        spec.latent_loss_weight = float(latent_loss_weight)
    train_payload = torch.load(split_feature_path(run_dir, "train"), map_location="cpu", weights_only=False)
    val_payload = torch.load(split_feature_path(run_dir, "val"), map_location="cpu", weights_only=False)
    train_ds = CachedFeatureDataset(train_payload, stage, spec.horizons)
    val_ds = CachedFeatureDataset(val_payload, stage, spec.horizons)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    input_dim = int(train_ds.z.shape[1] + train_ds.aux.shape[1])
    latent_dim = int(train_ds.z.shape[1])
    device = get_device()
    model = make_stage_model(stage, input_dim, latent_dim, len(spec.horizons), hidden_dim, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.0e-4)
    wb_name = os.environ.get("WANDB_NAME", f"{run_id}_{encoder}_{stage}")
    os.environ["WANDB_MODE"] = wandb_mode
    wb = wandb.init(
        project="sparsh-finetune-tactile-grasp",
        entity="junyuzhuzjy-zhejiang-university",
        dir=str(run_dir / stage),
        id=wb_name,
        name=wb_name,
        group="phase3_2_lightweight_world_model",
        tags=["sparsh", "tactile_grasp", "force-slip", "phase3_2", encoder, stage],
        notes=f"Phase3-2 {stage}: future slip-free probability / instability risk, not grasp success.",
        config={
            "run_id": run_id,
            "stage": stage,
            "encoder": encoder,
            "checkpoint": str(checkpoint),
            "horizons": list(spec.horizons),
            "input_schema": ["z_t_mean_pooled"] + AUX_SCHEMA,
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "latent_loss_weight": spec.latent_loss_weight,
            "raw_data_modified": False,
        },
    )
    best_score = -math.inf
    best_payload: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    stage_dir = run_dir / stage
    ckpt_dir = stage_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        bce_losses = []
        latent_losses = []
        for batch in train_loader:
            x = batch["x"].to(device)
            future_slip = batch["future_slip"].to(device)
            stability_target = 1.0 - future_slip
            logits, z_pred = model_logits(stage, model, x)
            bce = F.binary_cross_entropy_with_logits(logits, stability_target)
            latent_loss = logits.sum() * 0.0
            if z_pred is not None:
                z_next = batch["z_next"].to(device).detach()
                latent_loss = F.mse_loss(z_pred, z_next)
            loss = bce + spec.latent_loss_weight * latent_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            bce_losses.append(float(bce.detach().cpu()))
            latent_losses.append(float(latent_loss.detach().cpu()))
        val_metrics = evaluate_stage(stage, model, val_loader, device, spec.horizons, val_payload, horizons)
        primary_h = f"H{spec.horizons[0]}"
        primary = val_metrics[primary_h]
        auprc = primary.get("future_slip_auprc")
        f1 = primary.get("future_slip_f1") or 0.0
        score = float(auprc) if auprc is not None else float(f1)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_bce_loss": float(np.mean(bce_losses)),
            "train_latent_loss": float(np.mean(latent_losses)),
            "val": val_metrics,
            "selection_score": score,
        }
        history.append(record)
        log_payload = {
            "epoch": epoch,
            "train/loss": record["train_loss"],
            "train/bce_loss": record["train_bce_loss"],
            "train/latent_loss": record["train_latent_loss"],
            "val/selection_score": score,
            "val/future_slip_f1": primary.get("future_slip_f1"),
            "val/future_slip_auroc": primary.get("future_slip_auroc"),
            "val/future_slip_auprc": primary.get("future_slip_auprc"),
            "val/future_stability_auroc": primary.get("future_stability_auroc"),
            "val/future_stability_auprc": primary.get("future_stability_auprc"),
            "val/stability_calibration_error": primary.get("stability_calibration_error"),
        }
        wandb.log({k: v for k, v in log_payload.items() if v is not None}, step=epoch)
        if score > best_score:
            best_score = score
            best_payload = record
            torch.save(
                {
                    "stage": stage,
                    "encoder": encoder,
                    "run_id": run_id,
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "input_dim": input_dim,
                    "latent_dim": latent_dim,
                    "hidden_dim": hidden_dim,
                    "dropout": dropout,
                    "horizons": list(spec.horizons),
                    "metrics": val_metrics,
                    "checkpoint_source": str(checkpoint),
                },
                ckpt_dir / "best.pth",
            )
        if epoch == max_epochs:
            torch.save(model.state_dict(), ckpt_dir / "latest_state_dict.pth")
        write_json(stage_dir / "history.json", history)
    wb.finish()
    manifest = read_json(manifest_path(run_dir))
    base_metrics = manifest["base_current_metrics_val"]
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "stage": stage,
        "encoder": encoder,
        "source_decoupled_checkpoint": str(checkpoint),
        "best_epoch": best_payload["epoch"] if best_payload else None,
        "best_selection_score": best_score,
        "best_val": best_payload["val"] if best_payload else None,
        "final_val": history[-1]["val"] if history else None,
        "base_current_metrics_val": base_metrics,
        "input_schema": ["z_t_mean_pooled"] + AUX_SCHEMA,
        "target_definition": "future slip-free probability / instability risk; not grasp success",
        "raw_data_modified": False,
        "feature_manifest": str(manifest_path(run_dir)),
        "checkpoint": str(ckpt_dir / "best.pth"),
        "wall_time_seconds": time.time() - start,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"phase3_2_{stage}_report.json"
    md_path = report_dir / f"phase3_2_{stage}_report.md"
    summary["json_path"] = str(json_path)
    summary["md_path"] = str(md_path)
    write_json(json_path, summary)
    md_path.write_text(render_stage_report(summary), encoding="utf-8")
    return summary


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        f = float(value)
    except Exception:
        return str(value)
    if math.isnan(f) or math.isinf(f):
        return "n/a"
    return f"{f:.{digits}f}"


def render_stage_report(summary: dict[str, Any]) -> str:
    best = summary.get("best_val") or {}
    lines = [
        f"# Phase3-2 {summary['stage']} Report",
        "",
        f"- generated_at: `{summary['generated_at']}`",
        f"- run_id: `{summary['run_id']}`",
        f"- encoder: `{summary['encoder']}`",
        f"- source_decoupled_checkpoint: `{summary['source_decoupled_checkpoint']}`",
        f"- best_epoch: `{summary['best_epoch']}`",
        "- target: future slip-free probability / instability risk (`not grasp success`).",
        f"- raw_data_modified: `{summary['raw_data_modified']}`",
        "",
        "## Future prediction metrics",
        "",
        "| horizon | future slip F1 | future slip AUROC | future slip AUPRC | future stability AUROC | future stability AUPRC | stability ECE | lead time mean |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for h in best.get("horizons", []):
        row = best.get(f"H{h}", {})
        lines.append(
            f"| {h} | {fmt(row.get('future_slip_f1'))} | {fmt(row.get('future_slip_auroc'))} | {fmt(row.get('future_slip_auprc'))} | "
            f"{fmt(row.get('future_stability_auroc'))} | {fmt(row.get('future_stability_auprc'))} | {fmt(row.get('stability_calibration_error'))} | {fmt(row.get('lead_time_to_slip_onset_steps_mean'))} |"
        )
    if "latent_mse" in best:
        lines.extend(["", f"- latent_mse: `{fmt(best['latent_mse'], 6)}`"])
    base = summary["base_current_metrics_val"]
    con = base.get("consistency") or {}
    lines.extend(
        [
            "",
            "## Retained current force/slip metrics from frozen decoupled model",
            "",
            f"- force_rmse_mean_N: `{fmt(base.get('force_rmse_mean_N'))}`",
            f"- slip_f1: `{fmt(base.get('slip_f1'))}`",
            f"- slip_accuracy: `{fmt(base.get('slip_accuracy'))}`",
            f"- contradiction_rate: `{fmt(con.get('contradiction_rate'))}`",
            f"- monotonic_calibration_error: `{fmt(con.get('monotonic_calibration_error'))}`",
            "",
            "## Artifacts",
            "",
            f"- JSON: `{summary['json_path']}`",
            f"- checkpoint: `{summary['checkpoint']}`",
            f"- feature_manifest: `{summary['feature_manifest']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def render_summary(report: dict[str, Any]) -> str:
    lines = [
        "# Phase3 Lightweight Tactile World Model Summary",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- run_id: `{report['run_id']}`",
        f"- best_backbone: `{report['encoder']}`",
        f"- source_phase3_1_report: `{report['phase3_1_report']}`",
        "- target wording: future slip-free probability / instability risk, not grasp success.",
        "",
        "## Stage comparison",
        "",
        "| stage | horizons | best epoch | H1 future slip F1 | H1 future slip AUROC | H1 future slip AUPRC | H1 stability ECE | latent MSE |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for stage, summary in report["stage_reports"].items():
        best = summary.get("best_val") or {}
        h1 = best.get("H1", {})
        lines.append(
            f"| {stage} | {best.get('horizons')} | {summary.get('best_epoch')} | {fmt(h1.get('future_slip_f1'))} | "
            f"{fmt(h1.get('future_slip_auroc'))} | {fmt(h1.get('future_slip_auprc'))} | {fmt(h1.get('stability_calibration_error'))} | {fmt(best.get('latent_mse'), 6)} |"
        )
    base = report["base_current_metrics_val"]
    con = base.get("consistency") or {}
    lines.extend(
        [
            "",
            "## Frozen decoupled current-task metrics retained during P3-2",
            "",
            f"- force_rmse_mean_N: `{fmt(base.get('force_rmse_mean_N'))}`",
            f"- slip_f1: `{fmt(base.get('slip_f1'))}`",
            f"- slip_accuracy: `{fmt(base.get('slip_accuracy'))}`",
            f"- contradiction_rate: `{fmt(con.get('contradiction_rate'))}`",
            f"- monotonic_calibration_error: `{fmt(con.get('monotonic_calibration_error'))}`",
            "",
            "## Artifacts",
            "",
            f"- JSON: `{report['json_path']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run_all(args: argparse.Namespace) -> None:
    p3_report = read_json(Path(args.phase3_1_report))
    best = p3_report["best_backbone"]
    encoder = best["encoder"]
    checkpoint = Path(best["checkpoint"])
    stamp = args.stamp or now_stamp()
    run_id = args.run_id or f"phase3_2_world_model_{encoder}_{stamp}"
    report_dir = PHASE3_REPORT_ROOT / f"phase3_2_{stamp}"
    horizons = tuple(int(h) for h in args.horizons)
    manifest = ensure_features(run_id, encoder, checkpoint, horizons, args.precompute_batch_size, args.num_workers)
    stage_reports: dict[str, Any] = {}
    for stage in ["proxy", "latent", "multihorizon"]:
        stage_reports[stage] = train_stage(
            run_id=run_id,
            stage=stage,
            encoder=encoder,
            checkpoint=checkpoint,
            horizons=horizons,
            max_epochs=args.max_epochs,
            batch_size=args.train_batch_size,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            latent_loss_weight=args.latent_loss_weight if stage == "latent" else None,
            wandb_mode=args.wandb_mode,
            report_dir=report_dir,
        )
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "stamp": stamp,
        "encoder": encoder,
        "source_checkpoint": str(checkpoint),
        "phase3_1_report": str(Path(args.phase3_1_report)),
        "feature_manifest": str(manifest_path(PHASE3_RUN_ROOT / run_id)),
        "base_current_metrics_val": manifest["base_current_metrics_val"],
        "stage_reports": stage_reports,
        "raw_data_modified": False,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "phase3_world_model_summary.json"
    md_path = report_dir / "phase3_world_model_summary.md"
    summary["json_path"] = str(json_path)
    summary["md_path"] = str(md_path)
    write_json(json_path, summary)
    md_path.write_text(render_summary(summary), encoding="utf-8")
    current = PHASE3_REPORT_ROOT / "current_phase3_2_world_model_summary.md"
    current.write_text(
        "\n".join(
            [
                "# Current Phase3-2 World Model",
                "",
                f"- run_id: `{run_id}`",
                f"- encoder: `{encoder}`",
                f"- report: `{md_path}`",
                f"- generated_at: `{summary['generated_at']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(md_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-all", help="Precompute features and train all P3-2 heads")
    run.add_argument("--phase3-1-report", required=True)
    run.add_argument("--stamp", default=None)
    run.add_argument("--run-id", default=None)
    run.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    run.add_argument("--precompute-batch-size", type=int, default=128)
    run.add_argument("--train-batch-size", type=int, default=1024)
    run.add_argument("--num-workers", type=int, default=2)
    run.add_argument("--max-epochs", type=int, default=40)
    run.add_argument("--lr", type=float, default=1.0e-3)
    run.add_argument("--hidden-dim", type=int, default=512)
    run.add_argument("--dropout", type=float, default=0.1)
    run.add_argument("--latent-loss-weight", type=float, default=0.25)
    run.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=os.environ.get("WANDB_MODE", "online"))
    run.set_defaults(func=run_all)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
