#!/usr/bin/env python3
"""Phase 2-B shared force/slip multitask downstream training and diagnostics.

This script intentionally lives in tactile-grasp/sparsh-force-slip so Phase 2 code
is versioned with the force-slip workspace. It imports the upstream Sparsh encoder,
dataset, and baseline checkpoints without modifying /home/zjy/document/sparsh.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

SPARSH_REPO = Path("/home/zjy/document/sparsh")
if str(SPARSH_REPO) not in sys.path:
    sys.path.insert(0, str(SPARSH_REPO))

import hydra  # noqa: E402
import wandb  # noqa: E402
from tactile_ssl.data.vision_based_forces_slip_probes import VisionForceSlipDataset  # noqa: E402
from tactile_ssl.downstream_task.attentive_pooler import AttentivePooler  # noqa: E402
from tactile_ssl.model import VIT_EMBED_DIMS, vit_base  # noqa: E402

REPO = Path("/home/zjy/document/tactile-grasp")
WORKSPACE = REPO / "sparsh-force-slip"
PHASE1_RUN_ID_FILE = WORKSPACE / "phase1_run_id.txt"
PHASE1_RUN_ID = PHASE1_RUN_ID_FILE.read_text().strip() if PHASE1_RUN_ID_FILE.exists() else "phase1_gsmini_20260512_043331"
DERIVED_ROOT = Path(f"/vla1/zjy/sparsh_runs/force_slip_phase1/{PHASE1_RUN_ID}/derived_gsmini")
PHASE2_ROOT = Path("/vla1/zjy/sparsh_runs/force_slip_phase2")
EXP_ROOT = Path("/vla1/zjy/sparsh_runs/experiments")
REPORT_ROOT = WORKSPACE / "reports/phase2"
LOG_ROOT = WORKSPACE / "logs/phase2"

TRAIN_DATASETS = [
    "flat_batch_1_train",
    "flat_batch_2_train",
    "sharp_batch_1_train",
    "sharp_batch_2_train",
    "sphere_batch_1_train",
    "sphere_batch_2_train",
    "sphere_batch_3_train",
    "sphere_batch_4_train",
    "sphere_batch_5_train",
    "sphere_batch_6_train",
]
VAL_DATASETS = [
    "flat_batch_1_val",
    "flat_batch_2_val",
    "sharp_batch_1_val",
    "sharp_batch_2_val",
    "sphere_batch_1_val",
    "sphere_batch_2_val",
    "sphere_batch_3_val",
    "sphere_batch_4_val",
    "sphere_batch_5_val",
    "sphere_batch_6_val",
]

ENCODER_CHECKPOINTS = {
    "dinov2": Path("/vla1/zjy/sparsh_models/sparsh-dinov2-base/dinov2_vitbase.ckpt"),
    "mae": Path("/vla1/zjy/sparsh_models/sparsh-mae-base/mae_vitbase.ckpt"),
}

A_FORCE_EXPS = {
    "dinov2": "2026.05.12_04-39_phase1_gsmini_20260512_043331_dinov2_force_gsmini_20260512_043652",
    "mae": "2026.05.12_04-39_phase1_gsmini_20260512_043331_mae_force_gsmini_20260512_043652",
}
A_SLIP_EXPS = {
    "dinov2": "2026.05.13_01-21_phase1_gsmini_20260512_043331_dinov2_slip_allsource_diag_gsmini_20260513_012000",
    "mae": "2026.05.13_01-21_phase1_gsmini_20260512_043331_mae_slip_allsource_diag_gsmini_20260513_012000",
}

DEFAULT_MAX_ABS_FORCE = [1.5, 1.5, 2.0]
DEFAULT_MAX_DELTA_FORCE = [0.80, 0.80, 0.40]


@dataclass
class TrainConfig:
    encoder: str
    run_id: str
    max_epochs: int = 51
    batch_size: int = 100
    num_workers: int = 2
    lr: float = 1.0e-4
    lambda_slip: float = 1.0
    force_beta: float = 0.02
    slip_horizon: int = 0
    seed: int = 42
    validation_frequency: int = 5
    train_batches_limit: int | None = None
    val_batches_limit: int | None = None
    wandb_mode: str = "online"
    wandb_project: str = "sparsh-finetune-tactile-grasp"
    wandb_entity: str = "junyuzhuzjy-zhejiang-university"
    wandb_group: str = "phase2_b_shared_multitask"
    class_weights: tuple[float, float] = (0.1, 1.0)


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return str(obj)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def dataset_cfg(slip_horizon: int = 0) -> Any:
    return OmegaConf.create(
        {
            "sensor": "gelsight",
            "remove_bg": True,
            "out_format": "concat_ch_img",
            "num_frames": 2,
            "frame_stride": 5,
            "path_dataset": str(DERIVED_ROOT),
            "look_in_folder": False,
            "slip_horizon": int(slip_horizon),
            "max_abs_forceXYZ": DEFAULT_MAX_ABS_FORCE,
            "max_delta_forceXYZ": DEFAULT_MAX_DELTA_FORCE,
            "transforms": {"resize": [320, 240]},
        }
    )


def make_dataset(name: str, slip_horizon: int = 0) -> VisionForceSlipDataset:
    return VisionForceSlipDataset(config=dataset_cfg(slip_horizon), dataset_name=name)


def make_concat(names: Iterable[str], slip_horizon: int = 0) -> ConcatDataset:
    return ConcatDataset([make_dataset(name, slip_horizon) for name in names])


def make_loader(
    names: Iterable[str],
    slip_horizon: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader:
    dataset = make_concat(names, slip_horizon)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def load_encoder_weights(encoder: nn.Module, checkpoint_path: Path, encoder_type: str) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    if "jepa" in encoder_type:
        encoder_key = "target_encoder"
    elif "dino" in encoder_type:
        encoder_key = "teacher_encoder.backbone"
    else:
        encoder_key = "encoder"
    target_keys = [key for key in source_state.keys() if encoder_key in key]
    if not target_keys and encoder_key == "teacher_encoder.backbone":
        encoder_key = "teacher_encoder"
        target_keys = [key for key in source_state.keys() if encoder_key in key]
    if not target_keys:
        raise RuntimeError(f"No encoder keys matching {encoder_key!r} found in {checkpoint_path}")
    if "backbone" in target_keys[0] and "backbone" not in encoder_key:
        encoder_key = encoder_key + ".backbone"
    new_state = {key.replace(f"{encoder_key}.", ""): source_state[key] for key in target_keys}
    missing, unexpected = encoder.load_state_dict(new_state, strict=False)
    return {
        "checkpoint": str(checkpoint_path),
        "encoder_type": encoder_type,
        "source_prefix": encoder_key,
        "loaded_keys": len(new_state),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


class SharedForceSlipDecoder(nn.Module):
    """Shared pooler/trunk with separate signed force and slip heads."""

    def __init__(
        self,
        embed_dim_name: str = "base",
        num_heads: int = 12,
        depth: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        embed_dim = VIT_EMBED_DIMS[f"vit_{embed_dim_name}"]
        hidden_dim = embed_dim // 2
        trunk_dim = embed_dim // 4
        self.pooler = AttentivePooler(
            num_queries=1,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=4.0,
            depth=depth,
            norm_layer=nn.LayerNorm,
            init_std=0.02,
            qkv_bias=True,
            complete_block=True,
        )
        self.trunk = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, trunk_dim),
            nn.GELU(),
        )
        self.force_head = nn.Linear(trunk_dim, 3)
        self.slip_head = nn.Linear(trunk_dim, 2)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.pooler(z).squeeze(1)
        shared = self.trunk(shared)
        # Signed normalized force components in [-1, 1]. Fn/Ft/Fmag are derived at eval time in Newton units.
        force = torch.tanh(self.force_head(shared))
        slip = self.slip_head(shared)
        return {"force": force, "slip": slip}


class FrozenEncoderSharedForceSlip(nn.Module):
    def __init__(self, encoder_name: str) -> None:
        super().__init__()
        if encoder_name not in ENCODER_CHECKPOINTS:
            raise ValueError(f"Unknown encoder {encoder_name}; expected one of {sorted(ENCODER_CHECKPOINTS)}")
        self.encoder_name = encoder_name
        self.encoder = vit_base(
            img_size=[320, 240],
            in_chans=6,
            pos_embed_fn="sinusoidal",
            num_register_tokens=1,
        )
        self.load_info = load_encoder_weights(self.encoder, ENCODER_CHECKPOINTS[encoder_name], encoder_name)
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.decoder = SharedForceSlipDecoder(embed_dim_name="base", num_heads=12, depth=1, dropout=0.0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self.encoder.eval()
        with torch.no_grad():
            z = self.encoder(x)
        return self.decoder(z.detach())


def force_derived(force_n: np.ndarray) -> dict[str, np.ndarray]:
    ft = np.sqrt(np.square(force_n[:, 0]) + np.square(force_n[:, 1]))
    fn = np.abs(force_n[:, 2])
    fmag = np.sqrt(np.square(force_n).sum(axis=1))
    return {"Fn": fn, "Ft": ft, "Fmag": fmag, "ratio_Ft_over_Fn": ft / (fn + 1.0e-6)}


def summarize(force_gt_n: np.ndarray, force_pred_n: np.ndarray, label_gt: np.ndarray, slip_probs: np.ndarray) -> dict[str, Any]:
    pred_label = slip_probs.argmax(axis=1).astype(int)
    label_gt = label_gt.astype(int)
    force_err = force_pred_n - force_gt_n
    rmse_xyz = np.sqrt(np.mean(force_err**2, axis=0))
    mae_xyz = np.mean(np.abs(force_err), axis=0)
    gt_d = force_derived(force_gt_n)
    pred_d = force_derived(force_pred_n)
    derived_rmse = {
        key: float(np.sqrt(np.mean((pred_d[key] - gt_d[key]) ** 2))) for key in ["Fn", "Ft", "Fmag"]
    }
    cm = confusion_matrix(label_gt, pred_label, labels=[0, 1]).tolist()
    return {
        "n_samples": int(len(label_gt)),
        "positive_count": int(label_gt.sum()),
        "positive_ratio": float(label_gt.mean()) if len(label_gt) else 0.0,
        "force_rmse_xyz_N": rmse_xyz.tolist(),
        "force_rmse_mean_N": float(np.mean(rmse_xyz)),
        "force_mae_xyz_N": mae_xyz.tolist(),
        "force_mae_mean_N": float(np.mean(mae_xyz)),
        "force_derived_rmse_N": derived_rmse,
        "slip_accuracy": float(accuracy_score(label_gt, pred_label)) if len(label_gt) else 0.0,
        "slip_balanced_accuracy": float(balanced_accuracy_score(label_gt, pred_label)) if len(np.unique(label_gt)) > 1 else None,
        "slip_precision": float(precision_score(label_gt, pred_label, zero_division=0)),
        "slip_recall": float(recall_score(label_gt, pred_label, zero_division=0)),
        "slip_f1": float(f1_score(label_gt, pred_label, zero_division=0)),
        "slip_confusion_matrix_labels_0_1": cm,
        "mean_pred_slip_probability": float(slip_probs[:, 1].mean()) if len(slip_probs) else 0.0,
    }


def consistency_metrics(
    force_gt_n: np.ndarray,
    force_pred_n: np.ndarray,
    label_gt: np.ndarray,
    slip_probs: np.ndarray,
    reference: dict[str, Any],
) -> dict[str, Any]:
    pred_label = slip_probs.argmax(axis=1).astype(int)
    label_gt = label_gt.astype(int)
    ratio_pred = force_derived(force_pred_n)["ratio_Ft_over_Fn"]
    p_slip = slip_probs[:, 1]
    p20 = reference["ratio_percentiles"]["p20"]
    p80 = reference["ratio_percentiles"]["p80"]
    high = ratio_pred >= p80
    low = ratio_pred <= p20

    def safe_mean(mask: np.ndarray, values: np.ndarray) -> float | None:
        if int(mask.sum()) == 0:
            return None
        return float(values[mask].mean())

    high_no_slip = safe_mean(high, (pred_label == 0).astype(float))
    low_slip = safe_mean(low, (pred_label == 1).astype(float))
    denom = int(high.sum() + low.sum())
    contradiction = None
    if denom > 0:
        contradiction = float(((high & (pred_label == 0)).sum() + (low & (pred_label == 1)).sum()) / denom)
    high_slip_mask = high & (label_gt == 1)
    low_no_slip_mask = low & (label_gt == 0)
    high_recall = safe_mean(high_slip_mask, (pred_label == 1).astype(float))
    low_false_alarm = safe_mean(low_no_slip_mask, (pred_label == 1).astype(float))

    # Shared bin boundaries are frozen from train/val GT Ft/Fn distribution.
    edges = np.array(reference["ratio_bin_edges"], dtype=float)
    bin_ids = np.digitize(ratio_pred, edges[1:-1], right=False)
    bin_rows = []
    prev_p = None
    violation = 0.0
    for i in range(len(edges) - 1):
        mask = bin_ids == i
        if mask.sum() == 0:
            row = {"bin": i, "n": 0, "ratio_mean": None, "p_slip_mean": None, "violation": 0.0}
        else:
            p_mean = float(p_slip[mask].mean())
            ratio_mean = float(ratio_pred[mask].mean())
            v = 0.0 if prev_p is None or p_mean >= prev_p else prev_p - p_mean
            violation += v
            prev_p = p_mean
            row = {"bin": i, "n": int(mask.sum()), "ratio_mean": ratio_mean, "p_slip_mean": p_mean, "violation": v}
        bin_rows.append(row)
    return {
        "ratio_source": "model_predicted_force_N",
        "threshold_source": "train_val_ground_truth_force_ratio",
        "high_ratio_threshold_p80": p80,
        "low_ratio_threshold_p20": p20,
        "high_ratio_count": int(high.sum()),
        "low_ratio_count": int(low.sum()),
        "high_ratio_no_slip_contradiction": high_no_slip,
        "low_ratio_slip_contradiction": low_slip,
        "contradiction_rate": contradiction,
        "high_ratio_slip_recall": high_recall,
        "low_ratio_false_alarm": low_false_alarm,
        "monotonic_calibration_error": float(violation / max(1, len(edges) - 1)),
        "calibration_bins": bin_rows,
    }


def merge_arrays(chunks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = chunks[0].keys()
    return {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in keys}


def arrays_to_summary(arrays: dict[str, np.ndarray], reference: dict[str, Any] | None = None) -> dict[str, Any]:
    out = summarize(arrays["force_gt_n"], arrays["force_pred_n"], arrays["label_gt"], arrays["slip_probs"])
    if reference is not None:
        out["consistency"] = consistency_metrics(
            arrays["force_gt_n"], arrays["force_pred_n"], arrays["label_gt"], arrays["slip_probs"], reference
        )
    return out


def batch_arrays_from_outputs(batch: dict[str, torch.Tensor], force_pred: torch.Tensor, slip_logits: torch.Tensor) -> dict[str, np.ndarray]:
    force_scale = batch["force_scale"].detach().cpu().numpy()
    force_gt = batch["force"].detach().cpu().numpy()
    force_pred_np = force_pred.detach().cpu().numpy()
    label_gt = batch["slip_label"].detach().cpu().numpy().astype(int)
    slip_probs = F.softmax(slip_logits.detach(), dim=1).cpu().numpy()
    return {
        "force_gt_n": force_gt * force_scale,
        "force_pred_n": force_pred_np * force_scale,
        "label_gt": label_gt,
        "slip_probs": slip_probs,
    }


def train_epoch(
    model: FrozenEncoderSharedForceSlip,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: TrainConfig,
) -> dict[str, Any]:
    model.train()
    model.encoder.eval()
    class_weights = torch.tensor(cfg.class_weights, dtype=torch.float32, device=device)
    total_loss = 0.0
    total_force_loss = 0.0
    total_slip_loss = 0.0
    total_batches = 0
    chunks: list[dict[str, np.ndarray]] = []
    iterator = tqdm(loader, desc="train", leave=False)
    for batch_idx, batch in enumerate(iterator):
        if cfg.train_batches_limit is not None and batch_idx >= cfg.train_batches_limit:
            break
        x = batch["image"].to(device, non_blocking=True)
        force_gt = batch["force"].to(device, non_blocking=True)
        slip_gt = batch["slip_label"].to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss_force = F.smooth_l1_loss(out["force"], force_gt, beta=cfg.force_beta)
        loss_slip = F.cross_entropy(out["slip"], slip_gt, weight=class_weights)
        loss = loss_force + cfg.lambda_slip * loss_slip
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=10.0)
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        total_force_loss += float(loss_force.detach().cpu())
        total_slip_loss += float(loss_slip.detach().cpu())
        total_batches += 1
        chunks.append(batch_arrays_from_outputs(batch, out["force"].detach(), out["slip"].detach()))
        iterator.set_postfix(loss=f"{total_loss / max(1, total_batches):.4f}")
    arrays = merge_arrays(chunks)
    metrics = arrays_to_summary(arrays)
    metrics.update(
        {
            "loss": total_loss / max(1, total_batches),
            "force_loss": total_force_loss / max(1, total_batches),
            "slip_loss": total_slip_loss / max(1, total_batches),
            "batches": total_batches,
        }
    )
    return metrics


@torch.no_grad()
def evaluate_b_model(
    model: FrozenEncoderSharedForceSlip,
    names: list[str],
    device: torch.device,
    slip_horizon: int,
    batch_size: int,
    num_workers: int,
    reference: dict[str, Any] | None = None,
    limit_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    chunks_all: list[dict[str, np.ndarray]] = []
    per_dataset: dict[str, Any] = {}
    for name in names:
        loader = make_loader([name], slip_horizon, batch_size, num_workers, shuffle=False, drop_last=False)
        chunks: list[dict[str, np.ndarray]] = []
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"eval:{name}", leave=False)):
            if limit_batches is not None and batch_idx >= limit_batches:
                break
            x = batch["image"].to(device, non_blocking=True)
            out = model(x)
            chunks.append(batch_arrays_from_outputs(batch, out["force"], out["slip"]))
        arrays = merge_arrays(chunks)
        per_dataset[name] = arrays_to_summary(arrays, reference)
        chunks_all.append(arrays)
    arrays_all = merge_arrays(chunks_all)
    return {"aggregate": arrays_to_summary(arrays_all, reference), "per_dataset": per_dataset}


def save_b_checkpoint(
    path: Path,
    model: FrozenEncoderSharedForceSlip,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    cfg: TrainConfig,
    metrics: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "phase2_b_shared_multitask_v1",
        "epoch": int(epoch),
        "train_config": asdict(cfg),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "metrics": metrics or {},
        "encoder_load_info": model.load_info,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    torch.save(payload, path)


def load_b_checkpoint(path: Path, device: torch.device) -> tuple[FrozenEncoderSharedForceSlip, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg_dict = payload.get("train_config", {})
    encoder = cfg_dict.get("encoder", payload.get("encoder", "dinov2"))
    model = FrozenEncoderSharedForceSlip(encoder)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model, payload


def log_wandb(prefix: str, metrics: dict[str, Any], epoch: int) -> None:
    flat = {"epoch": epoch}
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            flat[f"{prefix}/{key}"] = float(value)
        elif key == "force_rmse_xyz_N":
            flat[f"{prefix}/rmse_Fx"] = float(value[0])
            flat[f"{prefix}/rmse_Fy"] = float(value[1])
            flat[f"{prefix}/rmse_Fz"] = float(value[2])
        elif key == "force_derived_rmse_N":
            for dkey, dval in value.items():
                flat[f"{prefix}/rmse_{dkey}"] = float(dval)
    wandb.log(flat)


def command_train(args: argparse.Namespace) -> None:
    cfg = TrainConfig(
        encoder=args.encoder,
        run_id=args.run_id or f"phase2_b_gsmini_{now_stamp()}",
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        lambda_slip=args.lambda_slip,
        force_beta=args.force_beta,
        slip_horizon=args.slip_horizon,
        seed=args.seed,
        validation_frequency=args.validation_frequency,
        train_batches_limit=args.limit_train_batches,
        val_batches_limit=args.limit_val_batches,
        wandb_mode=args.wandb_mode,
    )
    set_seed(cfg.seed)
    device = get_device()
    run_dir = PHASE2_ROOT / cfg.run_id / f"{cfg.encoder}_shared_multitask"
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "train_config.json", asdict(cfg))

    train_loader = make_loader(TRAIN_DATASETS, cfg.slip_horizon, cfg.batch_size, cfg.num_workers, shuffle=True, drop_last=True)
    model = FrozenEncoderSharedForceSlip(cfg.encoder).to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)

    wandb_name = os.environ.get("WANDB_NAME", f"{cfg.run_id}_{cfg.encoder}_b_shared_multitask")
    os.environ["WANDB_MODE"] = cfg.wandb_mode
    wb = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        dir=str(run_dir),
        id=wandb_name,
        name=wandb_name,
        group=cfg.wandb_group,
        tags=["sparsh", "tactile_grasp", "force-slip", "phase2", "B", cfg.encoder, "shared-multitask"],
        notes="Phase2-B shared pooler/trunk multitask decoder: force + slip, frozen Sparsh encoder, no consistency loss.",
        config={
            **asdict(cfg),
            "phase1_run_id": PHASE1_RUN_ID,
            "derived_root": str(DERIVED_ROOT),
            "train_datasets": TRAIN_DATASETS,
            "val_datasets": VAL_DATASETS,
            "code_workspace": str(WORKSPACE),
        },
    )

    history: list[dict[str, Any]] = []
    best_f1 = -math.inf
    best_epoch = -1
    start = time.time()
    final_val: dict[str, Any] | None = None
    try:
        for epoch in range(1, cfg.max_epochs + 1):
            print(f"[phase2-B] encoder={cfg.encoder} epoch={epoch}/{cfg.max_epochs}", flush=True)
            train_metrics = train_epoch(model, train_loader, optimizer, device, cfg)
            log_wandb("train", train_metrics, epoch)
            record = {"epoch": epoch, "train": train_metrics}
            should_val = epoch == cfg.max_epochs or epoch % cfg.validation_frequency == 0
            if should_val:
                val_metrics = evaluate_b_model(
                    model,
                    VAL_DATASETS,
                    device,
                    cfg.slip_horizon,
                    cfg.batch_size,
                    cfg.num_workers,
                    reference=None,
                    limit_batches=cfg.val_batches_limit,
                )["aggregate"]
                final_val = val_metrics
                log_wandb("val", val_metrics, epoch)
                record["val"] = val_metrics
                composite = val_metrics["slip_f1"] - max(0.0, val_metrics["force_rmse_mean_N"])
                if val_metrics["slip_f1"] > best_f1:
                    best_f1 = val_metrics["slip_f1"]
                    best_epoch = epoch
                    save_b_checkpoint(ckpt_dir / "best_f1.pth", model, optimizer, epoch, cfg, val_metrics)
                wandb.log({"val/composite_f1_minus_force_rmse": composite, "epoch": epoch})
                save_b_checkpoint(ckpt_dir / f"epoch-{epoch:04d}.pth", model, optimizer, epoch, cfg, val_metrics)
            save_b_checkpoint(ckpt_dir / "latest.pth", model, optimizer, epoch, cfg, record.get("val"))
            history.append(record)
            write_json(run_dir / "history.json", history)
        final_ckpt = ckpt_dir / f"epoch-{cfg.max_epochs:04d}.pth"
        if not final_ckpt.exists():
            save_b_checkpoint(final_ckpt, model, optimizer, cfg.max_epochs, cfg, final_val or {})
        final_eval = evaluate_b_model(
            model,
            VAL_DATASETS,
            device,
            cfg.slip_horizon,
            cfg.batch_size,
            cfg.num_workers,
            reference=None,
            limit_batches=cfg.val_batches_limit,
        )
        write_json(run_dir / "metrics_val.json", final_eval)
        summary = {
            "run_id": cfg.run_id,
            "encoder": cfg.encoder,
            "status": "completed",
            "run_dir": str(run_dir),
            "final_checkpoint": str(final_ckpt),
            "best_f1_checkpoint": str(ckpt_dir / "best_f1.pth"),
            "best_f1_epoch": best_epoch,
            "wall_time_seconds": time.time() - start,
            "final_val": final_eval["aggregate"],
            "wandb_name": wandb_name,
            "wandb_url": wb.url,
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        }
        write_json(run_dir / "training_summary.json", summary)
        print(json.dumps(summary, indent=2, default=json_default), flush=True)
    finally:
        wandb.finish()


def collect_force_ratio_reference(slip_horizon: int = 0) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    all_ratio: list[np.ndarray] = []
    for split_name, names in [("train", TRAIN_DATASETS), ("val", VAL_DATASETS)]:
        split_ratios = []
        for name in names:
            ds = make_dataset(name, slip_horizon)
            forces = []
            labels = []
            for idx in range(len(ds)):
                idx_trajectory = ds.idx2traj[idx]["trajectory"]
                idx_sample = ds.idx2traj[idx]["sample"]
                label, abs_force_norm, _ = ds._get_force_slip_labels(idx, idx_trajectory, idx_sample)
                forces.append(abs_force_norm * ds.max_abs_forceXYZ)
                labels.append(label)
            forces_np = np.asarray(forces, dtype=np.float64)
            ratios = force_derived(forces_np)["ratio_Ft_over_Fn"]
            split_ratios.append(ratios)
            rows.append(
                {
                    "split": split_name,
                    "dataset": name,
                    "n": int(len(ratios)),
                    "positive_ratio": float(np.mean(labels)) if labels else 0.0,
                    "ratio_p50": float(np.percentile(ratios, 50)),
                    "ratio_p65": float(np.percentile(ratios, 65)),
                    "ratio_p80": float(np.percentile(ratios, 80)),
                    "ratio_p95": float(np.percentile(ratios, 95)),
                }
            )
        split_arr = np.concatenate(split_ratios)
        rows.append(
            {
                "split": split_name,
                "dataset": "__aggregate__",
                "n": int(len(split_arr)),
                "ratio_p20": float(np.percentile(split_arr, 20)),
                "ratio_p50": float(np.percentile(split_arr, 50)),
                "ratio_p65": float(np.percentile(split_arr, 65)),
                "ratio_p80": float(np.percentile(split_arr, 80)),
                "ratio_p95": float(np.percentile(split_arr, 95)),
            }
        )
        all_ratio.append(split_arr)
    ratio = np.concatenate(all_ratio)
    percentiles = {
        "p20": float(np.percentile(ratio, 20)),
        "p50": float(np.percentile(ratio, 50)),
        "p65": float(np.percentile(ratio, 65)),
        "p80": float(np.percentile(ratio, 80)),
        "p95": float(np.percentile(ratio, 95)),
        "p99": float(np.percentile(ratio, 99)),
    }
    bin_edges = np.quantile(ratio, np.linspace(0.0, 1.0, 11)).astype(float)
    # Ensure strictly increasing-ish bin edges for np.digitize in degenerate tails.
    for i in range(1, len(bin_edges)):
        if bin_edges[i] <= bin_edges[i - 1]:
            bin_edges[i] = bin_edges[i - 1] + 1.0e-9
    sweep = []
    for tau_label in ["p50", "p65", "p80"]:
        for alpha in [5, 10, 20]:
            for beta_cons in [0.01, 0.05, 0.1]:
                sweep.append(
                    {
                        "tau_source": tau_label,
                        "tau": percentiles[tau_label],
                        "alpha": alpha,
                        "beta_cons": beta_cons,
                        "slip_horizon": slip_horizon,
                    }
                )
    return {
        "source": "train+val ground-truth absolute force labels from Phase1 derived Gelsight-mini splits",
        "phase1_run_id": PHASE1_RUN_ID,
        "derived_root": str(DERIVED_ROOT),
        "ratio_definition": "Ft/(Fn+1e-6), Ft=sqrt(Fx^2+Fy^2), Fn=abs(Fz), Newton units",
        "slip_horizon": slip_horizon,
        "ratio_percentiles": percentiles,
        "ratio_bin_edges": bin_edges.tolist(),
        "per_dataset_rows": rows,
        "tau_candidates": [
            {"label": "p50", "tau": percentiles["p50"]},
            {"label": "p65", "tau": percentiles["p65"]},
            {"label": "p80", "tau": percentiles["p80"]},
        ],
        "alpha_candidates": [5, 10, 20],
        "beta_cons_candidates": [0.01, 0.05, 0.1],
        "sweep_grid": sweep,
        "sweep_count": len(sweep),
        "test_set_used_for_tuning": False,
    }


@torch.no_grad()
def evaluate_a_models(encoder: str, reference: dict[str, Any], batch_size: int, num_workers: int) -> dict[str, Any]:
    device = get_device()
    force_exp = EXP_ROOT / A_FORCE_EXPS[encoder]
    slip_exp = EXP_ROOT / A_SLIP_EXPS[encoder]
    force_ckpt = force_exp / "checkpoints/epoch-0051.pth"
    slip_ckpt = slip_exp / "checkpoints/epoch-0051.pth"
    if not force_ckpt.exists():
        raise FileNotFoundError(force_ckpt)
    if not slip_ckpt.exists():
        raise FileNotFoundError(slip_ckpt)
    force_cfg = OmegaConf.load(force_exp / "config.yaml")
    slip_cfg = OmegaConf.load(slip_exp / "config.yaml")
    force_cfg.task.checkpoint_task = str(force_ckpt)
    slip_cfg.task.checkpoint_task = str(slip_ckpt)
    force_model = hydra.utils.instantiate(force_cfg.task).to(device).eval()
    slip_model = hydra.utils.instantiate(slip_cfg.task).to(device).eval()
    chunks_all: list[dict[str, np.ndarray]] = []
    per_dataset: dict[str, Any] = {}
    for name in VAL_DATASETS:
        loader = make_loader([name], 0, batch_size, num_workers, shuffle=False, drop_last=False)
        chunks = []
        for batch in tqdm(loader, desc=f"eval-A-{encoder}:{name}", leave=False):
            x = batch["image"].to(device, non_blocking=True)
            force_pred = force_model(x)
            slip_out = slip_model(x)
            chunks.append(batch_arrays_from_outputs(batch, force_pred, slip_out["slip"]))
        arrays = merge_arrays(chunks)
        per_dataset[name] = arrays_to_summary(arrays, reference)
        chunks_all.append(arrays)
    aggregate_arrays = merge_arrays(chunks_all)
    return {
        "encoder": encoder,
        "train_data": "A separate baseline: force all-source + slip all-source diagnostic",
        "force_experiment": str(force_exp),
        "force_checkpoint": str(force_ckpt),
        "slip_experiment": str(slip_exp),
        "slip_checkpoint": str(slip_ckpt),
        "aggregate": arrays_to_summary(aggregate_arrays, reference),
        "per_dataset": per_dataset,
    }


@torch.no_grad()
def evaluate_b_checkpoint_for_report(
    run_id: str,
    encoder: str,
    reference: dict[str, Any],
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    device = get_device()
    run_dir = PHASE2_ROOT / run_id / f"{encoder}_shared_multitask"
    # Step2 is a train/val diagnostic stage, so use the validation-selected B checkpoint
    # for A/B gate decisions. Keep epoch-0051/final checkpoints in the run directory for
    # traceability, but avoid making C eligibility depend on a potentially overfit final epoch.
    ckpt = run_dir / "checkpoints/best_f1.pth"
    if not ckpt.exists():
        ckpt = run_dir / "checkpoints/epoch-0051.pth"
    if not ckpt.exists():
        # Allow shorter smoke runs to be evaluated if requested.
        ckpts = sorted((run_dir / "checkpoints").glob("epoch-*.pth"))
        if not ckpts:
            raise FileNotFoundError(ckpt)
        ckpt = ckpts[-1]
    model, payload = load_b_checkpoint(ckpt, device)
    eval_res = evaluate_b_model(model, VAL_DATASETS, device, payload["train_config"].get("slip_horizon", 0), batch_size, num_workers, reference=reference)
    return {
        "encoder": encoder,
        "train_data": "B shared multitask all-source",
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt),
        "aggregate": eval_res["aggregate"],
        "per_dataset": eval_res["per_dataset"],
    }


def gate_status(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    a_force = a["aggregate"]["force_rmse_mean_N"]
    b_force = b["aggregate"]["force_rmse_mean_N"]
    a_f1 = a["aggregate"]["slip_f1"]
    b_f1 = b["aggregate"]["slip_f1"]
    force_increase_pct = (b_force / a_force - 1.0) * 100.0 if a_force else math.inf
    slip_f1_drop_pp = (a_f1 - b_f1) * 100.0
    force_hard = force_increase_pct > 10.0
    slip_hard = slip_f1_drop_pp > 2.0
    force_warning = 5.0 <= force_increase_pct <= 10.0
    slip_warning = 1.0 <= slip_f1_drop_pp <= 2.0
    if force_hard or slip_hard:
        status = "hard_fail"
    elif force_warning or slip_warning:
        status = "warning_band"
    else:
        status = "pass"
    return {
        "status": status,
        "force_rmse_mean_N_A": a_force,
        "force_rmse_mean_N_B": b_force,
        "force_rmse_increase_pct": force_increase_pct,
        "slip_f1_A": a_f1,
        "slip_f1_B": b_f1,
        "slip_f1_drop_pp": slip_f1_drop_pp,
        "force_warning": force_warning,
        "slip_warning": slip_warning,
        "force_hard_fail": force_hard,
        "slip_hard_fail": slip_hard,
    }


def render_report(report: dict[str, Any]) -> str:
    lines = [
        "# Phase 2-B Shared Multitask Decoder Diagnostic Report",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- phase2_b_run_id: `{report['run_id']}`",
        f"- phase1_run_id: `{report['phase1_run_id']}`",
        f"- derived_root: `{report['derived_root']}`",
        "- scope: Step1 B shared multitask decoder training + Step2 B diagnostics and C sweep plan.",
        "- test set tuning: `false` (train/val only).",
        "",
        "## Step1 B training outputs",
        "",
        "| encoder | checkpoint | W&B/run dir | val force RMSE mean N | val slip F1 | val slip recall | gate vs A |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for enc in report["encoders"]:
        b = report["b_evaluations"][enc]
        g = report["gate"][enc]
        a = b["aggregate"]
        lines.append(
            f"| {enc} | `{Path(b['checkpoint']).name}` | `{b['run_dir']}` | {a['force_rmse_mean_N']:.4f} | {a['slip_f1']:.4f} | {a['slip_recall']:.4f} | {g['status']} |"
        )
    lines.extend(["", "## A vs B validation comparison", ""])
    lines.append("| encoder | force RMSE A | force RMSE B | Δ force RMSE | slip F1 A | slip F1 B | F1 drop pp | gate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for enc in report["encoders"]:
        g = report["gate"][enc]
        lines.append(
            f"| {enc} | {g['force_rmse_mean_N_A']:.4f} | {g['force_rmse_mean_N_B']:.4f} | {g['force_rmse_increase_pct']:+.2f}% | {g['slip_f1_A']:.4f} | {g['slip_f1_B']:.4f} | {g['slip_f1_drop_pp']:+.2f} | {g['status']} |"
        )
    lines.extend(
        [
            "",
            "Gate definitions: pass if B force RMSE increase <=5% and slip F1 drop <1pp; warning band if force RMSE increase is 5-10% or slip F1 drop is 1-2pp; hard fail if force RMSE increase >10% or slip F1 drop >2pp.",
            "",
            "## Step2 tau/alpha/beta/slip_horizon decision",
            "",
        ]
    )
    ref = report["force_ratio_reference"]
    lines.append("Tau candidates are computed from train+val ground-truth force labels only, using `Ft/(Fn+1e-6)` in Newton units.")
    lines.append("")
    lines.append("| tau label | tau |")
    lines.append("|---|---:|")
    for item in ref["tau_candidates"]:
        lines.append(f"| {item['label']} | {item['tau']:.6f} |")
    lines.extend(
        [
            "",
            f"- alpha candidates: `{ref['alpha_candidates']}`",
            f"- beta_cons candidates: `{ref['beta_cons_candidates']}`",
            f"- sweep grid count: `{ref['sweep_count']}`",
            f"- slip_horizon fixed to `{ref['slip_horizon']}` for Phase2-B/C train/val work.",
            "- slip_horizon rationale: the official Sparsh downstream slip config and Phase1 alignment audit used horizon 0; keeping it fixed avoids using test data to tune temporal label semantics. If later changed, it must be swept on train/val only before any test evaluation.",
            "",
            "## Consistency diagnostic snapshot",
            "",
            "| encoder | model | contradiction rate | monotonic calib error | high-ratio recall | low-ratio false alarm |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for enc in report["encoders"]:
        for model_key, bucket in [("A", report["a_evaluations"][enc]), ("B", report["b_evaluations"][enc])]:
            c = bucket["aggregate"]["consistency"]

            def fmt(value: Any) -> str:
                return "n/a" if value is None else f"{value:.4f}"

            lines.append(
                f"| {enc} | {model_key} | {fmt(c['contradiction_rate'])} | {fmt(c['monotonic_calibration_error'])} | {fmt(c['high_ratio_slip_recall'])} | {fmt(c['low_ratio_false_alarm'])} |"
            )
    lines.extend(
        [
            "",
            "## C entry note",
            "",
            report["conclusion"],
            "",
            "## Artifacts",
            "",
            f"- JSON report: `{report['json_path']}`",
            f"- C sweep plan JSON: `{report['c_sweep_plan_path']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def render_sweep_plan(reference: dict[str, Any], run_id: str) -> str:
    tau_items = ", ".join([item["label"] + "=" + format(item["tau"], ".6f") for item in reference["tau_candidates"]])
    lines = [
        "# Phase 2-C Sweep Plan from Step2",
        "",
        f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- phase2_b_run_id: `{run_id}`",
        f"- source: `{reference['source']}`",
        f"- ratio_definition: `{reference['ratio_definition']}`",
        f"- test_set_used_for_tuning: `{reference['test_set_used_for_tuning']}`",
        "",
        "## Frozen candidate ranges",
        "",
        "| parameter | candidates |",
        "|---|---|",
        f"| tau | {tau_items} |",
        f"| alpha | {reference['alpha_candidates']} |",
        f"| beta_cons | {reference['beta_cons_candidates']} |",
        f"| slip_horizon | {reference['slip_horizon']} |",
        "",
        "## Grid",
        "",
        "| tau_source | tau | alpha | beta_cons | slip_horizon |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in reference["sweep_grid"]:
        lines.append(f"| {row['tau_source']} | {row['tau']:.6f} | {row['alpha']} | {row['beta_cons']} | {row['slip_horizon']} |")
    lines.append("")
    return "\n".join(lines) + "\n"


def command_report(args: argparse.Namespace) -> None:
    run_id = args.run_id
    encoders = args.encoders
    set_seed(args.seed)
    report_dir = REPORT_ROOT / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    reference_path = report_dir / "phase2_b_force_ratio_reference.json"
    if reference_path.exists() and not args.refresh_reference:
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
    else:
        reference = collect_force_ratio_reference(slip_horizon=args.slip_horizon)
        write_json(reference_path, reference)
    a_evals = {}
    b_evals = {}
    gates = {}
    cache_dir = report_dir / "eval_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for enc in encoders:
        a_cache = cache_dir / f"a_{enc}_allsource_val.json"
        b_cache = cache_dir / f"b_{enc}_allsource_val.json"
        if a_cache.exists() and not args.refresh_eval:
            a_eval = json.loads(a_cache.read_text(encoding="utf-8"))
        else:
            a_eval = evaluate_a_models(enc, reference, args.batch_size, args.num_workers)
            write_json(a_cache, a_eval)
        if b_cache.exists() and not args.refresh_eval:
            b_eval = json.loads(b_cache.read_text(encoding="utf-8"))
        else:
            b_eval = evaluate_b_checkpoint_for_report(run_id, enc, reference, args.batch_size, args.num_workers)
            write_json(b_cache, b_eval)
        a_evals[enc] = a_eval
        b_evals[enc] = b_eval
        gates[enc] = gate_status(a_eval, b_eval)
    hard_fails = [enc for enc, gate in gates.items() if gate["status"] == "hard_fail"]
    warnings = [enc for enc, gate in gates.items() if gate["status"] == "warning_band"]
    if hard_fails:
        conclusion = (
            "B triggers a hard fail for "
            + ", ".join(hard_fails)
            + "; Phase2-C may only be run as diagnostic-only for those encoders and must not be used for a formal improvement claim."
        )
    elif warnings:
        conclusion = (
            "B is inside the warning band for "
            + ", ".join(warnings)
            + "; Phase2-C can proceed, but reports must explicitly mark the main-metric risk."
        )
    else:
        conclusion = "B passes the Step1 gate on the evaluated train/val policy; Phase2-C can proceed as a formal candidate using the Step2 train/val-only sweep plan."
    sweep_json = report_dir / "phase2_c_sweep_plan.json"
    sweep_md = report_dir / "phase2_c_sweep_plan.md"
    write_json(sweep_json, reference)
    sweep_md.write_text(render_sweep_plan(reference, run_id), encoding="utf-8")
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "phase1_run_id": PHASE1_RUN_ID,
        "derived_root": str(DERIVED_ROOT),
        "encoders": encoders,
        "a_evaluations": a_evals,
        "b_evaluations": b_evals,
        "gate": gates,
        "force_ratio_reference": reference,
        "conclusion": conclusion,
        "json_path": str(report_dir / "phase2_b_diagnostic_report.json"),
        "c_sweep_plan_path": str(sweep_json),
    }
    json_path = report_dir / "phase2_b_diagnostic_report.json"
    md_path = report_dir / "phase2_b_diagnostic_report.md"
    write_json(json_path, report)
    md_path.write_text(render_report(report), encoding="utf-8")
    current = REPORT_ROOT / "current_phase2_b.md"
    current.write_text(
        f"# Current Phase2-B\n\n- run_id: `{run_id}`\n- report: `{md_path}`\n- C sweep plan: `{sweep_md}`\n- generated_at: `{report['generated_at']}`\n",
        encoding="utf-8",
    )
    print(json.dumps({"report": str(md_path), "json": str(json_path), "conclusion": conclusion}, indent=2), flush=True)


def command_smoke(args: argparse.Namespace) -> None:
    cfg = TrainConfig(
        encoder=args.encoder,
        run_id=args.run_id or f"phase2_b_smoke_{now_stamp()}",
        max_epochs=1,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_batches_limit=1,
        val_batches_limit=1,
        wandb_mode="disabled",
    )
    args_train = argparse.Namespace(
        encoder=cfg.encoder,
        run_id=cfg.run_id,
        max_epochs=1,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        lr=cfg.lr,
        lambda_slip=cfg.lambda_slip,
        force_beta=cfg.force_beta,
        slip_horizon=cfg.slip_horizon,
        seed=cfg.seed,
        validation_frequency=1,
        limit_train_batches=1,
        limit_val_batches=1,
        wandb_mode="disabled",
    )
    command_train(args_train)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train one Phase2-B shared multitask decoder")
    train.add_argument("--encoder", choices=sorted(ENCODER_CHECKPOINTS), required=True)
    train.add_argument("--run-id", default=None)
    train.add_argument("--max-epochs", type=int, default=51)
    train.add_argument("--batch-size", type=int, default=100)
    train.add_argument("--num-workers", type=int, default=2)
    train.add_argument("--lr", type=float, default=1.0e-4)
    train.add_argument("--lambda-slip", type=float, default=1.0)
    train.add_argument("--force-beta", type=float, default=0.02)
    train.add_argument("--slip-horizon", type=int, default=0)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--validation-frequency", type=int, default=5)
    train.add_argument("--limit-train-batches", type=int, default=None)
    train.add_argument("--limit-val-batches", type=int, default=None)
    train.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=os.environ.get("WANDB_MODE", "online"))
    train.set_defaults(func=command_train)

    report = sub.add_parser("report", help="Evaluate A/B and write Step2 diagnostics/sweep plan")
    report.add_argument("--run-id", required=True)
    report.add_argument("--encoders", nargs="+", choices=sorted(ENCODER_CHECKPOINTS), default=["dinov2", "mae"])
    report.add_argument("--batch-size", type=int, default=100)
    report.add_argument("--num-workers", type=int, default=2)
    report.add_argument("--slip-horizon", type=int, default=0)
    report.add_argument("--seed", type=int, default=42)
    report.add_argument("--refresh-eval", action="store_true")
    report.add_argument("--refresh-reference", action="store_true")
    report.set_defaults(func=command_report)

    smoke = sub.add_parser("smoke", help="One-batch smoke train/eval without W&B upload")
    smoke.add_argument("--encoder", choices=sorted(ENCODER_CHECKPOINTS), default="dinov2")
    smoke.add_argument("--run-id", default=None)
    smoke.add_argument("--batch-size", type=int, default=8)
    smoke.add_argument("--num-workers", type=int, default=0)
    smoke.set_defaults(func=command_smoke)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
