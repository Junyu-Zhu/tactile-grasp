#!/usr/bin/env python3
"""Phase4 paper-submission supplement experiments for force/slip coupling.

All outputs are derived artifacts. Raw datasets under /vla1/zjy are never
modified. The script intentionally reuses Phase2/Phase3 utilities so that data
splits, force units, and evaluation definitions stay aligned with prior phases.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import phase2_b_multitask as p2  # noqa: E402
import phase3_2_world_model as wm  # noqa: E402
import hydra  # noqa: E402
import wandb  # noqa: E402

REPO = Path("/home/zjy/document/tactile-grasp")
WORKSPACE = REPO / "sparsh-force-slip"
PHASE4_RUN_ROOT = Path("/vla1/zjy/sparsh_runs/force_slip_phase4")
PHASE4_REPORT_ROOT = WORKSPACE / "reports/phase4"
DEFAULT_HORIZONS = (1, 3, 5)
DECOUPLED_AUX_SCHEMA = list(wm.AUX_SCHEMA)
SEPARATE_AUX_SCHEMA = ["Fn_pred_N", "Ft_pred_N", "Ft_over_Fn_pred", "p_slip_current"]
INPUT_MODES = {
    "z_only": [],
    "z_p_slip": ["p_slip_current"],
    "z_force": ["Fn_pred_N", "Ft_pred_N", "Ft_over_Fn_pred"],
    "z_force_slip": ["Fn_pred_N", "Ft_pred_N", "Ft_over_Fn_pred", "p_slip_current"],
    "full": ["Fn_pred_N", "Ft_pred_N", "Ft_over_Fn_pred", "p_slip_current", "dFx_causal_N", "dFy_causal_N", "dFz_causal_N"],
}


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=p2.json_default), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except Exception:
        return str(value)
    if math.isnan(value) or math.isinf(value):
        return "n/a"
    return f"{value:.{digits}f}"


def resolve_experiment(spec: str | None, default: str) -> Path:
    if spec is None:
        spec = default
    path = Path(spec)
    if path.exists():
        return path
    return p2.resolve_experiment_dir(spec)


class SubsetFeatureDataset(Dataset):
    def __init__(self, payload: dict[str, Any], horizons: tuple[int, ...], aux_names: list[str]) -> None:
        self.z = payload["z"].float()
        aux = payload["aux"].float()
        schema = list(payload.get("aux_schema", []))
        missing = [name for name in aux_names if name not in schema]
        if missing:
            raise ValueError(f"Aux columns {missing} are unavailable in schema {schema}")
        idx = [schema.index(name) for name in aux_names]
        self.aux = aux[:, idx] if idx else torch.empty((aux.shape[0], 0), dtype=torch.float32)
        self.z_next = payload["z_next"].float()
        all_h = tuple(int(h) for h in payload["horizons"])
        self.h_indices = [all_h.index(int(h)) for h in horizons]
        self.future_slip = payload["future_slip"].float()

    def __len__(self) -> int:
        return int(self.z.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x = torch.cat([self.z[idx], self.aux[idx]], dim=0)
        return {"x": x, "z_next": self.z_next[idx], "future_slip": self.future_slip[idx, self.h_indices]}


@torch.no_grad()
def evaluate_separate_pair(
    encoder: str,
    force_exp: Path,
    slip_exp: Path,
    out_json: Path,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    dev = device()
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
    force_model = hydra.utils.instantiate(force_cfg.task).to(dev).eval()
    slip_model = hydra.utils.instantiate(slip_cfg.task).to(dev).eval()
    reference = p2.collect_force_ratio_reference(slip_horizon=0)
    chunks_all: list[dict[str, np.ndarray]] = []
    per_dataset: dict[str, Any] = {}
    for name in p2.VAL_DATASETS:
        loader = p2.make_loader([name], 0, batch_size, num_workers, shuffle=False, drop_last=False, encoder=encoder)
        chunks: list[dict[str, np.ndarray]] = []
        for batch in tqdm(loader, desc=f"eval-separate:{name}", leave=False):
            x = batch["image"].to(dev, non_blocking=True)
            force_pred = force_model(x)
            slip_out = slip_model(x)
            chunks.append(p2.batch_arrays_from_outputs(batch, force_pred, slip_out["slip"]))
        arrays = p2.merge_arrays(chunks)
        per_dataset[name] = p2.arrays_to_summary(arrays, reference)
        chunks_all.append(arrays)
    aggregate_arrays = p2.merge_arrays(chunks_all)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "kind": "separate_force_slip_eval",
        "encoder": encoder,
        "force_experiment": str(force_exp),
        "force_checkpoint": str(force_ckpt),
        "slip_experiment": str(slip_exp),
        "slip_checkpoint": str(slip_ckpt),
        "aggregate": p2.arrays_to_summary(aggregate_arrays, reference),
        "per_dataset": per_dataset,
        "raw_data_modified": False,
    }
    write_json(out_json, summary)
    return summary


@torch.no_grad()
def evaluate_decoupled_checkpoint(
    checkpoint: Path,
    out_json: Path,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    dev = device()
    model, payload = p2.load_b_checkpoint(checkpoint, dev)
    reference = p2.collect_force_ratio_reference(slip_horizon=0)
    eval_result = p2.evaluate_b_model(model, p2.VAL_DATASETS, dev, 0, batch_size, num_workers, reference)
    cfg = payload.get("train_config", {})
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "kind": "decoupled_multitask_eval",
        "encoder": cfg.get("encoder", "mae"),
        "decoder_variant": cfg.get("decoder_variant", "decoupled"),
        "checkpoint": str(checkpoint),
        "aggregate": eval_result["aggregate"],
        "per_dataset": eval_result["per_dataset"],
        "raw_data_modified": False,
    }
    write_json(out_json, summary)
    return summary


def manifest_path(run_dir: Path) -> Path:
    return run_dir / "features" / "feature_manifest.json"


def ensure_decoupled_features(
    run_id: str,
    checkpoint: Path,
    horizons: tuple[int, ...],
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    run_dir = PHASE4_RUN_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    train_info = wm.precompute_split("train", p2.TRAIN_DATASETS, "mae", checkpoint, run_dir, horizons, batch_size, num_workers)
    val_info = wm.precompute_split("val", p2.VAL_DATASETS, "mae", checkpoint, run_dir, horizons, batch_size, num_workers)
    val_payload = torch.load(wm.split_feature_path(run_dir, "val"), map_location="cpu", weights_only=False)
    reference = p2.collect_force_ratio_reference(slip_horizon=0)
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "feature_kind": "decoupled_joint",
        "run_id": run_id,
        "encoder": "mae",
        "checkpoint": str(checkpoint),
        "horizons": list(horizons),
        "feature_schema": {"latent": "mean pooled MAE tokens", "aux": DECOUPLED_AUX_SCHEMA},
        "train": train_info,
        "val": val_info,
        "base_current_metrics_val": p2.arrays_to_summary(wm.tensor_arrays_for_summary(val_payload), reference),
        "raw_data_modified": False,
        "derived_dataset": str(p2.DERIVED_ROOT),
    }
    write_json(manifest_path(run_dir), manifest)
    return manifest


@torch.no_grad()
def precompute_separate_split(
    split: str,
    names: list[str],
    run_dir: Path,
    encoder_checkpoint: Path,
    force_exp: Path,
    slip_exp: Path,
    horizons: tuple[int, ...],
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    out_path = wm.split_feature_path(run_dir, split)
    if out_path.exists():
        payload = torch.load(out_path, map_location="cpu", weights_only=False)
        return {"split": split, "path": str(out_path), "n_samples": int(payload["z"].shape[0]), "cached": True}
    dev = device()
    encoder_model, _ = p2.load_b_checkpoint(encoder_checkpoint, dev)
    force_ckpt = force_exp / "checkpoints/epoch-0051.pth"
    slip_ckpt = slip_exp / "checkpoints/epoch-0051.pth"
    force_cfg = OmegaConf.load(force_exp / "config.yaml")
    slip_cfg = OmegaConf.load(slip_exp / "config.yaml")
    force_cfg.task.checkpoint_task = str(force_ckpt)
    slip_cfg.task.checkpoint_task = str(slip_ckpt)
    force_model = hydra.utils.instantiate(force_cfg.task).to(dev).eval()
    slip_model = hydra.utils.instantiate(slip_cfg.task).to(dev).eval()
    loader = wm.make_future_loader(names, "mae", horizons, batch_size, num_workers)
    chunks: dict[str, list[Any]] = {k: [] for k in [
        "z", "z_next", "aux", "future_slip", "current_slip", "force_gt_n", "force_pred_n", "slip_probs", "metadata"
    ]}
    for batch in tqdm(loader, desc=f"precompute-separate:{split}"):
        x = batch["image"].to(dev, non_blocking=True)
        x_next = batch["next_image"].to(dev, non_blocking=True)
        z_tokens = encoder_model.encoder(x)
        z_next_tokens = encoder_model.encoder(x_next)
        z = wm.pool_latent(z_tokens).float()
        z_next = wm.pool_latent(z_next_tokens).float()
        force_scale = batch["force_scale"].to(dev, non_blocking=True)
        force_pred_norm = force_model(x)
        slip_out = slip_model(x)
        force_pred_n = force_pred_norm * force_scale
        force_gt_n = batch["force"].to(dev, non_blocking=True) * force_scale
        ft = torch.sqrt(torch.square(force_pred_n[:, 0]) + torch.square(force_pred_n[:, 1]))
        fn = torch.abs(force_pred_n[:, 2])
        ratio = ft / (fn + 1.0e-6)
        slip_probs = F.softmax(slip_out["slip"], dim=1)
        p_slip = slip_probs[:, 1]
        aux = torch.cat([fn[:, None], ft[:, None], ratio[:, None], p_slip[:, None]], dim=1)
        chunks["z"].append(z.cpu().to(torch.float16))
        chunks["z_next"].append(z_next.cpu().to(torch.float16))
        chunks["aux"].append(aux.cpu().float())
        chunks["future_slip"].append(batch["future_slip"].cpu().float())
        chunks["current_slip"].append(batch["slip_label"].cpu().long())
        chunks["force_gt_n"].append(force_gt_n.cpu().float())
        chunks["force_pred_n"].append(force_pred_n.cpu().float())
        chunks["slip_probs"].append(slip_probs.cpu().float())
        for dataset_name, traj, sample in zip(batch["dataset_name"], batch["trajectory_key"], batch["sample_index"].cpu().tolist()):
            chunks["metadata"].append({"dataset": dataset_name, "trajectory": str(traj), "sample": int(sample)})
    payload = {
        "split": split,
        "encoder": "mae",
        "feature_kind": "separate_late_fusion",
        "encoder_checkpoint": str(encoder_checkpoint),
        "force_experiment": str(force_exp),
        "slip_experiment": str(slip_exp),
        "horizons": list(horizons),
        "aux_schema": SEPARATE_AUX_SCHEMA,
        "z": torch.cat(chunks["z"], dim=0),
        "z_next": torch.cat(chunks["z_next"], dim=0),
        "aux": torch.cat(chunks["aux"], dim=0),
        "future_slip": torch.cat(chunks["future_slip"], dim=0),
        "current_slip": torch.cat(chunks["current_slip"], dim=0),
        "force_gt_n": torch.cat(chunks["force_gt_n"], dim=0),
        "force_pred_n": torch.cat(chunks["force_pred_n"], dim=0),
        "slip_probs": torch.cat(chunks["slip_probs"], dim=0),
        "metadata": chunks["metadata"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    return {"split": split, "path": str(out_path), "n_samples": int(payload["z"].shape[0]), "cached": False}


def ensure_separate_features(
    run_id: str,
    encoder_checkpoint: Path,
    force_exp: Path,
    slip_exp: Path,
    horizons: tuple[int, ...],
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    run_dir = PHASE4_RUN_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    train_info = precompute_separate_split("train", p2.TRAIN_DATASETS, run_dir, encoder_checkpoint, force_exp, slip_exp, horizons, batch_size, num_workers)
    val_info = precompute_separate_split("val", p2.VAL_DATASETS, run_dir, encoder_checkpoint, force_exp, slip_exp, horizons, batch_size, num_workers)
    val_payload = torch.load(wm.split_feature_path(run_dir, "val"), map_location="cpu", weights_only=False)
    reference = p2.collect_force_ratio_reference(slip_horizon=0)
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "feature_kind": "separate_late_fusion",
        "run_id": run_id,
        "encoder": "mae",
        "encoder_checkpoint": str(encoder_checkpoint),
        "force_experiment": str(force_exp),
        "slip_experiment": str(slip_exp),
        "horizons": list(horizons),
        "feature_schema": {"latent": "mean pooled MAE tokens", "aux": SEPARATE_AUX_SCHEMA},
        "train": train_info,
        "val": val_info,
        "base_current_metrics_val": p2.arrays_to_summary(wm.tensor_arrays_for_summary(val_payload), reference),
        "raw_data_modified": False,
        "derived_dataset": str(p2.DERIVED_ROOT),
    }
    write_json(manifest_path(run_dir), manifest)
    return manifest


def train_future_head(
    feature_run_id: str,
    experiment_name: str,
    input_mode: str,
    horizons: tuple[int, ...],
    max_epochs: int,
    batch_size: int,
    lr: float,
    hidden_dim: int,
    dropout: float,
    seed: int,
    wandb_mode: str,
    report_dir: Path,
) -> dict[str, Any]:
    if input_mode not in INPUT_MODES:
        raise ValueError(f"Unknown input_mode {input_mode}; expected {sorted(INPUT_MODES)}")
    set_seed(seed)
    run_dir = PHASE4_RUN_ROOT / feature_run_id
    train_payload = torch.load(wm.split_feature_path(run_dir, "train"), map_location="cpu", weights_only=False)
    val_payload = torch.load(wm.split_feature_path(run_dir, "val"), map_location="cpu", weights_only=False)
    aux_names = INPUT_MODES[input_mode]
    train_ds = SubsetFeatureDataset(train_payload, horizons, aux_names)
    val_ds = SubsetFeatureDataset(val_payload, horizons, aux_names)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    dev = device()
    model = wm.MLP(int(train_ds.z.shape[1] + train_ds.aux.shape[1]), hidden_dim, len(horizons), dropout).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.0e-4)
    wb_name = os.environ.get("WANDB_NAME", experiment_name)
    os.environ["WANDB_MODE"] = wandb_mode
    wb = wandb.init(
        project="sparsh-finetune-tactile-grasp",
        entity="junyuzhuzjy-zhejiang-university",
        dir=str(run_dir / "heads" / experiment_name),
        id=wb_name,
        name=wb_name,
        group="phase4_paper_future_stability",
        tags=["sparsh", "tactile_grasp", "force-slip", "phase4", input_mode],
        notes="Phase4 lightweight future stability head; target is future slip/instability, not grasp success.",
        config={
            "feature_run_id": feature_run_id,
            "experiment_name": experiment_name,
            "input_mode": input_mode,
            "input_schema": ["z_t_mean_pooled"] + aux_names,
            "horizons": list(horizons),
            "seed": seed,
            "raw_data_modified": False,
        },
    )
    best_score = -math.inf
    best_record: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    head_dir = run_dir / "heads" / experiment_name
    ckpt_dir = head_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            x = batch["x"].to(dev)
            target = 1.0 - batch["future_slip"].to(dev)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics = wm.evaluate_stage("multihorizon", model, val_loader, dev, horizons, val_payload, horizons)
        h1 = metrics.get("H1", {})
        score = h1.get("future_slip_auprc")
        if score is None:
            score = h1.get("future_slip_f1", 0.0)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val": metrics, "selection_score": float(score)}
        history.append(record)
        wandb.log(
            {
                "epoch": epoch,
                "train/loss": record["train_loss"],
                "val/selection_score": record["selection_score"],
                "val/H1_future_slip_f1": h1.get("future_slip_f1"),
                "val/H1_future_slip_auroc": h1.get("future_slip_auroc"),
                "val/H1_future_slip_auprc": h1.get("future_slip_auprc"),
                "val/H1_future_stability_auroc": h1.get("future_stability_auroc"),
                "val/H1_future_stability_auprc": h1.get("future_stability_auprc"),
                "val/H1_stability_ece": h1.get("stability_calibration_error"),
            },
            step=epoch,
        )
        if record["selection_score"] > best_score:
            best_score = record["selection_score"]
            best_record = record
            torch.save(
                {
                    "feature_run_id": feature_run_id,
                    "experiment_name": experiment_name,
                    "input_mode": input_mode,
                    "aux_names": aux_names,
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "horizons": list(horizons),
                    "metrics": metrics,
                    "seed": seed,
                },
                ckpt_dir / "best.pth",
            )
        write_json(head_dir / "history.json", history)
    wb.finish()
    manifest = read_json(manifest_path(run_dir))
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "feature_run_id": feature_run_id,
        "feature_kind": manifest.get("feature_kind"),
        "experiment_name": experiment_name,
        "input_mode": input_mode,
        "input_schema": ["z_t_mean_pooled"] + aux_names,
        "horizons": list(horizons),
        "seed": seed,
        "best_epoch": best_record["epoch"] if best_record else None,
        "best_selection_score": best_score,
        "best_val": best_record["val"] if best_record else None,
        "base_current_metrics_val": manifest.get("base_current_metrics_val"),
        "feature_manifest": str(manifest_path(run_dir)),
        "checkpoint": str(ckpt_dir / "best.pth"),
        "wall_time_seconds": time.time() - start,
        "raw_data_modified": False,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{experiment_name}_report.json"
    md_path = report_dir / f"{experiment_name}_report.md"
    summary["json_path"] = str(json_path)
    summary["md_path"] = str(md_path)
    write_json(json_path, summary)
    md_path.write_text(render_head_report(summary), encoding="utf-8")
    return summary


def render_head_report(summary: dict[str, Any]) -> str:
    best = summary.get("best_val") or {}
    lines = [
        f"# Phase4 Future Stability Head: {summary['experiment_name']}",
        "",
        f"- generated_at: `{summary['generated_at']}`",
        f"- feature_run_id: `{summary['feature_run_id']}`",
        f"- feature_kind: `{summary.get('feature_kind')}`",
        f"- input_mode: `{summary['input_mode']}`",
        f"- input_schema: `{summary['input_schema']}`",
        f"- best_epoch: `{summary['best_epoch']}`",
        f"- raw_data_modified: `{summary['raw_data_modified']}`",
        "",
        "| H | future slip F1 | future slip AUROC | future slip AUPRC | stability AUROC | stability AUPRC | ECE | lead mean |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for h in best.get("horizons", []):
        row = best.get(f"H{h}", {})
        lines.append(
            f"| {h} | {fmt(row.get('future_slip_f1'))} | {fmt(row.get('future_slip_auroc'))} | {fmt(row.get('future_slip_auprc'))} | "
            f"{fmt(row.get('future_stability_auroc'))} | {fmt(row.get('future_stability_auprc'))} | {fmt(row.get('stability_calibration_error'))} | {fmt(row.get('lead_time_to_slip_onset_steps_mean'))} |"
        )
    base = summary.get("base_current_metrics_val") or {}
    lines += [
        "",
        "## Current-task metrics from feature source",
        "",
        f"- force_rmse_mean_N: `{fmt(base.get('force_rmse_mean_N'))}`",
        f"- slip_f1: `{fmt(base.get('slip_f1'))}`",
        f"- slip_accuracy: `{fmt(base.get('slip_accuracy'))}`",
        "",
        "## Artifacts",
        "",
        f"- JSON: `{summary['json_path']}`",
        f"- checkpoint: `{summary['checkpoint']}`",
        f"- feature_manifest: `{summary['feature_manifest']}`",
    ]
    return "\n".join(lines) + "\n"


def render_comparison_report(title: str, payload: dict[str, Any]) -> str:
    lines = [
        f"# {title}",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- raw_data_modified: `{payload['raw_data_modified']}`",
        "",
        "| condition | feature kind | input mode | force RMSE | slip F1 | slip acc | H1 F1 | H1 AUROC | H1 AUPRC | H1 ECE | H3 F1 | H5 F1 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["conditions"]:
        base = row.get("base_current_metrics_val") or {}
        best = row.get("best_val") or {}
        h1, h3, h5 = best.get("H1", {}), best.get("H3", {}), best.get("H5", {})
        lines.append(
            f"| {row['condition']} | {row.get('feature_kind')} | {row.get('input_mode')} | {fmt(base.get('force_rmse_mean_N'))} | "
            f"{fmt(base.get('slip_f1'))} | {fmt(base.get('slip_accuracy'))} | {fmt(h1.get('future_slip_f1'))} | "
            f"{fmt(h1.get('future_slip_auroc'))} | {fmt(h1.get('future_slip_auprc'))} | {fmt(h1.get('stability_calibration_error'))} | "
            f"{fmt(h3.get('future_slip_f1'))} | {fmt(h5.get('future_slip_f1'))} |"
        )
    lines += ["", "## Notes", ""]
    lines += [f"- {note}" for note in payload.get("notes", [])]
    lines += ["", "## Artifacts", ""]
    for row in payload["conditions"]:
        lines.append(f"- {row['condition']}: `{row.get('json_path')}`")
    return "\n".join(lines) + "\n"


def command_eval_separate(args: argparse.Namespace) -> None:
    force_exp = resolve_experiment(args.force_exp, p2.A_FORCE_EXPS[args.encoder])
    slip_exp = resolve_experiment(args.slip_exp, p2.A_SLIP_EXPS[args.encoder])
    out = Path(args.out_json)
    summary = evaluate_separate_pair(args.encoder, force_exp, slip_exp, out, args.batch_size, args.num_workers)
    print(summary["aggregate"])
    print(out)


def command_eval_decoupled(args: argparse.Namespace) -> None:
    summary = evaluate_decoupled_checkpoint(Path(args.checkpoint), Path(args.out_json), args.batch_size, args.num_workers)
    print(summary["aggregate"])
    print(args.out_json)


def command_train_head(args: argparse.Namespace) -> None:
    horizons = tuple(int(h) for h in args.horizons)
    if args.feature_kind == "decoupled":
        ensure_decoupled_features(args.feature_run_id, Path(args.checkpoint), horizons, args.precompute_batch_size, args.num_workers)
    elif args.feature_kind == "separate":
        force_exp = resolve_experiment(args.force_exp, p2.A_FORCE_EXPS["mae"])
        slip_exp = resolve_experiment(args.slip_exp, p2.A_SLIP_EXPS["mae"])
        ensure_separate_features(args.feature_run_id, Path(args.checkpoint), force_exp, slip_exp, horizons, args.precompute_batch_size, args.num_workers)
    else:
        raise ValueError(args.feature_kind)
    report = train_future_head(
        feature_run_id=args.feature_run_id,
        experiment_name=args.experiment_name,
        input_mode=args.input_mode,
        horizons=horizons,
        max_epochs=args.max_epochs,
        batch_size=args.train_batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        seed=args.seed,
        wandb_mode=args.wandb_mode,
        report_dir=Path(args.report_dir),
    )
    print(report["md_path"])


def command_input_ablation(args: argparse.Namespace) -> None:
    stamp = args.stamp or now_stamp()
    horizons = tuple(int(h) for h in args.horizons)
    feature_run_id = args.feature_run_id or f"phase4_3_decoupled_features_{stamp}"
    report_dir = PHASE4_REPORT_ROOT / f"phase4_3_{stamp}"
    ensure_decoupled_features(feature_run_id, Path(args.checkpoint), horizons, args.precompute_batch_size, args.num_workers)
    conditions: list[dict[str, Any]] = []
    for mode in ["z_only", "z_p_slip", "z_force", "z_force_slip", "full"]:
        exp = f"phase4_3_{mode}_{stamp}"
        summary = train_future_head(feature_run_id, exp, mode, horizons, args.max_epochs, args.train_batch_size, args.lr, args.hidden_dim, args.dropout, args.seed, args.wandb_mode, report_dir)
        summary["condition"] = mode
        conditions.append(summary)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "phase": "phase4_3_world_model_input_ablation",
        "stamp": stamp,
        "checkpoint": str(Path(args.checkpoint)),
        "feature_run_id": feature_run_id,
        "conditions": conditions,
        "notes": [
            "Same MAE decoupled checkpoint, split, horizons, optimizer, and epoch budget are used for all input modes.",
            "full adds causal delta_force to z_t, force-derived scalars, and current slip probability.",
        ],
        "raw_data_modified": False,
    }
    json_path = report_dir / "phase4_3_world_model_input_ablation_report.json"
    md_path = report_dir / "phase4_3_world_model_input_ablation_report.md"
    payload["json_path"] = str(json_path)
    payload["md_path"] = str(md_path)
    write_json(json_path, payload)
    md_path.write_text(render_comparison_report("Phase4-3 World-model Input Ablation", payload), encoding="utf-8")
    print(md_path)


def command_late_fusion(args: argparse.Namespace) -> None:
    stamp = args.stamp or now_stamp()
    horizons = tuple(int(h) for h in args.horizons)
    report_dir = PHASE4_REPORT_ROOT / f"phase4_2_{stamp}"
    dec_feature_run = args.decoupled_feature_run_id or f"phase4_2_decoupled_features_{stamp}"
    sep_feature_run = args.separate_feature_run_id or f"phase4_2_separate_features_{stamp}"
    force_exp = resolve_experiment(args.force_exp, p2.A_FORCE_EXPS["mae"])
    slip_exp = resolve_experiment(args.slip_exp, p2.A_SLIP_EXPS["mae"])
    ensure_decoupled_features(dec_feature_run, Path(args.checkpoint), horizons, args.precompute_batch_size, args.num_workers)
    ensure_separate_features(sep_feature_run, Path(args.checkpoint), force_exp, slip_exp, horizons, args.precompute_batch_size, args.num_workers)
    specs = [
        ("separate_late_fusion", sep_feature_run, "z_force_slip"),
        ("decoupled_joint", dec_feature_run, "z_force_slip"),
        ("decoupled_joint_dynamics", dec_feature_run, "full"),
    ]
    conditions: list[dict[str, Any]] = []
    for condition, feature_run, mode in specs:
        exp = f"phase4_2_{condition}_{stamp}"
        summary = train_future_head(feature_run, exp, mode, horizons, args.max_epochs, args.train_batch_size, args.lr, args.hidden_dim, args.dropout, args.seed, args.wandb_mode, report_dir)
        summary["condition"] = condition
        conditions.append(summary)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "phase": "phase4_2_separate_late_fusion_vs_decoupled_joint",
        "stamp": stamp,
        "decoupled_checkpoint": str(Path(args.checkpoint)),
        "separate_force_experiment": str(force_exp),
        "separate_slip_experiment": str(slip_exp),
        "conditions": conditions,
        "notes": [
            "Separate late fusion uses the SPARSH-style force-only and slip-only MAE tasks for current predictions, then trains only a lightweight future-stability head.",
            "Decoupled joint uses one MAE decoupled multitask checkpoint; dynamics additionally includes causal delta_force.",
        ],
        "raw_data_modified": False,
    }
    json_path = report_dir / "phase4_2_late_fusion_vs_joint_report.json"
    md_path = report_dir / "phase4_2_late_fusion_vs_joint_report.md"
    payload["json_path"] = str(json_path)
    payload["md_path"] = str(md_path)
    write_json(json_path, payload)
    md_path.write_text(render_comparison_report("Phase4-2 Separate Late Fusion vs Decoupled Joint", payload), encoding="utf-8")
    print(md_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    evs = sub.add_parser("eval-separate")
    evs.add_argument("--encoder", default="mae")
    evs.add_argument("--force-exp", default=None)
    evs.add_argument("--slip-exp", default=None)
    evs.add_argument("--out-json", required=True)
    evs.add_argument("--batch-size", type=int, default=128)
    evs.add_argument("--num-workers", type=int, default=2)
    evs.set_defaults(func=command_eval_separate)

    evd = sub.add_parser("eval-decoupled")
    evd.add_argument("--checkpoint", required=True)
    evd.add_argument("--out-json", required=True)
    evd.add_argument("--batch-size", type=int, default=128)
    evd.add_argument("--num-workers", type=int, default=2)
    evd.set_defaults(func=command_eval_decoupled)

    th = sub.add_parser("train-head")
    th.add_argument("--feature-kind", choices=["decoupled", "separate"], required=True)
    th.add_argument("--feature-run-id", required=True)
    th.add_argument("--checkpoint", required=True, help="MAE decoupled checkpoint used for z_t and/or decoupled predictions")
    th.add_argument("--force-exp", default=None)
    th.add_argument("--slip-exp", default=None)
    th.add_argument("--experiment-name", required=True)
    th.add_argument("--input-mode", choices=sorted(INPUT_MODES), required=True)
    th.add_argument("--report-dir", required=True)
    th.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    th.add_argument("--precompute-batch-size", type=int, default=128)
    th.add_argument("--train-batch-size", type=int, default=1024)
    th.add_argument("--num-workers", type=int, default=2)
    th.add_argument("--max-epochs", type=int, default=40)
    th.add_argument("--lr", type=float, default=1.0e-3)
    th.add_argument("--hidden-dim", type=int, default=512)
    th.add_argument("--dropout", type=float, default=0.1)
    th.add_argument("--seed", type=int, default=42)
    th.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=os.environ.get("WANDB_MODE", "online"))
    th.set_defaults(func=command_train_head)

    p43 = sub.add_parser("input-ablation")
    p43.add_argument("--checkpoint", required=True)
    p43.add_argument("--stamp", default=None)
    p43.add_argument("--feature-run-id", default=None)
    p43.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    p43.add_argument("--precompute-batch-size", type=int, default=128)
    p43.add_argument("--train-batch-size", type=int, default=1024)
    p43.add_argument("--num-workers", type=int, default=2)
    p43.add_argument("--max-epochs", type=int, default=40)
    p43.add_argument("--lr", type=float, default=1.0e-3)
    p43.add_argument("--hidden-dim", type=int, default=512)
    p43.add_argument("--dropout", type=float, default=0.1)
    p43.add_argument("--seed", type=int, default=42)
    p43.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=os.environ.get("WANDB_MODE", "online"))
    p43.set_defaults(func=command_input_ablation)

    p42 = sub.add_parser("late-fusion")
    p42.add_argument("--checkpoint", required=True)
    p42.add_argument("--force-exp", default=None)
    p42.add_argument("--slip-exp", default=None)
    p42.add_argument("--stamp", default=None)
    p42.add_argument("--decoupled-feature-run-id", default=None)
    p42.add_argument("--separate-feature-run-id", default=None)
    p42.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    p42.add_argument("--precompute-batch-size", type=int, default=128)
    p42.add_argument("--train-batch-size", type=int, default=1024)
    p42.add_argument("--num-workers", type=int, default=2)
    p42.add_argument("--max-epochs", type=int, default=40)
    p42.add_argument("--lr", type=float, default=1.0e-3)
    p42.add_argument("--hidden-dim", type=int, default=512)
    p42.add_argument("--dropout", type=float, default=0.1)
    p42.add_argument("--seed", type=int, default=42)
    p42.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=os.environ.get("WANDB_MODE", "online"))
    p42.set_defaults(func=command_late_fusion)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
