"""Phase 5 diagnostic evaluation against local Sparsh assets.

This script is intentionally conservative: the supplied Sparsh model directory
contains SSL encoder checkpoints, not force/slip task-decoder checkpoints, so
it does not claim official TacBench metrics.  It checks the current Phase5
cube dataset with the Sparsh dataloader, runs a frozen encoder feature pass,
and trains tiny sklearn probes only as a data/representation separability
sanity check.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

SPARSH_REPO = Path("/home/zjy/Documents/grasp/sparsh")
SPARSH_DATA_ROOT = Path("/home/zjy/Documents/dataset1/sparsh/tactile_datasets")
SPARSH_MODEL_ROOT = Path("/home/zjy/Documents/dataset1/sparsh/sparsh_models")
DEFAULT_PHASE5_ROOT = Path("sim_dataset/phase5_cube_force_slip/sparsh_export_v2")
DEFAULT_DATASET_NAME = "cube_force_slip_v2"
DEFAULT_REPORT_ROOT = Path("artifacts/phase5_cube_force_slip/reports")
DEFAULT_ENCODER_CKPT = SPARSH_MODEL_ROOT / "sparsh-dino-small" / "dino_vitsmall.ckpt"

if SPARSH_REPO.as_posix() not in sys.path:
    sys.path.insert(0, SPARSH_REPO.as_posix())

from tactile_ssl.data.vision_based_forces_slip_probes import VisionForceSlipDataset  # noqa: E402
from tactile_ssl.model import vit_base, vit_small  # noqa: E402


def _dataset_config(dataset_root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        sensor="gelsight",
        remove_bg=True,
        out_format="concat_ch_img",
        num_frames=2,
        frame_stride=5,
        path_dataset=dataset_root.as_posix(),
        slip_horizon=0,
        max_abs_forceXYZ=[1.5, 1.5, 2.0],
        max_delta_forceXYZ=[0.8, 0.8, 0.8],
        transforms=SimpleNamespace(resize=[320, 240]),
    )


def _infer_encoder(checkpoint: Path) -> tuple[str, str, torch.nn.Module]:
    name = checkpoint.as_posix().lower()
    if "small" in name or "vitsmall" in name:
        size = "small"
        model = vit_small(img_size=[320, 240], in_chans=6, pos_embed_fn="sinusoidal", num_register_tokens=1)
    else:
        size = "base"
        model = vit_base(img_size=[320, 240], in_chans=6, pos_embed_fn="sinusoidal", num_register_tokens=1)

    if "dino" in name:
        encoder_key = "teacher_encoder.backbone"
        family = "dino"
    elif "jepa" in name:
        encoder_key = "target_encoder.backbone"
        family = "jepa"
    else:
        encoder_key = "encoder"
        family = "mae_or_other"

    # These checkpoints are local user-provided model artifacts. PyTorch >=2.6
    # defaults to weights_only=True, but Sparsh checkpoints contain scheduler
    # objects; use weights_only=False only for this trusted local path.
    checkpoint_obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint_obj.get("model", checkpoint_obj)
    target_keys = [key for key in state.keys() if encoder_key in key]
    if not target_keys and encoder_key.endswith(".backbone"):
        encoder_key = encoder_key.replace(".backbone", "")
        target_keys = [key for key in state.keys() if encoder_key in key]
    if not target_keys:
        raise KeyError(f"Could not find encoder keys containing {encoder_key!r} in {checkpoint}")
    loaded = {key.replace(f"{encoder_key}.", ""): state[key] for key in target_keys}
    missing, unexpected = model.load_state_dict(loaded, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys while loading encoder: {unexpected[:5]}")
    return f"sparsh-{family}-{size}", family, model


def _collect_metadata(dataset: VisionForceSlipDataset) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in range(len(dataset)):
        item = dataset.idx2traj[idx]
        traj_id = item["trajectory"]
        sample = int(item["sample"])
        traj = dataset.trajectories[traj_id]
        normal = float(np.asarray(traj.get("normal_force_n", [0.0]))[sample])
        valid_force = bool(np.asarray(traj.get("valid_force", [False]))[sample])
        valid_slip = bool(np.asarray(traj.get("valid_slip", [False]))[sample])
        slip_label = int(np.asarray(traj.get("slip_label", [0]))[sample])
        rows.append(
            {
                "dataset_idx": idx,
                "trajectory": traj_id,
                "sample": sample,
                "source_group": str(traj.get("source_group", "")),
                "is_phase5_new": traj_id.startswith("phase5_"),
                "normal_force_n": normal,
                "valid_force": valid_force,
                "valid_slip": valid_slip,
                "slip_label": slip_label,
                "nonzero_force": normal > 1e-6,
            }
        )
    return rows


def _extract_embeddings(dataset: VisionForceSlipDataset, model: torch.nn.Module, *, batch_size: int, device: str) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.to(device)
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            tokens = model(image)
            emb = tokens.mean(dim=1)
            chunks.append(emb.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def _force_probe(X: np.ndarray, rows: list[dict[str, Any]]) -> dict[str, Any]:
    y = np.asarray([row["normal_force_n"] for row in rows], dtype=np.float64)
    valid = np.asarray([row["valid_force"] for row in rows], dtype=bool)
    is_phase5 = np.asarray([row["is_phase5_new"] for row in rows], dtype=bool)
    train = valid & ~is_phase5
    test = valid & is_phase5
    result: dict[str, Any] = {
        "train_samples": int(train.sum()),
        "test_samples_phase5_new": int(test.sum()),
        "test_nonzero_force_samples": int((test & (y > 1e-6)).sum()),
    }
    if train.sum() < 20 or test.sum() < 10 or np.unique(y[train]).size < 2:
        result["status"] = "SKIPPED_INSUFFICIENT_SPLIT"
        return result

    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 4, 17)))
    model.fit(X[train], y[train])
    pred = model.predict(X[test])
    baseline = np.full_like(y[test], float(y[train].mean()))
    result.update(
        {
            "status": "PASS_DIAGNOSTIC_LINEAR_PROBE",
            "target": "normal_force_n_only_not_full_xyz",
            "phase5_all_valid_rmse_n": _rmse(y[test], pred),
            "phase5_all_valid_mae_n": float(mean_absolute_error(y[test], pred)),
            "phase5_all_valid_r2": float(r2_score(y[test], pred)) if np.unique(y[test]).size > 1 else None,
            "baseline_train_mean_rmse_n": _rmse(y[test], baseline),
            "baseline_train_mean_mae_n": float(mean_absolute_error(y[test], baseline)),
            "train_target_mean_n": float(y[train].mean()),
            "test_target_mean_n": float(y[test].mean()),
            "test_target_max_n": float(y[test].max()),
        }
    )
    contact_test = test & (y > 1e-6)
    if contact_test.sum() >= 5:
        pred_all = np.zeros_like(y)
        pred_all[test] = pred
        baseline_all = np.zeros_like(y)
        baseline_all[test] = baseline
        result.update(
            {
                "phase5_nonzero_rmse_n": _rmse(y[contact_test], pred_all[contact_test]),
                "phase5_nonzero_mae_n": float(mean_absolute_error(y[contact_test], pred_all[contact_test])),
                "baseline_nonzero_rmse_n": _rmse(y[contact_test], baseline_all[contact_test]),
                "baseline_nonzero_mae_n": float(mean_absolute_error(y[contact_test], baseline_all[contact_test])),
            }
        )
    return result


def _slip_probe(X: np.ndarray, rows: list[dict[str, Any]]) -> dict[str, Any]:
    y = np.asarray([row["slip_label"] for row in rows], dtype=np.int64)
    valid = np.asarray([row["valid_slip"] for row in rows], dtype=bool)
    is_phase5 = np.asarray([row["is_phase5_new"] for row in rows], dtype=bool)
    train = valid & ~is_phase5
    test = valid & is_phase5
    result: dict[str, Any] = {
        "train_samples": int(train.sum()),
        "test_samples_phase5_new": int(test.sum()),
        "train_positive": int((y[train] == 1).sum()),
        "train_negative": int((y[train] == 0).sum()),
        "test_positive": int((y[test] == 1).sum()),
        "test_negative": int((y[test] == 0).sum()),
    }
    if train.sum() < 20 or test.sum() < 10 or np.unique(y[train]).size < 2 or np.unique(y[test]).size < 2:
        result["status"] = "SKIPPED_INSUFFICIENT_SPLIT_OR_CLASSES"
        return result

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs"),
    )
    model.fit(X[train], y[train])
    pred = model.predict(X[test])
    majority = int(np.bincount(y[train]).argmax())
    base = np.full_like(y[test], majority)
    result.update(
        {
            "status": "PASS_DIAGNOSTIC_LINEAR_PROBE",
            "phase5_accuracy": float(accuracy_score(y[test], pred)),
            "phase5_balanced_accuracy": float(balanced_accuracy_score(y[test], pred)),
            "phase5_f1": float(f1_score(y[test], pred, zero_division=0)),
            "phase5_precision": float(precision_score(y[test], pred, zero_division=0)),
            "phase5_recall": float(recall_score(y[test], pred, zero_division=0)),
            "phase5_confusion_matrix_tn_fp_fn_tp": [int(v) for v in confusion_matrix(y[test], pred, labels=[0, 1]).ravel()],
            "baseline_majority_label": majority,
            "baseline_accuracy": float(accuracy_score(y[test], base)),
            "baseline_balanced_accuracy": float(balanced_accuracy_score(y[test], base)),
            "baseline_f1": float(f1_score(y[test], base, zero_division=0)),
        }
    )
    return result


def _model_inventory(model_root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(model_root.glob("sparsh-*/*")):
        if path.suffix in {".ckpt", ".safetensors"}:
            records.append({"path": path.as_posix(), "size_bytes": path.stat().st_size})
    return records


def _official_dataset_inventory(data_root: Path) -> dict[str, Any]:
    gs_force = data_root / "Gelsight-mini" / "gelsight-force-estimation"
    return {
        "sparsh_data_root": data_root.as_posix(),
        "gelsight_force_root": gs_force.as_posix(),
        "gelsight_force_exists": gs_force.exists(),
        "gelsight_force_dataset_slip_forces_files": len(list(gs_force.glob("*/*/dataset_slip_forces.pkl"))),
        "gelsight_force_dataset_gelsight_files": len(list(gs_force.glob("*/*/dataset_gelsight*.pkl"))),
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    force = report["diagnostics"].get("force_probe", {})
    slip = report["diagnostics"].get("slip_probe", {})
    manifest = report["phase5_manifest"]
    lines = [
        "# Phase 5 Sparsh model/data effect diagnostic",
        "",
        "## Local Sparsh assets",
        "",
        f"- Sparsh repo: `{report['sparsh_repo']}`",
        f"- Sparsh data root: `{report['official_dataset_inventory']['sparsh_data_root']}`",
        f"- Sparsh model root: `{report['sparsh_model_root']}`",
        f"- Encoder used for diagnostic: `{report['encoder']['checkpoint']}`",
        f"- Encoder family/size: `{report['encoder']['name']}`",
        "",
        "The supplied model directory contains SSL encoder checkpoints. It does not contain a task decoder checkpoint for `t1_force` or `t2_slip`, so this report does not claim official Sparsh/TacBench metrics.",
        "",
        "## Phase 5 dataset coverage",
        "",
        f"- Dataset root: `{manifest['dataset_root']}`",
        f"- Frames: {manifest['counts']['frames']}",
        f"- Trajectories: {manifest['counts']['trajectories']}",
        f"- Force-valid frames: {manifest['counts']['force_valid_frames']} / {manifest['counts']['frames']}",
        f"- Slip-valid frames: {manifest['counts']['slip_valid_frames']} / {manifest['counts']['frames']}",
        f"- Valid slip positives / negatives: {manifest['counts']['slip_positive_valid_frames']} / {manifest['counts']['slip_negative_valid_frames']}",
        f"- Gates: `{manifest['gates']}`",
        "",
        "## Frozen-Sparsh diagnostic probes",
        "",
        "Train split: older Phase3/Phase4 cube trajectories. Test split: newly collected `phase5_force_v2` and `phase5_slip_v2` trajectories. These probes are small sklearn heads on frozen encoder mean-pooled patch tokens, not the Sparsh task decoders.",
        "",
        "### Normal-force diagnostic",
        "",
        f"- Status: `{force.get('status')}`",
        f"- Train/test samples: {force.get('train_samples')} / {force.get('test_samples_phase5_new')}",
        f"- Test nonzero-force samples: {force.get('test_nonzero_force_samples')}",
    ]
    if force.get("status") == "PASS_DIAGNOSTIC_LINEAR_PROBE":
        lines += [
            f"- All valid RMSE: {force['phase5_all_valid_rmse_n']:.4f} N (baseline mean: {force['baseline_train_mean_rmse_n']:.4f} N)",
            f"- All valid MAE: {force['phase5_all_valid_mae_n']:.4f} N (baseline mean: {force['baseline_train_mean_mae_n']:.4f} N)",
            f"- Nonzero-force RMSE: {force.get('phase5_nonzero_rmse_n', float('nan')):.4f} N (baseline: {force.get('baseline_nonzero_rmse_n', float('nan')):.4f} N)",
            f"- Test max normal force: {force['test_target_max_n']:.4f} N",
            "- Interpretation: this only tests scalar normal force; Phase 5 still has no shear-force ground truth.",
        ]
    lines += [
        "",
        "### Slip diagnostic",
        "",
        f"- Status: `{slip.get('status')}`",
        f"- Train positives/negatives: {slip.get('train_positive')} / {slip.get('train_negative')}",
        f"- Test positives/negatives: {slip.get('test_positive')} / {slip.get('test_negative')}",
    ]
    if slip.get("status") == "PASS_DIAGNOSTIC_LINEAR_PROBE":
        lines += [
            f"- Balanced accuracy: {slip['phase5_balanced_accuracy']:.4f} (majority baseline: {slip['baseline_balanced_accuracy']:.4f})",
            f"- F1: {slip['phase5_f1']:.4f} (majority baseline: {slip['baseline_f1']:.4f})",
            f"- Precision/recall: {slip['phase5_precision']:.4f} / {slip['phase5_recall']:.4f}",
            f"- Confusion matrix [TN, FP, FN, TP]: `{slip['phase5_confusion_matrix_tn_fp_fn_tp']}`",
        ]
    lines += [
        "",
        "## Verdict",
        "",
        "- Data format effect: good. The current Phase 5 export loads through Sparsh and has mask-aware labels.",
        "- Label effect: improved over Phase 4. Force/slip labels are simulation-valid with masks, but force is scalar-normal only and slip is contact-frame simulation slip, not real TacBench sliding-probe ground truth.",
        "- Model effect: limited. Encoder-only checkpoints can support representation diagnostics or future probe training, but official force/slip evaluation still requires a trained task decoder checkpoint or a new probe-training run.",
        "- Domain-gap risk: high. Sparsh's documented force/slip datasets are real GelSight/DIGIT probe trajectories with ATI force and 2 mm shear slides; Phase 5 data is Isaac/TacEx cube grasp contact. Treat results as cube-simulation diagnostics only.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Phase 5 data effect with local Sparsh assets.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_PHASE5_ROOT)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--sparsh-repo", type=Path, default=SPARSH_REPO)
    parser.add_argument("--sparsh-data-root", type=Path, default=SPARSH_DATA_ROOT)
    parser.add_argument("--sparsh-model-root", type=Path, default=SPARSH_MODEL_ROOT)
    parser.add_argument("--encoder-checkpoint", type=Path, default=DEFAULT_ENCODER_CKPT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    manifest_path = args.dataset_root / args.dataset_name / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = VisionForceSlipDataset(config=_dataset_config(args.dataset_root), dataset_name=args.dataset_name)
    rows = _collect_metadata(dataset)

    encoder_name, encoder_family, encoder = _infer_encoder(args.encoder_checkpoint)
    embeddings = _extract_embeddings(dataset, encoder, batch_size=args.batch_size, device=args.device)

    report = {
        "sparsh_repo": args.sparsh_repo.as_posix(),
        "sparsh_model_root": args.sparsh_model_root.as_posix(),
        "official_dataset_inventory": _official_dataset_inventory(args.sparsh_data_root),
        "model_inventory": _model_inventory(args.sparsh_model_root),
        "encoder": {
            "name": encoder_name,
            "family": encoder_family,
            "checkpoint": args.encoder_checkpoint.as_posix(),
            "device": args.device,
            "embedding_shape": list(embeddings.shape),
        },
        "phase5_manifest": manifest,
        "dataset_len_sparsh_loader": len(dataset),
        "diagnostics": {
            "force_probe": _force_probe(embeddings, rows),
            "slip_probe": _slip_probe(embeddings, rows),
        },
    }
    args.report_root.mkdir(parents=True, exist_ok=True)
    json_path = args.report_root / "sparsh_model_data_effect.json"
    md_path = args.report_root / "sparsh_model_data_effect.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    print(json.dumps({"json": json_path.as_posix(), "markdown": md_path.as_posix(), "diagnostics": report["diagnostics"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
