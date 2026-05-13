#!/usr/bin/env python3
"""Aggregate-evaluate sphere-only and all-source slip checkpoints on shared val sets."""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

SPARSH_REPO = Path('/home/zjy/document/sparsh')
if str(SPARSH_REPO) not in sys.path:
    sys.path.insert(0, str(SPARSH_REPO))

import hydra  # noqa: E402

REPO = Path('/home/zjy/document/tactile-grasp')
RUN_ID = (REPO / 'sparsh-force-slip/phase1_run_id.txt').read_text().strip()
REPORT_DIR = REPO / 'sparsh-force-slip/reports/phase1' / RUN_ID
EXP_ROOT = Path('/vla1/zjy/sparsh_runs/experiments')

MODEL_EXPS = {
    'dinov2_sphere_only': '2026.05.12_04-39_phase1_gsmini_20260512_043331_dinov2_slip_gsmini_20260512_043652',
    'mae_sphere_only': '2026.05.12_04-39_phase1_gsmini_20260512_043331_mae_slip_gsmini_20260512_043652',
    'dinov2_allsource': '2026.05.13_01-21_phase1_gsmini_20260512_043331_dinov2_slip_allsource_diag_gsmini_20260513_012000',
    'mae_allsource': '2026.05.13_01-21_phase1_gsmini_20260512_043331_mae_slip_allsource_diag_gsmini_20260513_012000',
}

SPHERE_VAL = [f'sphere_batch_{i}_val' for i in range(1, 7)]
ALLSOURCE_VAL = ['flat_batch_1_val', 'flat_batch_2_val', 'sharp_batch_1_val', 'sharp_batch_2_val'] + SPHERE_VAL


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def load_wandb_summary(exp_dir: Path) -> dict[str, Any]:
    paths = list(exp_dir.glob('wandb/run-*/files/wandb-summary.json'))
    if not paths:
        return {}
    return json.loads(paths[0].read_text())


def load_cfg(exp_dir: Path, checkpoint: Path) -> Any:
    cfg = OmegaConf.load(exp_dir / 'config.yaml')
    cfg.task.checkpoint_task = str(checkpoint)
    # Avoid resolving hydra runtime interpolations that are irrelevant here.
    return cfg


def instantiate_model(cfg: Any, device: torch.device):
    model = hydra.utils.instantiate(cfg.task)
    model.to(device)
    model.eval()
    return model


def dataset_family(name: str) -> str:
    if name.startswith('flat_'):
        return 'flat'
    if name.startswith('sharp_'):
        return 'sharp'
    if name.startswith('sphere_'):
        return 'sphere'
    return 'other'


def summarize_arrays(gt: np.ndarray, pred: np.ndarray, probs: np.ndarray, force_gt_n: np.ndarray, force_pred_n: np.ndarray) -> dict[str, Any]:
    pred_label = pred.astype(int)
    gt = gt.astype(int)
    cm = confusion_matrix(gt, pred_label, labels=[0, 1])
    err = force_pred_n - force_gt_n
    return {
        'n_samples': int(len(gt)),
        'positive_count': int(gt.sum()),
        'positive_ratio': float(gt.mean()) if len(gt) else 0.0,
        'no_slip_baseline_accuracy': float((gt == 0).mean()) if len(gt) else 0.0,
        'accuracy': float(accuracy_score(gt, pred_label)) if len(gt) else 0.0,
        'balanced_accuracy': float(balanced_accuracy_score(gt, pred_label)) if len(np.unique(gt)) > 1 else None,
        'precision_slip': float(precision_score(gt, pred_label, zero_division=0)),
        'recall_slip': float(recall_score(gt, pred_label, zero_division=0)),
        'f1_slip': float(f1_score(gt, pred_label, zero_division=0)),
        'confusion_matrix_labels_0_1': cm.tolist(),
        'delta_force_rmse_N': np.sqrt(np.mean(err ** 2, axis=0)).tolist(),
        'delta_force_mae_N': np.mean(np.abs(err), axis=0).tolist(),
        'delta_force_rmse_mean_N': float(np.mean(np.sqrt(np.mean(err ** 2, axis=0)))),
        'delta_force_mae_mean_N': float(np.mean(np.mean(np.abs(err), axis=0))),
        'mean_pred_slip_probability': float(probs[:, 1].mean()) if len(probs) else 0.0,
    }


def eval_model(model, cfg: Any, dataset_names: list[str], device: torch.device, batch_size: int = 100) -> dict[str, Any]:
    all_gt, all_pred, all_probs, all_force_gt, all_force_pred = [], [], [], [], []
    per_dataset = {}
    per_family_arrays: dict[str, dict[str, list[np.ndarray]]] = {}
    for dataset_name in dataset_names:
        ds = hydra.utils.instantiate(cfg.data.dataset, dataset_name=dataset_name)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)
        d_gt, d_pred, d_probs, d_fgt, d_fpred = [], [], [], [], []
        for batch in tqdm(dl, desc=dataset_name, leave=False, disable=True):
            x = batch['image'].to(device, non_blocking=True)
            gt = batch['slip_label'].cpu().numpy().astype(int)
            force_gt = batch['delta_force'].cpu().numpy()
            scale = batch['delta_force_scale'].cpu().numpy()
            with torch.no_grad():
                out = model(x)
                probs = F.softmax(out['slip'], dim=1).detach().cpu().numpy()
                pred = probs.argmax(axis=1).astype(int)
                force_pred = out['force'].detach().cpu().numpy()
            d_gt.append(gt)
            d_pred.append(pred)
            d_probs.append(probs)
            d_fgt.append(force_gt * scale)
            d_fpred.append(force_pred * scale)
        gt = np.concatenate(d_gt)
        pred = np.concatenate(d_pred)
        probs = np.concatenate(d_probs)
        fgt = np.concatenate(d_fgt)
        fpred = np.concatenate(d_fpred)
        per_dataset[dataset_name] = summarize_arrays(gt, pred, probs, fgt, fpred)
        fam = dataset_family(dataset_name)
        fam_store = per_family_arrays.setdefault(fam, {'gt': [], 'pred': [], 'probs': [], 'fgt': [], 'fpred': []})
        fam_store['gt'].append(gt); fam_store['pred'].append(pred); fam_store['probs'].append(probs); fam_store['fgt'].append(fgt); fam_store['fpred'].append(fpred)
        all_gt.append(gt); all_pred.append(pred); all_probs.append(probs); all_force_gt.append(fgt); all_force_pred.append(fpred)
    aggregate = summarize_arrays(np.concatenate(all_gt), np.concatenate(all_pred), np.concatenate(all_probs), np.concatenate(all_force_gt), np.concatenate(all_force_pred))
    per_family = {}
    for fam, arrays in per_family_arrays.items():
        per_family[fam] = summarize_arrays(
            np.concatenate(arrays['gt']), np.concatenate(arrays['pred']), np.concatenate(arrays['probs']), np.concatenate(arrays['fgt']), np.concatenate(arrays['fpred'])
        )
    return {'aggregate': aggregate, 'per_family': per_family, 'per_dataset': per_dataset}


def pct(new: float, old: float) -> float | None:
    if old == 0:
        return None
    return (new / old - 1.0) * 100.0


def render_md(report: dict[str, Any]) -> str:
    rows = report['comparison']
    lines = [
        '# Slip Diagnostic Training Comparison',
        '',
        f"- generated_at: `{report['generated_at']}`",
        f"- run_id: `{report['run_id']}`",
        '- Scope: aggregate validation evaluation from saved `epoch-0051.pth` checkpoints, not W&B final-batch summaries.',
        '- Eval sets:',
        f"  - `sphere_val`: {', '.join(SPHERE_VAL)}",
        f"  - `allsource_val`: {', '.join(ALLSOURCE_VAL)}",
        '',
        '## Main aggregate comparison',
        '',
        '| encoder | train data | eval set | n | pos ratio | acc | F1 slip | recall slip | ΔF RMSE mean N |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for key in ['dinov2_sphere_only__sphere_val','dinov2_allsource__sphere_val','dinov2_sphere_only__allsource_val','dinov2_allsource__allsource_val','mae_sphere_only__sphere_val','mae_allsource__sphere_val','mae_sphere_only__allsource_val','mae_allsource__allsource_val']:
        r = report['evaluations'][key]['aggregate']
        model_key = report['evaluations'][key]['model_key']
        encoder = 'DINOv2' if 'dinov2' in model_key else 'MAE'
        train_data = 'all-source' if 'allsource' in model_key else 'sphere-only'
        eval_set = report['evaluations'][key]['eval_set']
        lines.append(f"| {encoder} | {train_data} | {eval_set} | {r['n_samples']} | {r['positive_ratio']:.4f} | {r['accuracy']:.4f} | {r['f1_slip']:.4f} | {r['recall_slip']:.4f} | {r['delta_force_rmse_mean_N']:.4f} |")
    lines.extend(['', '## Direct deltas', ''])
    for item in rows:
        lines.append(f"### {item['title']}")
        lines.append('')
        lines.append(f"- accuracy: `{item['old']['accuracy']:.4f}` -> `{item['new']['accuracy']:.4f}` ({item['delta']['accuracy_abs']:+.4f})")
        lines.append(f"- F1 slip: `{item['old']['f1_slip']:.4f}` -> `{item['new']['f1_slip']:.4f}` ({item['delta']['f1_abs']:+.4f})")
        lines.append(f"- recall slip: `{item['old']['recall_slip']:.4f}` -> `{item['new']['recall_slip']:.4f}` ({item['delta']['recall_abs']:+.4f})")
        lines.append(f"- mean Δforce RMSE N: `{item['old']['delta_force_rmse_mean_N']:.4f}` -> `{item['new']['delta_force_rmse_mean_N']:.4f}` ({item['delta']['rmse_mean_rel_pct']:+.1f}%)")
        lines.append('')
    lines.extend([
        '## Conclusion',
        '',
        report['conclusion'],
        '',
        '## Notes',
        '',
        '- W&B `wandb-summary.json` values are final logged batch values in this code path; the table above re-evaluates full validation sets from checkpoints for a fairer diagnostic.',
        '- The all-source run is still diagnostic. Do not replace the sphere-only slip baseline unless the chosen evaluation policy accepts the all-source tradeoff.',
    ])
    return '\n'.join(lines) + '\n'


def main() -> None:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    results: dict[str, Any] = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'run_id': RUN_ID,
        'device': str(device),
        'model_experiments': MODEL_EXPS,
        'eval_sets': {'sphere_val': SPHERE_VAL, 'allsource_val': ALLSOURCE_VAL},
        'evaluations': {},
        'wandb_summaries': {},
    }
    for model_key, exp_name in MODEL_EXPS.items():
        exp_dir = EXP_ROOT / exp_name
        ckpt = exp_dir / 'checkpoints/epoch-0051.pth'
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        cfg = load_cfg(exp_dir, ckpt)
        results['wandb_summaries'][model_key] = load_wandb_summary(exp_dir)
        print(f'Loading {model_key} from {ckpt}', flush=True)
        model = instantiate_model(cfg, device)
        for eval_set, names in [('sphere_val', SPHERE_VAL), ('allsource_val', ALLSOURCE_VAL)]:
            print(f'Evaluating {model_key} on {eval_set}', flush=True)
            res = eval_model(model, cfg, names, device)
            res['model_key'] = model_key
            res['eval_set'] = eval_set
            results['evaluations'][f'{model_key}__{eval_set}'] = res
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def comp(title, old_key, new_key):
        old = results['evaluations'][old_key]['aggregate']
        new = results['evaluations'][new_key]['aggregate']
        return {
            'title': title,
            'old_key': old_key,
            'new_key': new_key,
            'old': old,
            'new': new,
            'delta': {
                'accuracy_abs': new['accuracy'] - old['accuracy'],
                'f1_abs': new['f1_slip'] - old['f1_slip'],
                'recall_abs': new['recall_slip'] - old['recall_slip'],
                'rmse_mean_abs_N': new['delta_force_rmse_mean_N'] - old['delta_force_rmse_mean_N'],
                'rmse_mean_rel_pct': pct(new['delta_force_rmse_mean_N'], old['delta_force_rmse_mean_N']),
            },
        }
    results['comparison'] = [
        comp('DINOv2: all-source training vs sphere-only training on sphere_val', 'dinov2_sphere_only__sphere_val', 'dinov2_allsource__sphere_val'),
        comp('DINOv2: all-source training vs sphere-only training on allsource_val', 'dinov2_sphere_only__allsource_val', 'dinov2_allsource__allsource_val'),
        comp('MAE: all-source training vs sphere-only training on sphere_val', 'mae_sphere_only__sphere_val', 'mae_allsource__sphere_val'),
        comp('MAE: all-source training vs sphere-only training on allsource_val', 'mae_sphere_only__allsource_val', 'mae_allsource__allsource_val'),
    ]
    # Conservative interpretation with the metrics we just computed.
    din_all = results['evaluations']['dinov2_allsource__allsource_val']['aggregate']
    din_sph_all = results['evaluations']['dinov2_sphere_only__allsource_val']['aggregate']
    mae_all = results['evaluations']['mae_allsource__allsource_val']['aggregate']
    mae_sph_all = results['evaluations']['mae_sphere_only__allsource_val']['aggregate']
    results['conclusion'] = (
        'All-source slip training is useful as a diagnostic but should not automatically replace the sphere-only baseline. '
        f"On allsource_val, DINOv2 F1 changes from {din_sph_all['f1_slip']:.4f} to {din_all['f1_slip']:.4f}, "
        f"while MAE F1 changes from {mae_sph_all['f1_slip']:.4f} to {mae_all['f1_slip']:.4f}. "
        'Use the detailed table to decide the tradeoff; if the formal goal is broad flat+sharp+sphere coverage, all-source training is the relevant candidate, '
        'but if the formal goal is maximum sphere-only slip performance, keep the existing sphere-only baseline.'
    )
    out_json = REPORT_DIR / 'diagnostic_allsource_slip_comparison_aggregate_eval.json'
    out_md = REPORT_DIR / 'diagnostic_allsource_slip_comparison_aggregate_eval.md'
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=json_default) + '\n', encoding='utf-8')
    out_md.write_text(render_md(results), encoding='utf-8')
    print(out_json)
    print(out_md)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    main()
