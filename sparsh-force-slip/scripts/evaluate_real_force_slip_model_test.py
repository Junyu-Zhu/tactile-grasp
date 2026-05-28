#!/usr/bin/env python3
"""Evaluate trained force-slip/future-instability heads on real robot tactile image sequences.

This script is intentionally read-only for datasets and model checkpoints. It evaluates
sequence-level behavior because the real capture set has outcome names but no per-frame
force/slip labels.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
SPARSH_REPO = Path('/home/zjy/document/sparsh')
if str(SPARSH_REPO) not in sys.path:
    sys.path.insert(0, str(SPARSH_REPO))

import phase2_b_multitask as p2  # noqa: E402
import phase3_2_world_model as wm  # noqa: E402
from tactile_ssl.data.digit.utils import compute_diff, get_resize_transform  # noqa: E402

REPO = Path('/home/zjy/document/tactile-grasp')
WORKSPACE = REPO / 'sparsh-force-slip'
DEFAULT_DATASET = Path('/vla1/zjy/tactile_dataset/real-force-slip-model-test-dataset')
DEFAULT_STAGE1_CKPT = Path('/vla1/zjy/sparsh_runs/force_slip_phase2/phase_lambda_decoupled_mae_lam010_20260528_000000/mae_decoupled_multitask/checkpoints/epoch-0030.pth')
DEFAULT_STAGE2_CKPT = Path('/vla1/zjy/sparsh_runs/force_slip_phase4/phase_lambda_best_future_features_20260528_000000/heads/phase_lambda_best_full_20260528_000000/checkpoints/best.pth')
DEFAULT_REPORT_ROOT = WORKSPACE / 'reports/real_force_slip_model_test'
FORCE_SCALE = torch.tensor([1.5, 1.5, 2.0], dtype=torch.float32)


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
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding='utf-8')


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert('RGB'))


def preprocess_pair(cur_path: Path, prev_path: Path, bg: np.ndarray | None, transform) -> torch.Tensor:
    frames = []
    for path in (cur_path, prev_path):
        img = read_rgb(path)
        if bg is not None:
            img = compute_diff(img, bg, offset=0.5)
        h, w, _ = img.shape
        if h < w:
            # Match Sparsh preprocessing: rotate landscape GS frames to portrait.
            img = np.asarray(Image.fromarray(img).rotate(-90, expand=True))
            h, w, _ = img.shape
        ratio = 4 / 3
        if h / w != ratio:
            h2, w2 = int(h / ratio), w
            img = img[int((h - h2) / 2): int((h + h2) / 2), int((w - w2) / 2): int((w + w2) / 2)]
        frames.append(transform(Image.fromarray(img).convert('RGB')))
    return torch.cat(frames, dim=0)


def parse_sequence_name(name: str) -> tuple[str, str]:
    s = name.replace('-', '_').lower()
    if 'slip_fail' in s:
        outcome = 'slip_fail'
        obj = s.split('_slip_fail')[0]
    elif 'slip_success' in s:
        outcome = 'slip_success'
        obj = s.split('_slip_success')[0]
    elif 'hard_fail' in s:
        outcome = 'hard_fail'
        obj = s.split('_hard_fail')[0]
    elif s.endswith('_fail'):
        outcome = 'hard_fail'
        obj = s.rsplit('_fail', 1)[0]
    elif 'success' in s:
        outcome = 'success'
        obj = s.split('_success')[0]
    else:
        outcome = 'unknown'
        obj = re.split(r'[_-]', s)[0]
    return obj, outcome


def safe_mean(values: list[float]) -> float | None:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.mean(vals)) if vals else None


def top_quantile_mean(arr: np.ndarray, q: float = 0.9) -> float | None:
    if arr.size == 0:
        return None
    thr = np.quantile(arr, q)
    return float(arr[arr >= thr].mean())


@dataclass
class SequenceSide:
    sequence: str
    side: str
    object_name: str
    outcome: str
    image_paths: list[Path]


def discover_sequences(dataset_root: Path) -> list[SequenceSide]:
    out: list[SequenceSide] = []
    for seq_dir in sorted([p for p in dataset_root.iterdir() if p.is_dir()]):
        obj, outcome = parse_sequence_name(seq_dir.name)
        for side in ('left', 'right'):
            side_dir = seq_dir / side
            if not side_dir.exists():
                continue
            paths = sorted([p for p in side_dir.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}])
            if paths:
                out.append(SequenceSide(seq_dir.name, side, obj, outcome, paths))
    return out


def load_future_head(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location='cpu', weights_only=False)
    state = payload['model_state']
    in_dim = int(state['net.1.weight'].shape[1])
    hidden_dim = int(state['net.1.weight'].shape[0])
    horizons = tuple(int(h) for h in payload.get('horizons', [1, 3, 5]))
    model = wm.MLP(in_dim, hidden_dim, len(horizons), dropout=0.1).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, payload


@torch.no_grad()
def evaluate_sequence_side(
    item: SequenceSide,
    stage1: torch.nn.Module,
    future_head: torch.nn.Module | None,
    future_payload: dict[str, Any] | None,
    device: torch.device,
    batch_size: int,
    frame_stride_context: int,
    resize: tuple[int, int],
    use_first_frame_bg: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    transform = get_resize_transform(list(resize))
    paths = item.image_paths
    bg = read_rgb(paths[0]) if use_first_frame_bg and paths else None
    start = min(frame_stride_context, max(0, len(paths) - 1))
    valid_indices = list(range(start, len(paths)))
    rows: list[dict[str, Any]] = []
    if not valid_indices:
        return rows, {
            'sequence': item.sequence, 'side': item.side, 'object': item.object_name, 'outcome': item.outcome,
            'n_images': len(paths), 'n_eval_frames': 0, 'error': 'too_few_frames',
        }

    for chunk_start in tqdm(range(0, len(valid_indices), batch_size), desc=f'{item.sequence}/{item.side}', leave=False):
        inds = valid_indices[chunk_start: chunk_start + batch_size]
        tensors = []
        for idx in inds:
            prev_idx = max(0, idx - frame_stride_context)
            tensors.append(preprocess_pair(paths[idx], paths[prev_idx], bg, transform))
        x = torch.stack(tensors, dim=0).to(device, non_blocking=True)
        z_tokens = stage1.encoder(x)
        out = stage1.decoder(z_tokens)
        z = wm.pool_latent(z_tokens).float()
        force_n = out['force'] * FORCE_SCALE.to(device)
        slip_probs = F.softmax(out['slip'], dim=1)[:, 1]
        force_np = force_n.detach().cpu().numpy()
        p_slip_np = slip_probs.detach().cpu().numpy()
        z_cpu = z.detach().cpu()
        for local_i, idx in enumerate(inds):
            fx, fy, fz = [float(v) for v in force_np[local_i]]
            ft = float(math.sqrt(fx * fx + fy * fy))
            fn = float(abs(fz))
            fmag = float(math.sqrt(fx * fx + fy * fy + fz * fz))
            row = {
                'sequence': item.sequence,
                'side': item.side,
                'object': item.object_name,
                'outcome': item.outcome,
                'frame_index': int(idx),
                'frame_name': paths[idx].name,
                'time_s_at_60fps': float(idx / 60.0),
                'Fx_pred_N': fx,
                'Fy_pred_N': fy,
                'Fz_pred_N': fz,
                'Fn_pred_N': fn,
                'Ft_pred_N': ft,
                'Fmag_pred_N': fmag,
                'Ft_over_Fn_pred': float(ft / (fn + 1.0e-6)),
                'p_slip_current': float(p_slip_np[local_i]),
                '_z': z_cpu[local_i],
            }
            rows.append(row)

    # Causal predicted-force delta for future head: approximate unavailable real force delta with Stage-I force change.
    by_frame = {r['frame_index']: r for r in rows}
    for r in rows:
        prev = by_frame.get(max(start, int(r['frame_index']) - frame_stride_context))
        if prev is None:
            r['dFx_causal_N'] = 0.0
            r['dFy_causal_N'] = 0.0
            r['dFz_causal_N'] = 0.0
        else:
            r['dFx_causal_N'] = float(r['Fx_pred_N'] - prev['Fx_pred_N'])
            r['dFy_causal_N'] = float(r['Fy_pred_N'] - prev['Fy_pred_N'])
            r['dFz_causal_N'] = float(r['Fz_pred_N'] - prev['Fz_pred_N'])

    if future_head is not None and future_payload is not None:
        horizons = [int(h) for h in future_payload.get('horizons', [1, 3, 5])]
        aux_names = list(future_payload.get('aux_names', []))
        for chunk_start in range(0, len(rows), batch_size):
            chunk = rows[chunk_start: chunk_start + batch_size]
            z_batch = torch.stack([r['_z'] for r in chunk], dim=0).float()
            aux = torch.tensor([[float(r[name]) for name in aux_names] for r in chunk], dtype=torch.float32)
            x = torch.cat([z_batch, aux], dim=1).to(device)
            p_stable = torch.sigmoid(future_head(x)).detach().cpu().numpy()
            for r, probs in zip(chunk, p_stable):
                for h, p_s in zip(horizons, probs):
                    r[f'p_stable_H{h}'] = float(p_s)
                    r[f'p_instability_H{h}'] = float(1.0 - p_s)

    for r in rows:
        r.pop('_z', None)

    p_slip = np.array([r['p_slip_current'] for r in rows], dtype=float)
    fn = np.array([r['Fn_pred_N'] for r in rows], dtype=float)
    ft = np.array([r['Ft_pred_N'] for r in rows], dtype=float)
    fmag = np.array([r['Fmag_pred_N'] for r in rows], dtype=float)
    ratio = np.array([r['Ft_over_Fn_pred'] for r in rows], dtype=float)
    summary: dict[str, Any] = {
        'sequence': item.sequence,
        'side': item.side,
        'object': item.object_name,
        'outcome': item.outcome,
        'n_images': len(paths),
        'n_eval_frames': len(rows),
        'duration_s_at_60fps': float(len(paths) / 60.0),
        'eval_start_frame': int(start),
        'slip_prob_mean': float(p_slip.mean()),
        'slip_prob_max': float(p_slip.max()),
        'slip_prob_top10_mean': top_quantile_mean(p_slip, 0.9),
        'slip_pred_rate_p_ge_0p5': float((p_slip >= 0.5).mean()),
        'first_slip_frame_p_ge_0p5': int(rows[int(np.argmax(p_slip >= 0.5))]['frame_index']) if (p_slip >= 0.5).any() else None,
        'Fn_mean_N': float(fn.mean()), 'Fn_max_N': float(fn.max()),
        'Ft_mean_N': float(ft.mean()), 'Ft_max_N': float(ft.max()),
        'Fmag_mean_N': float(fmag.mean()), 'Fmag_max_N': float(fmag.max()),
        'Ft_over_Fn_mean': float(ratio.mean()), 'Ft_over_Fn_max': float(ratio.max()),
    }
    if future_payload is not None:
        for h in future_payload.get('horizons', [1, 3, 5]):
            key = f'p_instability_H{int(h)}'
            if key in rows[0]:
                arr = np.array([r[key] for r in rows], dtype=float)
                summary[f'{key}_mean'] = float(arr.mean())
                summary[f'{key}_max'] = float(arr.max())
                summary[f'{key}_top10_mean'] = top_quantile_mean(arr, 0.9)
                summary[f'{key}_rate_ge_0p5'] = float((arr >= 0.5).mean())
    return rows, summary


def aggregate_by(rows: list[dict[str, Any]], keys: tuple[str, ...], metric_keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for key, vals in sorted(groups.items()):
        rec = {k: v for k, v in zip(keys, key)}
        rec['n_sequence_sides'] = len(vals)
        for m in metric_keys:
            rec[m] = safe_mean([v.get(m) for v in vals])
        out.append(rec)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fieldnames = list(rows[0].keys())
    for r in rows[1:]:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def fmt(v: Any, digits: int = 3) -> str:
    if v is None:
        return 'n/a'
    try:
        return f'{float(v):.{digits}f}'
    except Exception:
        return str(v)


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        '# Real Force-Slip Model Test Report', '',
        f"- generated_at: `{payload['generated_at']}`",
        f"- dataset: `{payload['dataset_root']}`",
        f"- stage_i_checkpoint: `{payload['stage_i_checkpoint']}`",
        f"- stage_ii_future_checkpoint: `{payload.get('stage_ii_future_checkpoint')}`",
        f"- frame_rate: `60 fps`; input_resolution: `640x480`; evaluated_frames: `{payload['total_eval_frames']}`",
        f"- preprocessing: first frame background subtraction = `{payload['config']['use_first_frame_bg']}`, context stride = `{payload['config']['frame_stride_context']}` frames (~{payload['config']['frame_stride_context']/60:.3f}s), resized to `{payload['config']['resize']}`.",
        '',
        '## Important interpretation notes', '',
        '- The real dataset has sequence-level outcome names but no per-frame force/slip labels, so this is a weak-label deployment sanity check, not a calibrated accuracy/RMSE benchmark.',
        '- `hard_fail`/`*_fail` may include no contact; a contact-level slip model can output low slip for no-touch frames, so hard failures should be interpreted as out-of-distribution/no-contact cases rather than ordinary slip failures.',
        '- The future head was trained with force-delta inputs; here the unavailable real force delta is approximated by causal differences of Stage-I predicted force.',
        '',
        '## Outcome-level summary', '',
        '| outcome | seq-sides | pSlip mean | pSlip max | pSlip top10 | slip rate | H1 inst mean | H1 inst max | H5 inst mean | Fmag mean | Ft/Fn mean |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in payload['outcome_summary']:
        lines.append(f"| {r['outcome']} | {r['n_sequence_sides']} | {fmt(r.get('slip_prob_mean'))} | {fmt(r.get('slip_prob_max'))} | {fmt(r.get('slip_prob_top10_mean'))} | {fmt(r.get('slip_pred_rate_p_ge_0p5'))} | {fmt(r.get('p_instability_H1_mean'))} | {fmt(r.get('p_instability_H1_max'))} | {fmt(r.get('p_instability_H5_mean'))} | {fmt(r.get('Fmag_mean_N'))} | {fmt(r.get('Ft_over_Fn_mean'))} |")
    lines += ['', '## Object-level summary', '', '| object | outcome | seq-sides | pSlip mean | pSlip top10 | H1 inst mean | H5 inst mean | Fmag mean |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in payload['object_outcome_summary']:
        lines.append(f"| {r['object']} | {r['outcome']} | {r['n_sequence_sides']} | {fmt(r.get('slip_prob_mean'))} | {fmt(r.get('slip_prob_top10_mean'))} | {fmt(r.get('p_instability_H1_mean'))} | {fmt(r.get('p_instability_H5_mean'))} | {fmt(r.get('Fmag_mean_N'))} |")
    lines += ['', '## Sequence-side summary', '', '| sequence | side | outcome | frames | pSlip mean | pSlip max | pSlip top10 | slip rate | H1 inst mean | H5 inst mean | Fmag mean | first slip frame |', '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in payload['sequence_summaries']:
        lines.append(f"| {r['sequence']} | {r['side']} | {r['outcome']} | {r['n_eval_frames']} | {fmt(r.get('slip_prob_mean'))} | {fmt(r.get('slip_prob_max'))} | {fmt(r.get('slip_prob_top10_mean'))} | {fmt(r.get('slip_pred_rate_p_ge_0p5'))} | {fmt(r.get('p_instability_H1_mean'))} | {fmt(r.get('p_instability_H5_mean'))} | {fmt(r.get('Fmag_mean_N'))} | {r.get('first_slip_frame_p_ge_0p5')} |")
    interp = payload.get('interpretation', [])
    if interp:
        lines += ['', '## Automatic interpretation', '']
        lines += [f'- {x}' for x in interp]
    lines += ['', '## Output files', '']
    for k, v in payload.get('output_files', {}).items():
        lines.append(f'- {k}: `{v}`')
    return '\n'.join(lines) + '\n'


def build_interpretation(outcome_summary: list[dict[str, Any]]) -> list[str]:
    by = {r['outcome']: r for r in outcome_summary}
    notes = []
    if 'success' in by and 'slip_fail' in by:
        notes.append(f"Slip-fail vs success top-10% current slip probability: {fmt(by['slip_fail'].get('slip_prob_top10_mean'))} vs {fmt(by['success'].get('slip_prob_top10_mean'))}.")
    if 'slip_success' in by and 'success' in by:
        notes.append(f"Slip-success vs success mean H1 instability: {fmt(by['slip_success'].get('p_instability_H1_mean'))} vs {fmt(by['success'].get('p_instability_H1_mean'))}.")
    if 'hard_fail' in by:
        notes.append('Hard-fail sequences are treated separately because no-contact failures can produce low slip even when grasp outcome is bad.')
    if not notes:
        notes.append('Inspect sequence-level rows because the weak labels are too sparse for a single automatic conclusion.')
    return notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-root', type=Path, default=DEFAULT_DATASET)
    ap.add_argument('--stage1-checkpoint', type=Path, default=DEFAULT_STAGE1_CKPT)
    ap.add_argument('--stage2-checkpoint', type=Path, default=DEFAULT_STAGE2_CKPT)
    ap.add_argument('--report-root', type=Path, default=DEFAULT_REPORT_ROOT)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--frame-stride-context', type=int, default=5)
    ap.add_argument('--resize', type=int, nargs=2, default=[320, 240])
    ap.add_argument('--no-bg', action='store_true', help='Disable first-frame background subtraction.')
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = args.report_root / stamp
    report_dir.mkdir(parents=True, exist_ok=True)
    if not args.dataset_root.exists():
        raise FileNotFoundError(args.dataset_root)
    if not args.stage1_checkpoint.exists():
        raise FileNotFoundError(args.stage1_checkpoint)
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda:0')
    stage1, stage1_payload = p2.load_b_checkpoint(args.stage1_checkpoint, device)
    stage1.eval()
    future_head = None
    future_payload = None
    if args.stage2_checkpoint and args.stage2_checkpoint.exists():
        future_head, future_payload = load_future_head(args.stage2_checkpoint, device)

    sequence_items = discover_sequences(args.dataset_root)
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for item in sequence_items:
        rows, summary = evaluate_sequence_side(
            item, stage1, future_head, future_payload, device, args.batch_size,
            args.frame_stride_context, tuple(args.resize), not args.no_bg,
        )
        all_rows.extend(rows)
        summaries.append(summary)

    metric_keys = [
        'slip_prob_mean', 'slip_prob_max', 'slip_prob_top10_mean', 'slip_pred_rate_p_ge_0p5',
        'p_instability_H1_mean', 'p_instability_H1_max', 'p_instability_H1_top10_mean', 'p_instability_H1_rate_ge_0p5',
        'p_instability_H3_mean', 'p_instability_H5_mean', 'p_instability_H5_max', 'p_instability_H5_top10_mean',
        'Fn_mean_N', 'Ft_mean_N', 'Fmag_mean_N', 'Ft_over_Fn_mean',
    ]
    outcome_summary = aggregate_by(summaries, ('outcome',), metric_keys)
    object_outcome_summary = aggregate_by(summaries, ('object', 'outcome'), metric_keys)

    per_frame_csv = report_dir / 'real_force_slip_per_frame_predictions.csv'
    sequence_csv = report_dir / 'real_force_slip_sequence_summary.csv'
    outcome_csv = report_dir / 'real_force_slip_outcome_summary.csv'
    object_csv = report_dir / 'real_force_slip_object_outcome_summary.csv'
    write_csv(per_frame_csv, all_rows)
    write_csv(sequence_csv, summaries)
    write_csv(outcome_csv, outcome_summary)
    write_csv(object_csv, object_outcome_summary)

    payload = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'dataset_root': str(args.dataset_root),
        'stage_i_checkpoint': str(args.stage1_checkpoint),
        'stage_i_train_config': stage1_payload.get('train_config', {}),
        'stage_ii_future_checkpoint': str(args.stage2_checkpoint) if future_payload is not None else None,
        'stage_ii_payload_meta': {k: v for k, v in (future_payload or {}).items() if k != 'model_state'},
        'config': {
            'batch_size': args.batch_size,
            'frame_stride_context': args.frame_stride_context,
            'resize': list(args.resize),
            'use_first_frame_bg': not args.no_bg,
            'device': str(device),
        },
        'n_sequence_sides': len(sequence_items),
        'total_eval_frames': len(all_rows),
        'sequence_summaries': summaries,
        'outcome_summary': outcome_summary,
        'object_outcome_summary': object_outcome_summary,
        'interpretation': build_interpretation(outcome_summary),
        'output_files': {
            'json': str(report_dir / 'real_force_slip_model_test_report.json'),
            'markdown': str(report_dir / 'real_force_slip_model_test_report.md'),
            'per_frame_csv': str(per_frame_csv),
            'sequence_csv': str(sequence_csv),
            'outcome_csv': str(outcome_csv),
            'object_outcome_csv': str(object_csv),
        },
        'raw_data_modified': False,
    }
    write_json(report_dir / 'real_force_slip_model_test_report.json', payload)
    (report_dir / 'real_force_slip_model_test_report.md').write_text(render_md(payload), encoding='utf-8')
    print(json.dumps({'report_dir': str(report_dir), 'n_sequence_sides': len(sequence_items), 'total_eval_frames': len(all_rows)}, indent=2))


if __name__ == '__main__':
    main()
