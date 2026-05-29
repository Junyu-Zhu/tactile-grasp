#!/usr/bin/env python3
"""Evaluate force/slip and future-instability predictions on sidecar-annotated real tactile sequences."""
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
IMAGE_EXTS = {'.png', '.jpg', '.jpeg'}


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


def frame_id(path: Path) -> int:
    m = re.match(r'^(\d+)', path.name)
    if not m:
        raise ValueError(f'Cannot parse frame id from {path}')
    return int(m.group(1))


def parse_sequence_name(name: str) -> tuple[str, str]:
    s = name.replace('-', '_').lower()
    if 'slip_fail' in s:
        return s.split('_slip_fail')[0], 'slip_fail'
    if 'slip_success' in s:
        return s.split('_slip_success')[0], 'slip_success'
    if 'hard_fail' in s:
        return s.split('_hard_fail')[0], 'hard_fail'
    if s.endswith('_fail'):
        return s.rsplit('_fail', 1)[0], 'hard_fail'
    if 'success' in s:
        return s.split('_success')[0], 'success'
    return re.split(r'[_-]', s)[0], 'unknown'


def preprocess_pair(cur_path: Path, prev_path: Path, bg: np.ndarray | None, transform) -> torch.Tensor:
    frames = []
    for path in (cur_path, prev_path):
        img = read_rgb(path)
        if bg is not None:
            img = compute_diff(img, bg, offset=0.5)
        h, w, _ = img.shape
        if h < w:
            img = np.asarray(Image.fromarray(img).rotate(-90, expand=True))
            h, w, _ = img.shape
        ratio = 4 / 3
        if h / w != ratio:
            h2, w2 = int(h / ratio), w
            img = img[int((h - h2) / 2): int((h + h2) / 2), int((w - w2) / 2): int((w + w2) / 2)]
        frames.append(transform(Image.fromarray(img).convert('RGB')))
    return torch.cat(frames, dim=0)


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
    frame_map: dict[int, Path]
    sidecar: dict[str, Any]
    sidecar_path: Path


def discover_sequences(dataset_root: Path) -> list[SequenceSide]:
    out: list[SequenceSide] = []
    for seq_dir in sorted([p for p in dataset_root.iterdir() if p.is_dir()]):
        sidecar_path = seq_dir / 'real_force_slip_sidecar.json'
        sidecar = json.loads(sidecar_path.read_text(encoding='utf-8')) if sidecar_path.exists() else {}
        obj, outcome = sidecar.get('object'), sidecar.get('outcome')
        if not obj or not outcome:
            obj, outcome = parse_sequence_name(seq_dir.name)
        for side in ('left', 'right'):
            sd = sidecar.get(side, {}) if isinstance(sidecar.get(side, {}), dict) else {}
            if sd.get('valid', True) is False:
                continue
            side_dir = seq_dir / side
            if not side_dir.exists():
                continue
            paths = sorted([p for p in side_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS], key=frame_id)
            if not paths:
                continue
            fmap = {frame_id(p): p for p in paths}
            out.append(SequenceSide(seq_dir.name, side, obj, outcome, paths, fmap, sd, sidecar_path))
    return out


def selected_frame_ids(item: SequenceSide) -> list[int]:
    ids = sorted(item.frame_map)
    lo = item.sidecar.get('eval_start')
    hi = item.sidecar.get('eval_end')
    if lo is None or hi is None:
        # Conservative fallback if a sidecar is incomplete: use all frames.
        return ids
    return [i for i in ids if int(lo) <= i <= int(hi)]


def nearest_frame_at_or_before(frame_ids: list[int], target: int) -> int:
    best = frame_ids[0]
    for fid in frame_ids:
        if fid <= target:
            best = fid
        else:
            break
    return best


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
    all_ids = sorted(item.frame_map)
    eval_ids = selected_frame_ids(item)
    bg = read_rgb(item.image_paths[0]) if use_first_frame_bg else None
    rows: list[dict[str, Any]] = []
    if not eval_ids:
        return rows, {'sequence': item.sequence, 'side': item.side, 'object': item.object_name, 'outcome': item.outcome, 'n_eval_frames': 0, 'error': 'empty_sidecar_window'}

    for chunk_start in tqdm(range(0, len(eval_ids), batch_size), desc=f'{item.sequence}/{item.side}', leave=False):
        ids = eval_ids[chunk_start: chunk_start + batch_size]
        tensors = []
        for fid in ids:
            prev_fid = nearest_frame_at_or_before(all_ids, max(all_ids[0], fid - frame_stride_context))
            tensors.append(preprocess_pair(item.frame_map[fid], item.frame_map[prev_fid], bg, transform))
        x = torch.stack(tensors, dim=0).to(device, non_blocking=True)
        z_tokens = stage1.encoder(x)
        out = stage1.decoder(z_tokens)
        z = wm.pool_latent(z_tokens).float()
        force_n = out['force'] * FORCE_SCALE.to(device)
        p_slip = F.softmax(out['slip'], dim=1)[:, 1]
        force_np = force_n.detach().cpu().numpy()
        p_slip_np = p_slip.detach().cpu().numpy()
        z_cpu = z.detach().cpu()
        for local_i, fid in enumerate(ids):
            fx, fy, fz = [float(v) for v in force_np[local_i]]
            ft = float(math.sqrt(fx * fx + fy * fy))
            fn = float(abs(fz))
            fmag = float(math.sqrt(fx * fx + fy * fy + fz * fz))
            rows.append({
                'sequence': item.sequence, 'side': item.side, 'object': item.object_name, 'outcome': item.outcome,
                'frame_index': int(fid), 'frame_name': item.frame_map[fid].name, 'time_s_at_60fps': float(fid / 60.0),
                'window_source': 'sidecar_eval_window',
                'contact_start': item.sidecar.get('contact_start'), 'stable_start': item.sidecar.get('stable_start'),
                'stable_end': item.sidecar.get('stable_end'), 'slip_start': item.sidecar.get('slip_start'),
                'drop_start': item.sidecar.get('drop_start'), 'eval_start': item.sidecar.get('eval_start'), 'eval_end': item.sidecar.get('eval_end'),
                'Fx_pred_N': fx, 'Fy_pred_N': fy, 'Fz_pred_N': fz, 'Fn_pred_N': fn, 'Ft_pred_N': ft,
                'Fmag_pred_N': fmag, 'Ft_over_Fn_pred': float(ft / (fn + 1.0e-6)),
                'p_slip_current': float(p_slip_np[local_i]), '_z': z_cpu[local_i],
            })

    by_frame = {r['frame_index']: r for r in rows}
    eval_id_set = sorted(by_frame)
    for r in rows:
        prev_fid = nearest_frame_at_or_before(eval_id_set, max(eval_id_set[0], int(r['frame_index']) - frame_stride_context))
        prev = by_frame[prev_fid]
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

    p_slip_arr = np.array([r['p_slip_current'] for r in rows], dtype=float)
    fn = np.array([r['Fn_pred_N'] for r in rows], dtype=float)
    ft = np.array([r['Ft_pred_N'] for r in rows], dtype=float)
    fmag = np.array([r['Fmag_pred_N'] for r in rows], dtype=float)
    ratio = np.array([r['Ft_over_Fn_pred'] for r in rows], dtype=float)
    summary: dict[str, Any] = {
        'sequence': item.sequence, 'side': item.side, 'object': item.object_name, 'outcome': item.outcome,
        'n_images': len(item.image_paths), 'n_eval_frames': len(rows),
        'eval_start': item.sidecar.get('eval_start'), 'eval_end': item.sidecar.get('eval_end'),
        'contact_start': item.sidecar.get('contact_start'), 'stable_start': item.sidecar.get('stable_start'),
        'stable_end': item.sidecar.get('stable_end'), 'slip_start': item.sidecar.get('slip_start'), 'drop_start': item.sidecar.get('drop_start'),
        'duration_s_at_60fps': float((max(eval_ids) - min(eval_ids) + 1) / 60.0),
        'slip_prob_mean': float(p_slip_arr.mean()), 'slip_prob_max': float(p_slip_arr.max()),
        'slip_prob_top10_mean': top_quantile_mean(p_slip_arr, 0.9),
        'slip_pred_rate_p_ge_0p5': float((p_slip_arr >= 0.5).mean()),
        'first_slip_frame_p_ge_0p5': int(rows[int(np.argmax(p_slip_arr >= 0.5))]['frame_index']) if (p_slip_arr >= 0.5).any() else None,
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
        rec['n_eval_frames'] = int(sum(v.get('n_eval_frames', 0) for v in vals))
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
        '# Real Force-Slip Model Test Report (Sidecar Window)', '',
        f"- generated_at: `{payload['generated_at']}`",
        f"- dataset: `{payload['dataset_root']}`",
        f"- stage_i_checkpoint: `{payload['stage_i_checkpoint']}`",
        f"- stage_ii_future_checkpoint: `{payload.get('stage_ii_future_checkpoint')}`",
        f"- window_source: `sidecar eval_start/eval_end`",
        f"- frame_rate: `60 fps`; evaluated_frames: `{payload['total_eval_frames']}`; sequence_sides: `{payload['n_sequence_sides']}`",
        f"- preprocessing: first frame background subtraction = `{payload['config']['use_first_frame_bg']}`, context stride = `{payload['config']['frame_stride_context']}` frames, resized to `{payload['config']['resize']}`.",
        '', '## Interpretation notes', '',
        '- This run uses manually annotated sidecar windows, so it avoids most no-contact and irrelevant release frames.',
        '- The real dataset still has weak sequence-level labels, not per-frame force/slip ground truth; results are deployment diagnostics rather than calibrated accuracy/RMSE.',
        '- Stage-II future-instability uses predicted force deltas on real data, so treat it as exploratory unless it agrees with slip/force trends.',
        '', '## Outcome-level summary', '',
        '| outcome | sides | frames | pSlip mean | pSlip max | slip rate | H1 inst mean | H5 inst mean | Fmag mean | Ft/Fn mean |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in payload['outcome_summary']:
        lines.append(f"| {r['outcome']} | {r['n_sequence_sides']} | {r['n_eval_frames']} | {fmt(r.get('slip_prob_mean'))} | {fmt(r.get('slip_prob_max'))} | {fmt(r.get('slip_pred_rate_p_ge_0p5'))} | {fmt(r.get('p_instability_H1_mean'))} | {fmt(r.get('p_instability_H5_mean'))} | {fmt(r.get('Fmag_mean_N'))} | {fmt(r.get('Ft_over_Fn_mean'))} |")
    lines += ['', '## Sequence-side summary', '', '| sequence | side | outcome | window | frames | pSlip mean | slip rate | H1 inst mean | H5 inst mean | Fmag mean | first pSlip>=0.5 |', '|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in payload['sequence_summaries']:
        window = f"{r.get('eval_start')}-{r.get('eval_end')}"
        lines.append(f"| {r['sequence']} | {r['side']} | {r['outcome']} | {window} | {r['n_eval_frames']} | {fmt(r.get('slip_prob_mean'))} | {fmt(r.get('slip_pred_rate_p_ge_0p5'))} | {fmt(r.get('p_instability_H1_mean'))} | {fmt(r.get('p_instability_H5_mean'))} | {fmt(r.get('Fmag_mean_N'))} | {r.get('first_slip_frame_p_ge_0p5')} |")
    lines += ['', '## Automatic conclusion', '']
    for x in payload.get('interpretation', []):
        lines.append(f'- {x}')
    lines += ['', '## Output files', '']
    for k, v in payload.get('output_files', {}).items():
        lines.append(f'- {k}: `{v}`')
    return '\n'.join(lines) + '\n'


def build_interpretation(outcome_summary: list[dict[str, Any]]) -> list[str]:
    by = {r['outcome']: r for r in outcome_summary}
    notes = []
    if 'success' in by and 'slip_fail' in by:
        notes.append(f"Sidecar-window pSlip mean: success={fmt(by['success'].get('slip_prob_mean'))}, slip_fail={fmt(by['slip_fail'].get('slip_prob_mean'))}.")
        notes.append(f"Sidecar-window slip-rate: success={fmt(by['success'].get('slip_pred_rate_p_ge_0p5'))}, slip_fail={fmt(by['slip_fail'].get('slip_pred_rate_p_ge_0p5'))}.")
    if 'slip_success' in by and 'slip_fail' in by:
        notes.append(f"Slip-success vs slip-fail pSlip mean: {fmt(by['slip_success'].get('slip_prob_mean'))} vs {fmt(by['slip_fail'].get('slip_prob_mean'))}.")
    if 'success' in by and 'slip_fail' in by:
        notes.append(f"Mean force magnitude: success={fmt(by['success'].get('Fmag_mean_N'))} N, slip_fail={fmt(by['slip_fail'].get('Fmag_mean_N'))} N.")
    return notes or ['Inspect sequence-level rows; no automatic comparison was available.']


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-root', type=Path, default=DEFAULT_DATASET)
    ap.add_argument('--stage1-checkpoint', type=Path, default=DEFAULT_STAGE1_CKPT)
    ap.add_argument('--stage2-checkpoint', type=Path, default=DEFAULT_STAGE2_CKPT)
    ap.add_argument('--report-root', type=Path, default=DEFAULT_REPORT_ROOT)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--frame-stride-context', type=int, default=5)
    ap.add_argument('--resize', type=int, nargs=2, default=[320, 240])
    ap.add_argument('--no-bg', action='store_true')
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = args.report_root / stamp
    report_dir.mkdir(parents=True, exist_ok=True)
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
        rows, summary = evaluate_sequence_side(item, stage1, future_head, future_payload, device, args.batch_size, args.frame_stride_context, tuple(args.resize), not args.no_bg)
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
        'config': {'batch_size': args.batch_size, 'frame_stride_context': args.frame_stride_context, 'resize': list(args.resize), 'use_first_frame_bg': not args.no_bg, 'device': str(device), 'window_source': 'sidecar_eval_window'},
        'n_sequence_sides': len(sequence_items), 'total_eval_frames': len(all_rows),
        'sequence_summaries': summaries, 'outcome_summary': outcome_summary, 'object_outcome_summary': object_outcome_summary,
        'interpretation': build_interpretation(outcome_summary),
        'output_files': {
            'json': str(report_dir / 'real_force_slip_model_test_report.json'),
            'markdown': str(report_dir / 'real_force_slip_model_test_report.md'),
            'per_frame_csv': str(per_frame_csv), 'sequence_csv': str(sequence_csv),
            'outcome_csv': str(outcome_csv), 'object_outcome_csv': str(object_csv),
        },
        'raw_data_modified': False,
    }
    write_json(report_dir / 'real_force_slip_model_test_report.json', payload)
    (report_dir / 'real_force_slip_model_test_report.md').write_text(render_md(payload), encoding='utf-8')

    # Plot outcome bars if matplotlib is available.
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        labels = [r['outcome'] for r in outcome_summary]
        metrics = [('slip_prob_mean', 'Mean pSlip'), ('slip_pred_rate_p_ge_0p5', 'Slip rate'), ('p_instability_H1_mean', 'Mean H1 instability'), ('Fmag_mean_N', 'Mean Fmag (N)')]
        fig, axs = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        for ax, (key, title) in zip(axs.flat, metrics):
            ax.bar(labels, [r.get(key) or 0 for r in outcome_summary])
            ax.set_title(title)
            ax.tick_params(axis='x', rotation=25)
            ax.grid(axis='y', alpha=0.25)
        fig.suptitle('Sidecar-window real tactile model summary')
        png = report_dir / 'real_force_slip_outcome_bars.png'
        fig.savefig(png, dpi=180)
        payload['output_files']['outcome_bars'] = str(png)
        write_json(report_dir / 'real_force_slip_model_test_report.json', payload)
        (report_dir / 'real_force_slip_model_test_report.md').write_text(render_md(payload), encoding='utf-8')
    except Exception as exc:
        print(f'WARN plot skipped: {exc}', file=sys.stderr)

    print(json.dumps({'report_dir': str(report_dir), 'n_sequence_sides': len(sequence_items), 'total_eval_frames': len(all_rows)}, indent=2))


if __name__ == '__main__':
    main()
