#!/usr/bin/env python3
"""Evaluate real tactile force/slip deployment data with sidecar stable/slip windows.

The local real dataset is manually annotated per sequence and per sensor side with
1-based frame-id windows:
  - stable_windows: contact is stable, target future_instability=0
  - slip_windows: slip/incipient instability, target future_instability=1

This script evaluates every frame inside each annotated window and reports results
by window type, object, sequence, and left/right side.
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


def parse_object_from_sequence(name: str) -> str:
    s = name.replace('-', '_').lower()
    return re.split(r'[_-]', s)[0]


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


def safe_mean(values: list[Any]) -> float | None:
    vals = []
    for v in values:
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if not math.isnan(fv):
            vals.append(fv)
    return float(np.mean(vals)) if vals else None


def top_quantile_mean(arr: np.ndarray, q: float = 0.9) -> float | None:
    if arr.size == 0:
        return None
    thr = np.quantile(arr, q)
    return float(arr[arr >= thr].mean())


def auroc(y: np.ndarray, score: np.ndarray) -> float | None:
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    # Mann-Whitney U / pairwise probability, tie = 0.5.
    total = 0.0
    for p in pos:
        total += float((p > neg).sum()) + 0.5 * float((p == neg).sum())
    return float(total / (len(pos) * len(neg)))


def auprc(y: np.ndarray, score: np.ndarray) -> float | None:
    if y.sum() == 0 or y.sum() == len(y):
        return None
    order = np.argsort(-score)
    yy = y[order]
    tp = np.cumsum(yy == 1)
    fp = np.cumsum(yy == 0)
    precision = tp / np.maximum(tp + fp, 1)
    return float(precision[yy == 1].mean()) if (yy == 1).any() else None


def binary_metrics(rows: list[dict[str, Any]], score_key: str, threshold: float = 0.5) -> dict[str, Any]:
    vals = [(int(r['target_instability']), float(r[score_key])) for r in rows if r.get(score_key) is not None]
    if not vals:
        return {'score': score_key, 'n': 0}
    y = np.array([v[0] for v in vals], dtype=int)
    s = np.array([v[1] for v in vals], dtype=float)
    pred = (s >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and (precision + recall) else None
    return {
        'score': score_key,
        'n': int(len(y)),
        'n_stable': int((y == 0).sum()),
        'n_slip': int((y == 1).sum()),
        'threshold': threshold,
        'stable_mean': safe_mean(s[y == 0].tolist()),
        'slip_mean': safe_mean(s[y == 1].tolist()),
        'slip_minus_stable_gap': (safe_mean(s[y == 1].tolist()) - safe_mean(s[y == 0].tolist())) if (y == 0).any() and (y == 1).any() else None,
        'accuracy_at_0p5': float((pred == y).mean()),
        'precision_at_0p5': precision,
        'recall_at_0p5': recall,
        'f1_at_0p5': f1,
        'auroc': auroc(y, s),
        'auprc': auprc(y, s),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
    }


@dataclass
class EvalWindow:
    window_type: str
    window_index: int
    target_instability: int
    start: int
    end: int
    notes: str = ''


@dataclass
class SequenceSide:
    sequence: str
    side: str
    object_name: str
    image_paths: list[Path]
    frame_map: dict[int, Path]
    side_meta: dict[str, Any]
    sidecar: dict[str, Any]
    sidecar_path: Path


def discover_sequences(dataset_root: Path) -> list[SequenceSide]:
    out: list[SequenceSide] = []
    for seq_dir in sorted([p for p in dataset_root.iterdir() if p.is_dir()]):
        sidecar_path = seq_dir / 'real_force_slip_sidecar.json'
        sidecar = json.loads(sidecar_path.read_text(encoding='utf-8')) if sidecar_path.exists() else {}
        obj = sidecar.get('object') or parse_object_from_sequence(seq_dir.name)
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
            out.append(SequenceSide(seq_dir.name, side, obj, paths, fmap, sd, sidecar, sidecar_path))
    return out


def int_frame_value(v: Any) -> int:
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        m = re.match(r'^(\d+)', v.strip())
        if m:
            return int(m.group(1))
    raise ValueError(f'Invalid frame value: {v!r}')


def selected_windows(item: SequenceSide) -> list[EvalWindow]:
    ids = sorted(item.frame_map)
    lo_all, hi_all = ids[0], ids[-1]
    windows: list[EvalWindow] = []
    for window_type, target in [('stable', 0), ('slip', 1)]:
        raw_key = f'{window_type}_windows'
        raw_windows = item.side_meta.get(raw_key, []) or []
        for idx, raw in enumerate(raw_windows):
            start = max(lo_all, int_frame_value(raw.get('start')))
            end = min(hi_all, int_frame_value(raw.get('end')))
            if end < start:
                continue
            windows.append(EvalWindow(window_type, idx, target, start, end, str(raw.get('notes', ''))))
    return windows


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


def summarize_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    metric_keys = [
        'p_slip_current', 'Fn_pred_N', 'Ft_pred_N', 'Fmag_pred_N', 'Ft_over_Fn_pred',
        'p_instability_H1', 'p_instability_H3', 'p_instability_H5',
    ]
    for key, vals in sorted(groups.items()):
        rec = {k: v for k, v in zip(keys, key)}
        rec['n_frames'] = len(vals)
        rec['n_windows'] = len({(v['sequence'], v['side'], v['window_type'], v['window_index']) for v in vals})
        rec['n_sequence_sides'] = len({(v['sequence'], v['side']) for v in vals})
        for m in metric_keys:
            if any(m in v for v in vals):
                arr = np.array([float(v[m]) for v in vals if v.get(m) is not None], dtype=float)
                if arr.size:
                    rec[f'{m}_mean'] = float(arr.mean())
                    rec[f'{m}_max'] = float(arr.max())
                    rec[f'{m}_top10_mean'] = top_quantile_mean(arr, 0.9)
                    if m.startswith('p_'):
                        rec[f'{m}_rate_ge_0p5'] = float((arr >= 0.5).mean())
        out.append(rec)
    return out


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
) -> list[dict[str, Any]]:
    transform = get_resize_transform(list(resize))
    all_ids = sorted(item.frame_map)
    windows = selected_windows(item)
    if not windows:
        return []
    bg = read_rgb(item.image_paths[0]) if use_first_frame_bg else None
    tasks: list[tuple[EvalWindow, int]] = []
    for w in windows:
        for fid in all_ids:
            if w.start <= fid <= w.end:
                tasks.append((w, fid))
    rows: list[dict[str, Any]] = []
    for chunk_start in tqdm(range(0, len(tasks), batch_size), desc=f'{item.sequence}/{item.side}', leave=False):
        chunk = tasks[chunk_start: chunk_start + batch_size]
        tensors = []
        for _w, fid in chunk:
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
        for local_i, (w, fid) in enumerate(chunk):
            fx, fy, fz = [float(v) for v in force_np[local_i]]
            ft = float(math.sqrt(fx * fx + fy * fy))
            fn = float(abs(fz))
            fmag = float(math.sqrt(fx * fx + fy * fy + fz * fz))
            rows.append({
                'sequence': item.sequence,
                'object': item.object_name,
                'side': item.side,
                'window_type': w.window_type,
                'window_index': w.window_index,
                'target_instability': w.target_instability,
                'window_start': w.start,
                'window_end': w.end,
                'window_notes': w.notes,
                'frame_index': int(fid),
                'frame_name': item.frame_map[fid].name,
                'time_s_at_60fps': float((fid - 1) / 60.0),
                'schema': item.sidecar.get('schema'),
                'trial_result_for_reference_only': item.sidecar.get('trial_result_for_reference_only'),
                'Fx_pred_N': fx, 'Fy_pred_N': fy, 'Fz_pred_N': fz,
                'Fn_pred_N': fn, 'Ft_pred_N': ft, 'Fmag_pred_N': fmag,
                'Ft_over_Fn_pred': float(ft / (fn + 1.0e-6)),
                'p_slip_current': float(p_slip_np[local_i]), '_z': z_cpu[local_i],
            })

    # Compute causal force deltas within each sequence side using the closest evaluated previous frame.
    by_frame: dict[int, dict[str, Any]] = {}
    for r in rows:
        by_frame.setdefault(int(r['frame_index']), r)
    eval_id_set = sorted(by_frame)
    for r in rows:
        prev_fid = nearest_frame_at_or_before(eval_id_set, max(eval_id_set[0], int(r['frame_index']) - frame_stride_context))
        prev = by_frame[prev_fid]
        r['dFx_causal_N'] = float(r['Fx_pred_N'] - prev['Fx_pred_N'])
        r['dFy_causal_N'] = float(r['Fy_pred_N'] - prev['Fy_pred_N'])
        r['dFz_causal_N'] = float(r['Fz_pred_N'] - prev['Fz_pred_N'])

    if future_head is not None and future_payload is not None and rows:
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
    return rows


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
        '# Real Force-Slip Model Test Report (Stable/Slip Windows)', '',
        f"- generated_at: `{payload['generated_at']}`",
        f"- dataset: `{payload['dataset_root']}`",
        f"- stage_i_checkpoint: `{payload['stage_i_checkpoint']}`",
        f"- stage_ii_future_checkpoint: `{payload.get('stage_ii_future_checkpoint')}`",
        '- labels: `stable_windows -> target_instability=0`, `slip_windows -> target_instability=1`',
        f"- evaluated_frames: `{payload['total_eval_frames']}`; sequence_sides: `{payload['n_sequence_sides']}`; windows: `{payload['n_windows']}`",
        f"- preprocessing: first frame background subtraction = `{payload['config']['use_first_frame_bg']}`, context stride = `{payload['config']['frame_stride_context']}` frames, resized to `{payload['config']['resize']}`.",
        '', '## Interpretation notes', '',
        '- This run uses manually annotated local stable/slip windows and reports left/right sensors separately.',
        '- The labels are diagnostic deployment labels from the real image sequences, not calibrated force ground truth.',
        '- `p_slip_current` is the Stage-I slip probability; `p_instability_H*` is the Stage-II future-instability probability.',
        '', '## Binary separation metrics', '',
        '| score | n | stable mean | slip mean | gap | AUROC | AUPRC | F1@0.5 | Acc@0.5 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in payload['binary_metrics']:
        lines.append(f"| {r['score']} | {r.get('n', 0)} | {fmt(r.get('stable_mean'))} | {fmt(r.get('slip_mean'))} | {fmt(r.get('slip_minus_stable_gap'))} | {fmt(r.get('auroc'))} | {fmt(r.get('auprc'))} | {fmt(r.get('f1_at_0p5'))} | {fmt(r.get('accuracy_at_0p5'))} |")
    lines += ['', '## Stable vs slip window summary', '', '| window | sides | windows | frames | pSlip mean | H1 inst mean | H5 inst mean | Fmag mean | Ft/Fn mean |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in payload['window_type_summary']:
        lines.append(f"| {r['window_type']} | {r['n_sequence_sides']} | {r['n_windows']} | {r['n_frames']} | {fmt(r.get('p_slip_current_mean'))} | {fmt(r.get('p_instability_H1_mean'))} | {fmt(r.get('p_instability_H5_mean'))} | {fmt(r.get('Fmag_pred_N_mean'))} | {fmt(r.get('Ft_over_Fn_pred_mean'))} |")
    lines += ['', '## Object-level summary', '', '| object | window | sides | windows | frames | pSlip mean | H1 inst mean | H5 inst mean | Fmag mean |', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in payload['object_window_summary']:
        lines.append(f"| {r['object']} | {r['window_type']} | {r['n_sequence_sides']} | {r['n_windows']} | {r['n_frames']} | {fmt(r.get('p_slip_current_mean'))} | {fmt(r.get('p_instability_H1_mean'))} | {fmt(r.get('p_instability_H5_mean'))} | {fmt(r.get('Fmag_pred_N_mean'))} |")
    lines += ['', '## Sequence / side / window summary', '', '| sequence | side | window | range | frames | pSlip mean | slip rate | H1 inst mean | H5 inst mean | Fmag mean |', '|---|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in payload['sequence_side_window_summary']:
        rng = f"{r.get('window_start')}-{r.get('window_end')}"
        lines.append(f"| {r['sequence']} | {r['side']} | {r['window_type']} | {rng} | {r['n_frames']} | {fmt(r.get('p_slip_current_mean'))} | {fmt(r.get('p_slip_current_rate_ge_0p5'))} | {fmt(r.get('p_instability_H1_mean'))} | {fmt(r.get('p_instability_H5_mean'))} | {fmt(r.get('Fmag_pred_N_mean'))} |")
    lines += ['', '## Automatic conclusion', '']
    for x in payload.get('interpretation', []):
        lines.append(f'- {x}')
    lines += ['', '## Output files', '']
    for k, v in payload.get('output_files', {}).items():
        lines.append(f'- {k}: `{v}`')
    return '\n'.join(lines) + '\n'


def build_interpretation(metrics: list[dict[str, Any]]) -> list[str]:
    notes = []
    by = {m['score']: m for m in metrics}
    for score in ('p_slip_current', 'p_instability_H1', 'p_instability_H3', 'p_instability_H5'):
        if score in by:
            m = by[score]
            notes.append(f"{score}: stable_mean={fmt(m.get('stable_mean'))}, slip_mean={fmt(m.get('slip_mean'))}, gap={fmt(m.get('slip_minus_stable_gap'))}, AUROC={fmt(m.get('auroc'))}.")
    return notes or ['Inspect per-frame and sequence-side rows; no score metrics were available.']


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
    for item in sequence_items:
        all_rows.extend(evaluate_sequence_side(item, stage1, future_head, future_payload, device, args.batch_size, args.frame_stride_context, tuple(args.resize), not args.no_bg))

    sequence_side_window_summary = summarize_rows(all_rows, ('sequence', 'object', 'side', 'window_type', 'window_index', 'window_start', 'window_end'))
    window_type_summary = summarize_rows(all_rows, ('window_type',))
    object_window_summary = summarize_rows(all_rows, ('object', 'window_type'))
    sequence_summary = summarize_rows(all_rows, ('sequence', 'object', 'side'))
    side_summary = summarize_rows(all_rows, ('side', 'window_type'))

    score_keys = ['p_slip_current', 'p_instability_H1', 'p_instability_H3', 'p_instability_H5']
    binary = [binary_metrics(all_rows, k) for k in score_keys if any(k in r for r in all_rows)]

    per_frame_csv = report_dir / 'real_force_slip_per_frame_predictions.csv'
    seq_window_csv = report_dir / 'real_force_slip_sequence_side_window_summary.csv'
    window_csv = report_dir / 'real_force_slip_window_type_summary.csv'
    object_csv = report_dir / 'real_force_slip_object_window_summary.csv'
    sequence_csv = report_dir / 'real_force_slip_sequence_side_summary.csv'
    side_csv = report_dir / 'real_force_slip_side_summary.csv'
    metrics_csv = report_dir / 'real_force_slip_binary_metrics.csv'
    write_csv(per_frame_csv, all_rows)
    write_csv(seq_window_csv, sequence_side_window_summary)
    write_csv(window_csv, window_type_summary)
    write_csv(object_csv, object_window_summary)
    write_csv(sequence_csv, sequence_summary)
    write_csv(side_csv, side_summary)
    write_csv(metrics_csv, binary)

    n_windows = len({(r['sequence'], r['side'], r['window_type'], r['window_index']) for r in all_rows})
    payload = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'dataset_root': str(args.dataset_root),
        'stage_i_checkpoint': str(args.stage1_checkpoint),
        'stage_i_train_config': stage1_payload.get('train_config', {}),
        'stage_ii_future_checkpoint': str(args.stage2_checkpoint) if future_payload is not None else None,
        'stage_ii_payload_meta': {k: v for k, v in (future_payload or {}).items() if k != 'model_state'},
        'config': {'batch_size': args.batch_size, 'frame_stride_context': args.frame_stride_context, 'resize': list(args.resize), 'use_first_frame_bg': not args.no_bg, 'device': str(device), 'window_source': 'stable_windows/slip_windows'},
        'n_sequence_sides': len(sequence_items), 'n_windows': n_windows, 'total_eval_frames': len(all_rows),
        'binary_metrics': binary,
        'window_type_summary': window_type_summary,
        'object_window_summary': object_window_summary,
        'sequence_side_window_summary': sequence_side_window_summary,
        'sequence_side_summary': sequence_summary,
        'side_summary': side_summary,
        'interpretation': build_interpretation(binary),
        'output_files': {
            'json': str(report_dir / 'real_force_slip_model_test_report.json'),
            'markdown': str(report_dir / 'real_force_slip_model_test_report.md'),
            'per_frame_csv': str(per_frame_csv),
            'sequence_side_window_csv': str(seq_window_csv),
            'window_type_csv': str(window_csv),
            'object_window_csv': str(object_csv),
            'sequence_side_csv': str(sequence_csv),
            'side_csv': str(side_csv),
            'binary_metrics_csv': str(metrics_csv),
        },
        'raw_data_modified': False,
    }
    write_json(report_dir / 'real_force_slip_model_test_report.json', payload)
    (report_dir / 'real_force_slip_model_test_report.md').write_text(render_md(payload), encoding='utf-8')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        labels = [m['score'] for m in binary]
        fig, axs = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
        axs[0].bar(labels, [m.get('slip_minus_stable_gap') or 0 for m in binary])
        axs[0].set_title('Slip - stable score gap')
        axs[1].bar(labels, [m.get('auroc') or 0 for m in binary])
        axs[1].set_ylim(0, 1)
        axs[1].set_title('AUROC')
        axs[2].bar(labels, [m.get('f1_at_0p5') or 0 for m in binary])
        axs[2].set_ylim(0, 1)
        axs[2].set_title('F1@0.5')
        for ax in axs:
            ax.tick_params(axis='x', rotation=25)
            ax.grid(axis='y', alpha=0.25)
        fig.suptitle('Real stable/slip window separation')
        png = report_dir / 'real_force_slip_stable_slip_metrics.png'
        fig.savefig(png, dpi=180)
        payload['output_files']['stable_slip_metrics_png'] = str(png)
        write_json(report_dir / 'real_force_slip_model_test_report.json', payload)
        (report_dir / 'real_force_slip_model_test_report.md').write_text(render_md(payload), encoding='utf-8')
    except Exception as exc:
        print(f'WARN plot skipped: {exc}', file=sys.stderr)

    print(json.dumps({'report_dir': str(report_dir), 'n_sequence_sides': len(sequence_items), 'n_windows': n_windows, 'total_eval_frames': len(all_rows), 'binary_metrics': binary}, indent=2))


if __name__ == '__main__':
    main()
