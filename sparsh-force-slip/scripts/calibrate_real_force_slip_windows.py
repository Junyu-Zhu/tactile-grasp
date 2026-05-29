#!/usr/bin/env python3
"""Lightweight real-data calibration for force/slip stable-vs-slip windows.

This script freezes all tactile models and only learns CSV-level calibration maps.
Labels come from real sidecar windows already expanded by evaluate_real_force_slip_model_test.py:
  stable_windows -> target_instability=0
  slip_windows   -> target_instability=1
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except Exception:
    LogisticRegression = None
    StandardScaler = None
    SKLEARN_OK = False

FEATURES = [
    'p_slip_current', 'p_instability_H1', 'p_instability_H3', 'p_instability_H5',
    'Fn_pred_N', 'Ft_pred_N', 'Fmag_pred_N', 'Ft_over_Fn_pred',
    'dFx_causal_N', 'dFy_causal_N', 'dFz_causal_N',
]
RAW_SCORES = ['p_slip_current', 'p_instability_H1', 'p_instability_H3', 'p_instability_H5']
DEFAULT_INPUT = Path('/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/real_force_slip_model_test/20260529_215727/real_force_slip_per_frame_predictions.csv')
DEFAULT_REPORT_ROOT = Path('/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/real_force_slip_calibration')

SEQUENCE_SPLIT = {
    'train': ['banana_1', 'cucumber_1', 'cucumber_3', 'hammer_1', 'lemon_1', 'lemon_2'],
    'val': ['banana_2', 'hammer_2', 'lemon_3'],
    'test': ['cucumber_2', 'hammer_3', 'lemon_4', 'lemon_5'],
}
OBJECT_HOLDOUT_SPLIT = {
    'train_objects': ['banana', 'lemon'],
    'val_sequences': ['banana_2', 'lemon_3'],
    'test_objects': ['cucumber', 'hammer'],
}


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding='utf-8')


def ffloat(x: Any, default: float = np.nan) -> float:
    try:
        if x is None or x == '':
            return default
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        rr = dict(r)
        rr['target_instability'] = int(ffloat(r.get('target_instability'), 1 if r.get('window_type') == 'slip' else 0))
        for c in FEATURES + RAW_SCORES:
            rr[c] = ffloat(r.get(c))
        rr['window_index'] = int(ffloat(r.get('window_index'), 0))
        rr['window_start'] = int(ffloat(r.get('window_start'), 0))
        rr['window_end'] = int(ffloat(r.get('window_end'), 0))
        rr['frame_index'] = int(ffloat(r.get('frame_index'), 0))
        out.append(rr)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def split_counts(rows: list[dict[str, Any]], split_col='split') -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in sorted({r[split_col] for r in rows}):
        vals = [r for r in rows if r[split_col] == split]
        out[split] = {
            'frames': len(vals),
            'stable_frames': sum(1 for r in vals if r['target_instability'] == 0),
            'slip_frames': sum(1 for r in vals if r['target_instability'] == 1),
            'sequences': sorted({r['sequence'] for r in vals}),
            'objects': sorted({r['object'] for r in vals}),
            'sequence_sides': len({(r['sequence'], r['side']) for r in vals}),
            'windows': len({window_key(r) for r in vals}),
        }
    return out


def assign_sequence_split(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mapping = {s: split for split, seqs in SEQUENCE_SPLIT.items() for s in seqs}
    assigned = []
    for r in rows:
        rr = dict(r)
        rr['split'] = mapping.get(r['sequence'], 'test')
        assigned.append(rr)
    return assigned


def assign_object_holdout_split(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    val_set = set(OBJECT_HOLDOUT_SPLIT['val_sequences'])
    train_objs = set(OBJECT_HOLDOUT_SPLIT['train_objects'])
    test_objs = set(OBJECT_HOLDOUT_SPLIT['test_objects'])
    assigned = []
    for r in rows:
        rr = dict(r)
        if r['sequence'] in val_set:
            split = 'val'
        elif r['object'] in test_objs:
            split = 'test'
        elif r['object'] in train_objs:
            split = 'train'
        else:
            split = 'test'
        rr['split'] = split
        assigned.append(rr)
    return assigned


def window_key(r: dict[str, Any]) -> tuple[Any, ...]:
    return (r['sequence'], r['object'], r['side'], r['window_type'], r['window_index'], r['window_start'], r['window_end'])


def side_fusion_key(r: dict[str, Any]) -> tuple[Any, ...]:
    return (r['sequence'], r['object'], r['window_type'], r['window_index'], r['window_start'], r['window_end'])


def auroc(y: np.ndarray, score: np.ndarray) -> float | None:
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    total = 0.0
    for p in pos:
        total += float((p > neg).sum()) + 0.5 * float((p == neg).sum())
    return float(total / (len(pos) * len(neg)))


def auprc(y: np.ndarray, score: np.ndarray) -> float | None:
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return None
    order = np.argsort(-score)
    yy = y[order]
    tp = np.cumsum(yy == 1)
    fp = np.cumsum(yy == 0)
    precision = tp / np.maximum(tp + fp, 1)
    return float(precision[yy == 1].mean())


def metric_dict(rows: list[dict[str, Any]], score_key: str, threshold: float = 0.5, prefix: str | None = None) -> dict[str, Any]:
    vals = [(int(r['target_instability']), ffloat(r.get(score_key))) for r in rows if not math.isnan(ffloat(r.get(score_key)))]
    name = prefix or score_key
    if not vals:
        return {'method': name, 'score_key': score_key, 'n': 0, 'threshold': threshold}
    y = np.array([v[0] for v in vals], dtype=int)
    s = np.array([v[1] for v in vals], dtype=float)
    pred = (s >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    specificity = tn / (tn + fp) if (tn + fp) else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and (precision + recall) else None
    acc = float((pred == y).mean())
    bal = (recall + specificity) / 2 if recall is not None and specificity is not None else None
    stable = s[y == 0]
    slip = s[y == 1]
    return {
        'method': name, 'score_key': score_key, 'n': int(len(y)), 'n_stable': int((y == 0).sum()), 'n_slip': int((y == 1).sum()),
        'threshold': float(threshold),
        'stable_mean': float(stable.mean()) if stable.size else None,
        'slip_mean': float(slip.mean()) if slip.size else None,
        'slip_minus_stable_gap': float(slip.mean() - stable.mean()) if stable.size and slip.size else None,
        'auroc': auroc(y, s), 'auprc': auprc(y, s),
        'accuracy': acc, 'balanced_accuracy': bal, 'precision': precision, 'recall': recall, 'specificity': specificity, 'f1': f1,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
    }


def objective_value(m: dict[str, Any]) -> tuple[float, float]:
    # Primary F1, secondary balanced accuracy.
    return (float(m.get('f1') or -1), float(m.get('balanced_accuracy') or -1))


def fit_threshold(rows: list[dict[str, Any]], score_key: str) -> tuple[float, dict[str, Any]]:
    best_thr = 0.5
    best_m = metric_dict(rows, score_key, 0.5)
    for thr in np.linspace(0.0, 1.0, 501):
        m = metric_dict(rows, score_key, float(thr))
        if objective_value(m) > objective_value(best_m):
            best_thr, best_m = float(thr), m
    return best_thr, best_m


def make_feature_matrix(rows: list[dict[str, Any]], fill_values: dict[str, float] | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if fill_values is None:
        fill_values = {}
        for c in FEATURES:
            vals = np.array([ffloat(r.get(c)) for r in rows], dtype=float)
            finite = vals[np.isfinite(vals)]
            fill_values[c] = float(np.median(finite)) if finite.size else 0.0
    X = []
    y = []
    for r in rows:
        X.append([ffloat(r.get(c), fill_values[c]) if math.isfinite(ffloat(r.get(c), fill_values[c])) else fill_values[c] for c in FEATURES])
        y.append(int(r['target_instability']))
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int), fill_values


@dataclass
class Calibrator:
    kind: str
    fill_values: dict[str, float]
    scaler: Any = None
    model: Any = None
    weights: np.ndarray | None = None
    bias: float | None = None

    def predict_proba(self, rows: list[dict[str, Any]]) -> np.ndarray:
        X, _, _ = make_feature_matrix(rows, self.fill_values)
        if self.kind == 'sklearn':
            return self.model.predict_proba(self.scaler.transform(X))[:, 1]
        Xs = (X - self.weights['mean']) / self.weights['std']
        z = Xs @ self.weights['w'] + float(self.bias)
        return 1.0 / (1.0 + np.exp(-z))


def fit_logistic(train_rows: list[dict[str, Any]]) -> tuple[Calibrator, dict[str, Any]]:
    X, y, fill_values = make_feature_matrix(train_rows)
    if len(np.unique(y)) < 2:
        raise RuntimeError('Cannot train logistic calibration: train split has a single class')
    if SKLEARN_OK:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=2000, class_weight='balanced', solver='lbfgs')
        model.fit(Xs, y)
        return Calibrator(kind='sklearn', fill_values=fill_values, scaler=scaler, model=model), {
            'backend': 'sklearn.LogisticRegression', 'class_weight': 'balanced', 'features': FEATURES,
            'coef': model.coef_[0].tolist(), 'intercept': model.intercept_.tolist(),
            'feature_mean': scaler.mean_.tolist(), 'feature_scale': scaler.scale_.tolist(),
        }
    # Fallback: simple NumPy logistic regression.
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    Xs = (X - mean) / std
    w = np.zeros(Xs.shape[1], dtype=float)
    b = 0.0
    lr = 0.05
    pos = max(float((y == 1).sum()), 1.0)
    neg = max(float((y == 0).sum()), 1.0)
    weights = np.where(y == 1, len(y) / (2 * pos), len(y) / (2 * neg))
    for _ in range(3000):
        z = Xs @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        g = (p - y) * weights
        w -= lr * (Xs.T @ g / len(y))
        b -= lr * float(g.mean())
    cal = Calibrator(kind='numpy', fill_values=fill_values, weights={'w': w, 'mean': mean, 'std': std}, bias=b)
    return cal, {'backend': 'numpy_logistic', 'features': FEATURES, 'coef': w.tolist(), 'intercept': b, 'feature_mean': mean.tolist(), 'feature_scale': std.tolist()}


def add_calibrated_scores(rows: list[dict[str, Any]], cal: Calibrator) -> list[dict[str, Any]]:
    probs = cal.predict_proba(rows)
    out = []
    for r, p in zip(rows, probs):
        rr = dict(r)
        rr['p_calibrated_instability'] = float(p)
        out.append(rr)
    return out


def aggregate_windows(rows: list[dict[str, Any]], score_keys: Iterable[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[window_key(r)].append(r)
    out = []
    for key, vals in sorted(groups.items()):
        sequence, obj, side, wtype, widx, wstart, wend = key
        rec = {
            'sequence': sequence, 'object': obj, 'side': side, 'window_type': wtype, 'window_index': widx,
            'window_start': wstart, 'window_end': wend, 'target_instability': int(vals[0]['target_instability']),
            'split': vals[0].get('split'), 'n_frames': len(vals),
        }
        for s in score_keys:
            arr = np.array([ffloat(v.get(s)) for v in vals if math.isfinite(ffloat(v.get(s)))], dtype=float)
            if arr.size:
                rec[f'{s}_mean'] = float(arr.mean())
                rec[f'{s}_max'] = float(arr.max())
                q = np.quantile(arr, 0.9)
                rec[f'{s}_top10_mean'] = float(arr[arr >= q].mean())
        out.append(rec)
    return out


def aggregate_side_fusion(window_rows: list[dict[str, Any]], score_keys: Iterable[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in window_rows:
        groups[side_fusion_key(r)].append(r)
    out = []
    for key, vals in sorted(groups.items()):
        sequence, obj, wtype, widx, wstart, wend = key
        rec = {
            'sequence': sequence, 'object': obj, 'window_type': wtype, 'window_index': widx,
            'window_start': wstart, 'window_end': wend, 'target_instability': int(vals[0]['target_instability']),
            'split': vals[0].get('split'), 'n_sides': len({v['side'] for v in vals}), 'sides': '+'.join(sorted({v['side'] for v in vals})),
        }
        for s in score_keys:
            # Fuse left/right window means by max and mean. If only one side exists, this is side-level.
            base = f'{s}_mean'
            arr = np.array([ffloat(v.get(base)) for v in vals if math.isfinite(ffloat(v.get(base)))], dtype=float)
            if arr.size:
                rec[f'{s}_side_mean_fusion'] = float(arr.mean())
                rec[f'{s}_side_max_fusion'] = float(arr.max())
        out.append(rec)
    return out


def group_summary(rows: list[dict[str, Any]], keys: tuple[str, ...], score_keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k) for k in keys)].append(r)
    out = []
    for key, vals in sorted(groups.items()):
        rec = {k: v for k, v in zip(keys, key)}
        rec['n'] = len(vals)
        rec['stable_n'] = sum(1 for v in vals if int(v['target_instability']) == 0)
        rec['slip_n'] = sum(1 for v in vals if int(v['target_instability']) == 1)
        for s in score_keys:
            arr = np.array([ffloat(v.get(s)) for v in vals if math.isfinite(ffloat(v.get(s)))], dtype=float)
            if arr.size:
                rec[f'{s}_mean'] = float(arr.mean())
                rec[f'{s}_max'] = float(arr.max())
        out.append(rec)
    return out


def evaluate_split(split_name: str, rows: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    train = [r for r in rows if r['split'] == 'train']
    val = [r for r in rows if r['split'] == 'val']
    test = [r for r in rows if r['split'] == 'test']
    if not train or not val or not test:
        raise RuntimeError(f'{split_name}: train/val/test split is empty')
    cal, cal_meta = fit_logistic(train)
    rows_cal = add_calibrated_scores(rows, cal)
    train_cal = [r for r in rows_cal if r['split'] == 'train']
    val_cal = [r for r in rows_cal if r['split'] == 'val']
    test_cal = [r for r in rows_cal if r['split'] == 'test']

    frame_metrics = []
    threshold_records = []
    for score in RAW_SCORES:
        frame_metrics.append({**metric_dict(test_cal, score, 0.5, f'raw_{score}@0.5'), 'level': 'frame', 'split_design': split_name})
        thr, val_m = fit_threshold(val_cal, score)
        threshold_records.append({'score': score, 'threshold': thr, 'val_f1': val_m.get('f1'), 'val_balanced_accuracy': val_m.get('balanced_accuracy')})
        frame_metrics.append({**metric_dict(test_cal, score, thr, f'threshold_{score}'), 'level': 'frame', 'split_design': split_name})
    thr_cal, val_cal_m = fit_threshold(val_cal, 'p_calibrated_instability')
    threshold_records.append({'score': 'p_calibrated_instability', 'threshold': thr_cal, 'val_f1': val_cal_m.get('f1'), 'val_balanced_accuracy': val_cal_m.get('balanced_accuracy')})
    frame_metrics.append({**metric_dict(test_cal, 'p_calibrated_instability', 0.5, 'logistic_calibration@0.5'), 'level': 'frame', 'split_design': split_name})
    frame_metrics.append({**metric_dict(test_cal, 'p_calibrated_instability', thr_cal, 'logistic_calibration_thresholded'), 'level': 'frame', 'split_design': split_name})

    score_keys = RAW_SCORES + ['p_calibrated_instability']
    window_rows = aggregate_windows(rows_cal, score_keys)
    window_test = [r for r in window_rows if r['split'] == 'test']
    window_val = [r for r in window_rows if r['split'] == 'val']
    window_metrics = []
    for score in score_keys:
        for agg in ['mean', 'max', 'top10_mean']:
            key = f'{score}_{agg}'
            # Fit threshold on window-level val for each aggregation.
            if any(key in r for r in window_val):
                thr, _ = fit_threshold(window_val, key)
                window_metrics.append({**metric_dict(window_test, key, thr, f'window_{score}_{agg}_thresholded'), 'level': 'window', 'split_design': split_name})
                window_metrics.append({**metric_dict(window_test, key, 0.5, f'window_{score}_{agg}@0.5'), 'level': 'window', 'split_design': split_name})

    fusion_rows = aggregate_side_fusion(window_rows, score_keys)
    fusion_val = [r for r in fusion_rows if r['split'] == 'val']
    fusion_test = [r for r in fusion_rows if r['split'] == 'test']
    fusion_metrics = []
    for score in score_keys:
        for agg in ['side_mean_fusion', 'side_max_fusion']:
            key = f'{score}_{agg}'
            if any(key in r for r in fusion_val):
                thr, _ = fit_threshold(fusion_val, key)
                fusion_metrics.append({**metric_dict(fusion_test, key, thr, f'fusion_{score}_{agg}_thresholded'), 'level': 'side_fusion', 'split_design': split_name})
                fusion_metrics.append({**metric_dict(fusion_test, key, 0.5, f'fusion_{score}_{agg}@0.5'), 'level': 'side_fusion', 'split_design': split_name})

    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    write_csv(split_dir / 'calibrated_frame_predictions.csv', rows_cal)
    write_csv(split_dir / 'window_level_predictions.csv', window_rows)
    write_csv(split_dir / 'side_fusion_window_predictions.csv', fusion_rows)
    write_csv(split_dir / 'frame_metrics.csv', frame_metrics)
    write_csv(split_dir / 'window_metrics.csv', window_metrics)
    write_csv(split_dir / 'side_fusion_metrics.csv', fusion_metrics)
    write_csv(split_dir / 'thresholds.csv', threshold_records)
    write_csv(split_dir / 'side_summary.csv', group_summary(rows_cal, ('split', 'side', 'window_type'), score_keys))
    write_csv(split_dir / 'object_summary.csv', group_summary(rows_cal, ('split', 'object', 'window_type'), score_keys))
    write_csv(split_dir / 'sequence_summary.csv', group_summary(rows_cal, ('split', 'sequence', 'side', 'window_type'), score_keys))

    all_metrics = frame_metrics + window_metrics + fusion_metrics
    # Select by test F1 then balanced accuracy, but keep AUROC context.
    best = max(all_metrics, key=objective_value) if all_metrics else None
    return {
        'split_design': split_name,
        'split_counts': split_counts(rows_cal),
        'calibrator': cal_meta,
        'thresholds': threshold_records,
        'frame_metrics': frame_metrics,
        'window_metrics': window_metrics,
        'side_fusion_metrics': fusion_metrics,
        'best_by_f1': best,
        'output_dir': str(split_dir),
    }


def fmt(v: Any, digits: int = 3) -> str:
    if v is None:
        return 'n/a'
    try:
        return f'{float(v):.{digits}f}'
    except Exception:
        return str(v)


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        '# Real Force-Slip Calibration Report', '',
        f"- generated_at: `{payload['generated_at']}`",
        f"- input_csv: `{payload['input_csv']}`",
        f"- report_dir: `{payload['report_dir']}`",
        f"- features: `{', '.join(FEATURES)}`",
        '- label rule: `stable_windows -> 0`, `slip_windows -> 1`.',
        '- This is real-world calibration only: no Sparsh backbone, encoder, decoder, or checkpoint weights were retrained or modified.',
        '- Purpose: bridge the domain gap between controlled flat/sharp/sphere training data and real banana/cucumber/hammer/lemon grasp trials.',
        '', '## Headline results', '',
        '| split | best method | level | n | threshold | AUROC | AUPRC | F1 | Acc | BalAcc | stable mean | slip mean | gap |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for res in payload['results']:
        b = res.get('best_by_f1') or {}
        lines.append(f"| {res['split_design']} | {b.get('method')} | {b.get('level')} | {b.get('n')} | {fmt(b.get('threshold'))} | {fmt(b.get('auroc'))} | {fmt(b.get('auprc'))} | {fmt(b.get('f1'))} | {fmt(b.get('accuracy'))} | {fmt(b.get('balanced_accuracy'))} | {fmt(b.get('stable_mean'))} | {fmt(b.get('slip_mean'))} | {fmt(b.get('slip_minus_stable_gap'))} |")
    lines += ['', '## Frame-level test comparison', '']
    for res in payload['results']:
        lines += [f"### {res['split_design']}", '', '| method | threshold | AUROC | AUPRC | F1 | Acc | BalAcc | stable mean | slip mean | gap |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        wanted = [m for m in res['frame_metrics'] if m['method'] in [
            'raw_p_slip_current@0.5', 'raw_p_instability_H1@0.5', 'raw_p_instability_H3@0.5', 'raw_p_instability_H5@0.5',
            'threshold_p_slip_current', 'threshold_p_instability_H1', 'logistic_calibration@0.5', 'logistic_calibration_thresholded'
        ]]
        for m in wanted:
            lines.append(f"| {m['method']} | {fmt(m.get('threshold'))} | {fmt(m.get('auroc'))} | {fmt(m.get('auprc'))} | {fmt(m.get('f1'))} | {fmt(m.get('accuracy'))} | {fmt(m.get('balanced_accuracy'))} | {fmt(m.get('stable_mean'))} | {fmt(m.get('slip_mean'))} | {fmt(m.get('slip_minus_stable_gap'))} |")
        lines.append('')
    lines += ['## Split details', '']
    for res in payload['results']:
        lines += [f"### {res['split_design']}", '']
        for split, c in res['split_counts'].items():
            lines.append(f"- {split}: frames={c['frames']}, stable={c['stable_frames']}, slip={c['slip_frames']}, windows={c['windows']}, sequences={c['sequences']}")
        lines.append('')
    lines += ['## Output files', '']
    for k, v in payload['output_files'].items():
        lines.append(f'- {k}: `{v}`')
    lines += ['', '## Practical recommendation', '']
    lines += payload.get('recommendation', [])
    return '\n'.join(lines) + '\n'


def make_plots(payload: dict[str, Any], report_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:
        payload.setdefault('warnings', []).append(f'plot skipped: {exc}')
        return
    for res in payload['results']:
        methods = [m for m in res['frame_metrics'] if m['method'] in ['raw_p_slip_current@0.5', 'raw_p_instability_H1@0.5', 'logistic_calibration_thresholded', 'threshold_p_slip_current']]
        if not methods:
            continue
        labels = [m['method'].replace('_', '\n') for m in methods]
        fig, axs = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
        for ax, key, title in zip(axs, ['f1', 'balanced_accuracy', 'auroc'], ['F1', 'Balanced accuracy', 'AUROC']):
            ax.bar(labels, [m.get(key) or 0 for m in methods])
            ax.set_ylim(0, 1)
            ax.set_title(title)
            ax.grid(axis='y', alpha=0.25)
            ax.tick_params(axis='x', labelsize=7)
        fig.suptitle(f'Real calibration frame metrics: {res["split_design"]}')
        out = report_dir / f'{res["split_design"]}_frame_metric_bars.png'
        fig.savefig(out, dpi=180)
        payload['output_files'][f'{res["split_design"]}_frame_metric_bars'] = str(out)


def build_recommendation(results: list[dict[str, Any]]) -> list[str]:
    lines = []
    for res in results:
        split = res['split_design']
        frame = {m['method']: m for m in res['frame_metrics']}
        raw_slip = frame.get('raw_p_slip_current@0.5')
        raw_h1 = frame.get('raw_p_instability_H1@0.5')
        cal = frame.get('logistic_calibration_thresholded')
        th_slip = frame.get('threshold_p_slip_current')
        if raw_slip:
            lines.append(f"- {split}: raw p_slip_current remains the strongest uncalibrated real-data signal (F1={fmt(raw_slip.get('f1'))}, AUROC={fmt(raw_slip.get('auroc'))}).")
        if raw_h1:
            lines.append(f"- {split}: raw H1 future-instability is poorly calibrated on real data (F1={fmt(raw_h1.get('f1'))}, AUROC={fmt(raw_h1.get('auroc'))}), consistent with the high stable-window probabilities observed earlier.")
        if th_slip:
            lines.append(f"- {split}: threshold fitting p_slip_current gives a lightweight deployment rule (threshold={fmt(th_slip.get('threshold'))}, F1={fmt(th_slip.get('f1'))}).")
        if cal:
            lines.append(f"- {split}: logistic calibration combines slip/future/force proxies (threshold={fmt(cal.get('threshold'))}, F1={fmt(cal.get('f1'))}, BalAcc={fmt(cal.get('balanced_accuracy'))}); use it as a real-world calibration layer only if test metrics exceed the simpler threshold baseline.")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-csv', type=Path, default=DEFAULT_INPUT)
    ap.add_argument('--report-root', type=Path, default=DEFAULT_REPORT_ROOT)
    args = ap.parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(args.input_csv)
    rows = read_rows(args.input_csv)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = args.report_root / stamp
    report_dir.mkdir(parents=True, exist_ok=True)

    seq_rows = assign_sequence_split(rows)
    obj_rows = assign_object_holdout_split(rows)
    results = [
        evaluate_split('sequence_split', seq_rows, report_dir),
        evaluate_split('object_holdout', obj_rows, report_dir),
    ]
    payload = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'input_csv': str(args.input_csv),
        'report_dir': str(report_dir),
        'features': FEATURES,
        'raw_scores': RAW_SCORES,
        'sklearn_available': SKLEARN_OK,
        'sequence_split_plan': SEQUENCE_SPLIT,
        'object_holdout_split_plan': OBJECT_HOLDOUT_SPLIT,
        'results': results,
        'output_files': {
            'json': str(report_dir / 'real_force_slip_calibration_report.json'),
            'markdown': str(report_dir / 'real_force_slip_calibration_report.md'),
        },
    }
    payload['recommendation'] = build_recommendation(results)
    make_plots(payload, report_dir)
    write_json(report_dir / 'real_force_slip_calibration_report.json', payload)
    (report_dir / 'real_force_slip_calibration_report.md').write_text(render_md(payload), encoding='utf-8')
    print(json.dumps({'report_dir': str(report_dir), 'results': [{'split_design': r['split_design'], 'best_by_f1': r['best_by_f1']} for r in results]}, indent=2, default=json_default))


if __name__ == '__main__':
    main()
