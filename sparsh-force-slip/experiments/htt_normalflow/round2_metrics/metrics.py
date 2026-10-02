"""Pure NumPy metrics for the frozen round-2 protocol."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


THRESHOLD_GRID = np.r_[np.linspace(0.0, 1.0, 1001), 2.0]


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def confusion(y: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    y = np.asarray(y, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    return {
        "tn": int(np.sum(~y & ~pred)),
        "fp": int(np.sum(~y & pred)),
        "fn": int(np.sum(y & ~pred)),
        "tp": int(np.sum(y & pred)),
    }


def average_precision(y: np.ndarray, score: np.ndarray) -> float | None:
    """Non-interpolated AP with equal-score samples handled as one threshold."""
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score, dtype=float)
    positives = int(y.sum())
    if positives == 0 or positives == len(y):
        return None
    order = np.argsort(-score, kind="mergesort")
    ranked = y[order]
    ranked_score = score[order]
    ends = np.r_[np.flatnonzero(ranked_score[:-1] != ranked_score[1:]), len(ranked_score) - 1]
    tp = np.cumsum(ranked)[ends]
    fp = (ends + 1) - tp
    precision = tp / (tp + fp)
    recall = tp / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def binary_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=float)
    finite = np.isfinite(score) & np.isin(y, [0, 1])
    y, score = y[finite], score[finite]
    support = {"total": int(len(y)), "negative": int(np.sum(y == 0)), "positive": int(np.sum(y == 1))}
    quantile_levels = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
    def quantiles(values: np.ndarray) -> dict[str, float | None]:
        return {str(q): (float(np.quantile(values, q)) if len(values) else None) for q in quantile_levels}
    quantiles_all = quantiles(score)
    quantiles_by_class = {"negative": quantiles(score[y == 0]), "positive": quantiles(score[y == 1])}
    c = confusion(y, score >= threshold)
    per_class = {}
    for name, tp, fp, fn, class_support in (
        ("negative", c["tn"], c["fn"], c["fp"], support["negative"]),
        ("positive", c["tp"], c["fp"], c["fn"], support["positive"]),
    ):
        if class_support == 0:
            per_class[name] = None
            continue
        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, tp + fn)
        f1 = _ratio(2 * tp, 2 * tp + fp + fn)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1}
    both_classes = support["negative"] > 0 and support["positive"] > 0
    return {
        "evaluable": both_classes,
        "reason": None if both_classes else "both classes required for discrimination metrics; available one-class metrics retained",
        "threshold": float(threshold),
        "support": support,
        "score_quantiles": quantiles_all,
        "score_quantiles_by_true_class": quantiles_by_class,
        "confusion": c,
        **per_class,
        "macro_f1": float((per_class["negative"]["f1"] + per_class["positive"]["f1"]) / 2) if both_classes else None,
        "balanced_accuracy": float((per_class["negative"]["recall"] + per_class["positive"]["recall"]) / 2) if both_classes else None,
        "average_precision": average_precision(y, score) if both_classes else None,
        "brier": float(np.mean((score - y) ** 2)) if len(y) else None,
    }


def threshold_curve(y: np.ndarray, score: np.ndarray) -> list[dict]:
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=float)
    valid = np.isfinite(score) & np.isin(y, [0, 1])
    y, score = y[valid], score[valid]
    negative = np.sort(score[y == 0])
    positive = np.sort(score[y == 1])
    rows = []
    for threshold in THRESHOLD_GRID:
        fp = len(negative) - int(np.searchsorted(negative, threshold, side="left"))
        tp = len(positive) - int(np.searchsorted(positive, threshold, side="left"))
        recall = tp / len(positive) if len(positive) else None
        fpr = fp / len(negative) if len(negative) else None
        rows.append({
            "threshold": float(threshold),
            "balanced_accuracy": float((recall + 1.0 - fpr) / 2.0) if recall is not None and fpr is not None else None,
            "recall": None if recall is None else float(recall),
            "fpr": None if fpr is None else float(fpr),
        })
    return rows


def select_balanced_threshold(y: np.ndarray, score: np.ndarray) -> dict:
    curve = threshold_curve(y, score)
    y = np.asarray(y)
    support = {"total": int(len(y)), "negative": int(np.sum(y == 0)), "positive": int(np.sum(y == 1))}
    valid = [r for r in curve if r["balanced_accuracy"] is not None]
    if not valid:
        return {"threshold": None, "reason": "both classes required", "objective": None, "support": support}
    best_value = max(r["balanced_accuracy"] for r in valid)
    threshold = max(r["threshold"] for r in valid if np.isclose(r["balanced_accuracy"], best_value))
    return {"threshold": float(threshold), "reason": None, "objective": float(best_value), "support": support}


def select_fpr_threshold(y: np.ndarray, score: np.ndarray, max_fpr: float = 0.01) -> dict:
    curve = threshold_curve(y, score)
    y = np.asarray(y)
    support = {"total": int(len(y)), "negative": int(np.sum(y == 0)), "positive": int(np.sum(y == 1))}
    valid = [r for r in curve if r["fpr"] is not None and r["recall"] is not None and r["fpr"] <= max_fpr + 1e-12]
    if not valid:
        return {"threshold": None, "reason": "both classes required", "recall": None, "fpr": None, "support": support}
    best_recall = max(r["recall"] for r in valid)
    tied = [r for r in valid if np.isclose(r["recall"], best_recall)]
    chosen = max(tied, key=lambda r: r["threshold"])
    return {"threshold": chosen["threshold"], "reason": None, "recall": chosen["recall"], "fpr": chosen["fpr"], "support": support}


def group_bootstrap(
    y: np.ndarray,
    score: np.ndarray,
    groups: Sequence[str],
    threshold: float,
    reps: int = 200,
    seed: int = 20260913,
) -> dict:
    y, score, groups = np.asarray(y), np.asarray(score), np.asarray(groups, dtype=str)
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {k: [] for k in ("balanced_accuracy", "macro_f1", "average_precision", "brier")}
    for _ in range(reps):
        picked = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in picked]) if len(picked) else np.empty(0, int)
        row = binary_metrics(y[indices], score[indices], threshold)
        if not row["evaluable"]:
            continue
        for key in values:
            values[key].append(row[key])
    return {
        "seed": seed,
        "requested_replicates": reps,
        "valid_replicates": len(values["macro_f1"]),
        "ci95": {
            key: ({"low": float(np.quantile(vals, 0.025)), "high": float(np.quantile(vals, 0.975))} if vals else None)
            for key, vals in values.items()
        },
    }


def rising_edges(active: np.ndarray, cooldown: int = 5) -> np.ndarray:
    active = np.asarray(active, dtype=bool)
    candidates = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    accepted: list[int] = []
    for index in candidates:
        if not accepted or index - accepted[-1] > cooldown:
            accepted.append(int(index))
    return np.asarray(accepted, dtype=int)


def event_metrics(episodes: Sequence[dict], score_key: str, threshold: float, horizon: int, population: int) -> dict:
    onset_count = event_count = unsupported_events = hits = late = misses = false_edges = negative_frames = 0
    sequence_false_edges = sequence_negative_frames = 0
    leads: list[int] = []
    late_delays: list[int] = []
    episode_rows: list[dict] = []
    for episode in episodes:
        labels = np.asarray(episode["labels"])
        scores = np.asarray(episode[score_key], dtype=float)
        gross = np.flatnonzero(labels == 2)
        onset = int(gross[0]) if len(gross) else None
        edges = rising_edges(np.isfinite(scores) & (scores >= threshold), cooldown=5)
        if onset is None:
            full_horizon = np.arange(len(labels)) + horizon < len(labels)
            valid_negative = np.flatnonzero((labels == population) & np.isfinite(scores) & full_horizon)
            false = int(np.sum(np.isin(edges, valid_negative)))
            false_edges += false
            negative_frames += len(valid_negative)
            sequence_valid = np.flatnonzero(np.isin(labels, [0, 1]) & np.isfinite(scores) & full_horizon)
            sequence_false = int(np.sum(np.isin(edges, sequence_valid)))
            sequence_false_edges += sequence_false
            sequence_negative_frames += len(sequence_valid)
            episode_rows.append({"episode_id": episode["id"], "onset": None, "hit": None, "late": None, "miss": None, "lead_steps": None, "late_delay_steps": None, "population_false_alarm_edges": false, "sequence_false_alarm_edges": sequence_false})
            continue
        onset_count += 1
        warning_start = max(0, onset - horizon)
        eligible = np.arange(warning_start, onset)
        eligible = eligible[(eligible + horizon < len(labels)) & (labels[eligible] == population) & np.isfinite(scores[eligible])]
        hit_edges = edges[np.isin(edges, eligible)]
        pre_negative = np.arange(0, warning_start)
        pre_negative = pre_negative[(pre_negative + horizon < len(labels)) & (labels[pre_negative] == population) & np.isfinite(scores[pre_negative])]
        false = int(np.sum(np.isin(edges, pre_negative)))
        false_edges += false
        negative_frames += len(pre_negative)
        sequence_pre_negative = np.arange(0, warning_start)
        sequence_pre_negative = sequence_pre_negative[(sequence_pre_negative + horizon < len(labels)) & np.isin(labels[sequence_pre_negative], [0, 1]) & np.isfinite(scores[sequence_pre_negative])]
        sequence_false = int(np.sum(np.isin(edges, sequence_pre_negative)))
        sequence_false_edges += sequence_false
        sequence_negative_frames += len(sequence_pre_negative)
        if len(eligible) == 0:
            unsupported_events += 1
            episode_rows.append({"episode_id": episode["id"], "onset": onset, "supported": False, "hit": None, "late": None, "miss": None, "lead_steps": None, "late_delay_steps": None, "population_false_alarm_edges": false, "sequence_false_alarm_edges": sequence_false})
            continue
        event_count += 1
        if len(hit_edges):
            hit = int(hit_edges[0])
            lead = onset - hit
            hits += 1
            leads.append(lead)
            is_late = is_miss = False
            late_delay = None
        else:
            later = edges[(edges >= onset) & (edges <= min(len(scores) - 1, onset + horizon))]
            is_late = bool(len(later))
            late += int(is_late)
            late_delay = int(later[0] - onset) if is_late else None
            if late_delay is not None:
                late_delays.append(late_delay)
            is_miss = not is_late
            misses += int(is_miss)
            lead = None
        episode_rows.append({"episode_id": episode["id"], "onset": onset, "supported": True, "hit": bool(len(hit_edges)), "late": is_late, "miss": is_miss, "lead_steps": lead, "late_delay_steps": late_delay, "population_false_alarm_edges": false, "sequence_false_alarm_edges": sequence_false})
    return {
        "episodes_with_first_gross_onset": onset_count,
        "support_events": event_count,
        "unsupported_events_no_population_frame_in_horizon": unsupported_events,
        "hits": hits,
        "event_recall": _ratio(hits, event_count),
        "late": late,
        "late_diagnostic_window_steps_after_onset": horizon,
        "late_delay_steps": {
            "count": len(late_delays),
            "median": float(np.median(late_delays)) if late_delays else None,
            "mean": float(np.mean(late_delays)) if late_delays else None,
        },
        "misses": misses,
        "lead_steps": {
            "count": len(leads),
            "median": float(np.median(leads)) if leads else None,
            "mean": float(np.mean(leads)) if leads else None,
            "min": int(np.min(leads)) if leads else None,
            "max": int(np.max(leads)) if leads else None,
        },
        "population_conditioned_false_alarm_edges": false_edges,
        "population_conditioned_negative_frames": negative_frames,
        "population_conditioned_false_alarm_edges_per_negative_frame": _ratio(false_edges, negative_frames),
        "sequence_false_alarm_edges_before_warning_window": sequence_false_edges,
        "sequence_negative_frames_before_warning_window": sequence_negative_frames,
        "sequence_false_alarm_edges_per_negative_frame": _ratio(sequence_false_edges, sequence_negative_frames),
        "episodes": episode_rows,
    }


def future_event_support(episodes: Sequence[dict], horizon: int, population: int, score_key: str) -> dict:
    onset_count = supported = 0
    for episode in episodes:
        labels = np.asarray(episode["labels"])
        scores = np.asarray(episode[score_key], float)
        gross = np.flatnonzero(labels == 2)
        if not len(gross):
            continue
        onset_count += 1
        onset = int(gross[0])
        indices = np.arange(max(0, onset - horizon), onset)
        if np.any((indices + horizon < len(labels)) & (labels[indices] == population) & np.isfinite(scores[indices])):
            supported += 1
    return {
        "episodes_with_first_gross_onset": onset_count,
        "support_events": supported,
        "unsupported_events_no_population_frame_in_horizon": onset_count - supported,
    }


def pearson(x: np.ndarray, y: np.ndarray) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"n": int(len(x)), "r": None, "reason": "at least 3 varying pairs required"}
    return {"n": int(len(x)), "r": float(np.corrcoef(x, y)[0, 1]), "reason": None}


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
