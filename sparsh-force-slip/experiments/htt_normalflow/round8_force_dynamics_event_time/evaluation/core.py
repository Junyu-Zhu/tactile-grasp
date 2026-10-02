"""Reusable metrics for Round 8.

All functions consume complete per-episode timelines. Callers must fit models on
fit/selection/calibration roles and evaluate outer only after those objects are
frozen. Nothing in this module silently pools roles.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch

RULES = ("raw", "ema_0p5", "confirm2", "ema_0p5_confirm2")


def cumulative_risk_from_hazard(hazard: Sequence[float]) -> np.ndarray:
    q = np.asarray(hazard, dtype=float)
    if q.ndim != 1 or not len(q) or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError("hazards must be a finite one-dimensional probability vector")
    return 1.0 - np.cumprod(1.0 - q)


def average_precision(target: Sequence[int], score: Sequence[float]) -> float:
    y = np.asarray(target, dtype=np.int8)
    p = np.asarray(score, dtype=float)
    positives = int(y.sum())
    if positives == 0:
        return math.nan
    order = np.argsort(-p, kind="mergesort")
    ys, ps = y[order], p[order]
    ends = np.r_[np.flatnonzero(ps[:-1] != ps[1:]), len(ps) - 1]
    tp = np.cumsum(ys)[ends]
    return float(np.sum(np.diff(np.r_[0.0, tp / positives]) * (tp / (ends + 1))))


def causal_rule(values: Sequence[float], rule: str) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    if not len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("invalid probabilities")
    if rule == "raw":
        return p.copy()
    ema = np.empty_like(p)
    ema[0] = p[0]
    for index in range(1, len(p)):
        ema[index] = 0.5 * p[index] + 0.5 * ema[index - 1]
    if rule == "ema_0p5":
        return ema
    base = p if rule == "confirm2" else ema
    if rule not in ("confirm2", "ema_0p5_confirm2"):
        raise ValueError(f"unknown causal rule {rule!r}")
    result = np.zeros_like(base)
    result[1:] = np.minimum(base[:-1], base[1:])
    return result


def add_causal_rules(rows: list[dict[str, Any]], raw_key: str, prefix: str) -> None:
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_episode[str(row["episode_id"])].append(row)
    for episode, sequence in by_episode.items():
        sequence.sort(key=lambda row: int(row["t"]))
        times = [int(row["t"]) for row in sequence]
        if times != list(range(times[0], times[-1] + 1)):
            raise ValueError(f"non-contiguous timeline {episode}")
        scores = [float(row[raw_key]) for row in sequence]
        for rule in RULES:
            for row, value in zip(sequence, causal_rule(scores, rule)):
                row[f"{prefix}{rule}"] = float(value)


def _cohort(rows: Sequence[Mapping[str, Any]], mask_key: str) -> list[Mapping[str, Any]]:
    return [row for row in rows if bool(row[mask_key])]


def frame_metrics(rows: Sequence[Mapping[str, Any]], score_key: str, threshold: float, mask_key: str) -> dict[str, Any]:
    cohort = _cohort(rows, mask_key)
    y = np.asarray([int(row["target"]) for row in cohort], dtype=bool)
    score = np.asarray([float(row[score_key]) for row in cohort], dtype=float)
    if set(y.tolist()) != {False, True}:
        raise ValueError("frame cohort lacks both classes")
    alarm = score >= threshold
    tp = int(np.sum(alarm & y)); fp = int(np.sum(alarm & ~y))
    fn = int(np.sum(~alarm & y)); tn = int(np.sum(~alarm & ~y))
    recall = tp / (tp + fn); fpr = fp / (fp + tn)
    f1_positive = 2 * tp / max(1, 2 * tp + fp + fn)
    f1_negative = 2 * tn / max(1, 2 * tn + fp + fn)
    return {
        "n": len(cohort), "positive": int(y.sum()), "prevalence": float(y.mean()),
        "average_precision": average_precision(y, score), "brier": float(np.mean((score - y) ** 2)),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp, "frame_fpr": fpr, "frame_recall": recall,
        "balanced_accuracy": 0.5 * (recall + 1.0 - fpr), "macro_f1": 0.5 * (f1_positive + f1_negative),
        "never_alarm": bool(not alarm.any()),
    }


def _alarm_runs(times: Sequence[int], alarms: Sequence[bool]) -> tuple[int, int, int]:
    lengths: list[int] = []
    current = 0
    previous: int | None = None
    if len(times) != len(alarms):
        raise ValueError("alarm timeline length mismatch")
    for time, active in zip(times, alarms):
        if active:
            if previous is not None and time == previous + 1:
                current += 1
            else:
                if current:
                    lengths.append(current)
                current = 1
            previous = time
        else:
            if current:
                lengths.append(current)
            current, previous = 0, None
    if current:
        lengths.append(current)
    return len(lengths), sum(lengths), max(lengths, default=0)


def event_metrics(rows: Sequence[Mapping[str, Any]], score_key: str, threshold: float, mask_key: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Count only a new causal alarm start before onset as an event detection."""
    by_episode: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_episode[str(row["episode_id"])].append(row)
    records: list[dict[str, Any]] = []
    for episode in sorted(by_episode):
        sequence = sorted(by_episode[episode], key=lambda row: int(row["t"]))
        eligible = [row for row in sequence if bool(row[mask_key])]
        onset = sequence[0].get("onset")
        onset = None if onset is None else int(onset)
        left_censored = onset is not None and onset <= int(sequence[0]["t"])
        negative = [row for row in eligible if int(row["target"]) == 0]
        positive = [row for row in eligible if int(row["target"]) == 1]
        alarms = [float(row[score_key]) >= threshold for row in sequence]
        starts = [False] + [alarms[index] and not alarms[index - 1] for index in range(1, len(sequence))]
        alarm_by_t = {int(row["t"]): active for row, active in zip(sequence, alarms)}
        start_by_t = {int(row["t"]): start for row, start in zip(sequence, starts)}
        negative_alarms = [alarm_by_t[int(row["t"])] for row in negative]
        false_runs, false_duration, max_false_duration = _alarm_runs(
            [int(row["t"]) for row in negative], negative_alarms
        )
        positive_starts = [int(row["t"]) for row in positive if start_by_t[int(row["t"])]]
        positive_overlaps = [int(row["t"]) for row in positive if alarm_by_t[int(row["t"])]]
        uncensored = onset is not None and not left_censored and bool(positive)
        late_starts = [] if onset is None else [
            int(row["t"]) for row in sequence if int(row["t"]) >= onset and start_by_t[int(row["t"])]
        ]
        lead = onset - min(positive_starts) if onset is not None and positive_starts else None
        records.append({
            "episode_id": episode, "leakage_group": str(sequence[0]["leakage_group"]),
            "left_censored": left_censored, "right_censored_or_no_observed_onset": onset is None or (not left_censored and not positive),
            "negative_frames": len(negative), "positive_frames": len(positive),
            "trial_false_alarm": bool(any(negative_alarms)), "false_alarm_runs": false_runs,
            "false_alarm_starts": sum(start_by_t[int(row["t"])] for row in negative),
            "false_alarm_duration": false_duration, "max_false_alarm_duration": max_false_duration,
            "uncensored_event": uncensored, "event_detected": bool(positive_starts) if uncensored else None,
            "lead_frames": lead, "late_alarm": bool(late_starts) if uncensored and not positive_starts else False,
            "active_overlap_detected": bool(positive_overlaps) if uncensored else None,
            "ongoing_from_before_positive_window": bool(positive_overlaps and not positive_starts) if uncensored else None,
            "active_at_first_observable_row_start_censored": bool(alarms[0]),
            "observed_no_alarm_complete_timeline": not any(alarms),
        })
    negative_trials = [row for row in records if row["negative_frames"]]
    events = [row for row in records if row["uncensored_event"]]
    detected = [row for row in events if row["event_detected"]]
    run_count = sum(row["false_alarm_runs"] for row in records)
    return {
        "trials": len(records), "negative_window_trials": len(negative_trials),
        "trial_false_alarm_rate": float(np.mean([row["trial_false_alarm"] for row in negative_trials])) if negative_trials else math.nan,
        "false_alarm_runs": run_count, "false_alarm_starts": sum(row["false_alarm_starts"] for row in records),
        "false_alarm_duration": sum(row["false_alarm_duration"] for row in records),
        "mean_false_alarm_run_duration": sum(row["false_alarm_duration"] for row in records) / run_count if run_count else 0.0,
        "max_false_alarm_duration": max((row["max_false_alarm_duration"] for row in records), default=0),
        "uncensored_events": len(events), "left_censored_events": sum(row["left_censored"] for row in records),
        "right_censored_or_no_observed_onset_trials": sum(row["right_censored_or_no_observed_onset"] for row in records),
        "event_recall": float(np.mean([row["event_detected"] for row in events])) if events else math.nan,
        "active_overlap_recall": float(np.mean([row["active_overlap_detected"] for row in events])) if events else math.nan,
        "mean_lead_frames": float(np.mean([row["lead_frames"] for row in detected])) if detected else math.nan,
        "lead_ge_3_recall": float(np.mean([(row["lead_frames"] or -1) >= 3 for row in events])) if events else math.nan,
        "lead_ge_5_recall": float(np.mean([(row["lead_frames"] or -1) >= 5 for row in events])) if events else math.nan,
        "misses": sum(not row["event_detected"] for row in events), "late_alarms": sum(row["late_alarm"] for row in events),
        "never_alarm_trials": sum(row["observed_no_alarm_complete_timeline"] for row in records),
    }, records


def _threshold_curve(rows: Sequence[Mapping[str, Any]], score_key: str, mask_key: str) -> list[dict[str, float]]:
    cohort = _cohort(rows, mask_key)
    y = np.asarray([int(row["target"]) for row in cohort], dtype=np.int8)
    score = np.asarray([float(row[score_key]) for row in cohort], dtype=float)
    if set(y.tolist()) != {0, 1}:
        raise ValueError("calibration cohort lacks both classes")
    candidates = [math.nextafter(1.0, math.inf), *sorted(set(score.tolist()), reverse=True)]
    result = []
    for threshold in candidates:
        alarm = score >= threshold
        tp = int(np.sum(alarm & (y == 1))); fp = int(np.sum(alarm & (y == 0)))
        recall = tp / int(y.sum()); fpr = fp / int((y == 0).sum())
        result.append({"threshold": threshold, "frame_recall": recall, "frame_fpr": fpr,
                       "balanced_accuracy": 0.5 * (recall + 1.0 - fpr)})
    return result


def select_threshold(rows: Sequence[Mapping[str, Any]], score_key: str, mask_key: str, kind: str, target: float | None = None) -> dict[str, Any]:
    """Select a threshold on calibration only; callers transfer it unchanged."""
    curve = _threshold_curve(rows, score_key, mask_key)
    if kind == "fixed":
        threshold = 0.5
    elif kind == "maxBA":
        threshold = max(curve, key=lambda row: (row["balanced_accuracy"], -row["frame_fpr"], row["threshold"]))["threshold"]
    elif kind == "frame_fpr":
        valid = [row for row in curve if row["frame_fpr"] <= float(target) + 1e-15]
        threshold = max(valid, key=lambda row: (row["frame_recall"], -row["frame_fpr"], row["threshold"]))["threshold"]
    elif kind in ("trial_fa", "event_recall"):
        candidates = []
        for row in curve:
            event = event_metrics(rows, score_key, row["threshold"], mask_key)[0]
            candidates.append({**row, **event})
        if kind == "trial_fa":
            valid = [row for row in candidates if row["trial_false_alarm_rate"] <= float(target) + 1e-15]
            key = lambda row: (row["event_recall"], -row["trial_false_alarm_rate"], -row["false_alarm_starts"], -row["false_alarm_duration"], row["threshold"])
            choose = max
        else:
            valid = [row for row in candidates if row["event_recall"] >= float(target) - 1e-15]
            key = lambda row: (row["trial_false_alarm_rate"], row["false_alarm_starts"], row["false_alarm_duration"], -row["mean_lead_frames"] if math.isfinite(row["mean_lead_frames"]) else math.inf, -row["threshold"])
            choose = min
        if not valid:
            return {"status": "unavailable_calibration_constraint", "threshold": None}
        threshold = choose(valid, key=key)["threshold"]
    else:
        raise ValueError(f"unknown operating point {kind!r}")
    return {"status": "available", "threshold": float(threshold)}


def fit_shared_monotone_platt(rows_by_horizon: Mapping[int, Sequence[Mapping[str, Any]]], score_key: Callable[[int], str], mask_key: str) -> dict[str, Any]:
    """Fit one positive-slope logit map pooled across horizons.

    The identical increasing map preserves p(H1)<=p(H3)<=p(H5) for every row.
    """
    examples = [(int(row["target"]), float(row[score_key(horizon)]))
                for horizon, rows in rows_by_horizon.items() for row in rows if bool(row[mask_key])]
    y = torch.tensor([target for target, _ in examples], dtype=torch.float64)
    if not len(examples) or int(torch.unique(y).numel()) != 2:
        return {"status": "unavailable_single_class"}
    q = torch.tensor([score for _, score in examples], dtype=torch.float64).clamp(1e-6, 1 - 1e-6)
    x = torch.logit(q)
    raw_slope = torch.tensor(math.log(math.expm1(1.0)), dtype=torch.float64, requires_grad=True)
    intercept = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([raw_slope, intercept], lr=1.0, max_iter=100, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(torch.nn.functional.softplus(raw_slope) * x + intercept, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    slope = float(torch.nn.functional.softplus(raw_slope).detach())
    bias = float(intercept.detach())
    if not math.isfinite(slope) or not math.isfinite(bias) or slope <= 0:
        return {"status": "unavailable_nonfinite"}
    return {"status": "available", "slope": slope, "intercept": bias, "epsilon": 1e-6,
            "fit_rows": len(examples), "shared_across_horizons": sorted(rows_by_horizon)}


def apply_shared_platt(score: float, model: Mapping[str, Any]) -> float:
    if model.get("status") != "available":
        raise ValueError("calibration model is unavailable")
    epsilon = float(model["epsilon"])
    probability = min(max(float(score), epsilon), 1.0 - epsilon)
    logit = math.log(probability / (1.0 - probability))
    transformed = max(min(float(model["slope"]) * logit + float(model["intercept"]), 40.0), -40.0)
    return 1.0 / (1.0 + math.exp(-transformed))


def fit_state_linear_baseline(fit_rows: Mapping[str, Any], *, ridge: float = 1e-3) -> dict[str, Any]:
    """Fit a multi-output linear force forecast using fit_train only.

    Inputs are the complete nine-step signed XYZ history (27 values). A separate
    ridge fit is used at each future step because censoring masks differ.
    """
    history = torch.as_tensor(fit_rows["signed_force_xyz"]).detach().cpu().double()
    target = torch.as_tensor(fit_rows["future_force_target"]).detach().cpu().double()
    mask = torch.as_tensor(fit_rows["future_force_observed_mask"]).detach().cpu().bool()
    if history.ndim != 3 or tuple(history.shape[1:]) != (9, 3) or target.shape[1:] != (5, 3) or mask.shape[1:] != (5,):
        raise ValueError("invalid state-linear baseline tensors")
    x = history.reshape(len(history), -1).numpy()
    mean = x.mean(axis=0); std = np.maximum(x.std(axis=0), 1e-6)
    design_all = np.c_[np.ones(len(x)), (x - mean) / std]
    models = []
    for step in range(5):
        valid = mask[:, step].numpy()
        if int(valid.sum()) <= design_all.shape[1]:
            raise ValueError(f"insufficient fit rows for state linear step {step + 1}")
        design = design_all[valid]
        y = target[valid, step].numpy()
        penalty = np.diag(np.r_[0.0, np.full(design.shape[1] - 1, ridge)])
        weights = np.linalg.solve(design.T @ design / len(design) + penalty, design.T @ y / len(design))
        if not np.isfinite(weights).all():
            raise FloatingPointError("nonfinite state linear baseline")
        models.append({"step": step + 1, "fit_rows": int(valid.sum()), "weights": weights.tolist()})
    return {"schema": "round8_fit_only_state_linear_v1", "fit_role": "fit_train", "ridge": ridge,
            "input": "flattened signed_force_xyz t-8..t", "input_mean": mean.tolist(), "input_std": std.tolist(),
            "steps": models}


def predict_state_linear_baseline(model: Mapping[str, Any], rows: Mapping[str, Any]) -> np.ndarray:
    history = torch.as_tensor(rows["signed_force_xyz"]).detach().cpu().double()
    if history.ndim != 3 or tuple(history.shape[1:]) != (9, 3):
        raise ValueError("invalid state-linear prediction history")
    x = history.reshape(len(history), -1).numpy()
    mean = np.asarray(model["input_mean"], dtype=float); std = np.asarray(model["input_std"], dtype=float)
    design = np.c_[np.ones(len(x)), (x - mean) / std]
    predictions = np.stack([design @ np.asarray(step["weights"], dtype=float) for step in model["steps"]], axis=1)
    if predictions.shape != (len(history), 5, 3) or not np.isfinite(predictions).all():
        raise FloatingPointError("invalid state linear predictions")
    return predictions


def paired_cluster_bootstrap(
    candidate: Sequence[Mapping[str, Any]],
    comparator: Sequence[Mapping[str, Any]],
    statistic: Callable[[Sequence[Mapping[str, Any]]], float],
    *, repetitions: int = 200,
    seed: int = 8001,
) -> dict[str, Any]:
    """Paired bootstrap over complete leakage groups with identical row identity."""
    identity = lambda row: (str(row.get("seed", "")), str(row["episode_id"]), int(row.get("t", 0)))
    candidate_sorted = sorted(candidate, key=identity)
    comparator_sorted = sorted(comparator, key=identity)
    if [identity(row) for row in candidate_sorted] != [identity(row) for row in comparator_sorted]:
        raise ValueError("paired bootstrap row identity drift")
    if [str(row["leakage_group"]) for row in candidate_sorted] != [str(row["leakage_group"]) for row in comparator_sorted]:
        raise ValueError("paired bootstrap leakage-group drift")
    by_candidate: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_comparator: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidate_sorted:
        by_candidate[str(row["leakage_group"])].append(row)
    for row in comparator_sorted:
        by_comparator[str(row["leakage_group"])].append(row)
    groups = sorted(by_candidate)
    if groups != sorted(by_comparator):
        raise ValueError("paired bootstrap cluster identity drift")

    def materialize(source: Mapping[str, Sequence[Mapping[str, Any]]], draw: Sequence[str]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for copy_index, group in enumerate(draw):
            for row in source[group]:
                clone = dict(row)
                clone["episode_id"] = f"bootstrap_{copy_index}:{row['episode_id']}"
                clone["leakage_group"] = f"bootstrap_{copy_index}:{group}"
                result.append(clone)
        return result

    estimate = float(statistic(candidate_sorted) - statistic(comparator_sorted))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        draw = rng.choice(groups, size=len(groups), replace=True).tolist()
        difference = float(statistic(materialize(by_candidate, draw)) - statistic(materialize(by_comparator, draw)))
        if math.isfinite(difference):
            values.append(difference)
    return {"estimate": estimate, "ci_low": float(np.percentile(values, 2.5)) if values else None,
            "ci_high": float(np.percentile(values, 97.5)) if values else None,
            "valid_replicates": len(values), "requested_replicates": repetitions,
            "unit": "leakage_group", "paired": True}
