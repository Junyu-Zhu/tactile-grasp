"""Exact-memory replacement for the frozen R7 compact event threshold sweep."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np


def _descending_interval(thresholds: np.ndarray, score: float, previous: float) -> tuple[int, int]:
    """Indices satisfying previous < threshold <= score for descending thresholds."""
    start = int(np.searchsorted(-thresholds, -score, side="left"))
    stop = int(np.searchsorted(-thresholds, -previous, side="left"))
    return start, stop


def compact_event_curve(rows: list[dict[str, Any]], score_key: str, mask_key: str,
                        thresholds: list[float]) -> list[dict[str, Any]]:
    """Match R7 compact_event_curve without allocating threshold×frame matrices."""
    threshold_array = np.asarray(thresholds, dtype=float)
    if threshold_array.ndim != 1 or not len(threshold_array) or not np.isfinite(threshold_array).all():
        raise ValueError("invalid threshold vector")
    if np.any(threshold_array[1:] > threshold_array[:-1]):
        raise ValueError("thresholds must be descending")

    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_episode[str(row["episode_id"])].append(row)
    negative_max: list[float] = []
    negative_scores: list[float] = []
    start_difference = np.zeros(len(threshold_array) + 1, dtype=np.int64)
    events: list[tuple[int, list[tuple[int, float, float]]]] = []
    for sequence in by_episode.values():
        sequence = sorted(sequence, key=lambda row: int(row["t"]))
        eligible = [row for row in sequence if bool(row[mask_key])]
        negative = [row for row in eligible if int(row["target"]) == 0]
        positive = [row for row in eligible if int(row["target"]) == 1]
        previous = {int(row["t"]): (math.inf if index == 0 else float(sequence[index - 1][score_key]))
                    for index, row in enumerate(sequence)}
        if negative:
            scores = [float(row[score_key]) for row in negative]
            negative_max.append(max(scores)); negative_scores.extend(scores)
            for row, score in zip(negative, scores):
                start, stop = _descending_interval(threshold_array, score, previous[int(row["t"])])
                if start < stop:
                    start_difference[start] += 1; start_difference[stop] -= 1
        onset = sequence[0].get("onset")
        if onset is not None and int(onset) > min(int(row["t"]) for row in sequence) and positive:
            events.append((int(onset), [(int(row["t"]), float(row[score_key]), previous[int(row["t"])])
                                        for row in positive]))

    if negative_max:
        maxima = np.sort(np.asarray(negative_max, dtype=float))
        false_alarm = (len(maxima) - np.searchsorted(maxima, threshold_array, side="left")) / len(maxima)
    else:
        false_alarm = np.full(len(threshold_array), np.nan)
    if negative_scores:
        sorted_scores = np.sort(np.asarray(negative_scores, dtype=float))
        false_duration = len(sorted_scores) - np.searchsorted(sorted_scores, threshold_array, side="left")
    else:
        false_duration = np.zeros(len(threshold_array), dtype=np.int64)
    false_starts = np.cumsum(start_difference[:-1])

    detected_count = np.zeros(len(threshold_array), dtype=np.int64)
    lead_sum = np.zeros(len(threshold_array), dtype=float)
    lead_count = np.zeros(len(threshold_array), dtype=np.int64)
    sentinel = np.iinfo(np.int64).max
    for onset, points in events:
        first_time = np.full(len(threshold_array), sentinel, dtype=np.int64)
        for time, score, previous in points:
            start, stop = _descending_interval(threshold_array, score, previous)
            if start < stop:
                first_time[start:stop] = np.minimum(first_time[start:stop], time)
        detected = first_time != sentinel
        detected_count += detected
        lead_sum[detected] += onset - first_time[detected]
        lead_count[detected] += 1
    recall = detected_count / len(events) if events else np.full(len(threshold_array), np.nan)
    mean_lead = np.divide(lead_sum, lead_count, out=np.full(len(threshold_array), np.nan), where=lead_count > 0)
    return [{"threshold": float(threshold), "trial_false_alarm_rate": float(false_alarm[index]),
             "event_recall": float(recall[index]), "mean_lead_frames": float(mean_lead[index]),
             "false_alarm_starts": int(false_starts[index]), "false_alarm_duration": int(false_duration[index])}
            for index, threshold in enumerate(threshold_array)]
