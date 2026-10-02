#!/usr/bin/env python3
"""Round 13 fixed-protocol trial-level alarm calibration and evaluation.

This program is deliberately self contained.  It reads only frozen predictions and
endpoint tables, never imports a training stack, and never accepts validation data
in the threshold fitting interface.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import socket
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

GROUPS = ("V_original", "F_history_original", "V_class_trial_balanced", "F_class_trial_balanced")
FOLDS = tuple(f"htt_leave_p{i}" for i in range(1, 5))
SEEDS = (20260914, 20260915, 20260916)
ROLES = ("train", "calibration", "validation")
FAMILIES = ("trial_macro_static_FPR", "trial_any_static_alarm_rate")
ALPHAS = (0.01, 0.05, 0.10)
KS = (1, 2, 4)
NEVER = float(np.nextafter(np.float64(1.0), np.float64(np.inf)))
PAIR_METRICS = (
    "frame_static_FPR", "trial_macro_static_FPR", "trial_any_static_alarm_rate",
    "gross_recall", "balanced_accuracy", "macro_f1", "false_starts_per_trial",
    "static_alarming_frames", "static_support_trial_denominator",
    "static_alarming_frames_per_static_support_trial",
    "event_recall", "mean_detected_event_delay",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(clean(rows))
    os.replace(tmp, path)


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def verified(record: Mapping[str, str]) -> Path:
    path = Path(record["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != record["sha256"]:
        raise ValueError(f"SHA mismatch: {path}: {actual} != {record['sha256']}")
    return path


@dataclass(frozen=True)
class Episode:
    episode: str
    group: str
    probe: str
    t: np.ndarray
    stage: np.ndarray
    score: np.ndarray


def validate_index(index: Mapping) -> None:
    if index.get("schema") != "round13_startup_artifact_index_v1":
        raise ValueError("wrong startup index schema")
    runs = index["selected_runs"]
    expected = {(g, f, s) for g in GROUPS for f in FOLDS for s in SEEDS}
    actual = {(r["group"], r["fold"], int(r["seed"])) for r in runs}
    if len(runs) != 48 or actual != expected:
        raise ValueError("expected exact 4x4x3 run grid")
    for run in runs:
        if set(run["predictions"]) != set(ROLES) or set(run["endpoints"]) != set(ROLES):
            raise ValueError("missing role record")
        for role in ROLES:
            if "test" in run["predictions"][role]["path"].lower():
                raise ValueError("test-role path rejected")


def load_episodes(run: Mapping, role: str) -> list[Episode]:
    endpoint_path = verified(run["endpoints"][role])
    prediction_path = verified(run["predictions"][role])
    expected = {}
    ep_groups = {}
    ep_probes = {}
    for row in read_csv(endpoint_path):
        key = (row["episode_id"], int(row["t"]))
        if int(row["t"]) < 13 or key in expected:
            raise ValueError(f"invalid endpoint {key}")
        if row.get("role") != role or row.get("fold") != run["fold"]:
            raise ValueError("endpoint role/fold mismatch")
        stage = int(row["stage"])
        if stage not in (0, 1, 2):
            raise ValueError("unknown stage")
        group = row["leakage_group"]
        episode = row["episode_id"]
        if episode in ep_groups and ep_groups[episode] != group:
            raise ValueError("episode maps to multiple leakage groups")
        ep_groups[episode] = group
        ep_probes[episode] = row.get("probe", "unknown") or "unknown"
        expected[key] = stage
    observed = {}
    for row in read_csv(prediction_path):
        key = (row.get("episode_id", row.get("episode")), int(row.get("t", row.get("frame"))))
        if key in observed:
            raise ValueError("duplicate prediction")
        if key not in expected:
            if run.get("source_round") in ("round9", "round10", "round11"):
                continue
            raise ValueError("unexpected prediction endpoint")
        score = float(row.get("p_slip", row.get("probability", row.get("score", "nan"))))
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("invalid score")
        if int(row["stage"]) != expected[key] or row.get("role", role) != role:
            raise ValueError("prediction label/role mismatch")
        observed[key] = score
    if set(observed) != set(expected) or not observed:
        raise ValueError("missing prediction endpoints")
    by_episode = defaultdict(list)
    for (episode, t), stage in expected.items():
        by_episode[episode].append((t, stage, observed[(episode, t)]))
    result = []
    for episode, values in sorted(by_episode.items()):
        values.sort()
        result.append(Episode(
            episode, ep_groups[episode], ep_probes[episode],
            np.asarray([v[0] for v in values], dtype=np.int64),
            np.asarray([v[1] for v in values], dtype=np.int8),
            np.asarray([v[2] for v in values], dtype=np.float64),
        ))
    return result


def effective_score(ep: Episode, k: int) -> np.ndarray:
    """Causal rolling minimum; gaps reset and warm-up values are -inf."""
    q = np.full(len(ep.t), -np.inf, dtype=np.float64)
    start = 0
    for i in range(len(ep.t)):
        if i == 0 or ep.t[i] != ep.t[i - 1] + 1:
            start = i
        if i - start + 1 >= k:
            q[i] = float(np.min(ep.score[i - k + 1:i + 1]))
    return q


def state_alarm(ep: Episode, threshold: float, k: int) -> tuple[np.ndarray, np.ndarray]:
    alarm = np.zeros(len(ep.t), dtype=bool)
    starts = np.zeros(len(ep.t), dtype=bool)
    active = False
    count = 0
    for i, score in enumerate(ep.score):
        if i == 0 or ep.t[i] != ep.t[i - 1] + 1:
            active, count = False, 0
        if active:
            if score < threshold:
                active, count = False, 0
        else:
            count = count + 1 if score >= threshold else 0
            if count >= k:
                active = True
                starts[i] = True
        alarm[i] = active
    return alarm, starts


def episode_stats(ep: Episode, threshold: float, k: int) -> dict:
    alarm, starts = state_alarm(ep, threshold, k)
    static = ep.stage == 0
    gross = ep.stage == 2
    incipient = ep.stage == 1
    breaks = np.r_[True, np.diff(ep.t) != 1]
    events = hits = left = right = covered = prealarm = prealarm_delay0 = 0
    delay_sum = delay_n = 0
    for i in range(len(ep.t)):
        if not gross[i] or (i and not breaks[i] and gross[i - 1]):
            continue
        j = i + 1
        while j < len(ep.t) and not breaks[j] and gross[j]:
            j += 1
        hit = np.flatnonzero(alarm[i:j])
        covered += int(bool(len(hit)))
        is_left = bool(breaks[i])
        left += int(is_left)
        right += int(j == len(ep.t) or (j < len(ep.t) and breaks[j]))
        if not is_left:
            events += 1
            prior = bool(i > 0 and not breaks[i] and alarm[i - 1])
            prealarm += int(prior)
            if len(hit):
                hits += 1
                delay_sum += int(hit[0])
                delay_n += 1
                prealarm_delay0 += int(prior and int(hit[0]) == 0)
    longest = 0
    current = 0
    for i in range(len(ep.t)):
        if static[i] and alarm[i] and (i == 0 or (not breaks[i] and static[i - 1] and alarm[i - 1])):
            current += 1
        elif static[i] and alarm[i]:
            current = 1
        else:
            current = 0
        longest = max(longest, current)
    static_n = int(static.sum())
    fp = int((static & alarm).sum())
    tp = int((gross & alarm).sum())
    return {
        "episode": ep.episode, "leakage_group": ep.group, "probe": ep.probe,
        "static_frames": static_n, "static_fp": fp, "static_any_alarm": int(fp > 0),
        "gross_frames": int(gross.sum()), "gross_tp": tp,
        "tn": static_n - fp, "fp": fp, "fn": int(gross.sum()) - tp, "tp": tp,
        "false_starts": int((starts & static).sum()), "alarm_starts": int(starts.sum()),
        "longest_static_alarm_segment": longest, "events": events, "event_hits": hits,
        "left_censored_events": left, "right_boundary_events": right,
        "gross_segments": events + left, "segments_covered": covered,
        "preexisting_alarm_events": prealarm, "preexisting_alarm_delay0_events": prealarm_delay0,
        "delay_sum": delay_sum, "delay_n": delay_n,
        "trial_count": 1, "incipient_frames": int(incipient.sum()),
        "observed_alarm": int(alarm.any()),
    }


def div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def aggregate(stats: Sequence[Mapping], weights: Mapping[str, int] | None = None) -> dict:
    keys = (
        "static_frames", "static_fp", "gross_frames", "gross_tp", "tn", "fp", "fn", "tp",
        "false_starts", "alarm_starts", "events", "event_hits", "left_censored_events",
        "right_boundary_events", "gross_segments", "segments_covered", "preexisting_alarm_events",
        "preexisting_alarm_delay0_events", "delay_sum", "delay_n", "trial_count", "incipient_frames", "observed_alarm",
    )
    total = {key: 0.0 for key in keys}
    macro_sum = any_sum = static_trials = 0.0
    longest = 0
    for row in stats:
        w = 1 if weights is None else weights.get(row["leakage_group"], 0)
        if not w:
            continue
        for key in keys:
            total[key] += w * row[key]
        if row["static_frames"] > 0:
            static_trials += w
            macro_sum += w * row["static_fp"] / row["static_frames"]
            any_sum += w * row["static_any_alarm"]
        longest = max(longest, row["longest_static_alarm_segment"])
    tn, fp, fn, tp = (total[k] for k in ("tn", "fp", "fn", "tp"))
    out = dict(total)
    out.update({
        "static_support_trials": static_trials,
        "static_alarming_frames": total["static_fp"],
        "static_support_trial_denominator": static_trials,
        "static_alarming_frames_per_static_support_trial": div(total["static_fp"], static_trials),
        "frame_static_FPR": div(total["static_fp"], total["static_frames"]),
        "trial_macro_static_FPR": div(macro_sum, static_trials),
        "trial_any_static_alarm_rate": div(any_sum, static_trials),
        "gross_recall": div(total["gross_tp"], total["gross_frames"]),
        "balanced_accuracy": (1.0 - div(fp, tn + fp) + div(tp, tp + fn)) / 2.0 if tn + fp and tp + fn else float("nan"),
        "macro_f1": ((div(2 * tn, 2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0) +
                     (div(2 * tp, 2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)) / 2.0,
        "false_starts_per_trial": div(total["false_starts"], total["trial_count"]),
        "event_recall": div(total["event_hits"], total["events"]),
        "mean_detected_event_delay": div(total["delay_sum"], total["delay_n"]),
        "preexisting_alarm_event_rate": div(total["preexisting_alarm_events"], total["events"]),
        "preexisting_alarm_delay0_rate": div(total["preexisting_alarm_delay0_events"], total["events"]),
        "segment_coverage": div(total["segments_covered"], total["gross_segments"]),
        "longest_static_alarm_segment": longest,
        "observed_no_alarm": bool(total["observed_alarm"] == 0),
    })
    return out


def rank_metrics(episodes: Sequence[Episode]) -> dict:
    scores = np.concatenate([ep.score[ep.stage != 1] for ep in episodes])
    labels = np.concatenate([ep.stage[ep.stage != 1] == 2 for ep in episodes])
    if not labels.any() or labels.all():
        return {"AP": float("nan"), "pAUC": float("nan"), "positive_prevalence": float("nan")}
    order = np.argsort(-scores, kind="stable")
    labels = labels[order]
    scores = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]
    tp = np.cumsum(labels)[ends]
    fp = np.cumsum(~labels)[ends]
    rec = tp / labels.sum()
    precision = tp / (tp + fp)
    ap = float(np.sum(np.diff(np.r_[0.0, rec]) * precision))
    fpr = np.r_[0.0, fp / (~labels).sum()]
    tpr = np.r_[0.0, rec]
    stop = np.searchsorted(fpr, 0.1, side="right")
    xx = np.r_[fpr[:stop], 0.1]
    yy = np.r_[tpr[:stop], np.interp(0.1, fpr, tpr)]
    return {"AP": ap, "pAUC": float(np.trapezoid(yy, xx) / 0.1), "positive_prevalence": float(labels.mean())}


def candidate_thresholds(episodes: Sequence[Episode], k: int) -> np.ndarray:
    values = []
    for ep in episodes:
        q = effective_score(ep, k)
        mask = (ep.stage != 1) & np.isfinite(q)
        values.extend(q[mask].tolist())
    return np.asarray(sorted(set(values + [NEVER]), reverse=True), dtype=np.float64)


@dataclass
class FitCache:
    episodes: Sequence[Episode]
    k: int
    thresholds: np.ndarray
    static_n: np.ndarray
    gross_n: np.ndarray
    static_fp: np.ndarray
    gross_tp: np.ndarray
    candidate_groups: dict[float, set[str]]


def build_fit_cache(episodes: Sequence[Episode], k: int) -> FitCache:
    """Cache threshold sufficient statistics; avoids replaying trajectories per draw."""
    thresholds = candidate_thresholds(episodes, k)
    static_n = np.asarray([np.sum(ep.stage == 0) for ep in episodes], dtype=float)
    gross_n = np.asarray([np.sum(ep.stage == 2) for ep in episodes], dtype=float)
    static_fp = np.empty((len(episodes), len(thresholds)), dtype=np.float64)
    gross_tp = np.empty_like(static_fp)
    candidate_groups: dict[float, set[str]] = defaultdict(set)
    candidate_groups[NEVER].update(ep.group for ep in episodes)
    for i, ep in enumerate(episodes):
        q = effective_score(ep, k)
        qs = np.sort(q[(ep.stage == 0) & np.isfinite(q)])
        qg = np.sort(q[(ep.stage == 2) & np.isfinite(q)])
        # searchsorted on ascending arrays gives count(q >= threshold).
        static_fp[i] = len(qs) - np.searchsorted(qs, thresholds, side="left")
        gross_tp[i] = len(qg) - np.searchsorted(qg, thresholds, side="left")
        for value in np.unique(q[(ep.stage != 1) & np.isfinite(q)]):
            candidate_groups[float(value)].add(ep.group)
    return FitCache(episodes, k, thresholds, static_n, gross_n, static_fp, gross_tp, candidate_groups)


def fit_cached(cache: FitCache, family: str, alpha: float,
               weights: Mapping[str, int] | None = None,
               constraint_override: str | None = None) -> dict:
    if family not in FAMILIES or alpha not in ALPHAS:
        raise ValueError("unsupported fixed policy")
    ew = np.asarray([1 if weights is None else weights.get(ep.group, 0) for ep in cache.episodes], dtype=float)
    static_trials_mask = cache.static_n > 0
    static_trials = float(np.sum(ew[static_trials_mask]))
    gross_frames = float(np.sum(ew * cache.gross_n))
    if not static_trials or not gross_frames:
        return {"status": "unsupported", "threshold": None, "never_alarm": None,
                "static_support_trials": static_trials, "gross_support_frames": gross_frames}
    total_static = float(np.sum(ew * cache.static_n))
    fp = np.sum(cache.static_fp * ew[:, None], axis=0)
    tp = np.sum(cache.gross_tp * ew[:, None], axis=0)
    frame = fp / total_static
    macro = np.sum(np.divide(cache.static_fp, cache.static_n[:, None],
                             out=np.zeros_like(cache.static_fp), where=cache.static_n[:, None] > 0) * ew[:, None], axis=0) / static_trials
    any_rate = np.sum((cache.static_fp > 0) * ew[:, None], axis=0) / static_trials
    recall = tp / gross_frames
    metric_name = constraint_override or family
    constraint = {"frame_static_FPR": frame, FAMILIES[0]: macro, FAMILIES[1]: any_rate}[metric_name]
    selected_groups = {ep.group for ep, w in zip(cache.episodes, ew) if w > 0}
    allowed_candidate = np.asarray([
        th == NEVER or bool(cache.candidate_groups.get(float(th), set()) & selected_groups)
        for th in cache.thresholds], dtype=bool)
    feasible = np.flatnonzero((constraint <= alpha + 1e-15) & allowed_candidate)
    if not len(feasible):
        raise AssertionError("never-alarm candidate must be feasible")
    # Lexicographic max: recall, lower constrained error, higher threshold.
    best = max(feasible, key=lambda i: (recall[i], -constraint[i], cache.thresholds[i]))
    return {"status": "fit", "threshold": float(cache.thresholds[best]),
            "never_alarm": bool(cache.thresholds[best] > 1.0),
            "constraint_value": float(constraint[best]), "calibration_gross_recall": float(recall[best]),
            "static_support_trials": static_trials, "gross_support_frames": gross_frames}


def fit_policy(calibration: Sequence[Episode], family: str, alpha: float, k: int,
               weights: Mapping[str, int] | None = None) -> dict:
    """Fit on calibration episodes only.  No validation argument exists by design."""
    if family not in FAMILIES or k not in KS or alpha not in ALPHAS:
        raise ValueError("unsupported fixed policy")
    active_eps = [ep for ep in calibration if weights is None or weights.get(ep.group, 0) > 0]
    support_static = sum((weights.get(ep.group, 0) if weights else 1) for ep in active_eps if np.any(ep.stage == 0))
    support_gross = sum((weights.get(ep.group, 0) if weights else 1) * int(np.sum(ep.stage == 2)) for ep in active_eps)
    if not support_static or not support_gross:
        return {"status": "unsupported", "threshold": None, "never_alarm": None,
                "static_support_trials": support_static, "gross_support_frames": support_gross}
    best = None
    for threshold in candidate_thresholds(active_eps, k):
        stats = [episode_stats(ep, float(threshold), k) for ep in active_eps]
        agg = aggregate(stats, weights)
        constraint = agg[family]
        if np.isfinite(constraint) and constraint <= alpha + 1e-15:
            key = (agg["gross_recall"], -constraint, float(threshold))
            if best is None or key > best[0]:
                best = (key, float(threshold), agg)
    if best is None:
        raise AssertionError("never-alarm candidate must be feasible")
    return {"status": "fit", "threshold": best[1], "never_alarm": bool(best[1] > 1.0),
            "constraint_value": best[2][family], "calibration_gross_recall": best[2]["gross_recall"],
            "static_support_trials": support_static, "gross_support_frames": support_gross}


def policy_name(family: str, alpha: float, k: int) -> str:
    return f"{family}|a{alpha:.2f}|k{k}"


def evaluate_policy(episodes: Sequence[Episode], threshold: float, k: int) -> tuple[list[dict], dict]:
    stats = [episode_stats(ep, threshold, k) for ep in episodes]
    return stats, aggregate(stats)


def historical_subset(r12_metrics: Path, r12_trials: Path, allowed: set[tuple]) -> tuple[list[dict], list[dict]]:
    points = {"fixed_0.5", "maxBA", "FPR0.01", "FPR0.05", "FPR0.10"}
    metrics = [r for r in read_csv(r12_metrics)
               if (r["group"], r["fold"], int(r["seed"])) in allowed and r["point"] in points]
    trials = [r for r in read_csv(r12_trials)
              if (r["group"], r["fold"], int(r["seed"])) in allowed and r["point"] in points]
    if len(metrics) != 48 * 5 * 2 * 3:
        raise ValueError(f"historical metric subset incomplete: {len(metrics)}")
    return metrics, trials


def threshold_from_history(rows: Sequence[Mapping], ident: tuple, alpha: float) -> float:
    point = f"FPR{alpha:.2f}"
    found = [r for r in rows if (r["group"], r["fold"], int(r["seed"])) == ident
             and r["role"] == "calibration" and r["point"] == point and r["rule"] == "raw"]
    if len(found) != 1:
        raise ValueError(f"missing historical threshold {ident} {point}")
    return float(found[0]["threshold"])


def bootstrap_ci(values: list[float]) -> tuple[float, float, int]:
    valid = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if not len(valid):
        return float("nan"), float("nan"), 0
    return float(np.quantile(valid, 0.025, method="linear")), float(np.quantile(valid, 0.975, method="linear")), len(valid)


def machine_config() -> dict:
    return {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(), "hostname": socket.gethostname(),
        "platform": platform.platform(), "python": sys.version, "numpy": np.__version__,
        "cpu_count": os.cpu_count(), "XFORMERS_DISABLED": os.environ.get("XFORMERS_DISABLED"),
        "new_neural_trainings": 0, "device_policy": "CPU-only frozen prediction evaluation",
    }


def smoke() -> dict:
    def ep(name, t, stage, score, group="g"):
        return Episode(name, group, "p", np.asarray(t), np.asarray(stage), np.asarray(score, float))
    cases = []
    base = ep("e", [13, 14, 15, 17, 18, 19], [0, 1, 2, 0, 2, 2], [.7, .7, .7, .7, .4, .7])
    for k in KS:
        for th in (0.4, 0.7, NEVER):
            alarm, _ = state_alarm(base, th, k)
            q = effective_score(base, k)
            if not np.array_equal(alarm, q >= th):
                raise AssertionError("effective score/state mismatch")
            cases.append((k, th, alarm.tolist()))
    # >= ties, gap reset, incipient continuity, release, cold start.
    if state_alarm(base, .7, 2)[0].tolist() != [False, True, True, False, False, False]:
        raise AssertionError("state-machine boundary semantics")
    # Static-free/gross-free are unsupported, never-alarm is feasible otherwise.
    no_static = [ep("x", [13, 14], [2, 2], [.5, .6])]
    no_gross = [ep("x", [13, 14], [0, 0], [.5, .6])]
    if fit_policy(no_static, FAMILIES[0], .05, 1)["status"] != "unsupported":
        raise AssertionError("missing static support")
    if fit_policy(no_gross, FAMILIES[0], .05, 1)["status"] != "unsupported":
        raise AssertionError("missing gross support")
    # Macro and any denominators exclude static-free trials and preserve group multiplicity.
    eps = [ep("a", [13, 14], [0, 2], [.9, .9], "g1"), ep("b", [13, 14], [0, 0], [.1, .1], "g2"), no_static[0]]
    stats = [episode_stats(x, .5, 1) for x in eps]
    agg = aggregate(stats)
    if agg["static_support_trials"] != 2 or agg["trial_macro_static_FPR"] != .5 or agg["trial_any_static_alarm_rate"] != .5:
        raise AssertionError("trial denominator semantics")
    weighted = aggregate(stats, {"g1": 2, "g2": 1, "g": 0})
    if weighted["static_support_trials"] != 3 or not np.isclose(weighted["trial_macro_static_FPR"], 2 / 3):
        raise AssertionError("group multiplicity semantics")
    # Existing alarm before a non-left-censored gross event is marked and delay can be zero.
    pre = ep("pre", [13, 14, 15], [0, 0, 2], [.9, .9, .9])
    s = episode_stats(pre, .5, 1)
    if s["events"] != 1 or s["event_hits"] != 1 or s["preexisting_alarm_events"] != 1 or s["delay_sum"] != 0:
        raise AssertionError("preexisting alarm event semantics")
    if s["preexisting_alarm_delay0_events"] != 1:
        raise AssertionError("preexisting alarm delay-zero subset")
    # Candidate fitting has no validation parameter and tie-breaks to higher threshold.
    fit = fit_policy(eps[:2], FAMILIES[0], .10, 1)
    if fit["status"] != "fit" or fit["threshold"] < 0.0:
        raise AssertionError("calibration fitting")
    cache = build_fit_cache(eps[:2], 1)
    cached = fit_cached(cache, FAMILIES[0], .10)
    if cached["threshold"] != fit["threshold"]:
        raise AssertionError("cached/literal fitting mismatch")
    return {"status": "pass", "cases": len(cases), "checks": 10,
            "calibration_only_signature": "fit_policy(calibration,family,alpha,k,weights=None)"}


def run(args) -> None:
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    index = json.loads(args.index.read_text())
    validate_index(index)
    runs = index["selected_runs"]
    allowed = {(r["group"], r["fold"], int(r["seed"])) for r in runs}
    historical_metrics, historical_trials = historical_subset(args.r12_metrics, args.r12_trials, allowed)
    write_csv(root / "historical" / "metrics_48.csv", historical_metrics)
    write_csv(root / "historical" / "trials_48.csv", historical_trials)
    draws = json.loads(args.group_draws.read_text())
    datasets = {}
    canonical = {}
    mapping_rows = []
    input_hashes = {}
    for run in runs:
        ident = (run["group"], run["fold"], int(run["seed"]))
        datasets[ident] = {}
        for role in ROLES:
            episodes = load_episodes(run, role)
            datasets[ident][role] = episodes
            signature = [(e.episode, e.group, tuple(e.t), tuple(e.stage)) for e in episodes]
            key = (run["fold"], role)
            if key in canonical and canonical[key] != signature:
                raise ValueError(f"non-common endpoints: {key}")
            canonical[key] = signature
            for e in episodes:
                mapping_rows.append({"fold": run["fold"], "role": role, "episode": e.episode,
                                     "leakage_group": e.group, "probe": e.probe})
            for kind in ("predictions", "endpoints"):
                record = run[kind][role]
                input_hashes[record["path"]] = record["sha256"]
        role_groups = [{e.group for e in datasets[ident][role]} for role in ROLES]
        if any(role_groups[i] & role_groups[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError(f"role leakage {ident}")
    mapping_unique = list({(r["fold"], r["role"], r["episode"], r["leakage_group"], r["probe"]): r for r in mapping_rows}.values())
    write_csv(root / "audit" / "episode_group_mapping.csv", mapping_unique)
    multiplicity = defaultdict(set)
    for r in mapping_unique:
        multiplicity[(r["fold"], r["role"], r["leakage_group"])].add(r["episode"])
    write_csv(root / "audit" / "group_episode_multiplicity.csv", [
        {"fold": k[0], "role": k[1], "leakage_group": k[2], "episode_count": len(v),
         "episodes": "|".join(sorted(v))} for k, v in sorted(multiplicity.items())])

    thresholds_rows, metrics_rows, trial_rows, bootstrap_rows = [], [], [], []
    tail_rows, loo_rows, confirm4_metrics, confirm4_trials = [], [], [], []
    policy_stats = {}
    for run_i, run in enumerate(runs, 1):
        ident = (run["group"], run["fold"], int(run["seed"]))
        meta = {"group": ident[0], "fold": ident[1], "seed": ident[2]}
        cal = datasets[ident]["calibration"]
        fit_caches = {k: build_fit_cache(cal, k) for k in KS}
        cal_groups = sorted({e.group for e in cal})
        expected_draw_groups = set(draws[ident[1]][0])
        if set(cal_groups) != expected_draw_groups or len(draws[ident[1]]) != 200:
            raise ValueError("R12 group draws incompatible with current calibration groups")
        for role in ROLES:
            for ep in datasets[ident][role]:
                static = ep.score[ep.stage == 0]
                static_t = ep.t[ep.stage == 0]
                max_i = int(np.argmax(static)) if len(static) else None
                tail_rows.append({**meta, "role": role, "episode": ep.episode,
                                  "leakage_group": ep.group, "probe": ep.probe, "static_frames": len(static),
                                  "static_start_t": int(static_t.min()) if len(static_t) else None,
                                  "static_end_t": int(static_t.max()) if len(static_t) else None,
                                  "max_score_t": int(static_t[max_i]) if max_i is not None else None,
                                  "max_score_relative_position": div(max_i, len(static) - 1) if len(static) > 1 else (0.0 if len(static) else None),
                                  **{f"q{int(q*100):02d}": float(np.quantile(static, q)) if len(static) else None
                                     for q in (.5, .9, .95, .99, 1.0)}})
        for family in FAMILIES:
            for alpha in ALPHAS:
                for k in KS:
                    name = policy_name(family, alpha, k)
                    fit = fit_cached(fit_caches[k], family, alpha)
                    thresholds_rows.append({**meta, "policy": name, "family": family, "alpha": alpha, "k": k, **fit})
                    if fit["status"] != "fit":
                        continue
                    threshold = fit["threshold"]
                    for role in ROLES:
                        stats, agg = evaluate_policy(datasets[ident][role], threshold, k)
                        rank = rank_metrics(datasets[ident][role])
                        metrics_rows.append({**meta, "role": role, "policy": name, "family": family,
                                             "alpha": alpha, "k": k, "threshold": threshold,
                                             "selection": "calibration_filtered_constraint", **agg, **rank})
                        rows = [{**meta, "role": role, "policy": name, "family": family,
                                 "alpha": alpha, "k": k, "threshold": threshold, **s} for s in stats]
                        trial_rows.extend(rows)
                        policy_stats[(ident, role, name)] = stats
                    for draw_i, weights in enumerate(draws[ident[1]]):
                        boot = fit_cached(fit_caches[k], family, alpha, weights)
                        bootstrap_rows.append({**meta, "policy": name, "family": family, "alpha": alpha,
                                               "k": k, "draw": draw_i, **boot})
        # Raw FPR leave-one-group-out and confirm4 descriptive references.
        for alpha in ALPHAS:
            historical_threshold = threshold_from_history(historical_metrics, ident, alpha)
            reproduced = fit_cached(fit_caches[1], "trial_macro_static_FPR", alpha,
                                    constraint_override="frame_static_FPR")
            if reproduced["status"] != "fit" or reproduced["threshold"] != historical_threshold:
                raise ValueError(f"historical raw threshold mismatch {ident} alpha={alpha}: "
                                 f"{reproduced.get('threshold')} != {historical_threshold}")
            ref_name = f"historical_raw_FPR{alpha:.2f}|confirm4"
            for role in ROLES:
                stats, agg = evaluate_policy(datasets[ident][role], historical_threshold, 4)
                confirm4_metrics.append({**meta, "role": role, "policy": ref_name, "alpha": alpha, "k": 4,
                                         "threshold": historical_threshold, "selection": "historical_raw_threshold", **agg})
                rows = [{**meta, "role": role, "policy": ref_name, "alpha": alpha, "k": 4,
                         "threshold": historical_threshold, **s} for s in stats]
                confirm4_trials.extend(rows)
                policy_stats[(ident, role, ref_name)] = stats
            for removed in cal_groups:
                weights = {g: int(g != removed) for g in cal_groups}
                # Historical raw calibration recalculation uses frame FPR and k=1.
                fitted = fit_cached(fit_caches[1], "trial_macro_static_FPR", alpha, weights,
                                    constraint_override="frame_static_FPR")
                loo_rows.append({**meta, "alpha": alpha, "removed_group": removed,
                                 "full_threshold": historical_threshold,
                                 "loo_threshold": fitted["threshold"],
                                 "threshold_difference": fitted["threshold"] - historical_threshold if fitted["status"] == "fit" else None,
                                 "never_alarm": fitted["never_alarm"], "status": fitted["status"],
                                 "static_support_trials": fitted["static_support_trials"],
                                 "gross_frames": fitted["gross_support_frames"]})
        print(f"evaluated {run_i}/48 {ident}", flush=True)

    write_csv(root / "new_policies" / "thresholds.csv", thresholds_rows)
    write_csv(root / "new_policies" / "metrics.csv", metrics_rows)
    write_csv(root / "new_policies" / "trials.csv", trial_rows)
    write_csv(root / "new_policies" / "calibration_bootstrap.csv", bootstrap_rows)
    write_csv(root / "new_policies" / "confirm4_metrics.csv", confirm4_metrics)
    write_csv(root / "new_policies" / "confirm4_trials.csv", confirm4_trials)
    write_csv(root / "diagnostics" / "trial_static_tails.csv", tail_rows)
    write_csv(root / "diagnostics" / "raw_frame_fpr_leave_one_group_out.csv", loo_rows)

    # Compact uncertainty summary.
    grouped_boot = defaultdict(list)
    for r in bootstrap_rows:
        grouped_boot[(r["group"], r["fold"], r["seed"], r["policy"])].append(r)
    uncertainty = []
    for key, rows in sorted(grouped_boot.items()):
        valid = [r for r in rows if r["status"] == "fit"]
        vals = [float(r["threshold"]) for r in valid]
        uncertainty.append({"group": key[0], "fold": key[1], "seed": key[2], "policy": key[3],
                            "valid": len(valid), "requested": 200,
                            "threshold_q025": float(np.quantile(vals, .025, method="linear")) if vals else None,
                            "threshold_q50": float(np.quantile(vals, .5, method="linear")) if vals else None,
                            "threshold_q975": float(np.quantile(vals, .975, method="linear")) if vals else None,
                            "never_alarm_fraction": div(sum(float(r["never_alarm"]) for r in valid), len(valid))})
    write_csv(root / "new_policies" / "calibration_bootstrap_summary.csv", uncertainty)

    # Validation group bootstrap: fixed main calibration thresholds, shared draws across seeds/policies.
    validation_draws = {}
    for fi, fold in enumerate(FOLDS):
        names = sorted({e.group for e in datasets[(GROUPS[0], fold, SEEDS[0])]["validation"]})
        rng = np.random.default_rng(20260916 + fi)
        validation_draws[fold] = [dict(zip(names, np.bincount(rng.integers(0, len(names), len(names)), minlength=len(names)))) for _ in range(200)]
    pair_specs = []
    for family in FAMILIES:
        for alpha in ALPHAS:
            for k in KS:
                name = policy_name(family, alpha, k)
                for candidate, base in (("V_class_trial_balanced", "V_original"),
                                        ("F_class_trial_balanced", "F_history_original"),
                                        ("F_class_trial_balanced", "V_class_trial_balanced")):
                    pair_specs.append(("model", candidate, name, base, name))
                for group in GROUPS:
                    hist = f"historical_raw_FPR{alpha:.2f}|confirm4" if k == 4 else None
                    # k1/k2 historical trials are taken from the frozen R12 table below.
                    pair_specs.append(("rule_vs_historical", group, name, group, hist or f"HIST|{alpha:.2f}|k{k}"))
                if k in (2, 4):
                    for group in GROUPS:
                        pair_specs.append(("confirm_depth", group, name, group, policy_name(family, alpha, 1)))
    hist_trial_index = defaultdict(list)
    for r in historical_trials:
        if r["role"] != "validation" or not r["point"].startswith("FPR"):
            continue
        ident = (r["group"], r["fold"], int(r["seed"]))
        k = 1 if r["rule"] == "raw" else 2
        hist_trial_index[(ident, f"HIST|{float(r['point'][3:]):.2f}|k{k}")].append({
            "episode": r["episode"], "leakage_group": r["leakage_group"], "static_frames": int(r["tn"])+int(r["fp"]),
            "static_fp": int(r["fp"]), "static_any_alarm": int(int(r["fp"]) > 0),
            "gross_frames": int(r["tp"])+int(r["fn"]), "gross_tp": int(r["tp"]),
            "tn": int(r["tn"]), "fp": int(r["fp"]), "fn": int(r["fn"]), "tp": int(r["tp"]),
            "false_starts": int(r["false_starts"]), "alarm_starts": 0,
            "longest_static_alarm_segment": 0, "events": int(r["events"]), "event_hits": int(r["hits"]),
            "left_censored_events": int(r["left_censored"]), "right_boundary_events": int(r["right_boundary"]),
            "gross_segments": int(r["gross_segments"]), "segments_covered": int(r["segments_covered"]),
            "preexisting_alarm_events": 0, "delay_sum": int(r["delay_sum"]), "delay_n": int(r["delay_n"]),
            "preexisting_alarm_delay0_events": 0,
            "trial_count": 1, "incipient_frames": int(r["incipient_frames"]), "observed_alarm": int(r["observed_alarm"]),
        })
    paired_ci, paired_seed = [], []
    for fold in FOLDS:
        for kind, candidate, cand_policy, base, base_policy in pair_specs:
            reps = {metric: [] for metric in PAIR_METRICS}
            raw = {}
            for draw_i, weights in enumerate([None] + validation_draws[fold]):
                per_metric = {metric: [] for metric in PAIR_METRICS}
                for seed in SEEDS:
                    ci = (candidate, fold, seed); bi = (base, fold, seed)
                    cs = policy_stats[(ci, "validation", cand_policy)]
                    bs = hist_trial_index[(bi, base_policy)] if base_policy.startswith("HIST|") else policy_stats[(bi, "validation", base_policy)]
                    ca, ba = aggregate(cs, weights), aggregate(bs, weights)
                    for metric in PAIR_METRICS:
                        difference = ca[metric] - ba[metric]
                        per_metric[metric].append(difference)
                        if draw_i == 0:
                            paired_seed.append({"kind": kind, "fold": fold, "candidate_group": candidate,
                                                "candidate_policy": cand_policy, "base_group": base,
                                                "base_policy": base_policy, "seed": seed, "metric": metric,
                                                "difference": difference})
                for metric in PAIR_METRICS:
                    value = float(np.mean(per_metric[metric])) if all(np.isfinite(per_metric[metric])) else float("nan")
                    if draw_i == 0: raw[metric] = value
                    else: reps[metric].append(value)
            for metric in PAIR_METRICS:
                low, high, valid = bootstrap_ci(reps[metric])
                paired_ci.append({"kind": kind, "fold": fold, "candidate_group": candidate,
                                  "candidate_policy": cand_policy, "base_group": base, "base_policy": base_policy,
                                  "metric": metric, "difference": raw[metric], "ci_lower": low, "ci_upper": high,
                                  "valid_bootstraps": valid, "requested_bootstraps": 200,
                                  "unit": "complete_validation_leakage_group_shared_across_seeds"})
    write_csv(root / "paired" / "paired_ci.csv", paired_ci)
    write_csv(root / "paired" / "paired_seed_differences.csv", paired_seed)
    write_json(root / "paired" / "validation_group_draws.json", validation_draws)

    # Fixed case and train support.
    fixed = "htt/p3_sliding/0_press_13"
    fixed_rows = [r for r in trial_rows if r["fold"] == "htt_leave_p4" and r["role"] == "validation" and r["episode"] == fixed]
    fixed_rows += [r for r in confirm4_trials if r["fold"] == "htt_leave_p4" and r["role"] == "validation" and r["episode"] == fixed]
    fixed_rows += [r for r in historical_trials if r["fold"] == "htt_leave_p4" and r["role"] == "validation" and r["episode"] == fixed]
    write_csv(root / "fixed_case" / "all_rules.csv", fixed_rows)
    support = []
    for fold in FOLDS:
        episodes = datasets[(GROUPS[0], fold, SEEDS[0])]["train"]
        probes = Counter(e.probe for e in episodes)
        stages = Counter(int(x) for e in episodes for x in e.stage)
        static_lengths = []
        endpoint_minima = []
        for e in episodes:
            endpoint_minima.append(int(e.t.min()))
            cur = 0
            for i, st in enumerate(e.stage):
                if st == 0 and (i == 0 or e.t[i] == e.t[i-1] + 1): cur += 1
                elif st == 0: cur = 1
                else:
                    if cur: static_lengths.append(cur)
                    cur = 0
            if cur: static_lengths.append(cur)
        support.append({"fold": fold, "episodes": len(episodes), "leakage_groups": len({e.group for e in episodes}),
                        "probes": json.dumps(probes, sort_keys=True), "static_frames": stages[0],
                        "incipient_frames": stages[1], "gross_frames": stages[2],
                        "static_segments": len(static_lengths), "static_segment_min": min(static_lengths) if static_lengths else None,
                        "static_segment_median": float(np.median(static_lengths)) if static_lengths else None,
                        "static_segment_max": max(static_lengths) if static_lengths else None,
                        "episodes_without_static": sum(not np.any(e.stage == 0) for e in episodes),
                        "current_endpoint_min_t": min(endpoint_minima) if endpoint_minima else None,
                        "historical_window_endpoint_exclusions": "not_recomputed",
                        "exclusion_evidence": "protocol fixes common t>=13; exact early-frame count not asserted",
                        "force_change_without_slip_gt": "unavailable_no_predefined_physical_truth"})
    write_csv(root / "train_support" / "support_inventory.csv", support)

    # Validation contribution list under every formal policy.
    contributions = [r for r in trial_rows if r["role"] == "validation" and r["static_frames"] > 0]
    contributions.sort(key=lambda r: (r["group"], r["fold"], r["seed"], r["policy"], -r["static_fp"], r["episode"]))
    write_csv(root / "diagnostics" / "validation_trial_fp_contributions.csv", contributions)
    summary = {
        "schema": "round13_evaluation_summary_v1", "status": "complete",
        "completed_at": datetime.now(timezone.utc).astimezone().isoformat(), "new_neural_trainings": 0,
        "model_count": len(runs), "historical_workpoints_per_role": len(historical_metrics) // 3,
        "new_policy_workpoints_per_role": len(metrics_rows) // 3,
        "confirm4_reference_points_per_role": len(confirm4_metrics) // 3,
        "calibration_bootstrap_records": len(bootstrap_rows),
        "expected": {"models": 48, "historical_workpoints_per_role": 480,
                     "new_policy_workpoints_per_role": 864, "confirm4_reference_points_per_role": 144,
                     "calibration_bootstrap_records": 172800},
        "test_role_consumed": False, "validation_used_for_threshold_fitting": False,
        "input_hash_count": len(input_hashes), "input_hashes": input_hashes,
    }
    actual_workpoints = {
        "models": summary["model_count"],
        "historical_workpoints_per_role": summary["historical_workpoints_per_role"],
        "new_policy_workpoints_per_role": summary["new_policy_workpoints_per_role"],
        "confirm4_reference_points_per_role": summary["confirm4_reference_points_per_role"],
        "calibration_bootstrap_records": summary["calibration_bootstrap_records"],
    }
    if actual_workpoints != summary["expected"]:
        raise AssertionError(f"workpoint arithmetic failed: {summary}")
    write_json(root / "SUMMARY.json", summary)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--index", type=Path)
    p.add_argument("--r12-metrics", type=Path)
    p.add_argument("--r12-trials", type=Path)
    p.add_argument("--group-draws", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.smoke:
        print(json.dumps(smoke(), indent=2))
        return
    required = (args.index, args.r12_metrics, args.r12_trials, args.group_draws, args.output)
    if any(x is None for x in required):
        p.error("formal run requires --index --r12-metrics --r12-trials --group-draws --output")
    run(args)


if __name__ == "__main__":
    main()
