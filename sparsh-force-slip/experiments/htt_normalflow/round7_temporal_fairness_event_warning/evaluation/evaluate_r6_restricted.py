#!/usr/bin/env python3
"""Evaluate accepted Round-6 H1 predictions as an explicitly restricted diagnostic."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "protocol.json"
AMENDMENTS = tuple(HERE / f"AMENDMENT_0{i}.json" for i in (1, 2, 3))
GROUPS = ("A_visual", "B_force", "C_force_delta")
SEEDS = (20260914, 20260915, 20260916)
RULES = ("raw", "ema_0p5", "confirm2", "ema_0p5_confirm2")
FRAME_FPRS = (0.01, 0.05, 0.10)
TRIAL_FAS = (0.05, 0.10, 0.20)
EVENT_RECALLS = (0.50, 0.70, 0.90)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV {path}")
    keys = list(rows[0])
    if any(list(row) != keys for row in rows):
        raise ValueError(f"inconsistent CSV schema {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int8); score = np.asarray(score, dtype=float)
    positives = int(y.sum())
    if positives == 0:
        return math.nan
    order = np.argsort(-score, kind="mergesort")
    ys, ss = y[order], score[order]
    ends = np.r_[np.flatnonzero(ss[:-1] != ss[1:]), len(ss) - 1]
    tp = np.cumsum(ys)[ends]; precision = tp / (ends + 1); recall = tp / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def apply_rule(values: np.ndarray, rule: str) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("invalid rule input")
    if rule == "raw":
        return p.copy()
    ema = np.empty_like(p); ema[0] = p[0]
    for i in range(1, len(p)):
        ema[i] = 0.5 * p[i] + 0.5 * ema[i - 1]
    base = ema if rule == "ema_0p5_confirm2" else p
    if rule in ("confirm2", "ema_0p5_confirm2"):
        out = np.zeros_like(base); out[1:] = np.minimum(base[:-1], base[1:]); return out
    if rule == "ema_0p5":
        return ema
    raise ValueError(rule)


def add_rule_scores(rows: list[dict]) -> None:
    by_episode: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_episode[row["episode_id"]].append(row)
    for episode, seq in by_episode.items():
        seq.sort(key=lambda r: r["t"])
        times = [r["t"] for r in seq]
        if len(times) != len(set(times)) or times != list(range(times[0], times[-1] + 1)):
            raise ValueError(f"Round-6 restricted sequence is not contiguous: {episode}")
        p = np.asarray([r["p_raw"] for r in seq])
        for rule in RULES:
            for row, q in zip(seq, apply_rule(p, rule)):
                row[rule] = float(q)


def confusion(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y, dtype=bool); alarm = np.asarray(score) >= threshold
    tp = int(np.sum(alarm & y)); fp = int(np.sum(alarm & ~y)); fn = int(np.sum(~alarm & y)); tn = int(np.sum(~alarm & ~y))
    recall = tp / (tp + fn) if tp + fn else math.nan
    fpr = fp / (fp + tn) if fp + tn else math.nan
    specificity = tn / (tn + fp) if tn + fp else math.nan
    f1p = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    f1n = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "frame_fpr": fpr, "frame_recall": recall,
            "balanced_accuracy": 0.5 * (recall + specificity), "macro_f1": 0.5 * (f1p + f1n),
            "never_alarm": bool(not alarm.any())}


def threshold_curve(y: np.ndarray, score: np.ndarray) -> list[dict]:
    y = np.asarray(y, dtype=np.int8); score = np.asarray(score, dtype=float)
    if set(y.tolist()) != {0, 1}:
        raise ValueError("threshold population lacks both classes")
    order = np.argsort(-score, kind="mergesort"); ys, ss = y[order], score[order]
    ends = np.r_[np.flatnonzero(ss[:-1] != ss[1:]), len(ss) - 1]
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    rows = [{"threshold": math.nextafter(1.0, math.inf), "tp": 0, "fp": 0, "fn": pos, "tn": neg,
             "frame_recall": 0.0, "frame_fpr": 0.0, "balanced_accuracy": 0.5}]
    tp = np.cumsum(ys)[ends]; fp = ends + 1 - tp
    for i, end in enumerate(ends):
        tpi, fpi = int(tp[i]), int(fp[i]); rec, fpr = tpi / pos, fpi / neg
        rows.append({"threshold": float(ss[end]), "tp": tpi, "fp": fpi, "fn": pos - tpi, "tn": neg - fpi,
                     "frame_recall": rec, "frame_fpr": fpr, "balanced_accuracy": 0.5 * (rec + 1 - fpr)})
    return rows


def alarm_runs(times: np.ndarray, mask: np.ndarray) -> tuple[int, int, int, float]:
    times = np.asarray(times, dtype=np.int64); mask = np.asarray(mask, dtype=bool)
    lengths: list[int] = []; current = 0; previous = None
    for t, active in zip(times, mask):
        if active:
            if previous is not None and t == previous + 1:
                current += 1
            else:
                if current: lengths.append(current)
                current = 1
            previous = int(t)
        else:
            if current: lengths.append(current)
            current = 0; previous = None
    if current: lengths.append(current)
    return len(lengths), int(sum(lengths)), int(max(lengths, default=0)), float(np.mean(lengths)) if lengths else 0.0


def trial_event_metrics(rows: list[dict], score_key: str, threshold: float) -> tuple[dict, list[dict]]:
    records = []
    for episode in sorted({r["episode_id"] for r in rows}):
        seq = sorted((r for r in rows if r["episode_id"] == episode), key=lambda r: r["t"])
        onset = seq[0]["onset"]; negative = [r for r in seq if r["target"] == 0]; positive = [r for r in seq if r["target"] == 1]
        neg_alarm = np.asarray([r[score_key] >= threshold for r in negative], dtype=bool)
        starts, duration, max_duration, mean_duration = alarm_runs(np.asarray([r["t"] for r in negative]), neg_alarm)
        hits = [r["t"] for r in positive if r[score_key] >= threshold]
        has_event = onset is not None and bool(positive)
        records.append({"episode_id": episode, "leakage_group": seq[0]["leakage_group"], "onset_t": onset,
                        "negative_frames": len(negative), "positive_frames": len(positive), "trial_false_alarm": bool(neg_alarm.any()),
                        "false_alarm_starts": starts, "false_alarm_duration": duration, "max_false_alarm_duration": max_duration,
                        "mean_false_alarm_duration": mean_duration, "uncensored_event": has_event,
                        "event_detected": bool(hits) if has_event else None,
                        "first_alarm_t": min(hits) if hits else None,
                        "lead_frames": int(onset - min(hits)) if hits else None})
    negative_trials = [r for r in records if r["negative_frames"]]
    events = [r for r in records if r["uncensored_event"]]
    hits = [r for r in events if r["event_detected"]]
    summary = {"trials": len(records), "negative_window_trials": len(negative_trials),
               "trial_false_alarm_rate": float(np.mean([r["trial_false_alarm"] for r in negative_trials])) if negative_trials else math.nan,
               "false_alarm_starts": int(sum(r["false_alarm_starts"] for r in records)),
               "false_alarm_duration": int(sum(r["false_alarm_duration"] for r in records)),
               "mean_run_duration": float(np.mean([r["mean_false_alarm_duration"] for r in records if r["false_alarm_starts"]])) if any(r["false_alarm_starts"] for r in records) else 0.0,
               "uncensored_events": len(events), "event_recall": float(np.mean([r["event_detected"] for r in events])) if events else math.nan,
               "mean_lead_frames": float(np.mean([r["lead_frames"] for r in hits])) if hits else math.nan,
               "median_lead_frames": float(np.median([r["lead_frames"] for r in hits])) if hits else math.nan,
               "misses": int(sum(not r["event_detected"] for r in events)), "never_alarm_trials": int(sum(not r["trial_false_alarm"] and not r.get("event_detected") for r in records))}
    return summary, records


def compact_false_alarm_stats(rows: list[dict], score_key: str, thresholds: list[float]) -> dict[float, tuple[int, int]]:
    """Vectorized starts/duration for a small set of threshold ties."""
    current, previous, boundary = [], [], []
    by_episode = defaultdict(list)
    for row in rows: by_episode[row["episode_id"]].append(row)
    for seq in by_episode.values():
        negative = sorted((r for r in seq if r["target"] == 0), key=lambda r: r["t"])
        for i, row in enumerate(negative):
            current.append(row[score_key])
            contiguous = i > 0 and row["t"] == negative[i - 1]["t"] + 1
            previous.append(negative[i - 1][score_key] if contiguous else -math.inf)
            boundary.append(not contiguous)
    cur = np.asarray(current); prev = np.asarray(previous); start_boundary = np.asarray(boundary)
    result = {}
    for threshold in thresholds:
        active = cur >= threshold
        starts = int(np.sum(active & (start_boundary | (prev < threshold))))
        result[threshold] = (starts, int(np.sum(active)))
    return result


def select_threshold(rows: list[dict], score_key: str, kind: str, target: float | None = None) -> tuple[float | None, str]:
    y = np.asarray([r["target"] for r in rows], dtype=np.int8); score = np.asarray([r[score_key] for r in rows])
    curve = threshold_curve(y, score)
    if kind == "fixed": return 0.5, "available"
    if kind == "maxBA":
        row = max(curve, key=lambda r: (r["balanced_accuracy"], -r["frame_fpr"], r["threshold"]))
        return row["threshold"], "available"
    if kind == "frame_fpr":
        valid = [r for r in curve if r["frame_fpr"] <= float(target) + 1e-15]
        row = max(valid, key=lambda r: (r["frame_recall"], -r["frame_fpr"], r["threshold"]))
        return row["threshold"], "available"
    by_episode = defaultdict(list)
    for row in rows: by_episode[row["episode_id"]].append(row)
    negative_max = np.asarray([max(r[score_key] for r in seq if r["target"] == 0)
                               for seq in by_episode.values() if any(r["target"] == 0 for r in seq)])
    event_max = np.asarray([max(r[score_key] for r in seq if r["target"] == 1)
                            for seq in by_episode.values() if any(r["target"] == 1 for r in seq)])
    if not len(negative_max) or not len(event_max): return None, "unavailable_calibration_event_or_negative_trial"
    primary = []
    for row in curve:
        threshold = row["threshold"]
        compact = {"trial_false_alarm_rate": float(np.mean(negative_max >= threshold)),
                   "event_recall": float(np.mean(event_max >= threshold))}
        if kind == "trial_fa" and compact["trial_false_alarm_rate"] <= float(target) + 1e-15:
            primary.append((threshold, compact))
        elif kind == "event_recall" and compact["event_recall"] >= float(target) - 1e-15:
            primary.append((threshold, compact))
    if not primary:
        return None, "unavailable_calibration_constraint"
    if kind == "trial_fa":
        best_recall = max(x[1]["event_recall"] for x in primary); primary = [x for x in primary if x[1]["event_recall"] == best_recall]
        best_fa = min(x[1]["trial_false_alarm_rate"] for x in primary); primary = [x for x in primary if x[1]["trial_false_alarm_rate"] == best_fa]
    else:
        best_fa = min(x[1]["trial_false_alarm_rate"] for x in primary); primary = [x for x in primary if x[1]["trial_false_alarm_rate"] == best_fa]
    compact = compact_false_alarm_stats(rows, score_key, [x[0] for x in primary])
    detailed = [(threshold, {**values, "false_alarm_starts": compact[threshold][0], "false_alarm_duration": compact[threshold][1], "mean_lead_frames": 1.0}) for threshold, values in primary]
    if kind == "trial_fa":
        threshold, _ = max(detailed, key=lambda x: (-x[1]["false_alarm_starts"], -x[1]["false_alarm_duration"], x[0]))
    else:
        threshold, _ = max(detailed, key=lambda x: (-x[1]["false_alarm_starts"], -x[1]["false_alarm_duration"], x[1]["mean_lead_frames"] if math.isfinite(x[1]["mean_lead_frames"]) else -math.inf, x[0]))
    return threshold, "available"


def load_predictions(path: Path, expected_role: str) -> list[dict]:
    records = list(csv.DictReader(path.open()))
    required = {"episode_id", "leakage_group", "t", "first_current_slip_t", "target_future_H1", "p_future_raw"}
    if not records or required - set(records[0]): raise ValueError(f"invalid predictions {path}")
    seen = set(); rows = []
    for r in records:
        key = (r["episode_id"], int(r["t"]))
        if key in seen: raise ValueError(f"duplicate prediction {key}")
        seen.add(key); onset_raw = r["first_current_slip_t"]
        rows.append({"episode_id": r["episode_id"], "leakage_group": r["leakage_group"], "t": int(r["t"]),
                     "onset": None if onset_raw in ("", "None") else int(onset_raw), "target": int(r["target_future_H1"]),
                     "p_raw": float(r["p_future_raw"]), "role": expected_role})
    add_rule_scores(rows); return rows


def operating_specs() -> list[tuple[str, str, float | None]]:
    out = [("fixed_0.5", "fixed", 0.5), ("maxBA", "maxBA", None)]
    out += [(f"frame_FPR_{x:.2f}", "frame_fpr", x) for x in FRAME_FPRS]
    out += [(f"trial_FA_{x:.2f}", "trial_fa", x) for x in TRIAL_FAS]
    out += [(f"event_recall_{x:.2f}", "event_recall", x) for x in EVENT_RECALLS]
    return out


def evaluate(args) -> dict:
    protocol = json.loads(PROTOCOL.read_text())
    if protocol.get("status") != "frozen_before_round7_outer_scores": raise ValueError("evaluation protocol not frozen")
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    metric_rows, trial_rows, inputs = [], [], {}
    for group in GROUPS:
        for seed in SEEDS:
            folder = args.r6_root / "formal" / f"future_{group}_{seed}"
            summary_path = folder / "summary.json"; summary = json.loads(summary_path.read_text())
            if summary.get("status") != "complete" or not summary.get("formal") or summary.get("smoke") or summary.get("group") != group or int(summary.get("seed")) != seed:
                raise ValueError(f"invalid Round-6 formal identity {group}/{seed}")
            inputs[str(summary_path.resolve())] = sha256(summary_path)
            role_rows = {}
            for role in ("calibration", "outer"):
                spec = summary["artifacts"]["predictions"][role]; path = Path(spec["path"])
                if path.resolve() != (folder / f"predictions_{role}.csv").resolve() or sha256(path) != spec["sha256"]:
                    raise ValueError(f"prediction provenance mismatch {group}/{seed}/{role}")
                inputs[str(path.resolve())] = sha256(path); role_rows[role] = load_predictions(path, role)
            cal, outer = role_rows["calibration"], role_rows["outer"]
            for rule in RULES:
                for op, kind, target in operating_specs():
                    threshold, status = select_threshold(cal, rule, kind, target)
                    base = {"group": group, "seed": seed, "horizon": 1, "rule": rule, "operating_point": op,
                            "calibration_status": status, "threshold": threshold, "scope": "restricted_R6_eligible_H1_diagnostic"}
                    if threshold is None:
                        metric_rows.append({**base, **{k: "" for k in ("n", "positive", "prevalence", "average_precision", "brier", "tn", "fp", "fn", "tp", "frame_fpr", "frame_recall", "balanced_accuracy", "macro_f1", "never_alarm", "negative_window_trials", "trial_false_alarm_rate", "false_alarm_starts", "false_alarm_duration", "mean_run_duration", "uncensored_events", "event_recall", "mean_lead_frames", "median_lead_frames", "misses", "never_alarm_trials")}})
                        continue
                    y = np.asarray([r["target"] for r in outer]); score = np.asarray([r[rule] for r in outer])
                    frame = confusion(y, score, threshold); event, records = trial_event_metrics(outer, rule, threshold)
                    metric_rows.append({**base, "n": len(y), "positive": int(y.sum()), "prevalence": float(y.mean()),
                                        "average_precision": average_precision(y, score), "brier": float(np.mean((score - y) ** 2)), **frame, **event})
                    for record in records:
                        trial_rows.append({"group": group, "seed": seed, "horizon": 1, "rule": rule, "operating_point": op,
                                           "threshold": threshold, "scope": "restricted_R6_eligible_H1_diagnostic", **record})
    atomic_csv(out / "metrics.csv", metric_rows); atomic_csv(out / "trial_metrics.csv", trial_rows)
    inputs[str(PROTOCOL.resolve())] = sha256(PROTOCOL)
    for p in AMENDMENTS: inputs[str(p.resolve())] = sha256(p)
    outputs = {str(p.resolve()): sha256(p) for p in (out / "metrics.csv", out / "trial_metrics.csv")}
    summary = {"format": "round7_r6_restricted_event_diagnostic_v1", "status": "complete", "formal_round7": False,
               "expected_round6_runs": 9, "evaluated_round6_runs": 9, "rules": list(RULES),
               "scope": "Round-6 H1 predictions contain calibration/outer eligible endpoints only and no selection or complete post-label timeline; results are restricted historical diagnostics and cannot choose the Round-7 rule or establish deployable smoothing.",
               "source_hashes": inputs, "output_hashes": outputs}
    atomic_json(out / "summary.json", summary)
    report = """# Round-6 restricted event-rule diagnostic\n\nThis diagnostic reuses nine accepted Round-6 H1 runs. Stored predictions are contiguous over the eligible pre-onset range, but omit the internal selection role and the complete causal output timeline. Four fixed causal rules are therefore reported separately without rule-family selection. Calibration thresholds are transferred unchanged to outer. These values are historical diagnostics only and are not a Round-7 deployable comparison.\n"""
    (out / "README.md").write_text(report)
    print(json.dumps({"status": "complete", "runs": 9, "metric_rows": len(metric_rows), "trial_rows": len(trial_rows)}, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--r6-root", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); evaluate(a)


if __name__ == "__main__": main()
