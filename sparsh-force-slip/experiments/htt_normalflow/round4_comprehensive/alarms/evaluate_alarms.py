#!/usr/bin/env python3
"""Exact calibration-only operating points and causal temporal alarms."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


BUDGETS = (0.01, 0.05, 0.10)
RULES = tuple((k, ratio) for k in (1, 2, 3) for ratio in (1.0, 0.8))
BOOTSTRAPS = 200


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def read_predictions(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8", newline="") as stream:
        for raw in csv.DictReader(stream):
            row = {"episode": raw.get("episode_id", raw.get("episode")),
                   "t": int(raw["t"]), "stage": int(raw["stage"]),
                   "score": float(raw.get("p_slip", raw.get("probability")))}
            if row["episode"] is None or row["stage"] not in (0, 1, 2) or not 0 <= row["score"] <= 1:
                raise ValueError(f"invalid prediction row in {path}: {raw}")
            rows.append(row)
    if not rows:
        raise ValueError(f"empty predictions: {path}")
    seen = set()
    for row in rows:
        key = (row["episode"], row["t"])
        if key in seen: raise ValueError(f"duplicate prediction key {key} in {path}")
        seen.add(key)
    rows.sort(key=lambda x: (x["episode"], x["t"]))
    for episode in sorted({r["episode"] for r in rows}):
        indices = [r["t"] for r in rows if r["episode"] == episode]
        if indices != list(range(len(indices))):
            raise ValueError(f"non-contiguous episode {episode} in {path}")
    return rows


def confusion(stages: np.ndarray, alarms: np.ndarray) -> dict[str, Any]:
    mask = np.isin(stages, (0, 2)); y = stages[mask] == 2; pred = alarms[mask]
    tn = int(np.sum(~y & ~pred)); fp = int(np.sum(~y & pred))
    fn = int(np.sum(y & ~pred)); tp = int(np.sum(y & pred))
    tnr = tn / (tn + fp) if tn + fp else float("nan")
    tpr = tp / (tp + fn) if tp + fn else float("nan")
    f1_pos = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    f1_neg = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "confusion_matrix": [[tn, fp], [fn, tp]], "static_fpr": 1 - tnr,
            "gross_recall": tpr, "balanced_accuracy": (tnr + tpr) / 2,
            "macro_f1": (f1_pos + f1_neg) / 2, "positive_prevalence": float(np.mean(y))}


def average_precision(stages: np.ndarray, scores: np.ndarray) -> float:
    mask = np.isin(stages, (0, 2)); y = (stages[mask] == 2).astype(np.int64); s = scores[mask]
    positives = int(y.sum())
    if positives == 0: return float("nan")
    order = np.argsort(-s, kind="stable"); y, s = y[order], s[order]
    ends = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp = np.cumsum(y)[ends]; fp = (ends + 1) - tp
    recall = tp / positives; precision = tp / (tp + fp)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def raw_metrics(rows: Sequence[dict], threshold: float) -> dict[str, Any]:
    stage = np.asarray([r["stage"] for r in rows]); score = np.asarray([r["score"] for r in rows])
    result = confusion(stage, score >= threshold)
    alarms = score >= threshold
    result.update({"threshold": float(threshold), "never_alarm": bool(threshold > 1.0),
                   "observed_no_alarm": bool(not np.any(alarms)),
                   "observed_no_primary_alarm": bool(not np.any(alarms[np.isin(stage, (0, 2))])),
                   "average_precision": average_precision(stage, score),
                   "incipient_score": distribution(score[stage == 1])})
    return result


def distribution(values: np.ndarray) -> dict[str, Any]:
    if not len(values): return {"count": 0, "mean": None, "median": None, "q10": None, "q90": None}
    return {"count": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)),
            "q10": float(np.quantile(values, .1)), "q90": float(np.quantile(values, .9))}


def exact_thresholds(rows: Sequence[dict]) -> np.ndarray:
    scores = np.asarray([r["score"] for r in rows], np.float64)
    return np.r_[np.unique(scores), np.nextafter(1.0, math.inf)]


def threshold_candidates(rows: Sequence[dict]) -> list[dict[str, Any]]:
    """All exact-score operating points in O(N log N), including sentinel."""
    stage = np.asarray([r["stage"] for r in rows]); score = np.asarray([r["score"] for r in rows])
    mask = np.isin(stage, (0, 2)); y = stage[mask] == 2; primary_scores = score[mask]
    positives = int(y.sum()); negatives = int((~y).sum())
    by_score: dict[float, list[int]] = defaultdict(lambda: [0, 0])
    for value, positive in zip(primary_scores.tolist(), y.tolist()):
        by_score[value][int(positive)] += 1
    tp = fp = 0
    candidates = [{"threshold": float(np.nextafter(1.0, math.inf)), "tn": negatives, "fp": 0,
                   "fn": positives, "tp": 0, "static_fpr": 0.0, "gross_recall": 0.0,
                   "balanced_accuracy": 0.5}]
    for threshold in sorted(set(score.tolist()), reverse=True):
        negative_at, positive_at = by_score.get(threshold, (0, 0))
        fp += negative_at; tp += positive_at
        fpr = fp / negatives; recall = tp / positives
        candidates.append({"threshold": threshold, "tn": negatives - fp, "fp": fp,
                           "fn": positives - tp, "tp": tp, "static_fpr": fpr,
                           "gross_recall": recall, "balanced_accuracy": ((1 - fpr) + recall) / 2})
    return candidates


def choose_max_ba(rows: Sequence[dict]) -> dict[str, Any]:
    candidates = threshold_candidates(rows)
    return max(candidates, key=lambda x: (x["balanced_accuracy"], -x["static_fpr"], x["threshold"]))


def choose_budget(rows: Sequence[dict], budget: float) -> dict[str, Any]:
    candidates = threshold_candidates(rows)
    feasible = [x for x in candidates if x["static_fpr"] <= budget + 1e-15]
    return max(feasible, key=lambda x: (x["gross_recall"], -x["static_fpr"], x["threshold"]))


def split_episodes(rows: Sequence[dict]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = defaultdict(list)
    for row in rows: result[row["episode"]].append(row)
    return dict(result)


def alarm_sequence(rows: Sequence[dict], threshold: float, k: int, ratio: float) -> tuple[np.ndarray, np.ndarray]:
    alarms = np.zeros(len(rows), dtype=bool); starts = np.zeros(len(rows), dtype=bool)
    active = False; consecutive = 0
    for i, row in enumerate(rows):
        score = row["score"]
        if active:
            if score < ratio * threshold:
                active = False; consecutive = 0
        else:
            consecutive = consecutive + 1 if score >= threshold else 0
            if consecutive >= k:
                active = True; starts[i] = True
        alarms[i] = active
    return alarms, starts


def contiguous_gross_events(stages: np.ndarray) -> list[tuple[int, int]]:
    gross = stages == 2; padded = np.r_[False, gross, False]
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return list(zip(starts.tolist(), ends.tolist()))


def sequential_metrics(rows: Sequence[dict], threshold: float, k: int, ratio: float,
                       leakage_groups: Mapping[str, str] | None = None) -> tuple[dict, list[dict]]:
    trial_rows = []
    totals = {x: 0 for x in ("tn", "fp", "fn", "tp", "static_frames", "gross_frames", "false_starts", "events", "events_detected")}
    delays, frame_zero = [], 0
    all_stage, all_alarm, all_score = [], [], []
    for episode, episode_rows in split_episodes(rows).items():
        stage = np.asarray([r["stage"] for r in episode_rows]); score = np.asarray([r["score"] for r in episode_rows])
        alarm, starts = alarm_sequence(episode_rows, threshold, k, ratio)
        c = confusion(stage, alarm); events = contiguous_gross_events(stage)
        event_delays = []
        for start, end in events:
            frame_zero += int(start == 0)
            hits = np.flatnonzero(alarm[start:end])
            if len(hits): event_delays.append(int(hits[0]))
        row = {"episode": episode, "bootstrap_group": leakage_groups.get(episode, episode) if leakage_groups else episode,
               "frames": len(stage), "static_frames": int(np.sum(stage == 0)),
               "gross_frames": int(np.sum(stage == 2)), "incipient_frames": int(np.sum(stage == 1)),
               "tn": c["tn"], "fp": c["fp"], "fn": c["fn"], "tp": c["tp"],
               "false_alarm_starts": int(np.sum(starts & (stage == 0))),
               "static_frames_alarming": int(np.sum(alarm & (stage == 0))),
               "gross_events": len(events), "gross_events_detected": len(event_delays),
               "mean_detection_delay_frames": float(np.mean(event_delays)) if event_delays else None}
        trial_rows.append(row); delays.extend(event_delays)
        for key in ("tn", "fp", "fn", "tp"): totals[key] += row[key]
        totals["static_frames"] += row["static_frames"]; totals["gross_frames"] += row["gross_frames"]
        totals["false_starts"] += row["false_alarm_starts"]
        totals["events"] += row["gross_events"]; totals["events_detected"] += row["gross_events_detected"]
        all_stage.extend(stage.tolist()); all_alarm.extend(alarm.tolist()); all_score.extend(score.tolist())
    result = confusion(np.asarray(all_stage), np.asarray(all_alarm))
    result.update({"threshold": float(threshold), "confirmation_k": k, "release_ratio": ratio,
                   "never_alarm": bool(threshold > 1.0), "average_precision": average_precision(np.asarray(all_stage), np.asarray(all_score)),
                   "observed_no_alarm": bool(not np.any(all_alarm)),
                   "observed_no_primary_alarm": bool(not np.any(np.asarray(all_alarm)[np.isin(np.asarray(all_stage), (0, 2))])),
                   "false_alarm_starts": totals["false_starts"],
                   "false_alarm_starts_per_trial": totals["false_starts"] / len(trial_rows),
                   "gross_events": totals["events"], "gross_events_detected": totals["events_detected"],
                   "gross_event_recall": totals["events_detected"] / totals["events"] if totals["events"] else float("nan"),
                   "mean_detection_delay_frames": float(np.mean(delays)) if delays else None,
                   "median_detection_delay_frames": float(np.median(delays)) if delays else None,
                   "segments_beginning_at_frame_zero": frame_zero,
                   "incipient_score": distribution(np.asarray(all_score)[np.asarray(all_stage) == 1])})
    return result, trial_rows


def choose_rule(rows: Sequence[dict], threshold: float, budget: float,
                leakage_groups: Mapping[str, str] | None = None) -> tuple[dict, list[dict]]:
    candidates = []
    for k, ratio in RULES:
        metrics, trials = sequential_metrics(rows, threshold, k, ratio, leakage_groups)
        if metrics["static_fpr"] <= budget + 1e-15: candidates.append((metrics, trials))
    if not candidates:
        sentinel = np.nextafter(1.0, math.inf)
        metrics, trials = sequential_metrics(rows, sentinel, 1, 1.0, leakage_groups)
        metrics["fallback_reason"] = "no fixed-threshold rule met calibration FPR; explicit never-alarm sentinel"
        return metrics, trials
    return max(candidates, key=lambda x: (x[0]["gross_recall"], -x[0]["false_alarm_starts"],
                                          -x[0]["static_fpr"], -x[0]["confirmation_k"], x[0]["release_ratio"]))


def bootstrap_trials(trials: Sequence[dict], seed: int, repetitions: int = BOOTSTRAPS) -> dict[str, Any]:
    rng = np.random.default_rng(seed); groups = sorted({x["bootstrap_group"] for x in trials}); n = len(groups); draws = defaultdict(list)
    for _ in range(repetitions):
        sampled = rng.choice(groups, n, replace=True)
        selected = [row for group in sampled for row in trials if row["bootstrap_group"] == group]
        tn = sum(x["tn"] for x in selected); fp = sum(x["fp"] for x in selected)
        fn = sum(x["fn"] for x in selected); tp = sum(x["tp"] for x in selected)
        fpr = fp / (tn + fp) if tn + fp else float("nan"); recall = tp / (tp + fn) if tp + fn else float("nan")
        events = sum(x["gross_events"] for x in selected); detected = sum(x["gross_events_detected"] for x in selected)
        draws["static_fpr"].append(fpr); draws["gross_recall"].append(recall)
        draws["balanced_accuracy"].append(((1 - fpr) + recall) / 2)
        draws["gross_event_recall"].append(detected / events if events else float("nan"))
        draws["false_alarm_starts_per_trial"].append(sum(x["false_alarm_starts"] for x in selected) / len(selected))
    result = {key: np.nanquantile(value, [.025, .975]).tolist() for key, value in draws.items()}
    result["bootstrap_unit"] = "complete schema-2 leakage_group"
    result["groups"] = n
    return result


def evaluate_run(model: str, fold: str, seed: int, run_dir: Path, output: Path,
                 leakage_groups: Mapping[str, str], split_manifest_sha256: str) -> dict[str, Any]:
    calibration_path = run_dir / "predictions/calibration.csv"; validation_path = run_dir / "predictions/validation.csv"
    calibration = read_predictions(calibration_path); validation = read_predictions(validation_path)
    if set(split_episodes(calibration)) & set(split_episodes(validation)):
        raise ValueError(f"calibration/validation episode leakage: {run_dir}")
    calibration_groups = {leakage_groups[x] for x in split_episodes(calibration)}
    validation_groups = {leakage_groups[x] for x in split_episodes(validation)}
    if calibration_groups & validation_groups:
        raise ValueError(f"calibration/validation leakage-group overlap: {run_dir}")
    ops = {"fixed_0.5": {"threshold": 0.5}, "max_ba": choose_max_ba(calibration)}
    for budget in BUDGETS: ops[f"fpr_{budget:.2f}"] = choose_budget(calibration, budget)
    selected_episodes = set(split_episodes(calibration)) | set(split_episodes(validation))
    if not selected_episodes.issubset(leakage_groups): raise ValueError(f"episodes absent from schema-2 manifest: {selected_episodes-set(leakage_groups)}")
    group_members = defaultdict(set)
    for episode in selected_episodes: group_members[leakage_groups[episode]].add(episode)
    result = {"status": "complete", "model": model, "fold": fold, "seed": seed, "source": str(run_dir),
              "provenance": {"evaluator_sha256": sha256(Path(__file__).resolve()),
                             "master_protocol_sha256": sha256(Path(__file__).parents[1] / "PROTOCOL.md"),
                             "split_manifest_sha256": split_manifest_sha256,
                             "training_summary_sha256": sha256(run_dir / "training_summary.json") if (run_dir / "training_summary.json").is_file() else None,
                             "calibration_predictions_sha256": sha256(calibration_path),
                             "validation_predictions_sha256": sha256(validation_path)},
              "bootstrap_groups": len(group_members),
              "nonunique_leakage_groups": {key: sorted(value) for key, value in group_members.items() if len(value) > 1},
              "calibration_episodes": len(split_episodes(calibration)), "validation_episodes": len(split_episodes(validation)),
              "operating_points": {}}
    csv_rows = []
    for name, selected in ops.items():
        threshold = selected["threshold"]
        cal_raw, cal_trials = sequential_metrics(calibration, threshold, 1, 1.0, leakage_groups)
        val_raw, val_trials = sequential_metrics(validation, threshold, 1, 1.0, leakage_groups)
        item = {"selection": "fixed" if name == "fixed_0.5" else "calibration_only",
                "raw": {"calibration": cal_raw, "validation": val_raw,
                        "validation_trial_bootstrap_95ci": bootstrap_trials(val_trials, seed + sum(map(ord, name)))}}
        for role, trials in (("calibration", cal_trials), ("validation", val_trials)):
            csv_rows.extend(dict(model=model, fold=fold, seed=seed, operating_point=name, mode="raw", role=role, **x) for x in trials)
        if name.startswith("fpr_"):
            budget = float(name.split("_", 1)[1]); chosen, chosen_trials = choose_rule(calibration, threshold, budget, leakage_groups)
            val_seq, val_seq_trials = sequential_metrics(validation, chosen["threshold"], chosen["confirmation_k"], chosen["release_ratio"], leakage_groups)
            item["sequential"] = {"budget": budget, "calibration": chosen, "validation": val_seq,
                                  "validation_trial_bootstrap_95ci": bootstrap_trials(val_seq_trials, seed + 1000 + int(budget * 100))}
            csv_rows.extend(dict(model=model, fold=fold, seed=seed, operating_point=name, mode="sequential", role="calibration", **x) for x in chosen_trials)
            csv_rows.extend(dict(model=model, fold=fold, seed=seed, operating_point=name, mode="sequential", role="validation", **x) for x in val_seq_trials)
        result["operating_points"][name] = item
    target = output / "runs" / model / fold / f"seed_{seed}"
    atomic_json(target / "metrics.json", result)
    target.mkdir(parents=True, exist_ok=True)
    fields = list(csv_rows[0])
    temp = target / f"trials.csv.tmp.{os.getpid()}"
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(csv_rows)
    os.replace(temp, target / "trials.csv")
    return result


def identify_encoder_run(run_dir: Path) -> tuple[str, str, int] | None:
    text = "/".join(run_dir.parts)
    model = next((name for name in ("dino", "ijepa", "mae_letterbox") if f"/{name}/" in f"/{text}/"), None)
    fold_match = re.search(r"fold_p([1-4])", text); seed_match = re.search(r"seed_?(2026091[456])", text)
    if model and fold_match and seed_match:
        return model, f"htt_leave_p{fold_match.group(1)}", int(seed_match.group(1))
    # Fall back to columns, allowing a scheduler-specific directory layout.
    path = run_dir / "predictions/calibration.csv"
    if model and path.is_file():
        with path.open(encoding="utf-8", newline="") as stream: first = next(csv.DictReader(stream))
        return model, first["fold"], int(first["seed"])
    return None


def discover(args) -> list[tuple[str, str, int, Path]]:
    found = []
    for cal in sorted(args.r3_root.glob("runs/B/fold_p*/seed_*/predictions/calibration.csv")):
        run = cal.parents[1]; first = read_predictions(cal)[0]
        fold_number = re.search(r"fold_p([1-4])", str(run)).group(1)
        seed_number = int(re.search(r"seed_(\d+)", str(run)).group(1))
        found.append(("mae", f"htt_leave_p{fold_number}", seed_number, run))
    if args.encoder_root:
        for cal in sorted(args.encoder_root.glob("runs/*/*/*/predictions/calibration.csv")):
            run = cal.parents[1]; identity = identify_encoder_run(run)
            summary_path = run / "training_summary.json"
            if not identity or not summary_path.is_file(): continue
            training = json.loads(summary_path.read_text())
            if training.get("status") != "complete" or training.get("config", {}).get("smoke"):
                continue
            found.append((*identity, run))
    unique = {}
    for item in found: unique[item[:3]] = item
    return [unique[key] for key in sorted(unique)]


def load_leakage_groups(path: Path) -> dict[str, str]:
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 2: raise ValueError("alarms require schema-2 split manifest")
    mapping = {row["id"]: row["leakage_group"] for row in manifest["episodes"] if row.get("domain") == "htt"}
    if len(mapping) != len(set(mapping)): raise ValueError("duplicate episode ids in split manifest")
    return mapping


def paired_summary(results: Sequence[dict]) -> list[dict]:
    indexed = {(r["model"], r["fold"], r["seed"]): r for r in results}; rows = []
    for (model, fold, seed), result in indexed.items():
        if model == "mae" or ("mae", fold, seed) not in indexed: continue
        base = indexed[("mae", fold, seed)]
        for op in result["operating_points"]:
            for mode in ("raw", "sequential"):
                if mode not in result["operating_points"][op] or mode not in base["operating_points"][op]: continue
                current = result["operating_points"][op][mode]["validation"]
                reference = base["operating_points"][op][mode]["validation"]
                rows.append({"model": model, "fold": fold, "seed": seed, "operating_point": op, "mode": mode,
                             "delta_balanced_accuracy_vs_mae": current["balanced_accuracy"] - reference["balanced_accuracy"],
                             "delta_static_fpr_vs_mae": current["static_fpr"] - reference["static_fpr"],
                             "delta_gross_recall_vs_mae": current["gross_recall"] - reference["gross_recall"]})
    return rows


def render_summary(results: Sequence[dict], output: Path) -> None:
    paired = paired_summary(results)
    expected = {(model, f"htt_leave_p{fold}", seed) for model in ("mae", "dino", "ijepa", "mae_letterbox")
                for fold in range(1, 5) for seed in (20260914, 20260915, 20260916)}
    actual = {(r["model"], r["fold"], r["seed"]) for r in results}
    status = "complete" if actual == expected else "partial"
    descriptive = []
    run_rows = []
    for result in results:
        for op, value in result["operating_points"].items():
            for mode in ("raw", "sequential"):
                if mode not in value: continue
                metrics = value[mode]["validation"]
                run_rows.append({"model": result["model"], "fold": result["fold"], "seed": result["seed"],
                                 "operating_point": op, "mode": mode,
                                 **{key: metrics[key] for key in ("balanced_accuracy", "macro_f1", "average_precision",
                                                                  "positive_prevalence", "static_fpr", "gross_recall",
                                                                  "gross_event_recall", "false_alarm_starts_per_trial",
                                                                  "tn", "fp", "fn", "tp")}})
    for key in sorted({(r["model"], r["operating_point"], r["mode"]) for r in run_rows}):
        subset = [r for r in run_rows if (r["model"], r["operating_point"], r["mode"]) == key]
        row = {"model": key[0], "operating_point": key[1], "mode": key[2], "runs": len(subset)}
        for metric in ("balanced_accuracy", "macro_f1", "average_precision", "positive_prevalence", "static_fpr",
                       "gross_recall", "gross_event_recall", "false_alarm_starts_per_trial"):
            values = np.asarray([r[metric] for r in subset], float)
            row[f"{metric}_mean"] = float(np.nanmean(values)); row[f"{metric}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
        descriptive.append(row)
    summary = {"status": status, "runs_evaluated": len(results),
               "models": {m: sum(r["model"] == m for r in results) for m in sorted({r["model"] for r in results})},
               "expected_identities_present": actual == expected,
               "missing_identities": [list(x) for x in sorted(expected - actual)],
               "unexpected_identities": [list(x) for x in sorted(actual - expected)],
               "descriptive_mean_std_across_fold_seed_runs": descriptive,
               "paired_differences": paired, "protocol": {"budgets": BUDGETS, "rules": RULES, "bootstrap_replicates": BOOTSTRAPS}}
    atomic_json(output / "summary.json", summary)
    lines = ["# Round 4 低误报告警评估", "", f"已评估 {len(results)} 个模型运行；状态：`{status}`。",
             "阈值及连续告警规则只在 calibration 上确定，然后原样应用到 validation。", "",
             "结果为四折重叠开发评估，不是独立盲测；提前或延迟仅以帧报告。", "",
             "## 覆盖", ""]
    for model, count in summary["models"].items(): lines.append(f"- {model}: {count} runs")
    lines += ["", "## Validation 描述统计", "", "| 模型 | 工作点 | 模式 | BA | macro-F1 | AP | 正类占比 | FPR | gross召回 | 事件召回 | 误报启动/试次 |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in descriptive:
        lines.append(f"| {row['model']} | {row['operating_point']} | {row['mode']} | {row['balanced_accuracy_mean']:.4f}±{row['balanced_accuracy_std']:.4f} | {row['macro_f1_mean']:.4f}±{row['macro_f1_std']:.4f} | {row['average_precision_mean']:.4f}±{row['average_precision_std']:.4f} | {row['positive_prevalence_mean']:.4f}±{row['positive_prevalence_std']:.4f} | {row['static_fpr_mean']:.4f}±{row['static_fpr_std']:.4f} | {row['gross_recall_mean']:.4f}±{row['gross_recall_std']:.4f} | {row['gross_event_recall_mean']:.4f}±{row['gross_event_recall_std']:.4f} | {row['false_alarm_starts_per_trial_mean']:.4f}±{row['false_alarm_starts_per_trial_std']:.4f} |")
    lines += ["", "均值和标准差仅描述重叠的折与三个种子，不将其当作独立重复。每个运行的 `metrics.json` 包含固定 0.5、最大 BA、FPR 1%/5%/10% 工作点、实际 validation 指标、事件检测和 200 次完整 leakage-group bootstrap。`trials.csv` 保留逐试次计数。", ""]
    atomic_text(output / "SUMMARY_ZH.md", "\n".join(lines))
    if run_rows:
        target = output / "run_metrics.csv"; temp = target.with_name(target.name + f".tmp.{os.getpid()}")
        with temp.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(run_rows[0])); writer.writeheader(); writer.writerows(run_rows)
        os.replace(temp, target)
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        for model in summary["models"]:
            xs, ys = [], []
            for r in results:
                if r["model"] != model: continue
                for op in ("fpr_0.01", "fpr_0.05", "fpr_0.10"):
                    m = r["operating_points"][op]["sequential"]["validation"]; xs.append(m["static_fpr"]); ys.append(m["gross_recall"])
            ax.scatter(xs, ys, s=18, alpha=.7, label=model)
        ax.set(xlabel="Validation static FPR", ylabel="Validation gross recall", title="Calibration-selected sequential alarms")
        ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(output / "validation_operating_points.png", dpi=160); plt.close(fig)
    except ImportError:
        atomic_text(output / "plot_unavailable.txt", "matplotlib is unavailable; metrics remain in summary.json\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r3-root", type=Path, required=True)
    parser.add_argument("--encoder-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path,
                        default=Path("/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round1/splits.json"))
    parser.add_argument("--require-complete", action="store_true", help="require 12 runs each for mae,dino,ijepa,mae_letterbox")
    args = parser.parse_args(); runs = discover(args); leakage_groups = load_leakage_groups(args.split_manifest)
    if args.require_complete:
        expected = {(model, f"htt_leave_p{fold}", seed) for model in ("mae", "dino", "ijepa", "mae_letterbox")
                    for fold in range(1, 5) for seed in (20260914, 20260915, 20260916)}
        actual = {item[:3] for item in runs}
        if actual != expected: raise SystemExit(f"incomplete prediction identities: missing={sorted(expected-actual)}, unexpected={sorted(actual-expected)}")
    split_sha = sha256(args.split_manifest)
    results = [evaluate_run(*item, args.output, leakage_groups, split_sha) for item in runs]
    render_summary(results, args.output)
    print(json.dumps({"runs": len(results), "output": str(args.output)}, indent=2))


if __name__ == "__main__": main()
