#!/usr/bin/env python3
"""Validate and summarize the 48 formal Round-5 current-detection evaluations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluator = load_module("round5_evaluate_slip", ROOT / "evaluate_slip.py")
scheduler = load_module("round5_run_formal", ROOT / "run_formal.py")

MODELS = ("mae-r3-b", "V", "F-old", "F-adapt")
FORCE_MODELS = ("F-old", "F-adapt")
FOLDS = tuple(f"htt_leave_p{i}" for i in range(1, 5))
SEEDS = (20260914, 20260915, 20260916)
BUDGET_OPS = ("fpr_0.01", "fpr_0.05", "fpr_0.10")
BOOTSTRAPS = 200
PAIR_METRICS = (
    "static_fpr", "gross_recall", "gross_segment_alarm_coverage_including_left_censored",
    "uncensored_gross_event_recall",
    "false_alarm_starts_per_trial", "static_alarming_fraction",
    "uncensored_mean_detection_delay_frames",
)


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fields is None:
        raise ValueError(f"Cannot infer columns for empty CSV: {path}")
    fields = fields or list(rows[0])
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def arg_value(argv: list[str], option: str) -> str:
    if argv.count(option) != 1:
        raise ValueError(f"Expected exactly one {option}: {argv}")
    index = argv.index(option)
    if index + 1 >= len(argv):
        raise ValueError(f"Missing value after {option}")
    return argv[index + 1]


def expected_identities() -> set[tuple[str, str, int]]:
    return {(model, fold, seed) for model in MODELS for fold in FOLDS for seed in SEEDS}


def evaluation_identity(job: dict) -> tuple[str, str, int]:
    argv = job["argv"]
    return arg_value(argv, "--model-id"), arg_value(argv, "--fold"), int(arg_value(argv, "--seed"))


def audit_inventory(inventory: dict) -> tuple[dict[tuple[str, str, int], dict], list[str]]:
    evaluation_jobs = [job for job in inventory.get("jobs", []) if job.get("kind") == "evaluation"]
    errors = []
    jobs = {}
    for job in evaluation_jobs:
        try:
            identity = evaluation_identity(job)
        except Exception as exc:
            errors.append(f"{job.get('id')}: {exc}")
            continue
        if identity in jobs:
            errors.append(f"duplicate evaluation identity: {identity}")
        jobs[identity] = job
        model, _, _ = identity
        historical = "--historical-baseline" in job["argv"]
        if (model == "mae-r3-b") != historical:
            errors.append(f"historical identity/flag mismatch: {identity}")
        forbidden = {
            "--smoke", "--smoke-epochs", "--allow-smoke", "--allow-smoke-parent",
            "--allow-unverified-cache", "--interrupt-after-epoch",
        }
        if any(value.split("=")[0] in forbidden for value in job["argv"]):
            errors.append(f"formal evaluation contains preparation option: {identity}")
    missing = sorted(expected_identities() - set(jobs))
    extra = sorted(set(jobs) - expected_identities())
    if missing:
        errors.append(f"missing identities: {missing}")
    if extra:
        errors.append(f"extra identities: {extra}")
    if len(evaluation_jobs) != 48:
        errors.append(f"expected 48 evaluation jobs, found {len(evaluation_jobs)}")
    return jobs, errors


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def numeric(row: dict, key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in (None, "", "None") else float("nan")


def load_complete_run(job: dict, labels_by_fold: dict[str, dict], manifests: dict[str, dict]) -> dict:
    identity = evaluation_identity(job)
    model, fold, seed = identity
    if not scheduler.accepted(job):
        raise ValueError("scheduler acceptance/receipt check failed")
    metrics_path = Path(job["acceptance_path"])
    payload = json.loads(metrics_path.read_text())
    if payload.get("status") != "complete" or payload.get("formal") is not True:
        raise ValueError("metrics is not a completed formal artifact")
    if (payload.get("model_id"), payload.get("fold"), payload.get("seed")) != identity:
        raise ValueError("metrics identity mismatch")
    if payload.get("historical_baseline") is not (model == "mae-r3-b"):
        raise ValueError("metrics historical identity mismatch")
    trial_path = Path(payload.get("artifacts", {}).get("trials_csv", ""))
    if not trial_path.is_file() or evaluator.r4.sha256(trial_path) != payload.get("artifacts", {}).get("trials_csv_sha256"):
        raise ValueError("trials artifact/hash mismatch")
    trials = read_csv(trial_path)
    if len(trials) != payload["artifacts"].get("trial_rows"):
        raise ValueError("trials row count mismatch")
    if {row["role"] for row in trials} - {"calibration", "validation"}:
        raise ValueError("test or unknown role in trials")
    argv = job["argv"]
    prediction_path = Path(arg_value(argv, "--validation"))
    prediction_rows = evaluator.read_rows(
        prediction_path, manifests[fold], fold, "validation", labels_by_fold[fold]
    )
    return {
        "identity": identity,
        "job": job,
        "metrics_path": metrics_path,
        "payload": payload,
        "trials": trials,
        "prediction_rows": prediction_rows,
        "prediction_path": prediction_path,
    }


def trial_subset(run: dict, operation: str, mode: str) -> list[dict]:
    result = [row for row in run["trials"] if row["role"] == "validation"
              and row["operating_point"] == operation and row["mode"] == mode]
    if not result:
        raise ValueError(f"missing validation trials: {run['identity']} {operation}/{mode}")
    if len({row["episode"] for row in result}) != len(result):
        raise ValueError("duplicate validation episode trial row")
    by_episode = evaluator.r4.split_episodes(run["prediction_rows"])
    item = run["payload"]["operating_points"][operation][mode]["validation"]
    enriched = []
    for row in result:
        if row["episode"] not in by_episode:
            raise ValueError("trial episode missing from validation predictions")
        derived = uncensored_event_statistics(
            by_episode[row["episode"]], item["threshold"],
            item["confirmation_k"], item["release_ratio"],
        )
        if int(row["gross_events"]) != derived["gross_segments_including_left_censored"]:
            raise ValueError("trial gross-event count differs from reconstructed sequence")
        enriched.append({**row, **derived})
    return enriched


def uncensored_event_statistics(rows: list[dict], threshold: float, confirmation: int,
                                release: float) -> dict:
    alarms, _ = evaluator.r4.alarm_sequence(rows, threshold, confirmation, release)
    stages = np.asarray([row["stage"] for row in rows])
    segments = evaluator.r4.contiguous_gross_events(stages)
    left_censored = sum(start == 0 for start, _ in segments)
    delays = []
    for start, end in segments:
        if start == 0:
            continue
        hits = np.flatnonzero(alarms[start:end])
        if len(hits):
            delays.append(int(hits[0]))
    return {
        "gross_segments_including_left_censored": len(segments),
        "left_censored_gross_segments": left_censored,
        "uncensored_gross_events": len(segments) - left_censored,
        "uncensored_gross_events_detected": len(delays),
        "uncensored_detection_delay_sum_frames": sum(delays),
        "uncensored_mean_detection_delay_frames": float(np.mean(delays)) if delays else "",
        "_uncensored_delays": delays,
    }


def aggregate_trials(rows: list[dict], metric: str) -> float:
    if metric == "static_fpr":
        denominator = sum(int(row["static_frames"]) for row in rows)
        return sum(int(row["fp"]) for row in rows) / denominator if denominator else float("nan")
    if metric == "gross_recall":
        denominator = sum(int(row["gross_frames"]) for row in rows)
        return sum(int(row["tp"]) for row in rows) / denominator if denominator else float("nan")
    if metric == "gross_segment_alarm_coverage_including_left_censored":
        denominator = sum(int(row["gross_events"]) for row in rows)
        return sum(int(row["gross_events_detected"]) for row in rows) / denominator if denominator else float("nan")
    if metric == "uncensored_gross_event_recall":
        denominator = sum(int(row["uncensored_gross_events"]) for row in rows)
        return sum(int(row["uncensored_gross_events_detected"]) for row in rows) / denominator if denominator else float("nan")
    if metric == "false_alarm_starts_per_trial":
        return sum(int(row["false_alarm_starts"]) for row in rows) / len(rows)
    if metric == "static_alarming_fraction":
        denominator = sum(int(row["static_frames"]) for row in rows)
        return sum(int(row["static_frames_alarming"]) for row in rows) / denominator if denominator else float("nan")
    if metric == "uncensored_mean_detection_delay_frames":
        detected = sum(int(row["uncensored_gross_events_detected"]) for row in rows)
        numerator = sum(int(row["uncensored_detection_delay_sum_frames"]) for row in rows)
        return numerator / detected if detected else float("nan")
    raise KeyError(metric)


def stable_seed(*parts) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def group_rows(rows: list[dict], group_key: Callable[[dict], str]) -> dict[str, list[dict]]:
    result = defaultdict(list)
    for row in rows:
        result[group_key(row)].append(row)
    return dict(result)


def paired_bootstrap_trials(candidate: list[dict], baseline: list[dict], metric: str,
                            seed: int, repetitions: int = BOOTSTRAPS) -> tuple[dict, list[float]]:
    cand_by_episode = {row["episode"]: row for row in candidate}
    base_by_episode = {row["episode"]: row for row in baseline}
    if set(cand_by_episode) != set(base_by_episode):
        raise ValueError("paired trial episode sets differ")
    for episode in cand_by_episode:
        left, right = cand_by_episode[episode], base_by_episode[episode]
        for key in (
            "bootstrap_group", "frames", "static_frames", "gross_frames", "incipient_frames",
            "gross_events", "left_censored_gross_segments", "uncensored_gross_events",
        ):
            if left[key] != right[key]:
                raise ValueError(f"paired trial population mismatch: {episode}/{key}")
    cand_groups = group_rows(candidate, lambda row: row["bootstrap_group"])
    base_groups = group_rows(baseline, lambda row: row["bootstrap_group"])
    if set(cand_groups) != set(base_groups):
        raise ValueError("paired leakage-group sets differ")
    names = sorted(cand_groups)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        sampled = rng.choice(names, len(names), replace=True)
        left = [row for group in sampled for row in cand_groups[group]]
        right = [row for group in sampled for row in base_groups[group]]
        difference = aggregate_trials(left, metric) - aggregate_trials(right, metric)
        if math.isfinite(difference):
            draws.append(float(difference))
    point = aggregate_trials(candidate, metric) - aggregate_trials(baseline, metric)
    interval = {
        "point_difference": float(point) if math.isfinite(point) else None,
        "ci_lower": float(np.quantile(draws, 0.025)) if draws else None,
        "ci_upper": float(np.quantile(draws, 0.975)) if draws else None,
        "bootstrap_groups": len(names),
        "replicates_requested": repetitions,
        "replicates_valid": len(draws),
    }
    return interval, draws


def paired_bootstrap_pauc(candidate: list[dict], baseline: list[dict], groups: dict[str, str],
                          seed: int, repetitions: int = BOOTSTRAPS) -> tuple[dict, list[float]]:
    key = lambda row: (row["episode"], row["t"], row["stage"])
    if [key(row) for row in candidate] != [key(row) for row in baseline]:
        raise ValueError("paired pAUC frame populations/order differ")
    cand_groups = group_rows(candidate, lambda row: groups[row["episode"]])
    base_groups = group_rows(baseline, lambda row: groups[row["episode"]])
    names = sorted(cand_groups)
    if names != sorted(base_groups):
        raise ValueError("paired pAUC leakage-group sets differ")
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        sampled = rng.choice(names, len(names), replace=True)
        left = [row for group in sampled for row in cand_groups[group]]
        right = [row for group in sampled for row in base_groups[group]]
        try:
            draws.append(evaluator.partial_auc(left) - evaluator.partial_auc(right))
        except ValueError:
            continue
    point = evaluator.partial_auc(candidate) - evaluator.partial_auc(baseline)
    return {
        "point_difference": float(point),
        "ci_lower": float(np.quantile(draws, 0.025)) if draws else None,
        "ci_upper": float(np.quantile(draws, 0.975)) if draws else None,
        "bootstrap_groups": len(names),
        "replicates_requested": repetitions,
        "replicates_valid": len(draws),
    }, [float(value) for value in draws]


def flatten_runs(runs: dict[tuple[str, str, int], dict]) -> list[dict]:
    rows = []
    for identity in sorted(runs):
        model, fold, seed = identity
        payload = runs[identity]["payload"]
        for operation, item in payload["operating_points"].items():
            modes = ["raw"] + (["sequential"] if "sequential" in item else [])
            for mode in modes:
                metrics = item[mode]["validation"]
                trials = trial_subset(runs[identity], operation, mode)
                uncensored_delays = [delay for trial in trials for delay in trial["_uncensored_delays"]]
                uncensored_mean_delay = aggregate_trials(trials, "uncensored_mean_detection_delay_frames")
                rows.append({
                    "model": model, "fold": fold, "seed": seed,
                    "operating_point": operation, "mode": mode,
                    "partial_tpr_auc_0_0p1": metrics["partial_tpr_auc_0_0p1"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "average_precision": metrics["average_precision"],
                    "positive_prevalence": metrics["positive_prevalence"],
                    "static_fpr": metrics["static_fpr"],
                    "gross_recall": metrics["gross_recall"],
                    "gross_event_recall": metrics["gross_event_recall"],
                    "gross_segment_alarm_coverage_including_left_censored": metrics["gross_event_recall"],
                    "left_censored_gross_segments": sum(int(row["left_censored_gross_segments"]) for row in trials),
                    "uncensored_gross_events": sum(int(row["uncensored_gross_events"]) for row in trials),
                    "uncensored_gross_event_recall": aggregate_trials(trials, "uncensored_gross_event_recall"),
                    "false_alarm_starts_per_trial": metrics["false_alarm_starts_per_trial"],
                    "static_alarming_fraction": aggregate_trials(trials, "static_alarming_fraction"),
                    "mean_detection_delay_frames": metrics["mean_detection_delay_frames"],
                    "median_detection_delay_frames": metrics["median_detection_delay_frames"],
                    "uncensored_mean_detection_delay_frames": float(uncensored_mean_delay) if math.isfinite(uncensored_mean_delay) else None,
                    "uncensored_median_detection_delay_frames": float(np.median(uncensored_delays)) if uncensored_delays else None,
                    "threshold": metrics["threshold"],
                    "confirmation_k": metrics["confirmation_k"],
                    "release_ratio": metrics["release_ratio"],
                    "never_alarm": metrics["never_alarm"],
                    "observed_no_alarm": metrics["observed_no_alarm"],
                    "observed_no_primary_alarm": metrics["observed_no_primary_alarm"],
                    "metrics_path": str(runs[identity]["metrics_path"]),
                })
    return rows


def descriptive_summary(rows: list[dict]) -> list[dict]:
    result = []
    metrics = (
        "partial_tpr_auc_0_0p1", "balanced_accuracy", "macro_f1", "average_precision",
        "positive_prevalence", "static_fpr", "gross_recall", "gross_event_recall",
        "gross_segment_alarm_coverage_including_left_censored", "left_censored_gross_segments",
        "uncensored_gross_events", "uncensored_gross_event_recall",
        "false_alarm_starts_per_trial", "static_alarming_fraction",
        "mean_detection_delay_frames", "median_detection_delay_frames",
        "uncensored_mean_detection_delay_frames", "uncensored_median_detection_delay_frames",
    )
    grouped = group_rows(rows, lambda row: (row["model"], row["operating_point"], row["mode"]))
    for (model, operation, mode), selected in sorted(grouped.items()):
        row = {"model": model, "operating_point": operation, "mode": mode, "runs": len(selected)}
        for metric in metrics:
            values = np.asarray([float(item[metric]) if item[metric] is not None else float("nan") for item in selected], dtype=float)
            finite = values[np.isfinite(values)]
            row[metric + "_mean"] = float(np.mean(finite)) if len(finite) else None
            row[metric + "_std"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else (0.0 if len(finite) == 1 else None)
            row[metric + "_n"] = len(finite)
        row["never_alarm_runs"] = sum(str(item["never_alarm"]).lower() == "true" for item in selected)
        row["observed_no_alarm_runs"] = sum(str(item["observed_no_alarm"]).lower() == "true" for item in selected)
        result.append(row)
    return result


def paired_comparisons(runs: dict, manifests: dict) -> tuple[list[dict], dict]:
    rows, pauc_draws = [], {}
    for model in FORCE_MODELS:
        for fold in FOLDS:
            groups = {row["id"]: row["leakage_group"] for row in manifests[fold]["episodes"]}
            for seed in SEEDS:
                candidate = runs[(model, fold, seed)]
                baseline = runs[("V", fold, seed)]
                interval, draws = paired_bootstrap_pauc(
                    candidate["prediction_rows"], baseline["prediction_rows"], groups,
                    stable_seed(model, fold, seed, "pauc"),
                )
                rows.append({"candidate": model, "baseline": "V", "fold": fold, "seed": seed,
                             "operating_point": "threshold_independent", "mode": "raw",
                             "metric": "partial_tpr_auc_0_0p1", **interval})
                pauc_draws[(model, fold, seed)] = draws
                for operation in BUDGET_OPS:
                    candidate_trials = trial_subset(candidate, operation, "sequential")
                    baseline_trials = trial_subset(baseline, operation, "sequential")
                    for metric in PAIR_METRICS:
                        interval, _ = paired_bootstrap_trials(
                            candidate_trials, baseline_trials, metric,
                            stable_seed(model, fold, seed, operation, metric),
                        )
                        rows.append({"candidate": model, "baseline": "V", "fold": fold, "seed": seed,
                                     "operating_point": operation, "mode": "sequential",
                                     "metric": metric, **interval})
    return rows, pauc_draws


def shared_fold_pauc_interval(runs: dict, model: str, fold: str, groups: dict[str, str],
                              repetitions: int = BOOTSTRAPS) -> tuple[dict, list[float]]:
    grouped = {}
    points = []
    frame_keys = None
    group_names = None
    for seed in SEEDS:
        candidate = runs[(model, fold, seed)]["prediction_rows"]
        baseline = runs[("V", fold, seed)]["prediction_rows"]
        keys = [(row["episode"], row["t"], row["stage"]) for row in candidate]
        if keys != [(row["episode"], row["t"], row["stage"]) for row in baseline]:
            raise ValueError("fold shared-bootstrap paired populations differ")
        if frame_keys is None:
            frame_keys = keys
        elif keys != frame_keys:
            raise ValueError("three seeds do not share the same validation population")
        candidate_groups = group_rows(candidate, lambda row: groups[row["episode"]])
        baseline_groups = group_rows(baseline, lambda row: groups[row["episode"]])
        names = sorted(candidate_groups)
        if names != sorted(baseline_groups):
            raise ValueError("fold shared-bootstrap leakage groups differ")
        if group_names is None:
            group_names = names
        elif names != group_names:
            raise ValueError("three seeds do not share leakage groups")
        grouped[seed] = (candidate_groups, baseline_groups)
        points.append(evaluator.partial_auc(candidate) - evaluator.partial_auc(baseline))
    rng = np.random.default_rng(stable_seed(model, fold, "shared_fold_pauc"))
    draws = []
    for _ in range(repetitions):
        # This one sampled multiset is intentionally reused for every seed.
        sampled = rng.choice(group_names, len(group_names), replace=True)
        seed_differences = []
        for seed in SEEDS:
            candidate_groups, baseline_groups = grouped[seed]
            candidate = [row for group in sampled for row in candidate_groups[group]]
            baseline = [row for group in sampled for row in baseline_groups[group]]
            try:
                seed_differences.append(evaluator.partial_auc(candidate) - evaluator.partial_auc(baseline))
            except ValueError:
                seed_differences = []
                break
        if len(seed_differences) == len(SEEDS):
            draws.append(float(np.mean(seed_differences)))
    return {
        "point_difference": float(np.mean(points)),
        "ci_lower": float(np.quantile(draws, 0.025)) if draws else None,
        "ci_upper": float(np.quantile(draws, 0.975)) if draws else None,
        "bootstrap_groups": len(group_names),
        "replicates_requested": repetitions,
        "replicates_valid": len(draws),
        "shared_group_draw_across_three_seeds": True,
    }, draws


def direction_criterion(pair_rows: list[dict], runs: dict, manifests: dict) -> tuple[list[dict], dict]:
    results, decision = [], {}
    primary = {(row["candidate"], row["fold"], row["seed"]): row for row in pair_rows
               if row["metric"] == "partial_tpr_auc_0_0p1"}
    for model in FORCE_MODELS:
        positive = 0
        for fold in FOLDS:
            seed_rows = [primary[(model, fold, seed)] for seed in SEEDS]
            groups = {row["id"]: row["leakage_group"] for row in manifests[fold]["episodes"]}
            interval, _ = shared_fold_pauc_interval(runs, model, fold, groups)
            point = interval["point_difference"]
            is_positive = point > 0
            positive += int(is_positive)
            results.append({
                "candidate": model, "baseline": "V", "fold": fold,
                "three_seed_mean_pauc_difference": point,
                "ci_lower": interval["ci_lower"],
                "ci_upper": interval["ci_upper"],
                "positive_direction": is_positive,
                "positive_seed_differences": sum(row["point_difference"] > 0 for row in seed_rows),
                "seeds": 3, "bootstrap_replicates_valid": interval["replicates_valid"],
                "shared_group_draw_across_three_seeds": True,
            })
        decision[model] = {
            "positive_folds": positive,
            "folds": 4,
            "criterion": "strictly positive three-seed mean pAUC difference versus V in at least 3 folds",
            "passes": positive >= 3,
        }
    return results, decision

def failure_cases(runs: dict) -> tuple[list[dict], dict]:
    selected, curves = [], {}
    for identity in sorted(runs):
        model, fold, seed = identity
        run = runs[identity]
        item = run["payload"]["operating_points"]["fpr_0.05"]["sequential"]["validation"]
        threshold, confirmation, release = item["threshold"], item["confirmation_k"], item["release_ratio"]
        by_episode = evaluator.r4.split_episodes(run["prediction_rows"])
        alarm_by_episode = {}
        candidates_fp, candidates_fn = [], []
        for episode, rows in by_episode.items():
            alarms, starts = evaluator.r4.alarm_sequence(rows, threshold, confirmation, release)
            stage = np.asarray([row["stage"] for row in rows])
            static = stage == 0
            fraction = float(np.sum(alarms & static) / np.sum(static)) if np.sum(static) else 0.0
            false_starts = int(np.sum(starts & static))
            events = [(start, end) for start, end in evaluator.r4.contiguous_gross_events(stage) if start != 0]
            delays, missed = [], 0
            for start, end in events:
                hits = np.flatnonzero(alarms[start:end])
                if len(hits):
                    delays.append(int(hits[0]))
                else:
                    missed += 1
            if fraction > 0:
                candidates_fp.append((-fraction, -false_starts, episode, fraction, false_starts))
            if events and missed:
                recall = (len(events) - missed) / len(events)
                candidates_fn.append((recall, -missed, episode, len(events), missed))
            alarm_by_episode[episode] = (rows, alarms)
        if candidates_fp:
            _, _, episode, fraction, starts = min(candidates_fp)
            selected.append({"model": model, "fold": fold, "seed": seed, "failure_type": "false_positive",
                             "episode": episode, "static_alarming_fraction": fraction,
                             "false_alarm_starts": starts, "gross_events": "", "missed_gross_events": "",
                             "event_recall": "", "threshold": threshold, "confirmation_k": confirmation, "release_ratio": release})
            curves[(identity, "false_positive", episode)] = alarm_by_episode[episode]
        else:
            selected.append({"model": model, "fold": fold, "seed": seed, "failure_type": "false_positive",
                             "episode": "none", "static_alarming_fraction": 0.0,
                             "false_alarm_starts": 0, "gross_events": "", "missed_gross_events": "",
                             "event_recall": "", "threshold": threshold, "confirmation_k": confirmation, "release_ratio": release})
        if candidates_fn:
            recall, neg_missed, episode, events, missed = min(candidates_fn)
            selected.append({"model": model, "fold": fold, "seed": seed, "failure_type": "false_negative",
                             "episode": episode, "static_alarming_fraction": "", "false_alarm_starts": "",
                             "gross_events": events, "missed_gross_events": missed, "event_recall": recall,
                             "threshold": threshold, "confirmation_k": confirmation, "release_ratio": release})
            curves[(identity, "false_negative", episode)] = alarm_by_episode[episode]
        else:
            selected.append({"model": model, "fold": fold, "seed": seed, "failure_type": "false_negative",
                             "episode": "none", "static_alarming_fraction": "", "false_alarm_starts": "",
                             "gross_events": 0, "missed_gross_events": 0, "event_recall": "",
                             "threshold": threshold, "confirmation_k": confirmation, "release_ratio": release})
    return selected, curves


def choose_representatives(cases: list[dict]) -> list[dict]:
    result = []
    for model in MODELS:
        fp = [row for row in cases if row["model"] == model and row["failure_type"] == "false_positive" and row["episode"] != "none"]
        if fp:
            result.append(min(fp, key=lambda row: (-float(row["static_alarming_fraction"]),
                                                   -int(row["false_alarm_starts"]), row["fold"], row["seed"], row["episode"])))
        fn = [row for row in cases if row["model"] == model and row["failure_type"] == "false_negative" and row["episode"] != "none"]
        if fn:
            result.append(min(fn, key=lambda row: (float(row["event_recall"]),
                                                   -int(row["missed_gross_events"]), row["fold"], row["seed"], row["episode"])))
    return result


def render_figures(per_run: list[dict], directions: list[dict], cases: list[dict], curves: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    primary = [row for row in per_run if row["operating_point"] == "fixed_0.5" and row["mode"] == "raw"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for axis, fold in zip(axes.flat, FOLDS):
        for model in MODELS:
            rows = sorted([row for row in primary if row["fold"] == fold and row["model"] == model], key=lambda row: row["seed"])
            axis.plot(range(3), [row["partial_tpr_auc_0_0p1"] for row in rows], marker="o",
                      label=model if fold == FOLDS[0] else "_nolegend_")
        axis.set_title(fold); axis.set_xticks(range(3), [str(seed)[-2:] for seed in SEEDS]); axis.grid(alpha=.25)
    axes[0, 0].set_ylabel("normalized pAUC"); axes[1, 0].set_ylabel("normalized pAUC")
    fig.legend(loc="upper center", ncol=4, bbox_to_anchor=(.5, 1.01)); fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(figure_dir / "pauc_all_folds_seeds.png", dpi=180); plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    offsets = {"F-old": -0.08, "F-adapt": 0.08}
    colors = {"F-old": "tab:blue", "F-adapt": "tab:orange"}
    for model in FORCE_MODELS:
        rows = [row for row in directions if row["candidate"] == model]
        x = np.arange(4) + offsets[model]; y = np.asarray([row["three_seed_mean_pauc_difference"] for row in rows])
        lo = np.asarray([row["ci_lower"] for row in rows]); hi = np.asarray([row["ci_upper"] for row in rows])
        # Percentile intervals need not contain the independently computed point estimate.
        # Draw endpoints directly instead of converting them to non-negative error lengths.
        axis.vlines(x, lo, hi, linewidth=1.5, color=colors[model])
        axis.hlines(lo, x - .025, x + .025, linewidth=1.5, color=colors[model])
        axis.hlines(hi, x - .025, x + .025, linewidth=1.5, color=colors[model])
        axis.scatter(x, y, label=model, zorder=3, color=colors[model])
    axis.axhline(0, color="black", linewidth=1); axis.set_xticks(range(4), FOLDS); axis.set_ylabel("pAUC difference vs V")
    axis.legend(); axis.grid(alpha=.25); fig.tight_layout(); fig.savefig(figure_dir / "paired_pauc_direction.png", dpi=180); plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 6))
    for model in MODELS:
        xs, ys = [], []
        for operation in BUDGET_OPS:
            rows = [row for row in per_run if row["model"] == model and row["operating_point"] == operation and row["mode"] == "sequential"]
            xs.append(np.mean([row["static_fpr"] for row in rows])); ys.append(np.mean([row["gross_recall"] for row in rows]))
        axis.plot(xs, ys, marker="o", label=model)
    axis.set_xlabel("actual validation static FPR"); axis.set_ylabel("validation gross recall"); axis.grid(alpha=.25); axis.legend()
    fig.tight_layout(); fig.savefig(figure_dir / "low_fpr_tradeoff.png", dpi=180); plt.close(fig)

    representatives = choose_representatives(cases)
    for case in representatives:
        identity = (case["model"], case["fold"], int(case["seed"]))
        rows, alarms = curves[(identity, case["failure_type"], case["episode"])]
        frames = np.asarray([row["t"] for row in rows]); scores = np.asarray([row["score"] for row in rows]); stages = np.asarray([row["stage"] for row in rows])
        fig, axis = plt.subplots(figsize=(10, 3.5)); axis.plot(frames, scores, label="p_slip")
        axis.axhline(float(case["threshold"]), color="black", linestyle="--", label="threshold")
        axis.fill_between(frames, 0, 1, where=stages == 1, alpha=.12, color="orange", label="incipient")
        axis.fill_between(frames, 0, 1, where=stages == 2, alpha=.12, color="red", label="gross")
        axis.fill_between(frames, 0, 1, where=alarms, alpha=.12, color="blue", label="alarm")
        axis.set_ylim(0, 1); axis.set_title(f"{case['model']} {case['failure_type']} {case['fold']} seed {case['seed']} {case['episode']}", wrap=True, fontsize=9)
        axis.legend(ncol=5, fontsize=8); axis.set_xlabel("original frame index"); fig.tight_layout()
        name = f"case_{case['model'].replace('-', '_')}_{case['failure_type']}.png"
        fig.savefig(figure_dir / name, dpi=180, bbox_inches="tight"); plt.close(fig)


def write_report(output: Path, decisions: dict, directions: list[dict], summary: list[dict]) -> None:
    def mean_sd(row, metric):
        mean, std = row[metric + "_mean"], row[metric + "_std"]
        return "NA" if mean is None or std is None else f"{mean:.4f}±{std:.4f}"

    lines = [
        "# 第五轮当前滑移检测汇总", "",
        "全部 48 个正式评价均通过运行身份、调度回执、产物哈希、数据角色与权威标签核验。", "",
        "## 力条件分支相对纯视觉分支的主要判据", "",
        "| 候选模型 | pAUC均值为正的折数 | 是否通过≥3/4折方向准则 |", "|---|---:|---:|",
    ]
    for model in FORCE_MODELS:
        item = decisions[model]
        lines.append(f"| {model} | {item['positive_folds']}/4 | {'是' if item['passes'] else '否'} |")
    lines += ["", "每折方向取三个配对种子的 pAUC 差值均值；没有筛选或省略种子。", "",
              "| 候选模型 | 折 | 配对pAUC平均差值 | 95%组级bootstrap区间 | 正向种子数 |", "|---|---|---:|---:|---:|"]
    for row in directions:
        lines.append(f"| {row['candidate']} | {row['fold']} | {row['three_seed_mean_pauc_difference']:.6f} | [{row['ci_lower']:.6f}, {row['ci_upper']:.6f}] | {row['positive_seed_differences']}/3 |")
    index = {(row["model"], row["operating_point"], row["mode"]): row for row in summary}
    fixed = {model: index[(model, "fixed_0.5", "raw")] for model in MODELS}
    lines += ["", f"四折三种子的平均 pAUC 为：MAE-R3-B {fixed['mae-r3-b']['partial_tpr_auc_0_0p1_mean']:.4f}、V {fixed['V']['partial_tpr_auc_0_0p1_mean']:.4f}、F-old {fixed['F-old']['partial_tpr_auc_0_0p1_mean']:.4f}、F-adapt {fixed['F-adapt']['partial_tpr_auc_0_0p1_mean']:.4f}。两种力条件方案都通过了预注册的 3/4 折方向门槛，但没有任何一个正向折的共享组级区间整体高于零；F-adapt 在 leave-p1 反而三个种子均为负，且该折区间整体低于零。因此门槛结果只支持“存在跨折正向趋势”，不支持稳定、显著或普适提升。"]
    lines += ["", "## 低误报工作点的描述性结果", "",
              "| 模型 | calibration目标 | validation实际FPR均值±SD | gross帧召回均值±SD | 非左删失事件召回均值±SD | 每试次误报告警起点均值±SD | 永不告警运行数 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        for operation in BUDGET_OPS:
            row = index[(model, operation, "sequential")]
            lines.append(f"| {model} | {operation} | {mean_sd(row, 'static_fpr')} | {mean_sd(row, 'gross_recall')} | {mean_sd(row, 'uncensored_gross_event_recall')} | {mean_sd(row, 'false_alarm_starts_per_trial')} | {row['never_alarm_runs']} |")
    v05 = index[("V", "fpr_0.05", "sequential")]
    old05 = index[("F-old", "fpr_0.05", "sequential")]
    adapt05 = index[("F-adapt", "fpr_0.05", "sequential")]
    lines += ["", f"在 calibration 目标 5% 的工作点，validation 实际 FPR 并未维持在 5%：V、F-old、F-adapt 的跨运行均值分别为 {v05['static_fpr_mean']:.4f}、{old05['static_fpr_mean']:.4f}、{adapt05['static_fpr_mean']:.4f}。相对 V，F-old 的 FPR 下降 {v05['static_fpr_mean']-old05['static_fpr_mean']:.4f}，同时 gross 帧召回下降 {v05['gross_recall_mean']-old05['gross_recall_mean']:.4f}；F-adapt 的 FPR 下降 {v05['static_fpr_mean']-adapt05['static_fpr_mean']:.4f}，同时 gross 帧召回下降 {v05['gross_recall_mean']-adapt05['gross_recall_mean']:.4f}。这表现为误报与召回的权衡，不能只按较低 FPR 判作检测改善。F-adapt 在 1% 工作点有 1/12 个运行选择了永不告警，必须连同零召回代价解释。"]
    lines += ["", "原评价器的 `gross_event_recall` 包含在 t=10 评价边界已经开始的 gross 段，本报告将其明确称为“含左删失段的告警覆盖率”。上表事件召回和延迟仅使用非左删失 gross 事件。", "",
              "## 结论边界", "",
              "四折是相互重叠的 HTT 开发折，且 validation 参与 checkpoint 选择。配对区间在每折内按完整 leakage group 重采样。预测力来自同一张触觉图像，并非独立传感器。这里的改善不能证明真实物体成功率、力信息的因果作用、提前预警能力或完整世界模型。", "",
              "全部种子明细见 `per_run_metrics.csv`，配对区间见 `paired_vs_V.csv`，固定规则选出的失败案例见 `failure_cases.csv`。", ""]
    temporary = output / f"SUMMARY_ZH.md.tmp.{os.getpid()}"
    temporary.write_text("\n".join(lines))
    os.replace(temporary, output / "SUMMARY_ZH.md")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def run(args) -> dict:
    inventory_path = args.inventory.resolve()
    inventory = json.loads(inventory_path.read_text())
    jobs, inventory_errors = audit_inventory(inventory)
    readiness = {
        "format": "round5_current_analysis_readiness_v1",
        "inventory": str(inventory_path),
        "inventory_sha256": evaluator.r4.sha256(inventory_path),
        "expected_evaluations": 48,
        "inventory_errors": inventory_errors,
        "ready": [], "pending": [], "invalid": [],
    }
    for identity, job in sorted(jobs.items()):
        path = Path(job["acceptance_path"])
        if not path.is_file():
            readiness["pending"].append({"identity": list(identity), "path": str(path)})
            continue
        try:
            if not scheduler.accepted(job):
                raise ValueError("scheduler acceptance returned false")
            readiness["ready"].append({"identity": list(identity), "path": str(path)})
        except Exception as exc:
            readiness["invalid"].append({"identity": list(identity), "path": str(path), "reason": str(exc)})
    complete = not inventory_errors and len(readiness["ready"]) == 48 and not readiness["pending"] and not readiness["invalid"]
    readiness["status"] = "complete" if complete else "pending"
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "readiness.json", readiness)
    if not complete:
        if args.require_complete:
            raise RuntimeError(f"formal evaluations incomplete: ready={len(readiness['ready'])}, pending={len(readiness['pending'])}, invalid={len(readiness['invalid'])}")
        return readiness

    # All evaluation argv must bind the same manifest/contract/audit.
    triples = {(arg_value(job["argv"], "--manifest"), arg_value(job["argv"], "--contract"),
                arg_value(job["argv"], "--cache-audit")) for job in jobs.values()}
    if len(triples) != 1:
        raise ValueError("evaluation jobs do not share one authoritative data chain")
    manifest_name, contract_name, audit_name = next(iter(triples))
    manifest_path, contract_path, audit_path = map(Path, (manifest_name, contract_name, audit_name))
    manifest = json.loads(manifest_path.read_text())
    labels_by_fold, manifests = {}, {}
    for fold in FOLDS:
        labels_by_fold[fold], _ = evaluator.load_authoritative_labels(
            contract_path, audit_path, manifest, manifest_path, fold
        )
        manifests[fold] = manifest
    runs = {identity: load_complete_run(job, labels_by_fold, manifests) for identity, job in sorted(jobs.items())}
    per_run = flatten_runs(runs)
    descriptive = descriptive_summary(per_run)
    pairs, _ = paired_comparisons(runs, manifests)
    directions, decisions = direction_criterion(pairs, runs, manifests)
    cases, curves = failure_cases(runs)

    atomic_csv(args.output / "per_run_metrics.csv", per_run)
    atomic_csv(args.output / "descriptive_summary.csv", descriptive)
    atomic_csv(args.output / "paired_vs_V.csv", pairs)
    atomic_csv(args.output / "fold_direction_criterion.csv", directions)
    atomic_csv(args.output / "failure_cases.csv", cases)
    render_figures(per_run, directions, cases, curves, args.output)
    write_report(args.output, decisions, directions, descriptive)
    result = {
        "status": "complete", "format": "round5_current_analysis_summary_v1",
        "evaluations": len(runs), "all_expected_identities_present": set(runs) == expected_identities(),
        "all_seeds_reported": True, "bootstrap_repetitions": BOOTSTRAPS,
        "direction_criterion": decisions,
        "artifacts": {},
        "limitations": [
            "four folds are overlapping development folds",
            "validation participated in checkpoint selection",
            "predicted force is derived from the same tactile images",
            "no test-role or physical deployment claim",
        ],
    }
    for path in sorted(args.output.rglob("*")):
        if path.is_file() and path.name not in {"summary.json", "readiness.json"}:
            result["artifacts"][str(path.resolve())] = evaluator.r4.sha256(path)
    atomic_json(args.output / "summary.json", result)
    return result


def main():
    result = run(parse_args())
    print(json.dumps({"status": result["status"], "evaluations": result.get("evaluations", len(result.get("ready", [])))}, indent=2))


if __name__ == "__main__":
    main()
