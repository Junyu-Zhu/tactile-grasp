#!/usr/bin/env python3
"""Independent CPU-only integration audit for Round 4 deliverables."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


MODELS = ("mae", "dino", "ijepa", "mae_letterbox")
NEW_ENCODERS = ("dino", "ijepa", "mae_letterbox")
FOLDS = tuple(f"htt_leave_p{i}" for i in range(1, 5))
SEEDS = (20260914, 20260915, 20260916)
OPS = ("fixed_0.5", "max_ba", "fpr_0.01", "fpr_0.05", "fpr_0.10")
FPR_OPS = {"fpr_0.01": .01, "fpr_0.05": .05, "fpr_0.10": .10}
RULES = tuple((k, ratio) for k in (1, 2, 3) for ratio in (1.0, .8))
FLOAT_TOL = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(value, encoding="utf-8")
    os.replace(temp, path)


def read_predictions(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8", newline="") as stream:
        for raw in csv.DictReader(stream):
            episode = raw.get("episode_id", raw.get("episode"))
            score = float(raw.get("p_slip", raw.get("probability", "nan")))
            row = {"episode": episode, "t": int(raw["t"]), "stage": int(raw["stage"]), "score": score}
            if not episode or row["stage"] not in (0, 1, 2) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"invalid prediction row: {path}")
            rows.append(row)
    rows.sort(key=lambda row: (row["episode"], row["t"]))
    grouped = split_episodes(rows)
    if not rows or any([row["t"] for row in episode] != list(range(len(episode))) for episode in grouped.values()):
        raise ValueError(f"empty/non-contiguous predictions: {path}")
    if sum(len(value) for value in grouped.values()) != len({(row["episode"], row["t"]) for row in rows}):
        raise ValueError(f"duplicate prediction frame: {path}")
    return rows


def split_episodes(rows: Iterable[dict]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        result[row["episode"]].append(row)
    return dict(result)


def alarm_state(scores: np.ndarray, threshold: float, k: int, release_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    alarms = np.zeros(len(scores), dtype=bool); starts = np.zeros(len(scores), dtype=bool)
    active = False; consecutive = 0
    for index, score in enumerate(scores):
        if active:
            if score < release_ratio * threshold:
                active = False; consecutive = 0
        else:
            consecutive = consecutive + 1 if score >= threshold else 0
            if consecutive >= k:
                active = True; starts[index] = True
        alarms[index] = active
    return alarms, starts


def gross_events(stages: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, stages == 2, False]
    starts = np.flatnonzero(~padded[:-1] & padded[1:]); ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return list(zip(starts.tolist(), ends.tolist()))


def confusion(stages: np.ndarray, alarms: np.ndarray) -> dict[str, Any]:
    mask = np.isin(stages, (0, 2)); positive = stages[mask] == 2; predicted = alarms[mask]
    tn = int(np.sum(~positive & ~predicted)); fp = int(np.sum(~positive & predicted))
    fn = int(np.sum(positive & ~predicted)); tp = int(np.sum(positive & predicted))
    fpr = fp / (tn + fp) if tn + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "static_fpr": fpr,
            "gross_recall": recall, "balanced_accuracy": ((1 - fpr) + recall) / 2}


def independently_score(rows: list[dict], threshold: float, k: int = 1, release_ratio: float = 1.0) -> tuple[dict, list[dict]]:
    trials = []; all_stages = []; all_alarms = []; delays = []; frame_zero = 0
    for episode, episode_rows in split_episodes(rows).items():
        stages = np.asarray([row["stage"] for row in episode_rows], dtype=np.int8)
        scores = np.asarray([row["score"] for row in episode_rows], dtype=np.float64)
        alarms, starts = alarm_state(scores, threshold, k, release_ratio)
        metrics = confusion(stages, alarms); events = gross_events(stages); event_delays = []
        for start, end in events:
            frame_zero += int(start == 0)
            hit = np.flatnonzero(alarms[start:end])
            if len(hit):
                event_delays.append(int(hit[0]))
        delays.extend(event_delays)
        trials.append({"episode": episode, "frames": len(stages), "static_frames": int(np.sum(stages == 0)),
                       "gross_frames": int(np.sum(stages == 2)), "incipient_frames": int(np.sum(stages == 1)),
                       **{key: metrics[key] for key in ("tn", "fp", "fn", "tp")},
                       "false_alarm_starts": int(np.sum(starts & (stages == 0))),
                       "static_frames_alarming": int(np.sum(alarms & (stages == 0))),
                       "gross_events": len(events), "gross_events_detected": len(event_delays),
                       "mean_detection_delay_frames": None if not event_delays else float(np.mean(event_delays))})
        all_stages.extend(stages.tolist()); all_alarms.extend(alarms.tolist())
    result = confusion(np.asarray(all_stages), np.asarray(all_alarms))
    result.update({"threshold": float(threshold), "confirmation_k": k, "release_ratio": release_ratio,
                   "never_alarm": threshold > 1.0, "observed_no_alarm": not any(all_alarms),
                   "observed_no_primary_alarm": not np.any(np.asarray(all_alarms)[np.isin(np.asarray(all_stages), (0, 2))]),
                   "false_alarm_starts": sum(row["false_alarm_starts"] for row in trials),
                   "false_alarm_starts_per_trial": sum(row["false_alarm_starts"] for row in trials) / len(trials),
                   "gross_events": sum(row["gross_events"] for row in trials),
                   "gross_events_detected": sum(row["gross_events_detected"] for row in trials),
                   "gross_event_recall": (sum(row["gross_events_detected"] for row in trials) / sum(row["gross_events"] for row in trials)
                                          if sum(row["gross_events"] for row in trials) else float("nan")),
                   "mean_detection_delay_frames": None if not delays else float(np.mean(delays)),
                   "median_detection_delay_frames": None if not delays else float(np.median(delays)),
                   "segments_beginning_at_frame_zero": frame_zero})
    return result, trials


def threshold_candidates(rows: list[dict]) -> list[dict]:
    scores = np.asarray([row["score"] for row in rows]); stages = np.asarray([row["stage"] for row in rows])
    return [{"threshold": threshold, **confusion(stages, scores >= threshold)}
            for threshold in sorted(set(scores.tolist())) + [float(np.nextafter(1.0, math.inf))]]


def independently_select_threshold(rows: list[dict], op: str) -> float:
    if op == "fixed_0.5": return .5
    candidates = threshold_candidates(rows)
    if op == "max_ba":
        return max(candidates, key=lambda row: (row["balanced_accuracy"], -row["static_fpr"], row["threshold"]))["threshold"]
    budget = FPR_OPS[op]
    feasible = [row for row in candidates if row["static_fpr"] <= budget + 1e-15]
    return max(feasible, key=lambda row: (row["gross_recall"], -row["static_fpr"], row["threshold"]))["threshold"]


def independently_select_rule(rows: list[dict], threshold: float, budget: float) -> tuple[float, int, float]:
    candidates = []
    for k, ratio in RULES:
        metrics, _ = independently_score(rows, threshold, k, ratio)
        if metrics["static_fpr"] <= budget + 1e-15:
            candidates.append(metrics)
    if not candidates:
        return float(np.nextafter(1.0, math.inf)), 1, 1.0
    chosen = max(candidates, key=lambda row: (row["gross_recall"], -row["false_alarm_starts"],
                                               -row["static_fpr"], -row["confirmation_k"], row["release_ratio"]))
    return threshold, chosen["confirmation_k"], chosen["release_ratio"]


def same_number(left: Any, right: Any, tolerance: float = FLOAT_TOL) -> bool:
    if left is None or right is None: return left is right
    if isinstance(left, bool) or isinstance(right, bool): return left is right
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def compare_metrics(actual: dict, expected: dict, context: str, errors: list[str]) -> None:
    keys = ("tn", "fp", "fn", "tp", "static_fpr", "gross_recall", "balanced_accuracy", "threshold",
            "confirmation_k", "release_ratio", "never_alarm", "observed_no_alarm", "observed_no_primary_alarm",
            "false_alarm_starts", "false_alarm_starts_per_trial", "gross_events", "gross_events_detected",
            "gross_event_recall", "mean_detection_delay_frames", "median_detection_delay_frames",
            "segments_beginning_at_frame_zero")
    for key in keys:
        if key not in actual or not same_number(actual[key], expected[key]):
            errors.append(f"metric mismatch {context}/{key}: stored={actual.get(key)} independent={expected.get(key)}")


def read_trial_csv(path: Path) -> dict[tuple[str, str, str, str], dict]:
    result = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["operating_point"], row["mode"], row["role"], row["episode"])
            if key in result: raise ValueError(f"duplicate trial row: {path}/{key}")
            result[key] = row
    return result


def compare_trials(independent: list[dict], stored: dict, op: str, mode: str, role: str,
                   run_metrics: dict, context: str, errors: list[str]) -> None:
    keys = ("frames", "static_frames", "gross_frames", "incipient_frames", "tn", "fp", "fn", "tp",
            "false_alarm_starts", "static_frames_alarming", "gross_events", "gross_events_detected",
            "mean_detection_delay_frames")
    selected = []
    for trial in independent:
        key = (op, mode, role, trial["episode"]); row = stored.get(key)
        if row is None:
            errors.append(f"missing per-trial row {context}/{key}"); continue
        selected.append(row)
        for name in keys:
            raw = row[name]
            parsed = None if raw == "" else float(raw)
            if not same_number(parsed, trial[name]): errors.append(f"trial mismatch {context}/{key}/{name}")
    for name in ("tn", "fp", "fn", "tp", "false_alarm_starts", "gross_events", "gross_events_detected"):
        aggregate = sum(float(row[name]) for row in selected)
        if not same_number(aggregate, run_metrics[name]): errors.append(f"trial aggregate mismatch {context}/{op}/{mode}/{role}/{name}")


def check_alarm_runs(root: Path, r3_root: Path, errors: list[str]) -> dict:
    expected = {(model, fold, seed) for model in MODELS for fold in FOLDS for seed in SEEDS}
    metrics_paths = sorted((root / "alarms" / "runs").glob("*/*/seed_*/metrics.json"))
    actual = set(); raw_checks = sequential_checks = trial_checks = 0
    for path in metrics_paths:
        data = json.loads(path.read_text()); identity = (data.get("model"), data.get("fold"), data.get("seed")); actual.add(identity)
        context = "/".join(map(str, identity)); source = Path(data.get("source", ""))
        if data.get("status") != "complete" or "/smoke/" in f"/{source}/": errors.append(f"nonformal alarm source {context}")
        summary_path = source / "training_summary.json"
        if not summary_path.is_file(): errors.append(f"missing source summary {context}"); continue
        summary = json.loads(summary_path.read_text())
        if summary.get("status") != "complete" or summary.get("config", {}).get("smoke") is not False:
            errors.append(f"alarm prediction source is not formal {context}")
        expected_source = (r3_root / "runs" / "B" / f"fold_p{identity[1][-1]}" / f"seed_{identity[2]}"
                           if identity[0] == "mae" else root / "encoders" / "runs" / identity[0] / identity[1] / str(identity[2]))
        if source.resolve() != expected_source.resolve(): errors.append(f"unexpected formal prediction source {context}: {source}")
        source_identity_ok = ((summary.get("fold"), summary.get("seed")) == identity[1:] and
                              (identity[0] == "mae" or summary.get("encoder") == identity[0]))
        if not source_identity_ok: errors.append(f"alarm/source identity mismatch {context}")
        stored_summary_hash = data.get("provenance", {}).get("training_summary_sha256")
        if stored_summary_hash is None:
            errors.append(f"missing alarm training summary provenance {context}")
        elif sha256(summary_path) != stored_summary_hash:
            errors.append(f"alarm training summary hash mismatch {context}")
        roles = {}
        for role in ("calibration", "validation"):
            prediction = source / "predictions" / f"{role}.csv"
            if not prediction.is_file() or sha256(prediction) != data.get("provenance", {}).get(f"{role}_predictions_sha256"):
                errors.append(f"alarm source hash mismatch {context}/{role}"); continue
            roles[role] = read_predictions(prediction)
        if len(roles) != 2: continue
        stored_trials = read_trial_csv(path.parent / "trials.csv")
        expected_trial_keys = set()
        if set(data.get("operating_points", {})) != set(OPS): errors.append(f"operating point identity mismatch {context}")
        for op in OPS:
            item = data["operating_points"][op]
            expected_threshold = independently_select_threshold(roles["calibration"], op)
            threshold = item["raw"]["calibration"]["threshold"]
            if not same_number(threshold, expected_threshold): errors.append(f"threshold selection mismatch {context}/{op}")
            for role in ("calibration", "validation"):
                independent, trials = independently_score(roles[role], threshold)
                expected_trial_keys.update((op, "raw", role, trial["episode"]) for trial in trials)
                compare_metrics(item["raw"][role], independent, f"{context}/{op}/raw/{role}", errors)
                compare_trials(trials, stored_trials, op, "raw", role, independent, context, errors)
                raw_checks += 1; trial_checks += len(trials)
            if op in FPR_OPS:
                budget = FPR_OPS[op]
                if item["raw"]["calibration"]["static_fpr"] > budget + FLOAT_TOL:
                    errors.append(f"raw calibration FPR violation {context}/{op}")
                sequential = item.get("sequential", {}); cal_rule = sequential.get("calibration", {})
                seq_threshold = cal_rule.get("threshold"); k = cal_rule.get("confirmation_k"); ratio = cal_rule.get("release_ratio")
                if not (isinstance(k, int) and (k, ratio) in RULES) and not (seq_threshold > 1 and k == 1 and ratio == 1.0):
                    errors.append(f"invalid sequential rule {context}/{op}"); continue
                independently_selected = independently_select_rule(roles["calibration"], threshold, budget)
                if not (same_number(seq_threshold, independently_selected[0]) and k == independently_selected[1] and
                        same_number(ratio, independently_selected[2])):
                    errors.append(f"sequential rule selection mismatch {context}/{op}")
                for role in ("calibration", "validation"):
                    independent, trials = independently_score(roles[role], seq_threshold, k, ratio)
                    expected_trial_keys.update((op, "sequential", role, trial["episode"]) for trial in trials)
                    compare_metrics(sequential[role], independent, f"{context}/{op}/sequential/{role}", errors)
                    compare_trials(trials, stored_trials, op, "sequential", role, independent, context, errors)
                    sequential_checks += 1; trial_checks += len(trials)
                if sequential["calibration"]["static_fpr"] > budget + FLOAT_TOL:
                    errors.append(f"sequential calibration FPR violation {context}/{op}")
        if set(stored_trials) != expected_trial_keys:
            errors.append(f"per-trial identity mismatch {context}: missing={len(expected_trial_keys-set(stored_trials))} extra={len(set(stored_trials)-expected_trial_keys)}")
    if actual != expected: errors.append(f"alarm identities differ missing={sorted(expected-actual)} unexpected={sorted(actual-expected)}")
    audit = root / "alarms" / "FINAL_AUDIT.json"
    if not audit.is_file() or json.loads(audit.read_text()).get("status") != "pass": errors.append("alarms FINAL_AUDIT not pass")
    return {"expected_identities": 48, "found_identities": len(actual), "raw_role_recomputations": raw_checks,
            "sequential_role_recomputations": sequential_checks, "per_trial_rows_checked": trial_checks}


def collect_identity_status(root: Path, r3_root: Path, errors: list[str]) -> dict:
    encoder_expected = {(encoder, fold, seed) for encoder in NEW_ENCODERS for fold in FOLDS for seed in SEEDS}
    encoder_actual = set()
    for path in (root / "encoders" / "runs").glob("*/*/*/training_summary.json"):
        data = json.loads(path.read_text()); identity = (data.get("encoder"), data.get("fold"), data.get("seed"))
        if data.get("status") == "complete" and data.get("config", {}).get("smoke") is False: encoder_actual.add(identity)
    if encoder_actual != encoder_expected: errors.append(f"36 encoder identities differ missing={sorted(encoder_expected-encoder_actual)} unexpected={sorted(encoder_actual-encoder_expected)}")
    reused_expected = {(fold, seed) for fold in FOLDS for seed in SEEDS}; reused_actual = set()
    for fold in FOLDS:
        number = fold[-1]
        for seed in SEEDS:
            path = r3_root / "runs" / "B" / f"fold_p{number}" / f"seed_{seed}" / "training_summary.json"
            if path.is_file():
                data = json.loads(path.read_text())
                if (data.get("status") == "complete" and data.get("config", {}).get("smoke") is False and
                        data.get("config", {}).get("init") == "fresh"): reused_actual.add((fold, seed))
    if reused_actual != reused_expected: errors.append(f"12 reused MAE identities differ missing={sorted(reused_expected-reused_actual)}")
    nf_expected = {(variant, seed) for variant in ("C", "D") for seed in SEEDS}; nf_actual = set()
    for path in (root / "normalflow" / "runs").glob("*/*/metrics.json"):
        data = json.loads(path.read_text())
        if data.get("status") == "pass": nf_actual.add((data.get("variant"), data.get("seed")))
    if nf_actual != nf_expected: errors.append(f"6 NormalFlow identities differ missing={sorted(nf_expected-nf_actual)} unexpected={sorted(nf_actual-nf_expected)}")
    future_expected = {(model, seed) for model in ("mlp", "gru") for seed in SEEDS}; future_actual = set()
    for path in (root / "future" / "runs").glob("*/seed_*/training_summary.json"):
        data = json.loads(path.read_text())
        if data.get("status") == "complete" and data.get("config", {}).get("smoke") is False: future_actual.add((data.get("model"), data.get("seed")))
    if future_actual != future_expected: errors.append(f"6 future identities differ missing={sorted(future_expected-future_actual)} unexpected={sorted(future_actual-future_expected)}")
    return {"new_neural_expected": 48, "new_neural_found": len(encoder_actual) + len(nf_actual) + len(future_actual),
            "encoder_new": len(encoder_actual), "normalflow_new": len(nf_actual), "future_new": len(future_actual),
            "mae_reused_expected": 12, "mae_reused_found": len(reused_actual)}


def check_package_audits(root: Path, errors: list[str]) -> dict:
    paths = {"encoders": root / "encoders" / "final_audit.json",
             "normalflow": root / "normalflow" / "FINAL_AUDIT.json",
             "future": root / "future" / "final_audit.json"}
    statuses = {}
    for name, path in paths.items():
        if not path.is_file(): errors.append(f"missing {name} final audit"); statuses[name] = None; continue
        data = json.loads(path.read_text()); statuses[name] = data.get("status")
        if data.get("status") != "pass": errors.append(f"{name} final audit not pass")
        if name == "encoders" and (data.get("expected_runs") != 36 or data.get("verified_runs") != 36 or data.get("errors")):
            errors.append("encoders final audit does not cover 36 clean runs")
        if name == "normalflow" and (data.get("formal_runs") != 6 or data.get("errors")):
            errors.append("NormalFlow final audit does not cover 6 clean runs")
        if name == "future" and (data.get("failed") != 0 or data.get("passed", 0) < 1):
            errors.append("future final audit contains failed checks")
    frozen = root / "normalflow" / "formal_artifacts_pre_recovery_patch.json"
    recovery = root / "normalflow" / "recovery_proof.json"
    if not frozen.is_file() or json.loads(frozen.read_text()).get("status") != "complete":
        errors.append("NormalFlow frozen original manifest invalid")
    else:
        frozen_data = json.loads(frozen.read_text())
        for relative, expected_hash in frozen_data.get("artifacts", {}).items():
            artifact = root / "normalflow" / relative
            if not artifact.is_file() or sha256(artifact) != expected_hash:
                errors.append(f"NormalFlow original frozen artifact mismatch: {relative}")
    if not recovery.is_file() or json.loads(recovery.read_text()).get("status") != "pass": errors.append("NormalFlow recovery proof invalid")
    return {"statuses": statuses, "normalflow_original_manifest": str(frozen), "normalflow_recovery": str(recovery)}


def check_benchmarks(root: Path, errors: list[str]) -> dict:
    result = {}
    for encoder in MODELS:
        path = root / "benchmark" / f"{encoder}.json"
        if not path.is_file(): errors.append(f"missing benchmark {encoder}"); continue
        data = json.loads(path.read_text()); samples = np.asarray(data.get("milliseconds", []), dtype=float)
        valid = (data.get("encoder") == encoder and data.get("batch_size") == 1 and data.get("warmups") == 20 and
                 data.get("repeats") == 100 and len(samples) == 100 and np.isfinite(samples).all() and np.all(samples > 0) and
                 all(math.isfinite(float(data.get(key, float("nan")))) for key in ("median_ms", "p90_ms", "min_ms", "max_ms", "peak_allocated_bytes")) and
                 same_number(data["median_ms"], np.median(samples)) and same_number(data["p90_ms"], np.quantile(samples, .9)) and
                 same_number(data["min_ms"], samples.min()) and same_number(data["max_ms"], samples.max()) and
                 Path(data.get("checkpoint", "")).is_file() and sha256(Path(data["checkpoint"])) == data.get("checkpoint_sha256") and
                 bool(data.get("scope")) and bool(data.get("exclusions")) and bool(data.get("torch")))
        if not valid: errors.append(f"invalid benchmark metadata/finite samples {encoder}")
        result[encoder] = {"path": str(path), "sha256": sha256(path), "pass": valid,
                           "median_ms": data.get("median_ms"), "p90_ms": data.get("p90_ms")}
    return {"expected": 4, "verified": sum(item["pass"] for item in result.values()), "items": result}


def check_reviewers(root: Path, code_root: Path, errors: list[str]) -> dict:
    candidates = [code_root / "encoders" / "review_normalflow.md", code_root / "encoders" / "review_root_packages.md",
                  code_root / "future" / "review_encoders.md"]
    rows = []
    for path in candidates:
        valid = path.is_file() and len(path.read_text(encoding="utf-8").strip()) >= 100
        if not valid: errors.append(f"missing/empty reviewer report {path}")
        rows.append({"path": str(path), "sha256": sha256(path) if path.is_file() else None, "pass": valid})
    return {"expected": len(candidates), "verified": sum(row["pass"] for row in rows), "reports": rows}


def check_synthesis_inputs(root: Path, code_root: Path, errors: list[str]) -> dict:
    future_path = root / "future" / "evaluation" / "evaluation.json"
    padding_path = root / "future" / "startup_padding_audit.json"
    nf_path = root / "normalflow" / "summary.json"
    summarize_path = code_root / "summarize.py"
    required = (future_path, padding_path, nf_path, summarize_path)
    if not all(path.is_file() for path in required):
        errors.append("missing synthesis input/source artifact")
        return {"pass": False}
    future = json.loads(future_path.read_text()); padding = json.loads(padding_path.read_text()); nf = json.loads(nf_path.read_text())
    source = summarize_path.read_text(encoding="utf-8")
    def future_mean(method: str, metric: str) -> float:
        rows = [row for row in future["results"] if row["method"] == method and row["threshold_rule"] == "max_balanced_accuracy"]
        return float(np.mean([row["validation"][metric] for row in rows]))
    values = {"mlp_ap": future_mean("mlp", "average_precision"), "gru_ap": future_mean("gru", "average_precision"),
              "current_slip_ap": future_mean("adapted_current_slip", "average_precision"),
              "mlp_fpr": future_mean("mlp", "fpr"), "gru_fpr": future_mean("gru", "fpr")}
    expected_rounded = {"mlp_ap": .4652, "gru_ap": .5832, "current_slip_ap": .1860, "mlp_fpr": .7869, "gru_fpr": .7322}
    if any(round(values[key], 4) != value for key, value in expected_rounded.items()): errors.append("hard-coded Future summary values differ from artifacts")
    support = future["support"]; support_ok = (support["validation"]["eligible_frames"] == 77 and
        support["validation"]["positive_frames"] == 16 and support["validation"]["positive_episodes"] == 6 and
        support["calibration"]["negative_frames"] == 56)
    if not support_ok: errors.append("hard-coded Future support values differ from artifacts")
    padding_ok = (padding.get("provenance", {}).get("evaluation_sha256") == sha256(future_path) and
        padding["roles_h8"]["validation"]["startup_t3_7"]["positive_frames"] == 8 and
        padding["roles_h8"]["validation"]["strict_t_ge_8"]["positive_frames"] == 8 and
        padding["strict_shortest_supported_horizon"] == 8 and
        all(row["strata"]["startup_t3_7"]["fpr"] == 1.0 for row in padding["validation_fixed_threshold_strata"]))
    if not padding_ok: errors.append("hard-coded startup-padding summary values differ from artifacts")
    persistence = nf["baseline"]["persistence"]["metrics"]["overall"]
    linear = nf["baseline"]["linear_ar_ridge_1e-3"]["metrics"]["overall"]
    c, d = nf["mean_by_variant"]["C"], nf["mean_by_variant"]["D"]
    horizons = ("1", "3", "5")
    nf_ok = (all(c[h] < persistence[h] for h in horizons) and
             [c[h] < linear[h] for h in horizons] == [False, True, False] and
             all(d[h] > c[h] for h in horizons) and
             {row["parameters"] for row in nf["runs"] if row["variant"] == "C"} == {642048} and
             {row["parameters"] for row in nf["runs"] if row["variant"] == "D"} == {644370})
    if not nf_ok: errors.append("hard-coded NormalFlow interpretation/parameter values differ from artifacts")
    literals = ("0.4652/0.5832", "0.1860", "78.7%/73.2%", "642,048/644,370", "9,216")
    if any(literal not in source for literal in literals): errors.append("summarize.py no longer contains the independently checked hard-coded values")
    caution_phrases = ("不能单独称为可靠预警", "不能把保留当前状态造成的低 MSE解释为", "不能仅凭低 MSE 或 AP 提升宣称完整世界模型有效")
    # Match semantically important cautions while allowing punctuation/spacing edits.
    if caution_phrases[0] not in source or caution_phrases[2] not in source or "不能把保留当前状态造成的低 MSE" not in source:
        errors.append("summarize.py is missing required non-overclaiming cautions")
    return {"pass": nf_ok and support_ok and padding_ok and all(round(values[k], 4) == v for k, v in expected_rounded.items()),
            "future": values, "future_support": support, "padding_pass": padding_ok,
            "normalflow": {"persistence": persistence, "linear": linear, "C": c, "D": d,
                           "interpretation": "C beats persistence at H1/3/5; C beats linear only at H3; D is worse than C at H1/3/5"}}


def audit(root: Path, r3_root: Path, code_root: Path, output: Path) -> dict:
    errors: list[str] = []
    identities = collect_identity_status(root, r3_root, errors)
    packages = check_package_audits(root, errors)
    alarms = check_alarm_runs(root, r3_root, errors)
    benchmarks = check_benchmarks(root, errors)
    reviewers = check_reviewers(root, code_root, errors)
    synthesis = check_synthesis_inputs(root, code_root, errors)
    payload = {"status": "pass" if not errors else "fail", "errors": errors, "identities": identities,
               "package_audits": packages, "alarms": alarms, "benchmarks": benchmarks, "reviewers": reviewers,
               "synthesis_review": synthesis,
               "scope": "CPU-only independent recomputation; no training or GPU execution",
               "source_sha256": sha256(Path(__file__).resolve())}
    atomic_json(output / "independent_audit.json", payload)
    lines = ["# R4 独立集成验收", "", f"状态：`{payload['status']}`", "",
             f"- 新神经运行：{identities['new_neural_found']}/48；复用 MAE：{identities['mae_reused_found']}/12。",
             f"- 告警身份：{alarms['found_identities']}/48；独立重算 raw role 指标 {alarms['raw_role_recomputations']} 组、sequential role 指标 {alarms['sequential_role_recomputations']} 组。",
             f"- 逐试次记录核对：{alarms['per_trial_rows_checked']} 行。",
             f"- 延迟 benchmark：{benchmarks['verified']}/4；reviewer reports：{reviewers['verified']}/{reviewers['expected']}。", ""]
    if errors:
        lines += ["## 未通过项", ""] + [f"- {error}" for error in errors] + [""]
    else:
        lines += ["所有正式身份、上游恢复/冻结证据、原始预测哈希、阈值约束、逐帧混淆、顺序告警状态、事件起点/延迟及逐试次汇总均通过独立检查。", ""]
    atomic_text(output / "REPORT_ZH.md", "\n".join(lines))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--r3-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root.resolve(), args.r3_root.resolve(), args.code_root.resolve(), args.output.resolve())
    print(json.dumps({"status": result["status"], "errors": len(result["errors"]), "identities": result["identities"], "alarms": result["alarms"]}, indent=2))
    raise SystemExit(result["status"] != "pass")


if __name__ == "__main__":
    main()
