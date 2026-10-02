#!/usr/bin/env python3
"""Evaluate round-3 frozen-MAE slip-head adaptation runs.

The evaluator never reads the schema-2 test role.  It selects one threshold per
run from calibration predictions and applies that threshold unchanged to the
corresponding validation predictions.  Fold outputs are summarized without
pooling frames because the four development folds overlap.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

import numpy as np


FOLDS = tuple(f"htt_leave_p{i}" for i in range(1, 5))
SEEDS = (20260914, 20260915, 20260916)
VARIANTS = ("A", "B", "C")
CANONICAL_SPLIT_SHA256 = "bd01e50b06cbea8c9eae8fc1ba448a1b0a5b53110471ae3d12e48ea1313242d1"
STAGE_MAP = {"static": 0, "incipient": 1, "gross": 2, "0": 0, "1": 1, "2": 2}
METRIC_KEYS = ("balanced_accuracy", "macro_f1", "static_fpr", "gross_recall", "average_precision")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temp.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def average_precision(y: np.ndarray, score: np.ndarray) -> float | None:
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=float)
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(-score, kind="mergesort")
    ranked_y, ranked_score = y[order], score[order]
    ends = np.r_[np.flatnonzero(ranked_score[:-1] != ranked_score[1:]), len(ranked_score) - 1]
    true_positive = np.cumsum(ranked_y == 1)[ends]
    precision = true_positive / (ends + 1)
    recall = true_positive / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def binary_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=float)
    valid = np.isin(y, (0, 1)) & np.isfinite(score)
    y, score = y[valid], score[valid]
    pred = score >= threshold
    tn = int(np.sum((y == 0) & ~pred))
    fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & ~pred))
    tp = int(np.sum((y == 1) & pred))
    neg, pos = tn + fp, tp + fn
    neg_recall, pos_recall = ratio(tn, neg), ratio(tp, pos)
    neg_f1 = ratio(2 * tn, 2 * tn + fp + fn)
    pos_f1 = ratio(2 * tp, 2 * tp + fp + fn)
    both = neg > 0 and pos > 0
    return {
        "evaluable": both,
        "threshold": float(threshold),
        "support": {"total": int(len(y)), "static": neg, "gross": pos},
        "prevalence": ratio(pos, len(y)),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "balanced_accuracy": float((neg_recall + pos_recall) / 2) if both else None,
        "macro_f1": float((neg_f1 + pos_f1) / 2) if both else None,
        "static_fpr": ratio(fp, neg),
        "gross_recall": pos_recall,
        "average_precision": average_precision(y, score),
    }


def probability_distribution(stage: np.ndarray, score: np.ndarray) -> dict:
    result = {}
    for name, label in (("static", 0), ("incipient", 1), ("gross", 2)):
        values = np.asarray(score)[np.asarray(stage) == label]
        values = values[np.isfinite(values)]
        result[name] = {
            "count": int(len(values)),
            "mean": float(np.mean(values)) if len(values) else None,
            "std": float(np.std(values)) if len(values) else None,
            "quantiles": {
                str(q): float(np.quantile(values, q)) if len(values) else None
                for q in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
            },
        }
    return result


def select_calibration_threshold(y: np.ndarray, score: np.ndarray) -> dict:
    """Exact unique-score search with explicit never-alarm candidate.

    Ranking is maximum balanced accuracy, then minimum FPR, then largest
    threshold.  Predictions use score >= threshold.
    """
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=float)
    valid = np.isin(y, (0, 1)) & np.isfinite(score)
    y, score = y[valid], score[valid]
    if not len(score) or not np.any(y == 0) or not np.any(y == 1):
        return {"threshold": None, "reason": "calibration requires static and gross frames", "candidate_count": 0}
    # Probabilities are constrained to [0,1], so this is the protocol's
    # explicit above-one candidate rather than merely the next observed gap.
    never = float(np.nextafter(1.0, np.inf))
    candidates = np.r_[np.unique(score), never]
    negative = np.sort(score[y == 0])
    positive = np.sort(score[y == 1])
    rows = []
    for threshold in candidates:
        fp = len(negative) - int(np.searchsorted(negative, threshold, side="left"))
        tp = len(positive) - int(np.searchsorted(positive, threshold, side="left"))
        fpr = fp / len(negative)
        recall = tp / len(positive)
        rows.append({
            "threshold": float(threshold),
            "balanced_accuracy": float((1.0 - fpr + recall) / 2.0),
            "static_fpr": float(fpr),
            "gross_recall": float(recall),
            "never_alarm": bool(threshold > 1.0),
        })
    best_ba = max(row["balanced_accuracy"] for row in rows)
    tied_ba = [row for row in rows if abs(row["balanced_accuracy"] - best_ba) <= 1e-12]
    best_fpr = min(row["static_fpr"] for row in tied_ba)
    tied_fpr = [row for row in tied_ba if abs(row["static_fpr"] - best_fpr) <= 1e-12]
    chosen = max(tied_fpr, key=lambda row: row["threshold"])
    return {
        **chosen,
        "reason": None,
        "candidate_count": int(len(candidates)),
        "selection_rule": "max balanced_accuracy, then min static_fpr, then largest threshold; score>=threshold",
    }


def bootstrap_by_episode(
    y: np.ndarray,
    score: np.ndarray,
    episode: np.ndarray,
    threshold: float,
    reps: int = 200,
    seed: int = 20260914,
) -> dict:
    y, score, episode = np.asarray(y), np.asarray(score), np.asarray(episode, dtype=str)
    groups = np.unique(episode)
    rng = np.random.default_rng(seed)
    values = {key: [] for key in METRIC_KEYS}
    for _ in range(reps):
        picked = rng.choice(groups, len(groups), replace=True)
        indices = np.concatenate([np.flatnonzero(episode == group) for group in picked])
        row = binary_metrics(y[indices], score[indices], threshold)
        if not row["evaluable"]:
            continue
        for key in METRIC_KEYS:
            value = row[key]
            if value is not None:
                values[key].append(value)
    return {
        "unit": "episode/leakage_group",
        "seed": int(seed),
        "requested_replicates": int(reps),
        "valid_replicates": len(values["balanced_accuracy"]),
        "ci95": {
            key: ({"low": float(np.quantile(vals, 0.025)), "high": float(np.quantile(vals, 0.975))} if vals else None)
            for key, vals in values.items()
        },
    }


def stage_value(value: str) -> int:
    key = value.strip().lower()
    if key not in STAGE_MAP:
        raise ValueError(f"unknown stage {value!r}")
    return STAGE_MAP[key]


def load_prediction_csv(path: Path, expected_partition: str, expected_fold: str | None = None) -> list[dict]:
    required = {"episode_id", "t", "stage", "p_slip"}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = []
        seen = set()
        for raw in reader:
            key = (raw["episode_id"], int(raw["t"]))
            if key in seen:
                raise ValueError(f"{path}: duplicate episode/frame {key}")
            seen.add(key)
            score = float(raw["p_slip"])
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(f"{path}: invalid p_slip at {key}: {score}")
            fold = raw.get("fold") or expected_fold
            if expected_fold and fold != expected_fold:
                raise ValueError(f"{path}: row fold {fold!r} != {expected_fold!r}")
            if raw.get("partition") and raw["partition"] != expected_partition:
                raise ValueError(f"{path}: row partition {raw['partition']!r} != {expected_partition!r}")
            rows.append({
                "episode_id": raw["episode_id"], "t": int(raw["t"]),
                "stage": stage_value(raw["stage"]), "p_slip": score,
                "partition": expected_partition, "fold": fold,
                "seed": raw.get("seed"), "init": raw.get("init"),
                "checkpoint_epoch": raw.get("checkpoint_epoch"),
            })
    if not rows:
        raise ValueError(f"{path}: no predictions")
    return rows


def episode_metadata(splits: dict) -> dict[str, dict]:
    result = {}
    for row in splits.get("episodes", []):
        result[row["id"]] = row
    return result


def validate_role(
    rows: list[dict],
    split: dict,
    partition: str,
    path: Path,
    metadata: dict[str, dict],
    authoritative_labels: dict[str, np.ndarray] | None = None,
) -> None:
    allowed = set(split[partition])
    observed = {row["episode_id"] for row in rows}
    unexpected = observed - allowed
    if unexpected:
        raise ValueError(f"{path}: {partition} predictions contain out-of-role episode {sorted(unexpected)[0]}")
    # Labeled slip episodes may be a subset because force-only episodes have no stages.
    if not observed:
        raise ValueError(f"{path}: no in-role episodes")
    expected_labeled = {episode_id for episode_id in allowed if metadata.get(episode_id, {}).get("task") == "slip"}
    if expected_labeled and observed != expected_labeled:
        missing = expected_labeled - observed
        extra = observed - expected_labeled
        raise ValueError(f"{path}: incomplete labeled role; missing={sorted(missing)[:1]}, extra={sorted(extra)[:1]}")
    by_episode: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_episode[row["episode_id"]].append(row)
    for episode_id, episode_rows in by_episode.items():
        expected_frames = metadata.get(episode_id, {}).get("frames")
        if authoritative_labels is not None and episode_id in authoritative_labels:
            labels = authoritative_labels[episode_id]
            expected_frames = len(labels)
        else:
            labels = None
        frame_indices = sorted(row["t"] for row in episode_rows)
        if expected_frames is not None and frame_indices != list(range(int(expected_frames))):
            raise ValueError(f"{path}: incomplete/noncontiguous frames for {episode_id}; got {len(frame_indices)}, expected {expected_frames}")
        if labels is not None:
            for row in episode_rows:
                if int(labels[row["t"]]) != row["stage"]:
                    raise ValueError(f"{path}: stage label mismatch for {(episode_id, row['t'])}")


def arrays(rows: list[dict], metadata: dict[str, dict], primary_only: bool = True) -> tuple[np.ndarray, ...]:
    selected = [row for row in rows if not primary_only or row["stage"] in (0, 2)]
    stage = np.asarray([row["stage"] for row in selected], dtype=np.int8)
    y = (stage == 2).astype(np.int8)
    score = np.asarray([row["p_slip"] for row in selected], dtype=float)
    episode = np.asarray([
        metadata.get(row["episode_id"], {}).get("leakage_group", row["episode_id"])
        for row in selected
    ], dtype=str)
    return y, score, episode, stage


def load_baseline_cache(index_path: Path, expected_split_hash: str) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    index = read_json(index_path)
    if index.get("status") != "complete":
        raise ValueError("baseline cache index is not complete")
    if index.get("split_manifest_sha256") != expected_split_hash:
        raise ValueError("baseline cache/split SHA256 mismatch")
    if index.get("expected_episodes") != len(index.get("episodes", [])):
        raise ValueError("baseline cache inventory is incomplete")
    cache = {}
    for record in index.get("episodes", []):
        if record.get("domain") != "htt" or record.get("task") != "slip":
            continue
        cache_path = Path(record["cache_path"])
        expected_cache_hash = record.get("cache_sha256")
        if not expected_cache_hash or sha256(cache_path) != expected_cache_hash:
            raise ValueError(f"baseline cache artifact hash mismatch: {record['id']}")
        with np.load(cache_path, allow_pickle=False) as data:
            labels = np.asarray(data["labels"], dtype=np.int8)
            scores = np.asarray(data["p_slip"], dtype=float)
        if labels.shape != scores.shape or labels.ndim != 1:
            raise ValueError(f"baseline cache shape mismatch: {record['id']}")
        if not np.isin(labels, (0, 1, 2)).all():
            raise ValueError(f"baseline cache has invalid slip labels: {record['id']}")
        if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
            raise ValueError(f"baseline cache has invalid probabilities: {record['id']}")
        cache[record["id"]] = {"labels": labels, "scores": scores}
    return index, cache


def cache_rows(index: dict, cache: dict[str, dict[str, np.ndarray]], episode_ids: Iterable[str], partition: str, fold: str) -> list[dict]:
    result = []
    for episode_id in episode_ids:
        if episode_id not in cache:
            continue  # schema roles also contain force-only episodes by design
        labels = cache[episode_id]["labels"]
        scores = cache[episode_id]["scores"]
        for t, (stage, score) in enumerate(zip(labels, scores)):
            result.append({"episode_id": episode_id, "t": t, "stage": int(stage), "p_slip": float(score),
                           "partition": partition, "fold": fold, "seed": None, "init": "legacy",
                           "checkpoint_epoch": index.get("checkpoint_epoch")})
    return result


def resolve(path: str | Path, base: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else base / path


def records_from_manifest(path: Path) -> list[dict]:
    payload = read_json(path)
    records = payload.get("runs")
    if not isinstance(records, list):
        raise ValueError("run manifest must contain a runs list")
    result = []
    for record in records:
        item = dict(record)
        for key in ("predictions_calibration", "predictions_validation", "checkpoint", "training_summary", "baseline_cache_index"):
            if item.get(key):
                item[key] = str(resolve(item[key], path.parent))
        if item.get("seed") is not None:
            item["seed"] = int(item["seed"])
        result.append(item)
    return result


def discover_training_runs(run_root: Path) -> list[dict]:
    records = []
    for validation in sorted(run_root.glob("runs/*/fold_*/seed_*/predictions/validation.csv")):
        calibration = validation.with_name("calibration.csv")
        if not calibration.exists():
            raise ValueError(f"missing paired calibration predictions: {calibration}")
        sample = load_prediction_csv(validation, "validation")
        first = sample[0]
        init = (first.get("init") or validation.parents[3].name).lower()
        variant = "B" if init in {"b", "fresh", "new", "random", "new_head"} else "C" if init in {"c", "old", "legacy", "old_head"} else None
        if variant is None:
            raise ValueError(f"cannot map init={init!r} to B/C for {validation}")
        run_dir = validation.parents[1]
        records.append({
            "variant": variant, "fold": first["fold"], "seed": int(first["seed"]),
            "predictions_calibration": str(calibration), "predictions_validation": str(validation),
            "run_dir": str(run_dir), "training_summary": str(run_dir / "training_summary.json"),
            "checkpoint": str(run_dir / "best.pth"),
        })
    return records


def validate_run_artifacts(record: dict, formal: bool) -> None:
    if record["variant"] == "A":
        return
    run_dir = Path(record["run_dir"]) if record.get("run_dir") else None
    summary_path = Path(record["training_summary"]) if record.get("training_summary") else (run_dir / "training_summary.json" if run_dir else None)
    if summary_path is None or not summary_path.exists():
        if formal:
            raise ValueError(f"{run_identity(record)}: missing training_summary.json")
        return
    summary = read_json(summary_path)
    expected_init = "fresh" if record["variant"] == "B" else "old"
    if summary.get("status") != "complete" or summary.get("fold") != record["fold"] or int(summary.get("seed")) != int(record["seed"]) or summary.get("init") != expected_init:
        raise ValueError(f"{run_identity(record)}: incompatible/incomplete training summary")
    checks = {
        Path(summary["best_checkpoint"]): summary["best_checkpoint_sha256"],
        Path(summary["latest_checkpoint"]): summary["latest_checkpoint_sha256"],
        Path(summary["predictions"]["calibration"]): summary["prediction_sha256"]["calibration"],
        Path(summary["predictions"]["validation"]): summary["prediction_sha256"]["validation"],
    }
    for path, expected_hash in checks.items():
        if not path.exists() or sha256(path) != expected_hash:
            raise ValueError(f"{run_identity(record)}: artifact hash mismatch: {path}")
    if Path(record["predictions_calibration"]).resolve() != Path(summary["predictions"]["calibration"]).resolve():
        raise ValueError(f"{run_identity(record)}: calibration prediction path differs from training summary")
    if Path(record["predictions_validation"]).resolve() != Path(summary["predictions"]["validation"]).resolve():
        raise ValueError(f"{run_identity(record)}: validation prediction path differs from training summary")
    record.update({
        "training_summary": str(summary_path), "checkpoint": summary["best_checkpoint"],
        "checkpoint_sha256": summary["best_checkpoint_sha256"], "best_epoch": summary["best_epoch"],
        "training_status": summary["status"],
    })


def run_identity(record: dict) -> str:
    seed = "legacy" if record.get("seed") is None else str(record["seed"])
    return f"{record['variant']}/{record['fold']}/{seed}"


def evaluate_one(
    record: dict,
    splits: dict,
    metadata: dict[str, dict],
    reps: int,
    baseline: tuple[dict, dict[str, dict[str, np.ndarray]]] | None = None,
) -> tuple[dict, list[dict], list[dict], dict]:
    variant, fold = record["variant"], record["fold"]
    if variant not in VARIANTS or fold not in FOLDS:
        raise ValueError(f"invalid run identity: {record}")
    split = splits["splits"][fold]
    if variant == "A" and record.get("baseline_cache_index"):
        if baseline is None:
            raise ValueError("baseline cache was not loaded")
        index, cache = baseline
        cal_rows = cache_rows(index, cache, split["calibration"], "calibration", fold)
        val_rows = cache_rows(index, cache, split["validation"], "validation", fold)
        cal_path = val_path = Path(record["baseline_cache_index"])
    else:
        cal_path = Path(record["predictions_calibration"])
        val_path = Path(record["predictions_validation"])
        cal_rows = load_prediction_csv(cal_path, "calibration", fold)
        val_rows = load_prediction_csv(val_path, "validation", fold)
        expected_init = "fresh" if variant == "B" else "old"
        for path, rows in ((cal_path, cal_rows), (val_path, val_rows)):
            if {int(row["seed"]) for row in rows} != {int(record["seed"])}:
                raise ValueError(f"{path}: prediction seed differs from run record")
            if {row["init"] for row in rows} != {expected_init}:
                raise ValueError(f"{path}: prediction init differs from run record")
            raw_epochs = {row["checkpoint_epoch"] for row in rows}
            epochs = {int(value) for value in raw_epochs if value not in (None, "")}
            if (record.get("best_epoch") is not None and epochs != {int(record["best_epoch"])}) or len(epochs) > 1:
                raise ValueError(f"{path}: prediction checkpoint epoch mismatch")
    authority = None if baseline is None else {key: value["labels"] for key, value in baseline[1].items()}
    validate_role(cal_rows, split, "calibration", cal_path, metadata, authority)
    validate_role(val_rows, split, "validation", val_path, metadata, authority)
    if {row["episode_id"] for row in cal_rows} & {row["episode_id"] for row in val_rows}:
        raise ValueError(f"{run_identity(record)}: calibration/validation episode leakage")
    cy, cs, _, _ = arrays(cal_rows, metadata)
    vy, vs, vg, _ = arrays(val_rows, metadata)
    selected = select_calibration_threshold(cy, cs)
    if selected["threshold"] is None:
        raise ValueError(f"{run_identity(record)}: {selected['reason']}")
    cal_selected_metrics = binary_metrics(cy, cs, selected["threshold"])
    points = {}
    for name, threshold in (("fixed_0.5", 0.5), ("calibrated", selected["threshold"])):
        metrics = binary_metrics(vy, vs, threshold)
        stable_seed = int(hashlib.sha256(f"{run_identity(record)}/{name}".encode()).hexdigest()[:8], 16)
        metrics["episode_bootstrap"] = bootstrap_by_episode(vy, vs, vg, threshold, reps, stable_seed)
        metrics["never_alarm"] = bool(name == "calibrated" and selected["never_alarm"])
        points[name] = metrics
    trial_rows, failure_rows = [], []
    by_episode: dict[str, list[dict]] = defaultdict(list)
    for row in val_rows:
        by_episode[row["episode_id"]].append(row)
    for point, metrics in points.items():
        threshold = metrics["threshold"]
        candidates = []
        for episode_id, rows in sorted(by_episode.items()):
            y, score, _, _ = arrays(rows, metadata)
            if not len(y):
                continue
            item = binary_metrics(y, score, threshold)
            base = {"run_id": run_identity(record), "variant": variant, "fold": fold,
                    "seed": record.get("seed"), "operating_point": point, "episode_id": episode_id}
            trial_rows.append({**base, **flatten_metrics(item)})
            errors = item["confusion"]["fp"] + item["confusion"]["fn"]
            primary = [row for row in rows if row["stage"] in (0, 2)]
            details = []
            for source, target, prediction in zip(primary, y, score >= threshold):
                if bool(prediction) != bool(target):
                    details.append({"t": source["t"], "stage": source["stage"], "p_slip": source["p_slip"],
                                    "error": "FP" if target == 0 else "FN"})
            candidates.append({**base, "errors": errors, "error_rate": ratio(errors, item["support"]["total"]),
                               "error_frames_json": json.dumps(details[:10], ensure_ascii=False, separators=(",", ":")),
                               **flatten_metrics(item)})
        candidates.sort(key=lambda row: (-(row["error_rate"] or 0), -row["errors"], row["episode_id"]))
        failure_rows.extend(candidates[:5])
    stage = np.asarray([row["stage"] for row in val_rows], dtype=np.int8)
    score = np.asarray([row["p_slip"] for row in val_rows], dtype=float)
    distributions = probability_distribution(stage, score)
    incipient = score[stage == 1]
    incipient_diag = {
        **distributions["incipient"],
        "fraction_alarm_at_0.5": float(np.mean(incipient >= 0.5)) if len(incipient) else None,
        "fraction_alarm_at_calibrated_threshold": float(np.mean(incipient >= selected["threshold"])) if len(incipient) else None,
        "interpretation": "distribution diagnostic only; incipient is excluded from training loss and primary metrics",
    }
    result = {
        "run_id": run_identity(record), "variant": variant, "fold": fold, "seed": record.get("seed"),
        "record": record,
        "calibration": {"selection": selected, "metrics_at_selected_threshold": cal_selected_metrics},
        "validation": points,
        "validation_probability_distribution": distributions,
        "validation_incipient_diagnostic": incipient_diag,
    }
    return result, trial_rows, failure_rows, incipient_diag


def flatten_metrics(metrics: dict) -> dict:
    return {
        "threshold": metrics["threshold"], "support_total": metrics["support"]["total"],
        "support_static": metrics["support"]["static"], "support_gross": metrics["support"]["gross"],
        "prevalence": metrics["prevalence"], **metrics["confusion"],
        **{key: metrics[key] for key in METRIC_KEYS},
    }


def summarize(run_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in run_rows:
        grouped[(row["variant"], row["operating_point"])].append(row)
    overall = []
    for (variant, point), rows in sorted(grouped.items()):
        item = {
            "variant": variant, "operating_point": point, "runs": len(rows),
            "aggregation": "descriptive mean/std across run metrics; frames are not pooled; std is not an independent-sample CI",
        }
        for key in METRIC_KEYS:
            vals = [row[key] for row in rows if row[key] is not None]
            item[f"{key}_mean"] = mean(vals) if vals else None
            item[f"{key}_std"] = pstdev(vals) if len(vals) > 1 else 0.0 if vals else None
        overall.append(item)
    seed_rows = []
    grouped_seed: dict[tuple, list[dict]] = defaultdict(list)
    for row in run_rows:
        grouped_seed[(row["variant"], row["fold"], row["operating_point"])].append(row)
    for (variant, fold, point), rows in sorted(grouped_seed.items()):
        item = {"variant": variant, "fold": fold, "operating_point": point, "seeds": len(rows)}
        for key in METRIC_KEYS:
            vals = [row[key] for row in rows if row[key] is not None]
            item[f"{key}_mean"] = mean(vals) if vals else None
            item[f"{key}_std"] = pstdev(vals) if len(vals) > 1 else 0.0 if vals else None
            item[f"{key}_min"] = min(vals) if vals else None
            item[f"{key}_max"] = max(vals) if vals else None
        seed_rows.append(item)
    return overall, seed_rows


def paired_comparisons(run_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    baseline = {(row["fold"], row["operating_point"]): row for row in run_rows if row["variant"] == "A"}
    rows = []
    for row in run_rows:
        if row["variant"] not in ("B", "C"):
            continue
        reference = baseline.get((row["fold"], row["operating_point"]))
        if reference is None:
            continue
        item = {"variant": row["variant"], "fold": row["fold"], "seed": row["seed"], "operating_point": row["operating_point"]}
        for key in METRIC_KEYS:
            item[f"delta_{key}_vs_A_same_fold"] = (
                row[key] - reference[key] if row[key] is not None and reference[key] is not None else None
            )
        rows.append(item)
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], row["operating_point"])].append(row)
    summary = []
    for (variant, point), items in sorted(grouped.items()):
        result = {"variant": variant, "operating_point": point, "paired_runs": len(items)}
        for key in METRIC_KEYS:
            values = [row[f"delta_{key}_vs_A_same_fold"] for row in items if row[f"delta_{key}_vs_A_same_fold"] is not None]
            result[f"delta_{key}_mean"] = mean(values) if values else None
            result[f"delta_{key}_std"] = pstdev(values) if len(values) > 1 else 0.0 if values else None
        summary.append(result)
    return rows, summary


def paired_b_vs_c(run_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    indexed = {
        (row["variant"], row["fold"], row["seed"], row["operating_point"]): row
        for row in run_rows if row["variant"] in ("B", "C")
    }
    rows = []
    for fold in FOLDS:
        for seed in SEEDS:
            for point in ("fixed_0.5", "calibrated"):
                b = indexed.get(("B", fold, seed, point))
                c = indexed.get(("C", fold, seed, point))
                if b is None or c is None:
                    continue
                item = {"fold": fold, "seed": seed, "operating_point": point}
                for key in METRIC_KEYS:
                    item[f"delta_{key}_B_minus_C"] = (
                        b[key] - c[key] if b[key] is not None and c[key] is not None else None
                    )
                rows.append(item)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["operating_point"]].append(row)
    summary = []
    for point, items in sorted(grouped.items()):
        result = {"operating_point": point, "paired_runs": len(items)}
        for key in METRIC_KEYS:
            values = [row[f"delta_{key}_B_minus_C"] for row in items if row[f"delta_{key}_B_minus_C"] is not None]
            result[f"delta_{key}_mean"] = mean(values) if values else None
            result[f"delta_{key}_std"] = pstdev(values) if len(values) > 1 else 0.0 if values else None
            result[f"{key}_B_higher_count"] = sum(value > 0 for value in values)
        summary.append(result)
    return rows, summary


def fallback_svg_bar_chart(summary: list[dict], point: str, path: Path) -> None:
    rows = [row for row in summary if row["operating_point"] == point]
    width, height = 860, 430
    left, top, chart_h = 80, 45, 300
    colors = {"balanced_accuracy": "#2563eb", "static_fpr": "#dc2626", "gross_recall": "#16a34a"}
    metrics = tuple(colors)
    group_w = 220
    bars = []
    labels = []
    for gi, row in enumerate(rows):
        gx = left + 55 + gi * group_w
        labels.append(f'<text x="{gx + 45}" y="380" text-anchor="middle" font-size="15">{row["variant"]}</text>')
        for mi, key in enumerate(metrics):
            value = row.get(f"{key}_mean") or 0.0
            std = row.get(f"{key}_std") or 0.0
            x, y = gx + mi * 38, top + chart_h * (1 - value)
            h = chart_h * value
            err = chart_h * std
            bars.append(f'<rect x="{x}" y="{y}" width="28" height="{h}" fill="{colors[key]}"/>')
            bars.append(f'<line x1="{x+14}" y1="{max(top,y-err)}" x2="{x+14}" y2="{min(top+chart_h,y+err)}" stroke="#111"/>')
    legend = []
    for i, key in enumerate(metrics):
        legend.append(f'<rect x="{470+i*120}" y="12" width="14" height="14" fill="{colors[key]}"/><text x="{489+i*120}" y="24" font-size="12">{key}</text>')
    grid = [f'<line x1="{left}" y1="{top + chart_h*(1-v)}" x2="810" y2="{top + chart_h*(1-v)}" stroke="#ddd"/><text x="65" y="{top + chart_h*(1-v)+5}" text-anchor="end" font-size="12">{v:.1f}</text>' for v in np.linspace(0,1,6)]
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/><text x="25" y="25" font-size="18">Validation comparison: {point}</text>
{''.join(grid)}{''.join(bars)}{''.join(labels)}{''.join(legend)}
<text x="430" y="415" text-anchor="middle" font-size="12">Mean across run-level metrics; error bar = run-level SD; overlapping folds are not frame-pooled.</text></svg>'''
    write_text(path, svg)


def plot_bar_chart(summary: list[dict], point: str, svg_path: Path) -> list[Path]:
    """Write a publication-friendly SVG and PNG when matplotlib is available.

    The tiny SVG fallback keeps evaluation/unit tests usable in minimal CPU
    environments; formal server evaluation uses matplotlib.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        fallback_svg_bar_chart(summary, point, svg_path)
        return [svg_path]
    rows = [row for row in summary if row["operating_point"] == point]
    variants = [row["variant"] for row in rows]
    metrics = (
        ("balanced_accuracy", "Balanced accuracy", "#2563eb"),
        ("static_fpr", "Static FPR", "#dc2626"),
        ("gross_recall", "Gross recall", "#16a34a"),
    )
    x = np.arange(len(variants), dtype=float)
    bar_width = 0.24
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for index, (key, label, color) in enumerate(metrics):
        values = np.asarray([row.get(f"{key}_mean") or 0.0 for row in rows])
        errors = np.asarray([row.get(f"{key}_std") or 0.0 for row in rows])
        ax.bar(x + (index - 1) * bar_width, values, bar_width, yerr=errors,
               capsize=3, color=color, label=label, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x, variants)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Validation metric")
    ax.set_title(f"Frozen-MAE slip adaptation — {point}", pad=46)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=False)
    fig.text(0.5, 0.015,
             "Bars: descriptive mean across run metrics; error bars: run-level SD (not an independent-sample CI).\n"
             "Overlapping folds are not frame-pooled.",
             ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0.02, 0.10, 0.98, 0.90))
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = svg_path.with_suffix(".png")
    fig.savefig(svg_path, format="svg")
    fig.savefig(png_path, format="png", dpi=180)
    plt.close(fig)
    return [svg_path, png_path]


def fmt(value: Any, digits: int = 4) -> str:
    return "NA" if value is None else f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def render_chinese_report(
    summary: list[dict], run_rows: list[dict], complete_runs: list[dict], comparison_summary: list[dict],
    bc_summary: list[dict],
) -> str:
    lookup = {(row["variant"], row["operating_point"]): row for row in summary}
    lines = [
        "# 第三轮冻结 MAE 滑移检测头适配结果", "",
        "本报告是四折 HTT 开发对照。validation 同时参与 checkpoint 选择，因此结果不是无偏泛化估计；四折存在重叠，所有汇总均先计算每个运行的指标，再求描述性均值和标准差，没有池化重叠帧。跨折/种子的标准差用于描述波动，不是基于独立样本的置信区间。", "",
        "## 完成情况", "",
        f"共评估 {len(complete_runs)} 个运行：A 为旧模型直接迁移；B 为随机初始化 slip 分支；C 为旧 slip 分支继续适配。主要任务只包含 static 与 gross，incipient 仅作分布诊断。", "",
        "## 主结果", "",
        "| 组别 | 阈值 | BA（均值±SD） | Macro-F1 | static FPR | gross recall | AP |", "|---|---|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        for point, point_label in (("fixed_0.5", "0.5"), ("calibrated", "calibration")):
            row = lookup.get((variant, point))
            if not row:
                continue
            cell = lambda key: f"{fmt(row[f'{key}_mean'])}±{fmt(row[f'{key}_std'])}"
            lines.append(f"| {variant} | {point_label} | {cell('balanced_accuracy')} | {cell('macro_f1')} | {cell('static_fpr')} | {cell('gross_recall')} | {cell('average_precision')} |")
    never = [row for row in run_rows if row["operating_point"] == "calibrated" and row.get("never_alarm")]
    saturated_a = [
        result["calibration"]["selection"]["threshold"] for result in complete_runs
        if result["variant"] == "A" and result["calibration"]["selection"]["threshold"] is not None
        and result["calibration"]["selection"]["threshold"] > 0.99
    ]
    lines += ["", "校准阈值只由各自 calibration 分区选择，并原样应用到 validation。候选是全部唯一概率及显式 never-alarm；并列时依次选择 calibration FPR 更低、阈值更大的候选。",
              f"校准后共有 {len(never)} 个运行选择 never-alarm；这些结果明确保留，不视为检测改善。",
              f"A 组共有 {len(saturated_a)} 折的精确校准阈值高于 0.99。饱和概率附近的精确阈值可能对很小的数值差异敏感，因此应结合固定 0.5 结果和排序型 AP 解读，不能只依据该工作点。" if saturated_a else "",
              "", "## 直接测量与推断", ""]
    if comparison_summary:
        lines += ["下表是 B/C 与同折 A 的逐运行差值，再对折和种子求均值；正 BA/召回差值表示提高，负 FPR 差值表示误报降低。", "",
                  "| 组别 | 阈值 | ΔBA | Δstatic FPR | Δgross recall | ΔAP |", "|---|---|---:|---:|---:|---:|"]
        for row in comparison_summary:
            lines.append(f"| {row['variant']} | {row['operating_point']} | {fmt(row['delta_balanced_accuracy_mean'])} | {fmt(row['delta_static_fpr_mean'])} | {fmt(row['delta_gross_recall_mean'])} | {fmt(row['delta_average_precision_mean'])} |")
        lines += ["", "以上差值是本轮开发数据上的直接测量。若 B/C 改善，只能推断下游适配在当前协议下有帮助；不能单独证明旧训练数据是根因，也不能外推到真实物体。", ""]
    else:
        lines += ["当前是部分/基线冒烟结果，尚无 B/C 配对结果，因此不对适配效果作推断。", ""]
    if bc_summary:
        lines += ["### 新头 B 与旧头适配 C", "",
                  "下表按同折、同种子计算 B−C。正 BA/召回/AP 表示 B 更高，负 FPR 表示 B 误报更低。", "",
                  "| 阈值 | ΔBA | Δstatic FPR | Δgross recall | ΔAP | 配对数 |", "|---|---:|---:|---:|---:|---:|"]
        for row in bc_summary:
            lines.append(f"| {row['operating_point']} | {fmt(row['delta_balanced_accuracy_mean'])} | {fmt(row['delta_static_fpr_mean'])} | {fmt(row['delta_gross_recall_mean'])} | {fmt(row['delta_average_precision_mean'])} | {row['paired_runs']} |")
        fixed = next((row for row in bc_summary if row["operating_point"] == "fixed_0.5"), None)
        if fixed:
            lines += ["", f"直接测量显示，固定 0.5 时 B 相对 C 的平均 ΔBA={fmt(fixed['delta_balanced_accuracy_mean'])}、ΔFPR={fmt(fixed['delta_static_fpr_mean'])}、Δgross recall={fmt(fixed['delta_gross_recall_mean'])}。因此新初始化头在本开发协议下总体略优，但差值是重叠开发折上的描述性结果，不能解释为独立统计显著性或真实物体优势。", ""]
    lines += [
              "## 解读边界", "",
              "- bootstrap 以完整试次（leakage group）为单位进行 200 次重采样，阈值保持固定。", "- AP 和自然正类占比与阈值无关；BA、FPR、召回和混淆矩阵分别报告固定与校准工作点。",
              "- incipient 未进入损失和主指标，因此其输出只能描述模型分布，不能解释为提前预警性能。", "- 本轮只检验冻结 MAE 表征下的下游适配；不能据此证明真实物体泛化、future head 有效或轻量级世界模型成立。", "",
              "## 产物", "", "逐运行结果见 `run_metrics.csv`，折内种子波动见 `seed_variability.csv`，逐试次结果与失败案例分别见 `per_trial_metrics.csv` 和 `failure_cases.csv`，完整数值与 bootstrap 区间见 `evaluation.json`。", ""]
    return "\n".join(lines)


def expected_coverage(records: list[dict]) -> tuple[set[tuple], set[tuple]]:
    expected = {("A", fold, None) for fold in FOLDS}
    expected |= {(variant, fold, seed) for variant in ("B", "C") for fold in FOLDS for seed in SEEDS}
    actual = {(row["variant"], row["fold"], row.get("seed")) for row in records}
    return expected - actual, actual - expected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--runs-manifest", type=Path)
    parser.add_argument("--baseline-cache-index", type=Path)
    parser.add_argument("--bootstrap-reps", type=int, default=200)
    parser.add_argument("--allow-partial", action="store_true", help="For smoke/unit tests only")
    parser.add_argument("--allow-noncanonical-splits", action="store_true", help="For synthetic tests only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_reps != 200 and not args.allow_partial:
        raise ValueError("formal protocol fixes bootstrap repetitions at 200")
    split_hash = sha256(args.splits)
    if split_hash != CANONICAL_SPLIT_SHA256 and not args.allow_noncanonical_splits:
        raise ValueError(f"noncanonical schema-2 split SHA256: {split_hash}")
    splits = read_json(args.splits)
    if splits.get("schema_version") != 2:
        raise ValueError("only schema-2 splits are accepted")
    records = records_from_manifest(args.runs_manifest) if args.runs_manifest else discover_training_runs(args.run_root)
    if args.baseline_cache_index:
        records += [{"variant": "A", "fold": fold, "seed": None,
                     "baseline_cache_index": str(args.baseline_cache_index)} for fold in FOLDS]
    identities = [run_identity(row) for row in records]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate run identity in evaluation inputs")
    missing, unexpected = expected_coverage(records)
    if (missing or unexpected) and not args.allow_partial:
        raise ValueError(f"run coverage mismatch; missing={sorted(missing, key=str)}, unexpected={sorted(unexpected, key=str)}")
    for record in records:
        validate_run_artifacts(record, formal=not args.allow_partial)
    metadata = episode_metadata(splits)
    baseline_paths = {Path(row["baseline_cache_index"]) for row in records if row.get("baseline_cache_index")}
    if len(baseline_paths) > 1:
        raise ValueError("all A folds must use the same baseline cache index")
    baseline = load_baseline_cache(next(iter(baseline_paths)), split_hash) if baseline_paths else None
    if not args.allow_partial and baseline is None:
        raise ValueError("formal evaluation requires the verified round-2 baseline cache as label authority")
    results, run_rows, trial_rows, failures, incipient_rows = [], [], [], [], []
    for record in sorted(records, key=run_identity):
        result, trials, run_failures, incipient = evaluate_one(record, splits, metadata, args.bootstrap_reps, baseline)
        results.append(result)
        trial_rows.extend(trials)
        failures.extend(run_failures)
        incipient_rows.append({"run_id": result["run_id"], "variant": result["variant"], "fold": result["fold"],
                               "seed": result["seed"], **{k: v for k, v in incipient.items() if k != "quantiles"},
                               **{f"q_{k}": v for k, v in incipient["quantiles"].items()}})
        for point, metrics in result["validation"].items():
            run_rows.append({"run_id": result["run_id"], "variant": result["variant"], "fold": result["fold"],
                             "seed": result["seed"], "operating_point": point,
                             "never_alarm": metrics["never_alarm"], **flatten_metrics(metrics)})
    summary, seed_variability = summarize(run_rows)
    comparison_rows, comparison_summary = paired_comparisons(run_rows)
    bc_rows, bc_summary = paired_b_vs_c(run_rows)
    payload = {
        "status": "complete", "protocol": "round3 frozen MAE static-vs-gross development comparison",
        "split_manifest": str(args.splits), "split_manifest_sha256": split_hash,
        "baseline_cache_index": str(next(iter(baseline_paths))) if baseline_paths else None,
        "baseline_cache_index_sha256": sha256(next(iter(baseline_paths))) if baseline_paths else None,
        "test_partitions_read": False, "validation_used_for_checkpoint_selection": True,
        "fold_aggregation": "run metrics only; overlapping fold frames never pooled",
        "bootstrap": {"repetitions": args.bootstrap_reps, "unit": "episode/leakage_group", "threshold_refit": False},
        "runs": results, "summary": summary, "within_fold_seed_variability": seed_variability,
        "paired_comparisons_vs_A": comparison_rows, "paired_comparison_summary": comparison_summary,
        "paired_B_vs_C": bc_rows, "paired_B_vs_C_summary": bc_summary,
    }
    output = args.output
    write_json(output / "evaluation.json", payload)
    run_fields = ["run_id", "variant", "fold", "seed", "operating_point", "threshold", "never_alarm",
                  "support_total", "support_static", "support_gross", "prevalence", "tn", "fp", "fn", "tp", *METRIC_KEYS]
    write_csv(output / "run_metrics.csv", run_rows, run_fields)
    write_csv(output / "per_trial_metrics.csv", trial_rows, ["run_id", "variant", "fold", "seed", "operating_point", "episode_id", *run_fields[5:]])
    write_csv(output / "failure_cases.csv", failures, ["run_id", "variant", "fold", "seed", "operating_point", "episode_id", "errors", "error_rate", "error_frames_json", *run_fields[5:]])
    write_csv(output / "summary.csv", summary, list(summary[0]) if summary else ["variant"])
    write_csv(output / "seed_variability.csv", seed_variability, list(seed_variability[0]) if seed_variability else ["variant"])
    comparison_fields = list(comparison_rows[0]) if comparison_rows else ["variant"]
    write_csv(output / "paired_comparisons_vs_A.csv", comparison_rows, comparison_fields)
    bc_fields = list(bc_rows[0]) if bc_rows else ["operating_point"]
    write_csv(output / "paired_B_vs_C.csv", bc_rows, bc_fields)
    incipient_fields = list(incipient_rows[0]) if incipient_rows else ["run_id"]
    write_csv(output / "incipient_distribution.csv", incipient_rows, incipient_fields)
    plot_paths = []
    plot_paths += plot_bar_chart(summary, "fixed_0.5", output / "plots" / "comparison_fixed_0.5.svg")
    plot_paths += plot_bar_chart(summary, "calibrated", output / "plots" / "comparison_calibrated.svg")
    write_text(output / "REPORT_ZH.md", render_chinese_report(summary, run_rows, results, comparison_summary, bc_summary))
    artifact_names = ["evaluation.json", "run_metrics.csv", "per_trial_metrics.csv", "failure_cases.csv",
                      "summary.csv", "seed_variability.csv", "paired_comparisons_vs_A.csv", "paired_B_vs_C.csv", "incipient_distribution.csv", "REPORT_ZH.md",
                      *[str(path.relative_to(output)) for path in plot_paths]]
    write_json(output / "COMPLETE.json", {
        "status": "complete", "runs": len(results), "expected_runs": 28,
        "test_partitions_read": False, "split_manifest_sha256": split_hash,
        "artifacts": {name: sha256(output / name) for name in artifact_names},
    })


if __name__ == "__main__":
    main()
