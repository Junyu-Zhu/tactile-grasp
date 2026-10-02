#!/usr/bin/env python3
"""Evaluate cached frozen predictions without reading held-out test partitions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from metrics import (
    THRESHOLD_GRID,
    binary_metrics,
    event_metrics,
    future_event_support,
    group_bootstrap,
    json_ready,
    pearson,
    select_balanced_threshold,
    select_fpr_threshold,
    threshold_curve,
)


HORIZONS = (1, 3, 5)
POPULATIONS = {"current_static": 0, "current_incipient": 1}
DETECTION_TARGETS = {
    "static_vs_gross": {"keep": (0, 2), "positive": (2,)},
    "static_vs_any_slip": {"keep": (0, 1, 2), "positive": (1, 2)},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    tmp.replace(path)


def domain_name(value: str) -> str:
    value = value.lower()
    if value.startswith("htt"):
        return "htt"
    if value.startswith("normalflow"):
        return "normalflow"
    return value


def future_horizons(index: dict) -> tuple[int, ...]:
    candidates = [index.get("horizons"), index.get("future_horizons")]
    if isinstance(index.get("future_head"), dict):
        candidates.append(index["future_head"].get("horizons"))
    for value in candidates:
        if value is not None:
            result = tuple(int(v) for v in value)
            if result != HORIZONS:
                raise ValueError(f"Expected p_future columns {HORIZONS}, got {result}")
            return result
    raise ValueError("Cache index must explicitly record p_future horizon order")


def load_cache(index_path: Path) -> tuple[dict, dict[str, dict]]:
    index = read_json(index_path)
    future_horizons(index)
    required_top = ("split_manifest_sha256", "checkpoint_sha256", "preprocessing_fingerprint")
    missing_top = [key for key in required_top if not index.get(key)]
    if missing_top:
        raise ValueError(f"Cache index missing provenance: {missing_top}")
    episodes: dict[str, dict] = {}
    for row in index.get("episodes", []):
        episode_id = row["id"]
        if episode_id in episodes:
            raise ValueError(f"Duplicate cache episode id: {episode_id}")
        cache_path = Path(row["cache_path"])
        if not cache_path.is_absolute():
            cache_path = index_path.parent / cache_path
        with np.load(cache_path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        required = ("z", "force_pred", "p_slip", "p_future", "labels", "pose", "time", "frame_index")
        missing = [key for key in required if key not in arrays]
        if missing:
            raise ValueError(f"{episode_id}: missing cache arrays {missing}")
        n = int(row["frames"])
        for key in required:
            if len(arrays[key]) != n:
                raise ValueError(f"{episode_id}: {key} has {len(arrays[key])} rows, expected {n}")
        if arrays["z"].ndim != 2 or arrays["z"].shape[1] != 768:
            raise ValueError(f"{episode_id}: z must be [T,768]")
        if arrays["force_pred"].shape != (n, 3):
            raise ValueError(f"{episode_id}: force_pred must be [T,3]")
        if arrays["p_slip"].shape != (n,):
            raise ValueError(f"{episode_id}: p_slip must be [T]")
        if arrays["p_future"].shape != (n, 3):
            raise ValueError(f"{episode_id}: p_future must be [T,3]")
        if arrays["pose"].shape != (n, 4, 4) or arrays["time"].shape != (n,):
            raise ValueError(f"{episode_id}: pose/time shape mismatch")
        for key in ("p_slip", "p_future"):
            finite_values = arrays[key][np.isfinite(arrays[key])]
            if np.any((finite_values < 0) | (finite_values > 1)):
                raise ValueError(f"{episode_id}: {key} is not a probability")
        if not np.array_equal(arrays["frame_index"], np.arange(n)):
            raise ValueError(f"{episode_id}: noncontiguous frame_index")
        labels = arrays["labels"].astype(np.int8)
        if not np.isin(labels, [-1, 0, 1, 2]).all():
            raise ValueError(f"{episode_id}: invalid labels")
        episodes[episode_id] = {
            **row,
            **arrays,
            "id": episode_id,
            "domain": domain_name(row["domain"]),
            "cache_path": str(cache_path),
        }
    return index, episodes


def selected_episodes(ids: list[str], cache: dict[str, dict], expected_domain: str) -> list[dict]:
    missing = [episode_id for episode_id in ids if episode_id not in cache]
    if missing:
        raise ValueError(f"Cache lacks {len(missing)} selected episodes, first={missing[0]}")
    rows = [cache[episode_id] for episode_id in ids]
    wrong = [row["id"] for row in rows if row["domain"] != expected_domain]
    if wrong:
        raise ValueError(f"Unexpected domain in split: {wrong[0]}")
    return rows


def detection_arrays(episodes: list[dict], definition: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys, scores, groups = [], [], []
    for episode in episodes:
        labels = np.asarray(episode["labels"])
        score = np.asarray(episode["p_slip"], float)
        mask = np.isin(labels, definition["keep"]) & np.isfinite(score)
        ys.append(np.isin(labels[mask], definition["positive"]).astype(np.int8))
        scores.append(score[mask])
        groups.extend([episode.get("leakage_group", episode["id"])] * int(mask.sum()))
    return np.concatenate(ys), np.concatenate(scores), np.asarray(groups, str)


def detection_breakdowns(episodes: list[dict], definition: dict, threshold: float) -> dict:
    by_episode = {}
    by_probe: dict[str, list[dict]] = {}
    for episode in episodes:
        y, score, _ = detection_arrays([episode], definition)
        by_episode[episode["id"]] = binary_metrics(y, score, threshold)
        by_probe.setdefault(episode["group"], []).append(episode)
    probe_rows = {}
    for probe, rows in sorted(by_probe.items()):
        y, score, _ = detection_arrays(rows, definition)
        probe_rows[probe] = binary_metrics(y, score, threshold)
    return {"by_episode": by_episode, "by_probe": probe_rows}


def first_gross_frame_arrays(episodes: list[dict], horizon: int, population: int, score_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys, scores, groups = [], [], []
    for episode in episodes:
        labels = np.asarray(episode["labels"])
        score = np.asarray(episode[score_key], float)
        gross = np.flatnonzero(labels == 2)
        onset = int(gross[0]) if len(gross) else len(labels)
        t = np.arange(len(labels))
        mask = (t < onset) & (t + horizon < len(labels)) & (labels == population) & np.isfinite(score)
        y = ((t < onset) & (onset <= t + horizon))[mask].astype(np.int8)
        ys.append(y)
        scores.append(score[mask])
        groups.extend([episode.get("leakage_group", episode["id"])] * int(mask.sum()))
    return np.concatenate(ys), np.concatenate(scores), np.asarray(groups, str)


def future_any_arrays(episodes: list[dict], horizon: int, population: int, score_key: str) -> tuple[np.ndarray, np.ndarray]:
    ys, scores = [], []
    for episode in episodes:
        labels = np.asarray(episode["labels"])
        score = np.asarray(episode[score_key], float)
        gross = np.flatnonzero(labels == 2)
        onset = int(gross[0]) if len(gross) else len(labels)
        for t in range(min(onset, len(labels) - horizon)):
            if labels[t] == population and np.isfinite(score[t]):
                ys.append(int(np.any(labels[t + 1:t + horizon + 1] >= 1)))
                scores.append(float(score[t]))
    return np.asarray(ys, np.int8), np.asarray(scores, float)


def add_future_scores(episodes: list[dict], horizon_index: int) -> list[dict]:
    result = []
    for episode in episodes:
        row = dict(episode)
        row["score_future"] = np.asarray(episode["p_future"], float)[:, horizon_index]
        row["score_current"] = np.asarray(episode["p_slip"], float)
        row["score_gated"] = row["score_future"] * row["score_current"]
        result.append(row)
    return result


def future_monotonicity(episodes: list[dict], tolerance: float = 1e-6) -> dict:
    """Cumulative risk should not decrease as its horizon grows."""
    triples = []
    by_episode = {}
    for episode in episodes:
        risk = np.asarray(episode["p_future"], float)
        valid = np.isfinite(risk).all(axis=1)
        current = risk[valid]
        violation_13 = current[:, 0] > current[:, 1] + tolerance
        violation_35 = current[:, 1] > current[:, 2] + tolerance
        any_violation = violation_13 | violation_35
        by_episode[episode["id"]] = {
            "valid_frames": int(len(current)),
            "violating_frames": int(any_violation.sum()),
            "violation_fraction": float(any_violation.mean()) if len(current) else None,
        }
        if len(current):
            triples.append(current)
    risk = np.concatenate(triples) if triples else np.empty((0, 3))
    if not len(risk):
        return {"valid_frames": 0, "violating_frames": 0, "violation_fraction": None, "by_episode": by_episode}
    delta_13 = risk[:, 0] - risk[:, 1]
    delta_35 = risk[:, 1] - risk[:, 2]
    violation = (delta_13 > tolerance) | (delta_35 > tolerance)
    return {
        "definition": "violation when p(H1)>p(H3)+1e-6 or p(H3)>p(H5)+1e-6",
        "valid_frames": int(len(risk)),
        "violating_frames": int(violation.sum()),
        "violation_fraction": float(violation.mean()),
        "maximum_drop_H1_to_H3": float(np.maximum(delta_13, 0).max()),
        "maximum_drop_H3_to_H5": float(np.maximum(delta_35, 0).max()),
        "by_episode": by_episode,
    }


def future_breakdowns(
    episodes: list[dict],
    horizon: int,
    population: int,
    score_key: str,
    threshold: float,
    matched_fold_calibration: bool,
) -> dict:
    """Per-probe/episode views at one already-fixed fold-level threshold."""
    labeled = [episode for episode in episodes if np.any(np.isin(episode["labels"], [0, 1, 2]))]

    def summarize(rows: list[dict]) -> dict:
        y, score, _ = first_gross_frame_arrays(rows, horizon, population, score_key)
        any_y, any_score = future_any_arrays(rows, horizon, population, score_key)
        return {
            "threshold": float(threshold),
            "matched_fold_calibration": matched_fold_calibration,
            "first_gross_frame_metrics": binary_metrics(y, score, threshold),
            "first_gross_event_metrics": event_metrics(rows, score_key, threshold, horizon, population),
            "future_any_changed_semantics_frame_metrics": binary_metrics(any_y, any_score, threshold),
        }

    by_episode = {episode["id"]: summarize([episode]) for episode in labeled}
    grouped: dict[str, list[dict]] = {}
    for episode in labeled:
        grouped.setdefault(episode["group"], []).append(episode)
    by_probe = {probe: summarize(rows) for probe, rows in sorted(grouped.items())}
    return {"by_probe": by_probe, "by_episode": by_episode}


def evaluate_detection(splits: dict, cache: dict[str, dict]) -> tuple[dict, dict, dict]:
    report, calibration, curves = {}, {}, {}
    for fold in sorted(name for name in splits["splits"] if name.startswith("htt_leave_")):
        cal_eps = selected_episodes(splits["splits"][fold]["calibration"], cache, "htt")
        val_eps = selected_episodes(splits["splits"][fold]["validation"], cache, "htt")
        report[fold], calibration[fold], curves[fold] = {}, {}, {}
        for name, definition in DETECTION_TARGETS.items():
            cy, cs, _ = detection_arrays(cal_eps, definition)
            selected = select_balanced_threshold(cy, cs)
            calibration[fold][name] = {
                **selected,
                "selection_set_note": "descriptive metrics on calibration data are optimistic and not validation evidence",
                "B0_on_calibration": binary_metrics(cy, cs, 0.5),
                "B0_cal_on_calibration": None if selected["threshold"] is None else binary_metrics(cy, cs, selected["threshold"]),
            }
            vy, vs, vg = detection_arrays(val_eps, definition)
            variants = {"B0": 0.5, "B0_cal": selected["threshold"]}
            report[fold][name] = {}
            for variant, threshold in variants.items():
                if threshold is None:
                    report[fold][name][variant] = {"evaluable": False, "reason": "calibration threshold unavailable"}
                    continue
                aggregate = binary_metrics(vy, vs, threshold)
                aggregate["group_bootstrap"] = group_bootstrap(vy, vs, vg, threshold)
                report[fold][name][variant] = {
                    "aggregate": aggregate,
                    **detection_breakdowns(val_eps, definition, threshold),
                }
            curves[fold][name] = {"validation_supplement_only": True, "rows": threshold_curve(vy, vs)}
    return report, calibration, curves


def evaluate_future(splits: dict, cache: dict[str, dict]) -> tuple[dict, dict, dict]:
    report = {
        "interpretation": {
            "primary_target": "first gross onset in (t,t+H], full-horizon masked",
            "populations": "current-static primary; current-incipient separate",
            "training_deployment_covariate_shift": "historical head training used GT normalized delta force; this cache uses predicted-force t-(t-5) delta",
            "claim_boundary": "diagnostic transfer result, not same-input deployable accuracy",
        }
    }
    calibration, curves = {}, {}
    method_keys = {"future_head": "score_future", "current_p_slip": "score_current", "historical_gated": "score_gated"}
    for fold in sorted(name for name in splits["splits"] if name.startswith("htt_leave_")):
        cal_base = selected_episodes(splits["splits"][fold]["calibration"], cache, "htt")
        val_base = selected_episodes(splits["splits"][fold]["validation"], cache, "htt")
        report[fold], calibration[fold], curves[fold] = {}, {}, {}
        report[fold]["cross_horizon_monotonicity_on_validation"] = future_monotonicity(val_base)
        for hidx, horizon in enumerate(HORIZONS):
            cal_eps, val_eps = add_future_scores(cal_base, hidx), add_future_scores(val_base, hidx)
            hname = f"H{horizon}"
            report[fold][hname], calibration[fold][hname], curves[fold][hname] = {}, {}, {}
            for population_name, population in POPULATIONS.items():
                report[fold][hname][population_name] = {}
                calibration[fold][hname][population_name] = {}
                curves[fold][hname][population_name] = {}
                for method, score_key in method_keys.items():
                    cy, cs, _ = first_gross_frame_arrays(cal_eps, horizon, population, score_key)
                    selected = select_fpr_threshold(cy, cs, max_fpr=0.01)
                    calibration[fold][hname][population_name][method] = {
                        **selected,
                        "descriptive_probability_and_one_class_metrics_at_unselected_0_5": binary_metrics(cy, cs, 0.5),
                    }
                    vy, vs, vg = first_gross_frame_arrays(val_eps, horizon, population, score_key)
                    any_y, any_score = future_any_arrays(val_eps, horizon, population, score_key)
                    any_diagnostic = {
                        "definition": "any mapped class 1 or 2 in (t,t+H], restricted to pre-first-gross cohort",
                        "metrics_at_unselected_0_5": binary_metrics(any_y, any_score, 0.5),
                    }
                    threshold = selected["threshold"]
                    if threshold is None:
                        support_metrics = binary_metrics(vy, vs, 2.0)
                        result = {
                            "evaluable": False,
                            "reason": "calibration threshold unavailable; no validation operating point evaluated",
                            "first_gross_frame_support": support_metrics["support"],
                            "first_gross_event_support": future_event_support(val_eps, horizon, population, score_key),
                            "first_gross_frame_metrics": None,
                            "first_gross_event_metrics": None,
                            "descriptive_probability_and_one_class_metrics_at_unselected_0_5": binary_metrics(vy, vs, 0.5),
                            "descriptive_event_metrics_at_unselected_0_5": {
                                "matched_fpr_operating_point": False,
                                "metrics": event_metrics(val_eps, score_key, 0.5, horizon, population),
                            },
                            "future_any_changed_semantics_diagnostic": any_diagnostic,
                            "subgroups_at_unselected_0_5": future_breakdowns(
                                val_eps, horizon, population, score_key, 0.5, matched_fold_calibration=False
                            ),
                        }
                    else:
                        aggregate = binary_metrics(vy, vs, threshold)
                        aggregate["group_bootstrap"] = group_bootstrap(vy, vs, vg, threshold)
                        any_diagnostic["metrics_at_first_gross_calibrated_threshold"] = binary_metrics(any_y, any_score, threshold)
                        result = {
                            "first_gross_frame_metrics": aggregate,
                            "first_gross_event_metrics": event_metrics(val_eps, score_key, threshold, horizon, population),
                            "future_any_changed_semantics_diagnostic": any_diagnostic,
                            "subgroups_at_fold_calibrated_threshold": future_breakdowns(
                                val_eps, horizon, population, score_key, threshold, matched_fold_calibration=True
                            ),
                        }
                    report[fold][hname][population_name][method] = result
                    curves[fold][hname][population_name][method] = {
                        "validation_supplement_only": True,
                        "rows": threshold_curve(vy, vs),
                    }
    return report, calibration, curves


def rotation_angle(matrix: np.ndarray) -> float:
    cosine = np.clip((np.trace(matrix[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def normalflow_partition(episodes: list[dict]) -> dict:
    finite = total = 0
    all_z = []
    horizon_rows = {}
    image_changes, image_pose_translation, image_pose_rotation = [], [], []
    for episode in episodes:
        z = np.asarray(episode["z"], float)
        finite += int(np.isfinite(z).sum())
        total += int(z.size)
        all_z.append(z)
        if "image_delta_l1" in episode:
            image = np.asarray(episode["image_delta_l1"], float)
            pose = np.asarray(episode["pose"], float)
            for t in range(1, len(z)):
                if np.isfinite(image[t]) and np.isfinite(pose[[t - 1, t]]).all():
                    relative = np.linalg.solve(pose[t - 1], pose[t])
                    image_changes.append(image[t])
                    image_pose_translation.append(np.linalg.norm(relative[:3, 3]))
                    image_pose_rotation.append(rotation_angle(relative))
    stacked = np.concatenate(all_z, axis=0) if all_z else np.empty((0, 768))
    variances = np.var(stacked, axis=0) if len(stacked) else np.full(768, np.nan)
    for horizon in HORIZONS:
        feature_delta, pose_translation, pose_rotation, squared_errors = [], [], [], []
        for episode in episodes:
            z = np.asarray(episode["z"], float)
            pose = np.asarray(episode["pose"], float)
            for t in range(max(0, len(z) - horizon)):
                if np.isfinite(z[[t, t + horizon]]).all():
                    delta = z[t + horizon] - z[t]
                    feature_delta.append(np.linalg.norm(delta))
                    squared_errors.extend(np.square(delta).tolist())
                    if np.isfinite(pose[[t, t + horizon]]).all():
                        relative = np.linalg.solve(pose[t], pose[t + horizon])
                        pose_translation.append(np.linalg.norm(relative[:3, 3]))
                        pose_rotation.append(rotation_angle(relative))
                    else:
                        pose_translation.append(np.nan)
                        pose_rotation.append(np.nan)
        horizon_rows[f"H{horizon}"] = {
            "pairs": len(feature_delta),
            "persistence_mse": float(np.mean(squared_errors)) if squared_errors else None,
            "feature_delta_l2": {
                "mean": float(np.mean(feature_delta)) if feature_delta else None,
                "median": float(np.median(feature_delta)) if feature_delta else None,
            },
            "correlation_with_relative_translation_norm_meters": pearson(feature_delta, pose_translation),
            "correlation_with_relative_rotation_angle_radians": pearson(feature_delta, pose_rotation),
        }
    return {
        "episodes": len(episodes),
        "frames": int(sum(row["frames"] for row in episodes)),
        "finite_feature_fraction": float(finite / total) if total else None,
        "feature_dimension_variance": {
            "mean": float(np.nanmean(variances)) if len(stacked) else None,
            "median": float(np.nanmedian(variances)) if len(stacked) else None,
            "min": float(np.nanmin(variances)) if len(stacked) else None,
            "max": float(np.nanmax(variances)) if len(stacked) else None,
            "zero_variance_dimensions": int(np.sum(variances == 0)) if len(stacked) else None,
        },
        "horizons": horizon_rows,
        "adjacent_image_delta_l1": {
            "available": bool(image_changes),
            "pairs": len(image_changes),
            "correlation_with_relative_translation_norm_meters": pearson(image_changes, image_pose_translation),
            "correlation_with_relative_rotation_angle_radians": pearson(image_changes, image_pose_rotation),
        },
    }


def evaluate_normalflow(splits: dict, cache: dict[str, dict]) -> dict:
    result = {
        "interpretation": {
            "z": "pooled frozen representation, not a trained world-model state",
            "pose": "true_start_T_currs is start_T_curr (current sensor coordinates to start sensor coordinates), translation in metres; solve(T_t,T_future) maps future sensor coordinates into the t sensor frame",
            "pose_acquisition_chain_limit": "the upstream MoCap-to-sensor extrinsic/calibration chain was not independently verified from released arrays",
            "image": "adjacent raw RGB mean absolute difference when cache provides image_delta_l1",
            "excluded_claims": ["slip labels", "slip F1", "actions", "physical-time prediction", "world-model validity"],
            "official_code_evidence": [
                "https://github.com/rpl-cmu/normalflow_experiment/blob/9d1c1dd18239324afbf32b8cc87ae548a54bd1c0/track/track.py#L78-L121",
                "https://github.com/rpl-cmu/normalflow_experiment/blob/9d1c1dd18239324afbf32b8cc87ae548a54bd1c0/visualization/viz_track.py#L123-L153",
                "https://github.com/rpl-cmu/normalflow/blob/8c17b678f84e1e2e199dbd39039413abfc720220/normalflow/utils.py#L23-L37",
                "https://github.com/rpl-cmu/normalflow/blob/8c17b678f84e1e2e199dbd39039413abfc720220/normalflow/utils.py#L140-L158",
            ],
        }
    }
    split = splits["splits"]["normalflow_objects"]
    for partition in ("train", "validation"):
        result[partition] = normalflow_partition(selected_episodes(split[partition], cache, "normalflow"))
    return result


def require_complete_cache(index: dict) -> None:
    if index.get("status") != "complete":
        raise ValueError("Cache index status must be complete before metric evaluation")
    expected = index.get("expected_episodes")
    if not isinstance(expected, int) or len(index.get("episodes", [])) != expected:
        raise ValueError("Cache episode inventory does not match expected_episodes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index, cache = load_cache(args.index)
    require_complete_cache(index)
    splits = read_json(args.splits)
    if splits.get("schema_version") != 2:
        raise ValueError("Only authoritative schema-2 splits are accepted")
    actual_split_hash = sha256(args.splits)
    if index["split_manifest_sha256"] != actual_split_hash:
        raise ValueError("Cache/split fingerprint mismatch")
    detection, detection_cal, detection_curves = evaluate_detection(splits, cache)
    future, future_cal, future_curves = evaluate_future(splits, cache)
    normalflow = evaluate_normalflow(splits, cache)
    provenance = {
        "cache_index": str(args.index),
        "cache_index_sha256": sha256(args.index),
        "split_manifest": str(args.splits),
        "split_manifest_sha256": actual_split_hash,
        "checkpoint_sha256": index["checkpoint_sha256"],
        "future_checkpoint_sha256": index.get("future_checkpoint_sha256"),
        "future_head": index.get("future_head"),
        "preprocessing_fingerprint": index["preprocessing_fingerprint"],
        "threshold_grid": THRESHOLD_GRID.tolist(),
        "bootstrap": {"seed": 20260913, "repetitions": 200, "unit": "leakage_group or episode fallback"},
        "test_partitions_read": False,
        "scope_note": "four-fold development union can cover every probe; this is not a global blind test",
        "future_covariate_shift": "historical head trained with GT normalized delta-force feature; cache uses predicted-force t-(t-5) delta only",
    }
    outputs = {
        "provenance.json": provenance,
        "current_detection.json": detection,
        "current_calibration.json": detection_cal,
        "current_validation_threshold_curves.json": detection_curves,
        "future_warning.json": future,
        "future_calibration.json": future_cal,
        "future_validation_threshold_curves.json": future_curves,
        "normalflow_diagnostics.json": normalflow,
    }
    for name, payload in outputs.items():
        write_json(args.output / name, payload)
    write_json(args.output / "COMPLETE.json", {
        "status": "complete",
        "cache_index_sha256": provenance["cache_index_sha256"],
        "split_manifest_sha256": actual_split_hash,
        "files": sorted([*outputs, "COMPLETE.json"]),
    })


if __name__ == "__main__":
    main()
