#!/usr/bin/env python3
"""Render human-readable round-2 reports from completed immutable artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


DEFINITIONS = {
    "static_vs_gross": {"keep": (0, 2), "positive": (2,)},
    "static_vs_any_slip": {"keep": (0, 1, 2), "positive": (1, 2)},
}


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def verify_inputs(metrics_dir: Path, index_path: Path, split_path: Path) -> tuple[dict, dict, dict]:
    complete = read_json(metrics_dir / "COMPLETE.json")
    if complete.get("status") != "complete":
        raise ValueError("Metrics completion marker is absent or incomplete")
    index = read_json(index_path)
    if index.get("status") != "complete":
        raise ValueError("Cache index is not complete")
    if len(index.get("episodes", [])) != index.get("expected_episodes"):
        raise ValueError("Cache episode inventory is incomplete")
    splits = read_json(split_path)
    if splits.get("schema_version") != 2:
        raise ValueError("Only authoritative schema-2 splits are accepted")
    split_hash = sha256(split_path)
    index_hash = sha256(index_path)
    if index.get("split_manifest_sha256") != split_hash:
        raise ValueError("Cache/split fingerprint mismatch")
    if complete.get("split_manifest_sha256") != split_hash or complete.get("cache_index_sha256") != index_hash:
        raise ValueError("Metrics were not generated from the supplied cache and split")
    provenance = read_json(metrics_dir / "provenance.json")
    if provenance.get("test_partitions_read") is not False:
        raise ValueError("Metric provenance does not prove test partitions were excluded")
    return complete, index, splits


def detection_rows(metrics_dir: Path) -> list[dict]:
    detection = read_json(metrics_dir / "current_detection.json")
    calibration = read_json(metrics_dir / "current_calibration.json")
    rows = []
    for fold in sorted(detection):
        for definition in DEFINITIONS:
            selected = calibration[fold][definition]
            for variant in ("B0", "B0_cal"):
                payload = detection[fold][definition][variant]
                aggregate = payload.get("aggregate", payload)
                confusion = aggregate.get("confusion") or {}
                negative = aggregate.get("negative") or {}
                positive = aggregate.get("positive") or {}
                rows.append({
                    "fold": fold,
                    "definition": definition,
                    "variant": variant,
                    "threshold": aggregate.get("threshold"),
                    "calibration_selected_threshold": selected.get("threshold"),
                    "support_total": (aggregate.get("support") or {}).get("total"),
                    "support_negative": (aggregate.get("support") or {}).get("negative"),
                    "support_positive": (aggregate.get("support") or {}).get("positive"),
                    "tn": confusion.get("tn"),
                    "fp": confusion.get("fp"),
                    "fn": confusion.get("fn"),
                    "tp": confusion.get("tp"),
                    "negative_precision": negative.get("precision"),
                    "negative_recall": negative.get("recall"),
                    "negative_f1": negative.get("f1"),
                    "positive_precision": positive.get("precision"),
                    "positive_recall": positive.get("recall"),
                    "positive_f1": positive.get("f1"),
                    "macro_f1": aggregate.get("macro_f1"),
                    "balanced_accuracy": aggregate.get("balanced_accuracy"),
                    "average_precision": aggregate.get("average_precision"),
                    "brier": aggregate.get("brier"),
                    "bootstrap_valid_replicates": (aggregate.get("group_bootstrap") or {}).get("valid_replicates"),
                })
    return rows


def failure_rows(index: dict, splits: dict, calibration: dict) -> list[dict]:
    cache = {row["id"]: row for row in index["episodes"]}
    rows = []
    for fold in sorted(name for name in splits["splits"] if name.startswith("htt_leave_")):
        validation_ids = splits["splits"][fold]["validation"]
        for definition_name, definition in DEFINITIONS.items():
            episode_data = []
            for episode_id in validation_ids:
                record = cache.get(episode_id)
                if record is None:
                    raise ValueError(f"Cache lacks validation episode {episode_id}")
                cache_path = Path(record["cache_path"])
                with np.load(cache_path, allow_pickle=False) as data:
                    labels = data["labels"].astype(np.int8)
                    score = data["p_slip"].astype(float)
                valid = np.isin(labels, definition["keep"]) & np.isfinite(score)
                if not np.any(valid):
                    continue  # In particular, never rank unlabeled force/static episodes.
                target = np.isin(labels, definition["positive"])
                episode_data.append((episode_id, record, labels, score, valid, target))
            thresholds = [
                ("B0", 0.5),
                ("B0_cal", calibration[fold][definition_name].get("threshold")),
            ]
            for variant, threshold in thresholds:
                if threshold is None:
                    continue
                candidates = []
                for episode_id, record, labels, score, valid, target in episode_data:
                    prediction = score >= threshold
                    fp_indices = np.flatnonzero(valid & ~target & prediction)
                    fn_indices = np.flatnonzero(valid & target & ~prediction)
                    errors = np.sort(np.r_[fp_indices, fn_indices])
                    fp_set = set(fp_indices.tolist())
                    details = [
                        {
                            "frame": int(frame),
                            "p_slip": float(score[frame]),
                            "label": int(labels[frame]),
                            "error": "FP" if frame in fp_set else "FN",
                        }
                        for frame in errors[:10]
                    ]
                    support = int(valid.sum())
                    candidates.append({
                        "fold": fold,
                        "definition": definition_name,
                        "variant": variant,
                        "episode_id": episode_id,
                        "probe": record.get("group"),
                        "threshold": threshold,
                        "support": support,
                        "fp": int(len(fp_indices)),
                        "fn": int(len(fn_indices)),
                        "errors": int(len(errors)),
                        "error_rate": float(len(errors) / support),
                        "error_frames_json": json.dumps(details, ensure_ascii=False, separators=(",", ":")),
                    })
                candidates.sort(key=lambda row: (-row["error_rate"], -row["errors"], row["episode_id"]))
                rows.extend(candidates[:3])
    return rows


def future_table(future: dict, calibration: dict) -> list[dict]:
    rows = []
    for fold in sorted(key for key in future if key.startswith("htt_leave_")):
        for horizon in ("H1", "H3", "H5"):
            for population in ("current_static", "current_incipient"):
                for method in ("future_head", "current_p_slip", "historical_gated"):
                    payload = future[fold][horizon][population][method]
                    selected = calibration[fold][horizon][population][method]
                    first = payload.get("first_gross_frame_metrics")
                    descriptive = payload.get("descriptive_probability_and_one_class_metrics_at_unselected_0_5")
                    shown = first or descriptive or {}
                    event = payload.get("first_gross_event_metrics")
                    matched = selected.get("threshold") is not None
                    if event is None:
                        event = (payload.get("descriptive_event_metrics_at_unselected_0_5") or {}).get("metrics") or {}
                    any_metrics = payload["future_any_changed_semantics_diagnostic"]["metrics_at_unselected_0_5"]
                    support = shown.get("support") or {}
                    any_support = any_metrics.get("support") or {}
                    negative = shown.get("negative") or {}
                    positive = shown.get("positive") or {}
                    rows.append({
                        "fold": fold,
                        "horizon": horizon,
                        "population": population,
                        "method": method,
                        "selected_threshold": selected.get("threshold"),
                        "operating_threshold": selected.get("threshold") if matched else 0.5,
                        "matched_fold_calibration": matched,
                        "calibration_constrained_threshold": matched,
                        "matched_fold_calibration_definition": "threshold fitted on calibration at FPR<=1%; validation FPR is reported independently and need not match",
                        "firstgross_negative": support.get("negative"),
                        "firstgross_positive": support.get("positive"),
                        "firstgross_fpr": None if negative.get("recall") is None else 1.0 - negative["recall"],
                        "firstgross_recall": positive.get("recall"),
                        "firstgross_average_precision": shown.get("average_precision"),
                        "firstgross_brier": shown.get("brier"),
                        "event_support": event.get("support_events"),
                        "event_recall": event.get("event_recall"),
                        "event_misses": event.get("misses"),
                        "event_late": event.get("late"),
                        "event_lead_median_steps": (event.get("lead_steps") or {}).get("median"),
                        "event_lead_min_steps": (event.get("lead_steps") or {}).get("min"),
                        "event_lead_max_steps": (event.get("lead_steps") or {}).get("max"),
                        "event_late_delay_median_steps": (event.get("late_delay_steps") or {}).get("median"),
                        "population_false_edge_rate_per_negative_frame": event.get("population_conditioned_false_alarm_edges_per_negative_frame"),
                        "sequence_false_edge_rate_per_negative_frame": event.get("sequence_false_alarm_edges_per_negative_frame"),
                        "any_negative": any_support.get("negative"),
                        "any_positive": any_support.get("positive"),
                        "any_average_precision_at_unselected_0_5": any_metrics.get("average_precision"),
                        "any_brier_at_unselected_0_5": any_metrics.get("brier"),
                    })
    return rows


def future_failure_rows(future: dict, calibration: dict) -> list[dict]:
    rows = []
    for fold in sorted(key for key in future if key.startswith("htt_leave_")):
        for horizon in ("H1", "H3", "H5"):
            for population in ("current_static", "current_incipient"):
                for method in ("future_head", "current_p_slip", "historical_gated"):
                    payload = future[fold][horizon][population][method]
                    selected_threshold = calibration[fold][horizon][population][method].get("threshold")
                    matched = selected_threshold is not None
                    threshold = selected_threshold if matched else 0.5
                    event = payload.get("first_gross_event_metrics")
                    if event is None:
                        event = (payload.get("descriptive_event_metrics_at_unselected_0_5") or {}).get("metrics") or {}
                    for episode in event.get("episodes", []):
                        population_false = episode.get("population_false_alarm_edges", 0)
                        sequence_false = episode.get("sequence_false_alarm_edges", 0)
                        failure_types = []
                        if episode.get("miss") is True:
                            failure_types.append("miss")
                        if episode.get("late") is True:
                            failure_types.append("late")
                        if population_false:
                            failure_types.append("population_false_alarm")
                        if sequence_false:
                            failure_types.append("sequence_false_alarm")
                        if not failure_types:
                            continue
                        rows.append({
                            "fold": fold,
                            "horizon": horizon,
                            "population": population,
                            "method": method,
                            "episode_id": episode["episode_id"],
                            "threshold": threshold,
                            "matched_fold_calibration": matched,
                            "calibration_constrained_threshold": matched,
                            "matched_fold_calibration_definition": "threshold fitted on calibration at FPR<=1%; validation FPR is reported independently and need not match",
                            "failure_types": ";".join(failure_types),
                            "onset": episode.get("onset"),
                            "supported": episode.get("supported"),
                            "hit": episode.get("hit"),
                            "late": episode.get("late"),
                            "miss": episode.get("miss"),
                            "lead_steps": episode.get("lead_steps"),
                            "late_delay_steps": episode.get("late_delay_steps"),
                            "population_false_alarm_edges": population_false,
                            "sequence_false_alarm_edges": sequence_false,
                        })
    return rows


def render_markdown(
    metrics_dir: Path,
    index: dict,
    detection: list[dict],
    future_rows: list[dict],
) -> str:
    future = read_json(metrics_dir / "future_warning.json")
    normalflow = read_json(metrics_dir / "normalflow_diagnostics.json")
    lines = [
        "# Force-slip round-2 frozen baseline report",
        "",
        "This report uses completed frozen caches and schema-2 development partitions only. The four-fold development union can cover every HTT probe; it is not a global blind test.",
        "",
        "## Current detection",
        "",
        "| fold | definition | variant | threshold | support (neg/pos) | macro F1 | balanced accuracy | AP | Brier | FP/FN |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in detection:
        lines.append(
            f"| {row['fold']} | {row['definition']} | {row['variant']} | {fmt(row['threshold'], 3)} | "
            f"{row['support_negative']}/{row['support_positive']} | {fmt(row['macro_f1'])} | "
            f"{fmt(row['balanced_accuracy'])} | {fmt(row['average_precision'])} | {fmt(row['brier'])} | {row['fp']}/{row['fn']} |"
        )
    brier_pairs = {}
    for row in detection:
        brier_pairs.setdefault((row["fold"], row["definition"]), {})[row["variant"]] = row["brier"]
    brier_unchanged = all(
        values.get("B0") is None or values.get("B0_cal") is None or np.isclose(values["B0"], values["B0_cal"])
        for values in brier_pairs.values()
    )
    lines += [
        "",
        f"Threshold calibration changes decisions only. Brier is unchanged across B0/B0-cal: `{brier_unchanged}`.",
        "",
        "The failure-case CSV separately ranks the three highest-error validation episodes per fold and definition for B0 and B0-cal. Unlabeled force/static episodes are excluded.",
        "",
        "## Future warning",
        "",
        "First-gross and mapped future-any are separate targets. `NA` denotes an unsupported metric, not zero performance. A `True` calibration-constrained flag means only that the threshold was fitted on calibration data under FPR<=1%; validation operating points are reported as observed and are not necessarily matched. Fixed-0.5 results in one-class settings are descriptive.",
        "",
        "| fold | H | population | method | threshold/cal-constrained | first-gross neg/pos | validation FPR/recall/AP/Brier | event recall/support/miss/late | lead median[min,max] | pop/seq false-edge rate | any neg/pos/AP |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in future_rows:
        lines.append(
            f"| {row['fold']} | {row['horizon']} | {row['population']} | {row['method']} | {fmt(row['operating_threshold'], 3)}/{row['calibration_constrained_threshold']} | "
            f"{row['firstgross_negative']}/{row['firstgross_positive']} | {fmt(row['firstgross_fpr'])}/{fmt(row['firstgross_recall'])}/{fmt(row['firstgross_average_precision'])}/{fmt(row['firstgross_brier'])} | "
            f"{fmt(row['event_recall'])}/{row['event_support']}/{row['event_misses']}/{row['event_late']} | "
            f"{fmt(row['event_lead_median_steps'])}[{fmt(row['event_lead_min_steps'])},{fmt(row['event_lead_max_steps'])}] | "
            f"{fmt(row['population_false_edge_rate_per_negative_frame'])}/{fmt(row['sequence_false_edge_rate_per_negative_frame'])} | "
            f"{row['any_negative']}/{row['any_positive']}/{fmt(row['any_average_precision_at_unselected_0_5'])} |"
        )
    lines += [
        "",
        "HTT's bracket construction places incipient labels immediately before first gross. Consequently, current-static first-gross H1/H3/H5 has no positive warning examples, and current-incipient H5 has no negatives. Their unsupported discrimination and matched-FPR metrics remain `NA`; future-any may still have positives and is shown independently.",
        "",
        "The historical future head was trained with a ground-truth normalized delta-force feature, while this deployment cache uses predicted-force delta(t,t-5). This covariate shift limits the result to transfer diagnosis.",
        "",
        "The full aggregate future fields, including late-delay median and both false-edge rates, are in `future_comparison.csv`; raw subgroup and episode details remain in `future_warning.json`.",
        "",
        "## Cross-horizon monotonicity",
        "",
        "| fold | valid frames | violating frames | fraction | max H1→H3 drop | max H3→H5 drop |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for fold in sorted(key for key in future if key.startswith("htt_leave_")):
        row = future[fold]["cross_horizon_monotonicity_on_validation"]
        lines.append(
            f"| {fold} | {row['valid_frames']} | {row['violating_frames']} | {fmt(row['violation_fraction'])} | "
            f"{fmt(row.get('maximum_drop_H1_to_H3'))} | {fmt(row.get('maximum_drop_H3_to_H5'))} |"
        )
    lines += [
        "",
        "## NormalFlow frozen-feature diagnostics",
        "",
        "`z` is a pooled frozen representation, not a trained world-model state. `true_start_T_currs` is verified as start_T_curr: current-sensor coordinates map into the start sensor frame and translation is in metres. The unchanged solve(T_t,T_future) computation maps future-sensor coordinates into the time-t sensor frame. The upstream MoCap-to-sensor extrinsic/calibration chain remains independently unverified; these are not actions or slip evidence.",
        "",
        "| partition | frames | finite z | mean/median feature variance | H | persistence MSE | feature↔translation(m) r | feature↔rotation(rad) r |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    image_bullets = []
    for partition in ("train", "validation"):
        part = normalflow[partition]
        variance = part["feature_dimension_variance"]
        for horizon in ("H1", "H3", "H5"):
            hrow = part["horizons"][horizon]
            lines.append(
                f"| {partition} | {part['frames']} | {fmt(part['finite_feature_fraction'])} | "
                f"{fmt(variance['mean'])}/{fmt(variance['median'])} | {horizon} | {fmt(hrow['persistence_mse'], 6)} | "
                f"{fmt(hrow['correlation_with_relative_translation_norm_meters']['r'])} | "
                f"{fmt(hrow['correlation_with_relative_rotation_angle_radians']['r'])} |"
            )
        image = part["adjacent_image_delta_l1"]
        image_bullets.append(
            f"- {partition} adjacent image delta: available `{image['available']}`, pairs `{image['pairs']}`, "
            f"translation correlation in metres `{fmt(image['correlation_with_relative_translation_norm_meters']['r'])}`, "
            f"rotation correlation `{fmt(image['correlation_with_relative_rotation_angle_radians']['r'])}`."
        )
    lines += [
        "",
        *image_bullets,
        "",
        "Low persistence error alone does not establish a lightweight world model. NormalFlow has no slip labels, and its pose is not an action input.",
        "",
        "Pose direction/unit sources: [normalflow_experiment track.py](https://github.com/rpl-cmu/normalflow_experiment/blob/9d1c1dd18239324afbf32b8cc87ae548a54bd1c0/track/track.py#L78-L121), [viz_track.py](https://github.com/rpl-cmu/normalflow_experiment/blob/9d1c1dd18239324afbf32b8cc87ae548a54bd1c0/visualization/viz_track.py#L123-L153), and [normalflow utils.py](https://github.com/rpl-cmu/normalflow/blob/8c17b678f84e1e2e199dbd39039413abfc720220/normalflow/utils.py#L23-L37) ([transform implementation](https://github.com/rpl-cmu/normalflow/blob/8c17b678f84e1e2e199dbd39039413abfc720220/normalflow/utils.py#L140-L158)).",
        "",
        "## Provenance and files",
        "",
        f"- Frozen cache episodes: `{len(index['episodes'])}`",
        f"- Stage-I checkpoint SHA256: `{index.get('checkpoint_sha256')}`",
        f"- Future-head checkpoint SHA256: `{index.get('future_checkpoint_sha256')}`",
        "- Detection table: `detection_comparison.csv`",
        "- Full current-detection per-class metrics and TN/FP/FN/TP: [detection_comparison.csv](detection_comparison.csv); raw per-probe/episode metrics: [current_detection.json](../metrics/current_detection.json)",
        "- Full future aggregate comparison: [future_comparison.csv](future_comparison.csv); raw per-probe/episode metrics: [future_warning.json](../metrics/future_warning.json)",
        "- Validation failure cases: `failure_cases.csv`",
        "- Future warning misses, late detections and false alarms: `future_failure_cases.csv`",
        "- Calibration records: [current_calibration.json](../metrics/current_calibration.json) and [future_calibration.json](../metrics/future_calibration.json).",
    ]
    return "\n".join(lines) + "\n"


def render_reports(metrics_dir: Path, index_path: Path, split_path: Path, output: Path) -> None:
    _, index, splits = verify_inputs(metrics_dir, index_path, split_path)
    detection = detection_rows(metrics_dir)
    calibration = read_json(metrics_dir / "current_calibration.json")
    failures = failure_rows(index, splits, calibration)
    future_payload = read_json(metrics_dir / "future_warning.json")
    future_calibration = read_json(metrics_dir / "future_calibration.json")
    future_rows = future_table(future_payload, future_calibration)
    future_failures = future_failure_rows(future_payload, future_calibration)
    detection_fields = list(detection[0]) if detection else []
    failure_fields = list(failures[0]) if failures else [
        "fold", "definition", "variant", "episode_id", "probe", "threshold", "support",
        "fp", "fn", "errors", "error_rate", "error_frames_json",
    ]
    write_csv(output / "detection_comparison.csv", detection, detection_fields)
    write_csv(output / "future_comparison.csv", future_rows, list(future_rows[0]) if future_rows else [])
    write_csv(output / "failure_cases.csv", failures, failure_fields)
    future_failure_fields = list(future_failures[0]) if future_failures else [
        "fold", "horizon", "population", "method", "episode_id", "threshold",
        "matched_fold_calibration", "failure_types", "onset", "supported", "hit",
        "calibration_constrained_threshold", "matched_fold_calibration_definition",
        "late", "miss", "lead_steps", "late_delay_steps",
        "population_false_alarm_edges", "sequence_false_alarm_edges",
    ]
    write_csv(output / "future_failure_cases.csv", future_failures, future_failure_fields)
    markdown = render_markdown(metrics_dir, index, detection, future_rows)
    output.mkdir(parents=True, exist_ok=True)
    tmp = output / "REPORT.md.tmp"
    tmp.write_text(markdown, encoding="utf-8")
    tmp.replace(output / "REPORT.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_reports(args.metrics, args.index, args.splits, args.output)


if __name__ == "__main__":
    main()
