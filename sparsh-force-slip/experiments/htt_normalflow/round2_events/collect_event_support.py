#!/usr/bin/env python3
"""Summarize HTT first-gross event support in schema-2 development partitions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from adapters import future_onset  # noqa: E402


DEV_PARTITIONS = ("train", "validation", "calibration")
HORIZONS = (1, 3, 5)
STAGES = {0: "static", 1: "incipient", 2: "gross"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_path(row: dict) -> Path | None:
    candidates = [Path(path) for path in row.get("source_files", {}) if path.endswith(".labeled.npz")]
    if len(candidates) > 1:
        raise ValueError(f"Multiple label files for {row['id']}")
    return candidates[0] if candidates else None


def load_labels(row: dict) -> tuple[np.ndarray | None, dict | None, Path | None]:
    path = label_path(row)
    if path is None:
        return None, None, None
    with np.load(path, allow_pickle=False) as archive:
        labels = archive["sliding_labels_bracket"].astype(np.int64)
        meta = json.loads(str(archive["labeling_meta"].item()))
    if len(labels) != row["frames"]:
        raise ValueError(f"Label length mismatch: {row['id']}")
    if not np.isin(labels, tuple(STAGES)).all():
        raise ValueError(f"Unknown label value: {row['id']}")
    return labels, meta, path


def nullable_bool(value: bool | None):
    return "" if value is None else value


def episode_stats(fold: str, partition: str, row: dict, labels, meta, path) -> dict:
    result = {
        "fold": fold,
        "held_out_probe": fold.removeprefix("htt_leave_"),
        "partition": partition,
        "episode_id": row["id"],
        "probe": row["group"],
        "task": row["task"],
        "frames": row["frames"],
        "labels_available": labels is not None,
        "label_source": "sliding_labels_bracket" if labels is not None else "",
        "label_path": str(path) if path else "",
    }
    if labels is None:
        for stage in STAGES.values():
            result[f"frames_{stage}"] = ""
        for key in (
            "first_gross_index", "last_gross_index", "starts_gross", "no_gross", "no_static_prehistory",
            "frames_before_first_gross", "static_before_first_gross",
            "incipient_before_first_gross", "frames_recovery_after_first_gross",
            "post_or_current_gross_excluded",
        ):
            result[key] = ""
        for current in ("static", "incipient"):
            result[f"current_{current}_pre_event"] = ""
            for horizon in HORIZONS:
                for measure in ("eligible", "positive", "negative", "tail_masked", "positive_event"):
                    result[f"{current}_h{horizon}_{measure}"] = ""
        for key in (
            "meta_incipient_window", "meta_t_start", "meta_t_stop",
            "meta_first_gross_equals_t_start", "meta_last_gross_equals_t_stop",
            "meta_bracket_pattern_matches",
        ):
            result[key] = ""
        return result

    counts = {stage: int(np.sum(labels == value)) for value, stage in STAGES.items()}
    result.update({f"frames_{stage}": count for stage, count in counts.items()})
    gross_indices = np.flatnonzero(labels == 2)
    first_gross = int(gross_indices[0]) if len(gross_indices) else None
    last_gross = int(gross_indices[-1]) if len(gross_indices) else None
    pre_event = labels if first_gross is None else labels[:first_gross]
    result.update({
        "first_gross_index": "" if first_gross is None else first_gross,
        "last_gross_index": "" if last_gross is None else last_gross,
        "starts_gross": nullable_bool(first_gross == 0 if first_gross is not None else None),
        "no_gross": first_gross is None,
        "no_static_prehistory": nullable_bool(
            not np.any(pre_event == 0) if first_gross is not None else None
        ),
        "frames_before_first_gross": "" if first_gross is None else first_gross,
        "static_before_first_gross": "" if first_gross is None else int(np.sum(pre_event == 0)),
        "incipient_before_first_gross": "" if first_gross is None else int(np.sum(pre_event == 1)),
        "frames_recovery_after_first_gross": 0 if first_gross is None else int(np.sum(labels[first_gross:] == 0)),
        "post_or_current_gross_excluded": 0 if first_gross is None else len(labels) - first_gross,
        "current_static_pre_event": int(np.sum(pre_event == 0)),
        "current_incipient_pre_event": int(np.sum(pre_event == 1)),
    })
    for current_value, current_name in ((0, "static"), (1, "incipient")):
        for horizon in HORIZONS:
            eligible = positive = negative = tail_masked = 0
            for t in np.flatnonzero(labels == current_value):
                target, valid = future_onset(labels, int(t), (horizon,))
                if valid[0]:
                    eligible += 1
                    if target[0] == 1:
                        positive += 1
                    else:
                        negative += 1
                elif not np.any(labels[: t + 1] == 2) and t + horizon >= len(labels):
                    tail_masked += 1
            result[f"{current_name}_h{horizon}_eligible"] = eligible
            result[f"{current_name}_h{horizon}_positive"] = positive
            result[f"{current_name}_h{horizon}_negative"] = negative
            result[f"{current_name}_h{horizon}_tail_masked"] = tail_masked
            result[f"{current_name}_h{horizon}_positive_event"] = int(positive > 0)

    params = (meta or {}).get("params_bracket", {})
    incipient_window = params.get("incipient_window")
    t_start = params.get("t_start")
    t_stop = params.get("t_stop")
    expected_incipient_start = (
        max(0, first_gross - int(incipient_window))
        if first_gross is not None and incipient_window is not None else None
    )
    expected = None
    if first_gross is not None and expected_incipient_start is not None:
        expected = np.zeros(len(labels), dtype=np.int64)
        expected[expected_incipient_start:first_gross] = 1
        if t_stop is not None:
            expected[first_gross:int(t_stop) + 1] = 2
    result.update({
        "meta_incipient_window": "" if incipient_window is None else incipient_window,
        "meta_t_start": "" if t_start is None else t_start,
        "meta_t_stop": "" if t_stop is None else t_stop,
        "meta_first_gross_equals_t_start": nullable_bool(
            first_gross == t_start if first_gross is not None and t_start is not None else None
        ),
        "meta_last_gross_equals_t_stop": nullable_bool(
            last_gross == t_stop if last_gross is not None and t_stop is not None else None
        ),
        "meta_bracket_pattern_matches": nullable_bool(
            bool(np.array_equal(labels, expected)) if expected is not None else None
        ),
    })
    return result


def aggregate(rows: list[dict]) -> list[dict]:
    numeric_fields = [
        key for key in rows[0]
        if key.startswith("frames_") or key.startswith("current_")
        or any(key.endswith(suffix) for suffix in (
            "_eligible", "_positive", "_negative", "_tail_masked", "_positive_event"
        ))
    ]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["fold"], row["held_out_probe"], row["partition"], row["probe"])].append(row)
    output = []
    for key, members in sorted(groups.items()):
        item = dict(zip(("fold", "held_out_probe", "partition", "probe"), key))
        labeled = [row for row in members if row["labels_available"]]
        item.update({
            "episodes": len(members),
            "labeled_slip_episodes": len(labeled),
            "unlabeled_force_episodes": len(members) - len(labeled),
            "episodes_with_gross": sum(not row["no_gross"] for row in labeled),
            "episodes_starting_gross": sum(row["starts_gross"] is True for row in labeled),
            "episodes_without_gross": sum(row["no_gross"] is True for row in labeled),
            "episodes_without_static_prehistory": sum(row["no_static_prehistory"] is True for row in labeled),
        })
        for field in numeric_fields:
            item[field] = sum(int(row[field]) for row in labeled if row[field] != "")
        output.append(item)
    return output


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: dict):
    unique = summary["unique_labeled_episode_audit"]
    lines = [
        "# HTT development event support and bracket-label audit",
        "",
        "## Scope",
        "",
        "Only episode data and labels from `train`, `validation`, and `calibration` in each schema-2 HTT fold were read. "
        "Test ID lists were used only for an overlap assertion; no test episode data or labels were opened. "
        "Across cross-validation folds, every probe can appear "
        "in a development role for folds where it is not held out; fold-aggregated counts therefore "
        "repeat episodes and are not a global independent sample count.",
        "",
        "Force/static episodes have no slip labels and remain explicitly missing supervision. They are "
        "never counted as safe negative frames.",
        "",
        "## Unique labeled episodes (deduplicated across fold roles)",
        "",
        f"- Labeled sliding episodes: {unique['episodes']}.",
        f"- Episodes with a gross event: {unique['episodes_with_gross']}; starting gross: "
        f"{unique['episodes_starting_gross']}; without gross: {unique['episodes_without_gross']}.",
        f"- Episodes without any static prehistory: {unique['episodes_without_static_prehistory']}.",
        f"- Stage frames: static={unique['stage_frames']['static']}, "
        f"incipient={unique['stage_frames']['incipient']}, gross={unique['stage_frames']['gross']}.",
        f"- Pre-first-gross current frames: static={unique['pre_event_current_frames']['static']}, "
        f"incipient={unique['pre_event_current_frames']['incipient']}; post-gross recovery static frames "
        f"excluded by `future_onset`={unique['post_gross_recovery_static_frames']}.",
        "",
        "### First-gross target support (unique episodes)",
        "",
        "| current stage | horizon | eligible | positive | negative | tail masked | positive events |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {stage} | {horizon} | {unique['target_support'][stage][str(horizon)]['eligible']} | "
            f"{unique['target_support'][stage][str(horizon)]['positive']} | "
            f"{unique['target_support'][stage][str(horizon)]['negative']} | "
            f"{unique['target_support'][stage][str(horizon)]['tail_masked']} | "
            f"{unique['target_support'][stage][str(horizon)]['positive_event']} |"
            for stage in ("static", "incipient") for horizon in HORIZONS
        ],
        "",
        "The current-static cohort has no positive first-gross targets at 1/3/5 steps. This follows "
        "directly from the supplied five-step incipient backfill before `t_start`; discrimination or "
        "event recall for static-to-gross prediction is therefore not estimable at these horizons. "
        "The current-incipient cohort has event support, but it evaluates recognition within a label "
        "interval defined relative to the later gross boundary.",
        "",
        "## Label metadata evidence",
        "",
        f"- `incipient_window` values observed in metadata: {unique['meta']['incipient_window_values']}.",
        f"- First gross index equals metadata `t_start`: {unique['meta']['first_gross_equals_t_start']} "
        f"of {unique['episodes']} episodes.",
        f"- Last gross index equals metadata `t_stop`: {unique['meta']['last_gross_equals_t_stop']} "
        f"of {unique['episodes']} episodes.",
        f"- Exact empirical pattern `static -> up to incipient_window preceding frames -> gross from "
        f"t_start through t_stop (inclusive) -> static recovery` matches "
        f"{unique['meta']['bracket_pattern_matches']} of {unique['episodes']} episodes.",
        "- This is direct evidence from the supplied labels and per-file metadata that incipient labels "
        "use the later gross boundary. It is a proxy/backfilled stage, not an independently timed physical onset.",
        "- No label-generation source code was inspected; the report describes only the exact relationship "
        "observed between supplied arrays and metadata.",
        "",
        "## Reusable metric fields",
        "",
        "For each current-stage/horizon pair, the episode and grouped CSVs expose `eligible`, "
        "`positive`, `negative`, `tail_masked`, and `positive_event`. `positive_event` is one per episode "
        "when at least one eligible current frame predicts the single first-gross event, and should be "
        "used as the event-level support denominator rather than treating adjacent positive frames as independent events.",
        "",
        "Horizons are steps only: HTT has no verified time axis.",
        "",
        "## Artifacts",
        "",
        "- `episode_event_support.csv`: one row per fold/partition/episode role.",
        "- `group_event_support.csv`: sums by fold, partition, and probe.",
        "- `event_support.json`: provenance and machine-readable aggregate support.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 2:
        raise ValueError("Expected schema-2 manifest")

    episode_index = {row["id"]: row for row in manifest["episodes"]}
    rows = []
    cache = {}
    read_ids_by_fold = {}
    for fold, split in sorted(manifest["splits"].items()):
        if not fold.startswith("htt_leave_"):
            continue
        read_ids_by_fold[fold] = []
        test_ids = set(split["test"])
        for partition in DEV_PARTITIONS:
            if set(split[partition]) & test_ids:
                raise ValueError(f"Development/test overlap in {fold}/{partition}")
            for episode_id in split[partition]:
                row = episode_index[episode_id]
                if row["domain"] != "htt":
                    raise ValueError(f"Non-HTT row in {fold}: {episode_id}")
                read_ids_by_fold[fold].append(episode_id)
                if episode_id not in cache:
                    cache[episode_id] = load_labels(row)
                rows.append(episode_stats(fold, partition, row, *cache[episode_id]))

    if not rows:
        raise ValueError("No HTT development episodes found")
    grouped = aggregate(rows)
    labeled_unique = {row["episode_id"]: row for row in rows if row["labels_available"]}
    unique_rows = list(labeled_unique.values())
    unique_summary = {
        "episodes": len(unique_rows),
        "episodes_with_gross": sum(not row["no_gross"] for row in unique_rows),
        "episodes_starting_gross": sum(row["starts_gross"] is True for row in unique_rows),
        "episodes_without_gross": sum(row["no_gross"] is True for row in unique_rows),
        "episodes_without_static_prehistory": sum(row["no_static_prehistory"] is True for row in unique_rows),
        "stage_frames": {
            stage: sum(row[f"frames_{stage}"] for row in unique_rows) for stage in STAGES.values()
        },
        "pre_event_current_frames": {
            stage: sum(row[f"current_{stage}_pre_event"] for row in unique_rows)
            for stage in ("static", "incipient")
        },
        "post_gross_recovery_static_frames": sum(
            row["frames_recovery_after_first_gross"] for row in unique_rows
        ),
        "target_support": {
            stage: {
                str(horizon): {
                    measure: sum(row[f"{stage}_h{horizon}_{measure}"] for row in unique_rows)
                    for measure in ("eligible", "positive", "negative", "tail_masked", "positive_event")
                }
                for horizon in HORIZONS
            }
            for stage in ("static", "incipient")
        },
        "meta": {
            "incipient_window_values": sorted({row["meta_incipient_window"] for row in unique_rows}),
            "first_gross_equals_t_start": sum(row["meta_first_gross_equals_t_start"] is True for row in unique_rows),
            "last_gross_equals_t_stop": sum(row["meta_last_gross_equals_t_stop"] is True for row in unique_rows),
            "bracket_pattern_matches": sum(row["meta_bracket_pattern_matches"] is True for row in unique_rows),
        },
    }
    summary = {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "canonical_future_onset_source": str(EXPERIMENT_ROOT / "adapters.py"),
        "canonical_future_onset_source_sha256": sha256(EXPERIMENT_ROOT / "adapters.py"),
        "episode_data_partitions_read": list(DEV_PARTITIONS),
        "episode_data_partitions_not_read": ["test"],
        "test_partition_handling": "IDs used only to assert no overlap; no test episode data or labels opened",
        "horizons_steps": list(HORIZONS),
        "fold_read_counts": {fold: len(ids) for fold, ids in read_ids_by_fold.items()},
        "fold_read_ids": read_ids_by_fold,
        "cross_validation_context": (
            "Each fold excludes its held probe test partition. The same episode may appear in development "
            "roles in multiple other folds; summed fold rows are not independent global samples."
        ),
        "unique_labeled_episode_audit": unique_summary,
        "group_support": grouped,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "episode_event_support.csv", rows)
    write_csv(output_dir / "group_event_support.csv", grouped)
    (output_dir / "event_support.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(output_dir / "EVENT_SUPPORT.md", summary)
    print(json.dumps({
        "status": "passed",
        "fold_read_counts": summary["fold_read_counts"],
        "unique_labeled_episode_audit": unique_summary,
        "outputs": [str(output_dir / name) for name in (
            "episode_event_support.csv", "group_event_support.csv", "event_support.json", "EVENT_SUPPORT.md"
        )],
    }, indent=2))


if __name__ == "__main__":
    main()
