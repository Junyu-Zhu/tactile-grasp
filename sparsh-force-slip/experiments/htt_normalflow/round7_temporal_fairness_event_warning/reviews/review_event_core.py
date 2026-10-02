#!/usr/bin/env python3
"""Independent synthetic checks for the Round-7 event evaluation core."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    spec = importlib.util.spec_from_file_location("round7_event_core_under_review", path)
    value = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(value)
    return value


def row(episode: str, t: int, onset: int | None, target: int, score: float) -> dict:
    return {"episode_id": episode, "leakage_group": episode, "t": t, "onset": onset,
            "metric_mask": True, "target": target, "score": score}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    module = load(args.evaluator)
    checks = {}

    # An ongoing alarm that starts in the negative window is false-alarm burden,
    # active overlap, and not a newly detected event.
    ongoing = [row("ongoing", t, 4, target, score) for t, (target, score) in enumerate(((0, .1), (0, .9), (1, .9), (1, .9)))]
    summary, records = module.event_metrics(ongoing, "score", .5, "metric_mask")
    checks["ongoing_overlap_not_new_start"] = (
        summary["event_recall"] == 0 and summary["active_overlap_recall"] == 1
        and records[0]["ongoing_from_before_positive_window"]
        and records[0]["lead_frames"] is None and records[0]["false_alarm_starts"] == 1
    )

    # compact_event_curve must exactly match the full implementation at the same
    # thresholds for trial FA, event recall, and detected-event mean lead.
    parity_rows = ongoing + [
        row("new", 0, 4, 0, .1), row("new", 1, 4, 0, .2),
        row("new", 2, 4, 1, .8), row("new", 3, 4, 1, .9),
        row("negative", 0, None, 0, .1), row("negative", 1, None, 0, .7),
    ]
    parity = True
    details = []
    for threshold in (math.nextafter(1.0, math.inf), .85, .5, .15):
        full, _ = module.event_metrics(parity_rows, "score", threshold, "metric_mask")
        compact = module.compact_event_curve(parity_rows, "score", "metric_mask", [threshold])[0]
        current = {
            "threshold": threshold,
            "trial_false_alarm_rate_equal": compact["trial_false_alarm_rate"] == full["trial_false_alarm_rate"],
            "event_recall_equal": compact["event_recall"] == full["event_recall"],
            "mean_lead_equal": (
                math.isnan(compact["mean_lead_frames"]) and math.isnan(full["mean_lead_frames"])
            ) or compact["mean_lead_frames"] == full["mean_lead_frames"],
        }
        parity &= all(value for key, value in current.items() if key != "threshold")
        details.append(current)
    checks["compact_curve_parity"] = parity

    # A known first onset outside the feature-observable timeline and without an
    # eligible positive endpoint is right-censored, not an event miss or uncounted.
    censored = [row("censored", t, 8, 0, .1) for t in range(6)]
    censored_summary, censored_records = module.event_metrics(censored, "score", .5, "metric_mask")
    checks["known_onset_beyond_timeline_counted_right_censored"] = (
        not censored_records[0]["uncensored_event"]
        and censored_records[0]["right_censored_or_no_observed_onset"]
        and censored_summary["right_censored_or_no_observed_onset_trials"] == 1
    )

    # Mean duration over alarm runs is total alarm duration divided by the number
    # of observed maximal runs, rather than an unweighted mean of trial means.
    duration_rows = []
    for t, active in enumerate((1, 0, 1, 1, 1)):
        duration_rows.append(row("two_runs", t, None, 0, .9 if active else .1))
    duration_rows.append(row("one_run", 0, None, 0, .9))
    duration_summary, duration_records = module.event_metrics(duration_rows, "score", .5, "metric_mask")
    expected_mean = sum(record["false_alarm_duration"] for record in duration_records) / 3.0
    checks["mean_duration_weighted_by_alarm_runs"] = abs(duration_summary["mean_false_alarm_run_duration"] - expected_mean) < 1e-12
    checks["aggregate_max_duration_reported"] = duration_summary.get("max_false_alarm_duration") == 3

    # Cached and uncached calibration must select the same threshold, and distinct
    # score content must not reuse a stale cache entry.
    first = [row(f"a{i}", i, i + 1 if i % 2 else None, i % 2, score) for i, score in enumerate((.1, .9, .2, .8))]
    second = [row(f"a{i}", i, i + 1 if i % 2 else None, i % 2, score) for i, score in enumerate((.9, .1, .8, .2))]
    module._CAL_CACHE.clear()
    cached_first = module.select_threshold(first, "score", "metric_mask", "maxBA", None)[0]
    cached_second = module.select_threshold(second, "score", "metric_mask", "maxBA", None)[0]
    module._CAL_CACHE.clear()
    fresh_second = module.select_threshold(second, "score", "metric_mask", "maxBA", None)[0]
    checks["calibration_cache_content_bound"] = cached_first != cached_second and cached_second == fresh_second

    blockers = [name for name, passed in checks.items() if not passed]
    result = {
        "schema": "round7_event_core_independent_review_v1",
        "status": "pass" if not blockers else "changes_requested",
        "evaluator": {"path": str(args.evaluator.resolve()), "sha256": sha256(args.evaluator)},
        "checks": checks,
        "blockers": blockers,
        "details": {
            "compact_parity": details,
            "censor_record": censored_records[0],
            "duration_summary": duration_summary,
            "duration_expected_run_weighted_mean": expected_mean,
        },
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    print(json.dumps({"status": result["status"], "checks": len(checks), "blockers": blockers}))


if __name__ == "__main__":
    main()
