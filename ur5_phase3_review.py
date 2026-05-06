"""Phase 3 artifact alignment and label-quality review."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from ur5_phase3_schema import FRAME_MAP_COLUMNS, PROTOCOL_STAGES, TRIAL_REQUIRED_FIELDS, validate_metadata_fields

TACTILE_DIFF_PASS_THRESHOLD = 1.0e-5
TACTILE_CHANGED_PIXEL_FRACTION_THRESHOLD = 5.0e-4
TACTILE_CONTINUOUS_CHANGED_FRAME_THRESHOLD = 2
TACTILE_CONTACT_STAGES = {"contact_close", "hold", "micro_lift"}
REQUIRED_SUCCESS_CONTACT_SIDES = {"left", "right"}
MIN_MICRO_LIFT_OBJECT_FOLLOW_M = 0.0015


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_frame_map(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _trial_dirs(output_root: Path) -> list[Path]:
    return sorted(
        path
        for path in output_root.iterdir()
        if path.is_dir() and (path / "meta.json").is_file() and (path / "robot_state.json").is_file()
    )


def _tactile_response_summary(frame_rows: list[dict[str, str]]) -> dict[str, Any]:
    """Check that logged tactile RGB actually changes after contact.

    The original Phase3 review only verified that files existed.  That allowed
    constant TacEx background frames to pass even when the runtime sensor shell
    was not following the moving GSmini fingertip.  This mirrors the Phase2
    no-contact/contact comparison at trial-artifact level.
    """

    import numpy as np

    by_side: dict[str, dict[str, list[tuple[str, Path]]]] = {}
    for row in frame_rows:
        tactile_path = row.get("tactile_rgb_path", "")
        if not tactile_path:
            continue
        side = row.get("side", "")
        stage = row.get("action_stage", "")
        if not side or not Path(tactile_path).is_file():
            continue
        bucket = "contact" if stage in TACTILE_CONTACT_STAGES else "baseline"
        by_side.setdefault(side, {"baseline": [], "contact": []})[bucket].append((stage, Path(tactile_path)))

    side_results: dict[str, Any] = {}
    best_mean_abs_diff = 0.0
    best_changed_pixel_fraction = 0.0
    for side, buckets in by_side.items():
        if not buckets["baseline"] or not buckets["contact"]:
            side_results[side] = {
                "passed": False,
                "reason": "missing baseline or contact tactile_rgb frames",
            }
            continue
        baseline = np.load(buckets["baseline"][0][1]).astype("float32")
        side_best_mean = 0.0
        side_best_fraction = 0.0
        side_best_stage = None
        contact_frames: list[tuple[str, Any]] = []
        for stage, path in buckets["contact"]:
            contact = np.load(path).astype("float32")
            if contact.shape != baseline.shape:
                continue
            contact_frames.append((stage, contact))
            diff = np.abs(contact - baseline)
            mean_abs_diff = float(diff.mean()) if diff.size else 0.0
            changed_pixel_fraction = float((diff > TACTILE_DIFF_PASS_THRESHOLD).mean()) if diff.size else 0.0
            if mean_abs_diff > side_best_mean:
                side_best_mean = mean_abs_diff
                side_best_fraction = changed_pixel_fraction
                side_best_stage = stage
        consecutive_changed = 0
        max_consecutive_mean = 0.0
        previous = None
        for _stage, contact in contact_frames:
            if previous is not None and contact.shape == previous.shape:
                diff = np.abs(contact - previous)
                mean_abs_diff = float(diff.mean()) if diff.size else 0.0
                max_consecutive_mean = max(max_consecutive_mean, mean_abs_diff)
                if mean_abs_diff > TACTILE_DIFF_PASS_THRESHOLD:
                    consecutive_changed += 1
            previous = contact
        passed = (
            side_best_mean > TACTILE_DIFF_PASS_THRESHOLD
            and side_best_fraction > TACTILE_CHANGED_PIXEL_FRACTION_THRESHOLD
            and consecutive_changed >= TACTILE_CONTINUOUS_CHANGED_FRAME_THRESHOLD
        )
        side_results[side] = {
            "passed": passed,
            "best_stage": side_best_stage,
            "mean_abs_diff": side_best_mean,
            "changed_pixel_fraction": side_best_fraction,
            "contact_frame_count": len(contact_frames),
            "consecutive_changed_frames": consecutive_changed,
            "max_consecutive_mean_abs_diff": max_consecutive_mean,
            "threshold": TACTILE_DIFF_PASS_THRESHOLD,
            "changed_pixel_fraction_threshold": TACTILE_CHANGED_PIXEL_FRACTION_THRESHOLD,
            "continuous_changed_frame_threshold": TACTILE_CONTINUOUS_CHANGED_FRAME_THRESHOLD,
        }
        best_mean_abs_diff = max(best_mean_abs_diff, side_best_mean)
        best_changed_pixel_fraction = max(best_changed_pixel_fraction, side_best_fraction)
    return {
        "passed": bool(side_results) and all(result.get("passed") for result in side_results.values()),
        "best_mean_abs_diff": best_mean_abs_diff,
        "best_changed_pixel_fraction": best_changed_pixel_fraction,
        "sides": side_results,
    }


def _motion_success_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    bilateral_contact_step = None
    bilateral_contact_stage = None
    for sample in samples:
        if sample.get("action_stage") != "contact_close":
            continue
        sides = set(sample.get("contact_state", {}).get("contact_sides", []))
        if REQUIRED_SUCCESS_CONTACT_SIDES.issubset(sides):
            bilateral_contact_step = sample.get("index")
            bilateral_contact_stage = sample.get("action_stage")
            break

    hold_z = [
        sample.get("object_state", {}).get("root_position_m", [None, None, None])[2]
        for sample in samples
        if sample.get("action_stage") == "hold" and sample.get("object_state", {}).get("root_position_m")
    ]
    micro_lift_z = [
        sample.get("object_state", {}).get("root_position_m", [None, None, None])[2]
        for sample in samples
        if sample.get("action_stage") == "micro_lift" and sample.get("object_state", {}).get("root_position_m")
    ]
    pre_lift_z = hold_z[-1] if hold_z else None
    max_lift_z = max(micro_lift_z) if micro_lift_z else None
    object_lift_m = None
    if pre_lift_z is not None and max_lift_z is not None:
        object_lift_m = float(max_lift_z) - float(pre_lift_z)
    return {
        "bilateral_contact_passed": bilateral_contact_step is not None,
        "bilateral_contact_step": bilateral_contact_step,
        "bilateral_contact_stage": bilateral_contact_stage,
        "micro_lift_stage_present": bool(micro_lift_z),
        "pre_lift_object_z_m": pre_lift_z,
        "max_micro_lift_object_z_m": max_lift_z,
        "object_lift_m": object_lift_m,
        "min_required_object_lift_m": MIN_MICRO_LIFT_OBJECT_FOLLOW_M,
        "object_lift_passed": object_lift_m is not None and object_lift_m >= MIN_MICRO_LIFT_OBJECT_FOLLOW_M,
    }


def validate_trial_dir(trial_dir: Path) -> dict[str, Any]:
    meta_path = trial_dir / "meta.json"
    robot_state_path = trial_dir / "robot_state.json"
    frame_map_path = trial_dir / "frame_map.csv"
    issues: list[str] = []

    if not meta_path.is_file():
        return {"trial_dir": trial_dir.as_posix(), "passed": False, "issues": ["missing meta.json"]}
    meta = _load_json(meta_path)
    missing = validate_metadata_fields(meta)
    if missing:
        issues.append(f"meta missing required fields: {missing}")
    if not robot_state_path.is_file():
        issues.append("missing robot_state.json")
        robot_state = {"samples": []}
    else:
        robot_state = _load_json(robot_state_path)
    if not frame_map_path.is_file():
        issues.append("missing frame_map.csv")
        frame_rows: list[dict[str, str]] = []
    else:
        frame_rows = _read_frame_map(frame_map_path)

    samples = robot_state.get("samples", [])
    if not samples:
        issues.append("robot_state.json has no samples")
    if frame_rows:
        missing_columns = [column for column in FRAME_MAP_COLUMNS if column not in frame_rows[0]]
        if missing_columns:
            issues.append(f"frame_map missing columns: {missing_columns}")
    else:
        issues.append("frame_map has no tactile/frame rows")

    bad_state_indices: list[str] = []
    bad_stages: list[str] = []
    missing_tactile_files: list[str] = []
    for row in frame_rows:
        try:
            state_index = int(row.get("robot_state_index", "-1"))
        except ValueError:
            state_index = -1
        if state_index < 0 or state_index >= len(samples):
            bad_state_indices.append(row.get("frame_id", "unknown"))
        stage = row.get("action_stage", "")
        if stage not in PROTOCOL_STAGES:
            bad_stages.append(f"{row.get('frame_id', 'unknown')}:{stage}")
        for column in ("tactile_rgb_path", "camera_depth_path", "camera_rgb_path"):
            raw_path = row.get(column, "")
            if raw_path and not Path(raw_path).is_file():
                missing_tactile_files.append(raw_path)
    if bad_state_indices:
        issues.append(f"frame rows with invalid robot_state_index: {bad_state_indices[:10]}")
    if bad_stages:
        issues.append(f"frame rows with invalid stage: {bad_stages[:10]}")
    if missing_tactile_files:
        issues.append(f"missing tactile files referenced by frame_map: {missing_tactile_files[:10]}")

    motion_success = _motion_success_summary(samples)
    if not motion_success.get("bilateral_contact_passed"):
        issues.append("bilateral GSmini contact was not observed during contact_close")
    if not motion_success.get("micro_lift_stage_present"):
        issues.append("micro_lift stage was not observed")
    if not motion_success.get("object_lift_passed"):
        issues.append(
            "object did not follow the gripper during micro_lift "
            f"({float(motion_success.get('object_lift_m') or 0.0):.4f} m)"
        )

    tactile_response = _tactile_response_summary(frame_rows) if not missing_tactile_files else {"passed": False}
    if not tactile_response.get("passed"):
        issues.append("both tactile_rgb streams did not change continuously between baseline and contact/hold/lift frames")

    observed_stages = meta.get("action_stage", [])
    if "contact_close" not in observed_stages:
        issues.append("contact_close stage was not observed")
    if "release" not in observed_stages:
        issues.append("release stage was not observed")
    contact_onset = meta.get("contact_onset", {})
    if not isinstance(contact_onset, dict) or "detected" not in contact_onset:
        issues.append("contact_onset is not structured")
    if meta.get("success_label") and not contact_onset.get("detected"):
        issues.append("success_label=true but contact_onset.detected is false")
    alignment = meta.get("alignment_summary", {})
    if alignment.get("all_frames_have_robot_state") is not True:
        issues.append("alignment_summary does not prove all frames have robot state")

    return {
        "trial_dir": trial_dir.as_posix(),
        "trial_id": meta.get("trial_id"),
        "object_id": meta.get("object_id"),
        "protocol_variant": meta.get("protocol_variant"),
        "success_label": meta.get("success_label"),
        "failure_reason": meta.get("failure_reason"),
        "contact_onset": contact_onset,
        "robot_state_samples": len(samples),
        "frame_rows": len(frame_rows),
        "motion_success": motion_success,
        "tactile_response": tactile_response,
        "observed_stages": observed_stages,
        "passed": not issues,
        "issues": issues,
    }


def build_phase3_review(output_root: Path) -> dict[str, Any]:
    output_root = output_root.resolve()
    trials = [validate_trial_dir(path) for path in _trial_dirs(output_root)]
    failed_trials = [trial for trial in trials if not trial.get("passed") or not trial.get("success_label")]
    return {
        "output_root": output_root.as_posix(),
        "trial_count": len(trials),
        "artifact_validation_passed": bool(trials)
        and all(trial.get("passed") and trial.get("success_label") for trial in trials),
        "clean_trial_count": sum(1 for trial in trials if trial.get("passed") and trial.get("success_label")),
        "failed_trials": failed_trials,
        "trials": trials,
    }


def write_phase3_review(output_root: Path) -> Path:
    review = build_phase3_review(output_root)
    json_path = output_root / "phase3_review.json"
    md_path = output_root / "phase3_review.md"
    json_path.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Phase 3 Review",
        "",
        f"- Output root: `{review['output_root']}`",
        f"- Trial count: {review['trial_count']}",
        f"- Artifact validation passed: **{review['artifact_validation_passed']}**",
        f"- Clean trial count: {review['clean_trial_count']}",
        "",
        "| trial_id | object | protocol | success | contact_onset | frames | samples | issues |",
        "| --- | --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for trial in review["trials"]:
        onset = trial.get("contact_onset", {}).get("detected")
        issues = "; ".join(trial.get("issues", [])) or trial.get("failure_reason") or ""
        lines.append(
            f"| {trial.get('trial_id')} | {trial.get('object_id')} | {trial.get('protocol_variant')} | "
            f"{trial.get('success_label')} | {onset} | {trial.get('frame_rows')} | "
            f"{trial.get('robot_state_samples')} | {issues} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 3 trial artifacts and alignment.")
    parser.add_argument("--output_root", type=Path, default=Path("artifacts/phase3"))
    args = parser.parse_args()
    review_path = write_phase3_review(args.output_root)
    review = _load_json(review_path)
    print(json.dumps(review, ensure_ascii=False, indent=2))
    return 0 if review.get("artifact_validation_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
