"""Phase 3 artifact alignment and label-quality review."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from protac.compat.phase3_v1_schema import FRAME_MAP_COLUMNS, PROTOCOL_STAGES, validate_metadata_fields

# Match the established Phase2 response floor and require a spatially visible
# patch; sub-pixel renderer noise must not count as a meaningful tactile view.
TACTILE_DIFF_PASS_THRESHOLD = 1.0e-4
TACTILE_CHANGED_PIXEL_FRACTION_THRESHOLD = 1.0e-2
TACTILE_CONTINUOUS_CHANGED_FRAME_THRESHOLD = 2
TACTILE_CONTACT_STAGES = {"contact_close", "hold", "micro_lift"}
TACTILE_BASELINE_STAGES = {"reset", "pre_grasp"}
TACTILE_DEPTH_FAR_PLANE_MARGIN_M = 2.0e-4
TACTILE_DEPTH_MIN_SPATIAL_SPAN_M = 5.0e-5
REQUIRED_SUCCESS_CONTACT_SIDES = {"left", "right"}
MIN_MICRO_LIFT_OBJECT_FOLLOW_M = 0.015
ADAPTOR_CONTACT_FORCE_TOLERANCE_N = 0.05


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


def _tactile_baseline_stability_summary(
    frame_rows: list[dict[str, str]],
    *,
    sides: list[str],
) -> dict[str, Any]:
    """Require the no-contact reset/pre-grasp image to remain visually stable."""

    import numpy as np

    baseline_paths: dict[str, list[Path]] = {side: [] for side in sides}
    for row in frame_rows:
        side = row.get("side", "")
        tactile_path = Path(row.get("tactile_rgb_path", ""))
        physical_contact = str(row.get("contact_detected", "")).strip().lower() == "true"
        if (
            side in baseline_paths
            and row.get("action_stage") in TACTILE_BASELINE_STAGES
            and not physical_contact
            and tactile_path.is_file()
        ):
            baseline_paths[side].append(tactile_path)

    side_results: dict[str, Any] = {}
    for side, paths in baseline_paths.items():
        if len(paths) < 2:
            side_results[side] = {
                "passed": False,
                "frame_count": len(paths),
                "reason": "fewer than two no-contact reset/pre_grasp tactile_rgb frames",
            }
            continue
        reference = np.load(paths[0]).astype("float32")
        max_mean_abs_diff = 0.0
        max_changed_pixel_fraction = 0.0
        shape_mismatch_count = 0
        for path in paths[1:]:
            frame = np.load(path).astype("float32")
            if frame.shape != reference.shape:
                shape_mismatch_count += 1
                continue
            diff = np.abs(frame - reference)
            max_mean_abs_diff = max(
                max_mean_abs_diff,
                float(diff.mean()) if diff.size else 0.0,
            )
            max_changed_pixel_fraction = max(
                max_changed_pixel_fraction,
                float((diff > TACTILE_DIFF_PASS_THRESHOLD).mean()) if diff.size else 0.0,
            )
        passed = (
            shape_mismatch_count == 0
            and max_mean_abs_diff <= TACTILE_DIFF_PASS_THRESHOLD
            and max_changed_pixel_fraction <= TACTILE_CHANGED_PIXEL_FRACTION_THRESHOLD
        )
        side_results[side] = {
            "passed": passed,
            "frame_count": len(paths),
            "max_mean_abs_diff": max_mean_abs_diff,
            "max_changed_pixel_fraction": max_changed_pixel_fraction,
            "mean_abs_diff_threshold": TACTILE_DIFF_PASS_THRESHOLD,
            "changed_pixel_fraction_threshold": TACTILE_CHANGED_PIXEL_FRACTION_THRESHOLD,
            "shape_mismatch_count": shape_mismatch_count,
        }
    return {
        "passed": bool(side_results) and all(result.get("passed") for result in side_results.values()),
        "sides": side_results,
    }


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
        physical_gelpad_contact = str(row.get("contact_detected", "")).strip().lower() == "true"
        if stage in TACTILE_CONTACT_STAGES and physical_gelpad_contact:
            bucket = "contact"
        elif stage in TACTILE_BASELINE_STAGES or stage in TACTILE_CONTACT_STAGES:
            # Keep updating the baseline while the object approaches.  This
            # prevents camera proximity or adaptor-only contact from being
            # mislabelled as a tactile response.
            bucket = "baseline"
        else:
            continue
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
        baseline = np.load(buckets["baseline"][-1][1]).astype("float32")
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
        max_consecutive_changed = 0
        max_contact_to_contact_mean = 0.0
        previous = None
        for _stage, contact in contact_frames:
            baseline_diff = np.abs(contact - baseline)
            baseline_mean = float(baseline_diff.mean()) if baseline_diff.size else 0.0
            baseline_fraction = (
                float((baseline_diff > TACTILE_DIFF_PASS_THRESHOLD).mean())
                if baseline_diff.size
                else 0.0
            )
            changed_vs_baseline = (
                baseline_mean > TACTILE_DIFF_PASS_THRESHOLD
                and baseline_fraction > TACTILE_CHANGED_PIXEL_FRACTION_THRESHOLD
            )
            consecutive_changed = consecutive_changed + 1 if changed_vs_baseline else 0
            max_consecutive_changed = max(max_consecutive_changed, consecutive_changed)
            if previous is not None and contact.shape == previous.shape:
                diff = np.abs(contact - previous)
                mean_abs_diff = float(diff.mean()) if diff.size else 0.0
                max_contact_to_contact_mean = max(max_contact_to_contact_mean, mean_abs_diff)
            previous = contact
        passed = (
            side_best_mean > TACTILE_DIFF_PASS_THRESHOLD
            and side_best_fraction > TACTILE_CHANGED_PIXEL_FRACTION_THRESHOLD
            and max_consecutive_changed >= TACTILE_CONTINUOUS_CHANGED_FRAME_THRESHOLD
        )
        side_results[side] = {
            "passed": passed,
            "best_stage": side_best_stage,
            "mean_abs_diff": side_best_mean,
            "changed_pixel_fraction": side_best_fraction,
            "contact_frame_count": len(contact_frames),
            "consecutive_changed_frames": max_consecutive_changed,
            "max_contact_to_contact_mean_abs_diff": max_contact_to_contact_mean,
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
        contact_state = sample.get("contact_state", {})
        force_sides = set(contact_state.get("force_contact_sides", []))
        if contact_state.get("both_sides_force_contact") or REQUIRED_SUCCESS_CONTACT_SIDES.issubset(force_sides):
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
    grasp_contact_states = [
        sample.get("contact_state", {})
        for sample in samples
        if sample.get("action_stage") in TACTILE_CONTACT_STAGES
    ]
    max_adaptor_force_n = max(
        (
            float(state.get("max_adaptor_force_n", 0.0) or 0.0)
            for state in grasp_contact_states
        ),
        default=0.0,
    )
    adaptor_sensor_valid = bool(grasp_contact_states) and all(
        state.get("adaptor_sensor_valid") is True for state in grasp_contact_states
    )
    adaptor_clearance_passed = (
        adaptor_sensor_valid
        and max_adaptor_force_n <= ADAPTOR_CONTACT_FORCE_TOLERANCE_N
    )
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
        "adaptor_sensor_valid": adaptor_sensor_valid,
        "max_adaptor_force_n": max_adaptor_force_n,
        "adaptor_contact_force_tolerance_n": ADAPTOR_CONTACT_FORCE_TOLERANCE_N,
        "adaptor_clearance_passed": adaptor_clearance_passed,
    }


def _tactile_optical_depth_summary(
    samples: list[dict[str, Any]],
    *,
    sides: list[str],
    far_plane_m: float,
) -> dict[str, Any]:
    """Prove each TacEx camera observed geometry rather than its far plane.

    RGB-only comparison can be fooled by small renderer/background variations.
    The detached camera must also report a finite surface closer than the far
    plane and a non-flat spatial depth range during contact.  This caught the
    stale-camera-xform failure where both RGB streams changed slightly while
    every Taxim height-map pixel remained exactly at 32 mm.
    """

    results: dict[str, Any] = {}
    for side in sides:
        frame_count = 0
        closest_depth_m: float | None = None
        max_spatial_span_m = 0.0
        for sample in samples:
            if sample.get("action_stage") not in TACTILE_CONTACT_STAGES:
                continue
            summary = (
                sample.get("contact_state", {})
                .get("tactile_capture", {})
                .get("sides", {})
                .get(side, {})
                .get("camera_observation", {})
                .get("taxim_height_map", {})
            )
            if not summary.get("available"):
                continue
            min_depth = summary.get("min_m")
            max_depth = summary.get("max_m")
            if min_depth is None or max_depth is None:
                continue
            min_depth = float(min_depth)
            max_depth = float(max_depth)
            frame_count += 1
            closest_depth_m = min_depth if closest_depth_m is None else min(closest_depth_m, min_depth)
            max_spatial_span_m = max(max_spatial_span_m, max_depth - min_depth)
        observed_surface = (
            closest_depth_m is not None
            and closest_depth_m < far_plane_m - TACTILE_DEPTH_FAR_PLANE_MARGIN_M
        )
        spatially_resolved = max_spatial_span_m >= TACTILE_DEPTH_MIN_SPATIAL_SPAN_M
        results[side] = {
            "passed": observed_surface and spatially_resolved,
            "frame_count": frame_count,
            "closest_depth_m": closest_depth_m,
            "far_plane_m": far_plane_m,
            "far_plane_margin_m": TACTILE_DEPTH_FAR_PLANE_MARGIN_M,
            "max_spatial_span_m": max_spatial_span_m,
            "min_spatial_span_m": TACTILE_DEPTH_MIN_SPATIAL_SPAN_M,
            "observed_surface": observed_surface,
            "spatially_resolved": spatially_resolved,
        }
    return {
        "passed": bool(results) and all(result["passed"] for result in results.values()),
        "sides": results,
    }


def validate_trial_dir(trial_dir: Path) -> dict[str, Any]:
    meta_path = trial_dir / "meta.json"
    robot_state_path = trial_dir / "robot_state.json"
    frame_map_path = trial_dir / "frame_map.csv"
    issues: list[str] = []

    if not meta_path.is_file():
        return {"trial_dir": trial_dir.as_posix(), "passed": False, "issues": ["missing meta.json"]}
    meta = _load_json(meta_path)
    command_profile = meta.get("command_profile", {})
    tactile_setup = command_profile.get("tactile_setup", {})
    tactile_enabled = bool(tactile_setup.get("enabled"))
    tactile_sides = list(tactile_setup.get("sides", [])) if tactile_enabled else []
    missing = validate_metadata_fields(meta)
    if missing:
        issues.append(f"meta missing required fields: {missing}")
    robot_state: dict[str, Any]
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
    elif tactile_enabled:
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
    if not motion_success.get("adaptor_clearance_passed"):
        issues.append("adaptor/connector contacted the object or its contact sensor was unavailable")
    if not motion_success.get("bilateral_contact_passed"):
        issues.append("bilateral GSmini contact was not observed during contact_close")
    if not motion_success.get("micro_lift_stage_present"):
        issues.append("micro_lift stage was not observed")
    if not motion_success.get("object_lift_passed"):
        issues.append(
            "object did not follow the gripper during micro_lift "
            f"({float(motion_success.get('object_lift_m') or 0.0):.4f} m)"
        )

    if tactile_enabled:
        tactile_baseline_stability = (
            _tactile_baseline_stability_summary(frame_rows, sides=tactile_sides)
            if not missing_tactile_files
            else {"passed": False}
        )
        tactile_response = (
            _tactile_response_summary(frame_rows)
            if not missing_tactile_files
            else {"passed": False}
        )
        camera_checks = tactile_setup.get("camera_check", {})
        configured_far_planes = [
            float(check["clipping_range_m"][1])
            for side, check in camera_checks.items()
            if side in tactile_sides and check.get("clipping_range_m")
        ]
        far_plane_m = max(configured_far_planes, default=0.032)
        tactile_optical_depth = _tactile_optical_depth_summary(
            samples,
            sides=tactile_sides,
            far_plane_m=far_plane_m,
        )
        if not tactile_baseline_stability.get("passed"):
            issues.append("no-contact tactile_rgb streams changed during reset/pre_grasp")
        if not tactile_response.get("passed"):
            issues.append(
                "both tactile_rgb streams did not change continuously between baseline and contact/hold/lift frames"
            )
        if not tactile_optical_depth.get("passed"):
            issues.append(
                "TacEx contact height maps did not resolve object geometry below the camera far plane"
            )
    else:
        tactile_baseline_stability = {"passed": True, "skipped": "tactile disabled by runner options"}
        tactile_response = {"passed": True, "skipped": "tactile disabled by runner options"}
        tactile_optical_depth = {"passed": True, "skipped": "tactile disabled by runner options"}

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
        "tactile_baseline_stability": tactile_baseline_stability,
        "tactile_response": tactile_response,
        "tactile_optical_depth": tactile_optical_depth,
        "observed_stages": observed_stages,
        "passed": not issues,
        "issues": issues,
    }


def build_phase3_review(
    output_root: Path,
    *,
    trial_dirs: list[Path] | None = None,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    selected_dirs = _trial_dirs(output_root) if trial_dirs is None else [Path(path).resolve() for path in trial_dirs]
    trials = [validate_trial_dir(path) for path in selected_dirs]
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


def write_phase3_review(output_root: Path, *, trial_dirs: list[Path] | None = None) -> Path:
    review = build_phase3_review(output_root, trial_dirs=trial_dirs)
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
