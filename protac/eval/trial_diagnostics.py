"""Dependency-free Phase 3 trial diagnostics.

This module intentionally uses only the Python standard library so Phase 3
robot-state artifacts can be inspected without launching Isaac Sim.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

GRIPPER_JOINTS: tuple[str, ...] = (
    "finger_joint",
    "left_inner_finger_joint",
    "left_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    "right_inner_finger_joint",
    "right_inner_knuckle_joint",
)
DIAGNOSTIC_STAGES: tuple[str, ...] = ("contact_close", "hold")
SIDES: tuple[str, ...] = ("left", "right")
AXES: tuple[str, ...] = ("x", "y", "z")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _stat_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "peak": None}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "peak": max(abs(value) for value in values),
    }


def _span(points: list[list[float]]) -> dict[str, float | None]:
    spans: dict[str, float | None] = {}
    for axis_index, axis_name in enumerate(AXES):
        values = [point[axis_index] for point in points if len(point) > axis_index]
        spans[axis_name] = max(values) - min(values) if values else None
    return spans


def _resolve_robot_state_path(path: Path, *, trial_id: str | None = None) -> Path:
    if path.is_dir():
        candidate = path / "robot_state.json"
        if candidate.is_file():
            return candidate
        if trial_id is not None:
            selected = path / trial_id / "robot_state.json"
            if selected.is_file():
                return selected
        trial_dirs = sorted(child for child in path.iterdir() if child.is_dir() and (child / "robot_state.json").is_file())
        if len(trial_dirs) == 1:
            return trial_dirs[0] / "robot_state.json"
        raise FileNotFoundError(f"Could not resolve one robot_state.json under {path}")
    if path.name == "phase3_batch_summary.json":
        root = path.parent
        if trial_id is not None:
            selected = root / trial_id / "robot_state.json"
            if selected.is_file():
                return selected
        trial_dirs = sorted(child for child in root.iterdir() if child.is_dir() and (child / "robot_state.json").is_file())
        if len(trial_dirs) == 1:
            return trial_dirs[0] / "robot_state.json"
        raise FileNotFoundError(f"Could not resolve one robot_state.json next to {path}")
    return path


def _find_summary(path: Path, robot_state_path: Path) -> dict[str, Any] | None:
    candidates: list[Path] = []
    if path.is_file() and path.name == "phase3_batch_summary.json":
        candidates.append(path)
    if path.is_dir():
        candidates.append(path / "phase3_batch_summary.json")
    candidates.extend(
        [
            robot_state_path.parent / "phase3_batch_summary.json",
            robot_state_path.parent.parent / "phase3_batch_summary.json",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return _load_json(candidate)
    return None


def _find_trial_result(summary: dict[str, Any] | None, trial_id: str | None) -> dict[str, Any] | None:
    if not summary:
        return None
    results = summary.get("results")
    if not isinstance(results, list):
        return None
    if trial_id is None and len(results) == 1 and isinstance(results[0], dict):
        return results[0]
    for result in results:
        if isinstance(result, dict) and result.get("trial_id") == trial_id:
            return result
    return None


def _trial_success_and_lift(trial_result: dict[str, Any] | None, robot_state_path: Path) -> dict[str, Any]:
    meta: dict[str, Any] | None = None
    meta_path = robot_state_path.with_name("meta.json")
    if meta_path.is_file():
        meta = _load_json(meta_path)

    success = None
    if trial_result and "passed" in trial_result:
        success = bool(trial_result["passed"])
    elif meta and "success_label" in meta:
        success = bool(meta["success_label"])

    lift: dict[str, Any] = {"passed": None, "object_lift_m": None}
    micro_lift = trial_result.get("micro_lift") if trial_result else None
    if isinstance(micro_lift, dict):
        if "object_lift_passed" in micro_lift:
            lift["passed"] = bool(micro_lift["object_lift_passed"])
        lift_value = _safe_float(micro_lift.get("object_lift_m"))
        if lift_value is not None:
            lift["object_lift_m"] = lift_value
    return {"success": success, "lift": lift}


def analyze_trial(path: str | Path, *, trial_id: str | None = None) -> dict[str, Any]:
    """Return Phase 3 diagnostics for one trial path or summary path."""

    input_path = Path(path)
    robot_state_path = _resolve_robot_state_path(input_path, trial_id=trial_id)
    robot_state = _load_json(robot_state_path)
    samples = robot_state.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError(f"Expected samples list in {robot_state_path}")

    resolved_trial_id = trial_id or str(robot_state.get("trial_id") or robot_state_path.parent.name)
    summary = _find_summary(input_path, robot_state_path)
    trial_result = _find_trial_result(summary, resolved_trial_id)

    peak_velocity: dict[str, dict[str, float | None]] = {
        stage: {joint: None for joint in GRIPPER_JOINTS} for stage in DIAGNOSTIC_STAGES
    }
    gel_centers: dict[str, list[list[float]]] = {side: [] for side in SIDES}
    forces: dict[str, list[float]] = {side: [] for side in SIDES}

    for sample in samples:
        if not isinstance(sample, dict):
            continue
        stage = sample.get("action_stage")
        raw_contact_state = sample.get("contact_state")
        contact_state = raw_contact_state if isinstance(raw_contact_state, dict) else {}
        contact_active = bool(contact_state.get("contact_detected"))
        raw_joint_state = sample.get("joint_state")
        joint_state = raw_joint_state if isinstance(raw_joint_state, dict) else {}
        velocities = joint_state.get("velocity_rad_s")
        if stage in DIAGNOSTIC_STAGES and contact_active and isinstance(velocities, dict):
            for joint in GRIPPER_JOINTS:
                value = _safe_float(velocities.get(joint))
                if value is None:
                    continue
                current = peak_velocity[stage][joint]
                peak_velocity[stage][joint] = abs(value) if current is None else max(current, abs(value))

        if stage not in DIAGNOSTIC_STAGES or not contact_active:
            continue
        force_by_side = contact_state.get("force_by_side_n") if isinstance(contact_state, dict) else {}
        if isinstance(force_by_side, dict):
            for side in SIDES:
                value = _safe_float(force_by_side.get(side))
                if value is not None:
                    forces[side].append(value)

        geometry = contact_state.get("geometry") if isinstance(contact_state, dict) else {}
        sides = geometry.get("sides") if isinstance(geometry, dict) else {}
        if isinstance(sides, dict):
            for side in SIDES:
                side_geometry = sides.get(side)
                if not isinstance(side_geometry, dict):
                    continue
                center = side_geometry.get("center_world_m")
                if not isinstance(center, list):
                    continue
                point = [_safe_float(value) for value in center[:3]]
                if all(value is not None for value in point):
                    gel_centers[side].append([float(value) for value in point if value is not None])

    step_budget_used = _safe_float(trial_result.get("step_budget_used")) if trial_result else None
    result = {
        "trial_id": resolved_trial_id,
        "robot_state_path": robot_state_path.as_posix(),
        "summary_path": None,
        "total_steps": int(step_budget_used) if step_budget_used is not None else len(samples),
        "recorded_samples": len(samples),
        "trial": _trial_success_and_lift(trial_result, robot_state_path),
        "gripper_peak_velocity_rad_s": peak_velocity,
        "gel_aabb_center_span_m": {side: _span(points) for side, points in gel_centers.items()},
        "contact_force_n": {side: _stat_summary(values) for side, values in forces.items()},
    }
    if summary is not None:
        result["summary_path"] = (
            input_path.as_posix()
            if input_path.is_file() and input_path.name == "phase3_batch_summary.json"
            else (robot_state_path.parent.parent / "phase3_batch_summary.json").as_posix()
        )
    return result


def _flatten_numeric(data: Any, prefix: str = "") -> dict[str, float]:
    values: dict[str, float] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            values.update(_flatten_numeric(value, child_prefix))
    elif isinstance(data, (int, float)) and not isinstance(data, bool):
        value = _safe_float(data)
        if value is not None:
            values[prefix] = value
    return values


def compare_trials(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Compare candidate against baseline.

    ``reduction_rate`` is positive when the candidate is lower than baseline.
    This is the desired direction for jitter/velocity diagnostics.
    """

    baseline_values = _flatten_numeric(baseline)
    candidate_values = _flatten_numeric(candidate)
    rates: dict[str, float | None] = {}
    for key, baseline_value in baseline_values.items():
        if key in {"trial.success", "trial.lift.passed"}:
            continue
        candidate_value = candidate_values.get(key)
        if candidate_value is None:
            continue
        if baseline_value == 0.0:
            rates[key] = None if candidate_value != 0.0 else 0.0
        else:
            rates[key] = (baseline_value - candidate_value) / abs(baseline_value)

    return {
        "mode": "compare",
        "baseline": baseline,
        "candidate": candidate,
        "candidate_vs_baseline": {
            "positive_rate_means_candidate_lower": True,
            "improvement_rates": rates,
            "success_delta": _bool_delta(
                baseline.get("trial", {}).get("success"),
                candidate.get("trial", {}).get("success"),
            ),
            "lift_passed_delta": _bool_delta(
                baseline.get("trial", {}).get("lift", {}).get("passed"),
                candidate.get("trial", {}).get("lift", {}).get("passed"),
            ),
        },
    }


def _bool_delta(baseline: Any, candidate: Any) -> int | None:
    if not isinstance(baseline, bool) or not isinstance(candidate, bool):
        return None
    return int(candidate) - int(baseline)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze Phase 3 trial jitter, velocity, and force diagnostics.")
    parser.add_argument("baseline", help="Trial directory, robot_state.json, or phase3_batch_summary.json")
    parser.add_argument("candidate", nargs="?", help="Optional candidate trial for compare mode")
    parser.add_argument("--trial-id", help="Trial id to select when reading a batch summary")
    parser.add_argument("--candidate-trial-id", help="Candidate trial id to select when reading a batch summary")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation; use 0 for compact output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    baseline = analyze_trial(args.baseline, trial_id=args.trial_id)
    if args.candidate:
        candidate = analyze_trial(args.candidate, trial_id=args.candidate_trial_id)
        output = compare_trials(baseline, candidate)
    else:
        output = {"mode": "single", "trial": baseline}
    indent = None if args.indent == 0 else args.indent
    json.dump(output, sys.stdout, ensure_ascii=False, indent=indent, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
