"""Phase 3 artifact alignment and label-quality review."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from ur5_phase3_schema import FRAME_MAP_COLUMNS, PROTOCOL_STAGES, TRIAL_REQUIRED_FIELDS, validate_metadata_fields


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
        "artifact_validation_passed": bool(trials) and all(trial.get("passed") for trial in trials),
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
