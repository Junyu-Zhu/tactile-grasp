"""Strict Round-8 prediction schema and provenance checks."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Sequence

try:
    from .core import cumulative_risk_from_hazard
except ImportError:
    from core import cumulative_risk_from_hazard

BASE_COLUMNS = {
    "episode_id", "leakage_group", "t", "first_current_slip_t",
    "current_slip_label", "p_slip_current", "timeline_contiguous", "common_population",
}


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).lower() in {"true", "1"}:
        return True
    if str(value).lower() in {"false", "0"}:
        return False
    raise ValueError(f"invalid boolean {value!r}")


def _probability(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"invalid probability {name}={value!r}")
    return result


def load_timeline_csv(path: Path, horizons: Sequence[int] = (1, 3, 5), *, head_type: str) -> list[dict[str, Any]]:
    with Path(path).open(newline="") as handle:
        raw = list(csv.DictReader(handle))
    if not raw:
        raise ValueError(f"empty timeline {path}")
    required = set(BASE_COLUMNS)
    for horizon in horizons:
        required.update({f"eligible_H{horizon}", f"common_eligible_H{horizon}", f"right_censored_H{horizon}",
                         f"target_future_H{horizon}", f"p_future_H{horizon}_raw"})
    if missing := required - set(raw[0]):
        raise ValueError(f"timeline schema missing {sorted(missing)}")
    if head_type not in {"independent_sigmoid", "discrete_hazard"}:
        raise ValueError(f"unknown head_type {head_type!r}")
    if head_type == "discrete_hazard":
        hazard_columns = {f"q_future_step{step}_raw" for step in range(1, max(horizons) + 1)}
        if missing := hazard_columns - set(raw[0]):
            raise ValueError(f"hazard timeline missing {sorted(missing)}")

    result: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for source in raw:
        identity = (source["episode_id"], int(source["t"]))
        if identity in seen:
            raise ValueError(f"duplicate timeline row {identity}")
        seen.add(identity)
        onset_text = source["first_current_slip_t"]
        row: dict[str, Any] = {
            "episode_id": source["episode_id"], "leakage_group": source["leakage_group"], "t": int(source["t"]),
            "onset": None if onset_text in {"", "None"} else int(onset_text),
            "current_slip_label": int(source["current_slip_label"]),
            "p_slip_current": _probability(source["p_slip_current"], "p_slip_current"),
            "timeline_contiguous": _bool(source["timeline_contiguous"]), "common_population": _bool(source["common_population"]),
        }
        optional_float_columns = {
            "force_x": "force_x", "force_y": "force_y", "force_z": "force_z",
            "predicted_force_x": "force_x", "predicted_force_y": "force_y", "predicted_force_z": "force_z",
            "dforce_x": "delta_x", "dforce_y": "delta_y", "dforce_z": "delta_z",
            "latest_force_delta_x": "delta_x", "latest_force_delta_y": "delta_y", "latest_force_delta_z": "delta_z",
        }
        for source_name, internal_name in optional_float_columns.items():
            if source_name in source and source[source_name] not in {"", None}:
                value = float(source[source_name])
                if internal_name in row and not math.isclose(float(row[internal_name]), value, rel_tol=0.0, abs_tol=1e-7):
                    raise ValueError(f"conflicting aliases for {internal_name} at {identity}")
                row[internal_name] = value
        row.setdefault("delta_x", 0.0); row.setdefault("delta_y", 0.0); row.setdefault("delta_z", 0.0)
        if "latest_force_delta_valid" in source and source["latest_force_delta_valid"] not in {"", None}:
            row["delta_valid"] = _bool(source["latest_force_delta_valid"])
        for horizon in horizons:
            eligible = _bool(source[f"eligible_H{horizon}"])
            target_text = source[f"target_future_H{horizon}"]
            target = None if target_text == "" else int(target_text)
            if eligible != (target is not None) or target not in {None, 0, 1}:
                raise ValueError(f"target/mask drift {identity}/H{horizon}")
            if eligible:
                if row["current_slip_label"] != 0:
                    raise ValueError(f"eligible future endpoint is already slipping {identity}/H{horizon}")
                onset = row["onset"]
                if onset is not None and row["t"] >= onset:
                    raise ValueError(f"eligible future endpoint is at/after onset {identity}/H{horizon}")
                expected_target = int(onset is not None and onset <= row["t"] + horizon)
                if target != expected_target:
                    raise ValueError(f"future target/onset drift {identity}/H{horizon}")
            row[f"eligible_H{horizon}"] = eligible
            row[f"common_H{horizon}"] = _bool(source[f"common_eligible_H{horizon}"])
            row[f"right_censored_H{horizon}"] = _bool(source[f"right_censored_H{horizon}"])
            row[f"target_H{horizon}"] = target
            row[f"p_H{horizon}"] = _probability(source[f"p_future_H{horizon}_raw"], f"p_future_H{horizon}_raw")
        if head_type == "discrete_hazard":
            hazards = [_probability(source[f"q_future_step{step}_raw"], f"q_future_step{step}_raw")
                       for step in range(1, max(horizons) + 1)]
            cumulative = cumulative_risk_from_hazard(hazards)
            for horizon in horizons:
                if not math.isclose(row[f"p_H{horizon}"], float(cumulative[horizon - 1]), rel_tol=0.0, abs_tol=1e-6):
                    raise ValueError(f"hazard/cumulative mismatch {identity}/H{horizon}")
            row["hazard"] = hazards
        state_steps = []
        for step in range(1, max(horizons) + 1):
            mask_name = f"state_target_mask_tplus{step}"
            predicted_names = [f"predicted_state_force_tplus{step}_{axis}" for axis in "xyz"]
            target_names = [f"target_state_force_tplus{step}_{axis}" for axis in "xyz"]
            present = [name in source for name in [mask_name, *predicted_names, *target_names]]
            if any(present) and not all(present):
                raise ValueError(f"incomplete state schema {identity}/step{step}")
            if all(present):
                mask = _bool(source[mask_name])
                prediction = [float(source[name]) if source[name] != "" else math.nan for name in predicted_names]
                target = [float(source[name]) if source[name] != "" else math.nan for name in target_names]
                if mask and (not np_isfinite(prediction) or not np_isfinite(target)):
                    raise ValueError(f"nonfinite valid state target {identity}/step{step}")
                row[f"state_mask_step{step}"] = mask
                row[f"state_prediction_step{step}"] = prediction
                row[f"state_target_step{step}"] = target
                state_steps.append(step)
        if state_steps:
            row["state_steps"] = state_steps
        result.append(row)

    by_episode: dict[str, list[dict[str, Any]]] = {}
    for row in result:
        by_episode.setdefault(row["episode_id"], []).append(row)
    for episode, sequence in by_episode.items():
        sequence.sort(key=lambda row: row["t"])
        times = [row["t"] for row in sequence]
        if times != list(range(times[0], times[-1] + 1)):
            raise ValueError(f"non-contiguous episode {episode}")
        if sequence[0]["timeline_contiguous"] or any(not row["timeline_contiguous"] for row in sequence[1:]):
            raise ValueError(f"timeline_contiguous flag drift {episode}")
        if len({row["leakage_group"] for row in sequence}) != 1 or len({row["onset"] for row in sequence}) != 1:
            raise ValueError(f"episode metadata drift {episode}")
    return result


def np_isfinite(values: Sequence[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def validate_cross_run_identity(reference: Sequence[dict[str, Any]], candidate: Sequence[dict[str, Any]], horizons: Sequence[int] = (1, 3, 5)) -> None:
    def identity(row: dict[str, Any]) -> tuple[Any, ...]:
        return (row["episode_id"], row["leakage_group"], row["t"], row["onset"], row["current_slip_label"],
                row["common_population"], *(row[f"eligible_H{horizon}"] for horizon in horizons),
                *(row[f"target_H{horizon}"] for horizon in horizons))
    if [identity(row) for row in reference] != [identity(row) for row in candidate]:
        raise ValueError("cross-run population or label identity drift")
    upstream_fields = ("p_slip_current", "force_x", "force_y", "force_z", "delta_x", "delta_y", "delta_z")
    for left, right in zip(reference, candidate):
        for field in upstream_fields:
            if field in left and field in right and not math.isclose(
                float(left[field]), float(right[field]), rel_tol=0.0, abs_tol=1e-7
            ):
                raise ValueError(f"cross-run frozen-upstream identity drift: {field}")
        if "delta_valid" in left and "delta_valid" in right and left["delta_valid"] != right["delta_valid"]:
            raise ValueError("cross-run frozen-upstream identity drift: delta_valid")
    reference_has_state = any("state_steps" in row for row in reference)
    candidate_has_state = any("state_steps" in row for row in candidate)
    if reference_has_state and candidate_has_state:
        for left, right in zip(reference, candidate):
            if left.get("state_steps") != right.get("state_steps"):
                raise ValueError("cross-run state-step identity drift")
            for step in left.get("state_steps", []):
                if left[f"state_mask_step{step}"] != right[f"state_mask_step{step}"]:
                    raise ValueError("cross-run state-mask identity drift")
                if left[f"state_mask_step{step}"] and any(
                    not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-6)
                    for a, b in zip(left[f"state_target_step{step}"], right[f"state_target_step{step}"])
                ):
                    raise ValueError("cross-run state-target identity drift")
