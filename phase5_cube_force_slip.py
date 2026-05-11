"""Build Phase 5 cube force/slip v2 labels and Sparsh export.

Phase 5 is deliberately narrower than model training: it hardens cube-only
simulation labels so downstream Sparsh checks can distinguish format success
from metric-valid force/slip targets.  Heavy exported samples stay under the
repo-local ``sim_dataset/`` tree; reviewable reports are written to
``artifacts/phase5_cube_force_slip/reports``.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import pickle
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

DATASET_NAME = "cube_force_slip_v2"
DEFAULT_OUTPUT_ROOT = Path("sim_dataset/phase5_cube_force_slip")
DEFAULT_EXPORT_ROOT = DEFAULT_OUTPUT_ROOT / "sparsh_export_v2"
DEFAULT_REPORT_ROOT = Path("artifacts/phase5_cube_force_slip/reports")
FORCE_FRAME_NAME = "phase5_gripper_scalar_normal_v1"
SLIP_FRAME_NAME = "phase5_gripper_contact_frame_v1"
SLIP_VALID_STAGES = {"contact_close", "hold", "micro_lift"}
SLIP_MASKED_STAGES = {"release", "end_trial"}
FORCE_LEVEL_THRESHOLDS_N = {
    "light_upper": 0.25,
    "medium_upper": 0.80,
    "firm_upper": 1.50,
}


@dataclass(frozen=True)
class TrialSummary:
    trajectory_id: str
    source_group: str
    source_trial_id: str
    source_dir: str
    frame_rows: int
    samples: int
    contact_frames: int
    force_valid_frames: int
    slip_valid_frames: int
    slip_positive_frames: int
    max_normal_force_n: float
    stages: dict[str, int]
    force_regimes: dict[str, int]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_frame_map(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_tactile_image(path: Path) -> np.ndarray:
    image = np.load(path)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).round().astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=2)
    if image.ndim == 3 and image.shape[-1] > 3:
        image = image[:, :, :3]
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 tactile RGB image, got shape={image.shape}")
    return image


def _encode_png_bytes(image: np.ndarray) -> bytes:
    from PIL import Image

    with io.BytesIO() as buf:
        Image.fromarray(image, mode="RGB").save(buf, format="PNG")
        return buf.getvalue()


def _default_source_roots() -> list[Path]:
    candidates = [
        Path("artifacts/phase3"),
        Path("sim_dataset/phase4_collected/bridge_robustness"),
        Path("sim_dataset/phase4_collected/force_oriented"),
        Path("sim_dataset/phase4_collected/slip_oriented"),
        Path("sim_dataset/phase5_cube_force_slip/raw/force_v2"),
        Path("sim_dataset/phase5_cube_force_slip/raw/slip_v2"),
        Path("sim_dataset/phase5_cube_force_slip/raw/over_v2"),
    ]
    return [root for root in candidates if root.exists() and any(root.glob("phase3_cube_*"))]


def _source_group(source_root: Path) -> str:
    parts = source_root.parts
    if "phase5_cube_force_slip" in parts:
        return f"phase5_{source_root.name}"
    if source_root.as_posix().endswith("artifacts/phase3"):
        return "phase3_existing"
    return source_root.name


def _trajectory_id(source_root: Path, trial_dir: Path) -> str:
    return f"{_source_group(source_root)}__{trial_dir.name}"


def _state_samples(robot_state: dict[str, Any]) -> list[dict[str, Any]]:
    samples = robot_state.get("samples", [])
    return samples if isinstance(samples, list) else []


def _sample_for_row(robot_state: dict[str, Any], row: dict[str, str]) -> tuple[int, dict[str, Any]]:
    raw = row.get("robot_state_index", "")
    index = int(raw) if str(raw).strip() else -1
    samples = _state_samples(robot_state)
    sample = samples[index] if 0 <= index < len(samples) else {}
    return index, sample


def _contact_state(sample: dict[str, Any]) -> dict[str, Any]:
    state = sample.get("contact_state", {}) if isinstance(sample, dict) else {}
    return state if isinstance(state, dict) else {}


def _object_position(sample: dict[str, Any]) -> np.ndarray | None:
    obj = sample.get("object_state", {}) if isinstance(sample, dict) else {}
    if not isinstance(obj, dict):
        return None
    for key in ("grasp_center_world_m", "root_position_m"):
        value = obj.get(key)
        if isinstance(value, list) and len(value) >= 3:
            return np.asarray(value[:3], dtype=np.float64)
    return None


def _side_center(contact_state: dict[str, Any], side: str) -> np.ndarray | None:
    try:
        center = contact_state["geometry"]["sides"][side]["center_world_m"]
    except Exception:
        return None
    if isinstance(center, list) and len(center) >= 3:
        return np.asarray(center[:3], dtype=np.float64)
    return None


def _normalize(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return None
    return vector / norm


def _contact_frame(contact_state: dict[str, Any]) -> dict[str, Any]:
    left = _side_center(contact_state, "left")
    right = _side_center(contact_state, "right")
    if left is None or right is None:
        return {"valid": False, "reason": "missing_left_or_right_gelpad_center"}
    normal = _normalize(right - left)
    if normal is None:
        return {"valid": False, "reason": "degenerate_left_right_gelpad_axis"}

    # Tangent-b tracks world-up projected into the contact tangent plane, so
    # micro-lift motion is evaluated as slip-like tangential motion rather than
    # being mistaken for normal compression.
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    tangent_b = world_up - float(np.dot(world_up, normal)) * normal
    tangent_b_norm = _normalize(tangent_b)
    if tangent_b_norm is None:
        fallback = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        tangent_b_norm = _normalize(fallback - float(np.dot(fallback, normal)) * normal)
    if tangent_b_norm is None:
        return {"valid": False, "reason": "degenerate_tangent_axis"}
    tangent_a = _normalize(np.cross(tangent_b_norm, normal))
    if tangent_a is None:
        return {"valid": False, "reason": "degenerate_tangent_cross_axis"}
    origin = (left + right) * 0.5
    return {
        "valid": True,
        "origin_world_m": origin,
        "tangent_a_world": tangent_a,
        "tangent_b_world": tangent_b_norm,
        "normal_world": normal,
    }


def _relative_object_coords(sample: dict[str, Any]) -> tuple[np.ndarray | None, dict[str, Any]]:
    contact = _contact_state(sample)
    frame = _contact_frame(contact)
    pos = _object_position(sample)
    if not frame.get("valid") or pos is None:
        return None, frame
    rel = pos - frame["origin_world_m"]
    coords = np.asarray(
        [
            float(np.dot(rel, frame["tangent_a_world"])),
            float(np.dot(rel, frame["tangent_b_world"])),
            float(np.dot(rel, frame["normal_world"])),
        ],
        dtype=np.float64,
    )
    return coords, frame


def _normal_force_from_contact(contact_state: dict[str, Any]) -> tuple[float, str, bool]:
    force_by_side = contact_state.get("force_by_side_n", {}) or {}
    left = abs(float(force_by_side.get("left", 0.0) or 0.0))
    right = abs(float(force_by_side.get("right", 0.0) or 0.0))
    max_force = abs(float(contact_state.get("max_force_n", 0.0) or 0.0))
    normal_force_n = max(left, right, max_force)
    contact_detected = bool(contact_state.get("contact_detected", False))

    if normal_force_n > 1e-6:
        return normal_force_n, "sim_contact_sensor_scalar_normal", True
    if not contact_detected:
        return 0.0, "sim_no_contact_zero", True
    return 0.0, "invalid_geometry_contact_without_force_scalar", False


def _force_vector_label(contact_state: dict[str, Any]) -> tuple[np.ndarray, float, str, bool]:
    normal_force_n, source, valid = _normal_force_from_contact(contact_state)
    vector = np.asarray([0.0, 0.0, normal_force_n], dtype=np.float32)
    return vector, normal_force_n, source, valid


def _force_regime(contact_state: dict[str, Any], normal_force_n: float, valid: bool, stage: str) -> str:
    if stage in {"release", "end_trial"}:
        return "contact_release" if bool(contact_state.get("contact_detected", False)) else "release_no_contact"
    if stage in {"hold", "micro_lift"} and valid and normal_force_n > 0.0:
        prefix = "lift_and_hold_" if stage == "micro_lift" else "hold_"
    else:
        prefix = ""
    if not bool(contact_state.get("contact_detected", False)):
        return "no_contact"
    if not valid:
        return "contact_geometry_no_force_invalid"
    if normal_force_n < FORCE_LEVEL_THRESHOLDS_N["light_upper"]:
        return prefix + "light"
    if normal_force_n < FORCE_LEVEL_THRESHOLDS_N["medium_upper"]:
        return prefix + "medium"
    if normal_force_n < FORCE_LEVEL_THRESHOLDS_N["firm_upper"]:
        return prefix + "firm"
    return prefix + "over"


def _slip_state_labels(
    robot_state: dict[str, Any],
    *,
    threshold_m: float,
    horizon: int,
) -> dict[int, dict[str, Any]]:
    samples = _state_samples(robot_state)
    coords: list[np.ndarray | None] = []
    frame_valid: list[bool] = []
    frame_reason: list[str] = []
    for sample in samples:
        rel, frame = _relative_object_coords(sample)
        coords.append(rel)
        frame_valid.append(bool(frame.get("valid")))
        frame_reason.append(str(frame.get("reason", "ok")))

    labels: dict[int, dict[str, Any]] = {}
    for index, sample in enumerate(samples):
        stage = str(sample.get("action_stage", ""))
        contact = _contact_state(sample)
        contact_detected = bool(contact.get("contact_detected", False))
        release_masked = stage in SLIP_MASKED_STAGES
        valid = False
        label = 0
        source = "invalid"
        max_tangent_disp = 0.0

        if release_masked:
            source = "release_masked_not_metric_slip"
        elif not contact_detected:
            source = "no_contact_masked"
        elif stage not in SLIP_VALID_STAGES:
            source = f"stage_{stage or 'unknown'}_masked"
        elif not frame_valid[index] or coords[index] is None:
            source = f"missing_contact_frame:{frame_reason[index]}"
        else:
            valid = True
            current = coords[index]
            assert current is not None
            future_seen = False
            for future_index in range(index + 1, min(len(samples), index + horizon + 1)):
                future_sample = samples[future_index]
                future_stage = str(future_sample.get("action_stage", ""))
                if future_stage in SLIP_MASKED_STAGES:
                    continue
                if not bool(_contact_state(future_sample).get("contact_detected", False)):
                    continue
                future = coords[future_index]
                if future is None:
                    continue
                future_seen = True
                tangent_disp = float(np.linalg.norm(future[:2] - current[:2]))
                max_tangent_disp = max(max_tangent_disp, tangent_disp)
            label = 1 if max_tangent_disp >= threshold_m else 0
            source = (
                "sim_contact_frame_tangent_motion"
                if label
                else ("sim_contact_frame_stable" if future_seen else "sim_contact_frame_terminal_stable")
            )

        labels[index] = {
            "slip_label": int(label),
            "valid_slip": bool(valid),
            "slip_label_source": source,
            "slip_tangent_displacement_m": max_tangent_disp,
            "release_masked": bool(release_masked),
            "relative_object_coords_m": None if coords[index] is None else coords[index].astype(float).tolist(),
        }
    return labels


def _resolve_tactile_path(trial_dir: Path, row: dict[str, str]) -> Path:
    raw = row.get("tactile_rgb_path", "")
    if not raw:
        raise FileNotFoundError(f"Missing tactile_rgb_path in {trial_dir}")
    path = Path(raw)
    if path.exists():
        return path
    fallback = trial_dir / "tactile" / path.name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Missing tactile RGB array for {trial_dir}: {raw}")


def _iter_trials(source_roots: Iterable[Path]) -> Iterable[tuple[Path, Path]]:
    for source_root in source_roots:
        for trial_dir in sorted(source_root.glob("phase3_cube_*")):
            if (trial_dir / "meta.json").exists() and (trial_dir / "frame_map.csv").exists() and (trial_dir / "robot_state.json").exists():
                yield source_root, trial_dir


def _phase4_bridge_report() -> dict[str, Any] | None:
    path = Path("sim_dataset/phase4_sparsh_cube/bridge_report.json")
    if not path.exists():
        return None
    try:
        return _load_json(path)
    except Exception:
        return None


def build_phase5_export(
    source_roots: list[Path],
    *,
    export_root: Path,
    report_root: Path,
    slip_threshold_m: float,
    slip_horizon: int,
) -> dict[str, Any]:
    dataset_root = export_root / DATASET_NAME
    dataset_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    images: list[bytes] = []
    in_contact: list[int] = []
    trajectories: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    trial_summaries: list[TrialSummary] = []

    global_force_valid = 0
    global_force_invalid = 0
    global_slip_valid = 0
    global_slip_invalid = 0
    global_slip_positive_valid = 0
    global_slip_negative_valid = 0
    global_release_masked = 0
    force_sources = Counter()
    slip_sources = Counter()
    force_regimes = Counter()
    stage_counts = Counter()

    for source_root, trial_dir in _iter_trials(source_roots):
        source_group = _source_group(source_root)
        trajectory_id = _trajectory_id(source_root, trial_dir)
        frame_rows = _load_frame_map(trial_dir / "frame_map.csv")
        robot_state = _load_json(trial_dir / "robot_state.json")
        meta = _load_json(trial_dir / "meta.json")
        slip_by_state = _slip_state_labels(robot_state, threshold_m=slip_threshold_m, horizon=slip_horizon)

        indexes: list[int] = []
        forces: list[np.ndarray] = []
        slip_labels: list[int] = []
        valid_force_mask: list[bool] = []
        valid_slip_mask: list[bool] = []
        normal_forces: list[float] = []
        trajectory_force_sources: list[str] = []
        trajectory_slip_sources: list[str] = []
        trajectory_regimes = Counter()
        trajectory_stages = Counter()

        for row in frame_rows:
            tactile_path = _resolve_tactile_path(trial_dir, row)
            image = _load_tactile_image(tactile_path)
            global_index = len(images)
            images.append(_encode_png_bytes(image))

            robot_index, sample = _sample_for_row(robot_state, row)
            contact = _contact_state(sample)
            contact_detected = bool(contact.get("contact_detected", False))
            in_contact.append(1 if contact_detected else 0)
            stage = str(row.get("action_stage") or sample.get("action_stage") or "")
            stage_counts[stage] += 1
            trajectory_stages[stage] += 1

            force_vec, normal_force_n, force_source, force_valid = _force_vector_label(contact)
            slip_info = slip_by_state.get(robot_index, {})
            slip_label = int(slip_info.get("slip_label", 0))
            slip_valid = bool(slip_info.get("valid_slip", False))
            slip_source = str(slip_info.get("slip_label_source", "invalid"))
            force_regime = _force_regime(contact, normal_force_n, force_valid, stage)

            indexes.append(global_index)
            forces.append(force_vec)
            slip_labels.append(slip_label)
            valid_force_mask.append(force_valid)
            valid_slip_mask.append(slip_valid)
            normal_forces.append(normal_force_n)
            trajectory_force_sources.append(force_source)
            trajectory_slip_sources.append(slip_source)
            trajectory_regimes[force_regime] += 1

            force_sources[force_source] += 1
            slip_sources[slip_source] += 1
            force_regimes[force_regime] += 1
            if force_valid:
                global_force_valid += 1
            else:
                global_force_invalid += 1
            if slip_valid:
                global_slip_valid += 1
                if slip_label:
                    global_slip_positive_valid += 1
                else:
                    global_slip_negative_valid += 1
            else:
                global_slip_invalid += 1
            if bool(slip_info.get("release_masked", False)):
                global_release_masked += 1

            rel_coords = slip_info.get("relative_object_coords_m")
            rows.append(
                {
                    "global_frame_index": global_index,
                    "trajectory_id": trajectory_id,
                    "sample_in_trajectory": len(indexes) - 1,
                    "source_group": source_group,
                    "source_trial_id": trial_dir.name,
                    "source_dir": trial_dir.as_posix(),
                    "side": row.get("side", ""),
                    "frame_id": row.get("frame_id", ""),
                    "source_tactile_rgb_path": tactile_path.as_posix(),
                    "object_id": meta.get("object_id", ""),
                    "protocol_variant": meta.get("protocol_variant", ""),
                    "action_stage": stage,
                    "contact_detected": contact_detected,
                    "robot_state_index": robot_index,
                    "force_frame": FORCE_FRAME_NAME,
                    "force_order": "[tangent_a_n,tangent_b_n,normal_n]",
                    "force_units": "N",
                    "force_x_n": float(force_vec[0]),
                    "force_y_n": float(force_vec[1]),
                    "force_z_n": float(force_vec[2]),
                    "normal_force_n": normal_force_n,
                    "force_label_source": force_source,
                    "force_valid_for_metrics": force_valid,
                    "force_regime": force_regime,
                    "slip_frame": SLIP_FRAME_NAME,
                    "slip_label": slip_label,
                    "slip_valid_for_metrics": slip_valid,
                    "slip_label_source": slip_source,
                    "slip_tangent_displacement_m": float(slip_info.get("slip_tangent_displacement_m", 0.0) or 0.0),
                    "slip_threshold_m": slip_threshold_m,
                    "release_masked_for_slip": bool(slip_info.get("release_masked", False)),
                    "object_rel_tangent_a_m": "" if rel_coords is None else rel_coords[0],
                    "object_rel_tangent_b_m": "" if rel_coords is None else rel_coords[1],
                    "object_rel_normal_m": "" if rel_coords is None else rel_coords[2],
                    "label_source": f"force:{force_source};slip:{slip_source}",
                    "label_valid_for_metrics": bool(force_valid and slip_valid),
                }
            )

        trajectories[trajectory_id] = {
            "indexes": np.asarray(indexes, dtype=np.int64),
            "forces": np.asarray(forces, dtype=np.float32),
            "slip_label": np.asarray(slip_labels, dtype=np.int64),
            "valid_force": np.asarray(valid_force_mask, dtype=bool),
            "valid_slip": np.asarray(valid_slip_mask, dtype=bool),
            "normal_force_n": np.asarray(normal_forces, dtype=np.float32),
            "force_label_source": np.asarray(trajectory_force_sources, dtype=object),
            "slip_label_source": np.asarray(trajectory_slip_sources, dtype=object),
            "metadata": {
                "trial_id": trajectory_id,
                "source_trial_id": trial_dir.name,
                "object_id": meta.get("object_id", "cube"),
                "protocol_variant": meta.get("protocol_variant", ""),
                "force_frame": FORCE_FRAME_NAME,
                "force_axis_order": ["tangent_a_n", "tangent_b_n", "normal_n"],
                "force_sign_convention": "positive normal_n is compressive scalar-normal contact force",
                "slip_frame": SLIP_FRAME_NAME,
                "slip_threshold_m": slip_threshold_m,
                "slip_horizon_samples": slip_horizon,
                "source": "phase5_cube_force_slip",
            },
            "source_dir": trial_dir.as_posix(),
            "source_group": source_group,
        }
        trial_summaries.append(
            TrialSummary(
                trajectory_id=trajectory_id,
                source_group=source_group,
                source_trial_id=trial_dir.name,
                source_dir=trial_dir.as_posix(),
                frame_rows=len(frame_rows),
                samples=len(_state_samples(robot_state)),
                contact_frames=sum(1 for row in rows if row["trajectory_id"] == trajectory_id and row["contact_detected"]),
                force_valid_frames=int(sum(valid_force_mask)),
                slip_valid_frames=int(sum(valid_slip_mask)),
                slip_positive_frames=int(sum(1 for lab, valid in zip(slip_labels, valid_slip_mask) if valid and lab == 1)),
                max_normal_force_n=max(normal_forces) if normal_forces else 0.0,
                stages=dict(trajectory_stages),
                force_regimes=dict(trajectory_regimes),
            )
        )

    if not images:
        raise FileNotFoundError("No Phase3-style cube trials found for Phase5 export.")

    force_label_validity = "sim-valid" if global_force_valid > 0 else "invalid"
    slip_label_validity = (
        "sim-valid"
        if global_slip_positive_valid > 0 and global_slip_negative_valid > 0 and global_slip_valid > 0
        else "invalid"
    )
    format_gate_status = "PASS_STRUCTURAL" if images and trajectories and rows else "FAIL"
    eval_gate_status = (
        "BLOCKED_NO_CHECKPOINT_NO_TRUSTED_METRICS"
        if force_label_validity == "sim-valid" and slip_label_validity == "sim-valid"
        else "BLOCKED_LABEL_VALIDITY"
    )

    image_path = dataset_root / "dataset_gelsight_cube_v2.pkl"
    with image_path.open("wb") as f:
        pickle.dump(images, f, protocol=pickle.HIGHEST_PROTOCOL)

    force_slip = {
        "in_contact": np.asarray(in_contact, dtype=np.int64),
        "trajectories": trajectories,
        "valid_force_global": np.asarray([row["force_valid_for_metrics"] for row in rows], dtype=bool),
        "valid_slip_global": np.asarray([row["slip_valid_for_metrics"] for row in rows], dtype=bool),
        "source": {
            "dataset_name": DATASET_NAME,
            "phase": "phase5_cube_force_slip_v2",
            "source_roots": [root.as_posix() for root in source_roots],
            "force_frame": FORCE_FRAME_NAME,
            "force_vector_policy": "[0,0,normal_force_n] in gripper scalar-normal frame; never [left,right,max_force]",
            "slip_frame": SLIP_FRAME_NAME,
            "slip_policy": "contact-frame tangential relative object motion; release/end masked",
            "slip_threshold_m": slip_threshold_m,
            "slip_horizon_samples": slip_horizon,
            "force_label_validity": force_label_validity,
            "slip_label_validity": slip_label_validity,
            "metric_validity": "limited-sim-only-with-valid-masks-no-realworld-claim",
        },
    }
    force_slip_path = dataset_root / "dataset_slip_forces.pkl"
    with force_slip_path.open("wb") as f:
        pickle.dump(force_slip, f, protocol=pickle.HIGHEST_PROTOCOL)

    sample_index_path = dataset_root / "sample_index.csv"
    fieldnames = list(rows[0].keys())
    with sample_index_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    manifest = {
        "phase": "phase5_cube_force_slip_v2",
        "dataset_name": DATASET_NAME,
        "dataset_root": dataset_root.as_posix(),
        "export_root": export_root.as_posix(),
        "report_root": report_root.as_posix(),
        "files": {
            "dataset_gelsight_cube_v2.pkl": image_path.as_posix(),
            "dataset_slip_forces.pkl": force_slip_path.as_posix(),
            "sample_index.csv": sample_index_path.as_posix(),
            "export_manifest.json": (dataset_root / "export_manifest.json").as_posix(),
        },
        "source_roots": [root.as_posix() for root in source_roots],
        "counts": {
            "frames": len(images),
            "trajectories": len(trajectories),
            "in_contact_frames": int(sum(in_contact)),
            "force_valid_frames": global_force_valid,
            "force_invalid_frames": global_force_invalid,
            "slip_valid_frames": global_slip_valid,
            "slip_invalid_frames": global_slip_invalid,
            "slip_positive_valid_frames": global_slip_positive_valid,
            "slip_negative_valid_frames": global_slip_negative_valid,
            "release_masked_slip_frames": global_release_masked,
        },
        "force": {
            "frame": FORCE_FRAME_NAME,
            "order": ["tangent_a_n", "tangent_b_n", "normal_n"],
            "unit": "N",
            "normal_force_scalar_source": "max(abs(left_force_n), abs(right_force_n), abs(max_force_n)) from sim contact sensors when nonzero",
            "invalid_vector": "[left_force_n, right_force_n, max_force_n] is explicitly invalid for Phase5 metrics",
            "thresholds_n": FORCE_LEVEL_THRESHOLDS_N,
            "sources": dict(force_sources),
            "regimes": dict(force_regimes),
            "label_validity": force_label_validity,
        },
        "slip": {
            "frame": SLIP_FRAME_NAME,
            "valid_stages": sorted(SLIP_VALID_STAGES),
            "masked_stages": sorted(SLIP_MASKED_STAGES),
            "threshold_m": slip_threshold_m,
            "horizon_samples": slip_horizon,
            "sources": dict(slip_sources),
            "label_validity": slip_label_validity,
        },
        "gates": {
            "FORMAT_GATE": format_gate_status,
            "FORCE_LABEL_VALIDITY": force_label_validity,
            "SLIP_LABEL_VALIDITY": slip_label_validity,
            "EVAL_GATE": eval_gate_status,
        },
        "trial_summaries": [asdict(summary) for summary in trial_summaries],
    }
    manifest_path = dataset_root / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    _write_reports(report_root, manifest, rows)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def _md_table(counter: dict[str, int] | Counter) -> str:
    if not counter:
        return "| item | count |\n| --- | ---: |\n"
    def _sort_key(kv: tuple[str, Any]) -> tuple[int, str]:
        try:
            return (-int(kv[1]), str(kv[0]))
        except Exception:
            return (0, str(kv[0]))

    lines = ["| item | count |", "| --- | ---: |"]
    for key, value in sorted(counter.items(), key=_sort_key):
        lines.append(f"| `{key}` | {value} |")
    return "\n".join(lines) + "\n"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_reports(report_root: Path, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    counts = manifest["counts"]
    gates = manifest["gates"]
    phase4 = _phase4_bridge_report()
    phase4_gates = (phase4 or {}).get("gates", {}) if isinstance(phase4, dict) else {}

    _write(
        report_root / "phase4_audit.md",
        f"""# Phase 4 audit / proxy-boundary freeze

Phase 5 does **not** silently upgrade Phase 4 proxy labels.  Existing Phase 4
bridge outputs remain usable for format/dataloader smoke only unless a Phase 5
row-level valid mask marks a newly derived label as simulation-valid.

## Observed Phase 4 bridge gates

{_md_table(phase4_gates)}

## Phase 5 boundary rule

- Every exported row has `force_label_source`, `slip_label_source`, and
  `label_valid_for_metrics` in `sample_index.csv`.
- Phase 4's invalid/proxy force vector `[left_force_n, right_force_n, max_force_n]`
  is not reused as a metric vector.
- Geometry-only contact rows with zero force are kept for provenance and format
  continuity but have `force_valid_for_metrics=False`.
- Release/end rows are masked for slip metrics unless a future explicit release-slip
  protocol is introduced.
""",
    )

    _write(
        report_root / "force_vector_semantics.md",
        f"""# Force vector semantics

- Frame: `{FORCE_FRAME_NAME}`.
- Order: `[tangent_a_n, tangent_b_n, normal_n]`.
- Unit: Newtons.
- Sign: positive `normal_n` means compressive contact along the gripper scalar
  normal; tangential force components are intentionally zero because Phase 3/4
  recordings expose scalar contact magnitudes, not signed shear vectors.
- Valid metric vector: `[0, 0, normal_force_n]`, where `normal_force_n` is
  `max(abs(left_force_n), abs(right_force_n), abs(max_force_n))` from simulation
  contact sensors when nonzero.
- Valid no-contact vector: `[0, 0, 0]` with `force_label_source=sim_no_contact_zero`.
- Explicitly invalid for Phase 5 metrics: `[left_force_n, right_force_n, max_force_n]`.
- Invalid mask: geometry-contact frames with zero sensor force are exported but
  marked `force_valid_for_metrics=False`.

## Label-source counts

{_md_table(manifest['force']['sources'])}

## Force-regime counts

{_md_table(manifest['force']['regimes'])}
""",
    )

    _write(
        report_root / "force_trial_summary.md",
        f"""# Force trial summary

Exported frames: {counts['frames']}  
Trajectories: {counts['trajectories']}  
Force-valid frames: {counts['force_valid_frames']}  
Force-invalid/masked frames: {counts['force_invalid_frames']}  
Max normal force (N): {max(float(row['normal_force_n']) for row in rows):.6f}

Phase 5 force regimes are represented by `force_regime` in `sample_index.csv`.
The categories include `no_contact`, light/medium/firm/over contact bands, and
stage roles such as `lift_and_hold_*` and `contact_release` when present.

## Regime counts

{_md_table(manifest['force']['regimes'])}

## Source roots

"""
        + "\n".join(f"- `{root}`" for root in manifest["source_roots"]),
    )

    _write(
        report_root / "slip_contact_frame_rule.md",
        f"""# Slip contact-frame rule

- Frame: `{SLIP_FRAME_NAME}`.
- Origin: midpoint between left and right GelSight soft-mesh AABB centers from
  each simulation sample.
- Normal axis: left-to-right gelpad center axis (closing axis).
- Tangent axes: world-up projected into the plane orthogonal to the normal plus
  its orthogonal cross-axis.
- Label: `slip_label=1` when object relative motion in the two tangent axes
  reaches at least `{manifest['slip']['threshold_m']}` m within
  `{manifest['slip']['horizon_samples']}` future sample(s).
- Valid stages: {', '.join(f'`{s}`' for s in manifest['slip']['valid_stages'])}.
- Masked stages: {', '.join(f'`{s}`' for s in manifest['slip']['masked_stages'])}.
- Release/end rows are not counted as metric-valid slip labels.

This is simulation-valid for cube-only contact-frame consistency checks; it is
not a real-world tactile slip benchmark.
""",
    )

    _write(
        report_root / "slip_trial_summary.md",
        f"""# Slip trial summary

Slip-valid frames: {counts['slip_valid_frames']}  
Slip-invalid/masked frames: {counts['slip_invalid_frames']}  
Valid positive slip frames: {counts['slip_positive_valid_frames']}  
Valid no-slip frames: {counts['slip_negative_valid_frames']}  
Release-masked frames: {counts['release_masked_slip_frames']}

## Slip-source counts

{_md_table(manifest['slip']['sources'])}
""",
    )

    _write(
        report_root / "label_validity_gate.md",
        f"""# Label validity gate

- FORCE_LABEL_VALIDITY: **{gates['FORCE_LABEL_VALIDITY']}**
- SLIP_LABEL_VALIDITY: **{gates['SLIP_LABEL_VALIDITY']}**

The gate is mask-aware: metric consumers must filter with `valid_force` and
`valid_slip` from `dataset_slip_forces.pkl` or the corresponding CSV columns.
A row that exists for Sparsh format continuity is not automatically metric-valid.
""",
    )

    _write(
        report_root / "format_gate.md",
        f"""# Format gate

Initial structural status: **{gates['FORMAT_GATE']}**

Expected Phase 5 v2 export files:

- `{manifest['files']['dataset_gelsight_cube_v2.pkl']}`
- `{manifest['files']['dataset_slip_forces.pkl']}`
- `{manifest['files']['sample_index.csv']}`
- `{manifest['files']['export_manifest.json']}`

Run `phase5_sparsh_smoke.py` in the `tacex` environment to replace this
structural status with Sparsh dataloader evidence.
""",
    )

    _write(
        report_root / "eval_gate.md",
        f"""# Eval gate

EVAL_GATE: **{gates['EVAL_GATE']}**

No trusted force/slip metrics are emitted by the exporter.  Limited Sparsh
metrics require all of the following:

1. FORMAT_GATE passes through the Sparsh dataloader.
2. FORCE_LABEL_VALIDITY and SLIP_LABEL_VALIDITY are `sim-valid`.
3. Consumers apply valid-force and valid-slip masks.
4. A concrete Sparsh encoder/task checkpoint path is supplied.

Without a checkpoint, Phase 5 stops at label/format evidence.
""",
    )

    _write(
        report_root / "phase5_review.md",
        f"""# Phase 5 review

## Scope check

- Cube-only simulation: satisfied.
- Forcefield/chips_can/cracker_box/real robot/real data/RL/GraspNet/closed-loop:
  not introduced.
- Heavy data location: `{manifest['dataset_root']}` under repo-local `sim_dataset/`.
- Reviewable reports: `{manifest['report_root']}`.

## Gate summary

{_md_table(gates)}

## Counts

{_md_table(counts)}

## Remaining limitation

Labels are valid only for limited cube simulation checks with masks.  No real
Sparsh checkpoint was bundled, and no benchmark metrics should be claimed unless
`eval_gate.md` is updated by a checkpoint-backed evaluation run.
""",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Phase 5 cube force/slip v2 Sparsh export.")
    parser.add_argument(
        "--source-root",
        type=Path,
        action="append",
        default=None,
        help="Phase3-style source root containing phase3_cube_* trials. May be repeated.",
    )
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--slip-threshold-m", type=float, default=7.5e-4)
    parser.add_argument("--slip-horizon", type=int, default=3)
    args = parser.parse_args()

    source_roots = args.source_root or _default_source_roots()
    if not source_roots:
        raise FileNotFoundError("No Phase3/Phase4/Phase5 cube source roots found.")
    build_phase5_export(
        source_roots,
        export_root=args.export_root,
        report_root=args.report_root,
        slip_threshold_m=args.slip_threshold_m,
        slip_horizon=args.slip_horizon,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
