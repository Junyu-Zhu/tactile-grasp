"""Phase 3 aligned trial logging helpers."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from protac.compat.phase3_v1_schema import (
    FRAME_MAP_COLUMNS,
    PHASE_NAME,
    TRIAL_SCHEMA_VERSION,
    utc_now_iso,
    validate_metadata_fields,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "detach"):
        return _json_safe(value.detach().cpu().tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _tensor_to_numpy(value: Any):
    import numpy as np

    if hasattr(value, "detach"):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.shape and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in {1, 3} and array.shape[-1] not in {1, 3, 4}:
        array = array.transpose(1, 2, 0)
    return array


def _array_stats(array: Any) -> dict[str, Any]:
    import numpy as np

    arr = np.asarray(array, dtype=np.float32)
    return {
        "shape": list(arr.shape),
        "min": float(arr.min()) if arr.size else 0.0,
        "max": float(arr.max()) if arr.size else 0.0,
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size else 0.0,
    }


def _image_uint8(array: Any):
    import numpy as np

    arr = np.asarray(array)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.size and float(arr.max()) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        return arr[..., :3]
    return arr.reshape(arr.shape[0], -1)


def save_array(path: Path, array: Any) -> str:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, _tensor_to_numpy(array))
    return path.as_posix()


def save_preview_image(path: Path, array: Any) -> str:
    image = _image_uint8(array)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2  # type: ignore

        if image.ndim == 3 and image.shape[-1] == 3:
            cv2.imwrite(path.as_posix(), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(path.as_posix(), image)
        return path.as_posix()
    except Exception:
        fallback = path.with_suffix(".ppm" if image.ndim == 3 else ".pgm")
        if image.ndim == 3:
            fallback.write_bytes(b"P6\n%d %d\n255\n" % (image.shape[1], image.shape[0]) + image.tobytes())
        else:
            fallback.write_bytes(b"P5\n%d %d\n255\n" % (image.shape[1], image.shape[0]) + image.tobytes())
        return fallback.as_posix()


@dataclass(frozen=True)
class Phase3TrialPaths:
    trial_dir: Path
    tactile_dir: Path
    meta_json: Path
    robot_state_json: Path
    frame_map_csv: Path

    @classmethod
    def from_root(cls, output_root: Path, trial_id: str) -> "Phase3TrialPaths":
        trial_dir = output_root / trial_id
        return cls(
            trial_dir=trial_dir,
            tactile_dir=trial_dir / "tactile",
            meta_json=trial_dir / "meta.json",
            robot_state_json=trial_dir / "robot_state.json",
            frame_map_csv=trial_dir / "frame_map.csv",
        )


class Phase3TrialLogger:
    """Write aligned tactile frames, robot state samples, and frame-map rows."""

    def __init__(
        self,
        output_root: Path,
        *,
        trial_id: str,
        object_id: str,
        sensor_ids: tuple[str, ...],
        seed: int,
        protocol_variant: str,
        object_profile: dict[str, Any],
        command_profile: dict[str, Any],
        save_tactile_arrays: bool = True,
        save_preview_images: bool = True,
    ) -> None:
        self.paths = Phase3TrialPaths.from_root(output_root, trial_id)
        # A repeated deterministic trial reuses its id. Remove the previous
        # generated trial directory so stale tactile frames cannot leak into the
        # new frame map or confuse manual inspection.
        if self.paths.trial_dir.exists():
            shutil.rmtree(self.paths.trial_dir)
        self.paths.tactile_dir.mkdir(parents=True, exist_ok=True)
        self.trial_id = trial_id
        self.object_id = object_id
        self.sensor_ids = sensor_ids
        self.seed = seed
        self.protocol_variant = protocol_variant
        self.object_profile = object_profile
        self.command_profile = command_profile
        self.save_tactile_arrays = save_tactile_arrays
        self.save_preview_images = save_preview_images
        self.timestamp_start = utc_now_iso()
        self.timestamp_end: str | None = None
        self.robot_state_samples: list[dict[str, Any]] = []
        self.frame_rows: list[dict[str, Any]] = []
        self.frame_stats: dict[str, dict[str, Any]] = {}
        self._frame_counter = 0
        self._observed_stages: list[str] = []

    def _next_frame_id(self) -> str:
        self._frame_counter += 1
        return f"frame_{self._frame_counter:06d}"

    def record_sample(
        self,
        *,
        timestamp: str,
        action_stage: str,
        joint_state: dict[str, Any],
        gripper_state: dict[str, Any],
        object_state: dict[str, Any],
        contact_state: dict[str, Any],
        tactile_outputs: dict[str, dict[str, Any]],
        success_label: bool | None = None,
        failure_reason: str | None = None,
    ) -> dict[str, Any]:
        if action_stage not in self._observed_stages:
            self._observed_stages.append(action_stage)

        sample_index = len(self.robot_state_samples)
        frame_ids: list[str] = []
        for side, output in tactile_outputs.items():
            frame_id = self._next_frame_id()
            frame_ids.append(frame_id)
            files: dict[str, str] = {}
            for data_type in ("tactile_rgb", "camera_depth", "camera_rgb"):
                value = output.get(data_type)
                if value is None:
                    continue
                array = _tensor_to_numpy(value)
                stats_key = f"{side}:{frame_id}:{data_type}"
                self.frame_stats[stats_key] = _array_stats(array)
                if self.save_tactile_arrays:
                    files[data_type] = save_array(self.paths.tactile_dir / f"{side}_{frame_id}_{data_type}.npy", array)
                if self.save_preview_images and data_type in {"tactile_rgb", "camera_rgb"}:
                    files[f"{data_type}_preview"] = save_preview_image(
                        self.paths.tactile_dir / f"{side}_{frame_id}_{data_type}.png",
                        array,
                    )
            row = {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "action_stage": action_stage,
                "side": side,
                "sensor_id": side,
                "tactile_rgb_path": files.get("tactile_rgb", ""),
                "camera_depth_path": files.get("camera_depth", ""),
                "camera_rgb_path": files.get("camera_rgb", ""),
                "robot_state_index": sample_index,
                # A tactile frame is labelled contact only when this exact
                # gelpad reports force against the object.  Geometry proximity
                # or adaptor contact must not turn a Taxim frame into a positive
                # tactile-contact sample.
                "contact_detected": side in set(contact_state.get("force_contact_sides", [])),
                "success_label": "" if success_label is None else bool(success_label),
                "failure_reason": failure_reason or "",
            }
            self.frame_rows.append(row)

        state_sample = {
            "index": sample_index,
            "timestamp": timestamp,
            "action_stage": action_stage,
            "joint_state": _json_safe(joint_state),
            "gripper_state": _json_safe(gripper_state),
            "object_state": _json_safe(object_state),
            "contact_state": _json_safe(contact_state),
            "tactile_frame_ids": frame_ids,
        }
        self.robot_state_samples.append(state_sample)
        return state_sample

    def alignment_summary(self) -> dict[str, Any]:
        rows_with_state = [
            row for row in self.frame_rows if int(row["robot_state_index"]) < len(self.robot_state_samples)
        ]
        stages_from_frames = [row["action_stage"] for row in self.frame_rows]
        return {
            "robot_state_samples": len(self.robot_state_samples),
            "tactile_frame_rows": len(self.frame_rows),
            "all_frames_have_robot_state": len(rows_with_state) == len(self.frame_rows),
            "observed_stages": self._observed_stages,
            "frame_stages": sorted(set(stages_from_frames), key=stages_from_frames.index) if stages_from_frames else [],
            "frame_map_columns": list(FRAME_MAP_COLUMNS),
        }

    def finalize(
        self,
        *,
        success_label: bool,
        failure_reason: str | None,
        contact_onset: dict[str, Any],
    ) -> dict[str, Any]:
        self.timestamp_end = utc_now_iso()
        last_state = self.robot_state_samples[-1] if self.robot_state_samples else {}
        tactile_frame_id = {
            "count": self._frame_counter,
            "last": self.frame_rows[-1]["frame_id"] if self.frame_rows else None,
        }
        meta = {
            "schema_version": TRIAL_SCHEMA_VERSION,
            "trial_id": self.trial_id,
            "object_id": self.object_id,
            "sensor_id": list(self.sensor_ids),
            "seed": self.seed,
            "phase_name": PHASE_NAME,
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end,
            "action_stage": self._observed_stages,
            "joint_state": last_state.get("joint_state", {}),
            "gripper_state": last_state.get("gripper_state", {}),
            "tactile_frame_id": tactile_frame_id,
            "contact_onset": contact_onset,
            "success_label": bool(success_label),
            "failure_reason": failure_reason,
            "protocol_variant": self.protocol_variant,
            "object_profile": self.object_profile,
            "command_profile": self.command_profile,
            "output_files": {
                "meta": self.paths.meta_json.as_posix(),
                "robot_state": self.paths.robot_state_json.as_posix(),
                "frame_map": self.paths.frame_map_csv.as_posix(),
                "tactile_dir": self.paths.tactile_dir.as_posix(),
            },
            "alignment_summary": self.alignment_summary(),
        }
        meta = _json_safe(meta)
        missing = validate_metadata_fields(meta)
        if missing:
            raise RuntimeError(f"Phase3 metadata missing required fields: {missing}")
        self.paths.meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.paths.robot_state_json.write_text(
            json.dumps(
                {
                    "trial_id": self.trial_id,
                    "samples": self.robot_state_samples,
                    "frame_stats": self.frame_stats,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        with self.paths.frame_map_csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FRAME_MAP_COLUMNS, lineterminator="\n")
            writer.writeheader()
            for row in self.frame_rows:
                writer.writerow({column: row.get(column, "") for column in FRAME_MAP_COLUMNS})
        return meta
