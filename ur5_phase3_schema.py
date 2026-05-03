"""Phase 3 trial schema and protocol constants.

Phase 3 is deliberately limited to deterministic contact/grasp protocols plus
aligned tactile/robot-state logging.  This module is dependency-light so the
schema can be inspected, tested, and written without launching Isaac Sim.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PHASE3_SCOPE_SENTENCE = (
    "Phase 3 builds a minimal, repeatable contact/grasp trial protocol and saves "
    "aligned tactile frames plus robot state for later Phase 4/Sparsh work."
)
PHASE3_FORBIDDEN_SCOPE = (
    "No Sparsh inference/adaptation, no closed-loop tactile policy, no GraspNet, "
    "no RL, and no large-scale object generalization."
)
TRIAL_SCHEMA_VERSION = "phase3_trial_v1"
PHASE_NAME = "phase3_contact_grasp_logging"

TRIAL_REQUIRED_FIELDS: tuple[str, ...] = (
    "trial_id",
    "object_id",
    "sensor_id",
    "seed",
    "phase_name",
    "timestamp_start",
    "timestamp_end",
    "action_stage",
    "joint_state",
    "gripper_state",
    "tactile_frame_id",
    "contact_onset",
    "success_label",
    "failure_reason",
)

TRIAL_METADATA_FIELDS: tuple[str, ...] = (
    "schema_version",
    *TRIAL_REQUIRED_FIELDS,
    "protocol_variant",
    "object_profile",
    "command_profile",
    "output_files",
    "alignment_summary",
)

FRAME_MAP_COLUMNS: tuple[str, ...] = (
    "frame_id",
    "timestamp",
    "action_stage",
    "side",
    "sensor_id",
    "tactile_rgb_path",
    "camera_depth_path",
    "camera_rgb_path",
    "robot_state_index",
    "contact_detected",
    "success_label",
    "failure_reason",
)

ROBOT_STATE_SAMPLE_FIELDS: tuple[str, ...] = (
    "index",
    "timestamp",
    "action_stage",
    "joint_state",
    "gripper_state",
    "object_state",
    "contact_state",
    "tactile_frame_ids",
)

PROTOCOL_STAGES: tuple[str, ...] = (
    "reset",
    "pre_grasp",
    "contact_close",
    "hold",
    "micro_lift",
    "release",
    "end_trial",
)

PROTOCOL_VARIANTS: tuple[str, ...] = (
    "contact_only",
    "contact_hold",
    "contact_hold_micro_lift",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def make_trial_id(object_id: str, index: int) -> str:
    safe_object = object_id.lower().replace(" ", "_")
    return f"phase3_{safe_object}_{index:04d}"


@dataclass(frozen=True)
class TrialSchema:
    """Serializable description of the fixed Phase 3 trial contract."""

    schema_version: str = TRIAL_SCHEMA_VERSION
    scope: str = PHASE3_SCOPE_SENTENCE
    forbidden_scope: str = PHASE3_FORBIDDEN_SCOPE
    required_fields: tuple[str, ...] = TRIAL_REQUIRED_FIELDS
    metadata_fields: tuple[str, ...] = TRIAL_METADATA_FIELDS
    frame_map_columns: tuple[str, ...] = FRAME_MAP_COLUMNS
    robot_state_sample_fields: tuple[str, ...] = ROBOT_STATE_SAMPLE_FIELDS
    protocol_stages: tuple[str, ...] = PROTOCOL_STAGES
    protocol_variants: tuple[str, ...] = PROTOCOL_VARIANTS
    directory_layout: dict[str, str] = field(
        default_factory=lambda: {
            "meta": "artifacts/phase3/<trial_id>/meta.json",
            "tactile": "artifacts/phase3/<trial_id>/tactile/<side>_<frame_id>_<data_type>.<ext>",
            "robot_state": "artifacts/phase3/<trial_id>/robot_state.json",
            "frame_map": "artifacts/phase3/<trial_id>/frame_map.csv",
            "review": "artifacts/phase3/phase3_review.json",
        }
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope,
            "forbidden_scope": self.forbidden_scope,
            "required_fields": list(self.required_fields),
            "metadata_fields": list(self.metadata_fields),
            "frame_map_columns": list(self.frame_map_columns),
            "robot_state_sample_fields": list(self.robot_state_sample_fields),
            "protocol_stages": list(self.protocol_stages),
            "protocol_variants": list(self.protocol_variants),
            "directory_layout": self.directory_layout,
        }


def phase3_schema_dict() -> dict[str, Any]:
    return TrialSchema().as_dict()


def write_phase3_schema(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(phase3_schema_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def validate_metadata_fields(record: dict[str, Any]) -> list[str]:
    """Return missing required trial fields.

    The validator intentionally checks the fixed Phase 3 required field list
    rather than accepting whatever a trial runner happened to emit.
    """

    return [field_name for field_name in TRIAL_REQUIRED_FIELDS if field_name not in record]


def stage_is_valid(stage: str) -> bool:
    return stage in PROTOCOL_STAGES


def protocol_variant_is_valid(variant: str) -> bool:
    return variant in PROTOCOL_VARIANTS
