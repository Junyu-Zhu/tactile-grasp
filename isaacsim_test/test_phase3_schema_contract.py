from __future__ import annotations

import json

import protac.compat.phase3_v1_schema as schema


def test_phase3_v1_schema_contract_is_frozen() -> None:
    assert schema.TRIAL_SCHEMA_VERSION == "phase3_trial_v1"
    assert "No Sparsh inference/adaptation" in schema.PHASE3_FORBIDDEN_SCOPE
    assert "no closed-loop tactile policy" in schema.PHASE3_FORBIDDEN_SCOPE
    assert "no RL" in schema.PHASE3_FORBIDDEN_SCOPE
    assert schema.PROTOCOL_STAGES == (
        "reset", "pre_grasp", "contact_close", "hold", "micro_lift", "release", "end_trial"
    )
    assert schema.PROTOCOL_VARIANTS == ("contact_only", "contact_hold", "contact_hold_micro_lift")
    assert set(schema.TRIAL_REQUIRED_FIELDS).issubset(schema.TRIAL_METADATA_FIELDS)
    assert schema.FRAME_MAP_COLUMNS[0:4] == ("frame_id", "timestamp", "action_stage", "side")
    assert schema.ROBOT_STATE_SAMPLE_FIELDS[0:4] == ("index", "timestamp", "action_stage", "joint_state")


def test_phase3_schema_json_round_trip_is_stable(tmp_path) -> None:
    output = schema.write_phase3_schema(tmp_path / "trial_schema.json")
    assert json.loads(output.read_text()) == schema.phase3_schema_dict()
