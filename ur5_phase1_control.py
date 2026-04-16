from __future__ import annotations

import math
from typing import Iterable

import torch
from isaaclab.assets import Articulation

TABLE_TOP_HEIGHT = 0.72
TABLE_TRANSLATION = (0.6, 0.0, -0.0234)
UR5_BASE_OFFSET = (-0.053, -0.08, 0.01)
BANANA_REST_HEIGHT = TABLE_TOP_HEIGHT + 0.02

ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
GRIPPER_PRIMARY_JOINT_NAME = "robotiq_85_left_knuckle_joint"
GRIPPER_MIRROR_JOINT_NAME = "robotiq_85_right_knuckle_joint"
GRIPPER_LOG_JOINT_NAMES = [
    GRIPPER_PRIMARY_JOINT_NAME,
    GRIPPER_MIRROR_JOINT_NAME,
    "left_knuckle_finger_joint",
    "right_knuckle_finger_joint",
    "left_inner_finger_joint",
    "right_inner_finger_joint",
    "robotiq_85_left_finger_tip_joint",
    "robotiq_85_right_finger_tip_joint",
]
CONTROLLED_JOINT_NAMES = ARM_JOINT_NAMES + [GRIPPER_PRIMARY_JOINT_NAME, GRIPPER_MIRROR_JOINT_NAME]

RESET_ARM_JOINT_POS_DEG = {
    "shoulder_pan_joint": 1.8,
    "shoulder_lift_joint": -87.3,
    "elbow_joint": 50.9,
    "wrist_1_joint": -53.6,
    "wrist_2_joint": -90.1,
    "wrist_3_joint": -2.6,
}
PREGRASP_ARM_JOINT_POS_DEG = {
    "shoulder_pan_joint": -2.57,
    "shoulder_lift_joint": -63.89,
    "elbow_joint": 84.52,
    "wrist_1_joint": -110.63,
    "wrist_2_joint": -90.0,
    "wrist_3_joint": -155.28,
}

GRIPPER_OPEN_ANGLE_RAD = 0.0
GRIPPER_CLOSE_ANGLE_RAD = 0.6
ARM_SINGLE_JOINT_DELTA_RAD = 0.25
ARM_JOINT_TOLERANCE_RAD = 0.05
GRIPPER_JOINT_TOLERANCE_RAD = 0.03
RESET_JOINT_TOLERANCE_RAD = 0.02
RESET_GRIPPER_JOINT_TOLERANCE_RAD = 0.03
RESET_POSITION_TOLERANCE_M = 0.005
PREGRASP_JOINT_TOLERANCE_RAD = 0.05
TARGET_HOLD_STEPS = 30
ARM_MAX_STEPS = 180
GRIPPER_MAX_STEPS = 180
PREGRASP_MAX_STEPS = 240
DEFAULT_GRIPPER_CYCLES = 20
DEFAULT_RESET_TRIALS = 20
DEFAULT_PREGRASP_TRIALS = 20


def joint_targets_deg_to_rad(joint_targets_deg: dict[str, float]) -> dict[str, float]:
    return {joint_name: math.radians(angle_deg) for joint_name, angle_deg in joint_targets_deg.items()}


def named_joint_ids(robot: Articulation, joint_names: Iterable[str]) -> dict[str, int]:
    joint_names = list(joint_names)
    joint_ids, ordered_names = robot.find_joints(joint_names, preserve_order=True)
    return {name: joint_id for name, joint_id in zip(ordered_names, joint_ids, strict=True)}


def build_joint_target(
    robot: Articulation,
    joint_overrides_rad: dict[str, float],
    base_target: torch.Tensor | None = None,
) -> torch.Tensor:
    joint_target = robot.data.default_joint_pos.clone() if base_target is None else base_target.clone()
    joint_id_map = named_joint_ids(robot, joint_overrides_rad.keys())
    for joint_name, joint_value in joint_overrides_rad.items():
        joint_target[:, joint_id_map[joint_name]] = joint_value
    return joint_target


def reset_joint_overrides_rad() -> dict[str, float]:
    joint_overrides = joint_targets_deg_to_rad(RESET_ARM_JOINT_POS_DEG)
    joint_overrides[GRIPPER_PRIMARY_JOINT_NAME] = GRIPPER_OPEN_ANGLE_RAD
    joint_overrides[GRIPPER_MIRROR_JOINT_NAME] = GRIPPER_OPEN_ANGLE_RAD
    return joint_overrides


def pregrasp_joint_overrides_rad() -> dict[str, float]:
    joint_overrides = joint_targets_deg_to_rad(PREGRASP_ARM_JOINT_POS_DEG)
    joint_overrides[GRIPPER_PRIMARY_JOINT_NAME] = GRIPPER_OPEN_ANGLE_RAD
    joint_overrides[GRIPPER_MIRROR_JOINT_NAME] = GRIPPER_OPEN_ANGLE_RAD
    return joint_overrides


def build_reset_joint_target(robot: Articulation) -> torch.Tensor:
    return build_joint_target(robot, reset_joint_overrides_rad())


def build_pregrasp_joint_target(robot: Articulation) -> torch.Tensor:
    return build_joint_target(robot, pregrasp_joint_overrides_rad())


def build_gripper_joint_target(
    robot: Articulation,
    closed: bool,
    base_target: torch.Tensor | None = None,
) -> torch.Tensor:
    gripper_angle = GRIPPER_CLOSE_ANGLE_RAD if closed else GRIPPER_OPEN_ANGLE_RAD
    return build_joint_target(
        robot,
        {
            GRIPPER_PRIMARY_JOINT_NAME: gripper_angle,
            GRIPPER_MIRROR_JOINT_NAME: gripper_angle,
        },
        base_target=base_target,
    )


def current_joint_positions(robot: Articulation, joint_names: Iterable[str]) -> dict[str, float]:
    joint_id_map = named_joint_ids(robot, joint_names)
    return {joint_name: float(robot.data.joint_pos[0, joint_id].item()) for joint_name, joint_id in joint_id_map.items()}


def joint_errors(robot: Articulation, joint_target: torch.Tensor, joint_names: Iterable[str]) -> dict[str, float]:
    joint_id_map = named_joint_ids(robot, joint_names)
    return {
        joint_name: float(abs(robot.data.joint_pos[0, joint_id].item() - joint_target[0, joint_id].item()))
        for joint_name, joint_id in joint_id_map.items()
    }


def joint_target_values(joint_target: torch.Tensor, robot: Articulation, joint_names: Iterable[str]) -> dict[str, float]:
    joint_id_map = named_joint_ids(robot, joint_names)
    return {joint_name: float(joint_target[0, joint_id].item()) for joint_name, joint_id in joint_id_map.items()}


def pick_arm_probe_target(
    robot: Articulation,
    joint_name: str,
    delta_rad: float = ARM_SINGLE_JOINT_DELTA_RAD,
    safety_margin: float = 0.05,
) -> torch.Tensor:
    joint_target = build_reset_joint_target(robot)
    joint_id = named_joint_ids(robot, [joint_name])[joint_name]
    current_value = float(joint_target[0, joint_id].item())
    lower_limit = float(robot.data.soft_joint_pos_limits[0, joint_id, 0].item())
    upper_limit = float(robot.data.soft_joint_pos_limits[0, joint_id, 1].item())

    plus_target = current_value + delta_rad
    minus_target = current_value - delta_rad
    if plus_target <= upper_limit - safety_margin:
        selected_value = plus_target
    elif minus_target >= lower_limit + safety_margin:
        selected_value = minus_target
    else:
        selected_value = min(max(current_value, lower_limit), upper_limit)

    joint_target[:, joint_id] = selected_value
    return joint_target
