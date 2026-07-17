"""UR5/Robotiq joint conventions and reusable target builders."""

from __future__ import annotations

import math
from typing import Iterable

import torch
from isaaclab.assets import Articulation

TABLE_TOP_HEIGHT = 0.72
UR5_BASE_OFFSET = (-0.053, -0.08, 0.01)

ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
GRIPPER_CONTROL_JOINT_NAME = "finger_joint"
GRIPPER_MIMIC_MULTIPLIERS = {
    "finger_joint": 1.0,
    "left_inner_finger_joint": -1.0,
    "left_inner_knuckle_joint": 1.0,
    "right_outer_knuckle_joint": 1.0,
    "right_inner_finger_joint": -1.0,
    "right_inner_knuckle_joint": 1.0,
}
GRIPPER_CONTROL_JOINT_NAMES = list(GRIPPER_MIMIC_MULTIPLIERS.keys())
# Isaac's PhysX mimic constraint did not keep this open-chain Robotiq model
# symmetric once one GSmini pad touched the cube.  Match the known-good
# PyBullet controller instead: import the mimic children as ordinary joints and
# command every joint with the URDF multiplier.  These are position-drive
# targets, never contact-time joint-state teleports.
GRIPPER_COMMAND_JOINT_NAMES = GRIPPER_CONTROL_JOINT_NAMES.copy()
GRIPPER_LOG_JOINT_NAMES = GRIPPER_CONTROL_JOINT_NAMES.copy()
GRIPPER_OPEN_TARGET_RAD_BY_JOINT = {joint_name: 0.0 for joint_name in GRIPPER_CONTROL_JOINT_NAMES}
# Use a visibly closed Robotiq target for the integrated GSmini fingertips.
GRIPPER_CLOSE_TARGET_RAD = 0.45
# Robotiq 2F-85 keeps the fingertip pads parallel through a four-bar mimic
# relation. The inner-finger joints must mirror the outer knuckle target;
# otherwise any GSmini mounted on the pad will close at an angled gripper pose.
GRIPPER_CLOSE_INNER_FINGER_RAD = -GRIPPER_CLOSE_TARGET_RAD
GRIPPER_CLOSE_TARGET_RAD_BY_JOINT = {
    "finger_joint": GRIPPER_CLOSE_TARGET_RAD,
    "left_inner_finger_joint": GRIPPER_CLOSE_INNER_FINGER_RAD,
    "left_inner_knuckle_joint": GRIPPER_CLOSE_TARGET_RAD,
    "right_outer_knuckle_joint": GRIPPER_CLOSE_TARGET_RAD,
    "right_inner_finger_joint": GRIPPER_CLOSE_INNER_FINGER_RAD,
    "right_inner_knuckle_joint": GRIPPER_CLOSE_TARGET_RAD,
}
CONTROLLED_JOINT_NAMES = ARM_JOINT_NAMES + GRIPPER_CONTROL_JOINT_NAMES

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


def gripper_joint_overrides_rad(closed: bool) -> dict[str, float]:
    target_map = GRIPPER_CLOSE_TARGET_RAD_BY_JOINT if closed else GRIPPER_OPEN_TARGET_RAD_BY_JOINT
    return target_map.copy()


def reset_joint_overrides_rad() -> dict[str, float]:
    joint_overrides = joint_targets_deg_to_rad(RESET_ARM_JOINT_POS_DEG)
    joint_overrides.update(gripper_joint_overrides_rad(closed=False))
    return joint_overrides


def pregrasp_joint_overrides_rad() -> dict[str, float]:
    joint_overrides = joint_targets_deg_to_rad(PREGRASP_ARM_JOINT_POS_DEG)
    joint_overrides.update(gripper_joint_overrides_rad(closed=False))
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
    return build_joint_target(
        robot,
        gripper_joint_overrides_rad(closed=closed),
        base_target=base_target,
    )
