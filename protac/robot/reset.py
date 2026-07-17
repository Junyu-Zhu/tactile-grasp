"""Deterministic UR5/Robotiq reset used by the cube baseline."""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation

from protac.robot.control import (
    TABLE_TOP_HEIGHT,
    UR5_BASE_OFFSET,
    build_reset_joint_target,
)


def reset_robot(
    robot: Articulation,
    origin: torch.Tensor,
    joint_target: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reset the robot root, joints and command target without moving the object."""

    joint_target = build_reset_joint_target(robot) if joint_target is None else joint_target
    root_pose = robot.data.default_root_state.clone()
    root_pose[:, 0] = float(origin[0].item()) + UR5_BASE_OFFSET[0]
    root_pose[:, 1] = float(origin[1].item()) + UR5_BASE_OFFSET[1]
    root_pose[:, 2] = float(origin[2].item()) + TABLE_TOP_HEIGHT + UR5_BASE_OFFSET[2]
    root_velocity = torch.zeros_like(root_pose[:, 7:])
    joint_velocity = torch.zeros_like(robot.data.default_joint_vel)

    robot.write_root_pose_to_sim(root_pose[:, :7])
    robot.write_root_velocity_to_sim(root_velocity)
    robot.write_joint_state_to_sim(joint_target, joint_velocity)
    robot.set_joint_position_target(joint_target)
    robot.write_data_to_sim()
    robot.reset()
    return joint_target
