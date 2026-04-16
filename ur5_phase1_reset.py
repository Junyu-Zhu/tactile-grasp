from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject

from ur5_phase1_control import (
    BANANA_REST_HEIGHT,
    TABLE_TOP_HEIGHT,
    TABLE_TRANSLATION,
    UR5_BASE_OFFSET,
    build_reset_joint_target,
)


def expected_robot_base_position(origin: torch.Tensor, robot: Articulation) -> torch.Tensor:
    return torch.tensor(
        [
            float(origin[0].item()) + UR5_BASE_OFFSET[0],
            float(origin[1].item()) + UR5_BASE_OFFSET[1],
            float(origin[2].item()) + TABLE_TOP_HEIGHT + UR5_BASE_OFFSET[2],
        ],
        device=robot.device,
        dtype=robot.data.root_pose_w.dtype,
    )


def expected_banana_position(origin: torch.Tensor, banana: RigidObject) -> torch.Tensor:
    return torch.tensor(
        [
            float(origin[0].item()) + TABLE_TRANSLATION[0],
            float(origin[1].item()) + TABLE_TRANSLATION[1],
            float(origin[2].item()) + BANANA_REST_HEIGHT,
        ],
        device=banana.device,
        dtype=banana.data.root_pose_w.dtype,
    )


def reset_robot(
    robot: Articulation,
    origin: torch.Tensor,
    joint_target: torch.Tensor | None = None,
) -> torch.Tensor:
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


def reset_banana(banana: RigidObject, origin: torch.Tensor) -> None:
    object_pose = banana.data.default_root_state[:, :7].clone()
    object_pose[:, 0] = float(origin[0].item()) + TABLE_TRANSLATION[0]
    object_pose[:, 1] = float(origin[1].item()) + TABLE_TRANSLATION[1]
    object_pose[:, 2] = float(origin[2].item()) + BANANA_REST_HEIGHT
    object_pose[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=banana.device, dtype=object_pose.dtype)
    object_velocity = torch.zeros((object_pose.shape[0], 6), device=banana.device, dtype=object_pose.dtype)

    banana.write_root_pose_to_sim(object_pose)
    banana.write_root_velocity_to_sim(object_velocity)
    banana.reset()


def step_scene(
    sim: sim_utils.SimulationContext,
    robot: Articulation,
    banana: RigidObject,
    joint_target: torch.Tensor | None = None,
    steps: int = 1,
) -> None:
    sim_dt = sim.get_physics_dt()
    for _ in range(steps):
        if joint_target is not None:
            robot.set_joint_position_target(joint_target)
            robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
        banana.update(sim_dt)


def reset_scene(
    sim: sim_utils.SimulationContext,
    robot: Articulation,
    banana: RigidObject,
    origin: torch.Tensor,
    settle_steps: int = 150,
) -> torch.Tensor:
    joint_target = build_reset_joint_target(robot)
    reset_robot(robot, origin, joint_target)
    reset_banana(banana, origin)
    step_scene(sim, robot, banana, joint_target=joint_target, steps=settle_steps)
    return joint_target
