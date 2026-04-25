from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject

from ur5_phase1_control import (
    BANANA_REST_HEIGHT,
    GRIPPER_CONTROL_JOINT_NAMES,
    GRIPPER_KINEMATIC_STEP_RAD,
    TABLE_TOP_HEIGHT,
    TABLE_TRANSLATION,
    UR5_BASE_OFFSET,
    build_reset_joint_target,
    named_joint_ids,
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


def _gripper_joint_ids(robot: Articulation) -> list[int]:
    return list(named_joint_ids(robot, GRIPPER_CONTROL_JOINT_NAMES).values())


def _set_gripper_kinematic_cache(robot: Articulation, joint_target: torch.Tensor) -> None:
    joint_ids = _gripper_joint_ids(robot)
    setattr(robot, "_phase1_gripper_kinematic_pos", joint_target[:, joint_ids].detach().clone())


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
    _set_gripper_kinematic_cache(robot, joint_target)
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


def _stabilize_gripper_mimic_state(robot: Articulation, joint_target: torch.Tensor) -> None:
    """Move converted Robotiq mimic joints directly toward their targets.

    Isaac imports the Robotiq four-bar mimic chain as an open articulation for
    Phase 1. Position drives alone can leave the lightweight inner fingers and
    pad assembly lagging behind the outer knuckles, which makes the GSmini tips
    close at an angle and lets the fixed pad assembly visibly oscillate during
    arm motion. This bounded kinematic write keeps the gripper-only mimic joints
    coherent while preserving gradual visible motion.
    """

    joint_ids = _gripper_joint_ids(robot)
    desired_gripper_pos = joint_target[:, joint_ids]
    cached_gripper_pos = getattr(robot, "_phase1_gripper_kinematic_pos", None)
    if cached_gripper_pos is None or cached_gripper_pos.shape != desired_gripper_pos.shape:
        cached_gripper_pos = robot.data.joint_pos[:, joint_ids].detach().clone()
    cached_gripper_pos = cached_gripper_pos.to(device=desired_gripper_pos.device, dtype=desired_gripper_pos.dtype)
    delta = torch.clamp(
        desired_gripper_pos - cached_gripper_pos,
        min=-GRIPPER_KINEMATIC_STEP_RAD,
        max=GRIPPER_KINEMATIC_STEP_RAD,
    )
    next_gripper_pos = cached_gripper_pos + delta
    setattr(robot, "_phase1_gripper_kinematic_pos", next_gripper_pos.detach().clone())
    zero_gripper_vel = torch.zeros_like(next_gripper_pos)
    robot.write_joint_state_to_sim(next_gripper_pos, zero_gripper_vel, joint_ids=joint_ids)


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
        if joint_target is not None:
            _stabilize_gripper_mimic_state(robot, joint_target)


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
