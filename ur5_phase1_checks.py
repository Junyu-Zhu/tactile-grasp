from __future__ import annotations

import json
import math
from typing import Any

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject

from ur5_phase1_control import (
    ARM_JOINT_NAMES,
    ARM_JOINT_TOLERANCE_RAD,
    ARM_MAX_STEPS,
    CONTROLLED_JOINT_NAMES,
    DEFAULT_GRIPPER_CYCLES,
    DEFAULT_PREGRASP_TRIALS,
    DEFAULT_RESET_TRIALS,
    GRIPPER_CONTROL_JOINT_NAME,
    GRIPPER_CONTROL_JOINT_NAMES,
    GRIPPER_JOINT_TOLERANCE_RAD,
    GRIPPER_LOG_JOINT_NAMES,
    GRIPPER_MAX_STEPS,
    PREGRASP_JOINT_TOLERANCE_RAD,
    PREGRASP_MAX_STEPS,
    RESET_GRIPPER_JOINT_TOLERANCE_RAD,
    RESET_JOINT_TOLERANCE_RAD,
    RESET_POSITION_TOLERANCE_M,
    TARGET_HOLD_STEPS,
    build_gripper_joint_target,
    build_pregrasp_joint_target,
    build_reset_joint_target,
    current_joint_positions,
    joint_errors,
    joint_target_values,
    pick_arm_probe_target,
)
from ur5_phase1_reset import expected_banana_position, expected_robot_base_position, reset_scene, step_scene


def _max_abs(values: list[float]) -> float:
    return max((abs(value) for value in values), default=0.0)


def _is_finite_map(data: dict[str, float]) -> bool:
    return all(math.isfinite(value) for value in data.values())


def _wait_for_joint_target(
    sim: sim_utils.SimulationContext,
    robot: Articulation,
    banana: RigidObject,
    joint_target: torch.Tensor,
    joint_names: list[str],
    tolerance_rad: float,
    max_steps: int,
    hold_steps: int = TARGET_HOLD_STEPS,
) -> dict[str, Any]:
    stable_steps = 0
    peak_error = 0.0
    final_error_map: dict[str, float] = {}
    for step in range(1, max_steps + 1):
        step_scene(sim, robot, banana, joint_target=joint_target, steps=1)
        final_error_map = joint_errors(robot, joint_target, joint_names)
        max_error = _max_abs(list(final_error_map.values()))
        peak_error = max(peak_error, max_error)
        if _is_finite_map(final_error_map) and max_error <= tolerance_rad:
            stable_steps += 1
        else:
            stable_steps = 0
        if stable_steps >= hold_steps:
            return {
                "passed": True,
                "steps": step,
                "peak_error_rad": peak_error,
                "final_error_rad": max_error,
                "error_by_joint": final_error_map,
            }
    return {
        "passed": False,
        "steps": max_steps,
        "peak_error_rad": peak_error,
        "final_error_rad": _max_abs(list(final_error_map.values())),
        "error_by_joint": final_error_map,
    }


def _position_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(torch.max(torch.abs(actual - expected)).item())


def run_arm_joint_actuation_check(
    sim: sim_utils.SimulationContext,
    robot: Articulation,
    banana: RigidObject,
    origin: torch.Tensor,
) -> dict[str, Any]:
    print('[CHECK] Running arm joint actuation check...')
    summaries: list[dict[str, Any]] = []
    for joint_name in ARM_JOINT_NAMES:
        reset_scene(sim, robot, banana, origin)
        joint_target = pick_arm_probe_target(robot, joint_name)
        wait_result = _wait_for_joint_target(
            sim,
            robot,
            banana,
            joint_target,
            [joint_name],
            tolerance_rad=ARM_JOINT_TOLERANCE_RAD,
            max_steps=ARM_MAX_STEPS,
        )
        summaries.append(
            {
                "joint_name": joint_name,
                "target_rad": joint_target_values(joint_target, robot, [joint_name])[joint_name],
                **wait_result,
            }
        )
        if not wait_result["passed"]:
            return {"passed": False, "results": summaries}
    return {"passed": True, "results": summaries}


def run_gripper_cycle_check(
    sim: sim_utils.SimulationContext,
    robot: Articulation,
    banana: RigidObject,
    origin: torch.Tensor,
    cycles: int = DEFAULT_GRIPPER_CYCLES,
) -> dict[str, Any]:
    print(f'[CHECK] Running gripper open/close check for {cycles} cycles...')
    cycle_results: list[dict[str, Any]] = []
    base_target = build_reset_joint_target(robot)
    open_target = build_gripper_joint_target(robot, closed=False, base_target=base_target)
    close_target = build_gripper_joint_target(robot, closed=True, base_target=base_target)
    tracked_joints = GRIPPER_CONTROL_JOINT_NAMES

    reset_scene(sim, robot, banana, origin)
    for cycle_index in range(1, cycles + 1):
        close_result = _wait_for_joint_target(
            sim,
            robot,
            banana,
            close_target,
            tracked_joints,
            tolerance_rad=GRIPPER_JOINT_TOLERANCE_RAD,
            max_steps=GRIPPER_MAX_STEPS,
        )
        open_result = _wait_for_joint_target(
            sim,
            robot,
            banana,
            open_target,
            tracked_joints,
            tolerance_rad=GRIPPER_JOINT_TOLERANCE_RAD,
            max_steps=GRIPPER_MAX_STEPS,
        )
        cycle_summary = {
            "cycle": cycle_index,
            "close": close_result,
            "open": open_result,
            "gripper_joint_positions": current_joint_positions(robot, GRIPPER_LOG_JOINT_NAMES),
        }
        cycle_results.append(cycle_summary)
        if not (close_result["passed"] and open_result["passed"]):
            return {
                "passed": False,
                "primary_control_joint": GRIPPER_CONTROL_JOINT_NAME,
                "coordinated_gripper_joints": GRIPPER_LOG_JOINT_NAMES,
                "open_target_rad": joint_target_values(open_target, robot, tracked_joints),
                "close_target_rad": joint_target_values(close_target, robot, tracked_joints),
                "results": cycle_results,
            }
    return {
        "passed": True,
        "primary_control_joint": GRIPPER_CONTROL_JOINT_NAME,
        "coordinated_gripper_joints": GRIPPER_LOG_JOINT_NAMES,
        "open_target_rad": joint_target_values(open_target, robot, tracked_joints),
        "close_target_rad": joint_target_values(close_target, robot, tracked_joints),
        "results": cycle_results,
    }


def run_deterministic_reset_check(
    sim: sim_utils.SimulationContext,
    robot: Articulation,
    banana: RigidObject,
    origin: torch.Tensor,
    trials: int = DEFAULT_RESET_TRIALS,
) -> dict[str, Any]:
    print(f'[CHECK] Running deterministic reset check for {trials} trials...')
    trial_results: list[dict[str, Any]] = []
    expected_robot_position = expected_robot_base_position(origin, robot)
    expected_object_position = expected_banana_position(origin, banana)
    reset_target = build_reset_joint_target(robot)
    pregrasp_target = build_pregrasp_joint_target(robot)
    closed_pregrasp_target = build_gripper_joint_target(robot, closed=True, base_target=pregrasp_target)

    for trial_index in range(1, trials + 1):
        step_scene(sim, robot, banana, joint_target=pregrasp_target, steps=90)
        step_scene(sim, robot, banana, joint_target=closed_pregrasp_target, steps=90)
        reset_target = reset_scene(sim, robot, banana, origin)

        joint_error_map = joint_errors(robot, reset_target, CONTROLLED_JOINT_NAMES)
        robot_position_error = _position_error(robot.data.root_pose_w[0, :3], expected_robot_position)
        object_position_error = _position_error(banana.data.root_pose_w[0, :3], expected_object_position)
        arm_error_map = {joint_name: joint_error_map[joint_name] for joint_name in ARM_JOINT_NAMES}
        gripper_error_map = {
            joint_name: joint_error_map[joint_name]
            for joint_name in GRIPPER_CONTROL_JOINT_NAMES
            if joint_name in joint_error_map
        }
        trial_summary = {
            "trial": trial_index,
            "robot_position_error_m": robot_position_error,
            "banana_position_error_m": object_position_error,
            "joint_error_by_name": joint_error_map,
            "arm_max_joint_error_rad": _max_abs(list(arm_error_map.values())),
            "gripper_max_joint_error_rad": _max_abs(list(gripper_error_map.values())),
            "max_joint_error_rad": _max_abs(list(joint_error_map.values())),
        }
        trial_summary["passed"] = (
            _is_finite_map(joint_error_map)
            and len(gripper_error_map) == len(GRIPPER_CONTROL_JOINT_NAMES)
            and trial_summary["arm_max_joint_error_rad"] <= RESET_JOINT_TOLERANCE_RAD
            and trial_summary["gripper_max_joint_error_rad"] <= RESET_GRIPPER_JOINT_TOLERANCE_RAD
            and robot_position_error <= RESET_POSITION_TOLERANCE_M
            and object_position_error <= RESET_POSITION_TOLERANCE_M
        )
        trial_results.append(trial_summary)
        if not trial_summary["passed"]:
            return {"passed": False, "results": trial_results}
    return {"passed": True, "results": trial_results}


def run_pregrasp_check(
    sim: sim_utils.SimulationContext,
    robot: Articulation,
    banana: RigidObject,
    origin: torch.Tensor,
    trials: int = DEFAULT_PREGRASP_TRIALS,
) -> dict[str, Any]:
    print(f'[CHECK] Running fixed pre-grasp reach check for {trials} trials...')
    trial_results: list[dict[str, Any]] = []
    pregrasp_target = build_pregrasp_joint_target(robot)
    for trial_index in range(1, trials + 1):
        reset_scene(sim, robot, banana, origin)
        reach_result = _wait_for_joint_target(
            sim,
            robot,
            banana,
            pregrasp_target,
            ARM_JOINT_NAMES,
            tolerance_rad=PREGRASP_JOINT_TOLERANCE_RAD,
            max_steps=PREGRASP_MAX_STEPS,
        )
        trial_summary = {
            "trial": trial_index,
            "target_joint_rad": joint_target_values(pregrasp_target, robot, ARM_JOINT_NAMES),
            **reach_result,
            "ee_link_world_pos": [
                float(value)
                for value in robot.data.body_pose_w[
                    0,
                    robot.find_bodies(['ee_link'], preserve_order=True)[0][0],
                    :3,
                ].tolist()
            ],
        }
        trial_results.append(trial_summary)
        if not reach_result["passed"]:
            return {"passed": False, "results": trial_results}
    return {"passed": True, "results": trial_results}


def run_phase1_validation_suite(
    sim: sim_utils.SimulationContext,
    robot: Articulation,
    banana: RigidObject,
    origin: torch.Tensor,
    gripper_cycles: int = DEFAULT_GRIPPER_CYCLES,
    reset_trials: int = DEFAULT_RESET_TRIALS,
    pregrasp_trials: int = DEFAULT_PREGRASP_TRIALS,
) -> dict[str, Any]:
    results = {
        "arm_joint_actuation": run_arm_joint_actuation_check(sim, robot, banana, origin),
        "gripper_open_close": run_gripper_cycle_check(sim, robot, banana, origin, cycles=gripper_cycles),
        "deterministic_reset": run_deterministic_reset_check(sim, robot, banana, origin, trials=reset_trials),
        "fixed_pregrasp_reach": run_pregrasp_check(sim, robot, banana, origin, trials=pregrasp_trials),
    }
    results["passed"] = all(item["passed"] for key, item in results.items() if key != "passed")
    print('[CHECK] Phase 1 validation summary:')
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return results
