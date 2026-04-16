"""Phase 1 UR5e embodiment bring-up scene for Isaac Sim.

This is the single source-of-truth scene entry point for Phase 1 in
`tactile_grasp/phase1.md`.

It intentionally stays focused on Phase 1 embodiment bring-up:
- one UR5e + Robotiq loaded from the local USD config in `assets/ur5_usd/ur5.py`
- one local table USD
- one YCB banana loaded from the local URDF
- deterministic reset helpers for the robot and banana
- optional Phase 1 validation checks for arm actuation, gripper cycles,
  deterministic reset, and fixed pre-grasp reach

Usage:
    ./isaaclab.sh -p tactile_grasp/ur5_sim.py --headless --max_steps 5
    ./isaaclab.sh -p tactile_grasp/ur5_sim.py --headless --phase1-checks
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Phase 1 UR5e + table + banana Isaac Sim scene.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--max_steps", type=int, default=0, help="Maximum simulation steps to run. 0 keeps running.")
parser.add_argument("--phase1-checks", action="store_true", help="Run the Phase 1 validation suite and exit.")
parser.add_argument("--gripper-cycles", type=int, default=20, help="Number of open/close cycles for the gripper validation.")
parser.add_argument("--reset-trials", type=int, default=20, help="Number of deterministic reset trials to run.")
parser.add_argument("--pregrasp-trials", type=int, default=20, help="Number of fixed pre-grasp reach trials to run.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject

from ur5_phase1_checks import run_phase1_validation_suite
from ur5_phase1_control import build_reset_joint_target
from ur5_phase1_reset import reset_scene
from ur5_phase1_scene import (
    CAMERA_EYE,
    CAMERA_TARGET,
    RESET_INTERVAL,
    UR5_USD_PATH,
    TABLE_USD_PATH,
    BANANA_URDF_PATH,
    design_scene,
    report_scene_state,
)


def run_simulator(
    sim: sim_utils.SimulationContext,
    robots: dict[str, Articulation],
    bananas: dict[str, RigidObject],
    origins: torch.Tensor,
) -> None:
    step_count = 0
    robot_list = list(robots.values())
    banana_list = list(bananas.values())
    sim_dt = sim.get_physics_dt()

    for index, robot in enumerate(robot_list):
        reset_scene(sim, robot, banana_list[index], origins[index])
    report_scene_state(origins)
    print("[INFO] Reset scene to Phase 1 UR5e + table + banana state.")

    while simulation_app.is_running():
        if step_count > 0 and step_count % RESET_INTERVAL == 0:
            for index, robot in enumerate(robot_list):
                reset_scene(sim, robot, banana_list[index], origins[index])
            print(f"[INFO] Periodic reset at step {step_count}.")

        for robot in robot_list:
            joint_target = build_reset_joint_target(robot)
            robot.set_joint_position_target(joint_target)
            robot.write_data_to_sim()

        sim.step()
        for robot in robot_list:
            robot.update(sim_dt)
        for banana in banana_list:
            banana.update(sim_dt)

        step_count += 1
        if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
            break


def main() -> int:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(CAMERA_EYE, CAMERA_TARGET)

    robots, bananas, origins = design_scene(args_cli.num_envs)
    origins = origins.to(sim.device)

    sim.reset()
    print(f"[INFO] Loaded UR5 USD from: {UR5_USD_PATH}")
    print(f"[INFO] Loaded table USD from: {TABLE_USD_PATH}")
    print(f"[INFO] Loaded banana URDF from: {BANANA_URDF_PATH}")

    if args_cli.phase1_checks:
        if args_cli.num_envs != 1:
            raise ValueError("Phase 1 checks currently require --num_envs 1.")
        robot = next(iter(robots.values()))
        banana = next(iter(bananas.values()))
        results = run_phase1_validation_suite(
            sim,
            robot,
            banana,
            origins[0],
            gripper_cycles=args_cli.gripper_cycles,
            reset_trials=args_cli.reset_trials,
            pregrasp_trials=args_cli.pregrasp_trials,
        )
        if not results["passed"]:
            print("[RESULT] Phase 1 validation FAILED.")
            return 1
        print("[RESULT] Phase 1 validation PASSED.")
        return 0

    run_simulator(sim, robots, bananas, origins)
    return 0


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
