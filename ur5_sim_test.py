"""Visible Phase 1 demo for UR5e arm motion, Robotiq open/close, and pre-grasp reach.

This demo is for visual confirmation of Phase 1 Step 5 / Step 6 / Step 8.
Unlike `ur5_sim.py`, which defaults to a stable spawn and offers validation via
`--phase1-checks`, this script explicitly plays the motion sequence on screen.

Usage:
    ./isaaclab.sh -p tactile_grasp/ur5_sim_test.py
    ./isaaclab.sh -p tactile_grasp/ur5_sim_test.py --headless --gripper-cycles 2
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Demonstrate Phase 1 UR5 arm / gripper / pre-grasp motions.")
parser.add_argument("--gripper-cycles", type=int, default=2, help="How many visible gripper open/close cycles to play.")
parser.add_argument("--pregrasp-holds", type=int, default=1, help="How many visible pre-grasp repetitions to play.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils

from ur5_phase1_checks import (
    run_arm_joint_actuation_check,
    run_gripper_cycle_check,
    run_pregrasp_check,
)
from ur5_phase1_reset import reset_scene, step_scene
from ur5_phase1_scene import CAMERA_EYE, CAMERA_TARGET, BANANA_URDF_PATH, TABLE_USD_PATH, UR5_USD_PATH, design_scene, report_scene_state


def main() -> int:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
    sim.set_camera_view(CAMERA_EYE, CAMERA_TARGET)
    robots, bananas, origins = design_scene(num_envs=1)
    origins = origins.to(sim.device)
    sim.reset()

    robot = next(iter(robots.values()))
    banana = next(iter(bananas.values()))
    origin = origins[0]

    print(f"[INFO] Loaded UR5 USD from: {UR5_USD_PATH}")
    print(f"[INFO] Loaded table USD from: {TABLE_USD_PATH}")
    print(f"[INFO] Loaded banana URDF from: {BANANA_URDF_PATH}")
    report_scene_state(origins)

    reset_scene(sim, robot, banana, origin)
    step_scene(sim, robot, banana, steps=60)

    print("[DEMO] Step 5: 开始演示 6 个 arm joints 的小范围运动。")
    arm_result = run_arm_joint_actuation_check(sim, robot, banana, origin)
    if not arm_result["passed"]:
        print("[DEMO] Arm motion demo failed.")
        return 1
    step_scene(sim, robot, banana, steps=90)

    print(f"[DEMO] Step 6: 开始演示 Robotiq 开合，共 {args_cli.gripper_cycles} 次。")
    gripper_result = run_gripper_cycle_check(sim, robot, banana, origin, cycles=args_cli.gripper_cycles)
    if not gripper_result["passed"]:
        print("[DEMO] Gripper demo failed.")
        return 1
    step_scene(sim, robot, banana, steps=90)

    print(f"[DEMO] Step 8: 开始演示 fixed pre-grasp reach，共 {args_cli.pregrasp_holds} 次。")
    pregrasp_result = run_pregrasp_check(sim, robot, banana, origin, trials=args_cli.pregrasp_holds)
    if not pregrasp_result["passed"]:
        print("[DEMO] Pre-grasp demo failed.")
        return 1
    step_scene(sim, robot, banana, steps=180)

    print("[DEMO] 演示完成：Step 5 / 6 / 8 都已显示。")
    return 0


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
