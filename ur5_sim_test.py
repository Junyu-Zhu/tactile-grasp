"""Visible Phase 1 demo for the integrated UR5e embodiment.

This demo is for visual confirmation of the key observable Phase 1 gates:
- Step 4: clean spawn of robot + table + banana
- Step 5: arm joint actuation
- Step 6: gripper open/close
- Step 7: deterministic reset
- Step 8: fixed pre-grasp reach

Unlike `ur5_sim.py`, which defaults to a stable spawn and offers validation via
`--phase1-checks`, this script explicitly plays the motion sequence on screen.

Usage:
    ./isaaclab.sh -p tactile_grasp/ur5_sim_test.py
    ./isaaclab.sh -p tactile_grasp/ur5_sim_test.py --headless --gripper-cycles 2
    ./isaaclab.sh -p tactile_grasp/ur5_sim_test.py --headless --regenerate-robot-usd
"""

from __future__ import annotations

import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Demonstrate Phase 1 integrated UR5 embodiment milestones.")
parser.add_argument("--gripper-cycles", type=int, default=2, help="How many visible gripper open/close cycles to play.")
parser.add_argument("--pregrasp-holds", type=int, default=1, help="How many visible pre-grasp repetitions to play.")
parser.add_argument("--spawn-hold-steps", type=int, default=120, help="How long to hold the clean spawn for visual inspection.")
parser.add_argument("--reset-demo-trials", type=int, default=3, help="How many visible deterministic reset trials to play.")
parser.add_argument("--reset-disturb-steps", type=int, default=90, help="How many steps to spend moving away from reset before each visible reset.")
parser.add_argument("--reset-hold-steps", type=int, default=90, help="How long to hold after each visible reset.")
parser.add_argument(
    "--regenerate-robot-usd",
    action="store_true",
    help="Force regeneration of the USD derived from the canonical robot URDF before the demo scene loads.",
)
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
from ur5_phase1_control import (
    ARM_JOINT_NAMES,
    CONTROLLED_JOINT_NAMES,
    RESET_GRIPPER_JOINT_TOLERANCE_RAD,
    RESET_JOINT_TOLERANCE_RAD,
    RESET_POSITION_TOLERANCE_M,
    build_gripper_joint_target,
    build_pregrasp_joint_target,
    joint_errors,
)
from ur5_phase1_reset import (
    expected_banana_position,
    expected_robot_base_position,
    reset_scene,
    step_scene,
)
from ur5_phase1_scene import (
    BANANA_URDF_PATH,
    CAMERA_EYE,
    CAMERA_TARGET,
    TABLE_USD_PATH,
    design_scene,
    ensure_robot_usd_path,
    report_scene_state,
    resolve_robot_urdf_path,
)


def _position_error(actual, expected) -> float:
    return float((actual - expected).abs().max().item())


def run_visible_reset_demo(
    sim: sim_utils.SimulationContext,
    robot,
    banana,
    origin,
    *,
    trials: int,
    disturb_steps: int,
    hold_steps: int,
) -> bool:
    print(f"[DEMO] Step 7: 开始演示 deterministic reset，共 {trials} 次。")
    expected_robot_position = expected_robot_base_position(origin, robot)
    expected_banana_position_tensor = expected_banana_position(origin, banana)
    pregrasp_target = build_pregrasp_joint_target(robot)
    closed_pregrasp_target = build_gripper_joint_target(robot, closed=True, base_target=pregrasp_target)

    for trial_index in range(1, trials + 1):
        step_scene(sim, robot, banana, joint_target=pregrasp_target, steps=disturb_steps)
        step_scene(sim, robot, banana, joint_target=closed_pregrasp_target, steps=disturb_steps)
        reset_target = reset_scene(sim, robot, banana, origin)
        robot_position_error = _position_error(robot.data.root_pose_w[0, :3], expected_robot_position)
        banana_position_error = _position_error(banana.data.root_pose_w[0, :3], expected_banana_position_tensor)
        joint_error_map = joint_errors(robot, reset_target, CONTROLLED_JOINT_NAMES)
        arm_max_error = max(joint_error_map[joint_name] for joint_name in ARM_JOINT_NAMES)
        gripper_joint_names = [joint_name for joint_name in CONTROLLED_JOINT_NAMES if joint_name not in ARM_JOINT_NAMES]
        gripper_max_error = max(joint_error_map[joint_name] for joint_name in gripper_joint_names)
        trial_passed = (
            arm_max_error <= RESET_JOINT_TOLERANCE_RAD
            and gripper_max_error <= RESET_GRIPPER_JOINT_TOLERANCE_RAD
            and robot_position_error <= RESET_POSITION_TOLERANCE_M
            and banana_position_error <= RESET_POSITION_TOLERANCE_M
        )
        print(
            "[DEMO] Reset trial {trial}: robot_err={robot_err:.6f} m, banana_err={banana_err:.6f} m, "
            "arm_err={arm_err:.6f} rad, gripper_err={gripper_err:.6f} rad -> {status}".format(
                trial=trial_index,
                robot_err=robot_position_error,
                banana_err=banana_position_error,
                arm_err=arm_max_error,
                gripper_err=gripper_max_error,
                status="PASS" if trial_passed else "FAIL",
            )
        )
        step_scene(sim, robot, banana, joint_target=reset_target, steps=hold_steps)
        if not trial_passed:
            return False
    return True


def main() -> int:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
    sim.set_camera_view(CAMERA_EYE, CAMERA_TARGET)
    robot_urdf_path = resolve_robot_urdf_path()
    robot_usd_path = ensure_robot_usd_path(force_conversion=args_cli.regenerate_robot_usd)
    robots, bananas, origins = design_scene(num_envs=1, robot_usd_path=robot_usd_path)
    origins = origins.to(sim.device)
    sim.reset()

    robot = next(iter(robots.values()))
    banana = next(iter(bananas.values()))
    origin = origins[0]

    print(f"[INFO] Loaded canonical robot URDF from: {robot_urdf_path}")
    print(f"[INFO] Loaded generated robot USD from: {robot_usd_path}")
    print(f"[INFO] Loaded table USD from: {TABLE_USD_PATH}")
    print(f"[INFO] Loaded banana URDF from: {BANANA_URDF_PATH}")
    report_scene_state(origins, robot_urdf_path, robot_usd_path)

    reset_scene(sim, robot, banana, origin)
    print("[DEMO] Step 4: 展示 clean spawn（robot + table + banana）。")
    step_scene(sim, robot, banana, steps=args_cli.spawn_hold_steps)

    print("[DEMO] Step 5: 开始演示 6 个 arm joints 的小范围运动。")
    arm_result = run_arm_joint_actuation_check(sim, robot, banana, origin)
    if not arm_result["passed"]:
        print("[DEMO] Arm motion demo failed.")
        return 1
    step_scene(sim, robot, banana, steps=90)

    print(f"[DEMO] Step 6: 开始演示 gripper 开合，共 {args_cli.gripper_cycles} 次。")
    gripper_result = run_gripper_cycle_check(sim, robot, banana, origin, cycles=args_cli.gripper_cycles)
    if not gripper_result["passed"]:
        print("[DEMO] Gripper demo failed.")
        print(json.dumps(gripper_result, ensure_ascii=False, indent=2))
        return 1
    step_scene(sim, robot, banana, steps=90)

    reset_demo_passed = run_visible_reset_demo(
        sim,
        robot,
        banana,
        origin,
        trials=args_cli.reset_demo_trials,
        disturb_steps=args_cli.reset_disturb_steps,
        hold_steps=args_cli.reset_hold_steps,
    )
    if not reset_demo_passed:
        print("[DEMO] Deterministic reset demo failed.")
        return 1

    print(f"[DEMO] Step 8: 开始演示 fixed pre-grasp reach，共 {args_cli.pregrasp_holds} 次。")
    pregrasp_result = run_pregrasp_check(sim, robot, banana, origin, trials=args_cli.pregrasp_holds)
    if not pregrasp_result["passed"]:
        print("[DEMO] Pre-grasp demo failed.")
        return 1
    step_scene(sim, robot, banana, steps=180)

    print("[DEMO] 演示完成：Step 4 / 5 / 6 / 7 / 8 都已显示。")
    return 0


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
