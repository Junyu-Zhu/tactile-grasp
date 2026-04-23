"""Phase 2 single-side GSmini mount prototype for UR5e + Robotiq.

This script focuses only on Phase 2 Step 1 / 2 / 3:
- lock Phase 2 scope
- lock a single mount source-of-truth
- build a single-side GSmini + adaptor prototype on the Robotiq

It intentionally does **not** start tactile output, Sparsh, control, or grasp logic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Phase 2 single-side GSmini mount prototype.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--max_steps", type=int, default=0, help="Maximum simulation steps to run. 0 keeps running.")
parser.add_argument("--phase2-step3-checks", action="store_true", help="Run Step 3 single-side mount validation and exit.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils

from ur5_phase1_control import build_reset_joint_target
from ur5_phase1_reset import reset_scene
from ur5_phase1_scene import CAMERA_EYE, CAMERA_TARGET, design_scene, report_scene_state
from ur5_phase2_mount import (
    PHASE2_SCOPE_SENTENCE,
    mount_single_side_connector_only,
    read_left_fingertip_world_pose,
    run_single_side_mount_validation,
    source_of_truth_summary,
    sync_left_mount_to_runtime_fingertip,
)

PHASE2_DEBUG_CAMERA_OFFSET = (0.18, -0.20, 0.10)


def _write_phase2_step13_artifact(summary: dict[str, object]) -> Path:
    artifact_dir = Path(__file__).resolve().parent / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    out = artifact_dir / "phase2_step1_3_report.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return out


def _set_phase2_debug_camera(sim, robot) -> dict[str, list[float]]:
    fingertip_pos, _ = read_left_fingertip_world_pose(robot)
    target = [float(v) for v in fingertip_pos.tolist()]
    eye = [
        target[0] + PHASE2_DEBUG_CAMERA_OFFSET[0],
        target[1] + PHASE2_DEBUG_CAMERA_OFFSET[1],
        target[2] + PHASE2_DEBUG_CAMERA_OFFSET[2],
    ]
    sim.set_camera_view(tuple(eye), tuple(target))
    return {"eye": eye, "target": target}


def run_preview(sim, robots, bananas, origins) -> None:
    robot_list = list(robots.values())
    banana_list = list(bananas.values())
    sim_dt = sim.get_physics_dt()
    step_count = 0

    for idx, robot in enumerate(robot_list):
        reset_scene(sim, robot, banana_list[idx], origins[idx])
    preview_paths = mount_single_side_connector_only()

    for _ in range(5):
        for robot in robot_list:
            joint_target = build_reset_joint_target(robot)
            robot.set_joint_position_target(joint_target)
            robot.write_data_to_sim()
        sim.step()
        for robot in robot_list:
            robot.update(sim_dt)
        for banana in banana_list:
            banana.update(sim_dt)
        sync_left_mount_to_runtime_fingertip(robot_list[0], include_gsmini=False)

    debug_positions = sync_left_mount_to_runtime_fingertip(robot_list[0], include_gsmini=False)
    camera_debug = _set_phase2_debug_camera(sim, robot_list[0])
    report_scene_state(origins)
    print(f"[INFO] {PHASE2_SCOPE_SENTENCE}")
    print("[INFO] Phase 2 preview ready: connector-only manual tuning mode on the left fingertip.")
    print(json.dumps(source_of_truth_summary(), ensure_ascii=False, indent=2))
    print("[INFO] Phase 2 preview marker/camera debug:")
    print(
        json.dumps(
            {
                "camera": camera_debug,
                "world_positions": debug_positions,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    while simulation_app.is_running():
        for robot in robot_list:
            joint_target = build_reset_joint_target(robot)
            robot.set_joint_position_target(joint_target)
            robot.write_data_to_sim()
        sim.step()
        for robot in robot_list:
            robot.update(sim_dt)
        for banana in banana_list:
            banana.update(sim_dt)
        sync_left_mount_to_runtime_fingertip(robot_list[0], include_gsmini=False)
        step_count += 1
        if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
            break


def main() -> int:
    if args_cli.num_envs != 1:
        raise ValueError("Phase 2 Step 1/2/3 currently supports only --num_envs 1.")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
    sim.set_camera_view(CAMERA_EYE, CAMERA_TARGET)
    robots, bananas, origins = design_scene(args_cli.num_envs)
    origins = origins.to(sim.device)
    sim.reset()

    if args_cli.phase2_step3_checks:
        robot = next(iter(robots.values()))
        banana = next(iter(bananas.values()))
        summary = run_single_side_mount_validation(sim, robot, banana, origins[0])
        artifact = _write_phase2_step13_artifact(summary)
        print(f"[INFO] Wrote Phase 2 Step 1/2/3 artifact to: {artifact}")
        if not summary["single_side_mount"]["passed"]:
            print("[RESULT] Phase 2 Step 1/2/3 validation FAILED.")
            return 1
        print("[RESULT] Phase 2 Step 1/2/3 validation PASSED.")
        return 0

    run_preview(sim, robots, bananas, origins)
    return 0


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
