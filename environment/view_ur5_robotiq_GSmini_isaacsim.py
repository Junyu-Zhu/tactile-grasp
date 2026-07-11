"""Keep the integrated UR5e + Robotiq + GSmini assembly still in Isaac Sim.

Run from the tactile_grasp repository root:

    python environment/view_ur5_robotiq_GSmini_isaacsim.py

The GUI remains open at the same Phase 1 reset pose until its window is
closed.  ``--max_steps`` exists only for automated/headless smoke tests; the
default value of zero never ends the viewer loop on its own.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


# Running a script inside ``environment/`` otherwise omits tactile_grasp from
# Python's import path, while the shared Isaac scene helpers live at the repo
# root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher

from ur5_isaac_app import exit_headless_without_kit_shutdown, patch_headless_viewport_wait


patch_headless_viewport_wait()

parser = argparse.ArgumentParser(
    description="Open the integrated UR5e + Robotiq + GSmini assembly in a stationary Isaac Sim viewer."
)
parser.add_argument(
    "--regenerate-robot-usd",
    action="store_true",
    help="Regenerate the USD derived from the canonical URDF before opening the viewer.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Optional test limit. 0 (default) keeps the GUI open until you close it.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils

from ur5_phase1_reset import reset_scene
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


def run_stationary_viewer(sim: sim_utils.SimulationContext, robots, bananas, origins) -> None:
    """Hold all joints at the Isaac Phase 1 reset pose until the app closes."""

    robot_list = list(robots.values())
    banana_list = list(bananas.values())
    joint_targets = [
        reset_scene(sim, robot, banana, origin)
        for robot, banana, origin in zip(robot_list, banana_list, origins, strict=True)
    ]
    sim_dt = sim.get_physics_dt()
    step_count = 0

    print("[INFO] Viewer is holding the Isaac Phase 1 reset pose. Close the Isaac Sim window to exit.")
    while simulation_app.is_running():
        for robot, joint_target in zip(robot_list, joint_targets, strict=True):
            # This is a servo hold, not an animation: the target never changes.
            robot.set_joint_position_target(joint_target)
            robot.write_data_to_sim()

        sim.step()
        for robot in robot_list:
            robot.update(sim_dt)
        for banana in banana_list:
            banana.update(sim_dt)

        step_count += 1
        if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
            print(f"[INFO] Reached --max_steps={args_cli.max_steps}; closing viewer.")
            return


def main() -> int:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
    sim.set_camera_view(CAMERA_EYE, CAMERA_TARGET)

    robot_urdf_path = resolve_robot_urdf_path()
    robot_usd_path = ensure_robot_usd_path(force_conversion=args_cli.regenerate_robot_usd)
    robots, bananas, origins = design_scene(num_envs=1, robot_usd_path=robot_usd_path)
    origins = origins.to(sim.device)
    sim.reset()

    print(f"[INFO] Loaded canonical robot URDF from: {robot_urdf_path}")
    print(f"[INFO] Loaded generated robot USD from: {robot_usd_path}")
    print(f"[INFO] Loaded table USD from: {TABLE_USD_PATH}")
    print(f"[INFO] Loaded banana URDF from: {BANANA_URDF_PATH}")
    report_scene_state(origins, robot_urdf_path, robot_usd_path)

    run_stationary_viewer(sim, robots, bananas, origins)
    return 0


if __name__ == "__main__":
    exit_code = main()
    if args_cli.headless:
        exit_headless_without_kit_shutdown(exit_code)
    simulation_app.close()
    raise SystemExit(exit_code)
