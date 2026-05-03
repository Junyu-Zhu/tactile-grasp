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
parser.add_argument("--phase2-step4-checks", action="store_true", help="Run Step 4 gripper compatibility validation and exit.")
parser.add_argument("--phase2-tactile-checks", action="store_true", help="Run Step 5/6 tactile no-contact/contact validation and exit.")
parser.add_argument("--phase2-full-review", action="store_true", help="Run Phase 2 Step 1-9 review path and exit.")
parser.add_argument(
    "--phase2-tactile-live",
    action="store_true",
    help="Open a GUI live preview of Phase 2 TacEx tactile_rgb/camera_depth streams and keep simulation running.",
)
parser.add_argument(
    "--phase2-tactile-debug-vis",
    action="store_true",
    help="Enable TacEx debug visualization windows during tactile validation modes.",
)
parser.add_argument("--gripper-cycles", type=int, default=20, help="Open/close cycles for Phase 2 Step 4.")
parser.add_argument("--phase2-contact-repeats", type=int, default=3, help="Contact repetitions for Phase 2 Step 6.")
parser.add_argument("--phase2-log-frames", action="store_true", help="Save tactile/camera frame images to artifacts.")
parser.add_argument("--phase2-include-camera-rgb", action="store_true", help="Request camera_rgb in addition to tactile_rgb/camera_depth.")
parser.add_argument("--phase2-tactile-width", type=int, default=320, help="GelSight tactile/camera width.")
parser.add_argument("--phase2-tactile-height", type=int, default=240, help="GelSight tactile/camera height.")
parser.add_argument(
    "--regenerate-robot-usd",
    action="store_true",
    help="Force regeneration of the USD derived from the canonical robot URDF before the Phase 2 scene loads.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils

from ur5_phase1_control import build_reset_joint_target
from ur5_phase1_reset import reset_scene
from ur5_phase1_scene import (
    CAMERA_EYE,
    CAMERA_TARGET,
    design_scene,
    ensure_robot_usd_path,
    report_scene_state,
    resolve_robot_urdf_path,
)
from ur5_phase2_mount import (
    PHASE2_SCOPE_SENTENCE,
    clear_phase2_visual_prims,
    mount_phase2_sensor_shells,
    mount_single_side_connector_only,
    phase2_sensor_prim_paths,
    read_left_fingertip_world_pose,
    run_gripper_mount_compatibility_validation,
    run_single_side_mount_validation,
    source_of_truth_summary,
    sync_left_mount_to_runtime_fingertip,
    validate_phase2_sensor_camera_prims,
    validate_phase2_sensor_mounts,
)
from ur5_phase2_tactile import (
    Phase2TactileOptions,
    append_phase2_issue,
    build_phase2_gsmini_cfg,
    initialize_phase2_sensor,
    run_phase2_tactile_validation,
    summarize_exception_for_log,
    update_phase2_sensor,
    write_phase2_review_artifact,
)

PHASE2_DEBUG_CAMERA_OFFSET = (0.18, -0.20, 0.10)


def _write_phase2_step13_artifact(summary: dict[str, object]) -> Path:
    artifact_dir = Path(__file__).resolve().parent / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    out = artifact_dir / "phase2_step1_3_report.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    md = artifact_dir / "phase2_step1_3_report.md"
    md.write_text(
        "# Phase 2 Step 1/2/3 Report\n\n"
        f"- Scope: `{PHASE2_SCOPE_SENTENCE}`\n"
        f"- Canonical source-of-truth: `{summary['source_of_truth']['primary_robot_reference']}`\n"
        f"- Generated USD policy: {summary['source_of_truth']['generated_robot_asset_policy']}\n"
        f"- Single-side result: {'PASS' if summary['single_side_mount']['passed'] else 'FAIL'}\n\n"
        "## Source-of-truth summary\n"
        "```json\n"
        f"{json.dumps(summary['source_of_truth'], ensure_ascii=False, indent=2)}\n"
        "```\n",
        encoding="utf-8",
    )
    return out


def _write_phase2_step4_artifact(summary: dict[str, object]) -> Path:
    artifact_dir = Path(__file__).resolve().parent / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    out = artifact_dir / "phase2_step4_report.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (artifact_dir / "phase2_step4_report.md").write_text(
        "# Phase 2 Step 4 Report\n\n"
        f"- Result: {'PASS' if summary['passed'] else 'FAIL'}\n"
        f"- Cycles: {summary['cycles']}\n"
        f"- Max follow error: {summary['max_follow_error_m']} m\n"
        f"- Tolerance: {summary['tolerance_m']} m\n",
        encoding="utf-8",
    )
    return out


def _build_phase2_review(step13: dict[str, object], step4: dict[str, object], tactile: dict[str, object]) -> dict[str, object]:
    step5_passed = bool(tactile.get("step5_no_contact", {}).get("outputs", {}).get("tactile_rgb"))
    step6_passed = bool(tactile.get("step6_comparisons")) and all(
        item.get("passed") for item in tactile.get("step6_comparisons", [])
    )
    step7_mount = tactile.get("step7_dual_side_mount") or {}
    step7_left = tactile.get("step7_left_no_contact") or {}
    step7_right = tactile.get("step7_right_no_contact") or {}
    step7_passed = (
        bool(step7_mount.get("passed"))
        and bool(step7_left.get("outputs", {}).get("tactile_rgb"))
        and bool(step7_right.get("outputs", {}).get("tactile_rgb"))
    )
    step8_passed = bool(tactile.get("step8_log_dir"))
    steps = {
        "Step 1 - scope locked": {
            "passed": True,
            "evidence": PHASE2_SCOPE_SENTENCE,
        },
        "Step 2 - canonical mount source-of-truth": {
            "passed": True,
            "evidence": step13["source_of_truth"]["primary_robot_reference"],
            "notes": "Generated robot USD remains a derived loading asset.",
        },
        "Step 3 - single-side mount stable": {
            "passed": bool(step13["single_side_mount"]["passed"]),
            "evidence": "phase2_step1_3_report.json",
        },
        "Step 4 - single-side gripper open/close stable": {
            "passed": bool(step4.get("passed")),
            "evidence": "phase2_step4_report.json",
        },
        "Step 5 - GelSightMiniCfg no-contact tactile image": {
            "passed": step5_passed,
            "evidence": "phase2_tactile_report.json: step5_no_contact",
        },
        "Step 6 - no-contact/contact tactile image distinction": {
            "passed": step6_passed,
            "evidence": "phase2_tactile_report.json: step6_comparisons",
        },
        "Step 7 - dual-side mount and no-contact outputs": {
            "passed": step7_passed,
            "evidence": "phase2_tactile_report.json: step7_dual_side_mount + step7_left_no_contact + step7_right_no_contact",
        },
        "Step 8 - minimal tactile logging": {
            "passed": step8_passed,
            "evidence": str(tactile.get("step8_log_dir")),
        },
        "Step 9 - Phase 2 review": {
            "passed": True,
            "evidence": "phase2_review.json / phase2_review.md",
        },
    }
    return {"passed": all(bool(result["passed"]) for result in steps.values()), "steps": steps}


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


def run_preview(sim, robots, bananas, origins, robot_urdf_path, robot_usd_path) -> None:
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
    report_scene_state(origins, robot_urdf_path, robot_usd_path)
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


def _enable_phase2_tactile_debug_windows(include_camera_depth: bool = True) -> None:
    """Turn on TacEx GUI image windows for mounted Phase2 sensor case prims."""

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    for side, paths in phase2_sensor_prim_paths().items():
        case_prim = stage.GetPrimAtPath(paths["case"])
        if not case_prim.IsValid():
            print(f"[WARN] Cannot enable {side} tactile window; missing prim: {paths['case']}")
            continue
        for attr_name in ("debug_tactile_rgb", "debug_camera_depth"):
            if attr_name == "debug_camera_depth" and not include_camera_depth:
                continue
            attr = case_prim.GetAttribute(attr_name)
            if not attr:
                print(f"[WARN] TacEx debug attribute not found on {paths['case']}: {attr_name}")
                continue
            attr.Set(True)


def run_tactile_live_preview(sim, robots, bananas, origins, robot_urdf_path, robot_usd_path) -> None:
    """Run a GUI loop that continuously updates mounted TacEx GelSight outputs."""

    robot_list = list(robots.values())
    banana_list = list(bananas.values())
    robot = robot_list[0]
    sim_dt = sim.get_physics_dt()

    for idx, item in enumerate(robot_list):
        reset_scene(sim, item, banana_list[idx], origins[idx])
    for _ in range(5):
        sim.step()
        for item in robot_list:
            item.update(sim_dt)
        for banana in banana_list:
            banana.update(sim_dt)

    clear_phase2_visual_prims()
    mounted = mount_phase2_sensor_shells(("left", "right"), hide_render_geometry=True)
    camera_check = validate_phase2_sensor_camera_prims(("left", "right"))
    mount_check = validate_phase2_sensor_mounts(("left", "right"))
    failed = [side for side, check in mount_check.items() if not check["passed"]]
    missing_cameras = [side for side, check in camera_check.items() if not check["camera_exists"]]
    if failed or missing_cameras:
        raise RuntimeError(
            "Cannot start Phase2 tactile live preview; sensor mount/camera validation failed: "
            f"failed_mounts={failed}, missing_cameras={missing_cameras}"
        )

    sensors = []
    for side in ("left", "right"):
        cfg = build_phase2_gsmini_cfg(
            side,
            device=args_cli.device,
            resolution=(args_cli.phase2_tactile_width, args_cli.phase2_tactile_height),
            include_camera_depth=True,
            include_camera_rgb=args_cli.phase2_include_camera_rgb,
            debug_vis=True,
        )
        sensor = initialize_phase2_sensor(cfg)
        sensors.append((side, sensor))

    _enable_phase2_tactile_debug_windows(include_camera_depth=True)
    camera_debug = _set_phase2_debug_camera(sim, robot)
    report_scene_state(origins, robot_urdf_path, robot_usd_path)
    print("[INFO] Phase 2 tactile live preview ready.")
    print("[INFO] Close the Isaac Sim window or use --max_steps to stop.")
    print(
        json.dumps(
            {
                "debug_camera": camera_debug,
                "mounted_sensor_prims": mounted,
                "sensor_paths": phase2_sensor_prim_paths(),
                "windows": ["debug_tactile_rgb", "debug_camera_depth"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    step_count = 0
    while simulation_app.is_running():
        for item in robot_list:
            joint_target = build_reset_joint_target(item)
            item.set_joint_position_target(joint_target)
            item.write_data_to_sim()
        sim.step()
        for item in robot_list:
            item.update(sim_dt)
        for banana in banana_list:
            banana.update(sim_dt)
        for _, sensor in sensors:
            update_phase2_sensor(sensor, sim, dt=sim_dt)
        step_count += 1
        if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
            break


def main() -> int:
    if args_cli.num_envs != 1:
        raise ValueError("Phase 2 Step 1/2/3 currently supports only --num_envs 1.")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
    sim.set_camera_view(CAMERA_EYE, CAMERA_TARGET)
    robot_urdf_path = resolve_robot_urdf_path()
    robot_usd_path = ensure_robot_usd_path(force_conversion=args_cli.regenerate_robot_usd)
    robots, bananas, origins = design_scene(args_cli.num_envs, robot_usd_path=robot_usd_path)
    origins = origins.to(sim.device)
    print("[INFO] Phase 2 scene designed; resetting simulation.")
    sim.reset()
    print("[INFO] Phase 2 simulation reset complete.")

    if args_cli.phase2_tactile_live:
        run_tactile_live_preview(sim, robots, bananas, origins, robot_urdf_path, robot_usd_path)
        return 0

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

    if args_cli.phase2_step4_checks:
        robot = next(iter(robots.values()))
        banana = next(iter(bananas.values()))
        summary = run_gripper_mount_compatibility_validation(
            sim,
            robot,
            banana,
            origins[0],
            cycles=args_cli.gripper_cycles,
        )
        artifact = _write_phase2_step4_artifact(summary)
        print(f"[INFO] Wrote Phase 2 Step 4 artifact to: {artifact}")
        if not summary["passed"]:
            print("[RESULT] Phase 2 Step 4 validation FAILED.")
            return 1
        print("[RESULT] Phase 2 Step 4 validation PASSED.")
        return 0

    if args_cli.phase2_tactile_checks or args_cli.phase2_full_review:
        robot = next(iter(robots.values()))
        banana = next(iter(bananas.values()))
        options = Phase2TactileOptions(
            contact_repeats=args_cli.phase2_contact_repeats,
            save_frames=args_cli.phase2_log_frames,
            include_camera_rgb=args_cli.phase2_include_camera_rgb,
            resolution=(args_cli.phase2_tactile_width, args_cli.phase2_tactile_height),
            debug_vis=args_cli.phase2_tactile_debug_vis,
        )
        try:
            step13 = run_single_side_mount_validation(sim, robot, banana, origins[0])
            _write_phase2_step13_artifact(step13)
            step4 = run_gripper_mount_compatibility_validation(
                sim,
                robot,
                banana,
                origins[0],
                cycles=args_cli.gripper_cycles,
            )
            _write_phase2_step4_artifact(step4)
            tactile = run_phase2_tactile_validation(
                sim,
                robot,
                banana,
                origins[0],
                options=options,
                device=args_cli.device,
                run_dual_side=True,
            )
            if args_cli.phase2_full_review:
                review = _build_phase2_review(step13, step4, tactile)
                review_path = write_phase2_review_artifact(review)
                print(f"[INFO] Wrote Phase 2 review artifact to: {review_path}")
                if not review["passed"]:
                    append_phase2_issue(
                        "Phase2 full review did not pass all exit criteria",
                        json.dumps(review, ensure_ascii=False, indent=2),
                    )
                    print("[RESULT] Phase 2 full review FAILED.")
                    return 1
            if not tactile["passed"]:
                append_phase2_issue(
                    "Phase2 tactile validation did not pass",
                    json.dumps(tactile, ensure_ascii=False, indent=2),
                )
                print("[RESULT] Phase 2 tactile validation FAILED.")
                return 1
            print("[RESULT] Phase 2 tactile validation PASSED.")
            return 0
        except Exception as exc:
            summarize_exception_for_log(
                "ur5_phase2_sim.py --phase2-tactile-checks/--phase2-full-review",
                exc,
            )
            raise

    run_preview(sim, robots, bananas, origins, robot_urdf_path, robot_usd_path)
    return 0


if __name__ == "__main__":
    import os

    exit_code = 1
    fast_validation_exit = (
        args_cli.phase2_step3_checks
        or args_cli.phase2_step4_checks
        or args_cli.phase2_tactile_checks
        or args_cli.phase2_full_review
    )
    try:
        exit_code = main()
        if fast_validation_exit:
            # Isaac/Kit shutdown can block after validation while waiting for
            # background USD/resource operations.  The validation artifacts have
            # already been flushed at this point; hard-exit keeps CI/headless
            # Phase2 checks deterministic and leaves no child process behind.
            os._exit(exit_code)
    finally:
        if not fast_validation_exit:
            simulation_app.close(wait_for_replicator=False)
    raise SystemExit(exit_code)
