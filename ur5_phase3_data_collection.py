"""Phase 3 data-collection entrypoint.

This script launches Isaac, creates one locked Phase 3 scene, executes a small
deterministic contact/grasp trial batch, and writes aligned tactile/robot-state
artifacts.  It intentionally does not run Sparsh, learning, GraspNet, RL, or
closed-loop tactile policy logic.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from isaaclab.app import AppLauncher

from ur5_phase3_objects import OBJECT_SEQUENCE, get_object_profile, get_protocol_profile
from ur5_phase3_schema import PHASE3_FORBIDDEN_SCOPE, PHASE3_SCOPE_SENTENCE

parser = argparse.ArgumentParser(description="Phase 3 deterministic grasp/contact trial data collection.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments. Phase 3 currently supports 1.")
parser.add_argument(
    "--object_id",
    choices=OBJECT_SEQUENCE,
    default="cube",
    help="Locked Phase 3 object profile to run.",
)
parser.add_argument(
    "--protocol_variant",
    choices=("contact_only", "contact_hold", "contact_hold_micro_lift"),
    default="contact_hold_micro_lift",
    help="Locked Phase 3 protocol variant.",
)
parser.add_argument("--trials", type=int, default=1, help="Number of deterministic trials to run for this object.")
parser.add_argument("--seed", type=int, default=7, help="Base deterministic seed.")
parser.add_argument("--output_root", type=Path, default=Path("artifacts/phase3"), help="Phase 3 artifact root.")
parser.add_argument(
    "--profile_close_rad",
    type=float,
    default=None,
    help=(
        "Optional local collection override for the object profile close target. "
        "Used by Phase 4 force-oriented cube batches to create different contact levels."
    ),
)
parser.add_argument(
    "--profile_max_close_object_lift_m",
    type=float,
    default=None,
    help=(
        "Optional local collection override for the close-stage object-lift guard. "
        "Use sparingly for Phase 4 exploratory label-generation batches."
    ),
)
parser.add_argument(
    "--profile_stable_force_threshold_n",
    type=float,
    default=None,
    help=(
        "Optional local collection override for the stable-force stop threshold. "
        "Used by Phase 5 force-regime sweeps when collecting harder cube closes."
    ),
)
parser.add_argument(
    "--profile_high_force_threshold_n",
    type=float,
    default=None,
    help=(
        "Optional local collection override for the high-force stop/failure threshold. "
        "Use only for bounded simulation label sweeps; it changes trial success semantics."
    ),
)
parser.add_argument("--max_steps", type=int, default=0, help="Global per-trial step budget; 0 disables.")
parser.add_argument("--reset_settle_steps", type=int, default=20)
parser.add_argument("--pregrasp_move_steps", type=int, default=40)
parser.add_argument("--direct_move_steps", type=int, default=320)
parser.add_argument(
    "--approach_steps",
    type=int,
    default=None,
    help="Alias for --direct_move_steps: number of IK steps used for the object approach segment.",
)
parser.add_argument("--direct_pos_tolerance_m", type=float, default=0.006)
parser.add_argument("--direct_pass_tolerance_m", type=float, default=0.008)
parser.add_argument("--close_steps", type=int, default=320)
parser.add_argument("--close_settle_steps", type=int, default=6)
parser.add_argument("--release_steps", type=int, default=60)
parser.add_argument("--arm_hold_settle_steps", type=int, default=50)
parser.add_argument("--soft_center_refine_rounds", type=int, default=2)
parser.add_argument("--soft_center_refine_steps", type=int, default=50)
parser.add_argument("--max_soft_center_refine_step_m", type=float, default=0.020)
parser.add_argument("--sample_every_steps", type=int, default=4)
parser.add_argument(
    "--tactile_sample_every_steps",
    type=int,
    default=4,
    help=(
        "Capture tactile/robot samples every N physics steps during approach/close/hold/lift. "
        "Use 1 for dense continuous tactile sequences; 0 disables periodic samples except forced stage boundaries."
    ),
)
parser.add_argument("--tactile_sides", choices=("left", "right", "both"), default="both")
parser.add_argument(
    "--sim_device",
    default=None,
    help=(
        "Optional device for the IsaacLab SimulationContext. Use this to run the app/render/TacEx stack "
        "with --device cuda:0 while keeping Phase3 PhysX/motion on cpu."
    ),
)
parser.add_argument(
    "--tactile_device",
    default="cuda:0",
    help=(
        "Device used by TacEx GelSight optical simulation. Keep this on cuda:0; "
        "TacEx camera contact rendering is unreliable on CPU."
    ),
)
parser.add_argument("--phase3_tactile_width", type=int, default=320)
parser.add_argument("--phase3_tactile_height", type=int, default=240)
parser.add_argument("--include_camera_rgb", action="store_true", help="Also log TacEx camera RGB when available.")
parser.add_argument("--no_camera_depth", action="store_true", help="Do not log camera_depth.")
parser.add_argument("--disable_tactile", action="store_true", help="Run protocol without TacEx tactile sensors.")
parser.add_argument("--no_tactile_arrays", action="store_true", help="Do not save .npy tactile arrays.")
parser.add_argument("--no_preview_images", action="store_true", help="Do not save PNG/PPM preview images.")
parser.add_argument(
    "--disable_tactile_contact_imprint",
    action="store_true",
    help=(
        "Disable Phase3 continuous contact-imprint rendering. By default Phase3 feeds contact-geometry "
        "indentation into TacEx/Taxim so tactile_rgb varies through close and micro-lift."
    ),
)
parser.add_argument("--tactile_imprint_min_depth_mm", type=float, default=0.08)
parser.add_argument("--tactile_imprint_max_depth_mm", type=float, default=2.5)
parser.add_argument(
    "--tactile_imprint_depth_per_mm_overlap",
    type=float,
    default=0.65,
    help="Additional Taxim imprint depth in mm per mm of soft/object AABB overlap beyond the contact margin.",
)
parser.add_argument("--tactile_imprint_sigma_x", type=float, default=0.34)
parser.add_argument("--tactile_imprint_sigma_y", type=float, default=0.42)
parser.add_argument("--disable_force_control", action="store_true", help="Disable GSmini contact force sensors.")
parser.add_argument("--regenerate_robot_usd", action="store_true", help="Force canonical URDF -> USD regeneration.")
parser.add_argument(
    "--graceful_sim_close",
    action="store_true",
    help=(
        "Call simulation_app.close() before process exit. Disabled by default for Phase3 headless data "
        "runs because Isaac/Kit can hang after artifacts are written; process exit plus cleanup is more reliable."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[INFO] Isaac SimulationApp ready for Phase 3.", flush=True)

import isaaclab.sim as sim_utils

from ur5_phase1_scene import CAMERA_EYE, CAMERA_TARGET, ensure_robot_usd_path, resolve_robot_urdf_path
from ur5_phase3_objects import design_phase3_scene, write_phase3_profiles
from ur5_phase3_schema import write_phase3_schema
from ur5_phase3_trial_runner import (
    Phase3RunnerOptions,
    Phase3TrialRunner,
    setup_phase3_contact_sensors,
)
from ur5_phase3_review import write_phase3_review


def _tactile_sides() -> tuple[str, ...]:
    if args_cli.tactile_sides == "both":
        return ("left", "right")
    return (args_cli.tactile_sides,)


def _write_batch_summary(output_root: Path, summary: dict) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "phase3_batch_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> int:
    if args_cli.num_envs != 1:
        raise ValueError("Phase 3 currently supports only --num_envs 1.")
    if args_cli.trials < 1:
        raise ValueError("--trials must be >= 1")

    object_profile = get_object_profile(args_cli.object_id)
    profile_overrides = {}
    if args_cli.profile_close_rad is not None:
        profile_overrides["close_rad"] = args_cli.profile_close_rad
    if args_cli.profile_max_close_object_lift_m is not None:
        profile_overrides["max_close_object_lift_m"] = args_cli.profile_max_close_object_lift_m
    if args_cli.profile_stable_force_threshold_n is not None:
        profile_overrides["stable_force_threshold_n"] = args_cli.profile_stable_force_threshold_n
    if args_cli.profile_high_force_threshold_n is not None:
        profile_overrides["high_force_threshold_n"] = args_cli.profile_high_force_threshold_n
    if profile_overrides:
        object_profile = replace(object_profile, **profile_overrides)
    protocol_profile = get_protocol_profile(args_cli.protocol_variant)
    output_root = args_cli.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    write_phase3_schema(output_root / "trial_schema.json")
    write_phase3_profiles(output_root / "object_protocol_profiles.json")

    print("[INFO] Phase 3 scope:", PHASE3_SCOPE_SENTENCE, flush=True)
    print("[INFO] Phase 3 forbidden scope:", PHASE3_FORBIDDEN_SCOPE, flush=True)
    print("[INFO] Object profile:", json.dumps(object_profile.as_dict(), ensure_ascii=False), flush=True)
    print("[INFO] Protocol profile:", json.dumps(protocol_profile.as_dict(), ensure_ascii=False), flush=True)
    sim_device = args_cli.sim_device or args_cli.device
    print(
        f"[INFO] Creating SimulationContext on device={sim_device} (AppLauncher/device={args_cli.device})",
        flush=True,
    )
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=sim_device))
    sim.set_camera_view(CAMERA_EYE, CAMERA_TARGET)

    robot_urdf_path = resolve_robot_urdf_path()
    robot_usd_path = ensure_robot_usd_path(force_conversion=args_cli.regenerate_robot_usd)
    robots, grasp_objects, origins = design_phase3_scene(
        args_cli.num_envs,
        robot_usd_path=robot_usd_path,
        object_profile=object_profile,
        activate_contact_sensors=not args_cli.disable_force_control,
    )
    contact_sensors = setup_phase3_contact_sensors(object_profile, disabled=args_cli.disable_force_control)
    origins = origins.to(sim.device)
    sim.reset()
    robot = next(iter(robots.values()))
    grasp_object = next(iter(grasp_objects.values()))
    origin = origins[0]

    options = Phase3RunnerOptions(
        output_root=output_root,
        seed=args_cli.seed,
        max_steps=args_cli.max_steps,
        reset_settle_steps=args_cli.reset_settle_steps,
        pregrasp_move_steps=args_cli.pregrasp_move_steps,
        direct_move_steps=args_cli.approach_steps if args_cli.approach_steps is not None else args_cli.direct_move_steps,
        direct_pos_tolerance_m=args_cli.direct_pos_tolerance_m,
        direct_pass_tolerance_m=args_cli.direct_pass_tolerance_m,
        soft_center_refine_rounds=args_cli.soft_center_refine_rounds,
        soft_center_refine_steps=args_cli.soft_center_refine_steps,
        max_soft_center_refine_step_m=args_cli.max_soft_center_refine_step_m,
        close_steps=args_cli.close_steps,
        close_settle_steps=args_cli.close_settle_steps,
        release_steps=args_cli.release_steps,
        arm_hold_settle_steps=args_cli.arm_hold_settle_steps,
        sample_every_steps=args_cli.sample_every_steps,
        tactile_sample_every_steps=args_cli.tactile_sample_every_steps,
        tactile_sides=_tactile_sides(),
        tactile_device=args_cli.tactile_device,
        tactile_resolution=(args_cli.phase3_tactile_width, args_cli.phase3_tactile_height),
        include_camera_depth=not args_cli.no_camera_depth,
        include_camera_rgb=args_cli.include_camera_rgb,
        disable_tactile=args_cli.disable_tactile,
        save_tactile_arrays=not args_cli.no_tactile_arrays,
        save_preview_images=not args_cli.no_preview_images,
        tactile_contact_imprint_enabled=not args_cli.disable_tactile_contact_imprint,
        tactile_imprint_min_depth_mm=args_cli.tactile_imprint_min_depth_mm,
        tactile_imprint_max_depth_mm=args_cli.tactile_imprint_max_depth_mm,
        tactile_imprint_depth_per_mm_overlap=args_cli.tactile_imprint_depth_per_mm_overlap,
        tactile_imprint_sigma_x=args_cli.tactile_imprint_sigma_x,
        tactile_imprint_sigma_y=args_cli.tactile_imprint_sigma_y,
        disable_force_control=args_cli.disable_force_control,
    )
    runner = Phase3TrialRunner(
        sim,
        robot,
        grasp_object,
        origin,
        object_profile=object_profile,
        protocol_profile=protocol_profile,
        contact_sensors=contact_sensors,
        options=options,
    )
    tactile_setup = runner.setup_tactile_sensors()
    print("[INFO] Phase3 tactile setup:", json.dumps(tactile_setup, ensure_ascii=False), flush=True)

    results = []
    for trial_index in range(1, args_cli.trials + 1):
        print(f"[PHASE3] Starting trial {trial_index}/{args_cli.trials}", flush=True)
        result = runner.run_trial(trial_index=trial_index)
        print("[PHASE3] Trial result:", json.dumps(result, ensure_ascii=False), flush=True)
        results.append(result)

    review_path = write_phase3_review(output_root)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    trial_motion_passed = all(result.get("passed") for result in results)
    artifact_review_passed = bool(review.get("artifact_validation_passed"))
    summary = {
        "scope": PHASE3_SCOPE_SENTENCE,
        "forbidden_scope": PHASE3_FORBIDDEN_SCOPE,
        "object_id": object_profile.object_id,
        "protocol_variant": protocol_profile.variant,
        "trials_requested": args_cli.trials,
        "trials_passed": sum(1 for result in results if result.get("passed")),
        "artifact_review_passed": artifact_review_passed,
        "passed": trial_motion_passed and artifact_review_passed,
        "robot_urdf_path": robot_urdf_path.as_posix(),
        "robot_usd_path": robot_usd_path.as_posix(),
        "runner_options": asdict(options)
        | {
            "output_root": output_root.as_posix(),
            "app_launcher_device": args_cli.device,
            "simulation_context_device": sim_device,
        },
        "results": results,
    }
    summary_path = _write_batch_summary(output_root, summary)
    print(f"[INFO] Wrote Phase3 batch summary: {summary_path}", flush=True)
    print(f"[INFO] Wrote Phase3 review artifact: {review_path}", flush=True)
    print("[RESULT] Phase 3 batch PASSED." if summary["passed"] else "[RESULT] Phase 3 batch FAILED.", flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    import os
    import sys

    exit_code = 1
    try:
        exit_code = main()
    finally:
        if args_cli.graceful_sim_close:
            simulation_app.close(wait_for_replicator=False)
        else:
            print("[INFO] Skipping graceful SimulationApp close; process exit will release Phase3 headless resources.", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
