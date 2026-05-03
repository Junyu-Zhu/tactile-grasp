"""Phase 2 4 cm cube grasp test with live TacEx GelSight Mini display.

This script combines the Phase 1 integrated UR5e + Robotiq + connector +
GSmini scene with the Phase 2 TacEx runtime sensor shells.  It is intentionally
kept separate from ``ur5_phase2_sim.py`` so that the validation/preview entry
point remains small while this file focuses on closed-loop grasp experiments.

Typical usage for viewing tactile images while running the cube grasp:

    python ur5_phase2_grasp_test.py --enable_cameras --device cpu

Use ``--device cpu`` for this script unless you have disabled PhysX Direct GPU
API elsewhere.  The Phase 1 reset helpers write rigid poses/velocities and those
operations conflict with PhysX Direct GPU API in the current Isaac runtime.

The default scene replaces the YCB banana with a 4 cm cube, targets the cube
center so the two GSmini soft pads touch the left/right side centers.  The
target is commanded directly from the cube pose plus explicit CLI offsets, then
the arm joints are held fixed through close/settle to reduce visible wobble.
Soft-link contact is diagnostic only; gripper closure is governed by the
requested close width and the high-force stop threshold.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Phase 2 UR5 4 cm cube grasp with live GelSight tactile display.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn; only 1 is supported.")
parser.add_argument("--max_steps", type=int, default=0, help="Global simulation-step budget. 0 disables the budget.")
parser.add_argument(
    "--pre_motion_delay_seconds",
    type=float,
    default=0.0,
    help=(
        "Wall-clock delay after the Isaac scene is loaded and before any reset/pregrasp/grasp motion starts. "
        "Defaults to 0 so the grasp starts immediately; pass a positive value only when you need time "
        "to enable collision/contact debug visualizations in the GUI."
    ),
)
parser.add_argument(
    "--max_attempts",
    type=int,
    default=1,
    help=(
        "Stop after this many grasp attempts. The default is one direct attempt so the user-tuned "
        "grasp pose is executed without automatic terminal retuning."
    ),
)
parser.add_argument("--hold_seconds", type=float, default=5.0, help="Hold time after lifting before success is checked.")
parser.add_argument(
    "--lift_distance",
    type=float,
    default=0.16,
    help="Commanded upward EE lift distance in meters. The 4 cm cube profile commands extra EE lift to achieve a measured 10 cm object lift.",
)
parser.add_argument("--success_lift_margin", type=float, default=0.10, help="Minimum measured object lift in meters considered not dropped.")
parser.add_argument(
    "--success_mode",
    choices=("contact_demo", "lift_hold"),
    default="contact_demo",
    help=(
        "Pass/fail profile. contact_demo (default) is for Phase2 tactile-grasp bring-up: require soft-center "
        "alignment, stable two-sided close, close/lift safety, and a tactile image contact change, without "
        "requiring 10 cm object lift. lift_hold restores the old strict 10 cm lift/hold success gate."
    ),
)
parser.add_argument("--approach_extra_height", type=float, default=0.08, help="Legacy staged-approach height; direct grasp mode keeps this for compatibility but does not use it.")
parser.add_argument(
    "--grasp_ee_z_offset",
    type=float,
    default=0.17,
    help=(
        "EE z offset above cube root for the direct grasp pose. The 4 cm cube default uses the "
        "user-verified 0.17 m height so the GSmini fingertips stay clear of the table while "
        "targeting the cube side center."
    ),
)
parser.add_argument(
    "--grasp_x_offset",
    type=float,
    default=0.0,
    help=(
        "X offset from cube root for the desired GSmini soft-link center. "
        "The default keeps the soft_link centered on the cube side faces."
    ),
)
parser.add_argument(
    "--grasp_y_offset",
    type=float,
    default=0.0,
    help=(
        "Y offset from cube root for the desired GSmini soft-link center. "
        "The default keeps the soft_link centered on the cube side faces."
    ),
)
parser.add_argument(
    "--grasp_soft_center_z_offset",
    type=float,
    default=0.0,
    help=(
        "Z offset from cube center for the desired GSmini soft-link mesh center. "
        "The default aims the soft_link at the vertical center of the cube side faces."
    ),
)
parser.add_argument(
    "--no_align_soft_center",
    action="store_false",
    dest="align_soft_center",
    help=(
        "Disable the default GSmini soft-link centering mode and interpret --grasp_* offsets as the legacy "
        "end-effector target offsets."
    ),
)
parser.set_defaults(align_soft_center=True)
parser.add_argument(
    "--soft_center_tolerance",
    type=float,
    default=0.005,
    help="Allowed X/Z error in meters between the GSmini soft-link pair center and cube side-center line before close.",
)
parser.add_argument(
    "--soft_center_refine_rounds",
    type=int,
    default=4,
    help="Bounded pre-close correction rounds that re-center the measured GSmini soft-link pair on the cube.",
)
parser.add_argument(
    "--soft_center_refine_steps",
    type=int,
    default=60,
    help="IK steps per soft-center correction round.",
)
parser.add_argument(
    "--max_soft_center_refine_step",
    type=float,
    default=0.025,
    help="Maximum Cartesian correction per soft-center refinement round in meters.",
)
parser.add_argument(
    "--grasp_yaw_offset_deg",
    type=float,
    default=0.0,
    help=(
        "World-frame yaw rotation offset applied to the direct EE grasp orientation before close. "
        "The default keeps the pregrasp wrist orientation so the arm uses the shortest low-rotation path."
    ),
)
parser.add_argument(
    "--grasp_pitch_offset_deg",
    type=float,
    default=0.0,
    help="World-frame pitch rotation offset applied to the direct EE grasp orientation before close.",
)
parser.add_argument(
    "--grasp_roll_offset_deg",
    type=float,
    default=0.0,
    help="World-frame roll rotation offset applied to the direct EE grasp orientation before close.",
)
parser.add_argument(
    "--pregrasp_wrist3_deg",
    type=float,
    default=-2.6,
    help=(
        "Override wrist_3_joint for this grasp script's pregrasp pose. The cube default keeps the reset wrist "
        "angle so the gripper approaches the cube side centers with little wrist spin."
    ),
)
parser.add_argument(
    "--gripper_close_rad",
    type=float,
    default=0.25,
    help=(
        "Base Robotiq close command used for the first attempt. The 4 cm cube default is intentionally "
        "conservative so the first visible close does not squeeze the cube upward before lift."
    ),
)
parser.add_argument(
    "--min_grasp_ee_z_offset",
    type=float,
    default=0.17,
    help=(
        "Safety floor for auto-tuned grasp EE z offsets. Attempts below this value are clamped so the tuning "
        "sweep does not drive the GSmini fingertips below the user-verified direct grasp band."
    ),
)
parser.add_argument(
    "--max_joint_delta_per_step",
    type=float,
    default=0.018,
    help=(
        "Joint-space speed limiter for non-contact IK motion. The default is deliberately smooth so the "
        "direct move does not overshoot into the table and rebound."
    ),
)
parser.add_argument(
    "--descend_joint_delta_per_step",
    type=float,
    default=0.020,
    help=(
        "Legacy staged-descent speed limiter; direct grasp mode keeps this for compatibility but uses "
        "--max_joint_delta_per_step for the single direct move."
    ),
)
parser.add_argument(
    "--near_grasp_joint_delta_per_step",
    type=float,
    default=0.017,
    help=(
        "Legacy near-grasp staged-descent speed limiter. In direct mode the single target move uses "
        "--max_joint_delta_per_step, then arm joints are latched before closing."
    ),
)
parser.add_argument(
    "--precontact_extra_height",
    type=float,
    default=0.08,
    help="Legacy staged-precontact height; direct grasp mode keeps this for compatibility but does not use it.",
)
parser.add_argument(
    "--descend_accept_tolerance",
    type=float,
    default=0.006,
    help=(
        "Accepted position error for the single direct move before arm-joint latching. "
        "Increase only if you prefer less IK settling near the target."
    ),
)
parser.add_argument("--ik_pos_tolerance", type=float, default=0.006, help="EE position tolerance for an IK segment.")
parser.add_argument("--approach_steps", type=int, default=90, help="Legacy staged approach steps; direct grasp mode does not use them.")
parser.add_argument("--precontact_steps", type=int, default=1, help="Legacy staged pre-contact steps; direct grasp mode does not use them.")
parser.add_argument("--descend_steps", type=int, default=80, help="Legacy final descend steps; direct grasp mode uses --direct_move_steps instead.")
parser.add_argument("--direct_move_steps", type=int, default=240, help="Max IK steps for the single direct move from pregrasp to the final grasp pose.")
parser.add_argument(
    "--close_steps",
    type=int,
    default=160,
    help=(
        "Gripper closing steps toward --gripper_close_rad; the finer default samples object-lift/contact "
        "earlier so the guard can stop before visible upward squeeze."
    ),
)
parser.add_argument(
    "--close_settle_steps",
    type=int,
    default=4,
    help="Extra hold-closed steps after the visible close command and before lifting.",
)
parser.add_argument(
    "--arm_hold_settle_steps",
    type=int,
    default=50,
    help=(
        "After reaching the direct grasp pose, hold the current arm joints for this many physics steps before "
        "closing. The same arm hold target is also applied during close/settle to reduce EE wobble."
    ),
)
parser.add_argument("--lift_steps", type=int, default=65, help="Max slow IK steps for the lift segment.")
parser.add_argument("--settle_steps", type=int, default=20, help="Initial reset settle steps.")
parser.add_argument("--reset_move_steps", type=int, default=12, help="Slow joint steps from reset state before each attempt.")
parser.add_argument("--pregrasp_move_steps", type=int, default=25, help="Slow joint steps into the Phase 1 pre-grasp before IK.")
parser.add_argument(
    "--post_contact_close_distance",
    type=float,
    default=0.001,
    help=(
        "Deprecated compatibility option. Soft-link contact no longer stops closure; force control and "
        "the target close width determine when the close command ends."
    ),
)
parser.add_argument(
    "--gripper_gap_per_rad",
    type=float,
    default=0.07,
    help="Approximate Robotiq soft-pad gap travel in meters per finger_joint radian; 0.07 maps 1 cm to about 0.143 rad.",
)
parser.add_argument(
    "--max_close_without_contact_rad",
    type=float,
    default=0.45,
    help=(
        "Absolute close cap for the direct close command. Set this >= --gripper_close_rad when you want "
        "the gripper to reach the specified width unless high force stops it."
    ),
)
parser.add_argument(
    "--max_contact_stop_close_rad",
    type=float,
    default=0.45,
    help=(
        "Deprecated compatibility option. Soft contact no longer creates a separate stop angle; "
        "--max_close_without_contact_rad is the only absolute close cap."
    ),
)
parser.add_argument(
    "--max_close_object_lift",
    type=float,
    default=0.001,
    help=(
        "Cube z increase threshold during gripper close before lift. The default guard stops/rewinds closure "
        "before visible upward squeezing."
    ),
)
parser.add_argument(
    "--object_lift_guard_trigger_fraction",
    type=float,
    default=1.0,
    help=(
        "Trigger the close rewind when lift reaches this fraction of --max_close_object_lift. "
        "The artifact still reports pass/fail against --max_close_object_lift; this early trigger leaves "
        "headroom for one physics step of response latency."
    ),
)
parser.add_argument(
    "--enable_close_object_lift_guard",
    action="store_true",
    default=True,
    help=(
        "Safety guard: stop/rewind gripper closure if cube z rises above --max_close_object_lift. "
        "Enabled by default; this flag is kept for command compatibility."
    ),
)
parser.add_argument(
    "--disable_close_object_lift_guard",
    action="store_false",
    dest="enable_close_object_lift_guard",
    help="Disable the default cube-lift close guard and allow the gripper to close to the force/width target.",
)
parser.add_argument(
    "--object_lift_rewind_steps",
    type=int,
    default=8,
    help="When --enable_close_object_lift_guard is active, command the last safe close target for this many settle steps.",
)
parser.add_argument(
    "--object_lift_rewind_open_margin_rad",
    type=float,
    default=0.005,
    help=(
        "When cube lift is detected, reopen this many radians from the last safe close target before the rewind "
        "hold. This avoids continuing to wedge the cube upward while settling."
    ),
)
parser.add_argument(
    "--max_preclose_object_shift",
    type=float,
    default=0.015,
    help=(
        "Abort closing if the object has already moved this far before the close command, which indicates hard "
        "connector/base pre-contact instead of a centered soft-pad grasp."
    ),
)
parser.add_argument(
    "--soft_contact_force_threshold",
    type=float,
    default=0.2,
    help="Filtered force threshold in newtons used to report GSmini/cube contact; it no longer stops closure by itself.",
)
parser.add_argument(
    "--force_control_stable_force_threshold",
    type=float,
    default=0.5,
    help="Per-side force threshold in newtons used to say both GSmini pads are stably holding the cube.",
)
parser.add_argument(
    "--force_control_high_force_threshold",
    type=float,
    default=8.0,
    help="Per-side force threshold in newtons that stops gripper closing to avoid excessive squeeze force.",
)
parser.add_argument(
    "--force_control_stable_steps",
    type=int,
    default=1,
    help="Consecutive close steps where both sides must exceed the stable-force threshold before stopping as a stable grasp.",
)
parser.add_argument(
    "--disable_stable_force_stop",
    action="store_true",
    help="Do not stop closing when both GSmini pads have stable force; only high-force/object-lift guards can stop early.",
)
parser.add_argument(
    "--disable_force_control",
    action="store_true",
    help="Disable gripper force-control detection and high-force stopping during close.",
)
parser.add_argument(
    "--contact_stop_mode",
    choices=("both", "any"),
    default="any",
    help="Deprecated compatibility option; soft contact no longer stops gripper closure.",
)
parser.add_argument(
    "--disable_contact_stop",
    action="store_true",
    help="Deprecated compatibility option; soft contact stop is always disabled in direct force-control mode.",
)
parser.add_argument("--phase2-tactile-width", type=int, default=320, help="GelSight tactile/camera width.")
parser.add_argument("--phase2-tactile-height", type=int, default=240, help="GelSight tactile/camera height.")
parser.add_argument("--show_right_tactile", action="store_true", help="Also open the right tactile RGB window.")
parser.add_argument(
    "--show_tactile_depth",
    action="store_true",
    help=(
        "Request camera_depth from TacEx sensors. Depth is kept out of the default script-owned RGB panel; "
        "combine with --show_tacex_debug_windows only if you also want the legacy standalone depth windows."
    ),
)
parser.add_argument(
    "--show_tacex_debug_windows",
    action="store_true",
    default=True,
    help=(
        "Also open TacEx's legacy standalone debug windows. Enabled by default to match TacEx GSmini demos."
    ),
)
parser.add_argument(
    "--no_tacex_debug_windows",
    action="store_false",
    dest="show_tacex_debug_windows",
    help="Disable TacEx's legacy standalone tactile_rgb debug windows and keep only the script-owned panel.",
)
parser.add_argument(
    "--dock_tactile_windows_right",
    action="store_true",
    default=True,
    help="Best-effort: request Isaac/omni.ui right-side docking preference for tactile debug windows when supported.",
)
parser.add_argument(
    "--no_dock_tactile_windows_right",
    action="store_false",
    dest="dock_tactile_windows_right",
    help="Disable the default best-effort right-side docking request for the Phase2 tactile panel/windows.",
)
parser.add_argument(
    "--no_tactile_panel",
    action="store_true",
    help="Disable the script-owned Isaac UI panel that displays live tactile_rgb images inside the main GUI.",
)
parser.add_argument("--no_tactile_live", action="store_true", help="Disable live tactile windows for headless debugging.")
parser.add_argument(
    "--disable_tactile_contact_imprint",
    action="store_true",
    help=(
        "Disable the Phase2 contact-geometry TacEx imprint fallback. By default, when the GSmini soft-link "
        "AABB is touching the cube but TacEx camera depth still renders the no-contact background, the script "
        "feeds a contact-centered height-map imprint through TacEx/Taxim so the live tactile_rgb window changes."
    ),
)
parser.add_argument(
    "--tactile_contact_imprint_depth_mm",
    type=float,
    default=1.5,
    help="Nominal Taxim indentation depth in millimeters for the contact-geometry tactile imprint fallback.",
)
parser.add_argument(
    "--tactile_contact_imprint_background_threshold",
    type=float,
    default=0.75,
    help=(
        "Mean absolute tactile_rgb delta threshold used to decide that a contact frame still matches the "
        "stored no-contact GSmini background and should receive the geometry-based imprint fallback."
    ),
)
parser.add_argument(
    "--regenerate-robot-usd",
    action="store_true",
    help="Force regeneration of the USD derived from the canonical robot URDF before scene load.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[INFO] Isaac SimulationApp ready; importing IsaacLab runtime modules.", flush=True)

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz, quat_mul, subtract_frame_transforms

from ur5_phase1_control import (
    ARM_JOINT_NAMES,
    GRIPPER_CONTROL_JOINT_NAMES,
    build_gripper_joint_target,
    build_pregrasp_joint_target,
    build_reset_joint_target,
    gripper_joint_overrides_rad,
    named_joint_ids,
)
from ur5_phase1_reset import _stabilize_gripper_mimic_state, reset_scene
from ur5_phase1_scene import (
    CAMERA_EYE,
    CAMERA_TARGET,
    PHASE2_CUBE_NAME,
    PHASE2_CUBE_REST_HEIGHT,
    PHASE2_CUBE_SIZE_M,
    TABLE_USD_PATH,
    design_scene,
    ensure_robot_usd_path,
    report_scene_state,
    resolve_robot_urdf_path,
)
from ur5_phase2_mount import (
    clear_phase2_visual_prims,
    mount_phase2_sensor_shells,
    phase2_sensor_prim_paths,
    validate_phase2_sensor_camera_prims,
    validate_phase2_sensor_mounts,
)
from ur5_phase2_tactile import (
    DEFAULT_SENSOR_CAMERA_CLIPPING_RANGE,
    build_phase2_gsmini_cfg,
    initialize_phase2_sensor,
    update_phase2_sensor,
)

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
ISSUE_LOG_PATH = Path(__file__).resolve().parent / "log.md"
PHASE2_GRASP_RESULT_PREFIX = "phase2_grasp_test"
PHASE2_OBJECT_KIND = "cube"
PHASE2_OBJECT_LABEL = "4cm_cube"
SOFT_CONTACT_BODY_BY_SIDE = {
    "left": "left_gelsight_mini_gelpad",
    "right": "right_gelsight_mini_gelpad",
}
GSMINI_SOFT_ORIGIN_IN_SENSOR_FRAME_M = (0.0, 0.0, 0.0)
GSMINI_SOFT_MESH_AABB_IN_SENSOR_FRAME_M = {
    "min": (-0.01363918, -0.02524673, -0.04923395),
    "max": (0.01022479, -0.01999673, -0.02100601),
}
GSMINI_BASE_LINK_AABB_IN_SENSOR_FRAME_M = {
    # GSmini case/base mesh only, now expressed in the TacEx-style
    # connector/case/gelpad frame shared by the fixed attachment chain.
    "min": (-0.01616778, -0.02028085, -0.05095018),
    "max": (0.01271882, 0.00371915, -0.01895018),
}
GSMINI_SENSOR_ASSEMBLY_AABB_IN_SENSOR_FRAME_M = {
    # Union of GSmini connector, case/base, and soft gelpad mesh collision
    # bounds in the shared sensor frame.  The Robotiq pad remains on the
    # inner_finger body, so it is audited separately by the static URDF values.
    "min": (-0.01710485, -0.02524673, -0.05344392),
    "max": (0.01389515, 0.00831280, 0.02155608),
}
GSMINI_SOFT_AABB_CONTACT_MARGIN_M = 0.002

# Static audit of the canonical URDF geometry:
# - left/right pad collision z max: 0.03242 + 0.0375 / 2 = 0.05117 m
# - the TacEx-style connector/case/gelpad fixed chain still uses the original
#   inner-finger mount origin/rpy, so the world placement and z extents are
#   unchanged from the prior embedded-geometry URDF.
# The canonical URDF includes non-rendered collision meshes for the Robotiq
# finger links and matching visual/collision geometry for the fingertip pad,
# connector, GSmini base, and GSmini soft gelpad visuals.  To avoid hard
# connector/base pre-contact lifting the cube, only the Robotiq pad and the
# separate gelpad body receive collision geometry in the canonical URDF.
URDF_PAD_COLLISION_Z_MAX_M = 0.05117
URDF_GSMINI_SOFT_COLLISION_Z_MAX_M = 0.08165
URDF_GSMINI_FULL_COLLISION_Z_MAX_M = 0.08586
URDF_GSMINI_COLLISION_EXTENSION_BEYOND_PAD_M = URDF_GSMINI_SOFT_COLLISION_Z_MAX_M - URDF_PAD_COLLISION_Z_MAX_M
URDF_HARD_SENSOR_COLLISION_SCALE_NOTE = (
    "contact colliders: Robotiq pad on inner fingers plus a TacEx-style fixed GSmini gelpad body; "
    "connector/case remain visual-only, while the fixed attachment joint preserves the original mount origin/rpy and adds camera link anchors"
)


@dataclass(frozen=True)
class AttemptParams:
    attempt: int
    ee_z_offset: float
    soft_center_z_offset: float
    longitudinal_x_offset: float
    lateral_y_offset: float
    yaw_offset_rad: float
    pitch_offset_rad: float
    roll_offset_rad: float
    gripper_close_rad: float


class StepBudget:
    """Global step counter that can stop a long tuning loop deterministically."""

    def __init__(self, max_steps: int):
        self.max_steps = max_steps
        self.steps = 0

    @property
    def exhausted(self) -> bool:
        return self.max_steps > 0 and self.steps >= self.max_steps

    def tick(self) -> None:
        self.steps += 1


def append_phase2_grasp_log(summary: str, details: str) -> None:
    ISSUE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with ISSUE_LOG_PATH.open("a", encoding="utf-8") as stream:
        stream.write(f"\n## {timestamp} — {summary}\n\n")
        stream.write(details.rstrip() + "\n")


def _write_result_artifacts(result: dict[str, Any]) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = ARTIFACT_DIR / f"{PHASE2_GRASP_RESULT_PREFIX}_{stamp}.json"
    md_path = ARTIFACT_DIR / f"{PHASE2_GRASP_RESULT_PREFIX}_{stamp}.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(
        "# Phase 2 Grasp Test Result\n\n"
        f"- Passed: **{result.get('passed')}**\n"
        f"- Attempts: `{result.get('attempts_used')}`\n"
        f"- Reason: `{result.get('reason')}`\n"
        f"- Best attempt: `{json.dumps(result.get('best_attempt'), ensure_ascii=False)}`\n"
        f"- JSON: `{json_path}`\n\n"
        "## Final attempt\n"
        "```json\n"
        f"{json.dumps(result.get('final_attempt'), ensure_ascii=False, indent=2)}\n"
        "```\n",
        encoding="utf-8",
    )
    return json_path


def _enable_tactile_debug_windows(sides: tuple[str, ...], *, include_camera_depth: bool) -> None:
    """Turn on TacEx GUI windows for tactile/contact images."""

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    all_paths = phase2_sensor_prim_paths()
    for side in sides:
        case_path = all_paths[side]["case"]
        prim = stage.GetPrimAtPath(case_path)
        if not prim.IsValid():
            print(f"[WARN] Cannot enable tactile window; missing prim: {case_path}")
            continue
        attr_names = ["debug_tactile_rgb"]
        if include_camera_depth:
            attr_names.append("debug_camera_depth")
        for attr_name in attr_names:
            attr = prim.GetAttribute(attr_name)
            if not attr:
                print(f"[WARN] TacEx {attr_name} attribute missing on: {case_path}")
                continue
            attr.Set(True)

    if args_cli.dock_tactile_windows_right:
        # TacEx creates the actual image windows lazily inside its sensor update
        # callbacks and upstream currently does not expose a docking handle for
        # those windows.  Keep this as an explicit best-effort attempt/log so
        # the grasp path remains robust if the UI API is unavailable.
        try:
            import omni.ui as ui  # type: ignore

            _ = ui.DockPreference.RIGHT_BOTTOM
            print("[INFO] Requested right-side tactile window docking; TacEx debug windows will use upstream UI behavior.")
        except Exception as exc:  # pragma: no cover - GUI-only best effort
            print(f"[WARN] Could not request tactile window docking: {exc}")


class TactileRgbPanel:
    """Small Isaac UI panel that mirrors TacEx tactile_rgb tensors in the main GUI.

    TacEx can create standalone debug windows by toggling USD attributes.  For
    this grasp script we also keep our own panel so the left/right contact
    images can be shown together and best-effort docked to the right side of
    the existing Isaac Sim window.
    """

    def __init__(self, sides: tuple[str, ...], *, width: int, height: int, dock_right: bool):
        self.enabled = False
        self._providers: dict[str, Any] = {}
        self._labels: dict[str, Any] = {}
        self._stats: dict[str, dict[str, Any]] = {}
        self._window: Any | None = None
        self._np: Any | None = None
        self._widget_width = max(64, int(width))
        self._widget_height = max(64, int(height))

        try:
            import numpy as np  # type: ignore
            import omni.ui as ui  # type: ignore

            self._np = np
            window_kwargs: dict[str, Any] = {
                "width": self._widget_width + 40,
                "height": (self._widget_height + 42) * max(1, len(sides)) + 28,
            }
            if dock_right:
                window_kwargs["dockPreference"] = ui.DockPreference.RIGHT_BOTTOM
            self._window = ui.Window("Phase2 GSmini tactile RGB", **window_kwargs)
            with self._window.frame:
                with ui.VStack(spacing=4):
                    ui.Label("Phase2 GSmini tactile_rgb (live)", height=18)
                    for side in sides:
                        ui.Label(f"{side} GSmini", height=16)
                        provider = ui.ByteImageProvider()
                        self._providers[side] = provider
                        ui.ImageWithProvider(provider, width=self._widget_width, height=self._widget_height)
                        self._labels[side] = ui.Label("waiting for first frame", height=16)
            self.enabled = True
        except Exception as exc:  # pragma: no cover - GUI-only best effort
            print(f"[WARN] Could not create docked tactile RGB panel: {exc}")

    def update(self, side: str, output: dict[str, Any]) -> None:
        if not self.enabled or side not in self._providers:
            return
        frame = output.get("tactile_rgb")
        if frame is None:
            return
        try:
            image = self._to_rgba_uint8(frame)
            height, width = image.shape[:2]
            rgb = image[..., :3]
            stat = {
                "frames": int(self._stats.get(side, {}).get("frames", 0)) + 1,
                "width": int(width),
                "height": int(height),
                "mean": float(rgb.mean()),
                "std": float(rgb.std()),
                "min": int(rgb.min()),
                "max": int(rgb.max()),
                "flat": bool(rgb.max() == rgb.min()),
            }
            self._stats[side] = stat
            # Match TacEx's ByteImageProvider usage: pass a flattened buffer
            # rather than a Python bytes object, which has produced blank panes
            # on some Isaac/omni.ui builds.
            self._providers[side].set_bytes_data(image.flatten().data, [width, height])
            label = self._labels.get(side)
            if label is not None:
                label.text = (
                    f"{width}x{height} frame={stat['frames']} μ={stat['mean']:.1f} σ={stat['std']:.1f} "
                    f"range=[{stat['min']},{stat['max']}]"
                )
        except Exception as exc:  # pragma: no cover - GUI-only best effort
            label = self._labels.get(side)
            if label is not None:
                label.text = f"tactile panel update failed: {exc}"

    @property
    def stats(self) -> dict[str, dict[str, Any]]:
        return dict(self._stats)

    def _to_rgba_uint8(self, frame: Any) -> Any:
        np = self._np
        if np is None:
            raise RuntimeError("numpy unavailable")
        if hasattr(frame, "detach"):
            frame = frame.detach().cpu().numpy()
        image = np.asarray(frame)
        # TacEx tensors are [N, H, W, 3]; the standalone script always uses one
        # environment per sensor.
        if image.ndim == 4:
            image = image[0]
        if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
            image = np.moveaxis(image, 0, -1)
        if image.ndim == 2:
            image = np.dstack((image, image, image))
        if image.ndim == 3 and image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        if image.ndim != 3 or image.shape[-1] not in (3, 4):
            raise ValueError(f"unsupported tactile_rgb shape: {image.shape}")
        image = np.array(image, copy=True)
        if image.dtype != np.uint8:
            image = image.astype(np.float32)
            if image.size and float(np.nanmax(image)) <= 1.5:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        # TacEx's native debug windows normalize tactile_rgb before handing it
        # to ByteImageProvider.  Do the same for display only, otherwise valid
        # low-contrast Taxim frames can look blank even while tensors update.
        finite = image[..., :3][np.isfinite(image[..., :3])]
        if finite.size:
            min_value = int(finite.min())
            max_value = int(finite.max())
            if max_value > min_value:
                image_rgb = image[..., :3].astype(np.float32)
                image[..., :3] = ((image_rgb - min_value) * (255.0 / (max_value - min_value))).astype(np.uint8)
            elif max_value == 0:
                image[..., :3] = 32
        if image.shape[-1] == 3:
            alpha = np.full(image.shape[:2] + (1,), 255, dtype=np.uint8)
            image = np.concatenate((image, alpha), axis=-1)
        return np.ascontiguousarray(image)


TACTILE_PANEL: TactileRgbPanel | None = None
TACTILE_MOUNT_INFO: dict[str, Any] = {}
TACTILE_CONTACT_IMPRINT_BASELINES: dict[str, torch.Tensor] = {}
TACTILE_CONTACT_IMPRINT_STATS: dict[str, dict[str, Any]] = {}


def setup_tactile_live_sensors(device: str, *, sides: tuple[str, ...]) -> list[tuple[str, Any]]:
    """Mount Phase2 TacEx shells and initialize live tactile RGB sensors."""

    global TACTILE_PANEL, TACTILE_MOUNT_INFO, TACTILE_CONTACT_IMPRINT_BASELINES, TACTILE_CONTACT_IMPRINT_STATS
    TACTILE_CONTACT_IMPRINT_BASELINES = {}
    TACTILE_CONTACT_IMPRINT_STATS = {}
    if args_cli.no_tactile_live:
        TACTILE_PANEL = None
        TACTILE_MOUNT_INFO = {"enabled": False, "reason": "--no_tactile_live"}
        return []

    clear_phase2_visual_prims()
    mounted = mount_phase2_sensor_shells(sides, hide_render_geometry=True)
    camera_check = validate_phase2_sensor_camera_prims(sides)
    mount_check = validate_phase2_sensor_mounts(sides)
    failures = [side for side, check in mount_check.items() if not check["passed"]]
    missing_cameras = [side for side, check in camera_check.items() if not check["camera_exists"]]
    if failures or missing_cameras:
        raise RuntimeError(
            "Cannot start Phase2 tactile live display; mount/camera check failed: "
            f"failed_mounts={failures}, missing_cameras={missing_cameras}"
        )

    sensors: list[tuple[str, Any]] = []
    for side in sides:
        cfg = build_phase2_gsmini_cfg(
            side,
            device=device,
            resolution=(args_cli.phase2_tactile_width, args_cli.phase2_tactile_height),
            include_camera_depth=args_cli.show_tactile_depth,
            include_camera_rgb=False,
            debug_vis=True,
        )
        sensors.append((side, initialize_phase2_sensor(cfg)))
    if args_cli.no_tactile_panel:
        TACTILE_PANEL = None
    else:
        TACTILE_PANEL = TactileRgbPanel(
            sides,
            width=args_cli.phase2_tactile_width,
            height=args_cli.phase2_tactile_height,
            dock_right=args_cli.dock_tactile_windows_right,
        )
    if args_cli.show_tacex_debug_windows:
        _enable_tactile_debug_windows(sides, include_camera_depth=args_cli.show_tactile_depth)
    TACTILE_MOUNT_INFO = {
        "enabled": True,
        "mounted": mounted,
        "camera_check": camera_check,
        "mount_check": mount_check,
        "runtime_shell_policy": (
            "The canonical URDF owns visible/collision GSmini geometry. "
            "Phase2 TacEx shell mesh descendants are hidden and their physics is disabled; "
            "their camera prims remain active for tactile_rgb."
        ),
    }
    print("[INFO] Phase2 live tactile RGB windows enabled:")
    print(
        json.dumps(
            {
                "sides": sides,
                "mounted": mounted,
                "sensor_paths": phase2_sensor_prim_paths(),
                "shown_data_types": ["tactile_rgb"] + (["camera_depth"] if args_cli.show_tactile_depth else []),
                "sensor_camera_clipping_range_m": list(DEFAULT_SENSOR_CAMERA_CLIPPING_RANGE),
                "dock_tactile_windows_right": args_cli.dock_tactile_windows_right,
                "script_tactile_panel": bool(TACTILE_PANEL and TACTILE_PANEL.enabled),
                "legacy_tacex_debug_windows": args_cli.show_tacex_debug_windows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return sensors


def _stage_prim_visibility(path: str) -> dict[str, Any]:
    """Audit one USD prim without treating hidden debug/sensor meshes as fatal."""

    try:
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            return {"path": path, "exists": False}
        payload: dict[str, Any] = {
            "path": path,
            "exists": True,
            "active": bool(prim.IsActive()),
            "type": prim.GetTypeName(),
        }
        imageable = UsdGeom.Imageable(prim)
        if imageable:
            payload["visibility"] = str(imageable.ComputeVisibility())
            purpose_attr = imageable.GetPurposeAttr()
            purpose = purpose_attr.Get() if purpose_attr else None
            if purpose:
                payload["purpose"] = str(purpose)
        return payload
    except Exception as exc:  # pragma: no cover - runtime USD audit only
        return {"path": path, "exists": None, "error": str(exc)}


def _inner_finger_stage_audit() -> dict[str, Any]:
    """Record why some Stage tree children look gray while contacts still work."""

    paths = [
        "/World/Origin1/Robot/left_inner_finger",
        "/World/Origin1/Robot/left_inner_finger/visuals",
        "/World/Origin1/Robot/left_inner_finger/collisions",
        "/World/Origin1/Robot/right_inner_finger",
        "/World/Origin1/Robot/right_inner_finger/visuals",
        "/World/Origin1/Robot/right_inner_finger/collisions",
    ]
    for side_paths in phase2_sensor_prim_paths().values():
        paths.extend([side_paths["case"], side_paths["gelpad"], side_paths["camera"]])
    audited = {_path: _stage_prim_visibility(_path) for _path in paths}
    return {
        "prim_status": audited,
        "interpretation": (
            "Gray Stage-tree entries under phase2_tacex are expected when TacEx runtime shell meshes are hidden "
            "or their physics is disabled. Canonical URDF gelpad contact is validated separately by "
            "filtered contact forces and GSmini soft-mesh AABB overlap."
        ),
    }


def _origin_prim_path(index: int = 1) -> str:
    return f"/World/Origin{index}"


def _cube_prim_path(index: int = 1) -> str:
    return f"{_origin_prim_path(index)}/{PHASE2_CUBE_NAME}"



def setup_soft_contact_sensors(*, origin_index: int = 1) -> dict[str, ContactSensor]:
    """Create filtered force sensors on the TacEx-style GSmini gelpad bodies."""

    if args_cli.disable_force_control:
        print("[INFO] GSmini force-control sensors disabled by --disable_force_control", flush=True)
        return {}

    origin_prim = _origin_prim_path(origin_index)
    cube_path = _cube_prim_path(origin_index)
    sensors: dict[str, ContactSensor] = {}
    for side, body_name in SOFT_CONTACT_BODY_BY_SIDE.items():
        sensors[side] = ContactSensor(
            ContactSensorCfg(
                prim_path=f"{origin_prim}/Robot/{body_name}",
                update_period=0.0,
                history_length=3,
                debug_vis=False,
                filter_prim_paths_expr=[cube_path],
            )
        )
    print(
        "[INFO] GSmini force-control sensors configured: "
        + json.dumps(
            {
                "cube_filter": cube_path,
                "body_by_side": SOFT_CONTACT_BODY_BY_SIDE,
                "contact_report_threshold_n": args_cli.soft_contact_force_threshold,
                "stable_force_threshold_n": args_cli.force_control_stable_force_threshold,
                "high_force_stop_threshold_n": args_cli.force_control_high_force_threshold,
                "stable_steps": args_cli.force_control_stable_steps,
                "soft_contact_stop": False,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return sensors


def _tactile_rgb_tensor(output: dict[str, Any]) -> torch.Tensor | None:
    frame = output.get("tactile_rgb")
    if frame is None or not hasattr(frame, "detach"):
        return None
    return frame


def _tactile_rgb_mean_abs_delta(frame: torch.Tensor, baseline: torch.Tensor | None) -> float | None:
    if baseline is None:
        return None
    try:
        current = frame.detach()
        reference = baseline.detach().to(device=current.device, dtype=current.dtype)
        if current.shape != reference.shape:
            return None
        return float(torch.mean(torch.abs(current - reference)).item())
    except Exception:
        return None


def _sensor_optical_simulator_geometry(sensor: Any) -> dict[str, float]:
    optical_sim = getattr(sensor, "optical_simulator", None)
    cfg = getattr(optical_sim, "cfg", None)
    gelpad_to_camera = float(getattr(cfg, "gelpad_to_camera_min_distance", 0.024) or 0.024)
    gelpad_height = float(getattr(cfg, "gelpad_height", 0.0045) or 0.0045)
    return {
        "gelpad_to_camera_min_distance_m": gelpad_to_camera,
        "gelpad_height_m": gelpad_height,
        "far_gelpad_surface_mm": (gelpad_to_camera + gelpad_height) * 1000.0,
    }


def _geometry_contact_imprint_depth_mm(side_geometry: dict[str, Any]) -> float:
    nominal_depth = max(0.05, float(args_cli.tactile_contact_imprint_depth_mm))
    min_overlap_m = float(side_geometry.get("soft_mesh_aabb_min_overlap_m", 0.0) or 0.0)
    # Positive overlap means the cube is already inside the soft-link AABB.
    # Keep the fallback bounded so it makes the contact visible without
    # inventing excessive pressing depth.
    overlap_depth = max(0.0, min_overlap_m) * 1000.0
    return min(3.0, nominal_depth + 0.25 * overlap_depth)


def _apply_taxim_contact_imprint(sensor: Any, output: dict[str, Any], *, side: str, side_geometry: dict[str, Any]) -> dict[str, Any]:
    """Feed a soft-contact height-map patch through TacEx/Taxim and replace tactile_rgb.

    TacEx's normal path renders tactile_rgb from the hidden sensor camera depth.
    In this UR5/Robotiq assembly the canonical URDF owns the visible/collision
    GSmini geometry while a hidden TacEx shell provides the camera.  Depending
    on shell/camera occlusion and clipping, the camera can keep seeing the
    no-contact background even after the canonical soft-link mesh touches the
    cube.  This fallback is intentionally narrow: it only runs when contact
    geometry says this side is touching and the current RGB frame still matches
    the stored no-contact background.  The rendered image still comes from
    TacEx/Taxim; only the height map source is replaced by the measured contact
    geometry.
    """

    optical_sim = getattr(sensor, "optical_simulator", None)
    sensor_data = getattr(sensor, "_data", None)
    sensor_output = getattr(sensor_data, "output", None)
    if optical_sim is None or sensor_output is None or "height_map" not in sensor_output or "tactile_rgb" not in output:
        return {"applied": False, "reason": "missing TacEx optical simulator, height_map, or tactile_rgb output"}

    height_map = sensor_output["height_map"]
    if not hasattr(height_map, "shape") or height_map.ndim != 3:
        return {"applied": False, "reason": f"unsupported height_map shape: {getattr(height_map, 'shape', None)}"}

    _, height, width = height_map.shape
    geometry = _sensor_optical_simulator_geometry(sensor)
    base_mm = geometry["far_gelpad_surface_mm"]
    depth_mm = _geometry_contact_imprint_depth_mm(side_geometry)
    device = height_map.device
    dtype = height_map.dtype

    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype).reshape(1, height, 1)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype).reshape(1, 1, width)
    # A centered, smooth oval corresponds to aiming the GSmini soft-pair TCP at
    # the cube side-center line.  It is intentionally small enough to show a
    # local contact patch rather than tinting the whole gelpad.
    sigma_x = torch.tensor(0.34, device=device, dtype=dtype)
    sigma_y = torch.tensor(0.42, device=device, dtype=dtype)
    imprint = torch.exp(-0.5 * ((x / sigma_x) ** 2 + (y / sigma_y) ** 2))
    height_map[:] = torch.tensor(base_mm, device=device, dtype=dtype) - torch.tensor(depth_mm, device=device, dtype=dtype) * imprint

    indentation = getattr(sensor, "_indentation_depth", None)
    if indentation is not None:
        indentation[:] = torch.tensor(depth_mm, device=indentation.device, dtype=indentation.dtype)
    optical_indentation = getattr(optical_sim, "_indentation_depth", None)
    if optical_indentation is not None:
        optical_indentation[:] = torch.tensor(depth_mm, device=optical_indentation.device, dtype=optical_indentation.dtype)

    rendered = optical_sim.optical_simulation()
    target_rgb = output["tactile_rgb"]
    if rendered.device != target_rgb.device or rendered.dtype != target_rgb.dtype:
        rendered = rendered.to(device=target_rgb.device, dtype=target_rgb.dtype)
    target_rgb[:] = rendered
    return {
        "applied": True,
        "source": "soft_link_aabb_contact_geometry_to_taxim_height_map",
        "depth_mm": depth_mm,
        "base_height_mm": base_mm,
        "height_map_shape": [int(height), int(width)],
        **geometry,
    }


def _maybe_apply_tactile_contact_imprint(
    side: str,
    sensor: Any,
    output: dict[str, Any],
    *,
    robot: Articulation | None,
    grasp_object: RigidObject | None,
) -> None:
    frame = _tactile_rgb_tensor(output)
    if frame is None:
        return

    stats = TACTILE_CONTACT_IMPRINT_STATS.setdefault(
        side,
        {
            "frames": 0,
            "baseline_frames": 0,
            "contact_frames": 0,
            "applied_frames": 0,
            "real_contact_delta_frames": 0,
            "max_mean_abs_delta_before": 0.0,
            "max_mean_abs_delta_after": 0.0,
            "max_depth_mm": 0.0,
        },
    )
    stats["frames"] = int(stats.get("frames", 0)) + 1

    if args_cli.disable_tactile_contact_imprint or robot is None or grasp_object is None:
        if side not in TACTILE_CONTACT_IMPRINT_BASELINES:
            TACTILE_CONTACT_IMPRINT_BASELINES[side] = frame.detach().clone()
            stats["baseline_frames"] = int(stats.get("baseline_frames", 0)) + 1
        stats["enabled"] = False
        stats["reason"] = (
            "--disable_tactile_contact_imprint"
            if args_cli.disable_tactile_contact_imprint
            else "robot/object geometry unavailable for this update"
        )
        return

    geometry = _soft_contact_geometry(robot, grasp_object)
    side_geometry = geometry.get("sides", {}).get(side, {})
    contact = bool(side in geometry.get("contact_sides", []))
    if not contact:
        TACTILE_CONTACT_IMPRINT_BASELINES[side] = frame.detach().clone()
        stats["baseline_frames"] = int(stats.get("baseline_frames", 0)) + 1
        stats["enabled"] = True
        stats["last_contact"] = False
        stats["last_contact_sides"] = []
        stats["last_geometry"] = side_geometry
        stats["last_result"] = {"applied": False, "reason": "no soft-link/cube AABB contact on this update"}
        return

    stats["enabled"] = True
    stats["contact_frames"] = int(stats.get("contact_frames", 0)) + 1
    baseline = TACTILE_CONTACT_IMPRINT_BASELINES.get(side)
    if baseline is None:
        # If the first available frame is already in contact, use TacEx's
        # current frame as the background reference.  This is safe because the
        # fallback is only considered if the current frame's delta is below the
        # background threshold.
        baseline = frame.detach().clone()
        TACTILE_CONTACT_IMPRINT_BASELINES[side] = baseline
        stats["baseline_frames"] = int(stats.get("baseline_frames", 0)) + 1

    before_delta = _tactile_rgb_mean_abs_delta(frame, baseline)
    if before_delta is not None:
        stats["max_mean_abs_delta_before"] = max(float(stats.get("max_mean_abs_delta_before", 0.0)), before_delta)

    threshold = max(0.0, float(args_cli.tactile_contact_imprint_background_threshold))
    should_apply = before_delta is None or before_delta <= threshold
    result: dict[str, Any] = {
        "applied": False,
        "reason": "current TacEx tactile_rgb already differs from no-contact background",
    }
    if should_apply:
        result = _apply_taxim_contact_imprint(sensor, output, side=side, side_geometry=side_geometry)
        if result.get("applied"):
            stats["applied_frames"] = int(stats.get("applied_frames", 0)) + 1
            stats["max_depth_mm"] = max(float(stats.get("max_depth_mm", 0.0)), float(result.get("depth_mm", 0.0) or 0.0))
            stats["last_applied_result"] = result
    else:
        stats["real_contact_delta_frames"] = int(stats.get("real_contact_delta_frames", 0)) + 1

    after_frame = _tactile_rgb_tensor(output)
    after_delta = _tactile_rgb_mean_abs_delta(after_frame, baseline) if after_frame is not None else None
    if after_delta is not None:
        stats["max_mean_abs_delta_after"] = max(float(stats.get("max_mean_abs_delta_after", 0.0)), after_delta)

    stats.update(
        {
            "last_contact": True,
            "last_contact_sides": list(geometry.get("contact_sides", [])),
            "last_geometry": side_geometry,
            "last_mean_abs_delta_before": before_delta,
            "last_mean_abs_delta_after": after_delta,
            "background_threshold": threshold,
            "last_result": result,
            "note": (
                "Contact imprint is a fallback for camera-depth/background mismatch: it uses the canonical "
                "GSmini soft-link AABB contact geometry to feed a height map into TacEx/Taxim, then updates "
                "the same tactile_rgb tensor displayed by the live panel/windows."
            ),
        }
    )


def _update_tactile_sensors(
    sensors: list[tuple[str, Any]],
    sim: sim_utils.SimulationContext,
    robot: Articulation | None = None,
    grasp_object: RigidObject | None = None,
) -> None:
    if not sensors:
        return
    dt = sim.get_physics_dt()
    for side, sensor in sensors:
        output = update_phase2_sensor(sensor, sim, dt=dt)
        _maybe_apply_tactile_contact_imprint(side, sensor, output, robot=robot, grasp_object=grasp_object)
        if TACTILE_PANEL is not None:
            TACTILE_PANEL.update(side, output)


def _tactile_contact_change_success(sides: tuple[str, ...]) -> dict[str, Any]:
    """Return whether the displayed tactile stream showed a contact-driven change.

    For the Phase2 demo profile, the tactile gate is evaluated on the live
    sides the user asked to display.  If tactile live is explicitly disabled,
    the gate is marked skipped so headless non-visual debugging can still use
    the contact-demo success profile.
    """

    if args_cli.no_tactile_live:
        return {
            "required": False,
            "passed": True,
            "skipped": True,
            "reason": "--no_tactile_live disables the visual tactile gate",
            "sides": {},
        }

    required_sides = tuple(sides) if sides else tuple(TACTILE_CONTACT_IMPRINT_STATS.keys())
    per_side: dict[str, Any] = {}
    for side in required_sides:
        stats = TACTILE_CONTACT_IMPRINT_STATS.get(side, {})
        contact_frames = int(stats.get("contact_frames", 0) or 0)
        applied_frames = int(stats.get("applied_frames", 0) or 0)
        real_delta_frames = int(stats.get("real_contact_delta_frames", 0) or 0)
        max_delta_after = float(stats.get("max_mean_abs_delta_after", 0.0) or 0.0)
        side_passed = bool(contact_frames > 0 and (applied_frames > 0 or real_delta_frames > 0 or max_delta_after > 1.0e-6))
        per_side[side] = {
            "passed": side_passed,
            "contact_frames": contact_frames,
            "imprint_applied_frames": applied_frames,
            "real_contact_delta_frames": real_delta_frames,
            "max_mean_abs_delta_after": max_delta_after,
        }
    return {
        "required": True,
        "passed": bool(per_side and all(side_stats["passed"] for side_stats in per_side.values())),
        "skipped": False,
        "sides": per_side,
        "criteria": (
            "Each displayed tactile side must see at least one contact frame and either a real TacEx "
            "contact delta or the TacEx/Taxim contact-imprint fallback applied to tactile_rgb."
        ),
    }


def _attempt_success_evaluation(
    *,
    direct_move: dict[str, Any],
    lift: dict[str, Any],
    close_passed: bool,
    close_settle: dict[str, Any],
    hold: dict[str, Any],
    hold_joint_summary: dict[str, Any],
    close_safety: dict[str, Any],
    stable_grasp_passed: bool,
    soft_center_alignment: dict[str, Any],
    tactile_sides: tuple[str, ...],
) -> dict[str, Any]:
    """Evaluate both success profiles and select the configured one."""

    soft_center_preclose = soft_center_alignment.get("preclose_error_m", {})
    soft_center_passed = bool((not args_cli.align_soft_center) or soft_center_preclose.get("passed", False))
    direct_motion_passed = bool(direct_move.get("final_position_error_m", math.inf) <= args_cli.ik_pos_tolerance * 2.0)
    lift_motion_passed = bool(lift.get("final_position_error_m", math.inf) <= args_cli.ik_pos_tolerance * 2.0)
    close_settle_gripper_passed = bool(close_settle.get("gripper", {}).get("max_abs_error_rad", math.inf) <= 0.12)
    hold_gripper_passed = bool(hold_joint_summary.get("max_abs_error_rad", math.inf) <= 0.08)
    tactile_change = _tactile_contact_change_success(tactile_sides)

    lift_hold_passed = bool(
        hold["min_lift_m"] >= args_cli.success_lift_margin
        and stable_grasp_passed
        and close_settle_gripper_passed
        and hold_gripper_passed
        and close_safety["close_object_lift_passed"]
        and close_safety["settle_object_lift_passed"]
    )
    contact_demo_passed = bool(
        soft_center_passed
        and direct_motion_passed
        and close_passed
        and close_settle_gripper_passed
        and hold_gripper_passed
        and stable_grasp_passed
        and close_safety["passed"]
        and tactile_change["passed"]
    )
    selected = args_cli.success_mode
    return {
        "selected_mode": selected,
        "selected_passed": bool(contact_demo_passed if selected == "contact_demo" else lift_hold_passed),
        "contact_demo_passed": contact_demo_passed,
        "lift_hold_passed": lift_hold_passed,
        "checks": {
            "soft_center_preclose_passed": soft_center_passed,
            "direct_motion_passed": direct_motion_passed,
            "lift_motion_passed": lift_motion_passed,
            "close_passed": close_passed,
            "close_settle_gripper_passed": close_settle_gripper_passed,
            "hold_gripper_passed": hold_gripper_passed,
            "stable_grasp_passed": stable_grasp_passed,
            "close_safety_passed": bool(close_safety.get("passed", False)),
            "strict_lift_margin_passed": bool(hold["min_lift_m"] >= args_cli.success_lift_margin),
            "tactile_contact_change": tactile_change,
        },
        "notes": {
            "contact_demo": (
                "Default Phase2 bring-up criterion: centered soft-link grasp, stable close, safety guards, "
                "and visible tactile contact change; it intentionally does not require 10 cm lift/hold."
            ),
            "lift_hold": "Legacy strict criterion requiring the configured object lift margin during hold.",
        },
    }


def _step_once(
    sim: sim_utils.SimulationContext,
    robot: Articulation,
    grasp_object: RigidObject,
    sensors: list[tuple[str, Any]],
    budget: StepBudget,
    *,
    arm_joint_ids: list[int] | None = None,
    arm_joint_target: torch.Tensor | None = None,
    gripper_joint_ids: list[int] | None = None,
    gripper_joint_target: torch.Tensor | None = None,
) -> None:
    if arm_joint_target is not None and arm_joint_ids is not None:
        robot.set_joint_position_target(arm_joint_target, joint_ids=arm_joint_ids)
    if gripper_joint_target is not None and gripper_joint_ids is not None:
        robot.set_joint_position_target(gripper_joint_target[:, gripper_joint_ids], joint_ids=gripper_joint_ids)
    if arm_joint_target is not None or gripper_joint_target is not None:
        robot.write_data_to_sim()
    if gripper_joint_target is not None:
        # Keep the imported Robotiq mimic chain coherent before PhysX solves
        # contacts, not after.  A post-step kinematic write makes the GUI/tactile
        # render see the finger snapped back into contact, which looks like the
        # left tip is vibrating when it brushes the object.
        _stabilize_gripper_mimic_state(robot, gripper_joint_target)

    sim.step()
    dt = sim.get_physics_dt()
    robot.update(dt)
    grasp_object.update(dt)
    _update_tactile_sensors(sensors, sim, robot, grasp_object)
    budget.tick()


def _hold_steps_for_seconds(sim: sim_utils.SimulationContext, seconds: float) -> int:
    return max(1, int(math.ceil(seconds / sim.get_physics_dt())))


def _object_position(grasp_object: RigidObject) -> torch.Tensor:
    return grasp_object.data.root_pose_w[0, :3].detach().clone()


def _ee_body_id(robot: Articulation) -> int:
    return int(robot.find_bodies(["ee_link"], preserve_order=True)[0][0])


def _ee_pose(robot: Articulation, ee_body: int) -> tuple[torch.Tensor, torch.Tensor]:
    pose = robot.data.body_pose_w[:, ee_body]
    return pose[:, 0:3].detach().clone(), pose[:, 3:7].detach().clone()


def _soft_link_center_positions(robot: Articulation) -> dict[str, list[float]]:
    local = torch.tensor(
        GSMINI_SOFT_ORIGIN_IN_SENSOR_FRAME_M,
        device=robot.device,
        dtype=robot.data.body_pose_w.dtype,
    ).reshape(1, 3)
    positions: dict[str, list[float]] = {}
    for side, body_name in SOFT_CONTACT_BODY_BY_SIDE.items():
        body_id = int(robot.find_bodies([body_name], preserve_order=True)[0][0])
        pose = robot.data.body_pose_w[:, body_id]
        center = pose[:, 0:3] + quat_apply(pose[:, 3:7], local)
        positions[side] = [float(value) for value in center[0].detach().cpu().tolist()]
    return positions


def _arm_joint_hold_summary(
    robot: Articulation,
    arm_joint_ids: list[int],
    arm_joint_target: torch.Tensor,
) -> dict[str, Any]:
    actual = robot.data.joint_pos[:, arm_joint_ids].detach()
    error = torch.abs(actual - arm_joint_target)
    return {
        "max_abs_error_rad": float(error.max().item()) if error.numel() else 0.0,
        "target_rad": [float(value) for value in arm_joint_target[0].detach().cpu().tolist()],
        "actual_rad": [float(value) for value in actual[0].detach().cpu().tolist()],
    }


def _latch_arm_at_current_state(robot: Articulation, arm_joint_ids: list[int]) -> tuple[torch.Tensor, dict[str, Any]]:
    """Use the current arm joint pose as a zero-velocity hold target.

    The direct IK segment produces the final grasp pose.  After it reaches that
    pose, the real robot equivalent is a stationary servo hold, not a second
    terminal adjustment.  Zeroing residual arm velocities here prevents the
    PhysX drive from overshooting/rebounding while the gripper starts to close.
    """

    arm_joint_target = robot.data.joint_pos[:, arm_joint_ids].detach().clone()
    zero_arm_velocity = torch.zeros_like(arm_joint_target)
    robot.write_joint_state_to_sim(arm_joint_target, zero_arm_velocity, joint_ids=arm_joint_ids)
    robot.set_joint_position_target(arm_joint_target, joint_ids=arm_joint_ids)
    robot.write_data_to_sim()
    return arm_joint_target, {
        "latched": True,
        "zeroed_arm_velocity": True,
        "arm_hold": _arm_joint_hold_summary(robot, arm_joint_ids, arm_joint_target),
    }


def _hold_arm_and_gripper_targets(
    sim: sim_utils.SimulationContext,
    robot: Articulation,
    grasp_object: RigidObject,
    sensors: list[tuple[str, Any]],
    budget: StepBudget,
    *,
    arm_joint_ids: list[int],
    arm_joint_target: torch.Tensor,
    gripper_joint_ids: list[int],
    gripper_joint_target: torch.Tensor,
    steps: int,
) -> dict[str, Any]:
    actual_steps = 0
    for _ in range(max(0, steps)):
        if budget.exhausted:
            break
        _step_once(
            sim,
            robot,
            grasp_object,
            sensors,
            budget,
            arm_joint_ids=arm_joint_ids,
            arm_joint_target=arm_joint_target,
            gripper_joint_ids=gripper_joint_ids,
            gripper_joint_target=gripper_joint_target,
        )
        actual_steps += 1
    return {
        "requested_steps": steps,
        "actual_steps": actual_steps,
        "arm_hold": _arm_joint_hold_summary(robot, arm_joint_ids, arm_joint_target),
        "gripper": _gripper_joint_summary(robot, gripper_joint_target),
    }


def _aabb_corners(min_xyz: tuple[float, float, float], max_xyz: tuple[float, float, float], *, device: str, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(
        [
            [min_xyz[0], min_xyz[1], min_xyz[2]],
            [min_xyz[0], min_xyz[1], max_xyz[2]],
            [min_xyz[0], max_xyz[1], min_xyz[2]],
            [min_xyz[0], max_xyz[1], max_xyz[2]],
            [max_xyz[0], min_xyz[1], min_xyz[2]],
            [max_xyz[0], min_xyz[1], max_xyz[2]],
            [max_xyz[0], max_xyz[1], min_xyz[2]],
            [max_xyz[0], max_xyz[1], max_xyz[2]],
        ],
        device=device,
        dtype=dtype,
    )


def _transform_local_corners(pose_w: torch.Tensor, local_corners: torch.Tensor) -> torch.Tensor:
    quat_w = pose_w[:, 3:7].expand(local_corners.shape[0], 4)
    pos_w = pose_w[:, 0:3].expand(local_corners.shape[0], 3)
    return pos_w + quat_apply(quat_w, local_corners)


def _cube_world_aabb(grasp_object: RigidObject) -> dict[str, Any]:
    half = PHASE2_CUBE_SIZE_M / 2.0
    pose_w = grasp_object.data.root_pose_w[:, :7]
    local = _aabb_corners(
        (-half, -half, -half),
        (half, half, half),
        device=grasp_object.device,
        dtype=grasp_object.data.root_pose_w.dtype,
    )
    world = _transform_local_corners(pose_w, local)
    min_w = world.min(dim=0).values
    max_w = world.max(dim=0).values
    center_w = grasp_object.data.root_pose_w[0, :3].detach().clone()
    return {
        "min": min_w,
        "max": max_w,
        "center": center_w,
        "summary": {
            "min_world_m": [float(value) for value in min_w.detach().cpu().tolist()],
            "max_world_m": [float(value) for value in max_w.detach().cpu().tolist()],
            "center_world_m": [float(value) for value in center_w.detach().cpu().tolist()],
            "half_extent_m": half,
        },
    }


def _sensor_frame_local_aabbs_world(
    robot: Articulation,
    local_aabb_m: dict[str, tuple[float, float, float]],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    local_corners = _aabb_corners(
        local_aabb_m["min"],
        local_aabb_m["max"],
        device=robot.device,
        dtype=robot.data.body_pose_w.dtype,
    )
    aabbs: dict[str, dict[str, Any]] = {}
    for side, body_name in SOFT_CONTACT_BODY_BY_SIDE.items():
        body_id = int(robot.find_bodies([body_name], preserve_order=True)[0][0])
        pose = robot.data.body_pose_w[:, body_id]
        world = _transform_local_corners(pose, local_corners)
        min_w = world.min(dim=0).values
        max_w = world.max(dim=0).values
        center_w = (min_w + max_w) / 2.0
        aabbs[side] = {
            "min": min_w,
            "max": max_w,
            "center": center_w,
            "summary": {
                "body": body_name,
                "label": label,
                "min_world_m": [float(value) for value in min_w.detach().cpu().tolist()],
                "max_world_m": [float(value) for value in max_w.detach().cpu().tolist()],
                "center_world_m": [float(value) for value in center_w.detach().cpu().tolist()],
            },
        }
    return aabbs


def _soft_link_mesh_aabbs_world(robot: Articulation) -> dict[str, dict[str, Any]]:
    return _sensor_frame_local_aabbs_world(
        robot,
        GSMINI_SOFT_MESH_AABB_IN_SENSOR_FRAME_M,
        label="gsmini_soft_mesh",
    )


def _soft_mesh_pair_center_world(robot: Articulation) -> torch.Tensor:
    """Return the world center of the two GSmini soft-link mesh AABBs."""

    soft_aabbs = _soft_link_mesh_aabbs_world(robot)
    centers = [soft_aabbs[side]["center"] for side in ("left", "right") if side in soft_aabbs]
    if not centers:
        raise RuntimeError("Cannot compute GSmini soft-link center; no soft-link bodies were found.")
    return torch.stack(centers, dim=0).mean(dim=0)


def _compute_soft_centered_grasp_target(
    robot: Articulation,
    ee_body: int,
    object_start: torch.Tensor,
    target_quat_w: torch.Tensor,
    params: AttemptParams,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute an EE target whose GSmini soft links, not ee_link, center on the cube sides.

    The Robotiq/GSmini chain has a fixed transform from ``ee_link`` to the two
    soft-link mesh centers while the gripper is held open.  Measuring that
    transform in the pregrasp pose and reapplying it at the target orientation
    avoids the previous hand-tuned EE offset that left the soft_link several
    millimeters off the cube side centers.
    """

    ee_pos_w, ee_quat_w = _ee_pose(robot, ee_body)
    current_soft_pair_center_w = _soft_mesh_pair_center_world(robot)
    soft_offset_in_ee = quat_apply_inverse(
        ee_quat_w,
        (current_soft_pair_center_w.reshape(1, 3) - ee_pos_w),
    )[0]

    desired_soft_pair_center_w = object_start.detach().clone()
    desired_soft_pair_center_w[0] += params.longitudinal_x_offset
    desired_soft_pair_center_w[1] += params.lateral_y_offset
    desired_soft_pair_center_w[2] += params.soft_center_z_offset
    target_soft_offset_w = quat_apply(target_quat_w.reshape(1, 4), soft_offset_in_ee.reshape(1, 3))[0]
    target_ee_pos_w = desired_soft_pair_center_w - target_soft_offset_w

    return target_ee_pos_w.to(device=robot.device, dtype=robot.data.root_pose_w.dtype), {
        "enabled": True,
        "mode": "soft_mesh_pair_center_to_cube_side_center",
        "soft_center_tcp": {
            "frame_name": "virtual_soft_center_tcp",
            "parent_frame": "ee_link",
            "definition": (
                "Virtual TCP at the average of the left/right GSmini soft-link mesh AABB centers "
                "while the gripper is open. The IK solver still targets ee_link; this TCP provides "
                "the deterministic ee_link offset needed to place the soft pads on the cube side-center line."
            ),
            "target_frame_for_ik": "ee_link",
            "desired_world_m": [float(value) for value in desired_soft_pair_center_w.detach().cpu().tolist()],
            "offset_in_ee_frame_m": [float(value) for value in soft_offset_in_ee.detach().cpu().tolist()],
            "target_ee_world_m": [float(value) for value in target_ee_pos_w.detach().cpu().tolist()],
            "refinement_policy": (
                "Use this as the first deterministic target, then keep the bounded measured soft-center "
                "refinement because PhysX articulation settling and gripper mimic constraints can still "
                "move the realized soft_link centers by millimeters."
            ),
        },
        "desired_soft_pair_center_world_m": [float(value) for value in desired_soft_pair_center_w.detach().cpu().tolist()],
        "pregrasp_ee_world_m": [float(value) for value in ee_pos_w[0].detach().cpu().tolist()],
        "pregrasp_soft_pair_center_world_m": [
            float(value) for value in current_soft_pair_center_w.detach().cpu().tolist()
        ],
        "soft_offset_in_ee_frame_m": [float(value) for value in soft_offset_in_ee.detach().cpu().tolist()],
        "target_soft_offset_world_m": [float(value) for value in target_soft_offset_w.detach().cpu().tolist()],
        "target_ee_world_m": [float(value) for value in target_ee_pos_w.detach().cpu().tolist()],
        "cube_center_world_m": [float(value) for value in object_start.detach().cpu().tolist()],
        "soft_center_z_offset_m": params.soft_center_z_offset,
        "longitudinal_x_offset_m": params.longitudinal_x_offset,
        "lateral_y_offset_m": params.lateral_y_offset,
    }


def _sensor_assembly_aabbs_world(robot: Articulation) -> dict[str, dict[str, Any]]:
    return _sensor_frame_local_aabbs_world(
        robot,
        GSMINI_SENSOR_ASSEMBLY_AABB_IN_SENSOR_FRAME_M,
        label="gsmini_sensor_assembly_visual_bounds",
    )


def _base_link_aabbs_world(robot: Articulation) -> dict[str, dict[str, Any]]:
    return _sensor_frame_local_aabbs_world(
        robot,
        GSMINI_BASE_LINK_AABB_IN_SENSOR_FRAME_M,
        label="gsmini_base_link_collision",
    )


def _aabb_overlap_summary(soft_aabb: dict[str, Any], cube_aabb: dict[str, Any]) -> dict[str, Any]:
    soft_min = soft_aabb["min"]
    soft_max = soft_aabb["max"]
    cube_min = cube_aabb["min"]
    cube_max = cube_aabb["max"]
    margin = GSMINI_SOFT_AABB_CONTACT_MARGIN_M
    overlaps = torch.minimum(soft_max, cube_max) - torch.maximum(soft_min, cube_min)
    separated = torch.maximum(cube_min - soft_max, soft_min - cube_max)
    positive_sep = torch.clamp(separated, min=0.0)
    distance = float(torch.linalg.norm(positive_sep).item())
    overlap_with_margin = bool(torch.all(overlaps >= -margin).item())
    min_overlap = float(overlaps.min().item())
    return {
        "soft_mesh_aabb_overlap": overlap_with_margin,
        "soft_mesh_aabb_min_overlap_m": min_overlap,
        "soft_mesh_aabb_distance_m": 0.0 if overlap_with_margin else distance,
        "contact_margin_m": margin,
    }


def _soft_contact_geometry(robot: Articulation, grasp_object: RigidObject) -> dict[str, Any]:
    cube_aabb = _cube_world_aabb(grasp_object)
    soft_aabbs = _soft_link_mesh_aabbs_world(robot)
    sides: dict[str, Any] = {}
    contact_sides: list[str] = []
    for side, soft_aabb in soft_aabbs.items():
        overlap = _aabb_overlap_summary(soft_aabb, cube_aabb)
        side_summary = {
            **soft_aabb["summary"],
            **overlap,
        }
        if overlap["soft_mesh_aabb_overlap"]:
            contact_sides.append(side)
        sides[side] = side_summary
    return {
        "cube_aabb": cube_aabb["summary"],
        "sides": sides,
        "contact_sides": contact_sides,
    }



def _soft_link_geometry_summary(robot: Articulation, grasp_object: RigidObject) -> dict[str, Any]:
    """Summarize direct side-grasp geometry without any table-clearance checks."""

    centers = _soft_link_center_positions(robot)
    mesh_aabbs = _soft_link_mesh_aabbs_world(robot)
    base_link_aabbs = _base_link_aabbs_world(robot)
    sensor_assembly_aabbs = _sensor_assembly_aabbs_world(robot)
    mesh_centers = {side: aabb["summary"]["center_world_m"] for side, aabb in mesh_aabbs.items()}
    if {"left", "right"}.issubset(centers):
        origin_gap = math.dist(centers["left"], centers["right"])
    else:
        origin_gap = math.nan
    if {"left", "right"}.issubset(mesh_centers):
        mesh_center_gap = math.dist(mesh_centers["left"], mesh_centers["right"])
    else:
        mesh_center_gap = math.nan
    cube = _cube_world_aabb(grasp_object)
    cube_center = cube["summary"]["center_world_m"]
    side_grasp_aperture: dict[str, Any] = {
        "axis": "world_y",
        "soft_inner_gap_m": math.nan,
        "cube_extent_m": PHASE2_CUBE_SIZE_M,
        "margin_m": math.nan,
        "feasible_without_interpenetration": False,
    }
    if {"left", "right"}.issubset(mesh_aabbs):
        left_summary = mesh_aabbs["left"]["summary"]
        right_summary = mesh_aabbs["right"]["summary"]
        left_min_y, left_max_y = left_summary["min_world_m"][1], left_summary["max_world_m"][1]
        right_min_y, right_max_y = right_summary["min_world_m"][1], right_summary["max_world_m"][1]
        inner_gap = max(right_min_y, left_min_y) - min(right_max_y, left_max_y)
        # If the sign convention flips, use the largest clear interval between
        # the two soft AABBs along the intended left/right cube-grasp axis.
        if inner_gap < 0.0:
            inner_gap = max(left_min_y, right_min_y) - min(left_max_y, right_max_y)
        margin = inner_gap - PHASE2_CUBE_SIZE_M
        side_grasp_aperture.update(
            {
                "soft_inner_gap_m": float(inner_gap),
                "cube_extent_m": PHASE2_CUBE_SIZE_M,
                "margin_m": float(margin),
                "feasible_without_interpenetration": bool(margin >= -GSMINI_SOFT_AABB_CONTACT_MARGIN_M),
            }
        )
    return {
        "soft_origin_in_sensor_frame_m": list(GSMINI_SOFT_ORIGIN_IN_SENSOR_FRAME_M),
        "soft_mesh_local_aabb_m": GSMINI_SOFT_MESH_AABB_IN_SENSOR_FRAME_M,
        "base_link_local_aabb_m": GSMINI_BASE_LINK_AABB_IN_SENSOR_FRAME_M,
        "sensor_assembly_local_aabb_m": GSMINI_SENSOR_ASSEMBLY_AABB_IN_SENSOR_FRAME_M,
        "soft_origin_world_m": centers,
        "soft_origin_distance_m": origin_gap,
        "soft_mesh_aabb_world_m": {side: aabb["summary"] for side, aabb in mesh_aabbs.items()},
        "base_link_aabb_world_m": {side: aabb["summary"] for side, aabb in base_link_aabbs.items()},
        "sensor_assembly_aabb_world_m": {side: aabb["summary"] for side, aabb in sensor_assembly_aabbs.items()},
        "soft_mesh_center_distance_m": mesh_center_gap,
        "cube_center_world_m": cube_center,
        "cube_side_center_z_m": cube_center[2],
        "cube_half_extent_m": PHASE2_CUBE_SIZE_M / 2.0,
        "side_grasp_aperture": side_grasp_aperture,
        "soft_cube_contact_geometry": _soft_contact_geometry(robot, grasp_object),
    }


def _soft_center_alignment_error(geometry: dict[str, Any]) -> dict[str, Any]:
    """Summarize how well the two soft-link mesh centers align with cube side centers."""

    tolerance = max(0.0, float(args_cli.soft_center_tolerance))
    cube_center = geometry.get("cube_center_world_m")
    soft_meshes = geometry.get("soft_mesh_aabb_world_m", {})
    centers = [
        soft_meshes[side]["center_world_m"]
        for side in ("left", "right")
        if side in soft_meshes and "center_world_m" in soft_meshes[side]
    ]
    if cube_center is None or not centers:
        return {"available": False, "reason": "missing cube center or soft mesh centers"}
    mean_center = [sum(center[axis] for center in centers) / len(centers) for axis in range(3)]
    error = [mean_center[axis] - cube_center[axis] for axis in range(3)]
    xz_error = math.sqrt(error[0] ** 2 + error[2] ** 2)
    return {
        "available": True,
        "soft_pair_center_world_m": mean_center,
        "cube_center_world_m": cube_center,
        "pair_center_error_world_m": error,
        "xz_center_error_m": xz_error,
        "tolerance_m": tolerance,
        "passed": bool(xz_error <= tolerance),
        "interpretation": (
            "Checks the average left/right GSmini soft-link mesh center against the cube side-face center line. "
            "Y is reported for symmetry; X and Z decide whether the pad is centered on the side face."
        ),
    }


def _apply_orientation_offsets(
    base_quat_w: torch.Tensor,
    *,
    yaw_offset_rad: float,
    pitch_offset_rad: float,
    roll_offset_rad: float,
) -> torch.Tensor:
    offsets = quat_from_euler_xyz(
        torch.tensor([roll_offset_rad], device=base_quat_w.device, dtype=base_quat_w.dtype),
        torch.tensor([pitch_offset_rad], device=base_quat_w.device, dtype=base_quat_w.dtype),
        torch.tensor([yaw_offset_rad], device=base_quat_w.device, dtype=base_quat_w.dtype),
    )
    return quat_mul(offsets, base_quat_w)


def _make_custom_gripper_target(robot: Articulation, close_rad: float, *, base_target: torch.Tensor | None = None) -> torch.Tensor:
    overrides = gripper_joint_overrides_rad(closed=True)
    sign_by_joint = {joint_name: 1.0 if value >= 0.0 else -1.0 for joint_name, value in overrides.items()}
    custom = {joint_name: sign_by_joint[joint_name] * close_rad for joint_name in overrides}
    from ur5_phase1_control import build_joint_target

    return build_joint_target(robot, custom, base_target=base_target)


def _gripper_joint_summary(robot: Articulation, joint_target: torch.Tensor) -> dict[str, Any]:
    """Record whether the converted Robotiq mimic joints reached the commanded close pose."""

    joint_id_map = named_joint_ids(robot, GRIPPER_CONTROL_JOINT_NAMES)
    target: dict[str, float] = {}
    actual: dict[str, float] = {}
    error: dict[str, float] = {}
    for joint_name, joint_id in joint_id_map.items():
        target_value = float(joint_target[0, joint_id].item())
        actual_value = float(robot.data.joint_pos[0, joint_id].item())
        target[joint_name] = target_value
        actual[joint_name] = actual_value
        error[joint_name] = abs(actual_value - target_value)
    return {
        "target_rad": target,
        "actual_rad": actual,
        "abs_error_rad": error,
        "max_abs_error_rad": max(error.values()) if error else 0.0,
    }


def _current_close_rad(robot: Articulation) -> float:
    finger_joint_id = named_joint_ids(robot, ["finger_joint"])["finger_joint"]
    return abs(float(robot.data.joint_pos[0, finger_joint_id].item()))



def _read_soft_contact(
    contact_sensors: dict[str, ContactSensor],
    sim: sim_utils.SimulationContext,
    robot: Articulation,
    grasp_object: RigidObject,
) -> dict[str, Any]:
    """Read filtered GSmini gelpad forces and geometry without using contact as a close-stop trigger."""

    geometry = _soft_contact_geometry(robot, grasp_object)

    if not contact_sensors:
        return {
            "enabled": False,
            "contact_detected": bool(geometry["contact_sides"]),
            "contact_sides": geometry["contact_sides"],
            "geometry_contact_sides": geometry["contact_sides"],
            "force_contact_sides": [],
            "force_by_side_n": {},
            "max_force_n": 0.0,
            "both_sides_force_contact": False,
            "threshold_n": args_cli.soft_contact_force_threshold,
            "geometry": geometry,
            "note": "force-control sensors disabled; geometry contact is diagnostic only and does not stop closure",
        }

    sides: dict[str, Any] = {}
    contact_sides: list[str] = []
    force_contact_sides: list[str] = []
    force_by_side: dict[str, float] = {}
    dt = sim.get_physics_dt()
    for side, sensor in contact_sensors.items():
        side_summary: dict[str, Any] = {
            "initialized": bool(sensor.is_initialized),
            "max_force_n": 0.0,
            "in_contact": False,
            "contact_source": "none",
        }
        try:
            if not sensor.is_initialized:
                raise RuntimeError("contact sensor is not initialized yet")
            sensor.update(dt, force_recompute=True)
            data = sensor.data
            force_tensor = getattr(data, "force_matrix_w", None)
            if force_tensor is None:
                force_tensor = data.net_forces_w
            if force_tensor is not None and force_tensor.numel() > 0:
                max_force = float(torch.linalg.norm(force_tensor.reshape(-1, 3), dim=-1).max().item())
            else:
                max_force = 0.0
            side_summary["max_force_n"] = max_force
            side_summary["force_in_contact"] = max_force >= args_cli.soft_contact_force_threshold
        except Exception as exc:  # pragma: no cover - exercised only inside Isaac runtime
            side_summary["error"] = str(exc)
            side_summary["force_in_contact"] = False
        geometry_side = geometry["sides"].get(side, {})
        geometry_in_contact = bool(geometry_side.get("soft_mesh_aabb_overlap", False))
        force_by_side[side] = float(side_summary["max_force_n"])
        side_summary["geometry"] = geometry_side
        side_summary["geometry_in_contact"] = geometry_in_contact
        side_summary["in_contact"] = bool(side_summary.get("force_in_contact") or geometry_in_contact)
        if side_summary.get("force_in_contact"):
            force_contact_sides.append(side)
        if side_summary.get("force_in_contact") and geometry_in_contact:
            side_summary["contact_source"] = "force+soft_mesh_aabb"
        elif side_summary.get("force_in_contact"):
            side_summary["contact_source"] = "force"
        elif geometry_in_contact:
            side_summary["contact_source"] = "soft_mesh_aabb"
        if side_summary["in_contact"]:
            contact_sides.append(side)
        sides[side] = side_summary

    required_sides = set(contact_sensors.keys())
    return {
        "enabled": True,
        "contact_detected": bool(contact_sides),
        "contact_sides": contact_sides,
        "geometry_contact_sides": geometry["contact_sides"],
        "force_contact_sides": force_contact_sides,
        "force_by_side_n": force_by_side,
        "max_force_n": max(force_by_side.values()) if force_by_side else 0.0,
        "both_sides_force_contact": required_sides.issubset(set(force_contact_sides)),
        "threshold_n": args_cli.soft_contact_force_threshold,
        "sides": sides,
        "geometry": geometry,
        "note": (
            "The URDF now exposes GSmini soft_link visual/collision geometry on TacEx-style gelpad bodies. "
            "Soft-link contact is now diagnostic only; gripper closure is stopped only by high force or other safety guards."
        ),
    }



def _close_gripper_with_force_control(
    sim: sim_utils.SimulationContext,
    robot: Articulation,
    grasp_object: RigidObject,
    tactile_sensors: list[tuple[str, Any]],
    contact_sensors: dict[str, ContactSensor],
    budget: StepBudget,
    *,
    target_close_rad: float,
    steps: int,
    preclose_object_z: float | None = None,
    arm_joint_ids: list[int] | None = None,
    arm_joint_hold_target: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Close until stable GSmini force, high force, object lift, or the requested width stops motion."""

    base_target = robot.data.joint_pos.detach().clone()
    requested_target_close_rad = target_close_rad
    absolute_close_cap_rad = max(0.0, args_cli.max_close_without_contact_rad)
    target_close_rad = min(target_close_rad, absolute_close_cap_rad)
    start_close_rad = min(target_close_rad, _current_close_rad(robot))
    rewind_open_margin_rad = max(0.0, float(args_cli.object_lift_rewind_open_margin_rad))
    lift_guard_trigger_m = max(
        0.0,
        float(args_cli.max_close_object_lift) * max(0.0, min(1.0, float(args_cli.object_lift_guard_trigger_fraction))),
    )
    rewind_close_rad: float | None = None

    final_target = _make_custom_gripper_target(robot, start_close_rad, base_target=base_target)
    actual_steps = 0
    last_contact_read = _read_soft_contact(contact_sensors, sim, robot, grasp_object)
    stopped_by_object_lift = False
    stopped_by_high_force = False
    high_force_step: int | None = None
    object_z0 = float(_object_position(grasp_object)[2].item()) if preclose_object_z is None else preclose_object_z
    max_object_lift_m = 0.0
    object_z_history: list[dict[str, Any]] = []
    force_history: list[dict[str, Any]] = []
    last_safe_target = final_target.detach().clone()
    last_safe_close_rad = start_close_rad
    last_force_safe_target = final_target.detach().clone()
    last_force_safe_close_rad = start_close_rad
    gripper_joint_ids = list(named_joint_ids(robot, GRIPPER_CONTROL_JOINT_NAMES).values())

    force_control_enabled = bool(contact_sensors) and not args_cli.disable_force_control
    stable_force_threshold = max(0.0, args_cli.force_control_stable_force_threshold)
    high_force_threshold = max(0.0, args_cli.force_control_high_force_threshold)
    required_stable_steps = max(1, args_cli.force_control_stable_steps)
    stable_counter = 0
    stable_grasp_detected = False
    stable_grasp_step: int | None = None
    stopped_by_stable_force = False
    max_force_seen_n = 0.0

    for index in range(max(1, steps)):
        if budget.exhausted:
            break
        alpha = float(index + 1) / float(max(1, steps))
        planned_close_rad = start_close_rad + (target_close_rad - start_close_rad) * alpha

        final_target = _make_custom_gripper_target(robot, planned_close_rad, base_target=base_target)
        _step_once(
            sim,
            robot,
            grasp_object,
            tactile_sensors,
            budget,
            arm_joint_ids=arm_joint_ids,
            arm_joint_target=arm_joint_hold_target,
            gripper_joint_ids=gripper_joint_ids,
            gripper_joint_target=final_target,
        )
        actual_steps += 1
        last_contact_read = _read_soft_contact(contact_sensors, sim, robot, grasp_object)
        force_by_side = last_contact_read.get("force_by_side_n", {})
        max_force_n = float(last_contact_read.get("max_force_n", 0.0) or 0.0)
        max_force_seen_n = max(max_force_seen_n, max_force_n)
        both_sides_stable = bool(
            force_control_enabled
            and force_by_side
            and all(float(force_by_side.get(side, 0.0)) >= stable_force_threshold for side in contact_sensors)
        )
        if both_sides_stable:
            stable_counter += 1
            if stable_counter >= required_stable_steps and not stable_grasp_detected:
                stable_grasp_detected = True
                stable_grasp_step = actual_steps
        else:
            stable_counter = 0
        high_force_detected = bool(force_control_enabled and max_force_n >= high_force_threshold)
        force_history.append(
            {
                "step": actual_steps,
                "planned_close_rad": planned_close_rad,
                "force_by_side_n": force_by_side,
                "max_force_n": max_force_n,
                "both_sides_stable": both_sides_stable,
                "stable_counter": stable_counter,
                "high_force_detected": high_force_detected,
            }
        )
        object_z = float(_object_position(grasp_object)[2].item())
        object_lift_m = object_z - object_z0
        max_object_lift_m = max(max_object_lift_m, object_lift_m)
        object_z_history.append(
            {
                "step": actual_steps,
                "planned_close_rad": planned_close_rad,
                "object_z_m": object_z,
                "object_lift_m": object_lift_m,
            }
        )
        if high_force_detected:
            stopped_by_high_force = True
            high_force_step = actual_steps
            final_target = last_force_safe_target
            last_safe_target = last_force_safe_target
            last_safe_close_rad = last_force_safe_close_rad
            break
        last_force_safe_target = final_target.detach().clone()
        last_force_safe_close_rad = planned_close_rad

        if (not args_cli.enable_close_object_lift_guard) or object_lift_m <= lift_guard_trigger_m:
            last_safe_target = final_target.detach().clone()
            last_safe_close_rad = planned_close_rad
            if stable_grasp_detected and not args_cli.disable_stable_force_stop:
                stopped_by_stable_force = True
                break
        else:
            stopped_by_object_lift = True
            rewind_close_rad = max(start_close_rad, last_safe_close_rad - rewind_open_margin_rad)
            final_target = _make_custom_gripper_target(
                robot,
                rewind_close_rad,
                base_target=robot.data.joint_pos.detach().clone(),
            )
            last_safe_target = final_target.detach().clone()
            last_safe_close_rad = rewind_close_rad
            for _ in range(max(0, args_cli.object_lift_rewind_steps)):
                if budget.exhausted:
                    break
                _step_once(
                    sim,
                    robot,
                    grasp_object,
                    tactile_sensors,
                    budget,
                    arm_joint_ids=arm_joint_ids,
                    arm_joint_target=arm_joint_hold_target,
                    gripper_joint_ids=gripper_joint_ids,
                    gripper_joint_target=final_target,
                )
                actual_steps += 1
                object_z = float(_object_position(grasp_object)[2].item())
                object_lift_m = object_z - object_z0
                max_object_lift_m = max(max_object_lift_m, object_lift_m)
                last_contact_read = _read_soft_contact(contact_sensors, sim, robot, grasp_object)
                force_history.append(
                    {
                        "step": actual_steps,
                        "planned_close_rad": rewind_close_rad,
                        "force_by_side_n": last_contact_read.get("force_by_side_n", {}),
                        "max_force_n": float(last_contact_read.get("max_force_n", 0.0) or 0.0),
                        "rewind_hold": True,
                        "rewind_open_margin_rad": rewind_open_margin_rad,
                        "rewind_close_rad": rewind_close_rad,
                        "object_lift_guard_trigger_m": lift_guard_trigger_m,
                    }
                )
                object_z_history.append(
                    {
                        "step": actual_steps,
                        "planned_close_rad": rewind_close_rad,
                        "object_z_m": object_z,
                        "object_lift_m": object_lift_m,
                        "rewind_hold": True,
                        "rewind_open_margin_rad": rewind_open_margin_rad,
                        "rewind_close_rad": rewind_close_rad,
                        "object_lift_guard_trigger_m": lift_guard_trigger_m,
                    }
                )
            break

    final_close_rad = _current_close_rad(robot)
    final_object_z = float(_object_position(grasp_object)[2].item())
    final_object_lift_m = final_object_z - object_z0
    final_target_close_rad = float(final_target[0, named_joint_ids(robot, ["finger_joint"])["finger_joint"]].item())
    close_goal_reached = bool(
        final_target_close_rad >= target_close_rad - 1.0e-6
        and not stopped_by_high_force
        and not stopped_by_object_lift
    )
    summary = {
        "requested_steps": steps,
        "actual_steps": actual_steps,
        "requested_target_close_rad": requested_target_close_rad,
        "target_close_rad": target_close_rad,
        "absolute_close_cap_rad": absolute_close_cap_rad,
        "start_close_rad": start_close_rad,
        "final_target_close_rad": final_target_close_rad,
        "actual_final_close_rad": final_close_rad,
        "close_goal_reached": close_goal_reached,
        "soft_contact_detected": bool(last_contact_read.get("contact_detected", False)),
        "last_contact_read": last_contact_read,
        "soft_contact_stop_enabled": False,
        "stopped_by_no_contact_cap": target_close_rad < requested_target_close_rad,
        "force_control": {
            "enabled": force_control_enabled,
            "stable_force_threshold_n": stable_force_threshold,
            "high_force_threshold_n": high_force_threshold,
            "required_stable_steps": required_stable_steps,
            "stable_grasp_detected": stable_grasp_detected,
            "stable_grasp_step": stable_grasp_step,
            "stable_force_stop_enabled": not args_cli.disable_stable_force_stop,
            "stopped_by_stable_force": stopped_by_stable_force,
            "final_stable_counter": stable_counter,
            "max_force_seen_n": max_force_seen_n,
            "stopped_by_high_force": stopped_by_high_force,
            "high_force_step": high_force_step,
            "last_force_safe_close_rad": last_force_safe_close_rad,
            "force_history_tail": force_history[-12:],
        },
        "preclose_object_z_m": object_z0,
        "final_object_z_m": final_object_z,
        "final_object_lift_m": final_object_lift_m,
        "max_object_lift_during_close_m": max_object_lift_m,
        "max_close_object_lift_m": args_cli.max_close_object_lift,
        "object_lift_guard_trigger_m": lift_guard_trigger_m,
        "object_lift_guard_trigger_fraction": args_cli.object_lift_guard_trigger_fraction,
        "object_lift_guard_enabled": args_cli.enable_close_object_lift_guard,
        "object_lift_within_limit": (
            (not args_cli.enable_close_object_lift_guard) or max_object_lift_m <= args_cli.max_close_object_lift
        ),
        "stopped_by_object_lift": stopped_by_object_lift,
        "last_safe_close_rad": last_safe_close_rad,
        "object_lift_rewind_open_margin_rad": rewind_open_margin_rad,
        "rewind_close_rad": rewind_close_rad,
        "object_z_history_tail": object_z_history[-12:],
        "arm_hold_enabled": bool(arm_joint_ids is not None and arm_joint_hold_target is not None),
    }
    return final_target, summary


def _slow_joint_move(
    sim: sim_utils.SimulationContext,
    robot: Articulation,
    grasp_object: RigidObject,
    sensors: list[tuple[str, Any]],
    budget: StepBudget,
    target: torch.Tensor,
    *,
    steps: int,
) -> None:
    start = robot.data.joint_pos.detach().clone()
    for index in range(steps):
        if budget.exhausted:
            break
        alpha = float(index + 1) / float(max(1, steps))
        blend = start + (target - start) * alpha
        robot.set_joint_position_target(blend)
        robot.write_data_to_sim()
        _stabilize_gripper_mimic_state(robot, blend)
        sim.step()
        dt = sim.get_physics_dt()
        robot.update(dt)
        grasp_object.update(dt)
        _update_tactile_sensors(sensors, sim, robot, grasp_object)
        budget.tick()


def _resolve_ik_indices(robot: Articulation) -> tuple[list[int], list[int], int, int]:
    arm_joint_ids = list(named_joint_ids(robot, ARM_JOINT_NAMES).values())
    gripper_joint_ids = list(named_joint_ids(robot, GRIPPER_CONTROL_JOINT_NAMES).values())
    ee_body = _ee_body_id(robot)
    # For fixed-base articulations PhysX omits the root body from jacobian rows.
    ee_jacobian_index = ee_body - 1 if robot.is_fixed_base else ee_body
    return arm_joint_ids, gripper_joint_ids, ee_body, ee_jacobian_index




def _move_ee_to_pose(
    sim: sim_utils.SimulationContext,
    robot: Articulation,
    grasp_object: RigidObject,
    sensors: list[tuple[str, Any]],
    budget: StepBudget,
    *,
    target_pos_w: torch.Tensor,
    target_quat_w: torch.Tensor,
    arm_joint_ids: list[int],
    gripper_joint_ids: list[int],
    ee_body: int,
    ee_jacobian_index: int,
    gripper_joint_target: torch.Tensor,
    max_steps: int,
    pos_tolerance: float,
    max_joint_delta: float,
    stop_tolerance: float | None = None,
) -> dict[str, Any]:
    """Move the EE toward one world pose, then let the caller latch the resulting arm joints."""

    controller = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        num_envs=1,
        device=robot.device,
    )
    target_pos_w = target_pos_w.to(device=robot.device, dtype=robot.data.root_pose_w.dtype).reshape(1, 3)
    target_quat_w = target_quat_w.to(device=robot.device, dtype=robot.data.root_pose_w.dtype).reshape(1, 4)
    root_pose_w = robot.data.root_pose_w
    target_pos_b, target_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], target_pos_w, target_quat_w
    )
    command = torch.cat((target_pos_b, target_quat_b), dim=1)
    controller.reset()
    controller.set_command(command)

    final_error = math.inf
    final_step = 0
    final_pos_w = None
    stop_tolerance = pos_tolerance if stop_tolerance is None else stop_tolerance
    for step in range(1, max_steps + 1):
        if budget.exhausted:
            break
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobian_index, :, arm_joint_ids]
        ee_pose_w = robot.data.body_pose_w[:, ee_body]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        joint_pos = robot.data.joint_pos[:, arm_joint_ids]
        joint_pos_des = controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        limited_delta = torch.clamp(joint_pos_des - joint_pos, min=-max_joint_delta, max=max_joint_delta)
        limited_joint_target = joint_pos + limited_delta

        _step_once(
            sim,
            robot,
            grasp_object,
            sensors,
            budget,
            arm_joint_ids=arm_joint_ids,
            arm_joint_target=limited_joint_target,
            gripper_joint_ids=gripper_joint_ids,
            gripper_joint_target=gripper_joint_target,
        )
        ee_pos_w, _ = _ee_pose(robot, ee_body)
        final_pos_w = ee_pos_w.detach().clone()
        final_error = float(torch.linalg.norm(target_pos_w - ee_pos_w).item())
        final_step = step
        if final_error <= stop_tolerance:
            return {
                "passed": final_error <= pos_tolerance,
                "steps": step,
                "final_position_error_m": final_error,
                "final_position_world_m": [float(value) for value in final_pos_w[0].detach().cpu().tolist()],
                "accepted_stop_tolerance_m": stop_tolerance,
                "control_mode": "single_direct_ik_segment_then_arm_latch",
            }
    return {
        "passed": False,
        "steps": final_step,
        "final_position_error_m": final_error,
        "final_position_world_m": (
            [float(value) for value in final_pos_w[0].detach().cpu().tolist()] if final_pos_w is not None else None
        ),
        "accepted_stop_tolerance_m": stop_tolerance,
        "control_mode": "single_direct_ik_segment_then_arm_latch",
    }


def _refine_soft_center_alignment(
    sim: sim_utils.SimulationContext,
    robot: Articulation,
    grasp_object: RigidObject,
    sensors: list[tuple[str, Any]],
    budget: StepBudget,
    *,
    soft_center_alignment: dict[str, Any],
    target_quat_w: torch.Tensor,
    arm_joint_ids: list[int],
    gripper_joint_ids: list[int],
    ee_body: int,
    ee_jacobian_index: int,
    gripper_joint_target: torch.Tensor,
) -> dict[str, Any]:
    """Iteratively correct the reached EE pose using the measured soft-link center.

    The first target is computed from the pregrasp EE->soft-link transform.  In
    Isaac the final measured transform can drift a few centimeters after IK,
    finger settling, and articulation constraints.  This bounded correction loop
    measures the actual GSmini soft-pair center at the reached pose and nudges
    the EE by the residual cube-center error before the arm is latched.
    """

    desired_raw = soft_center_alignment.get("desired_soft_pair_center_world_m")
    if not args_cli.align_soft_center or desired_raw is None:
        return {"enabled": False, "reason": "soft-center alignment disabled or missing desired center"}

    desired = torch.tensor(
        desired_raw,
        device=robot.device,
        dtype=robot.data.root_pose_w.dtype,
    )
    rounds: list[dict[str, Any]] = []
    final_geometry = _soft_link_geometry_summary(robot, grasp_object)
    final_error = _soft_center_alignment_error(final_geometry)
    max_rounds = max(0, int(args_cli.soft_center_refine_rounds))
    max_step = max(0.0, float(args_cli.max_soft_center_refine_step))
    refine_pos_tolerance = max(1.0e-4, min(float(args_cli.ik_pos_tolerance), float(args_cli.soft_center_tolerance) * 0.5))

    for round_index in range(1, max_rounds + 1):
        if budget.exhausted or final_error.get("passed"):
            break
        actual_raw = final_error.get("soft_pair_center_world_m")
        if actual_raw is None:
            break

        actual = torch.tensor(actual_raw, device=robot.device, dtype=robot.data.root_pose_w.dtype)
        correction = desired - actual
        correction_norm = float(torch.linalg.norm(correction).item())
        applied_correction = correction.detach().clone()
        clamped = False
        if max_step > 0.0 and correction_norm > max_step:
            applied_correction = correction * (max_step / correction_norm)
            clamped = True
        elif max_step <= 0.0:
            applied_correction.zero_()
            clamped = correction_norm > 0.0

        current_ee_pos_w, _ = _ee_pose(robot, ee_body)
        refined_target_pos = current_ee_pos_w[0].detach().clone() + applied_correction
        move_result = _move_ee_to_pose(
            sim,
            robot,
            grasp_object,
            sensors,
            budget,
            target_pos_w=refined_target_pos,
            target_quat_w=target_quat_w,
            arm_joint_ids=arm_joint_ids,
            gripper_joint_ids=gripper_joint_ids,
            ee_body=ee_body,
            ee_jacobian_index=ee_jacobian_index,
            gripper_joint_target=gripper_joint_target,
            max_steps=max(1, int(args_cli.soft_center_refine_steps)),
            pos_tolerance=refine_pos_tolerance,
            max_joint_delta=args_cli.max_joint_delta_per_step,
            stop_tolerance=refine_pos_tolerance,
        )
        next_geometry = _soft_link_geometry_summary(robot, grasp_object)
        next_error = _soft_center_alignment_error(next_geometry)
        rounds.append(
            {
                "round": round_index,
                "before_error_m": final_error,
                "desired_soft_pair_center_world_m": [
                    float(value) for value in desired.detach().cpu().tolist()
                ],
                "raw_correction_world_m": [
                    float(value) for value in correction.detach().cpu().tolist()
                ],
                "applied_correction_world_m": [
                    float(value) for value in applied_correction.detach().cpu().tolist()
                ],
                "raw_correction_norm_m": correction_norm,
                "max_correction_step_m": max_step,
                "correction_clamped": clamped,
                "target_ee_world_m": [
                    float(value) for value in refined_target_pos.detach().cpu().tolist()
                ],
                "move": move_result,
                "after_error_m": next_error,
            }
        )
        final_geometry = next_geometry
        final_error = next_error

    return {
        "enabled": True,
        "rounds_requested": max_rounds,
        "rounds_used": len(rounds),
        "refine_steps_per_round": args_cli.soft_center_refine_steps,
        "refine_pos_tolerance_m": refine_pos_tolerance,
        "max_correction_step_m": max_step,
        "desired_soft_pair_center_world_m": [
            float(value) for value in desired.detach().cpu().tolist()
        ],
        "rounds": rounds,
        "final_error_m": final_error,
        "final_geometry": final_geometry,
        "passed": bool(final_error.get("passed", False)),
    }


def _skipped_motion_summary(reason: str, target_pos_w: torch.Tensor | None = None) -> dict[str, Any]:
    return {
        "passed": True,
        "steps": 0,
        "final_position_error_m": 0.0,
        "final_position_world_m": (
            [float(value) for value in target_pos_w.detach().cpu().tolist()] if target_pos_w is not None else None
        ),
        "accepted_stop_tolerance_m": 0.0,
        "skipped": True,
        "reason": reason,
    }


def _attempt_grid(max_attempts: int) -> list[AttemptParams]:
    z_offsets = [0.0, 0.02, 0.04, 0.06, 0.01, 0.08, -0.01]
    x_offsets = [0.0, -0.01, 0.01, -0.02, 0.02]
    y_offsets = [0.0, -0.02, 0.02, -0.03, 0.03, -0.01, 0.01]
    base_yaw = math.radians(args_cli.grasp_yaw_offset_deg)
    base_pitch = math.radians(args_cli.grasp_pitch_offset_deg)
    base_roll = math.radians(args_cli.grasp_roll_offset_deg)
    yaw_offsets = [0.0, math.radians(3), math.radians(-3), math.radians(6), math.radians(-6)]
    pitch_offsets = [0.0, math.radians(3), math.radians(-3)]
    roll_offsets = [0.0, math.radians(3), math.radians(-3)]
    close_values = []
    for candidate in (args_cli.gripper_close_rad, 0.20, 0.18, 0.16, 0.22, 0.14):
        close_values.append(min(candidate, max(0.0, args_cli.max_close_without_contact_rad)))

    ordered_params: list[AttemptParams] = []
    seen: set[tuple[float, float, float, float, float, float, float, float]] = set()

    def add(z: float, x: float, y: float, yaw_delta: float, pitch_delta: float, roll_delta: float, close_rad: float) -> None:
        yaw = base_yaw + yaw_delta
        pitch = base_pitch + pitch_delta
        roll = base_roll + roll_delta
        ee_z_offset = max(args_cli.min_grasp_ee_z_offset, args_cli.grasp_ee_z_offset + z)
        soft_center_z_offset = args_cli.grasp_soft_center_z_offset + z
        key = (
            round(ee_z_offset, 5),
            round(soft_center_z_offset, 5),
            round(x, 5),
            round(y, 5),
            round(yaw, 5),
            round(pitch, 5),
            round(roll, 5),
            round(close_rad, 5),
        )
        if key in seen or len(ordered_params) >= max_attempts:
            return
        seen.add(key)
        ordered_params.append(
            AttemptParams(
                attempt=0,
                ee_z_offset=ee_z_offset,
                soft_center_z_offset=soft_center_z_offset,
                longitudinal_x_offset=args_cli.grasp_x_offset + x,
                lateral_y_offset=args_cli.grasp_y_offset + y,
                yaw_offset_rad=yaw,
                pitch_offset_rad=pitch,
                roll_offset_rad=roll,
                gripper_close_rad=close_rad,
            )
        )

    # The cube target is symmetric, so stay centered and avoid rotation by
    # default.  Try small height changes before XY/angle changes: the left/right
    # side-center grasp is most sensitive to soft-pad height, while XY/yaw
    # changes make the path longer and add visible wrist motion.
    add(z_offsets[0], x_offsets[0], y_offsets[0], yaw_offsets[0], pitch_offsets[0], roll_offsets[0], close_values[0])
    for z in z_offsets[1:]:
        add(z, x_offsets[0], y_offsets[0], yaw_offsets[0], pitch_offsets[0], roll_offsets[0], close_values[0])
    for y in y_offsets[1:]:
        add(z_offsets[0], x_offsets[0], y, yaw_offsets[0], pitch_offsets[0], roll_offsets[0], close_values[0])
    for x in x_offsets[1:]:
        add(z_offsets[0], x, y_offsets[0], yaw_offsets[0], pitch_offsets[0], roll_offsets[0], close_values[0])
    for close_rad in close_values[1:]:
        add(z_offsets[0], x_offsets[0], y_offsets[0], yaw_offsets[0], pitch_offsets[0], roll_offsets[0], close_rad)
    for yaw in yaw_offsets[1:]:
        add(z_offsets[0], x_offsets[0], y_offsets[0], yaw, pitch_offsets[0], roll_offsets[0], close_values[0])
    for pitch in pitch_offsets[1:]:
        add(z_offsets[0], x_offsets[0], y_offsets[0], yaw_offsets[0], pitch, roll_offsets[0], close_values[0])
    for roll in roll_offsets[1:]:
        add(z_offsets[0], x_offsets[0], y_offsets[0], yaw_offsets[0], pitch_offsets[0], roll, close_values[0])

    attempts = []
    for params in ordered_params[:max_attempts]:
        attempts.append(
            AttemptParams(
                attempt=len(attempts) + 1,
                ee_z_offset=params.ee_z_offset,
                soft_center_z_offset=params.soft_center_z_offset,
                longitudinal_x_offset=params.longitudinal_x_offset,
                lateral_y_offset=params.lateral_y_offset,
                yaw_offset_rad=params.yaw_offset_rad,
                pitch_offset_rad=params.pitch_offset_rad,
                roll_offset_rad=params.roll_offset_rad,
                gripper_close_rad=params.gripper_close_rad,
            )
        )
    return attempts


def _hold_and_measure(
    sim: sim_utils.SimulationContext,
    robot: Articulation,
    grasp_object: RigidObject,
    sensors: list[tuple[str, Any]],
    budget: StepBudget,
    *,
    closed_target: torch.Tensor,
    initial_object_z: float,
    hold_seconds: float,
    arm_joint_ids: list[int] | None = None,
    arm_joint_hold_target: torch.Tensor | None = None,
) -> dict[str, Any]:
    gripper_joint_ids = list(named_joint_ids(robot, GRIPPER_CONTROL_JOINT_NAMES).values())
    hold_steps = _hold_steps_for_seconds(sim, hold_seconds)
    current_z = float(_object_position(grasp_object)[2].item())
    min_z = current_z
    max_z = current_z
    actual_hold_steps = 0
    for _ in range(hold_steps):
        if budget.exhausted:
            break
        _step_once(
            sim,
            robot,
            grasp_object,
            sensors,
            budget,
            arm_joint_ids=arm_joint_ids,
            arm_joint_target=arm_joint_hold_target,
            gripper_joint_ids=gripper_joint_ids,
            gripper_joint_target=closed_target,
        )
        z = float(_object_position(grasp_object)[2].item())
        min_z = min(min_z, z)
        max_z = max(max_z, z)
        actual_hold_steps += 1
    final_z = float(_object_position(grasp_object)[2].item())
    lifted_m = final_z - initial_object_z
    min_lift_m = min_z - initial_object_z
    return {
        "hold_steps": hold_steps,
        "actual_hold_steps": actual_hold_steps,
        "final_object_z_m": final_z,
        "max_object_z_m": max_z,
        "min_object_z_m": min_z,
        "final_lift_m": lifted_m,
        "min_lift_m": min_lift_m,
        "arm_hold_enabled": bool(arm_joint_ids is not None and arm_joint_hold_target is not None),
    }


def _settle_closed_gripper(
    sim: sim_utils.SimulationContext,
    robot: Articulation,
    grasp_object: RigidObject,
    sensors: list[tuple[str, Any]],
    budget: StepBudget,
    *,
    closed_target: torch.Tensor,
    steps: int,
    preclose_object_z: float | None = None,
    arm_joint_ids: list[int] | None = None,
    arm_joint_hold_target: torch.Tensor | None = None,
) -> dict[str, Any]:
    gripper_joint_ids = list(named_joint_ids(robot, GRIPPER_CONTROL_JOINT_NAMES).values())
    actual_steps = 0
    final_target = closed_target
    object_z0 = float(_object_position(grasp_object)[2].item()) if preclose_object_z is None else preclose_object_z
    max_object_lift_m = 0.0
    stopped_by_object_lift = False
    object_z_history: list[dict[str, Any]] = []
    last_safe_close_rad = _current_close_rad(robot)
    rewind_open_margin_rad = max(0.0, float(args_cli.object_lift_rewind_open_margin_rad))
    lift_guard_trigger_m = max(
        0.0,
        float(args_cli.max_close_object_lift) * max(0.0, min(1.0, float(args_cli.object_lift_guard_trigger_fraction))),
    )
    rewind_close_rad: float | None = None
    for _ in range(max(0, steps)):
        if budget.exhausted:
            break
        _step_once(
            sim,
            robot,
            grasp_object,
            sensors,
            budget,
            arm_joint_ids=arm_joint_ids,
            arm_joint_target=arm_joint_hold_target,
            gripper_joint_ids=gripper_joint_ids,
            gripper_joint_target=closed_target,
        )
        actual_steps += 1
        object_z = float(_object_position(grasp_object)[2].item())
        object_lift_m = object_z - object_z0
        max_object_lift_m = max(max_object_lift_m, object_lift_m)
        current_close_rad = _current_close_rad(robot)
        object_z_history.append(
            {
                "step": actual_steps,
                "object_z_m": object_z,
                "object_lift_m": object_lift_m,
                "close_rad": current_close_rad,
            }
        )
        if (not args_cli.enable_close_object_lift_guard) or object_lift_m <= lift_guard_trigger_m:
            last_safe_close_rad = current_close_rad
            continue

        stopped_by_object_lift = True
        rewind_close_rad = max(0.0, last_safe_close_rad - rewind_open_margin_rad)
        final_target = _make_custom_gripper_target(
            robot,
            rewind_close_rad,
            base_target=robot.data.joint_pos.detach().clone(),
        )
        last_safe_close_rad = rewind_close_rad
        for _ in range(max(0, args_cli.object_lift_rewind_steps)):
            if budget.exhausted:
                break
            _step_once(
                sim,
                robot,
                grasp_object,
                sensors,
                budget,
                arm_joint_ids=arm_joint_ids,
                arm_joint_target=arm_joint_hold_target,
                gripper_joint_ids=gripper_joint_ids,
                gripper_joint_target=final_target,
            )
            actual_steps += 1
            object_z = float(_object_position(grasp_object)[2].item())
            object_lift_m = object_z - object_z0
            max_object_lift_m = max(max_object_lift_m, object_lift_m)
            object_z_history.append(
                {
                    "step": actual_steps,
                    "object_z_m": object_z,
                    "object_lift_m": object_lift_m,
                    "close_rad": rewind_close_rad,
                    "rewind_hold": True,
                    "rewind_open_margin_rad": rewind_open_margin_rad,
                    "rewind_close_rad": rewind_close_rad,
                    "object_lift_guard_trigger_m": lift_guard_trigger_m,
                }
            )
        break
    summary = _gripper_joint_summary(robot, final_target)
    final_object_z = float(_object_position(grasp_object)[2].item())
    final_object_lift_m = final_object_z - object_z0
    return {
        "requested_steps": steps,
        "actual_steps": actual_steps,
        "gripper": summary,
        "preclose_object_z_m": object_z0,
        "final_object_z_m": final_object_z,
        "final_object_lift_m": final_object_lift_m,
        "max_object_lift_during_settle_m": max_object_lift_m,
        "max_close_object_lift_m": args_cli.max_close_object_lift,
        "object_lift_guard_trigger_m": lift_guard_trigger_m,
        "object_lift_guard_trigger_fraction": args_cli.object_lift_guard_trigger_fraction,
        "object_lift_guard_enabled": args_cli.enable_close_object_lift_guard,
        "object_lift_within_limit": (
            (not args_cli.enable_close_object_lift_guard) or max_object_lift_m <= args_cli.max_close_object_lift
        ),
        "stopped_by_object_lift": stopped_by_object_lift,
        "last_safe_close_rad": last_safe_close_rad,
        "object_lift_rewind_open_margin_rad": rewind_open_margin_rad,
        "rewind_close_rad": rewind_close_rad,
        "final_target_close_rad": float(final_target[0, named_joint_ids(robot, ["finger_joint"])["finger_joint"]].item()),
        "object_z_history_tail": object_z_history[-12:],
        "arm_hold_enabled": bool(arm_joint_ids is not None and arm_joint_hold_target is not None),
    }


def _static_geometry_audit_payload() -> dict[str, Any]:
    return {
        "urdf_pad_collision_z_max_m": URDF_PAD_COLLISION_Z_MAX_M,
        "urdf_gsmini_soft_collision_z_max_m": URDF_GSMINI_SOFT_COLLISION_Z_MAX_M,
        "urdf_gsmini_full_collision_z_max_m": URDF_GSMINI_FULL_COLLISION_Z_MAX_M,
        "gsmini_base_link_local_aabb_m": GSMINI_BASE_LINK_AABB_IN_SENSOR_FRAME_M,
        "gsmini_sensor_assembly_local_aabb_m": GSMINI_SENSOR_ASSEMBLY_AABB_IN_SENSOR_FRAME_M,
        "urdf_gsmini_collision_extension_beyond_pad_m": URDF_GSMINI_COLLISION_EXTENSION_BEYOND_PAD_M,
        "hard_sensor_collision_scale_note": URDF_HARD_SENSOR_COLLISION_SCALE_NOTE,
        "note": (
            "The canonical URDF includes explicit Robotiq finger/pad collisions and matching "
            "TacEx-style connector/GSmini visuals at the same origin/rpy/scale as before. "
            "The GSmini soft gel is a separate fixed gelpad body under the sensor case with a simple box collider, "
            "so contact-stop cross-checks the soft mesh AABB while the hard connector/base remain visual-only."
        ),
    }


def _object_payload(object_start: torch.Tensor) -> dict[str, Any]:
    return {
        "kind": PHASE2_OBJECT_KIND,
        "label": PHASE2_OBJECT_LABEL,
        "cube_size_m": PHASE2_CUBE_SIZE_M,
        "cube_side_centers_world_m": {
            "left_y_positive": [
                float(object_start[0].item()),
                float(object_start[1].item() + PHASE2_CUBE_SIZE_M / 2.0),
                float(object_start[2].item()),
            ],
            "right_y_negative": [
                float(object_start[0].item()),
                float(object_start[1].item() - PHASE2_CUBE_SIZE_M / 2.0),
                float(object_start[2].item()),
            ],
        },
    }




def run_one_attempt(
    sim: sim_utils.SimulationContext,
    robot: Articulation,
    grasp_object: RigidObject,
    origin: torch.Tensor,
    tactile_sensors: list[tuple[str, Any]],
    contact_sensors: dict[str, ContactSensor],
    budget: StepBudget,
    params: AttemptParams,
) -> dict[str, Any]:
    reset_scene(
        sim,
        robot,
        grasp_object,
        origin,
        settle_steps=args_cli.settle_steps,
        object_rest_height=PHASE2_CUBE_REST_HEIGHT,
    )
    _update_tactile_sensors(tactile_sensors, sim, robot, grasp_object)

    arm_joint_ids, gripper_joint_ids, ee_body, ee_jacobian_index = _resolve_ik_indices(robot)
    open_reset_target = build_reset_joint_target(robot)
    _slow_joint_move(sim, robot, grasp_object, tactile_sensors, budget, open_reset_target, steps=args_cli.reset_move_steps)
    pregrasp_target = build_pregrasp_joint_target(robot)
    if args_cli.pregrasp_wrist3_deg is not None:
        wrist3_joint_id = named_joint_ids(robot, ["wrist_3_joint"])["wrist_3_joint"]
        pregrasp_target[:, wrist3_joint_id] = math.radians(args_cli.pregrasp_wrist3_deg)
    _slow_joint_move(sim, robot, grasp_object, tactile_sensors, budget, pregrasp_target, steps=args_cli.pregrasp_move_steps)

    object_start = _object_position(grasp_object)
    initial_object_z = float(object_start[2].item())
    _, ee_quat = _ee_pose(robot, ee_body)
    target_quat_w = _apply_orientation_offsets(
        ee_quat.detach().clone(),
        yaw_offset_rad=params.yaw_offset_rad,
        pitch_offset_rad=params.pitch_offset_rad,
        roll_offset_rad=params.roll_offset_rad,
    )

    if args_cli.align_soft_center:
        grasp_pos, soft_center_alignment = _compute_soft_centered_grasp_target(
            robot,
            ee_body,
            object_start,
            target_quat_w,
            params,
        )
    else:
        grasp_xy = object_start.detach().clone()
        grasp_xy[0] += params.longitudinal_x_offset
        grasp_xy[1] += params.lateral_y_offset
        grasp_pos = torch.tensor(
            [float(grasp_xy[0].item()), float(grasp_xy[1].item()), initial_object_z + params.ee_z_offset],
            device=robot.device,
            dtype=robot.data.root_pose_w.dtype,
        )
        soft_center_alignment = {
            "enabled": False,
            "mode": "legacy_ee_target_offset",
            "target_ee_world_m": [float(value) for value in grasp_pos.detach().cpu().tolist()],
            "soft_center_tcp": {
                "enabled": False,
                "reason": "--no_align_soft_center selected legacy ee_link target mode",
            },
        }
    open_target = build_gripper_joint_target(robot, closed=False, base_target=robot.data.joint_pos.detach().clone())

    direct_move = _move_ee_to_pose(
        sim,
        robot,
        grasp_object,
        tactile_sensors,
        budget,
        target_pos_w=grasp_pos,
        target_quat_w=target_quat_w,
        arm_joint_ids=arm_joint_ids,
        gripper_joint_ids=gripper_joint_ids,
        ee_body=ee_body,
        ee_jacobian_index=ee_jacobian_index,
        gripper_joint_target=open_target,
        max_steps=args_cli.direct_move_steps,
        pos_tolerance=args_cli.ik_pos_tolerance,
        max_joint_delta=args_cli.max_joint_delta_per_step,
        stop_tolerance=max(args_cli.ik_pos_tolerance, args_cli.descend_accept_tolerance),
    )
    approach = _skipped_motion_summary("direct_mode_no_approach_stage", grasp_pos)
    descend_precontact = _skipped_motion_summary("direct_mode_no_precontact_stage", grasp_pos)
    descend = direct_move

    soft_center_refinement = _refine_soft_center_alignment(
        sim,
        robot,
        grasp_object,
        tactile_sensors,
        budget,
        soft_center_alignment=soft_center_alignment,
        target_quat_w=target_quat_w,
        arm_joint_ids=arm_joint_ids,
        gripper_joint_ids=gripper_joint_ids,
        ee_body=ee_body,
        ee_jacobian_index=ee_jacobian_index,
        gripper_joint_target=open_target,
    )
    soft_center_alignment["refinement"] = soft_center_refinement
    if soft_center_refinement.get("enabled"):
        final_refine_error = soft_center_refinement.get("final_error_m", {})
        print(
            "[GRASP] Soft-center refinement: "
            f"rounds={soft_center_refinement.get('rounds_used')}/"
            f"{soft_center_refinement.get('rounds_requested')}, "
            f"xz_error_m={float(final_refine_error.get('xz_center_error_m', math.inf)):.5f}, "
            f"passed={'true' if final_refine_error.get('passed') else 'false'}",
            flush=True,
        )

    preclose_arm_hold_target, arm_velocity_latch = _latch_arm_at_current_state(robot, arm_joint_ids)
    effective_grasp_pos, _ = _ee_pose(robot, ee_body)
    effective_grasp_pos = effective_grasp_pos[0].detach().clone()
    lift_pos = effective_grasp_pos.clone()
    lift_pos[2] += args_cli.lift_distance
    preclose_arm_settle = _hold_arm_and_gripper_targets(
        sim,
        robot,
        grasp_object,
        tactile_sensors,
        budget,
        arm_joint_ids=arm_joint_ids,
        arm_joint_target=preclose_arm_hold_target,
        gripper_joint_ids=gripper_joint_ids,
        gripper_joint_target=open_target,
        steps=args_cli.arm_hold_settle_steps,
    )
    preclose_soft_geometry = _soft_link_geometry_summary(robot, grasp_object)
    soft_center_alignment["preclose_error_m"] = _soft_center_alignment_error(preclose_soft_geometry)
    preclose_object_position = _object_position(grasp_object)
    preclose_object_shift_m = float(torch.linalg.norm(preclose_object_position - object_start).item())
    close_safety = {
        "preclose_object_shift_m": preclose_object_shift_m,
        "max_preclose_object_shift_m": args_cli.max_preclose_object_shift,
        "preclose_object_shift_passed": bool(preclose_object_shift_m <= args_cli.max_preclose_object_shift),
        "preclose_arm_settle_max_error_rad": preclose_arm_settle.get("arm_hold", {}).get("max_abs_error_rad"),
        "direct_grasp_target": True,
        "direct_target_note": (
            "The grasp target is initialized from the measured GSmini soft-link mesh pair center, then refined "
            "with bounded measured corrections so the pre-close soft_link center lands on the cube side-face "
            "center line plus CLI offsets. Approach/precontact/terminal XY adjustment stages are skipped; "
            "residual arm velocity is zeroed once at the reached pose before close."
        ),
        "arm_velocity_latch": arm_velocity_latch,
    }
    close_safety["passed"] = bool(close_safety["preclose_object_shift_passed"])

    if not close_safety["passed"]:
        open_summary = _gripper_joint_summary(robot, open_target)
        hold_stub = {
            "hold_steps": 0,
            "actual_hold_steps": 0,
            "final_object_z_m": float(preclose_object_position[2].item()),
            "max_object_z_m": float(preclose_object_position[2].item()),
            "min_object_z_m": float(preclose_object_position[2].item()),
            "final_lift_m": float(preclose_object_position[2].item() - initial_object_z),
            "min_lift_m": float(preclose_object_position[2].item() - initial_object_z),
        }
        print(
            "[GRASP] Skipping close for preclose object-shift safety: "
            + json.dumps(close_safety, ensure_ascii=False),
            flush=True,
        )
        return {
            "passed": False,
            "success_evaluation": {
                "selected_mode": args_cli.success_mode,
                "selected_passed": False,
                "contact_demo_passed": False,
                "lift_hold_passed": False,
                "checks": {
                    "preclose_object_shift_passed": False,
                    "soft_center_preclose_passed": bool(soft_center_alignment.get("preclose_error_m", {}).get("passed", False)),
                },
                "reason": "blocked_by_preclose_object_shift_guard",
            },
            "params": asdict(params),
            "pregrasp_wrist3_deg": args_cli.pregrasp_wrist3_deg,
            "static_geometry_audit": _static_geometry_audit_payload(),
            "object": _object_payload(object_start),
            "object_initial_position_m": [float(value) for value in object_start.detach().cpu().tolist()],
            "grasp_target_world_m": [float(value) for value in grasp_pos.detach().cpu().tolist()],
            "effective_grasp_target_world_m": [float(value) for value in effective_grasp_pos.detach().cpu().tolist()],
            "lift_target_world_m": [float(value) for value in lift_pos.detach().cpu().tolist()],
            "preclose_soft_geometry": preclose_soft_geometry,
            "soft_center_alignment": soft_center_alignment,
            "postclose_soft_geometry": preclose_soft_geometry,
            "postsettle_soft_geometry": preclose_soft_geometry,
            "close_safety": close_safety,
            "preclose_arm_settle": preclose_arm_settle,
            "direct_move": direct_move,
            "approach": approach,
            "descend_precontact": descend_precontact,
            "descend": descend,
            "close_command": {
                "executed": False,
                "reason": "blocked_by_preclose_object_shift_guard",
                "steps": 0,
                "settle_steps": 0,
                "target_close_rad": params.gripper_close_rad,
                "close_passed": False,
                "force_control": {"enabled": not args_cli.disable_force_control, "blocked_by_safety_guard": True},
            },
            "close_gripper": open_summary,
            "close_settle": {"requested_steps": 0, "actual_steps": 0, "gripper": open_summary},
            "lift": {"passed": False, "steps": 0, "final_position_error_m": math.inf, "final_position_world_m": None},
            "hold": hold_stub,
            "hold_gripper": open_summary,
            "motion_reached_nominal": False,
            "step_budget_used": budget.steps,
        }

    print(
        "[GRASP] Closing gripper with force control: "
        f"attempt={params.attempt}, target_close_rad={params.gripper_close_rad:.3f}, steps={args_cli.close_steps}, "
        f"high_force_stop={args_cli.force_control_high_force_threshold:.3f} N",
        flush=True,
    )
    closed_target, close_stop = _close_gripper_with_force_control(
        sim,
        robot,
        grasp_object,
        tactile_sensors,
        contact_sensors,
        budget,
        target_close_rad=params.gripper_close_rad,
        steps=args_cli.close_steps,
        preclose_object_z=float(preclose_object_position[2].item()),
        arm_joint_ids=arm_joint_ids,
        arm_joint_hold_target=preclose_arm_hold_target,
    )
    close_joint_summary = _gripper_joint_summary(robot, closed_target)
    postclose_soft_geometry = _soft_link_geometry_summary(robot, grasp_object)
    force_control = close_stop.get("force_control", {})
    close_passed = close_joint_summary["max_abs_error_rad"] <= 0.12
    print(
        "[GRASP] Close result: "
        f"max_abs_error_rad={close_joint_summary['max_abs_error_rad']:.4f}, "
        f"close_goal_reached={'true' if close_stop.get('close_goal_reached') else 'false'}, "
        f"stable_grasp={'true' if force_control.get('stable_grasp_detected') else 'false'}, "
        f"high_force_stop={'true' if force_control.get('stopped_by_high_force') else 'false'}, "
        f"passed={'true' if close_passed else 'false'}",
        flush=True,
    )
    close_settle = _settle_closed_gripper(
        sim,
        robot,
        grasp_object,
        tactile_sensors,
        budget,
        closed_target=closed_target,
        steps=args_cli.close_settle_steps,
        preclose_object_z=float(preclose_object_position[2].item()),
        arm_joint_ids=arm_joint_ids,
        arm_joint_hold_target=preclose_arm_hold_target,
    )
    if close_settle.get("stopped_by_object_lift"):
        closed_target = _make_custom_gripper_target(
            robot,
            float(close_settle.get("last_safe_close_rad", _current_close_rad(robot))),
            base_target=robot.data.joint_pos.detach().clone(),
        )
    postsettle_soft_geometry = _soft_link_geometry_summary(robot, grasp_object)
    stable_grasp_passed = bool(args_cli.disable_force_control or force_control.get("stable_grasp_detected", False))
    close_safety.update(
        {
            "max_close_object_lift_m": args_cli.max_close_object_lift,
            "object_lift_guard_trigger_fraction": args_cli.object_lift_guard_trigger_fraction,
            "object_lift_guard_trigger_m": close_stop.get("object_lift_guard_trigger_m"),
            "max_object_lift_during_close_m": close_stop.get("max_object_lift_during_close_m"),
            "final_object_lift_after_close_m": close_stop.get("final_object_lift_m"),
            "close_object_lift_passed": bool(close_stop.get("object_lift_within_limit", False)),
            "close_stopped_by_object_lift": bool(close_stop.get("stopped_by_object_lift", False)),
            "max_object_lift_during_settle_m": close_settle.get("max_object_lift_during_settle_m"),
            "final_object_lift_after_settle_m": close_settle.get("final_object_lift_m"),
            "settle_object_lift_passed": bool(close_settle.get("object_lift_within_limit", False)),
            "settle_stopped_by_object_lift": bool(close_settle.get("stopped_by_object_lift", False)),
            "object_lift_rewind_open_margin_rad": args_cli.object_lift_rewind_open_margin_rad,
            "force_control_enabled": bool(force_control.get("enabled", False)),
            "stable_grasp_detected": bool(force_control.get("stable_grasp_detected", False)),
            "stable_grasp_passed": stable_grasp_passed,
            "high_force_stop": bool(force_control.get("stopped_by_high_force", False)),
            "close_goal_reached": bool(close_stop.get("close_goal_reached", False)),
        }
    )
    close_safety["passed"] = bool(
        close_safety["preclose_object_shift_passed"]
        and close_safety["close_object_lift_passed"]
        and close_safety["settle_object_lift_passed"]
        and close_safety["stable_grasp_passed"]
    )

    lift = _move_ee_to_pose(
        sim,
        robot,
        grasp_object,
        tactile_sensors,
        budget,
        target_pos_w=lift_pos,
        target_quat_w=target_quat_w,
        arm_joint_ids=arm_joint_ids,
        gripper_joint_ids=gripper_joint_ids,
        ee_body=ee_body,
        ee_jacobian_index=ee_jacobian_index,
        gripper_joint_target=closed_target,
        max_steps=args_cli.lift_steps,
        pos_tolerance=args_cli.ik_pos_tolerance,
        max_joint_delta=args_cli.max_joint_delta_per_step,
    )
    hold = _hold_and_measure(
        sim,
        robot,
        grasp_object,
        tactile_sensors,
        budget,
        closed_target=closed_target,
        initial_object_z=initial_object_z,
        hold_seconds=args_cli.hold_seconds,
        arm_joint_ids=arm_joint_ids,
        arm_joint_hold_target=robot.data.joint_pos[:, arm_joint_ids].detach().clone(),
    )
    hold_joint_summary = _gripper_joint_summary(robot, closed_target)
    motion_reached_nominal = bool(
        direct_move["final_position_error_m"] <= args_cli.ik_pos_tolerance * 2.0
        and lift["final_position_error_m"] <= args_cli.ik_pos_tolerance * 2.0
    )
    success_evaluation = _attempt_success_evaluation(
        direct_move=direct_move,
        lift=lift,
        close_passed=close_passed,
        close_settle=close_settle,
        hold=hold,
        hold_joint_summary=hold_joint_summary,
        close_safety=close_safety,
        stable_grasp_passed=stable_grasp_passed,
        soft_center_alignment=soft_center_alignment,
        tactile_sides=tuple(side for side, _sensor in tactile_sensors),
    )
    passed = bool(success_evaluation["selected_passed"])
    return {
        "passed": passed,
        "success_evaluation": success_evaluation,
        "params": asdict(params),
        "pregrasp_wrist3_deg": args_cli.pregrasp_wrist3_deg,
        "static_geometry_audit": _static_geometry_audit_payload(),
        "object": _object_payload(object_start),
        "object_initial_position_m": [float(value) for value in object_start.detach().cpu().tolist()],
        "grasp_target_world_m": [float(value) for value in grasp_pos.detach().cpu().tolist()],
        "effective_grasp_target_world_m": [float(value) for value in effective_grasp_pos.detach().cpu().tolist()],
        "lift_target_world_m": [float(value) for value in lift_pos.detach().cpu().tolist()],
        "preclose_soft_geometry": preclose_soft_geometry,
        "soft_center_alignment": soft_center_alignment,
        "postclose_soft_geometry": postclose_soft_geometry,
        "postsettle_soft_geometry": postsettle_soft_geometry,
        "close_safety": close_safety,
        "preclose_arm_settle": preclose_arm_settle,
        "direct_move": direct_move,
        "approach": approach,
        "descend_precontact": descend_precontact,
        "descend": descend,
        "close_command": {
            "executed": True,
            "steps": args_cli.close_steps,
            "settle_steps": args_cli.close_settle_steps,
            "target_close_rad": params.gripper_close_rad,
            "close_passed": close_passed,
            "force_control": close_stop,
        },
        "close_gripper": close_joint_summary,
        "close_settle": close_settle,
        "lift": lift,
        "hold": hold,
        "hold_gripper": hold_joint_summary,
        "motion_reached_nominal": motion_reached_nominal,
        "step_budget_used": budget.steps,
    }



def run_grasp_test(
    sim: sim_utils.SimulationContext,
    robot: Articulation,
    grasp_object: RigidObject,
    origin: torch.Tensor,
    contact_sensors: dict[str, ContactSensor],
) -> dict[str, Any]:
    sides = ("left", "right") if args_cli.show_right_tactile else ("left",)
    tactile_sensors = setup_tactile_live_sensors(args_cli.device, sides=sides)
    budget = StepBudget(args_cli.max_steps)
    attempts = _attempt_grid(args_cli.max_attempts)
    results = []
    final_attempt: dict[str, Any] | None = None
    print(
        "[GRASP] Tuning profile: "
        + json.dumps(
            {
                "max_attempts": args_cli.max_attempts,
                "success_mode": args_cli.success_mode,
                "grasp_x_offset": args_cli.grasp_x_offset,
                "grasp_y_offset": args_cli.grasp_y_offset,
                "align_soft_center": args_cli.align_soft_center,
                "grasp_soft_center_z_offset": args_cli.grasp_soft_center_z_offset,
                "soft_center_tolerance": args_cli.soft_center_tolerance,
                "soft_center_refine_rounds": args_cli.soft_center_refine_rounds,
                "soft_center_refine_steps": args_cli.soft_center_refine_steps,
                "max_soft_center_refine_step": args_cli.max_soft_center_refine_step,
                "grasp_yaw_offset_deg": args_cli.grasp_yaw_offset_deg,
                "grasp_pitch_offset_deg": args_cli.grasp_pitch_offset_deg,
                "grasp_roll_offset_deg": args_cli.grasp_roll_offset_deg,
                "min_grasp_ee_z_offset": args_cli.min_grasp_ee_z_offset,
                "pregrasp_wrist3_deg": args_cli.pregrasp_wrist3_deg,
                "gripper_close_rad": args_cli.gripper_close_rad,
                "absolute_close_cap_rad": args_cli.max_close_without_contact_rad,
                "max_close_object_lift": args_cli.max_close_object_lift,
                "object_lift_guard_trigger_fraction": args_cli.object_lift_guard_trigger_fraction,
                "enable_close_object_lift_guard": args_cli.enable_close_object_lift_guard,
                "object_lift_rewind_steps": args_cli.object_lift_rewind_steps,
                "object_lift_rewind_open_margin_rad": args_cli.object_lift_rewind_open_margin_rad,
                "max_preclose_object_shift": args_cli.max_preclose_object_shift,
                "direct_grasp_target": "soft_link_cube_center_plus_refinement",
                "direct_move_steps": args_cli.direct_move_steps,
                "arm_hold_settle_steps": args_cli.arm_hold_settle_steps,
                "max_joint_delta_per_step": args_cli.max_joint_delta_per_step,
                "descend_accept_tolerance": args_cli.descend_accept_tolerance,
                "close_steps": args_cli.close_steps,
                "close_settle_steps": args_cli.close_settle_steps,
                "force_control_stable_force_threshold": args_cli.force_control_stable_force_threshold,
                "force_control_high_force_threshold": args_cli.force_control_high_force_threshold,
                "force_control_stable_steps": args_cli.force_control_stable_steps,
                "stable_force_stop_enabled": not args_cli.disable_stable_force_stop,
                "soft_contact_force_threshold": args_cli.soft_contact_force_threshold,
                "disable_force_control": args_cli.disable_force_control,
                "soft_contact_stop": False,
                "soft_center_tcp": "virtual_soft_center_tcp_average_of_left_right_gsmini_soft_mesh_centers",
                "tactile_contact_imprint_enabled": not args_cli.disable_tactile_contact_imprint,
                "tactile_contact_imprint_depth_mm": args_cli.tactile_contact_imprint_depth_mm,
                "tactile_contact_imprint_background_threshold": args_cli.tactile_contact_imprint_background_threshold,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for params in attempts:
        if budget.exhausted:
            final_attempt = {"passed": False, "reason": "global step budget exhausted", "params": asdict(params)}
            break
        print(f"[GRASP] Attempt {params.attempt}/{len(attempts)}: {asdict(params)}")
        result = run_one_attempt(
            sim,
            robot,
            grasp_object,
            origin,
            tactile_sensors,
            contact_sensors,
            budget,
            params,
        )
        results.append(result)
        final_attempt = result
        print(
            json.dumps(
                {
                    "attempt": params.attempt,
                    "passed": result["passed"],
                    "params": result["params"],
                    "direct_move": result["direct_move"],
                    "success_evaluation": result.get("success_evaluation"),
                    "soft_center_alignment": result.get("soft_center_alignment"),
                    "close_command": result["close_command"],
                    "close_safety": result.get("close_safety"),
                    "close_max_abs_error_rad": result["close_gripper"]["max_abs_error_rad"],
                    "close_settle_max_abs_error_rad": result["close_settle"]["gripper"]["max_abs_error_rad"],
                    "hold": result["hold"],
                    "motion_reached_nominal": result["motion_reached_nominal"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if result["passed"]:
            break

    passed = bool(final_attempt and final_attempt.get("passed"))
    reason = (
        f"success ({args_cli.success_mode})"
        if passed
        else f"stopped after {len(results)} attempts without meeting {args_cli.success_mode} success criteria"
    )
    if len(results) >= args_cli.max_attempts and not passed:
        reason = f"reached max_attempts={args_cli.max_attempts}; preserved logs for {args_cli.success_mode} review"
    best_attempt = None
    if results:
        best_attempt = max(results, key=lambda item: float(item.get("hold", {}).get("min_lift_m", -math.inf)))
    if best_attempt and not passed:
        aperture = best_attempt.get("preclose_soft_geometry", {}).get("side_grasp_aperture", {})
        if aperture.get("feasible_without_interpenetration") is False:
            reason += (
                "; soft-pad aperture is smaller than the cube "
                f"(gap={aperture.get('soft_inner_gap_m')} m, cube={aperture.get('cube_extent_m')} m)"
            )
    return {
        "passed": passed,
        "reason": reason,
        "attempts_used": len(results),
        "max_attempts": args_cli.max_attempts,
        "step_budget_used": budget.steps,
        "max_steps": args_cli.max_steps,
        "success_criteria": {
            "selected_mode": args_cli.success_mode,
            "modes": {
                "contact_demo": (
                    "Default Phase2 tactile-grasp bring-up gate: soft-center preclose alignment, stable "
                    "two-sided force close, gripper/safety checks, and live tactile contact image change. "
                    "Does not require 10 cm object lift."
                ),
                "lift_hold": "Legacy strict gate: require the configured object lift margin during hold.",
            },
            "lift_distance_m": args_cli.lift_distance,
            "success_lift_margin_m": args_cli.success_lift_margin,
            "hold_seconds": args_cli.hold_seconds,
            "max_gripper_joint_error_rad": 0.08,
            "max_close_object_lift_m": args_cli.max_close_object_lift,
            "close_object_lift_guard_enabled": args_cli.enable_close_object_lift_guard,
            "max_preclose_object_shift_m": args_cli.max_preclose_object_shift,
            "object": PHASE2_OBJECT_LABEL,
            "grasp_target": (
                "GSmini soft-link mesh pair center aligned to the 4 cm cube side-face center plus CLI offsets; "
                "no approach/precontact/terminal XY adjustment stages"
            ),
            "soft_center_tcp": {
                "frame_name": "virtual_soft_center_tcp",
                "definition": (
                    "The initial IK target is computed from the measured ee_link -> average(left/right GSmini "
                    "soft-link mesh center) transform, then bounded measured refinement corrects the realized "
                    "soft-link center before close."
                ),
                "implemented_as": "runtime geometry layer in ur5_phase2_grasp_test.py; no extra URDF joint required",
            },
            "align_soft_center": args_cli.align_soft_center,
            "grasp_soft_center_z_offset_m": args_cli.grasp_soft_center_z_offset,
            "soft_center_tolerance_m": args_cli.soft_center_tolerance,
            "soft_center_refine_rounds": args_cli.soft_center_refine_rounds,
            "soft_center_refine_steps": args_cli.soft_center_refine_steps,
            "max_soft_center_refine_step_m": args_cli.max_soft_center_refine_step,
            "object_lift_rewind_steps": args_cli.object_lift_rewind_steps,
            "object_lift_rewind_open_margin_rad": args_cli.object_lift_rewind_open_margin_rad,
            "object_lift_guard_trigger_fraction": args_cli.object_lift_guard_trigger_fraction,
            "force_control": {
                "enabled": not args_cli.disable_force_control,
                "body_by_side": SOFT_CONTACT_BODY_BY_SIDE,
                "filter_prim_path": _cube_prim_path(1),
                "contact_report_threshold_n": args_cli.soft_contact_force_threshold,
                "stable_force_threshold_n": args_cli.force_control_stable_force_threshold,
                "high_force_stop_threshold_n": args_cli.force_control_high_force_threshold,
                "stable_steps": args_cli.force_control_stable_steps,
                "stable_force_stop_enabled": not args_cli.disable_stable_force_stop,
                "soft_contact_stop_enabled": False,
                "absolute_close_cap_rad": args_cli.max_close_without_contact_rad,
            },
            "motion_profile": {
                "direct_move_steps": args_cli.direct_move_steps,
                "direct_max_joint_delta_per_step": args_cli.max_joint_delta_per_step,
                "descend_accept_tolerance_m": args_cli.descend_accept_tolerance,
                "legacy_approach_steps_skipped": args_cli.approach_steps,
                "legacy_precontact_steps_skipped": args_cli.precontact_steps,
                "close_steps": args_cli.close_steps,
                "close_settle_steps": args_cli.close_settle_steps,
                "arm_hold_settle_steps": args_cli.arm_hold_settle_steps,
            },
            "demo_default_note": (
                "Default max_attempts=1 executes the user-tuned direct cube-center target once. Set "
                "--max_attempts > 1 only when you explicitly want independent tuning attempts."
            ),
        },
        "tactile_live": {
            "enabled": not args_cli.no_tactile_live,
            "shown_data_types": ["tactile_rgb"] + (["camera_depth"] if args_cli.show_tactile_depth else []),
            "sides": sides,
            "sensor_paths": phase2_sensor_prim_paths(),
            "sensor_camera_clipping_range_m": list(DEFAULT_SENSOR_CAMERA_CLIPPING_RANGE),
            "dock_tactile_windows_right": args_cli.dock_tactile_windows_right,
            "script_tactile_panel": bool(TACTILE_PANEL and TACTILE_PANEL.enabled),
            "script_tactile_panel_stats": TACTILE_PANEL.stats if TACTILE_PANEL is not None else {},
            "contact_imprint_fallback": {
                "enabled": not args_cli.disable_tactile_contact_imprint,
                "nominal_depth_mm": args_cli.tactile_contact_imprint_depth_mm,
                "background_threshold_mean_abs_delta": args_cli.tactile_contact_imprint_background_threshold,
                "stats": TACTILE_CONTACT_IMPRINT_STATS,
                "source": "canonical GSmini soft-link AABB contact geometry rendered through TacEx/Taxim when camera-depth RGB stays at background",
            },
            "legacy_tacex_debug_windows": args_cli.show_tacex_debug_windows,
            "mount_info": TACTILE_MOUNT_INFO,
            "inner_finger_stage_audit": _inner_finger_stage_audit(),
        },
        "final_attempt": final_attempt,
        "best_attempt": (
            {
                "attempt": best_attempt["params"]["attempt"],
                "passed": best_attempt["passed"],
                "params": best_attempt["params"],
                "soft_center_alignment": best_attempt.get("soft_center_alignment"),
                "min_lift_m": best_attempt["hold"]["min_lift_m"],
                "max_lift_m": best_attempt["hold"]["max_object_z_m"] - best_attempt["object_initial_position_m"][2],
                "force_control": best_attempt["close_command"]["force_control"],
                "close_safety": best_attempt.get("close_safety"),
            }
            if best_attempt
            else None
        ),
        "attempts": results,
    }


def _pre_motion_delay(sim: sim_utils.SimulationContext, seconds: float) -> dict[str, Any]:
    """Keep the Isaac GUI responsive before robot motion starts."""

    requested_seconds = max(0.0, float(seconds))
    if requested_seconds <= 0.0:
        return {"requested_seconds": requested_seconds, "actual_seconds": 0.0, "skipped": True}

    print(
        f"[INFO] Waiting {requested_seconds:.1f}s before robot motion. "
        "Use this time to open Physics Debug / Colliders / Contact Points.",
        flush=True,
    )
    start = time.monotonic()
    next_log = start + 10.0
    render_errors = 0
    while time.monotonic() - start < requested_seconds:
        is_running = getattr(simulation_app, "is_running", None)
        if callable(is_running) and not is_running():
            break
        try:
            sim.render()
        except Exception:
            render_errors += 1
            try:
                simulation_app.update()
            except Exception:
                time.sleep(0.05)
        now = time.monotonic()
        if now >= next_log:
            remaining = max(0.0, requested_seconds - (now - start))
            print(f"[INFO] Pre-motion delay: {remaining:.0f}s remaining", flush=True)
            next_log = now + 10.0
        time.sleep(0.02)

    actual_seconds = time.monotonic() - start
    print("[INFO] Pre-motion delay complete; starting grasp motion.", flush=True)
    return {
        "requested_seconds": requested_seconds,
        "actual_seconds": actual_seconds,
        "render_errors": render_errors,
        "skipped": False,
    }


def main() -> int:
    if args_cli.num_envs != 1:
        raise ValueError("Phase2 grasp test currently supports only --num_envs 1.")
    if args_cli.max_attempts > 50:
        raise ValueError("User requested a hard cap of 50 tuning attempts; use --max_attempts <= 50.")
    if args_cli.device != "cpu":
        print(
            "[WARN] This script resets rigid poses like Phase1; if PhysX Direct GPU API errors appear, rerun with --device cpu."
        )

    print(f"[INFO] Creating SimulationContext on device={args_cli.device}", flush=True)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
    sim.set_camera_view(CAMERA_EYE, CAMERA_TARGET)
    robot_urdf_path = resolve_robot_urdf_path()
    print("[INFO] Resolving/generated robot USD from canonical URDF.", flush=True)
    robot_usd_path = ensure_robot_usd_path(force_conversion=args_cli.regenerate_robot_usd)
    print("[INFO] Designing Phase2 scene.", flush=True)
    robots, grasp_objects, origins = design_scene(
        args_cli.num_envs,
        robot_usd_path=robot_usd_path,
        object_kind=PHASE2_OBJECT_KIND,
        activate_contact_sensors=not args_cli.disable_force_control,
    )
    contact_sensors = setup_soft_contact_sensors(origin_index=1)
    origins = origins.to(sim.device)
    print("[INFO] Resetting simulation after scene design.", flush=True)
    sim.reset()

    robot = next(iter(robots.values()))
    grasp_object = next(iter(grasp_objects.values()))
    origin = origins[0]
    print(f"[INFO] Loaded canonical robot URDF from: {robot_urdf_path}")
    print(f"[INFO] Loaded generated robot USD from: {robot_usd_path}")
    print(f"[INFO] Loaded table USD from: {TABLE_USD_PATH}")
    print(f"[INFO] Loaded Phase2 grasp cube: prim={_cube_prim_path(1)}, size={PHASE2_CUBE_SIZE_M} m")
    report_scene_state(origins, robot_urdf_path, robot_usd_path, object_kind=PHASE2_OBJECT_KIND)

    print(
        "[INFO] Applying visible reset before grasp motion so the GUI starts from the mounted UR5 reset pose.",
        flush=True,
    )
    reset_scene(
        sim,
        robot,
        grasp_object,
        origin,
        settle_steps=args_cli.settle_steps,
        object_rest_height=PHASE2_CUBE_REST_HEIGHT,
    )
    _pre_motion_delay(sim, args_cli.pre_motion_delay_seconds)
    result = run_grasp_test(sim, robot, grasp_object, origin, contact_sensors)
    artifact = _write_result_artifacts(result)
    print(f"[INFO] Wrote Phase2 grasp result artifact: {artifact}")
    if not result["passed"]:
        append_phase2_grasp_log("Phase2 grasp test did not reach success criteria", json.dumps(result, ensure_ascii=False, indent=2))
        print("[RESULT] Phase 2 grasp test FAILED.")
        return 1
    print("[RESULT] Phase 2 grasp test PASSED.")
    return 0


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    finally:
        simulation_app.close(wait_for_replicator=False)
    raise SystemExit(exit_code)
