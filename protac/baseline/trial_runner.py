"""Deterministic Phase 3 contact/grasp trial runner orchestration."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor

from protac.grasp.contact_geometry import (
    phase3_object_prim_path as phase3_object_prim_path,
    setup_phase3_contact_sensors as setup_phase3_contact_sensors,
)
from protac.compat.phase3_v1_logger import Phase3TrialLogger
from protac.grasp.nominal_controller import (
    Phase3MotionMixin,
    StepBudget,
)
from protac.scene.object_profiles import Phase3ObjectProfile, Phase3ProtocolProfile
from protac.compat.phase3_v1_schema import make_trial_id


@dataclass(frozen=True)
class Phase3RunnerOptions:
    output_root: Path
    seed: int = 7
    max_steps: int = 0
    reset_settle_steps: int = 20
    pregrasp_move_steps: int = 40
    direct_move_steps: int = 320
    direct_pos_tolerance_m: float = 0.006
    direct_pass_tolerance_m: float = 0.008
    max_joint_delta_per_step: float = 0.006
    approach_clearance_m: float = 0.080
    soft_center_refine_rounds: int = 2
    soft_center_refine_steps: int = 50
    max_soft_center_refine_step_m: float = 0.020
    close_steps: int = 320
    close_settle_steps: int = 6
    release_steps: int = 60
    arm_hold_settle_steps: int = 50
    stable_force_steps: int = 6
    post_lift_hold_steps: int = 180
    sample_every_steps: int = 16
    tactile_sample_every_steps: int = 4
    tactile_sides: tuple[str, ...] = ("left", "right")
    tactile_device: str = "cuda:0"
    tactile_resolution: tuple[int, int] = (320, 240)
    include_camera_depth: bool = False
    tactile_debug_vis: bool = False
    tactile_live_every_steps: int = 4
    disable_tactile: bool = False
    save_tactile_arrays: bool = True
    save_preview_images: bool = True
    disable_force_control: bool = False
    realtime_pacing: bool = False


class Phase3TrialRunner(Phase3MotionMixin):
    def __init__(
        self,
        sim: sim_utils.SimulationContext,
        robot: Articulation,
        grasp_object: RigidObject,
        origin: torch.Tensor,
        *,
        object_profile: Phase3ObjectProfile,
        protocol_profile: Phase3ProtocolProfile,
        contact_sensors: dict[str, dict[str, ContactSensor]],
        options: Phase3RunnerOptions,
        viewport_geometry_setup: dict[str, dict[str, object]] | None = None,
    ) -> None:
        self.sim = sim
        self.robot = robot
        self.grasp_object = grasp_object
        self.origin = origin
        self.object_profile = object_profile
        self.protocol_profile = protocol_profile
        self.contact_sensors = contact_sensors
        self.options = options
        self.viewport_geometry_setup = viewport_geometry_setup or {}
        self.budget = StepBudget(options.max_steps)
        self.tactile_sensors: list[tuple[str, Any]] = []
        self.tactile_sensor_cfgs: dict[str, Any] = {}
        self.tactile_setup: dict[str, Any] = {"enabled": False}
        self._last_tactile_capture: dict[str, Any] = {"mode": "not_initialized"}
        self.contact_onset: dict[str, Any] | None = None
        self._last_contact_state: dict[str, Any] = {"contact_detected": False}
        self._last_tactile_outputs: dict[str, dict[str, Any]] = {}
        self._tactile_render_ready = False
        self._last_tactile_sensor_sync: dict[str, Any] = {}
        self._next_step_deadline: float | None = None
        self._max_adaptor_force_seen_n = 0.0
        self._adaptor_contact_detected = False
        self._adaptor_sensor_error_detected = False
        random.seed(options.seed)
        torch.manual_seed(options.seed)

    def setup_tactile_sensors(self) -> dict[str, Any]:
        if self.options.disable_tactile:
            self.tactile_setup = {"enabled": False, "reason": "disabled_by_cli"}
            return self.tactile_setup

        from protac.tactile.mount import (
            clear_legacy_visual_prims,
            mount_sensor_shells,
            sensor_prim_paths,
            set_viewport_guide_purpose_enabled,
            sync_sensor_shells_to_robot,
            validate_sensor_camera_prims,
            validate_sensor_mounts,
        )
        from protac.tactile.sensor import (
            build_gsmini_cfg,
            enable_tactile_debug_windows,
            initialize_gsmini_sensor,
            position_tactile_debug_windows,
        )

        sides = tuple(self.options.tactile_sides)
        clear_legacy_visual_prims()
        if not self.viewport_geometry_setup:
            raise RuntimeError(
                "Phase3 tactile render geometry must be configured before sim.reset() "
                "to keep the articulation's PhysX tensor view valid."
            )
        shell_spawn = mount_sensor_shells(
            sides,
            hide_render_geometry=True,
            configure_canonical_gelpad_as_guide=False,
        )
        # Guide purpose is a persistent global Hydra setting in Isaac Sim 4.5,
        # not a main-viewport-only flag.  Keep it off so robot colliders and the
        # hidden canonical gel never enter TacEx's TiledCamera render products.
        set_viewport_guide_purpose_enabled(False)
        initial_sensor_sync = sync_sensor_shells_to_robot(self.robot, sides)
        camera_check = validate_sensor_camera_prims(sides)
        mount_check = validate_sensor_mounts(sides)
        failed = [
            side
            for side in sides
            if not camera_check.get(side, {}).get("camera_exists") or not mount_check.get(side, {}).get("passed")
        ]
        if failed:
            raise RuntimeError(f"Phase3 tactile sensor shell validation failed for sides: {failed}")

        self.tactile_sensors = []
        self.tactile_sensor_cfgs = {}
        for side in sides:
            cfg = build_gsmini_cfg(
                side,
                device=self.options.tactile_device,
                resolution=self.options.tactile_resolution,
                include_camera_depth=self.options.include_camera_depth,
                debug_vis=self.options.tactile_debug_vis,
            )
            self.tactile_sensor_cfgs[side] = cfg
            sensor = initialize_gsmini_sensor(cfg)
            self.tactile_sensors.append((side, sensor))
        # Prime both continuously-enabled TiledCamera products before turning
        # on TacEx's native debug attributes.  Their callbacks can then display
        # a real, calibrated no-contact frame rather than uninitialized data.
        self._capture_tactile_outputs()
        initial_clean_capture: dict[str, Any] = {
            "performed": True,
            "capture": self._last_tactile_capture,
        }
        debug_attributes = (
            enable_tactile_debug_windows(
                sides,
                include_camera_depth=self.options.include_camera_depth,
            )
            if self.options.tactile_debug_vis
            else {}
        )
        debug_windows = {
            "enabled": self.options.tactile_debug_vis,
            "mode": "tacex_native_sensor_debug_vis",
            "user_draggable": True,
            "sides": list(sides) if self.options.tactile_debug_vis else [],
            "attributes": debug_attributes,
        }
        if self.options.tactile_debug_vis:
            # One ordinary app/render update lets TacEx create its own windows;
            # this code only moves those native windows so left/right do not
            # overlap.
            self.sim.render(mode=self.sim.RenderMode.FULL_RENDERING)
            debug_windows["initial_positions"] = position_tactile_debug_windows(
                self.tactile_sensors
            )
        self.tactile_setup = {
            "enabled": True,
            "sides": list(sides),
            "sensor_paths": sensor_prim_paths(),
            "shell_spawn": shell_spawn,
            "camera_check": camera_check,
            "mount_check": mount_check,
            "initial_sensor_sync": initial_sensor_sync,
            "resolution": list(self.options.tactile_resolution),
            "output_data_types": ["tactile_rgb"],
            "mount_mode": "detached TacEx Sensor.usd synchronized from canonical URDF GSmini case bodies",
            "debug_windows": debug_windows,
            "initial_clean_capture": initial_clean_capture,
            "viewport_gel": {
                "render_mesh_visible": True,
                "render_mode": "tacex_style_secondary_ray_isolation_with_opaque_blue_primary_view",
                "physics_and_contact_enabled": True,
                "sensor_isolation": "TacEx primvars:invisibleToSecondaryRays on canonical gel mesh",
                "pre_reset_geometry_setup": self.viewport_geometry_setup,
            },
        }
        return self.tactile_setup

    def teardown_tactile_viewport(self) -> None:
        from protac.tactile.mount import set_viewport_guide_purpose_enabled

        # Do not restore a stale persistent ``guide`` value: it is the source
        # of both the striped robot and TacEx camera occlusion.
        set_viewport_guide_purpose_enabled(False)

    def run_trial(self, *, trial_index: int) -> dict[str, Any]:
        trial_id = make_trial_id(self.object_profile.object_id, trial_index)
        logger = Phase3TrialLogger(
            self.options.output_root,
            trial_id=trial_id,
            object_id=self.object_profile.object_id,
            sensor_ids=self.options.tactile_sides,
            seed=self.options.seed + trial_index,
            protocol_variant=self.protocol_profile.variant,
            object_profile=self.object_profile.as_dict(),
            command_profile={
                "runner_options": asdict(self.options) | {"output_root": self.options.output_root.as_posix()},
                "protocol_profile": self.protocol_profile.as_dict(),
                "tactile_setup": self.tactile_setup,
            },
            save_tactile_arrays=self.options.save_tactile_arrays,
            save_preview_images=self.options.save_preview_images,
        )
        self.contact_onset = None
        self.budget = StepBudget(self.options.max_steps)
        self._next_step_deadline = None
        self._max_adaptor_force_seen_n = 0.0
        self._adaptor_contact_detected = False
        self._adaptor_sensor_error_detected = False

        self._reset_scene()
        reset_target = self._reset_joint_target()
        for _ in range(self.options.reset_settle_steps):
            if self.budget.exhausted:
                break
            self._step_once(logger, "reset", gripper_joint_target=reset_target, gripper_joint_ids=None)
        self._log_sample(logger, "reset", force_tactile=True)

        pregrasp_target = self._pregrasp_target()
        pregrasp_move = self._slow_joint_move(
            logger,
            pregrasp_target,
            stage="pre_grasp",
            steps=self.options.pregrasp_move_steps,
        )
        self._log_sample(logger, "pre_grasp", force_tactile=True)

        arm_joint_ids, gripper_joint_ids, ee_body, ee_jacobian_index = self._resolve_ik_indices()
        open_gripper_target = self._open_gripper_target()
        if self.object_profile.grasp_ee_world_pos_m is not None:
            _, current_quat_w = self._ee_pose(ee_body)
            target_quat_w = (
                torch.tensor(
                    self.object_profile.grasp_ee_world_quat_wxyz,
                    device=self.robot.device,
                    dtype=self.robot.data.root_pose_w.dtype,
                ).reshape(1, 4)
                if self.object_profile.grasp_ee_world_quat_wxyz is not None
                else current_quat_w
            )
            target_ee_pos = self.origin + torch.tensor(
                self.object_profile.grasp_ee_world_pos_m,
                device=self.robot.device,
                dtype=self.robot.data.root_pose_w.dtype,
            )
            hover_ee_pos = target_ee_pos + torch.tensor(
                [0.0, 0.0, self.options.approach_clearance_m],
                device=self.robot.device,
                dtype=self.robot.data.root_pose_w.dtype,
            )
            hover_move = self._move_ee_to_pose(
                logger,
                stage="pre_grasp",
                target_pos_w=hover_ee_pos,
                target_quat_w=target_quat_w[0],
                arm_joint_ids=arm_joint_ids,
                gripper_joint_ids=gripper_joint_ids,
                ee_body=ee_body,
                ee_jacobian_index=ee_jacobian_index,
                gripper_joint_target=open_gripper_target,
                max_steps=self.object_profile.fast_grasp_move_steps,
                pos_tolerance=self.options.direct_pos_tolerance_m,
                max_joint_delta=self.options.max_joint_delta_per_step,
                orientation_tolerance_rad=0.01,
            )
            initial_pose_move = self._move_ee_to_pose(
                logger,
                stage="pre_grasp",
                target_pos_w=target_ee_pos,
                target_quat_w=target_quat_w[0],
                arm_joint_ids=arm_joint_ids,
                gripper_joint_ids=gripper_joint_ids,
                ee_body=ee_body,
                ee_jacobian_index=ee_jacobian_index,
                gripper_joint_target=open_gripper_target,
                max_steps=self.object_profile.fast_grasp_move_steps,
                pos_tolerance=self.options.direct_pos_tolerance_m,
                max_joint_delta=self.options.max_joint_delta_per_step,
                orientation_tolerance_rad=0.01,
            )
            approach_offset_w = torch.tensor(
                self.object_profile.approach_offset_world_m,
                device=self.robot.device,
                dtype=self.robot.data.root_pose_w.dtype,
            )
            desired_center = self._object_grasp_center_world() + approach_offset_w
            actual_center = self._soft_mesh_pair_center_world()
            pre_correction_error = desired_center - actual_center
            correction_limit_m = 0.012
            bounded_correction = torch.clamp(
                pre_correction_error,
                min=-correction_limit_m,
                max=correction_limit_m,
            )
            current_ee_pos, _ = self._ee_pose(ee_body)
            center_correction = self._move_ee_to_pose(
                logger,
                stage="pre_grasp",
                target_pos_w=current_ee_pos[0] + bounded_correction,
                target_quat_w=target_quat_w[0],
                arm_joint_ids=arm_joint_ids,
                gripper_joint_ids=gripper_joint_ids,
                ee_body=ee_body,
                ee_jacobian_index=ee_jacobian_index,
                gripper_joint_target=open_gripper_target,
                max_steps=min(80, self.object_profile.fast_grasp_move_steps),
                pos_tolerance=0.004,
                max_joint_delta=self.options.max_joint_delta_per_step,
                orientation_tolerance_rad=0.01,
            )
            post_correction_error = desired_center - self._soft_mesh_pair_center_world()
            center_error = float(torch.linalg.norm(post_correction_error).item())
            direct_move = {
                "mode": "verified_fixed_cube_cartesian_pose",
                "passed": (
                    center_error <= self.options.direct_pass_tolerance_m
                    and center_correction["final_orientation_error_rad"] <= 0.03
                ),
                "steps": hover_move["steps"] + initial_pose_move["steps"] + center_correction["steps"],
                "hover_move": hover_move,
                "initial_pose_move": initial_pose_move,
                "center_correction": center_correction,
                "target_position_world_m": initial_pose_move["target_position_world_m"],
                "target_orientation_world_wxyz": initial_pose_move["target_orientation_world_wxyz"],
                "final_position_world_m": center_correction["final_position_world_m"],
                "final_orientation_world_wxyz": center_correction["final_orientation_world_wxyz"],
                "ee_position_error_m": center_correction["final_position_error_m"],
                "soft_pair_center_error_m": center_error,
                "final_position_error_m": center_error,
                "final_orientation_error_rad": center_correction["final_orientation_error_rad"],
                "pos_tolerance_m": self.options.direct_pass_tolerance_m,
                "orientation_tolerance_rad": 0.03,
            }
            soft_center_target = {
                "mode": "verified_fixed_cube_cartesian_pose",
                "target_ee_world_m": list(self.object_profile.grasp_ee_world_pos_m),
                "target_ee_world_quat_wxyz": target_quat_w[0].detach().cpu().tolist(),
                "approach_clearance_m": self.options.approach_clearance_m,
                "hover_ee_world_m": hover_ee_pos.detach().cpu().tolist(),
                "grasp_center_offset_world_m": approach_offset_w.detach().cpu().tolist(),
            }
            refinement = {
                "enabled": True,
                "mode": "single_bounded_soft_center_correction",
                "pre_error_m": pre_correction_error.detach().cpu().tolist(),
                "bounded_correction_m": bounded_correction.detach().cpu().tolist(),
                "correction_limit_m": correction_limit_m,
                "move": center_correction,
                "post_error_m": post_correction_error.detach().cpu().tolist(),
                "final_error_norm_m": center_error,
            }
        else:
            _, target_quat_w = self._ee_pose(ee_body)
            target_ee_pos, soft_center_target = self._compute_soft_centered_target(
                ee_body=ee_body,
                target_quat_w=target_quat_w[0],
            )
            direct_move = self._move_ee_to_pose(
                logger,
                stage="pre_grasp",
                target_pos_w=target_ee_pos,
                target_quat_w=target_quat_w[0],
                arm_joint_ids=arm_joint_ids,
                gripper_joint_ids=gripper_joint_ids,
                ee_body=ee_body,
                ee_jacobian_index=ee_jacobian_index,
                gripper_joint_target=open_gripper_target,
                max_steps=self.options.direct_move_steps,
                pos_tolerance=self.options.direct_pos_tolerance_m,
                max_joint_delta=self.options.max_joint_delta_per_step,
            )
            refinement = self._refine_soft_center(
                logger,
                target_quat_w=target_quat_w[0],
                arm_joint_ids=arm_joint_ids,
                gripper_joint_ids=gripper_joint_ids,
                ee_body=ee_body,
                ee_jacobian_index=ee_jacobian_index,
                gripper_joint_target=open_gripper_target,
            )
        arm_hold_target, latch = self._latch_arm(arm_joint_ids)
        post_refine_stage = "contact_close" if self.contact_onset is not None else "pre_grasp"
        self._hold_targets(
            logger,
            post_refine_stage,
            arm_joint_ids=arm_joint_ids,
            arm_joint_target=arm_hold_target,
            gripper_joint_ids=gripper_joint_ids,
            gripper_joint_target=open_gripper_target,
            steps=self.options.arm_hold_settle_steps,
        )
        self._log_sample(logger, post_refine_stage, force_tactile=True)

        close_target, close_summary = self._close_gripper(
            logger,
            arm_joint_ids=arm_joint_ids,
            arm_joint_target=arm_hold_target,
            gripper_joint_ids=gripper_joint_ids,
        )
        self._log_sample(logger, "contact_close", force_tactile=True)
        hold_summary = self._run_hold_if_enabled(logger, arm_joint_ids, arm_hold_target, gripper_joint_ids, close_target)
        micro_lift_summary = self._run_micro_lift_if_enabled(
            logger,
            arm_joint_ids,
            gripper_joint_ids,
            ee_body,
            ee_jacobian_index,
            close_target,
        )
        lift_arm_target, lift_latch = self._latch_arm(arm_joint_ids)
        post_lift_hold = self._hold_targets(
            logger,
            "micro_lift",
            arm_joint_ids=arm_joint_ids,
            arm_joint_target=lift_arm_target,
            gripper_joint_ids=gripper_joint_ids,
            gripper_joint_target=close_target,
            steps=self.options.post_lift_hold_steps,
        )
        retained_object_z = float(self.grasp_object.data.root_pose_w[0, 2].item())
        if micro_lift_summary.get("enabled"):
            pre_lift_z = float(micro_lift_summary.get("pre_lift_object_z_m", retained_object_z))
            retained_lift_m = retained_object_z - pre_lift_z
            micro_lift_summary["post_lift_hold"] = post_lift_hold
            micro_lift_summary["retained_object_z_m"] = retained_object_z
            micro_lift_summary["retained_object_lift_m"] = retained_lift_m
            micro_lift_summary["object_lift_m"] = retained_lift_m
            micro_lift_summary["object_lift_passed"] = retained_lift_m >= float(
                micro_lift_summary.get("min_required_object_lift_m", 0.0)
            )
        release_summary = self._run_release(logger, arm_joint_ids, lift_arm_target, gripper_joint_ids)
        self._log_sample(logger, "end_trial", force_tactile=True)

        close_summary["max_adaptor_force_seen_n"] = self._max_adaptor_force_seen_n
        close_summary["adaptor_contact_detected"] = self._adaptor_contact_detected
        close_summary["adaptor_sensor_valid"] = not self._adaptor_sensor_error_detected
        contact_onset = self.contact_onset or {"detected": False}
        failed_reasons = self._failure_reasons(direct_move, contact_onset, close_summary, micro_lift_summary)
        success_label = not failed_reasons
        failure_reason = "; ".join(failed_reasons) if failed_reasons else None
        meta = logger.finalize(success_label=success_label, failure_reason=failure_reason, contact_onset=contact_onset)
        return {
            "trial_id": trial_id,
            "passed": success_label,
            "failure_reason": failure_reason,
            "meta_path": logger.paths.meta_json.as_posix(),
            "trial_dir": logger.paths.trial_dir.as_posix(),
            "step_budget_used": self.budget.steps,
            "object_id": self.object_profile.object_id,
            "protocol_variant": self.protocol_profile.variant,
            "tactile_setup": self.tactile_setup,
            "pregrasp_move": pregrasp_move,
            "soft_center_target": soft_center_target,
            "direct_move": direct_move,
            "soft_center_refinement": refinement,
            "arm_latch": {"pre_close": latch, "post_lift": lift_latch},
            "close": close_summary,
            "hold": hold_summary,
            "micro_lift": micro_lift_summary,
            "release": release_summary,
            "contact_onset": contact_onset,
            "alignment_summary": meta.get("alignment_summary", {}),
        }
