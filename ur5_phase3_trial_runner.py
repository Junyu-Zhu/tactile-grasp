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

from ur5_phase3_geometry import phase3_object_prim_path, setup_phase3_contact_sensors
from ur5_phase3_logging import Phase3TrialLogger
from ur5_phase3_motion import (
    Phase3MotionMixin,
    StepBudget,
)
from ur5_phase3_objects import Phase3ObjectProfile, Phase3ProtocolProfile
from ur5_phase3_schema import make_trial_id


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
    max_joint_delta_per_step: float = 0.018
    soft_center_refine_rounds: int = 2
    soft_center_refine_steps: int = 50
    max_soft_center_refine_step_m: float = 0.020
    close_steps: int = 320
    close_settle_steps: int = 6
    release_steps: int = 60
    arm_hold_settle_steps: int = 50
    stable_force_steps: int = 1
    sample_every_steps: int = 4
    tactile_sample_every_steps: int = 4
    tactile_sides: tuple[str, ...] = ("left", "right")
    tactile_device: str = "cuda:0"
    tactile_resolution: tuple[int, int] = (320, 240)
    include_camera_depth: bool = True
    include_camera_rgb: bool = False
    disable_tactile: bool = False
    save_tactile_arrays: bool = True
    save_preview_images: bool = True
    tactile_contact_imprint_enabled: bool = True
    tactile_imprint_min_depth_mm: float = 0.08
    tactile_imprint_max_depth_mm: float = 2.5
    tactile_imprint_depth_per_mm_overlap: float = 0.65
    tactile_imprint_sigma_x: float = 0.34
    tactile_imprint_sigma_y: float = 0.42
    disable_force_control: bool = False


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
        contact_sensors: dict[str, ContactSensor],
        options: Phase3RunnerOptions,
    ) -> None:
        self.sim = sim
        self.robot = robot
        self.grasp_object = grasp_object
        self.origin = origin
        self.object_profile = object_profile
        self.protocol_profile = protocol_profile
        self.contact_sensors = contact_sensors
        self.options = options
        self.budget = StepBudget(options.max_steps)
        self.tactile_sensors: list[tuple[str, Any]] = []
        self.tactile_sensor_cfgs: dict[str, Any] = {}
        self.tactile_setup: dict[str, Any] = {"enabled": False}
        self._last_tactile_pose_sync: dict[str, Any] = {"enabled": False}
        self._last_tactile_contact_proxy: dict[str, Any] = {"enabled": False}
        self._last_tactile_imprint: dict[str, Any] = {"enabled": False}
        self._tactile_imprint_baselines: dict[str, torch.Tensor] = {}
        self._tactile_imprint_stats: dict[str, dict[str, Any]] = {}
        self._last_tactile_sensor_refresh: dict[str, Any] = {"refreshed": False}
        self._last_tactile_scene_signature: tuple[Any, ...] | None = None
        self._last_tactile_scene_signature_by_side: dict[str, tuple[Any, ...]] = {}
        self.contact_onset: dict[str, Any] | None = None
        self._last_contact_state: dict[str, Any] = {"contact_detected": False}
        self._last_tactile_outputs: dict[str, dict[str, Any]] = {}
        random.seed(options.seed)
        torch.manual_seed(options.seed)

    def setup_tactile_sensors(self) -> dict[str, Any]:
        if self.options.disable_tactile:
            self.tactile_setup = {"enabled": False, "reason": "disabled_by_cli"}
            return self.tactile_setup

        from ur5_phase2_mount import (
            clear_phase2_visual_prims,
            mount_phase2_sensor_shells,
            phase2_sensor_prim_paths,
            validate_phase2_sensor_camera_prims,
            validate_phase2_sensor_mounts,
        )
        from ur5_phase2_tactile import build_phase2_gsmini_cfg, initialize_phase2_sensor

        sides = tuple(self.options.tactile_sides)
        clear_phase2_visual_prims()
        shell_spawn = mount_phase2_sensor_shells(sides, hide_render_geometry=True)
        camera_check = validate_phase2_sensor_camera_prims(sides)
        mount_check = validate_phase2_sensor_mounts(sides)
        failed = [
            side
            for side in sides
            if not camera_check.get(side, {}).get("camera_exists") or not mount_check.get(side, {}).get("passed")
        ]
        if failed:
            raise RuntimeError(f"Phase3 tactile sensor shell validation failed for sides: {failed}")

        self.tactile_sensors = []
        self.tactile_sensor_cfgs = {}
        self._last_tactile_scene_signature_by_side = {}
        for side in sides:
            cfg = build_phase2_gsmini_cfg(
                side,
                device=self.options.tactile_device,
                resolution=self.options.tactile_resolution,
                include_camera_depth=self.options.include_camera_depth,
                include_camera_rgb=self.options.include_camera_rgb,
                debug_vis=False,
            )
            self.tactile_sensor_cfgs[side] = cfg
            self.tactile_sensors.append((side, initialize_phase2_sensor(cfg)))
            self._last_tactile_scene_signature_by_side[side] = ("no_probe",)
        self._last_tactile_scene_signature = ("no_contact_proxy",)

        # Keep the TacEx runtime shells in the same fingertip-child local-offset
        # mode that Phase2 validates.  TiledCamera render products can retain a
        # stale view when these referenced camera prims are repeatedly rewritten
        # to world poses, so Phase3 gates Phase2-style camera-visible contact
        # probes by the actual runtime contact state instead of moving the
        # sensor shell hierarchy every logged sample.
        initial_pose_sync = {
            "enabled": False,
            "mode": "phase2_fingertip_child_local_offsets",
            "reason": "preserve Phase2 TacEx render-product behavior",
        }
        self.tactile_setup = {
            "enabled": True,
            "sides": list(sides),
            "sensor_paths": phase2_sensor_prim_paths(),
            "shell_spawn": shell_spawn,
            "camera_check": camera_check,
            "mount_check": mount_check,
            "resolution": list(self.options.tactile_resolution),
            "include_camera_depth": self.options.include_camera_depth,
            "include_camera_rgb": self.options.include_camera_rgb,
            "pose_sync": {
                "mode": "phase2_fingertip_child_local_offsets",
                "initial": initial_pose_sync,
            },
            "contact_proxy": {
                "enabled": False,
                "mode": "disabled_replaced_by_continuous_taxim_contact_imprint",
            },
            "contact_imprint": {
                "enabled": self.options.tactile_contact_imprint_enabled,
                "mode": "soft_object_overlap_to_continuous_taxim_height_map",
                "min_depth_mm": self.options.tactile_imprint_min_depth_mm,
                "max_depth_mm": self.options.tactile_imprint_max_depth_mm,
                "depth_per_mm_overlap": self.options.tactile_imprint_depth_per_mm_overlap,
            },
        }
        return self.tactile_setup

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

        self._reset_scene()
        reset_target = self._reset_joint_target()
        for _ in range(self.options.reset_settle_steps):
            if self.budget.exhausted:
                break
            self._step_once(logger, "reset", gripper_joint_target=reset_target, gripper_joint_ids=None)
        self._log_sample(logger, "reset", force_tactile=True)

        pregrasp_target = self._pregrasp_target()
        self._slow_joint_move(logger, pregrasp_target, stage="pre_grasp", steps=self.options.pregrasp_move_steps)
        self._log_sample(logger, "pre_grasp", force_tactile=True)

        arm_joint_ids, gripper_joint_ids, ee_body, ee_jacobian_index = self._resolve_ik_indices()
        open_gripper_target = self._open_gripper_target()
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
        release_summary = self._run_release(logger, arm_joint_ids, arm_hold_target, gripper_joint_ids)
        self._log_sample(logger, "end_trial", force_tactile=True)

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
            "soft_center_target": soft_center_target,
            "direct_move": direct_move,
            "soft_center_refinement": refinement,
            "arm_latch": latch,
            "close": close_summary,
            "hold": hold_summary,
            "micro_lift": micro_lift_summary,
            "release": release_summary,
            "contact_onset": contact_onset,
            "alignment_summary": meta.get("alignment_summary", {}),
        }
