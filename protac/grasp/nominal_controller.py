# mypy: disable-error-code="attr-defined, has-type"
"""Phase 3 deterministic motion helpers built on locked geometry helpers."""

from __future__ import annotations

import math
import time
from typing import Any

import torch

from isaaclab.assets import Articulation
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, subtract_frame_transforms

from protac.tactile.gsmini_contract import ISAACLAB_CAMERA_DATA_FORWARD_AXIS
from protac.robot.control import (
    ARM_JOINT_NAMES,
    GRIPPER_COMMAND_JOINT_NAMES,
    build_gripper_joint_target,
    build_pregrasp_joint_target,
    build_reset_joint_target,
    gripper_joint_overrides_rad,
    named_joint_ids,
)
from protac.robot.reset import reset_robot
from protac.grasp.contact_geometry import Phase3GeometryMixin, _list_tensor
from protac.compat.phase3_v1_logger import Phase3TrialLogger
from protac.compat.phase3_v1_schema import move_result_accepted, utc_now_iso

PHASE3_DEFAULT_PREGRASP_WRIST3_DEG = -2.6


class StepBudget:
    def __init__(self, max_steps: int) -> None:
        self.max_steps = int(max_steps)
        self.steps = 0

    @property
    def exhausted(self) -> bool:
        return self.max_steps > 0 and self.steps >= self.max_steps

    def tick(self) -> None:
        self.steps += 1


def _make_custom_gripper_target(
    robot: Articulation,
    close_rad: float,
    *,
    base_target: torch.Tensor | None = None,
) -> torch.Tensor:
    from protac.robot.control import build_joint_target

    overrides = gripper_joint_overrides_rad(closed=True)
    sign_by_joint = {joint_name: 1.0 if value >= 0.0 else -1.0 for joint_name, value in overrides.items()}
    custom = {joint_name: sign_by_joint[joint_name] * close_rad for joint_name in overrides}
    return build_joint_target(robot, custom, base_target=base_target)


class Phase3MotionMixin(Phase3GeometryMixin):
    contact_onset: dict[str, Any] | None
    _next_step_deadline: float | None
    _TACTILE_CONTACT_STAGES = frozenset({"contact_close", "hold", "micro_lift"})

    def _sample_period_for_stage(self, stage: str) -> int:
        if stage in self._TACTILE_CONTACT_STAGES:
            return int(self.options.tactile_sample_every_steps)
        return int(self.options.sample_every_steps)

    def _pace_realtime(self) -> None:
        """Cap interactive playback at the physics rate without slowing headless runs."""

        if not self.options.realtime_pacing:
            return
        dt = float(self.sim.get_physics_dt())
        now = time.monotonic()
        if self._next_step_deadline is None or now - self._next_step_deadline > 0.25:
            self._next_step_deadline = now
        self._next_step_deadline += dt
        delay = self._next_step_deadline - time.monotonic()
        if delay > 0.0:
            time.sleep(delay)

    def _tactile_camera_observation(self, sensor: Any) -> dict[str, Any]:
        """Summarize camera/object geometry and raw depth for runtime audits."""

        camera = getattr(sensor, "camera", None)
        data = getattr(camera, "data", None)
        if data is None or getattr(data, "pos_w", None) is None:
            return {"available": False}
        position = data.pos_w[0].detach()
        quat_world = data.quat_w_world[0].detach()
        # CameraData.quat_w_world is already converted by IsaacLab from the
        # authored OpenGL -Z camera frame to its +X-forward world convention.
        forward = quat_apply(
            quat_world.reshape(1, 4),
            torch.tensor(
                [ISAACLAB_CAMERA_DATA_FORWARD_AXIS],
                device=quat_world.device,
                dtype=quat_world.dtype,
            ),
        )[0]
        target = self._object_grasp_center_world().to(device=position.device, dtype=position.dtype)
        camera_to_target = target - position
        axial_distance = float(torch.dot(camera_to_target, forward).item())
        lateral_distance = float(torch.linalg.norm(camera_to_target - axial_distance * forward).item())

        sensor_output = getattr(getattr(sensor, "_data", None), "output", {})
        height_map = sensor_output.get("height_map")
        height_map_summary: dict[str, Any] = {"available": False}
        if height_map is not None:
            depth_m = height_map.detach() / 1000.0
            finite = torch.isfinite(depth_m)
            finite_values = depth_m[finite]
            height_map_summary = {
                "available": True,
                "finite_fraction": float(finite.float().mean().item()),
                "min_m": float(finite_values.min().item()) if finite_values.numel() else None,
                "max_m": float(finite_values.max().item()) if finite_values.numel() else None,
            }
        return {
            "available": True,
            "position_world_m": _list_tensor(position),
            "forward_world": _list_tensor(forward),
            "forward_axis_convention": "IsaacLab CameraData world convention (+X forward, +Z up)",
            "object_grasp_center_world_m": _list_tensor(target),
            "object_center_axial_distance_m": axial_distance,
            "object_center_lateral_distance_m": lateral_distance,
            "taxim_height_map": height_map_summary,
        }

    def _capture_tactile_outputs(self) -> dict[str, dict[str, Any]]:
        if not self.tactile_sensors:
            return {}
        from protac.tactile.sensor import update_tactile_sensor

        if not self._tactile_render_ready:
            self._sync_tactile_sensors_and_render()

        outputs: dict[str, dict[str, Any]] = {}
        dt = self.sim.get_physics_dt()
        capture_diagnostics: dict[str, Any] = {}
        try:
            for side, sensor in self.tactile_sensors:
                sensor_output = update_tactile_sensor(sensor, self.sim, dt=dt, render_ticks=0)
                tactile_rgb = sensor_output.get("tactile_rgb")
                if tactile_rgb is None:
                    raise RuntimeError("TacEx sensor did not return tactile_rgb")
                outputs[side] = {"tactile_rgb": tactile_rgb}
                capture_diagnostics[side] = {
                    "camera_observation": self._tactile_camera_observation(sensor),
                }
        except Exception as exc:  # pragma: no cover - Isaac runtime only
            side_name = locals().get("side", "unknown")
            raise RuntimeError(f"TacEx {side_name} tactile capture failed: {exc}") from exc
        if self.options.tactile_debug_vis:
            # TacEx owns the window, image provider, conversion and display.
            # This normal app update presents the outputs computed above; no
            # duplicate custom renderer or HydraTexture state toggling is used.
            self._render_tactile_frame()
            self._tactile_render_ready = True
        live_frame_counts = {
            side: int(sensor.frame[0].item())
            for side, sensor in self.tactile_sensors
        }
        self._last_tactile_capture = {
            "mode": "tacex_native_render_then_sensor_update",
            "viewport_capture_isolation": "canonical_gel_visible_to_primary_rays_and_hidden_from_secondary_rays",
            "sensor_sync": self._last_tactile_sensor_sync,
            "sides": capture_diagnostics,
            "live_frame_counts": live_frame_counts,
        }
        self._last_tactile_outputs = outputs
        return outputs

    def _sync_tactile_sensors_and_render(self) -> None:
        """Synchronize detached TacEx shells, then render one clean camera frame."""

        if self.tactile_sensors:
            from protac.tactile.mount import sync_sensor_shells_to_robot

            self._last_tactile_sensor_sync = sync_sensor_shells_to_robot(
                self.robot,
                tuple(side for side, _ in self.tactile_sensors),
                sensor_instances=dict(self.tactile_sensors),
            )
        self._render_tactile_frame()
        self._tactile_render_ready = True

    def _render_tactile_frame(self) -> None:
        """Render cameras without requesting an unsupported headless GUI mode."""

        if self.sim.has_gui():
            self.sim.render(mode=self.sim.RenderMode.FULL_RENDERING)
        else:
            self.sim.render()

    def _advance_simulation_frame(self) -> float:
        """Advance physics, synchronize detached TacEx cameras, then render."""

        self._tactile_render_ready = False
        self.sim.step(render=False)
        dt = self.sim.get_physics_dt()
        self.robot.update(dt)
        self.grasp_object.update(dt)
        self._sync_tactile_sensors_and_render()
        return dt

    def _reset_object(self) -> None:
        root_pose = self.grasp_object.data.default_root_state[:, :7].clone()
        root_pose[:, 0] = float(self.origin[0].item()) + self.object_profile.root_position_m[0]
        root_pose[:, 1] = float(self.origin[1].item()) + self.object_profile.root_position_m[1]
        root_pose[:, 2] = float(self.origin[2].item()) + self.object_profile.root_position_m[2]
        root_pose[:, 3:7] = torch.tensor(
            self.object_profile.root_rot_wxyz,
            device=self.grasp_object.device,
            dtype=root_pose.dtype,
        )
        root_velocity = torch.zeros((root_pose.shape[0], 6), device=self.grasp_object.device, dtype=root_pose.dtype)
        self.grasp_object.write_root_pose_to_sim(root_pose)
        self.grasp_object.write_root_velocity_to_sim(root_velocity)
        self.grasp_object.reset()

    def _reset_scene(self) -> torch.Tensor:
        target = build_reset_joint_target(self.robot)
        reset_robot(self.robot, self.origin, target)
        self._reset_object()
        self._tactile_render_ready = False
        return target

    def _log_sample(
        self,
        logger: Phase3TrialLogger,
        stage: str,
        *,
        force_tactile: bool = False,
        tactile_outputs: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        contact_state = self._read_contact_state()
        if tactile_outputs is None:
            tactile_outputs = self._capture_tactile_outputs() if (force_tactile or self.tactile_sensors) else {}
        contact_state = dict(contact_state)
        contact_state["tactile_capture"] = self._last_tactile_capture
        logger.record_sample(
            timestamp=utc_now_iso(),
            action_stage=stage,
            joint_state=self._joint_state(),
            gripper_state=self._gripper_state(),
            object_state=self._object_state(),
            contact_state=contact_state,
            tactile_outputs=tactile_outputs,
        )

    def _step_once(
        self,
        logger: Phase3TrialLogger,
        stage: str,
        *,
        arm_joint_ids: list[int] | None = None,
        arm_joint_target: torch.Tensor | None = None,
        gripper_joint_ids: list[int] | None = None,
        gripper_joint_target: torch.Tensor | None = None,
        force_log: bool = False,
    ) -> None:
        if arm_joint_target is not None and arm_joint_ids is not None:
            self.robot.set_joint_position_target(arm_joint_target, joint_ids=arm_joint_ids)
        if gripper_joint_target is not None and gripper_joint_ids is not None:
            self.robot.set_joint_position_target(gripper_joint_target[:, gripper_joint_ids], joint_ids=gripper_joint_ids)
        if arm_joint_target is not None or gripper_joint_target is not None:
            self.robot.write_data_to_sim()
        self._advance_simulation_frame()
        self.budget.tick()
        contact_state = self._read_contact_state()
        onset_stage = stage in {"contact_close", "hold", "micro_lift"}
        onset_has_force = bool(contact_state.get("force_contact_sides"))
        if contact_state.get("contact_detected") and self.contact_onset is None and (onset_stage or onset_has_force):
            self.contact_onset = {
                "detected": True,
                "step": self.budget.steps,
                "timestamp": utc_now_iso(),
                "action_stage": stage,
                "contact_sides": contact_state.get("contact_sides", []),
                "force_by_side_n": contact_state.get("force_by_side_n", {}),
            }
            force_log = True
        sample_every_steps = self._sample_period_for_stage(stage)
        live_outputs = None
        live_period = max(1, int(self.options.tactile_live_every_steps))
        if self.options.tactile_debug_vis and self.tactile_sensors and self.budget.steps % live_period == 0:
            live_outputs = self._capture_tactile_outputs()
        if force_log or (sample_every_steps > 0 and self.budget.steps % sample_every_steps == 0):
            self._log_sample(logger, stage, force_tactile=force_log, tactile_outputs=live_outputs)
        self._pace_realtime()

    def _resolve_ik_indices(self) -> tuple[list[int], list[int], int, int]:
        arm_joint_ids = list(named_joint_ids(self.robot, ARM_JOINT_NAMES).values())
        gripper_joint_ids = list(named_joint_ids(self.robot, GRIPPER_COMMAND_JOINT_NAMES).values())
        ee_body = int(self.robot.find_bodies(["ee_link"], preserve_order=True)[0][0])
        ee_jacobian_index = ee_body - 1 if self.robot.is_fixed_base else ee_body
        return arm_joint_ids, gripper_joint_ids, ee_body, ee_jacobian_index

    def _ee_pose(self, ee_body: int) -> tuple[torch.Tensor, torch.Tensor]:
        pose = self.robot.data.body_pose_w[:, ee_body]
        return pose[:, 0:3].detach().clone(), pose[:, 3:7].detach().clone()

    def _slow_joint_move(
        self,
        logger: Phase3TrialLogger,
        target: torch.Tensor,
        *,
        stage: str,
        steps: int,
    ) -> dict[str, Any]:
        start = self.robot.data.joint_pos.detach().clone()
        actual_steps = 0
        for index in range(max(1, steps)):
            if self.budget.exhausted:
                break
            phase = float(index + 1) / float(max(1, steps))
            alpha = phase * phase * (3.0 - 2.0 * phase)
            blend = start + (target - start) * alpha
            self.robot.set_joint_position_target(blend)
            self.robot.write_data_to_sim()
            self._advance_simulation_frame()
            self.budget.tick()
            actual_steps += 1
            sample_every_steps = self._sample_period_for_stage(stage)
            if sample_every_steps > 0 and self.budget.steps % sample_every_steps == 0:
                self._log_sample(logger, stage)
            elif (
                self.options.tactile_debug_vis
                and self.tactile_sensors
                and self.budget.steps % max(1, int(self.options.tactile_live_every_steps)) == 0
            ):
                self._capture_tactile_outputs()
            self._pace_realtime()
        return {
            "mode": "joint_space_smoothstep",
            "requested_steps": steps,
            "actual_steps": actual_steps,
        }

    def _move_ee_to_pose(
        self,
        logger: Phase3TrialLogger,
        *,
        stage: str,
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
        orientation_tolerance_rad: float | None = None,
        trajectory_steps: int = 0,
    ) -> dict[str, Any]:
        controller = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            num_envs=1,
            device=self.robot.device,
        )
        target_pos_w = target_pos_w.to(device=self.robot.device, dtype=self.robot.data.root_pose_w.dtype).reshape(1, 3)
        target_quat_w = target_quat_w.to(device=self.robot.device, dtype=self.robot.data.root_pose_w.dtype).reshape(1, 4)
        root_pose_w = self.robot.data.root_pose_w
        target_pos_b, target_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], target_pos_w, target_quat_w
        )
        start_pose_w = self.robot.data.body_pose_w[:, ee_body].detach().clone()
        start_pos_b, start_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], start_pose_w[:, 0:3], start_pose_w[:, 3:7]
        )
        controller.reset()
        controller.set_command(torch.cat((target_pos_b, target_quat_b), dim=1))
        final_error = math.inf
        final_orientation_error = math.inf
        final_pos_w: torch.Tensor | None = None
        final_quat_w: torch.Tensor | None = None
        final_step = 0
        for step in range(1, max_steps + 1):
            if self.budget.exhausted:
                break
            trajectory_complete = True
            if trajectory_steps > 0:
                phase = min(1.0, float(step) / float(trajectory_steps))
                alpha = phase**3 * (10.0 - 15.0 * phase + 6.0 * phase**2)
                command_pos_b = start_pos_b + (target_pos_b - start_pos_b) * alpha
                # Contact-preserving lifts keep orientation fixed.  For the
                # first few steps use the measured start quaternion to avoid an
                # unnecessary orientation impulse, then command the final one.
                command_quat_b = start_quat_b if step == 1 else target_quat_b
                controller.set_command(torch.cat((command_pos_b, command_quat_b), dim=1))
                trajectory_complete = phase >= 1.0
            jacobian = self.robot.root_physx_view.get_jacobians()[:, ee_jacobian_index, :, arm_joint_ids]
            ee_pose_w = self.robot.data.body_pose_w[:, ee_body]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            joint_pos = self.robot.data.joint_pos[:, arm_joint_ids]
            joint_pos_des = controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
            limited_target = joint_pos + torch.clamp(joint_pos_des - joint_pos, min=-max_joint_delta, max=max_joint_delta)
            self._step_once(
                logger,
                stage,
                arm_joint_ids=arm_joint_ids,
                arm_joint_target=limited_target,
                gripper_joint_ids=gripper_joint_ids,
                gripper_joint_target=gripper_joint_target,
            )
            ee_pos_w, ee_quat_w = self._ee_pose(ee_body)
            final_pos_w = ee_pos_w.detach().clone()
            final_quat_w = ee_quat_w.detach().clone()
            final_error = float(torch.linalg.norm(target_pos_w - ee_pos_w).item())
            quaternion_dot = torch.sum(target_quat_w * ee_quat_w, dim=1).abs().clamp(max=1.0)
            final_orientation_error = float((2.0 * torch.acos(quaternion_dot))[0].item())
            final_step = step
            orientation_reached = (
                orientation_tolerance_rad is None
                or final_orientation_error <= orientation_tolerance_rad
            )
            if trajectory_complete and final_error <= pos_tolerance and orientation_reached:
                break
        orientation_passed = (
            orientation_tolerance_rad is None
            or final_orientation_error <= orientation_tolerance_rad
        )
        return {
            "passed": final_error <= pos_tolerance and orientation_passed,
            "steps": final_step,
            "target_position_world_m": _list_tensor(target_pos_w[0]),
            "target_orientation_world_wxyz": _list_tensor(target_quat_w[0]),
            "final_position_world_m": _list_tensor(final_pos_w[0]) if final_pos_w is not None else None,
            "final_orientation_world_wxyz": _list_tensor(final_quat_w[0]) if final_quat_w is not None else None,
            "final_position_error_m": final_error,
            "final_orientation_error_rad": final_orientation_error,
            "pos_tolerance_m": pos_tolerance,
            "orientation_tolerance_rad": orientation_tolerance_rad,
            "trajectory": "cartesian_minimum_jerk" if trajectory_steps > 0 else "direct_dls_servo",
            "trajectory_steps": trajectory_steps,
        }

    def _compute_soft_centered_target(self, *, ee_body: int, target_quat_w: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        ee_pos_w, ee_quat_w = self._ee_pose(ee_body)
        current_soft_pair_center_w = self._soft_mesh_pair_center_world()
        soft_offset_in_ee = quat_apply_inverse(ee_quat_w, (current_soft_pair_center_w.reshape(1, 3) - ee_pos_w))[0]
        desired_soft_pair_center_w = self._object_grasp_center_world()
        approach_offset_w = torch.tensor(
            getattr(self.object_profile, "approach_offset_world_m", (0.0, 0.0, 0.0)),
            device=self.robot.device,
            dtype=self.robot.data.root_pose_w.dtype,
        )
        desired_soft_pair_center_w = desired_soft_pair_center_w + approach_offset_w
        target_soft_offset_w = quat_apply(target_quat_w.reshape(1, 4), soft_offset_in_ee.reshape(1, 3))[0]
        target_ee_pos_w = desired_soft_pair_center_w - target_soft_offset_w
        return target_ee_pos_w.to(device=self.robot.device, dtype=self.robot.data.root_pose_w.dtype), {
            "mode": "soft_mesh_pair_center_to_object_grasp_center",
            "desired_soft_pair_center_world_m": _list_tensor(desired_soft_pair_center_w),
            "approach_offset_world_m": _list_tensor(approach_offset_w),
            "pregrasp_ee_world_m": _list_tensor(ee_pos_w[0]),
            "pregrasp_soft_pair_center_world_m": _list_tensor(current_soft_pair_center_w),
            "soft_offset_in_ee_frame_m": _list_tensor(soft_offset_in_ee),
            "target_ee_world_m": _list_tensor(target_ee_pos_w),
        }

    def _refine_soft_center(
        self,
        logger: Phase3TrialLogger,
        *,
        target_quat_w: torch.Tensor,
        arm_joint_ids: list[int],
        gripper_joint_ids: list[int],
        ee_body: int,
        ee_jacobian_index: int,
        gripper_joint_target: torch.Tensor,
    ) -> dict[str, Any]:
        approach_offset_w = torch.tensor(
            getattr(self.object_profile, "approach_offset_world_m", (0.0, 0.0, 0.0)),
            device=self.robot.device,
            dtype=self.robot.data.root_pose_w.dtype,
        )
        desired = self._object_grasp_center_world() + approach_offset_w
        rounds: list[dict[str, Any]] = []
        stage = "contact_close" if getattr(self.object_profile, "contact_approach_refine", False) else "pre_grasp"
        for round_index in range(1, self.options.soft_center_refine_rounds + 1):
            if self.budget.exhausted:
                break
            current = self._soft_mesh_pair_center_world()
            error = desired - current
            step_vec = torch.clamp(error, min=-self.options.max_soft_center_refine_step_m, max=self.options.max_soft_center_refine_step_m)
            ee_pos, _ = self._ee_pose(ee_body)
            move = self._move_ee_to_pose(
                logger,
                stage=stage,
                target_pos_w=ee_pos[0] + step_vec,
                target_quat_w=target_quat_w,
                arm_joint_ids=arm_joint_ids,
                gripper_joint_ids=gripper_joint_ids,
                ee_body=ee_body,
                ee_jacobian_index=ee_jacobian_index,
                gripper_joint_target=gripper_joint_target,
                max_steps=self.options.soft_center_refine_steps,
                pos_tolerance=self.options.direct_pos_tolerance_m,
                max_joint_delta=self.options.max_joint_delta_per_step,
            )
            post_error = desired - self._soft_mesh_pair_center_world()
            rounds.append({
                "round": round_index,
                "pre_error_m": _list_tensor(error),
                "step_m": _list_tensor(step_vec),
                "move": move,
                "post_error_m": _list_tensor(post_error),
                "post_error_norm_m": float(torch.linalg.norm(post_error).item()),
            })
            if rounds[-1]["post_error_norm_m"] <= self.options.direct_pos_tolerance_m:
                break
        return {
            "enabled": self.options.soft_center_refine_rounds > 0,
            "stage": stage,
            "desired_soft_pair_center_world_m": _list_tensor(desired),
            "rounds": rounds,
            "final_error_norm_m": rounds[-1]["post_error_norm_m"] if rounds else None,
        }

    def _latch_arm(self, arm_joint_ids: list[int]) -> tuple[torch.Tensor, dict[str, Any]]:
        arm_joint_target = self.robot.data.joint_pos[:, arm_joint_ids].detach().clone()
        self.robot.set_joint_position_target(arm_joint_target, joint_ids=arm_joint_ids)
        self.robot.write_data_to_sim()
        return arm_joint_target, {
            "latched": True,
            "mode": "hold_current_target_without_state_teleport",
            "target_rad": _list_tensor(arm_joint_target[0]),
        }

    def _hold_targets(
        self,
        logger: Phase3TrialLogger,
        stage: str,
        *,
        arm_joint_ids: list[int],
        arm_joint_target: torch.Tensor,
        gripper_joint_ids: list[int],
        gripper_joint_target: torch.Tensor,
        steps: int,
    ) -> dict[str, Any]:
        actual_steps = 0
        for _ in range(max(0, steps)):
            if self.budget.exhausted:
                break
            self._step_once(
                logger,
                stage,
                arm_joint_ids=arm_joint_ids,
                arm_joint_target=arm_joint_target,
                gripper_joint_ids=gripper_joint_ids,
                gripper_joint_target=gripper_joint_target,
            )
            actual_steps += 1
        return {"requested_steps": steps, "actual_steps": actual_steps}

    def _close_gripper(
        self,
        logger: Phase3TrialLogger,
        *,
        arm_joint_ids: list[int],
        arm_joint_target: torch.Tensor,
        gripper_joint_ids: list[int],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        base_target = self.robot.data.joint_pos.detach().clone()
        finger_id = named_joint_ids(self.robot, ["finger_joint"])["finger_joint"]
        start_close_rad = abs(float(base_target[0, finger_id].item()))
        target_close_rad = self.object_profile.close_rad
        final_target = _make_custom_gripper_target(self.robot, start_close_rad, base_target=base_target)
        object_z0 = float(self.grasp_object.data.root_pose_w[0, 2].item())
        force_history: list[dict[str, Any]] = []
        stable_counter = 0
        stopped_by_stable_force = False
        stopped_by_high_force = False
        stopped_by_object_lift = False
        max_object_lift_m = 0.0
        bilateral_contact_detected = False
        bilateral_contact_step: int | None = None
        bilateral_force_contact_detected = False
        bilateral_force_contact_step: int | None = None
        max_force_seen_n = 0.0
        max_adaptor_force_seen_n = 0.0
        final_contact_state: dict[str, Any] = {}
        for index in range(max(1, self.options.close_steps)):
            if self.budget.exhausted:
                break
            alpha = float(index + 1) / float(max(1, self.options.close_steps))
            planned_close_rad = start_close_rad + (target_close_rad - start_close_rad) * alpha
            final_target = _make_custom_gripper_target(self.robot, planned_close_rad, base_target=base_target)
            self._step_once(
                logger,
                "contact_close",
                arm_joint_ids=arm_joint_ids,
                arm_joint_target=arm_joint_target,
                gripper_joint_ids=gripper_joint_ids,
                gripper_joint_target=final_target,
            )
            contact_state = self._last_contact_state
            final_contact_state = contact_state
            force_by_side = contact_state.get("force_by_side_n", {})
            adaptor_force_by_side = contact_state.get("adaptor_force_by_side_n", {})
            contact_sides = set(contact_state.get("contact_sides", []))
            if {"left", "right"}.issubset(contact_sides):
                bilateral_contact_detected = True
                if bilateral_contact_step is None:
                    bilateral_contact_step = index + 1
            both_sides_stable = bool(force_by_side and all(float(force_by_side.get(side, 0.0)) >= self.object_profile.stable_force_threshold_n for side in ("left", "right")))
            if both_sides_stable:
                bilateral_force_contact_detected = True
                if bilateral_force_contact_step is None:
                    bilateral_force_contact_step = index + 1
            stable_counter = stable_counter + 1 if both_sides_stable else 0
            max_force_n = float(contact_state.get("max_force_n", 0.0) or 0.0)
            max_force_seen_n = max(max_force_seen_n, max_force_n)
            max_adaptor_force_n = float(contact_state.get("max_adaptor_force_n", 0.0) or 0.0)
            max_adaptor_force_seen_n = max(max_adaptor_force_seen_n, max_adaptor_force_n)
            object_lift_m = float(self.grasp_object.data.root_pose_w[0, 2].item()) - object_z0
            max_object_lift_m = max(max_object_lift_m, object_lift_m)
            force_history.append({
                "step": index + 1,
                "planned_close_rad": planned_close_rad,
                "contact_sides": sorted(contact_sides),
                "force_by_side_n": force_by_side,
                "adaptor_force_by_side_n": adaptor_force_by_side,
                "max_force_n": max_force_n,
                "max_adaptor_force_n": max_adaptor_force_n,
                "both_sides_stable": both_sides_stable,
                "bilateral_contact_detected": bilateral_contact_detected,
                "bilateral_force_contact_detected": bilateral_force_contact_detected,
                "object_lift_m": object_lift_m,
            })
            if max_force_n >= self.object_profile.high_force_threshold_n:
                stopped_by_high_force = True
                break
            if object_lift_m > self.object_profile.max_close_object_lift_m:
                stopped_by_object_lift = True
                break
            if stable_counter >= self.options.stable_force_steps:
                stopped_by_stable_force = True
                break
        settle = self._hold_targets(
            logger,
            "contact_close",
            arm_joint_ids=arm_joint_ids,
            arm_joint_target=arm_joint_target,
            gripper_joint_ids=gripper_joint_ids,
            gripper_joint_target=final_target,
            steps=self.options.close_settle_steps,
        )
        return final_target, {
            "requested_target_close_rad": target_close_rad,
            "start_close_rad": start_close_rad,
            "stopped_by_stable_force": stopped_by_stable_force,
            "stopped_by_high_force": stopped_by_high_force,
            "stopped_by_object_lift": stopped_by_object_lift,
            "stable_counter": stable_counter,
            "bilateral_contact_detected": bilateral_contact_detected,
            "bilateral_contact_step": bilateral_contact_step,
            "bilateral_force_contact_detected": bilateral_force_contact_detected,
            "bilateral_force_contact_step": bilateral_force_contact_step,
            "max_force_seen_n": max_force_seen_n,
            "max_adaptor_force_seen_n": max_adaptor_force_seen_n,
            "max_object_lift_m": max_object_lift_m,
            "final_contact_state": final_contact_state,
            "force_history_tail": force_history[-20:],
            "settle": settle,
        }

    def _pregrasp_target(self) -> torch.Tensor:
        if getattr(self.object_profile, "pregrasp_mode", "") == "reset_clearance":
            return build_reset_joint_target(self.robot)
        target = build_pregrasp_joint_target(self.robot)
        target[:, named_joint_ids(self.robot, ["wrist_3_joint"])["wrist_3_joint"]] = math.radians(PHASE3_DEFAULT_PREGRASP_WRIST3_DEG)
        return target

    def _reset_joint_target(self) -> torch.Tensor:
        return build_reset_joint_target(self.robot)

    def _open_gripper_target(self) -> torch.Tensor:
        return build_gripper_joint_target(self.robot, closed=False, base_target=self.robot.data.joint_pos)

    def _run_hold_if_enabled(self, logger: Phase3TrialLogger, arm_joint_ids: list[int], arm_hold_target: torch.Tensor, gripper_joint_ids: list[int], close_target: torch.Tensor) -> dict[str, Any]:
        if "hold" not in self.protocol_profile.stages:
            return {"requested_steps": 0, "actual_steps": 0}
        hold_seconds = max(self.protocol_profile.hold_seconds, self.object_profile.hold_seconds)
        hold_steps = max(1, int(math.ceil(hold_seconds / self.sim.get_physics_dt())))
        summary = self._hold_targets(logger, "hold", arm_joint_ids=arm_joint_ids, arm_joint_target=arm_hold_target, gripper_joint_ids=gripper_joint_ids, gripper_joint_target=close_target, steps=hold_steps)
        self._log_sample(logger, "hold", force_tactile=True)
        return summary

    def _run_micro_lift_if_enabled(self, logger: Phase3TrialLogger, arm_joint_ids: list[int], gripper_joint_ids: list[int], ee_body: int, ee_jacobian_index: int, close_target: torch.Tensor) -> dict[str, Any]:
        if "micro_lift" not in self.protocol_profile.stages:
            return {"enabled": False}
        ee_pos, ee_quat = self._ee_pose(ee_body)
        distance = max(self.protocol_profile.micro_lift_distance_m, self.object_profile.micro_lift_distance_m)
        object_z0 = float(self.grasp_object.data.root_pose_w[0, 2].item())
        move = self._move_ee_to_pose(
            logger,
            stage="micro_lift",
            target_pos_w=ee_pos[0] + torch.tensor([0.0, 0.0, distance], device=self.robot.device),
            target_quat_w=ee_quat[0],
            arm_joint_ids=arm_joint_ids,
            gripper_joint_ids=gripper_joint_ids,
            ee_body=ee_body,
            ee_jacobian_index=ee_jacobian_index,
            gripper_joint_target=close_target,
            max_steps=max(180, self.options.direct_move_steps),
            pos_tolerance=min(0.002, self.options.direct_pos_tolerance_m),
            max_joint_delta=self.options.max_joint_delta_per_step,
            trajectory_steps=max(120, self.options.direct_move_steps // 2),
        )
        final_object_z = float(self.grasp_object.data.root_pose_w[0, 2].item())
        object_lift_m = final_object_z - object_z0
        min_required_object_lift_m = max(0.015, distance * 0.55)
        self._log_sample(logger, "micro_lift", force_tactile=True)
        return {
            "enabled": True,
            "distance_m": distance,
            "move": move,
            "pre_lift_object_z_m": object_z0,
            "final_object_z_m": final_object_z,
            "object_lift_m": object_lift_m,
            "min_required_object_lift_m": min_required_object_lift_m,
            "object_lift_passed": object_lift_m >= min_required_object_lift_m,
        }

    def _run_release(self, logger: Phase3TrialLogger, arm_joint_ids: list[int], arm_hold_target: torch.Tensor, gripper_joint_ids: list[int]) -> dict[str, Any]:
        release_target = self._open_gripper_target()
        summary = self._hold_targets(logger, "release", arm_joint_ids=arm_joint_ids, arm_joint_target=arm_hold_target, gripper_joint_ids=gripper_joint_ids, gripper_joint_target=release_target, steps=self.options.release_steps)
        self._log_sample(logger, "release", force_tactile=True)
        return summary

    def _failure_reasons(
        self,
        direct_move: dict[str, Any],
        contact_onset: dict[str, Any],
        close_summary: dict[str, Any],
        micro_lift_summary: dict[str, Any],
    ) -> list[str]:
        failed_reasons: list[str] = []
        if self.budget.exhausted:
            failed_reasons.append("global step budget exhausted")
        direct_accepted = move_result_accepted(
            direct_move,
            position_fallback_tolerance_m=self.options.direct_pass_tolerance_m,
        )
        direct_move["accepted_pass_tolerance_m"] = self.options.direct_pass_tolerance_m
        direct_move["accepted_for_trial"] = direct_accepted
        orientation_required = direct_move.get("orientation_tolerance_rad") is not None
        if not direct_accepted and (orientation_required or not contact_onset.get("detected")):
            failed_reasons.append("direct pre-grasp IK target not reached")
        if not contact_onset.get("detected"):
            failed_reasons.append("contact_onset not detected")
        if not close_summary.get("bilateral_contact_detected"):
            failed_reasons.append("bilateral GSmini contact not detected during close")
        if not close_summary.get("bilateral_force_contact_detected") or not close_summary.get(
            "stopped_by_stable_force"
        ):
            failed_reasons.append("stable bilateral force grasp not established")
        if close_summary.get("stopped_by_high_force"):
            failed_reasons.append("close stopped by high force")
        if close_summary.get("stopped_by_object_lift"):
            failed_reasons.append("close stopped by object lift guard")
        if not close_summary.get("adaptor_sensor_valid", False):
            failed_reasons.append("adaptor contact sensor validation failed")
        if close_summary.get("adaptor_contact_detected"):
            failed_reasons.append("adaptor contacted cube during grasp")
        if not micro_lift_summary.get("enabled"):
            failed_reasons.append("micro lift stage not enabled by protocol")
        else:
            lift_move = micro_lift_summary.get("move", {})
            lift_accepted = move_result_accepted(
                lift_move,
                position_fallback_tolerance_m=self.options.direct_pass_tolerance_m,
            )
            lift_move["accepted_pass_tolerance_m"] = self.options.direct_pass_tolerance_m
            lift_move["accepted_for_trial"] = lift_accepted
            if not lift_accepted:
                failed_reasons.append("micro lift IK target not reached")
            if not micro_lift_summary.get("object_lift_passed"):
                failed_reasons.append(
                    "object did not lift with the gripper "
                    f"({float(micro_lift_summary.get('object_lift_m', 0.0)):.4f} m)"
                )
        return failed_reasons
