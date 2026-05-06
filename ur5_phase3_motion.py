"""Phase 3 deterministic motion helpers built on locked geometry helpers."""

from __future__ import annotations

import math
from typing import Any

import torch

from isaaclab.assets import Articulation
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import quat_apply, quat_apply_inverse, subtract_frame_transforms

from ur5_phase1_control import (
    ARM_JOINT_NAMES,
    GRIPPER_CONTROL_JOINT_NAMES,
    build_gripper_joint_target,
    build_pregrasp_joint_target,
    build_reset_joint_target,
    gripper_joint_overrides_rad,
    named_joint_ids,
)
from ur5_phase1_reset import _stabilize_gripper_mimic_state, reset_robot
from ur5_phase3_geometry import Phase3GeometryMixin, _list_tensor
from ur5_phase3_logging import Phase3TrialLogger
from ur5_phase3_schema import utc_now_iso

PHASE3_DEFAULT_PREGRASP_WRIST3_DEG = -2.6
TACTILE_FINGERTIP_BODY_BY_SIDE = {
    "left": "left_inner_finger",
    "right": "right_inner_finger",
}
TACTILE_CONTACT_PROXY_ROOT = "/World/Phase3TactileContactProxy"
TACTILE_CONTACT_PROXY_DISTANCE_M = 0.026
TACTILE_CONTACT_PROXY_SCALE = (0.010, 0.010, 0.010)
TACTILE_CONTACT_PROXY_AXES = (
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0),
)


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
    from ur5_phase1_control import build_joint_target

    overrides = gripper_joint_overrides_rad(closed=True)
    sign_by_joint = {joint_name: 1.0 if value >= 0.0 else -1.0 for joint_name, value in overrides.items()}
    custom = {joint_name: sign_by_joint[joint_name] * close_rad for joint_name in overrides}
    return build_joint_target(robot, custom, base_target=base_target)


class Phase3MotionMixin(Phase3GeometryMixin):
    def _delete_tactile_contact_proxies(self) -> None:
        import isaacsim.core.utils.prims as prim_utils

        if prim_utils.is_prim_path_valid(TACTILE_CONTACT_PROXY_ROOT):
            prim_utils.delete_prim(TACTILE_CONTACT_PROXY_ROOT)

    def _set_proxy_cube_pose(self, prim_path: str, position: Any) -> None:
        import isaacsim.core.utils.prims as prim_utils
        from pxr import Gf, UsdGeom
        import omni.usd

        if not prim_utils.is_prim_path_valid(prim_path):
            prim_utils.create_prim(
                prim_path,
                "Cube",
                translation=(float(position[0]), float(position[1]), float(position[2])),
                scale=TACTILE_CONTACT_PROXY_SCALE,
            )
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))
        )
        xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*TACTILE_CONTACT_PROXY_SCALE))

    def _update_tactile_contact_proxies(self, contact_state: dict[str, Any]) -> dict[str, Any]:
        """Spawn camera-visible, non-physics contact witnesses for TacEx.

        IsaacLab PhysX body poses and USD camera render products do not expose
        the canonical URDF GSmini collision contact directly to TacEx's internal
        depth camera.  Phase2 already used a camera-visible probe to prove the
        GSmini optical stack.  Phase3 gates this probe by the *actual* contact
        state so tactile_rgb changes only when the scripted grasp has reached
        contact/hold.
        """

        import isaacsim.core.utils.prims as prim_utils
        import omni.usd
        from pxr import Gf, Usd, UsdGeom
        from ur5_phase2_mount import phase2_sensor_prim_paths

        contact_sides = set(contact_state.get("contact_sides", []))
        if not contact_sides:
            self._delete_tactile_contact_proxies()
            self._last_tactile_contact_proxy = {"enabled": False, "reason": "no_contact"}
            return self._last_tactile_contact_proxy

        if not prim_utils.is_prim_path_valid(TACTILE_CONTACT_PROXY_ROOT):
            prim_utils.create_prim(TACTILE_CONTACT_PROXY_ROOT, "Xform")
        stage = omni.usd.get_context().get_stage()
        summary: dict[str, Any] = {"enabled": True, "distance_m": TACTILE_CONTACT_PROXY_DISTANCE_M, "sides": {}}
        for side, paths in phase2_sensor_prim_paths().items():
            side_root = f"{TACTILE_CONTACT_PROXY_ROOT}/{side}"
            if side not in contact_sides:
                if prim_utils.is_prim_path_valid(side_root):
                    prim_utils.delete_prim(side_root)
                continue
            if not prim_utils.is_prim_path_valid(side_root):
                prim_utils.create_prim(side_root, "Xform")
            camera_prim = stage.GetPrimAtPath(paths["camera"])
            if not camera_prim.IsValid():
                continue
            camera_xform = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            camera_pos = camera_xform.ExtractTranslation()
            positions = []
            for index, axis_tuple in enumerate(TACTILE_CONTACT_PROXY_AXES):
                axis = Gf.Vec3d(*axis_tuple)
                direction = camera_xform.TransformDir(axis)
                direction.Normalize()
                # Small lateral staggering avoids exact overlap while keeping
                # all probes close to the TacEx optical contact distance.
                lateral = (index - 2.5) * 0.0004
                proxy_pos = camera_pos + direction * TACTILE_CONTACT_PROXY_DISTANCE_M + Gf.Vec3d(lateral, 0.0, 0.0)
                child_path = f"{side_root}/probe_{index}"
                self._set_proxy_cube_pose(child_path, proxy_pos)
                positions.append([float(proxy_pos[0]), float(proxy_pos[1]), float(proxy_pos[2])])
            summary["sides"][side] = {
                "camera": paths["camera"],
                "probes": positions,
            }
        self._last_tactile_contact_proxy = summary
        return summary

    @staticmethod
    def _tactile_scene_signature(contact_proxy: dict[str, Any]) -> tuple[Any, ...]:
        """Return a coarse renderer-scene signature for TacEx sensor refreshes.

        Phase2 rebuilds the GelSight sensor after adding its camera-visible
        contact probe because a standalone TiledCamera view can otherwise keep
        returning the pre-probe render product.  Phase3 keeps the same rule but
        only keys on contact-proxy presence/side set; moving an already-visible
        proxy should be picked up by render ticks without recreating sensors on
        every logged sample.
        """

        if not contact_proxy.get("enabled"):
            return ("no_contact_proxy",)
        sides = tuple(sorted(contact_proxy.get("sides", {}).keys()))
        return ("contact_proxy", sides)

    def _refresh_tactile_sensors_for_scene_signature(self, signature: tuple[Any, ...]) -> dict[str, Any]:
        if signature == self._last_tactile_scene_signature:
            return {"refreshed": False, "signature": list(signature)}
        if not getattr(self, "tactile_sensor_cfgs", None):
            self._last_tactile_scene_signature = signature
            return {"refreshed": False, "signature": list(signature), "reason": "missing_sensor_cfgs"}

        from ur5_phase2_tactile import initialize_phase2_sensor

        refreshed: list[str] = []
        rebuilt: list[tuple[str, Any]] = []
        for side, sensor in self.tactile_sensors:
            cfg = self.tactile_sensor_cfgs.get(side)
            if cfg is None:
                rebuilt.append((side, sensor))
                continue
            rebuilt.append((side, initialize_phase2_sensor(cfg)))
            refreshed.append(side)
        self.tactile_sensors = rebuilt
        self._last_tactile_scene_signature = signature
        return {
            "refreshed": bool(refreshed),
            "signature": list(signature),
            "sides": refreshed,
        }

    def _delete_phase2_static_contact_probe(self) -> None:
        import isaacsim.core.utils.prims as prim_utils
        from ur5_phase2_tactile import STATIC_CONTACT_PROBE_PATH

        if prim_utils.is_prim_path_valid(STATIC_CONTACT_PROBE_PATH):
            prim_utils.delete_prim(STATIC_CONTACT_PROBE_PATH)

    def _set_tactile_shell_world_pose(self, prim_path: str, position: torch.Tensor, orientation: torch.Tensor) -> None:
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"Phase3 tactile shell prim is missing: {prim_path}")
        pos = [float(value) for value in position.detach().cpu().reshape(-1).tolist()]
        quat = [float(value) for value in orientation.detach().cpu().reshape(-1).tolist()]
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        # The TacEx camera shells are runtime sensor-only prims.  They must
        # follow IsaacLab's PhysX articulation body poses, not stale USD child
        # transforms under the URDF link.  Resetting the inherited xform stack
        # lets Phase3 write the exact world pose every capture.
        xform.SetResetXformStack(True)
        xform.AddTranslateOp().Set(Gf.Vec3d(*pos))
        xform.AddOrientOp().Set(Gf.Quatf(quat[0], Gf.Vec3f(quat[1], quat[2], quat[3])))

    def _sync_tactile_sensor_shells_to_runtime_fingers(self) -> dict[str, Any]:
        if not self.tactile_sensors:
            return {"enabled": False, "reason": "no_tactile_sensors"}

        from ur5_phase2_mount import (
            CASE_LOCAL_QUAT_WXYZ,
            CASE_LOCAL_TRANSLATION,
            CONNECTOR_LOCAL_QUAT_WXYZ,
            CONNECTOR_LOCAL_TRANSLATION,
            GELPAD_LOCAL_QUAT_WXYZ,
            GELPAD_LOCAL_TRANSLATION,
            phase2_sensor_prim_paths,
        )

        pose_summary: dict[str, Any] = {"enabled": True, "sides": {}}
        device = self.robot.device
        dtype = self.robot.data.body_pose_w.dtype
        for side, _sensor in self.tactile_sensors:
            body_name = TACTILE_FINGERTIP_BODY_BY_SIDE[side]
            body_id = int(self.robot.find_bodies([body_name], preserve_order=True)[0][0])
            fingertip_pose = self.robot.data.body_pose_w[:, body_id]
            fingertip_pos = fingertip_pose[:, 0:3]
            fingertip_quat = fingertip_pose[:, 3:7]
            connector_pos, connector_quat = math_utils.combine_frame_transforms(
                fingertip_pos,
                fingertip_quat,
                torch.tensor([CONNECTOR_LOCAL_TRANSLATION], device=device, dtype=dtype),
                torch.tensor([CONNECTOR_LOCAL_QUAT_WXYZ], device=device, dtype=dtype),
            )
            case_pos, case_quat = math_utils.combine_frame_transforms(
                connector_pos,
                connector_quat,
                torch.tensor([CASE_LOCAL_TRANSLATION], device=device, dtype=dtype),
                torch.tensor([CASE_LOCAL_QUAT_WXYZ], device=device, dtype=dtype),
            )
            gelpad_pos, gelpad_quat = math_utils.combine_frame_transforms(
                case_pos,
                case_quat,
                torch.tensor([GELPAD_LOCAL_TRANSLATION], device=device, dtype=dtype),
                torch.tensor([GELPAD_LOCAL_QUAT_WXYZ], device=device, dtype=dtype),
            )
            paths = phase2_sensor_prim_paths()[side]
            self._set_tactile_shell_world_pose(paths["case"], case_pos[0], case_quat[0])
            self._set_tactile_shell_world_pose(paths["gelpad"], gelpad_pos[0], gelpad_quat[0])
            pose_summary["sides"][side] = {
                "finger_body": body_name,
                "sensor_case_world_m": _list_tensor(case_pos[0]),
                "sensor_gelpad_world_m": _list_tensor(gelpad_pos[0]),
            }
        self._last_tactile_pose_sync = pose_summary
        return pose_summary

    @staticmethod
    def _tactile_rgb_tensor(output: dict[str, Any]) -> torch.Tensor | None:
        frame = output.get("tactile_rgb")
        if frame is None or not hasattr(frame, "detach"):
            return None
        return frame.detach()

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def _clamp_unit(value: float, limit: float = 0.60) -> float:
        return max(-limit, min(limit, float(value)))

    def _continuous_imprint_depth_mm(self, side_geometry: dict[str, Any], contact_state: dict[str, Any], side: str) -> float:
        min_depth = max(0.01, float(self.options.tactile_imprint_min_depth_mm))
        max_depth = max(min_depth, float(self.options.tactile_imprint_max_depth_mm))
        margin_m = float(side_geometry.get("contact_margin_m", 0.0) or 0.0)
        min_overlap_m = float(side_geometry.get("soft_mesh_aabb_min_overlap_m", -margin_m) or -margin_m)
        overlap_inside_margin_mm = max(0.0, min_overlap_m + margin_m) * 1000.0
        geometry_depth = min_depth + max(0.0, float(self.options.tactile_imprint_depth_per_mm_overlap)) * overlap_inside_margin_mm

        force_by_side = contact_state.get("force_by_side_n", {}) if isinstance(contact_state, dict) else {}
        force_n = max(0.0, float(force_by_side.get(side, 0.0) or 0.0))
        force_depth = 0.0
        if self.object_profile.stable_force_threshold_n > 0.0:
            force_depth = min_depth * min(2.0, force_n / self.object_profile.stable_force_threshold_n)
        return max(min_depth, min(max_depth, geometry_depth + force_depth))

    def _continuous_imprint_center_xy(self, side_geometry: dict[str, Any], contact_state: dict[str, Any]) -> tuple[float, float]:
        geometry = contact_state.get("geometry", {}) if isinstance(contact_state, dict) else {}
        object_center = geometry.get("object_aabb", {}).get("center_world_m")
        soft_center = side_geometry.get("center_world_m")
        if not object_center or not soft_center:
            return (0.0, 0.0)
        # Keep this deliberately heuristic and bounded: the goal is to preserve a
        # continuous, contact-relative patch inside the TacEx/Taxim optical model,
        # not to claim a calibrated pixel-to-world registration.
        lateral_scale_m = 0.035
        vertical_scale_m = 0.035
        center_x = self._clamp_unit((float(object_center[1]) - float(soft_center[1])) / lateral_scale_m)
        center_y = self._clamp_unit(-(float(object_center[2]) - float(soft_center[2])) / vertical_scale_m)
        return (center_x, center_y)

    def _apply_continuous_taxim_imprint(
        self,
        sensor: Any,
        output: dict[str, Any],
        *,
        side: str,
        side_geometry: dict[str, Any],
        contact_state: dict[str, Any],
    ) -> dict[str, Any]:
        optical_sim = getattr(sensor, "optical_simulator", None)
        sensor_data = getattr(sensor, "_data", None)
        sensor_output = getattr(sensor_data, "output", None)
        if optical_sim is None or sensor_output is None or "height_map" not in sensor_output or "tactile_rgb" not in output:
            return {"applied": False, "reason": "missing TacEx optical simulator, height_map, or tactile_rgb output"}

        height_map = sensor_output["height_map"]
        if not hasattr(height_map, "shape") or height_map.ndim != 3:
            return {"applied": False, "reason": f"unsupported height_map shape: {getattr(height_map, 'shape', None)}"}

        _, height, width = height_map.shape
        geometry = self._sensor_optical_simulator_geometry(sensor)
        base_mm = geometry["far_gelpad_surface_mm"]
        depth_mm = self._continuous_imprint_depth_mm(side_geometry, contact_state, side)
        center_x, center_y = self._continuous_imprint_center_xy(side_geometry, contact_state)
        device = height_map.device
        dtype = height_map.dtype
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype).reshape(1, height, 1)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype).reshape(1, 1, width)
        sigma_x = torch.tensor(max(0.05, float(self.options.tactile_imprint_sigma_x)), device=device, dtype=dtype)
        sigma_y = torch.tensor(max(0.05, float(self.options.tactile_imprint_sigma_y)), device=device, dtype=dtype)
        imprint = torch.exp(
            -0.5
            * (
                ((x - torch.tensor(center_x, device=device, dtype=dtype)) / sigma_x) ** 2
                + ((y - torch.tensor(center_y, device=device, dtype=dtype)) / sigma_y) ** 2
            )
        )
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
            "source": "phase3_continuous_soft_object_overlap_to_taxim_height_map",
            "side": side,
            "depth_mm": depth_mm,
            "center_xy": [center_x, center_y],
            "base_height_mm": base_mm,
            "height_map_shape": [int(height), int(width)],
            "soft_mesh_aabb_min_overlap_m": side_geometry.get("soft_mesh_aabb_min_overlap_m"),
            **geometry,
        }

    def _maybe_apply_continuous_tactile_imprint(
        self,
        side: str,
        sensor: Any,
        output: dict[str, Any],
        contact_state: dict[str, Any],
    ) -> dict[str, Any]:
        frame = self._tactile_rgb_tensor(output)
        stats = self._tactile_imprint_stats.setdefault(
            side,
            {
                "frames": 0,
                "baseline_frames": 0,
                "contact_frames": 0,
                "applied_frames": 0,
                "max_mean_abs_delta_before": 0.0,
                "max_mean_abs_delta_after": 0.0,
                "max_depth_mm": 0.0,
            },
        )
        stats["frames"] = int(stats.get("frames", 0)) + 1
        if frame is None:
            return {"applied": False, "reason": "missing tactile_rgb frame"}

        geometry = contact_state.get("geometry", {}) if isinstance(contact_state, dict) else {}
        side_geometry = geometry.get("sides", {}).get(side, {})
        contact_sides = set(contact_state.get("contact_sides", [])) if isinstance(contact_state, dict) else set()
        if not self.options.tactile_contact_imprint_enabled:
            self._tactile_imprint_baselines.setdefault(side, frame.detach().clone())
            return {"applied": False, "reason": "continuous imprint disabled by options"}
        if side not in contact_sides:
            self._tactile_imprint_baselines[side] = frame.detach().clone()
            stats["baseline_frames"] = int(stats.get("baseline_frames", 0)) + 1
            return {"applied": False, "reason": "no contact on this side"}

        stats["contact_frames"] = int(stats.get("contact_frames", 0)) + 1
        baseline = self._tactile_imprint_baselines.get(side)
        before_delta = self._tactile_rgb_mean_abs_delta(frame, baseline)
        if before_delta is not None:
            stats["max_mean_abs_delta_before"] = max(float(stats.get("max_mean_abs_delta_before", 0.0)), before_delta)

        result = self._apply_continuous_taxim_imprint(
            sensor,
            output,
            side=side,
            side_geometry=side_geometry,
            contact_state=contact_state,
        )
        after_frame = self._tactile_rgb_tensor(output)
        after_delta = self._tactile_rgb_mean_abs_delta(after_frame, baseline) if after_frame is not None else None
        if after_delta is not None:
            stats["max_mean_abs_delta_after"] = max(float(stats.get("max_mean_abs_delta_after", 0.0)), after_delta)
        if result.get("applied"):
            stats["applied_frames"] = int(stats.get("applied_frames", 0)) + 1
            stats["max_depth_mm"] = max(float(stats.get("max_depth_mm", 0.0)), float(result.get("depth_mm", 0.0) or 0.0))
        stats.update(
            {
                "last_contact": True,
                "last_contact_sides": sorted(contact_sides),
                "last_mean_abs_delta_before": before_delta,
                "last_mean_abs_delta_after": after_delta,
                "last_result": result,
            }
        )
        return result

    def _capture_tactile_outputs(self) -> dict[str, dict[str, Any]]:
        if not self.tactile_sensors:
            return {}
        from ur5_phase2_tactile import update_phase2_sensor

        outputs: dict[str, dict[str, Any]] = {}
        dt = self.sim.get_physics_dt()
        contact_sides = set(self._last_contact_state.get("contact_sides", []))
        self._delete_tactile_contact_proxies()
        self._delete_phase2_static_contact_probe()
        rebuilt_sensors: list[tuple[str, Any]] = []
        imprint_results: dict[str, Any] = {}
        for side, sensor in self.tactile_sensors:
            try:
                outputs[side] = dict(update_phase2_sensor(sensor, self.sim, dt=dt))
                imprint_results[side] = self._maybe_apply_continuous_tactile_imprint(
                    side,
                    sensor,
                    outputs[side],
                    self._last_contact_state,
                )
            except Exception as exc:  # pragma: no cover - Isaac runtime only
                outputs[side] = {"error": str(exc)}
                imprint_results[side] = {"applied": False, "error": str(exc)}
            rebuilt_sensors.append((side, sensor))
        self.tactile_sensors = rebuilt_sensors
        self._last_tactile_sensor_refresh = {
            "refreshed": False,
            "sides": [],
            "mode": "continuous_imprint_no_static_probe_rebuild",
        }
        self._last_tactile_contact_proxy = {
            "enabled": False,
            "mode": "disabled_replaced_by_continuous_taxim_contact_imprint",
        }
        self._last_tactile_imprint = {
            "enabled": self.options.tactile_contact_imprint_enabled,
            "contact_sides": sorted(contact_sides),
            "mode": "continuous_taxim_height_map",
            "results": imprint_results,
            "stats": self._tactile_imprint_stats,
        }
        self._last_tactile_outputs = outputs
        return outputs

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
        return target

    def _log_sample(self, logger: Phase3TrialLogger, stage: str, *, force_tactile: bool = False) -> None:
        contact_state = self._read_contact_state()
        tactile_outputs = self._capture_tactile_outputs() if (force_tactile or self.tactile_sensors) else {}
        contact_state = dict(contact_state)
        contact_state["tactile_imprint"] = self._last_tactile_imprint
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
        if gripper_joint_target is not None:
            _stabilize_gripper_mimic_state(self.robot, gripper_joint_target)

        self.sim.step()
        dt = self.sim.get_physics_dt()
        self.robot.update(dt)
        self.grasp_object.update(dt)
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
        sample_every_steps = (
            self.options.tactile_sample_every_steps
            if self.options.tactile_sample_every_steps is not None
            else self.options.sample_every_steps
        )
        if force_log or (sample_every_steps > 0 and self.budget.steps % sample_every_steps == 0):
            self._log_sample(logger, stage, force_tactile=force_log)

    def _resolve_ik_indices(self) -> tuple[list[int], list[int], int, int]:
        arm_joint_ids = list(named_joint_ids(self.robot, ARM_JOINT_NAMES).values())
        gripper_joint_ids = list(named_joint_ids(self.robot, GRIPPER_CONTROL_JOINT_NAMES).values())
        ee_body = int(self.robot.find_bodies(["ee_link"], preserve_order=True)[0][0])
        ee_jacobian_index = ee_body - 1 if self.robot.is_fixed_base else ee_body
        return arm_joint_ids, gripper_joint_ids, ee_body, ee_jacobian_index

    def _ee_pose(self, ee_body: int) -> tuple[torch.Tensor, torch.Tensor]:
        pose = self.robot.data.body_pose_w[:, ee_body]
        return pose[:, 0:3].detach().clone(), pose[:, 3:7].detach().clone()

    def _slow_joint_move(self, logger: Phase3TrialLogger, target: torch.Tensor, *, stage: str, steps: int) -> None:
        start = self.robot.data.joint_pos.detach().clone()
        for index in range(max(1, steps)):
            if self.budget.exhausted:
                break
            alpha = float(index + 1) / float(max(1, steps))
            blend = start + (target - start) * alpha
            self.robot.set_joint_position_target(blend)
            self.robot.write_data_to_sim()
            _stabilize_gripper_mimic_state(self.robot, blend)
            self.sim.step()
            dt = self.sim.get_physics_dt()
            self.robot.update(dt)
            self.grasp_object.update(dt)
            self.budget.tick()
            sample_every_steps = (
                self.options.tactile_sample_every_steps
                if self.options.tactile_sample_every_steps is not None
                else self.options.sample_every_steps
            )
            if sample_every_steps > 0 and self.budget.steps % sample_every_steps == 0:
                self._log_sample(logger, stage)

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
        controller.reset()
        controller.set_command(torch.cat((target_pos_b, target_quat_b), dim=1))
        final_error = math.inf
        final_pos_w: torch.Tensor | None = None
        final_step = 0
        for step in range(1, max_steps + 1):
            if self.budget.exhausted:
                break
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
            ee_pos_w, _ = self._ee_pose(ee_body)
            final_pos_w = ee_pos_w.detach().clone()
            final_error = float(torch.linalg.norm(target_pos_w - ee_pos_w).item())
            final_step = step
            if final_error <= pos_tolerance:
                break
        return {
            "passed": final_error <= pos_tolerance,
            "steps": final_step,
            "target_position_world_m": _list_tensor(target_pos_w[0]),
            "final_position_world_m": _list_tensor(final_pos_w[0]) if final_pos_w is not None else None,
            "final_position_error_m": final_error,
            "pos_tolerance_m": pos_tolerance,
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
        self.robot.write_joint_state_to_sim(arm_joint_target, torch.zeros_like(arm_joint_target), joint_ids=arm_joint_ids)
        self.robot.set_joint_position_target(arm_joint_target, joint_ids=arm_joint_ids)
        self.robot.write_data_to_sim()
        return arm_joint_target, {"latched": True, "target_rad": _list_tensor(arm_joint_target[0])}

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
        max_force_seen_n = 0.0
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
            contact_sides = set(contact_state.get("contact_sides", []))
            if {"left", "right"}.issubset(contact_sides):
                bilateral_contact_detected = True
                if bilateral_contact_step is None:
                    bilateral_contact_step = index + 1
            both_sides_stable = bool(force_by_side and all(float(force_by_side.get(side, 0.0)) >= self.object_profile.stable_force_threshold_n for side in ("left", "right")))
            stable_counter = stable_counter + 1 if both_sides_stable else 0
            max_force_n = float(contact_state.get("max_force_n", 0.0) or 0.0)
            max_force_seen_n = max(max_force_seen_n, max_force_n)
            object_lift_m = float(self.grasp_object.data.root_pose_w[0, 2].item()) - object_z0
            max_object_lift_m = max(max_object_lift_m, object_lift_m)
            force_history.append({
                "step": index + 1,
                "planned_close_rad": planned_close_rad,
                "contact_sides": sorted(contact_sides),
                "force_by_side_n": force_by_side,
                "max_force_n": max_force_n,
                "both_sides_stable": both_sides_stable,
                "bilateral_contact_detected": bilateral_contact_detected,
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
            "max_force_seen_n": max_force_seen_n,
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
            max_steps=max(20, self.options.direct_move_steps // 4),
            pos_tolerance=self.options.direct_pos_tolerance_m,
            max_joint_delta=self.options.max_joint_delta_per_step,
        )
        final_object_z = float(self.grasp_object.data.root_pose_w[0, 2].item())
        object_lift_m = final_object_z - object_z0
        min_required_object_lift_m = max(0.0015, min(0.004, distance * 0.25))
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
        direct_error = float(direct_move.get("final_position_error_m", math.inf) or math.inf)
        direct_accepted = bool(direct_move.get("passed")) or direct_error <= self.options.direct_pass_tolerance_m
        direct_move["accepted_pass_tolerance_m"] = self.options.direct_pass_tolerance_m
        direct_move["accepted_for_trial"] = direct_accepted
        if not direct_accepted and not contact_onset.get("detected"):
            failed_reasons.append("direct pre-grasp IK target not reached")
        if not contact_onset.get("detected"):
            failed_reasons.append("contact_onset not detected")
        if not close_summary.get("bilateral_contact_detected"):
            failed_reasons.append("bilateral GSmini contact not detected during close")
        if close_summary.get("stopped_by_high_force"):
            failed_reasons.append("close stopped by high force")
        if close_summary.get("stopped_by_object_lift"):
            failed_reasons.append("close stopped by object lift guard")
        if not micro_lift_summary.get("enabled"):
            failed_reasons.append("micro lift stage not enabled by protocol")
        else:
            lift_move = micro_lift_summary.get("move", {})
            lift_error = float(lift_move.get("final_position_error_m", math.inf) or math.inf)
            lift_accepted = bool(lift_move.get("passed")) or lift_error <= self.options.direct_pass_tolerance_m
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
