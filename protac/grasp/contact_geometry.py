# mypy: disable-error-code=attr-defined
"""Phase 3 object geometry and contact-state helpers."""

from __future__ import annotations

from typing import Any

import torch

from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils.math import quat_apply

from protac.robot.control import GRIPPER_CONTROL_JOINT_NAMES, named_joint_ids
from protac.scene.object_profiles import Phase3ObjectProfile

SOFT_CONTACT_BODY_BY_SIDE = {
    "left": "left_gelsight_mini_gelpad",
    "right": "right_gelsight_mini_gelpad",
}
ADAPTOR_CONTACT_BODY_BY_SIDE = {
    "left": "left_gelsight_connector",
    "right": "right_gelsight_connector",
}
ADAPTOR_CONTACT_FORCE_TOLERANCE_N = 0.05
GSMINI_SOFT_MESH_AABB_IN_SENSOR_FRAME_M = {
    "min": (-0.01363918, -0.02524673, -0.04923395),
    "max": (0.01022479, -0.01999673, -0.02100601),
}
GSMINI_SOFT_AABB_CONTACT_MARGIN_M = 0.002


def phase3_object_prim_path(profile: Phase3ObjectProfile, *, origin_index: int = 1) -> str:
    return f"/World/Origin{origin_index}/{profile.prim_name}"


def _make_object_filtered_contact_sensor(
    origin_prim: str,
    body_name: str,
    object_path: str,
) -> ContactSensor:
    return ContactSensor(
        ContactSensorCfg(
            prim_path=f"{origin_prim}/Robot/{body_name}",
            update_period=0.0,
            history_length=3,
            debug_vis=False,
            filter_prim_paths_expr=[object_path],
        )
    )


def setup_phase3_contact_sensors(
    profile: Phase3ObjectProfile,
    *,
    origin_index: int = 1,
    disabled: bool = False,
) -> dict[str, dict[str, ContactSensor]]:
    if disabled:
        return {}
    origin_prim = f"/World/Origin{origin_index}"
    object_path = phase3_object_prim_path(profile, origin_index=origin_index)
    sensor_body_groups = {
        "gelpad": SOFT_CONTACT_BODY_BY_SIDE,
        "adaptor": ADAPTOR_CONTACT_BODY_BY_SIDE,
    }
    return {
        group: {
            side: _make_object_filtered_contact_sensor(origin_prim, body_name, object_path)
            for side, body_name in body_by_side.items()
        }
        for group, body_by_side in sensor_body_groups.items()
    }


def _list_tensor(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().cpu().reshape(-1).tolist()]


def _aabb_corners(
    min_xyz: tuple[float, float, float],
    max_xyz: tuple[float, float, float],
    *,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
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


class Phase3GeometryMixin:
    @staticmethod
    def _filtered_contact_force(sensor: ContactSensor, dt: float) -> tuple[float, str | None]:
        try:
            if not sensor.is_initialized:
                raise RuntimeError("contact sensor is not initialized yet")
            sensor.update(dt, force_recompute=True)
            force_tensor = getattr(sensor.data, "force_matrix_w", None)
            if force_tensor is None:
                force_tensor = sensor.data.net_forces_w
            if force_tensor is None or force_tensor.numel() == 0:
                return 0.0, None
            force = float(torch.linalg.norm(force_tensor.reshape(-1, 3), dim=-1).max().item())
            return force, None
        except Exception as exc:  # pragma: no cover - Isaac runtime only
            return 0.0, f"{type(exc).__name__}: {exc}"

    def _read_contact_sensor_group(
        self,
        sensors: dict[str, ContactSensor],
        dt: float,
        contact_threshold_n: float,
        *,
        threshold_is_inclusive: bool,
    ) -> tuple[dict[str, float], list[str], dict[str, str]]:
        force_by_side: dict[str, float] = {}
        contact_sides: list[str] = []
        errors: dict[str, str] = {}
        for side, sensor in sensors.items():
            force, error = self._filtered_contact_force(sensor, dt)
            force_by_side[side] = force
            if error:
                errors[side] = error
            force_is_contact = force >= contact_threshold_n
            if not threshold_is_inclusive:
                force_is_contact = force > contact_threshold_n
            if force_is_contact:
                contact_sides.append(side)
        return force_by_side, contact_sides, errors

    def _joint_state(self) -> dict[str, Any]:
        names = list(getattr(self.robot, "joint_names", []))
        pos = self.robot.data.joint_pos[0].detach().cpu().tolist()
        vel = self.robot.data.joint_vel[0].detach().cpu().tolist()
        if not names:
            names = [f"joint_{index}" for index in range(len(pos))]
        return {
            "position_rad": {name: float(pos[index]) for index, name in enumerate(names[: len(pos)])},
            "velocity_rad_s": {name: float(vel[index]) for index, name in enumerate(names[: len(vel)])},
        }

    def _gripper_state(self) -> dict[str, Any]:
        joint_id_map = named_joint_ids(self.robot, GRIPPER_CONTROL_JOINT_NAMES)
        joints = {
            joint_name: float(self.robot.data.joint_pos[0, joint_id].item())
            for joint_name, joint_id in joint_id_map.items()
        }
        return {
            "control_joint_position_rad": joints,
            "finger_joint_abs_rad": abs(float(joints.get("finger_joint", 0.0))),
        }

    def _object_state(self) -> dict[str, Any]:
        pose = self.grasp_object.data.root_pose_w[0, :7]
        return {
            "root_pose_wxyz_m": _list_tensor(pose),
            "root_position_m": _list_tensor(pose[:3]),
            "root_quat_wxyz": _list_tensor(pose[3:7]),
            "grasp_center_world_m": _list_tensor(self._object_grasp_center_world()),
            "world_aabb": self._object_world_aabb()["summary"],
        }

    def _object_world_aabb(self) -> dict[str, Any]:
        pose_w = self.grasp_object.data.root_pose_w[:, :7]
        local = _aabb_corners(
            self.object_profile.local_aabb.min_m,
            self.object_profile.local_aabb.max_m,
            device=self.grasp_object.device,
            dtype=self.grasp_object.data.root_pose_w.dtype,
        )
        world = _transform_local_corners(pose_w, local)
        min_w = world.min(dim=0).values
        max_w = world.max(dim=0).values
        center_w = (min_w + max_w) / 2.0
        return {
            "min": min_w,
            "max": max_w,
            "center": center_w,
            "summary": {
                "min_world_m": _list_tensor(min_w),
                "max_world_m": _list_tensor(max_w),
                "center_world_m": _list_tensor(center_w),
                "local_aabb_m": {
                    "min": list(self.object_profile.local_aabb.min_m),
                    "max": list(self.object_profile.local_aabb.max_m),
                },
            },
        }

    def _object_grasp_center_world(self) -> torch.Tensor:
        pose_w = self.grasp_object.data.root_pose_w[:, :7]
        local = torch.tensor(
            self.object_profile.grasp_center_local_m,
            device=self.grasp_object.device,
            dtype=self.grasp_object.data.root_pose_w.dtype,
        ).reshape(1, 3)
        return (pose_w[:, 0:3] + quat_apply(pose_w[:, 3:7], local))[0].detach().clone()

    def _soft_link_mesh_aabbs_world(self) -> dict[str, dict[str, Any]]:
        local_corners = _aabb_corners(
            GSMINI_SOFT_MESH_AABB_IN_SENSOR_FRAME_M["min"],
            GSMINI_SOFT_MESH_AABB_IN_SENSOR_FRAME_M["max"],
            device=self.robot.device,
            dtype=self.robot.data.body_pose_w.dtype,
        )
        aabbs: dict[str, dict[str, Any]] = {}
        for side, body_name in SOFT_CONTACT_BODY_BY_SIDE.items():
            body_id = int(self.robot.find_bodies([body_name], preserve_order=True)[0][0])
            pose = self.robot.data.body_pose_w[:, body_id]
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
                    "min_world_m": _list_tensor(min_w),
                    "max_world_m": _list_tensor(max_w),
                    "center_world_m": _list_tensor(center_w),
                },
            }
        return aabbs

    def _soft_mesh_pair_center_world(self) -> torch.Tensor:
        aabbs = self._soft_link_mesh_aabbs_world()
        return torch.stack([aabbs[side]["center"] for side in ("left", "right")], dim=0).mean(dim=0)

    def _soft_contact_geometry(self) -> dict[str, Any]:
        object_aabb = self._object_world_aabb()
        soft_aabbs = self._soft_link_mesh_aabbs_world()
        sides: dict[str, Any] = {}
        contact_sides: list[str] = []
        for side, soft_aabb in soft_aabbs.items():
            overlaps = torch.minimum(soft_aabb["max"], object_aabb["max"]) - torch.maximum(
                soft_aabb["min"],
                object_aabb["min"],
            )
            separated = torch.maximum(object_aabb["min"] - soft_aabb["max"], soft_aabb["min"] - object_aabb["max"])
            positive_sep = torch.clamp(separated, min=0.0)
            distance = float(torch.linalg.norm(positive_sep).item())
            overlap_with_margin = bool(torch.all(overlaps >= -GSMINI_SOFT_AABB_CONTACT_MARGIN_M).item())
            if overlap_with_margin:
                contact_sides.append(side)
            sides[side] = {
                **soft_aabb["summary"],
                "soft_mesh_aabb_overlap": overlap_with_margin,
                "soft_mesh_aabb_min_overlap_m": float(overlaps.min().item()),
                "soft_mesh_aabb_distance_m": 0.0 if overlap_with_margin else distance,
                "contact_margin_m": GSMINI_SOFT_AABB_CONTACT_MARGIN_M,
            }
        return {
            "object_aabb": object_aabb["summary"],
            "sides": sides,
            "contact_sides": contact_sides,
        }

    def _read_contact_state(self) -> dict[str, Any]:
        geometry = self._soft_contact_geometry()
        force_by_side: dict[str, float] = {}
        adaptor_force_by_side: dict[str, float] = {}
        force_contact_sides: list[str] = []
        adaptor_contact_sides: list[str] = []
        sensor_errors: dict[str, dict[str, str]] = {"gelpad": {}, "adaptor": {}}
        if self.contact_sensors:
            dt = self.sim.get_physics_dt()
            force_by_side, force_contact_sides, sensor_errors["gelpad"] = self._read_contact_sensor_group(
                self.contact_sensors.get("gelpad", {}),
                dt,
                self.object_profile.stable_force_threshold_n,
                threshold_is_inclusive=True,
            )
            (
                adaptor_force_by_side,
                adaptor_contact_sides,
                sensor_errors["adaptor"],
            ) = self._read_contact_sensor_group(
                self.contact_sensors.get("adaptor", {}),
                dt,
                ADAPTOR_CONTACT_FORCE_TOLERANCE_N,
                threshold_is_inclusive=False,
            )
        contact_sides = sorted(set(geometry["contact_sides"]) | set(force_contact_sides))
        adaptor_sensor_valid = (
            set(adaptor_force_by_side) == {"left", "right"}
            and not sensor_errors["adaptor"]
        )
        max_adaptor_force_n = max(adaptor_force_by_side.values(), default=0.0)
        self._max_adaptor_force_seen_n = max(
            float(getattr(self, "_max_adaptor_force_seen_n", 0.0)),
            max_adaptor_force_n,
        )
        self._adaptor_contact_detected = bool(
            getattr(self, "_adaptor_contact_detected", False) or adaptor_contact_sides
        )
        self._adaptor_sensor_error_detected = bool(
            getattr(self, "_adaptor_sensor_error_detected", False) or not adaptor_sensor_valid
        )
        state = {
            "enabled": bool(self.contact_sensors),
            "contact_detected": bool(contact_sides),
            "contact_sides": contact_sides,
            "geometry_contact_sides": geometry["contact_sides"],
            "force_contact_sides": force_contact_sides,
            "force_by_side_n": force_by_side,
            "max_force_n": max(force_by_side.values()) if force_by_side else 0.0,
            "both_sides_force_contact": {"left", "right"}.issubset(set(force_contact_sides)),
            "adaptor_force_by_side_n": adaptor_force_by_side,
            "adaptor_contact_sides": adaptor_contact_sides,
            "max_adaptor_force_n": max_adaptor_force_n,
            "adaptor_contact_force_tolerance_n": ADAPTOR_CONTACT_FORCE_TOLERANCE_N,
            "adaptor_sensor_valid": adaptor_sensor_valid,
            "contact_sensor_errors": sensor_errors,
            "stable_force_threshold_n": self.object_profile.stable_force_threshold_n,
            "high_force_threshold_n": self.object_profile.high_force_threshold_n,
            "geometry": geometry,
        }
        self._last_contact_state = state
        return state
