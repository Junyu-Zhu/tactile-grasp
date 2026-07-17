"""Locked Phase 3 object set and protocol profiles.

The static profile data in this module is import-safe outside Isaac.  IsaacLab
scene construction is kept inside functions so schema/profile checks can run in
normal Python without launching SimulationApp.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[2]

# Copied from the Phase 1/2 table convention to keep this profile module
# dependency-light.  The scene builder below still reuses Phase 1 scene helpers
# after Isaac has been launched.
TABLE_TOP_HEIGHT_M = 0.72
TABLE_TRANSLATION_M = (0.6, 0.0, -0.0234)


@dataclass(frozen=True)
class LocalAabb:
    min_m: tuple[float, float, float]
    max_m: tuple[float, float, float]

    @property
    def size_m(self) -> tuple[float, float, float]:
        return (
            self.max_m[0] - self.min_m[0],
            self.max_m[1] - self.min_m[1],
            self.max_m[2] - self.min_m[2],
        )

    @property
    def center_m(self) -> tuple[float, float, float]:
        return (
            (self.min_m[0] + self.max_m[0]) / 2.0,
            (self.min_m[1] + self.max_m[1]) / 2.0,
            (self.min_m[2] + self.max_m[2]) / 2.0,
        )


def _scaled_aabb(
    min_m: tuple[float, float, float],
    max_m: tuple[float, float, float],
    scale: float | tuple[float, float, float],
) -> LocalAabb:
    if isinstance(scale, tuple):
        sx, sy, sz = scale
    else:
        sx = sy = sz = scale
    return LocalAabb(
        min_m=(min_m[0] * sx, min_m[1] * sy, min_m[2] * sz),
        max_m=(max_m[0] * sx, max_m[1] * sy, max_m[2] * sz),
    )


def _quat_apply_wxyz(
    quat: tuple[float, float, float, float],
    vec: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Rotate a vector by a wxyz quaternion without importing torch/Isaac."""

    w, x, y, z = quat
    vx, vy, vz = vec
    # q * v * q^-1, expanded for dependency-light profile calculations.
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


@dataclass(frozen=True)
class Phase3ObjectProfile:
    object_id: str
    label: str
    object_kind: str
    prim_name: str
    urdf_rel_path: str | None
    local_aabb: LocalAabb
    mass_kg: float
    root_rot_wxyz: tuple[float, float, float, float]
    grasp_center_local_m: tuple[float, float, float]
    close_rad: float
    stable_force_threshold_n: float
    high_force_threshold_n: float
    max_close_object_lift_m: float
    hold_seconds: float
    micro_lift_distance_m: float
    pregrasp_mode: str
    contact_approach_refine: bool
    tuning_note: str
    spawn_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    approach_offset_world_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    grasp_ee_world_pos_m: tuple[float, float, float] | None = None
    grasp_ee_world_quat_wxyz: tuple[float, float, float, float] | None = None
    fast_grasp_move_steps: int = 80

    @property
    def rest_root_z_m(self) -> float:
        min_xyz, max_xyz = self.local_aabb.min_m, self.local_aabb.max_m
        corners = (
            (min_xyz[0], min_xyz[1], min_xyz[2]),
            (min_xyz[0], min_xyz[1], max_xyz[2]),
            (min_xyz[0], max_xyz[1], min_xyz[2]),
            (min_xyz[0], max_xyz[1], max_xyz[2]),
            (max_xyz[0], min_xyz[1], min_xyz[2]),
            (max_xyz[0], min_xyz[1], max_xyz[2]),
            (max_xyz[0], max_xyz[1], min_xyz[2]),
            (max_xyz[0], max_xyz[1], max_xyz[2]),
        )
        rotated_min_z = min(_quat_apply_wxyz(self.root_rot_wxyz, corner)[2] for corner in corners)
        return TABLE_TOP_HEIGHT_M - rotated_min_z

    @property
    def root_position_m(self) -> tuple[float, float, float]:
        return (TABLE_TRANSLATION_M[0], TABLE_TRANSLATION_M[1], self.rest_root_z_m)

    @property
    def urdf_path(self) -> Path | None:
        return PROJECT_DIR / self.urdf_rel_path if self.urdf_rel_path else None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["local_aabb"]["size_m"] = list(self.local_aabb.size_m)
        payload["local_aabb"]["center_m"] = list(self.local_aabb.center_m)
        payload["root_position_m"] = list(self.root_position_m)
        payload["rest_root_z_m"] = self.rest_root_z_m
        payload["urdf_path"] = self.urdf_path.as_posix() if self.urdf_path else None
        return payload


@dataclass(frozen=True)
class Phase3ProtocolProfile:
    variant: str
    stages: tuple[str, ...]
    hold_seconds: float
    micro_lift_distance_m: float
    description: str

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["stages"] = list(self.stages)
        return payload


def _yaw_quat_wxyz(yaw_rad: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw_rad / 2.0), 0.0, 0.0, math.sin(yaw_rad / 2.0))


def _pitch_quat_wxyz(pitch_rad: float) -> tuple[float, float, float, float]:
    return (math.cos(pitch_rad / 2.0), 0.0, math.sin(pitch_rad / 2.0), 0.0)


CUBE_PROFILE = Phase3ObjectProfile(
    object_id="cube",
    label="4cm_cube",
    object_kind="cube",
    prim_name="GraspCube",
    urdf_rel_path=None,
    local_aabb=LocalAabb(min_m=(-0.02, -0.02, -0.02), max_m=(0.02, 0.02, 0.02)),
    mass_kg=0.08,
    root_rot_wxyz=(1.0, 0.0, 0.0, 0.0),
    grasp_center_local_m=(0.0, 0.0, 0.0),
    # The calibrated adaptor/GSmini mount needs about 0.42 rad before its soft
    # surfaces reach a 40 mm cube.  A 0.45 rad request gives the force-stop loop
    # This is a safe force-seeking travel limit, not a position that must be
    # reached.  The close loop stops earlier after stable bilateral force.  A
    # 0.45 rad limit only produced about 0.9/0.5 N in the measured Isaac run,
    # which is insufficient to support an 80 g cube at default friction.
    close_rad=0.55,
    stable_force_threshold_n=1.5,
    high_force_threshold_n=12.0,
    # The centered GSmini tips can raise the light cube by about 3.3 mm on the
    # final close step; keep the guard above that verified grasp transient but
    # well below the intentional 15 mm micro-lift.
    max_close_object_lift_m=0.0035,
    hold_seconds=2.0,
    micro_lift_distance_m=0.035,
    pregrasp_mode="phase2_side_pregrasp",
    contact_approach_refine=False,
    # Raise the GSmini soft-pair center by 4 mm from the previous baseline so
    # the soft faces, rather than the lower adaptor edge, carry the cube.  The
    # runtime adaptor contact sensors make this clearance an enforced result,
    # not a visual assumption.
    approach_offset_world_m=(0.0, 0.0, 0.010),
    # Verified pre-contact EE pose from the deterministic cube baseline.  A
    # straight Cartesian DLS approach avoids both the old AABB-refinement
    # oscillation and the object-sweeping arc of direct joint interpolation.
    grasp_ee_world_pos_m=(0.597046256, 0.000392442, 0.935164974),
    grasp_ee_world_quat_wxyz=(0.71106805, -0.00254552, 0.70311543, 0.00210681),
    fast_grasp_move_steps=240,
    tuning_note=(
        "Phase2-proven cube profile retuned for Phase3 visible grasp logging: side grasp at the "
        "cube center with a moderate close target, bilateral-contact success checks, and the "
        "default micro-lift protocol so a GUI run shows whether the cube is actually held."
    ),
)

_CHIPS_CAN_SCALE = 0.55
_CHIPS_CAN_AABB = _scaled_aabb(
    min_m=(-0.042753130197522, -0.025356302224099, 0.00570114748552),
    max_m=(0.03188893683255, 0.04928576480597299, 0.24733659625053),
    scale=_CHIPS_CAN_SCALE,
)

CHIPS_CAN_PROFILE = Phase3ObjectProfile(
    object_id="chips_can",
    label="YcbChipsCan",
    object_kind="urdf",
    prim_name="YcbChipsCan",
    urdf_rel_path="ycb_objects/YcbChipsCan/model.urdf",
    local_aabb=_CHIPS_CAN_AABB,
    mass_kg=0.10,
    # Lay the can on its side: local z (can height) becomes world x, while the
    # gripper still closes across the can's circular cross-section near center.
    root_rot_wxyz=_pitch_quat_wxyz(math.pi / 2.0),
    grasp_center_local_m=_CHIPS_CAN_AABB.center_m,
    close_rad=0.23,
    stable_force_threshold_n=0.45,
    high_force_threshold_n=6.0,
    max_close_object_lift_m=0.0035,
    hold_seconds=2.0,
    micro_lift_distance_m=0.012,
    pregrasp_mode="phase2_side_pregrasp",
    contact_approach_refine=False,
    tuning_note=(
        "Side-lying scaled can profile: shrink the can to 55% so the GSmini-padded Robotiq opening "
        "can surround the circular section with margin, rotate the can 90 degrees about local y so "
        "its axis lies along world x, and target the scaled local AABB center for a lift-capable side grasp."
    ),
    spawn_scale=(_CHIPS_CAN_SCALE, _CHIPS_CAN_SCALE, _CHIPS_CAN_SCALE),
)

_CRACKER_BOX_SCALE = 0.55
_CRACKER_BOX_AABB = _scaled_aabb(
    min_m=(-0.003362, -0.089567, -0.108358),
    max_m=(0.058277, 0.0679, 0.098405),
    scale=_CRACKER_BOX_SCALE,
)
_CRACKER_BOX_GRASP_CENTER = (
    _CRACKER_BOX_AABB.center_m[0],
    _CRACKER_BOX_AABB.center_m[1],
    _CRACKER_BOX_AABB.center_m[2] + 0.020,
)

CRACKER_BOX_PROFILE = Phase3ObjectProfile(
    object_id="cracker_box",
    label="YcbCrackerBox",
    object_kind="urdf",
    prim_name="YcbCrackerBox",
    urdf_rel_path="ycb_objects/YcbCrackerBox/model.urdf",
    local_aabb=_CRACKER_BOX_AABB,
    mass_kg=0.10,
    # Rotate the box 90 degrees about z so the thin local-x dimension is along
    # the gripper closing axis.  Without this, the world-y width is too large
    # for a Robotiq 2F-85 side grasp.
    root_rot_wxyz=_yaw_quat_wxyz(math.pi / 2.0),
    grasp_center_local_m=_CRACKER_BOX_GRASP_CENTER,
    close_rad=0.35,
    stable_force_threshold_n=0.35,
    high_force_threshold_n=7.0,
    max_close_object_lift_m=0.003,
    hold_seconds=2.5,
    micro_lift_distance_m=0.010,
    pregrasp_mode="phase2_side_pregrasp",
    contact_approach_refine=False,
    tuning_note=(
        "Scaled box profile: shrink the YCB cracker box to 55% so the Robotiq+GSmini opening can "
        "surround it, yaw-rotate the narrow side into the closing axis, and target an upper-mid "
        "scaled AABB grasp point and close farther so both GSmini pads engage before micro-lift; "
        "the upper-mid target keeps the arm out of the too-low side-grasp band observed in GUI runs "
        "while still pressing the box near its lift-capable side band."
    ),
    spawn_scale=(_CRACKER_BOX_SCALE, _CRACKER_BOX_SCALE, _CRACKER_BOX_SCALE),
    approach_offset_world_m=(0.0, 0.0, 0.002),
)

OBJECT_PROFILES: dict[str, Phase3ObjectProfile] = {
    profile.object_id: profile
    for profile in (CUBE_PROFILE, CHIPS_CAN_PROFILE, CRACKER_BOX_PROFILE)
}
OBJECT_SEQUENCE: tuple[str, ...] = ("cube", "chips_can", "cracker_box")

PROTOCOL_PROFILES: dict[str, Phase3ProtocolProfile] = {
    "contact_only": Phase3ProtocolProfile(
        variant="contact_only",
        stages=("reset", "pre_grasp", "contact_close", "release", "end_trial"),
        hold_seconds=0.0,
        micro_lift_distance_m=0.0,
        description="Close until contact evidence is logged, then release.",
    ),
    "contact_hold": Phase3ProtocolProfile(
        variant="contact_hold",
        stages=("reset", "pre_grasp", "contact_close", "hold", "release", "end_trial"),
        hold_seconds=2.0,
        micro_lift_distance_m=0.0,
        description="Recommended first stable variant: contact close plus fixed hold.",
    ),
    "contact_hold_micro_lift": Phase3ProtocolProfile(
        variant="contact_hold_micro_lift",
        stages=("reset", "pre_grasp", "contact_close", "hold", "micro_lift", "release", "end_trial"),
        hold_seconds=2.0,
        micro_lift_distance_m=0.01,
        description="Add a small deterministic lift after contact-hold succeeds.",
    ),
}


def get_object_profile(object_id: str) -> Phase3ObjectProfile:
    try:
        return OBJECT_PROFILES[object_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported Phase3 object_id={object_id!r}; choose one of {list(OBJECT_PROFILES)}") from exc


def get_protocol_profile(variant: str) -> Phase3ProtocolProfile:
    try:
        return PROTOCOL_PROFILES[variant]
    except KeyError as exc:
        raise ValueError(f"Unsupported Phase3 protocol variant={variant!r}; choose one of {list(PROTOCOL_PROFILES)}") from exc


def phase3_profiles_dict() -> dict[str, Any]:
    return {
        "object_sequence": list(OBJECT_SEQUENCE),
        "objects": {object_id: profile.as_dict() for object_id, profile in OBJECT_PROFILES.items()},
        "protocols": {variant: profile.as_dict() for variant, profile in PROTOCOL_PROFILES.items()},
    }


def write_phase3_profiles(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(phase3_profiles_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def make_phase3_object_cfg(profile: Phase3ObjectProfile, prim_path: str):
    """Build a RigidObjectCfg for the locked object profile.

    Importing IsaacLab here keeps static profile checks independent from the
    Isaac runtime.
    """

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    init_state = RigidObjectCfg.InitialStateCfg(pos=profile.root_position_m, rot=profile.root_rot_wxyz)
    common_rigid = sim_utils.RigidBodyPropertiesCfg(
        rigid_body_enabled=True,
        disable_gravity=False,
        linear_damping=1.0,
        angular_damping=2.0,
        max_linear_velocity=1.0,
        max_angular_velocity=180.0,
        max_depenetration_velocity=1.0,
        max_contact_impulse=50.0,
        enable_gyroscopic_forces=False,
        retain_accelerations=True,
        solver_position_iteration_count=32,
        solver_velocity_iteration_count=12,
        sleep_threshold=0.001,
        stabilization_threshold=0.0005,
    )
    common_collision = sim_utils.CollisionPropertiesCfg(
        collision_enabled=True,
        contact_offset=0.003,
        rest_offset=0.0,
        torsional_patch_radius=0.01,
        min_torsional_patch_radius=0.002,
    )
    common_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=2.5,
        dynamic_friction=2.0,
        restitution=0.0,
        friction_combine_mode="max",
        restitution_combine_mode="multiply",
    )

    if profile.object_kind == "cube":
        return RigidObjectCfg(
            prim_path=prim_path,
            spawn=sim_utils.CuboidCfg(
                size=profile.local_aabb.size_m,
                rigid_props=common_rigid,
                mass_props=sim_utils.MassPropertiesCfg(mass=profile.mass_kg),
                collision_props=common_collision,
                physics_material=common_material,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 1.0, 1.0),
                    metallic=0.0,
                    roughness=0.55,
                ),
            ),
            init_state=init_state,
        )

    if profile.object_kind == "urdf":
        urdf_path = profile.urdf_path
        if urdf_path is None or not urdf_path.is_file():
            raise FileNotFoundError(f"Phase3 object URDF not found for {profile.object_id}: {urdf_path}")
        return RigidObjectCfg(
            prim_path=prim_path,
            spawn=sim_utils.UrdfFileCfg(
                asset_path=urdf_path.as_posix(),
                scale=profile.spawn_scale,
                fix_base=False,
                joint_drive=None,
                rigid_props=common_rigid,
                mass_props=sim_utils.MassPropertiesCfg(mass=profile.mass_kg),
                collision_props=common_collision,
            ),
            init_state=init_state,
        )

    raise ValueError(f"Unsupported Phase3 object_kind={profile.object_kind!r}")


def design_phase3_scene(
    num_envs: int,
    *,
    robot_usd_path: Path,
    object_profile: Phase3ObjectProfile,
    activate_contact_sensors: bool = True,
):
    """Design one Phase 3 scene with the selected object profile."""

    import isaacsim.core.utils.prims as prim_utils  # type: ignore
    import isaaclab.sim as sim_utils
    import torch
    from isaaclab.assets import Articulation, RigidObject
    from protac.robot.asset import (
        TABLE_USD_PATH,
        build_robot_cfg,
        compute_grid_origins,
        spawn_robot_mount_base,
    )

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))
    light_cfg.func("/World/Light", light_cfg)
    table_cfg = sim_utils.UsdFileCfg(
        usd_path=TABLE_USD_PATH.as_posix(),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    )

    origins = compute_grid_origins(num_envs, spacing=2.5)
    robots: dict[str, Articulation] = {}
    objects: dict[str, RigidObject] = {}
    for index, origin in enumerate(origins, start=1):
        origin_prim = f"/World/Origin{index}"
        prim_utils.create_prim(origin_prim, "Xform", translation=origin)
        table_cfg.func(f"{origin_prim}/Table", table_cfg, translation=TABLE_TRANSLATION_M)
        spawn_robot_mount_base(f"{origin_prim}/RobotMountBase")
        robots[f"ur5_{index}"] = Articulation(
            cfg=build_robot_cfg(
                f"{origin_prim}/Robot",
                robot_usd_path,
                activate_contact_sensors=activate_contact_sensors,
            )
        )
        objects[f"{object_profile.object_id}_{index}"] = RigidObject(
            cfg=make_phase3_object_cfg(object_profile, f"{origin_prim}/{object_profile.prim_name}")
        )
    return robots, objects, torch.tensor(origins, dtype=torch.float32)
