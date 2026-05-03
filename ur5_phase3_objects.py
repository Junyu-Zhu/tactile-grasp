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

PROJECT_DIR = Path(__file__).resolve().parent

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
        return tuple(self.max_m[index] - self.min_m[index] for index in range(3))

    @property
    def center_m(self) -> tuple[float, float, float]:
        return tuple((self.min_m[index] + self.max_m[index]) / 2.0 for index in range(3))


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

    @property
    def rest_root_z_m(self) -> float:
        return TABLE_TOP_HEIGHT_M - self.local_aabb.min_m[2]

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
    close_rad=0.25,
    stable_force_threshold_n=0.5,
    high_force_threshold_n=8.0,
    max_close_object_lift_m=0.004,
    hold_seconds=2.0,
    micro_lift_distance_m=0.015,
    pregrasp_mode="phase2_side_pregrasp",
    contact_approach_refine=False,
    tuning_note=(
        "Phase2-proven cube profile: side grasp at the cube center and conservative close width; "
        "Phase3 allows a small <=4 mm close-induced lift because early runtime trials reached "
        "stable two-sided force around 1.9-3.1 mm before hold."
    ),
)

CHIPS_CAN_PROFILE = Phase3ObjectProfile(
    object_id="chips_can",
    label="YcbChipsCan",
    object_kind="urdf",
    prim_name="YcbChipsCan",
    urdf_rel_path="ycb_objects/YcbChipsCan/model.urdf",
    local_aabb=LocalAabb(
        min_m=(-0.042753130197522, -0.025356302224099, 0.00570114748552),
        max_m=(0.03188893683255, 0.04928576480597299, 0.24733659625053),
    ),
    mass_kg=0.205,
    root_rot_wxyz=(1.0, 0.0, 0.0, 0.0),
    grasp_center_local_m=(-0.060, 0.011964731290936996, 0.126518871868025),
    close_rad=0.35,
    stable_force_threshold_n=0.35,
    high_force_threshold_n=6.0,
    max_close_object_lift_m=0.006,
    hold_seconds=2.0,
    micro_lift_distance_m=0.012,
    pregrasp_mode="reset_clearance",
    contact_approach_refine=True,
    tuning_note=(
        "Upright can profile: keep the reset-clearance pregrasp, then target the near-side "
        "mid-height edge instead of the full cylinder center because the GSmini fingertips reduce "
        "the usable open gap below the can diameter. Runtime tuning treats the final soft-center "
        "approach as the contact stage so the can is touched without tipping before hold."
    ),
)

CRACKER_BOX_PROFILE = Phase3ObjectProfile(
    object_id="cracker_box",
    label="YcbCrackerBox",
    object_kind="urdf",
    prim_name="YcbCrackerBox",
    urdf_rel_path="ycb_objects/YcbCrackerBox/model.urdf",
    local_aabb=LocalAabb(
        min_m=(-0.003362, -0.089567, -0.108358),
        max_m=(0.058277, 0.0679, 0.098405),
    ),
    mass_kg=0.411,
    # Rotate the box 90 degrees about z so the thin local-x dimension is along
    # the gripper closing axis.  Without this, the world-y width is too large
    # for a Robotiq 2F-85 side grasp.
    root_rot_wxyz=_yaw_quat_wxyz(math.pi / 2.0),
    grasp_center_local_m=(0.027457500000000003, -0.010833499999999996, -0.004976499999999995),
    close_rad=0.22,
    stable_force_threshold_n=0.45,
    high_force_threshold_n=7.0,
    max_close_object_lift_m=0.004,
    hold_seconds=2.5,
    micro_lift_distance_m=0.010,
    pregrasp_mode="reset_clearance",
    contact_approach_refine=False,
    tuning_note=(
        "Box profile: yaw-rotate so the narrow side is gripped; use a shorter micro-lift because "
        "the tall box is heavier and more tip-prone than the cube."
    ),
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
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
        friction_combine_mode="multiply",
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
    from ur5_phase1_scene import (
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
