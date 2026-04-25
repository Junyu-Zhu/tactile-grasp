from __future__ import annotations

import math
from pathlib import Path

import isaacsim.core.utils.prims as prim_utils  # type: ignore
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.sim.utils import bind_visual_material

from ur5_phase1_control import (
    ARM_JOINT_NAMES,
    BANANA_REST_HEIGHT,
    GRIPPER_CONTROL_JOINT_NAMES,
    RESET_ARM_JOINT_POS_DEG,
    TABLE_TOP_HEIGHT,
    TABLE_TRANSLATION,
    UR5_BASE_OFFSET,
    gripper_joint_overrides_rad,
    joint_targets_deg_to_rad,
)

BANANA_NAME = "YcbBanana"
BANANA_MASS = 0.09
CAMERA_EYE = (2.6, -2.2, 1.8)
CAMERA_TARGET = (0.6, 0.0, 0.75)
RESET_INTERVAL = 600
ROBOT_MOUNT_BASE_RADIUS = 0.18
ROBOT_MOUNT_BASE_HEIGHT = 0.02
ROBOT_MOUNT_BASE_COLOR = (0.18, 0.18, 0.17)
ROBOT_MOUNT_BASE_SEGMENTS = 96

_TABLE_ENV_DIR = Path(__file__).resolve().parent / "environment"
_TABLE_LEGACY_DIR = Path(__file__).resolve().parent / "assets" / "ur5_usd"
_ROBOT_ENV_DIR = Path(__file__).resolve().parent / "environment" / "ur5_robotiq_GSmini"
_ROBOT_URDF_DIR = _ROBOT_ENV_DIR / "urdf"
_ROBOT_USD_DIR = _ROBOT_ENV_DIR / "usd"
_ROBOT_USD_FILE_NAME = "ur5_robotiq_GSmini.usd"
_YCB_DIR = Path(__file__).resolve().parent / "ycb_objects"
_ROBOT_USD_REQUIRED_CONFIG_LINES = (
    "merge_fixed_joints: false",
    "self_collision: false",
    "finger_joint: 10000.0",
    "finger_joint: 500.0",
)
_ROBOT_USD_BASE_LAYER = _ROBOT_USD_DIR / "configuration" / "ur5_robotiq_GSmini_base.usd"
_GS_MINI_VISUAL_MATERIAL_PATCHES = {
    "GSminiConnectorDarkGray": {
        "color": (0.35, 0.35, 0.35),
        "mesh_paths": (
            "/visuals/left_inner_finger/left_gelsight_connector_visual/mesh",
            "/visuals/right_inner_finger/right_gelsight_connector_visual/mesh",
        ),
    },
    "GSminiBaseBlack": {
        "color": (0.05, 0.05, 0.05),
        "mesh_paths": (
            "/visuals/left_inner_finger/left_gelsight_mini_base_visual/mesh",
            "/visuals/right_inner_finger/right_gelsight_mini_base_visual/mesh",
        ),
    },
    "GSminiSoftBlue": {
        "color": (0.25, 0.60, 1.0),
        "mesh_paths": (
            "/visuals/left_inner_finger/left_gelsight_mini_soft_visual/mesh",
            "/visuals/right_inner_finger/right_gelsight_mini_soft_visual/mesh",
        ),
    },
}


def resolve_robot_urdf_path() -> Path:
    urdf_path = _ROBOT_URDF_DIR / "ur5_robotiq_GSmini.urdf"
    if not urdf_path.is_file():
        raise FileNotFoundError(f"Canonical robot URDF not found: {urdf_path}")
    return urdf_path


def _robot_usd_config_matches_expected() -> bool:
    config_path = _ROBOT_USD_DIR / "config.yaml"
    if not config_path.is_file():
        return False
    config_text = config_path.read_text(encoding="utf-8")
    return all(required_line in config_text for required_line in _ROBOT_USD_REQUIRED_CONFIG_LINES)


def _define_preview_surface_material(stage, material_path: str, color: tuple[float, float, float]):
    """Create or update a USD preview material without relying on Kit commands."""

    from pxr import Gf, Sdf, UsdShade  # type: ignore

    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _patch_gsmini_usd_visual_materials() -> None:
    """Repair generated USD material bindings for STL-based GSmini visuals.

    Isaac's URDF importer maps the embedded GSmini STL visuals to a shared
    white ``DefaultMaterial`` even when the URDF visual entries define inline
    colors.  Patch the generated base layer directly so connector/base/soft
    colors remain correct after every URDF->USD regeneration.
    """

    if not _ROBOT_USD_BASE_LAYER.is_file():
        return

    from pxr import Usd, UsdShade  # type: ignore

    stage = Usd.Stage.Open(_ROBOT_USD_BASE_LAYER.as_posix())
    if stage is None:
        raise RuntimeError(f"Unable to open robot USD base layer for material patching: {_ROBOT_USD_BASE_LAYER}")

    for material_name, patch in _GS_MINI_VISUAL_MATERIAL_PATCHES.items():
        material = _define_preview_surface_material(
            stage,
            f"/ur5_robotiq_GSmini/Looks/{material_name}",
            patch["color"],
        )
        for mesh_path in patch["mesh_paths"]:
            mesh_prim = stage.GetPrimAtPath(mesh_path)
            if not mesh_prim.IsValid():
                raise RuntimeError(f"Expected GSmini mesh prim missing from generated USD: {mesh_path}")
            UsdShade.MaterialBindingAPI(mesh_prim).Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)

    stage.GetRootLayer().Save()


def ensure_robot_usd_path(force_conversion: bool = False) -> Path:
    urdf_path = resolve_robot_urdf_path()
    _ROBOT_USD_DIR.mkdir(parents=True, exist_ok=True)
    usd_path = _ROBOT_USD_DIR / _ROBOT_USD_FILE_NAME
    should_force_conversion = (
        force_conversion
        or not usd_path.is_file()
        or urdf_path.stat().st_mtime > usd_path.stat().st_mtime
        or not _robot_usd_config_matches_expected()
    )
    converter_cfg = UrdfConverterCfg(
        asset_path=urdf_path.as_posix(),
        usd_dir=_ROBOT_USD_DIR.as_posix(),
        usd_file_name=_ROBOT_USD_FILE_NAME,
        force_usd_conversion=should_force_conversion,
        make_instanceable=False,
        fix_base=True,
        root_link_name="base_link",
        # Keep fixed joints unmerged for Robotiq link/body naming, but embed
        # GSmini/connector visuals directly in the fingertip pads so they do
        # not rely on lightweight fixed-joint rigid bodies during arm motion.
        merge_fixed_joints=False,
        convert_mimic_joints_to_normal_joints=True,
        collider_type="convex_decomposition",
        self_collision=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness={
                    **{joint_name: 800.0 for joint_name in ARM_JOINT_NAMES},
                    **{joint_name: 1.0e4 for joint_name in GRIPPER_CONTROL_JOINT_NAMES},
                },
                damping={
                    **{joint_name: 40.0 for joint_name in ARM_JOINT_NAMES},
                    **{joint_name: 5.0e2 for joint_name in GRIPPER_CONTROL_JOINT_NAMES},
                },
            )
        ),
    )
    usd_path = Path(UrdfConverter(converter_cfg).usd_path)
    if not usd_path.is_file():
        raise FileNotFoundError(f"Robot USD was not generated from URDF: {usd_path}")
    _patch_gsmini_usd_visual_materials()
    return usd_path


def resolve_table_usd_path() -> Path:
    env_usd_path = _TABLE_ENV_DIR / "table.usd"
    if env_usd_path.is_file():
        return env_usd_path

    legacy_usd_path = _TABLE_LEGACY_DIR / "table.usd"
    if legacy_usd_path.is_file():
        _TABLE_ENV_DIR.mkdir(parents=True, exist_ok=True)
        env_usd_path.write_bytes(legacy_usd_path.read_bytes())
        return env_usd_path

    raise FileNotFoundError(f"Table USD not found in either location: {env_usd_path} or {legacy_usd_path}")


def resolve_banana_urdf_path() -> Path:
    urdf_path = _YCB_DIR / BANANA_NAME / "model.urdf"
    if not urdf_path.is_file():
        raise FileNotFoundError(f"YCB banana URDF not found: {urdf_path}")
    return urdf_path


TABLE_USD_PATH = resolve_table_usd_path()
BANANA_URDF_PATH = resolve_banana_urdf_path()


def _reset_joint_state_map() -> dict[str, float]:
    joint_overrides = joint_targets_deg_to_rad(RESET_ARM_JOINT_POS_DEG)
    joint_overrides.update(gripper_joint_overrides_rad(closed=False))
    return joint_overrides


def build_robot_cfg(prim_path: str, robot_usd_path: Path) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=robot_usd_path.as_posix(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_depenetration_velocity=5.0,
                enable_gyroscopic_forces=True,
                disable_gravity=True,
            ),
            activate_contact_sensors=False,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
            copy_from_source=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(
                UR5_BASE_OFFSET[0],
                UR5_BASE_OFFSET[1],
                TABLE_TOP_HEIGHT + UR5_BASE_OFFSET[2],
            ),
            joint_pos=_reset_joint_state_map(),
        ),
        actuators={
            "arm_actuator": ImplicitActuatorCfg(
                joint_names_expr=ARM_JOINT_NAMES,
                velocity_limit_sim=50.0,
                effort_limit_sim=87.0,
                stiffness=800.0,
                damping=40.0,
            ),
            "gripper_actuator": ImplicitActuatorCfg(
                joint_names_expr=GRIPPER_CONTROL_JOINT_NAMES,
                effort_limit_sim=1000.0,
                velocity_limit_sim=2.0,
                stiffness=1.0e4,
                damping=5.0e2,
            ),
        },
    )


def compute_grid_origins(num_envs: int, spacing: float = 2.5) -> list[list[float]]:
    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}")
    num_cols = int(torch.ceil(torch.sqrt(torch.tensor(float(num_envs)))).item())
    num_rows = int(torch.ceil(torch.tensor(float(num_envs / num_cols))).item())
    origins = []
    for i in range(num_envs):
        row = i // num_cols
        col = i % num_cols
        origins.append(
            [
                spacing * (row - (num_rows - 1) / 2.0),
                spacing * (col - (num_cols - 1) / 2.0),
                0.0,
            ]
        )
    return origins


def make_banana_cfg(prim_path: str) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UrdfFileCfg(
            asset_path=BANANA_URDF_PATH.as_posix(),
            fix_base=False,
            joint_drive=None,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=False,
                linear_damping=1.0,
                angular_damping=2.0,
                max_linear_velocity=1.0,
                max_angular_velocity=360.0,
                max_depenetration_velocity=1.0,
                max_contact_impulse=10.0,
                enable_gyroscopic_forces=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=BANANA_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002,
                rest_offset=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(TABLE_TRANSLATION[0], TABLE_TRANSLATION[1], BANANA_REST_HEIGHT),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


def robot_mount_base_translation() -> tuple[float, float, float]:
    return (
        UR5_BASE_OFFSET[0],
        UR5_BASE_OFFSET[1],
        TABLE_TOP_HEIGHT + ROBOT_MOUNT_BASE_HEIGHT / 2.0,
    )


def _cylinder_mesh_arrays(radius: float, height: float, segments: int) -> tuple[list[tuple[float, float, float]], list[int], list[int]]:
    half_height = height / 2.0
    bottom_ring: list[tuple[float, float, float]] = []
    top_ring: list[tuple[float, float, float]] = []
    for index in range(segments):
        angle = 2.0 * math.pi * index / segments
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        bottom_ring.append((x, y, -half_height))
        top_ring.append((x, y, half_height))

    points = bottom_ring + top_ring
    face_vertex_counts: list[int] = []
    face_vertex_indices: list[int] = []

    for index in range(segments):
        next_index = (index + 1) % segments
        face_vertex_counts.append(4)
        face_vertex_indices.extend([index, next_index, segments + next_index, segments + index])

    face_vertex_counts.append(segments)
    face_vertex_indices.extend(range(segments, 2 * segments))
    face_vertex_counts.append(segments)
    face_vertex_indices.extend(reversed(range(segments)))

    return points, face_vertex_indices, face_vertex_counts


def spawn_robot_mount_base(prim_path: str) -> None:
    """Spawn a smooth visual mounting base that fills the table/UR5 gap.

    The Phase 1 robot pose intentionally follows the TacEx
    ``load_ur5_in_isaacsim.py`` convention:
    ``TABLE_TOP_HEIGHT + UR5_BASE_OFFSET[2]``.  A thin graphite plate makes the
    robot look table-mounted rather than visually floating, without adding
    collision contacts that could disturb the fixed-base articulation.
    """

    points, face_vertex_indices, face_vertex_counts = _cylinder_mesh_arrays(
        ROBOT_MOUNT_BASE_RADIUS,
        ROBOT_MOUNT_BASE_HEIGHT,
        ROBOT_MOUNT_BASE_SEGMENTS,
    )
    prim_utils.create_prim(prim_path, "Xform", translation=robot_mount_base_translation())
    prim_utils.create_prim(f"{prim_path}/geometry", "Xform")
    mesh_path = f"{prim_path}/geometry/mesh"
    prim_utils.create_prim(
        mesh_path,
        "Mesh",
        attributes={
            "points": points,
            "faceVertexIndices": face_vertex_indices,
            "faceVertexCounts": face_vertex_counts,
            "subdivisionScheme": "none",
        },
    )
    prim_utils.create_prim(f"{prim_path}/Looks", "Scope")
    material_path = f"{prim_path}/Looks/graphite"
    material_cfg = sim_utils.PreviewSurfaceCfg(
        diffuse_color=ROBOT_MOUNT_BASE_COLOR,
        metallic=0.0,
        roughness=0.35,
    )
    material_cfg.func(material_path, material_cfg)
    bind_visual_material(mesh_path, material_path)


def design_scene(num_envs: int, *, robot_usd_path: Path) -> tuple[dict[str, Articulation], dict[str, RigidObject], torch.Tensor]:
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
    bananas: dict[str, RigidObject] = {}

    for index, origin in enumerate(origins, start=1):
        origin_prim = f"/World/Origin{index}"
        prim_utils.create_prim(origin_prim, "Xform", translation=origin)
        table_cfg.func(f"{origin_prim}/Table", table_cfg, translation=TABLE_TRANSLATION)
        spawn_robot_mount_base(f"{origin_prim}/RobotMountBase")
        robots[f"ur5_{index}"] = Articulation(cfg=build_robot_cfg(f"{origin_prim}/Robot", robot_usd_path))
        bananas[f"banana_{index}"] = RigidObject(cfg=make_banana_cfg(f"{origin_prim}/{BANANA_NAME}"))

    return robots, bananas, torch.tensor(origins, dtype=torch.float32)


def report_scene_state(origins: torch.Tensor, robot_urdf_path: Path, robot_usd_path: Path) -> None:
    base_pose = (
        float(origins[0, 0].item()) + UR5_BASE_OFFSET[0],
        float(origins[0, 1].item()) + UR5_BASE_OFFSET[1],
        float(origins[0, 2].item()) + TABLE_TOP_HEIGHT + UR5_BASE_OFFSET[2],
    )
    banana_pose = (
        float(origins[0, 0].item()) + TABLE_TRANSLATION[0],
        float(origins[0, 1].item()) + TABLE_TRANSLATION[1],
        float(origins[0, 2].item()) + BANANA_REST_HEIGHT,
    )
    mount_base_local_pose = robot_mount_base_translation()
    mount_base_pose = (
        float(origins[0, 0].item()) + mount_base_local_pose[0],
        float(origins[0, 1].item()) + mount_base_local_pose[1],
        float(origins[0, 2].item()) + mount_base_local_pose[2],
    )
    print("[INFO] Phase 1 source of truth = tactile_grasp/ur5_sim.py")
    print(f"[INFO] Canonical robot URDF: {robot_urdf_path}")
    print(f"[INFO] Generated robot USD: {robot_usd_path}")
    print(f"[INFO] Table translation reference: {TABLE_TRANSLATION}")
    print(f"[INFO] UR5 base pose reference: {base_pose}")
    print(
        "[INFO] UR5 visual mount base: "
        f"pose={mount_base_pose}, radius={ROBOT_MOUNT_BASE_RADIUS}, height={ROBOT_MOUNT_BASE_HEIGHT}"
    )
    print(f"[INFO] Banana pose reference: {banana_pose}")
