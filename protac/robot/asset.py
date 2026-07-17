"""Canonical UR5/Robotiq/GSmini asset conversion and scene primitives."""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TypedDict

import isaacsim.core.utils.prims as prim_utils  # type: ignore
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.sim.utils import bind_visual_material

from protac.robot.control import (
    ARM_JOINT_NAMES,
    GRIPPER_COMMAND_JOINT_NAMES,
    RESET_ARM_JOINT_POS_DEG,
    TABLE_TOP_HEIGHT,
    UR5_BASE_OFFSET,
    gripper_joint_overrides_rad,
    joint_targets_deg_to_rad,
)

CAMERA_EYE = (2.6, -2.2, 1.8)
CAMERA_TARGET = (0.6, 0.0, 0.75)
ROBOT_MOUNT_BASE_RADIUS = 0.18
ROBOT_MOUNT_BASE_HEIGHT = 0.02
ROBOT_MOUNT_BASE_COLOR = (0.18, 0.18, 0.17)
ROBOT_MOUNT_BASE_SEGMENTS = 96

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TABLE_ENV_DIR = PROJECT_ROOT / "environment"
_TABLE_LEGACY_DIR = PROJECT_ROOT / "assets" / "ur5_usd"
_ROBOT_ENV_DIR = PROJECT_ROOT / "environment" / "ur5_robotiq_GSmini"
_ROBOT_URDF_DIR = _ROBOT_ENV_DIR / "urdf"
_ROBOT_USD_DIR = _ROBOT_ENV_DIR / "usd"
_ROBOT_URDF_FILE_NAME = "ur5_robotiq_GSmini_new.urdf"
_ROBOT_USD_FILE_NAME = "ur5_robotiq_GSmini_new.usd"
_ROBOT_USD_REQUIRED_CONFIG_LINES = (
    "merge_fixed_joints: false",
    "convert_mimic_joints_to_normal_joints: false",
    "self_collision: false",
    "finger_joint: 2500.0",
    "right_inner_knuckle_joint: 2500.0",
    "finger_joint: 120.0",
    "right_inner_knuckle_joint: 120.0",
)
_ROBOT_USD_BASE_LAYER = _ROBOT_USD_DIR / "configuration" / "ur5_robotiq_GSmini_new_base.usd"
_USD_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
class _MaterialPatch(TypedDict):
    color: tuple[float, float, float]
    mesh_paths: tuple[str, ...]


_GS_MINI_VISUAL_MATERIAL_PATCHES: dict[str, _MaterialPatch] = {
    "GSminiConnectorDarkGray": {
        "color": (0.35, 0.35, 0.35),
        "mesh_paths": (
            "/visuals/left_gelsight_connector/left_gelsight_connector_visual/mesh",
            "/visuals/right_gelsight_connector/right_gelsight_connector_visual/mesh",
        ),
    },
    "GSminiBaseBlack": {
        "color": (0.05, 0.05, 0.05),
        "mesh_paths": (
            "/visuals/left_gelsight_mini_case/left_gelsight_mini_base_visual/mesh",
            "/visuals/right_gelsight_mini_case/right_gelsight_mini_base_visual/mesh",
        ),
    },
    "GSminiSoftBlue": {
        "color": (0.25, 0.60, 1.0),
        "mesh_paths": (
            "/visuals/left_gelsight_mini_gelpad/left_gelsight_mini_soft_visual/mesh",
            "/visuals/right_gelsight_mini_gelpad/right_gelsight_mini_soft_visual/mesh",
        ),
    },
}


def resolve_robot_urdf_path() -> Path:
    urdf_path = _ROBOT_URDF_DIR / _ROBOT_URDF_FILE_NAME
    if not urdf_path.is_file():
        raise FileNotFoundError(f"Canonical robot URDF not found: {urdf_path}")
    return urdf_path


def _resolve_robot_mesh_paths(urdf_path: Path) -> tuple[Path, ...]:
    """Resolve and validate direct mesh dependencies of the canonical URDF.

    Isaac Sim 4.5 derives a USD prim name from each mesh basename without
    sanitizing hyphens.  Reject such names before the importer can leave an
    empty, apparently generated USD behind.
    """

    mesh_paths: list[Path] = []
    for mesh in ET.parse(urdf_path).getroot().iter("mesh"):
        filename = mesh.get("filename")
        if not filename:
            raise ValueError(f"URDF mesh entry has no filename: {urdf_path}")
        if filename.startswith("package://"):
            raise ValueError(f"Canonical URDF must use resolvable local mesh paths, got: {filename}")

        mesh_path = (urdf_path.parent / filename).resolve()
        if not _USD_IDENTIFIER_PATTERN.fullmatch(mesh_path.stem):
            raise ValueError(
                "Isaac Sim requires a USD-safe mesh basename containing only letters, digits, and underscores; "
                f"rename {mesh_path.name!r} and update {urdf_path.name}."
            )
        if not mesh_path.is_file():
            raise FileNotFoundError(f"URDF mesh asset not found: {mesh_path}")
        mesh_paths.append(mesh_path)

    return tuple(dict.fromkeys(mesh_paths))


def _robot_sources_are_newer_than_usd(source_paths: tuple[Path, ...], usd_path: Path) -> bool:
    """Return whether the URDF or a referenced mesh changed after conversion."""

    if not usd_path.is_file():
        return True
    usd_mtime_ns = usd_path.stat().st_mtime_ns
    return any(source_path.stat().st_mtime_ns > usd_mtime_ns for source_path in source_paths)


def _robot_usd_config_matches_expected() -> bool:
    config_path = _ROBOT_USD_DIR / "config.yaml"
    if not config_path.is_file():
        return False
    config_text = config_path.read_text(encoding="utf-8")
    required_lines = (
        *_ROBOT_USD_REQUIRED_CONFIG_LINES,
        f"asset_path: {resolve_robot_urdf_path().as_posix()}",
        f"usd_file_name: {_ROBOT_USD_FILE_NAME}",
    )
    return all(required_line in config_text for required_line in required_lines)


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

    missing_mesh_paths: list[str] = []
    for material_name, patch in _GS_MINI_VISUAL_MATERIAL_PATCHES.items():
        material = _define_preview_surface_material(
            stage,
            f"/ur5_robotiq_GSmini/Looks/{material_name}",
            patch["color"],
        )
        for mesh_path in patch["mesh_paths"]:
            mesh_prim = stage.GetPrimAtPath(mesh_path)
            if not mesh_prim.IsValid():
                missing_mesh_paths.append(mesh_path)
                continue
            UsdShade.MaterialBindingAPI(mesh_prim).Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)

    if missing_mesh_paths:
        missing = "\n  - ".join(missing_mesh_paths)
        raise RuntimeError(
            "URDF conversion produced an incomplete robot USD; expected GSmini visual meshes are missing:\n"
            f"  - {missing}"
        )

    stage.GetRootLayer().Save()


def _validate_generated_robot_usd(usd_path: Path) -> None:
    """Reject the empty USD layers that Isaac's importer can leave on failure."""

    from pxr import Usd  # type: ignore

    stage = Usd.Stage.Open(usd_path.as_posix())
    if stage is None or not stage.GetDefaultPrim().IsValid():
        raise RuntimeError(
            "URDF conversion did not produce a valid robot USD with a default prim. "
            "Inspect the preceding Isaac URDF importer errors; mesh basenames must be valid USD identifiers. "
            f"Output: {usd_path}"
        )

    prim_names = {prim.GetName() for prim in stage.Traverse()}
    required_prim_names = {"base_link", "left_gelsight_connector", "right_gelsight_connector"}
    missing_prim_names = sorted(required_prim_names - prim_names)
    if missing_prim_names:
        raise RuntimeError(
            f"Generated robot USD is incomplete; missing required prims {missing_prim_names}: {usd_path}"
        )


def ensure_robot_usd_path(force_conversion: bool = False) -> Path:
    urdf_path = resolve_robot_urdf_path()
    source_paths = (urdf_path, *_resolve_robot_mesh_paths(urdf_path))
    _ROBOT_USD_DIR.mkdir(parents=True, exist_ok=True)
    usd_path = _ROBOT_USD_DIR / _ROBOT_USD_FILE_NAME
    sources_changed = _robot_sources_are_newer_than_usd(source_paths, usd_path)
    config_matches = _robot_usd_config_matches_expected()
    should_force_conversion = (
        force_conversion
        or not usd_path.is_file()
        or sources_changed
        or not config_matches
    )
    if should_force_conversion:
        reasons = []
        if force_conversion:
            reasons.append("explicit request")
        if not usd_path.is_file():
            reasons.append("USD is missing")
        if sources_changed:
            reasons.append("URDF or referenced mesh is newer")
        if not config_matches:
            reasons.append("converter configuration changed")
        print(f"[INFO] Regenerating robot USD ({', '.join(reasons)}): {usd_path}")
    converter_cfg = UrdfConverterCfg(
        asset_path=urdf_path.as_posix(),
        usd_dir=_ROBOT_USD_DIR.as_posix(),
        usd_file_name=_ROBOT_USD_FILE_NAME,
        force_usd_conversion=should_force_conversion,
        make_instanceable=False,
        fix_base=True,
        root_link_name="base_link",
        # Keep fixed joints unmerged so the TacEx-style connector/case/gelpad
        # attachment chain and camera links survive URDF->USD conversion.
        merge_fixed_joints=False,
        # Ignore URDF mimic constraints and retain all six gripper joints as
        # independently driven joints.  This mirrors the stable PyBullet
        # reference controller and avoids contact-load asymmetry observed with
        # PhysxMimicJointAPI on this open-chain Robotiq asset.
        convert_mimic_joints_to_normal_joints=False,
        collider_type="convex_decomposition",
        self_collision=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness={
                    **{joint_name: 800.0 for joint_name in ARM_JOINT_NAMES},
                    **{joint_name: 2.5e3 for joint_name in GRIPPER_COMMAND_JOINT_NAMES},
                },
                damping={
                    **{joint_name: 40.0 for joint_name in ARM_JOINT_NAMES},
                    **{joint_name: 1.2e2 for joint_name in GRIPPER_COMMAND_JOINT_NAMES},
                },
            )
        ),
    )
    usd_path = Path(UrdfConverter(converter_cfg).usd_path)
    if not usd_path.is_file():
        raise FileNotFoundError(f"Robot USD was not generated from URDF: {usd_path}")
    _validate_generated_robot_usd(usd_path)
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


TABLE_USD_PATH = resolve_table_usd_path()


def _reset_joint_state_map() -> dict[str, float]:
    joint_overrides = joint_targets_deg_to_rad(RESET_ARM_JOINT_POS_DEG)
    joint_overrides.update(gripper_joint_overrides_rad(closed=False))
    return joint_overrides


def build_robot_cfg(prim_path: str, robot_usd_path: Path, *, activate_contact_sensors: bool = False) -> ArticulationCfg:
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
            activate_contact_sensors=activate_contact_sensors,
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
                joint_names_expr=GRIPPER_COMMAND_JOINT_NAMES,
                effort_limit_sim=200.0,
                velocity_limit_sim=1.0,
                stiffness=2.5e3,
                damping=1.2e2,
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

    The robot pose intentionally follows the TacEx
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

