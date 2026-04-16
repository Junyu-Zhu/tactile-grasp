from __future__ import annotations

import importlib.util
from pathlib import Path

import isaacsim.core.utils.prims as prim_utils  # type: ignore
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg

from ur5_phase1_control import BANANA_REST_HEIGHT, TABLE_TOP_HEIGHT, TABLE_TRANSLATION, UR5_BASE_OFFSET

BANANA_NAME = "YcbBanana"
BANANA_MASS = 0.09
CAMERA_EYE = (2.6, -2.2, 1.8)
CAMERA_TARGET = (0.6, 0.0, 0.75)
RESET_INTERVAL = 600

_ASSETS_DIR = Path(__file__).resolve().parent / "assets" / "ur5_usd"
_YCB_DIR = Path(__file__).resolve().parent / "ycb_objects"


def load_local_ur5_gripper_cfg() -> ArticulationCfg:
    cfg_path = _ASSETS_DIR / "ur5.py"
    spec = importlib.util.spec_from_file_location("tactile_grasp_local_ur5_cfg", cfg_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load UR5 config module from: {cfg_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.UR5_GRIPPER_CFG


def resolve_ur5_usd_path() -> Path:
    usd_path = _ASSETS_DIR / "ur5_moveit.usd"
    if not usd_path.is_file():
        raise FileNotFoundError(f"UR5 USD not found: {usd_path}")
    return usd_path


def resolve_table_usd_path() -> Path:
    usd_path = _ASSETS_DIR / "table.usd"
    if not usd_path.is_file():
        raise FileNotFoundError(f"Table USD not found: {usd_path}")
    return usd_path


def resolve_banana_urdf_path() -> Path:
    urdf_path = _YCB_DIR / BANANA_NAME / "model.urdf"
    if not urdf_path.is_file():
        raise FileNotFoundError(f"YCB banana URDF not found: {urdf_path}")
    return urdf_path


UR5_GRIPPER_CFG = load_local_ur5_gripper_cfg()
UR5_USD_PATH = resolve_ur5_usd_path()
TABLE_USD_PATH = resolve_table_usd_path()
BANANA_URDF_PATH = resolve_banana_urdf_path()


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


def make_robot_cfg(prim_path: str) -> ArticulationCfg:
    robot_cfg = UR5_GRIPPER_CFG.replace(prim_path=prim_path)
    robot_cfg.spawn.usd_path = UR5_USD_PATH.as_posix()
    robot_cfg.init_state.pos = (
        UR5_BASE_OFFSET[0],
        UR5_BASE_OFFSET[1],
        TABLE_TOP_HEIGHT + UR5_BASE_OFFSET[2],
    )
    return robot_cfg


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


def design_scene(num_envs: int) -> tuple[dict[str, Articulation], dict[str, RigidObject], torch.Tensor]:
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
        robots[f"ur5_{index}"] = Articulation(cfg=make_robot_cfg(f"{origin_prim}/Robot"))
        bananas[f"banana_{index}"] = RigidObject(cfg=make_banana_cfg(f"{origin_prim}/{BANANA_NAME}"))

    return robots, bananas, torch.tensor(origins, dtype=torch.float32)


def report_scene_state(origins: torch.Tensor) -> None:
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
    print("[INFO] Phase 1 source of truth = tactile_grasp/ur5_sim.py")
    print(f"[INFO] Table translation reference: {TABLE_TRANSLATION}")
    print(f"[INFO] UR5 base pose reference: {base_pose}")
    print(f"[INFO] Banana pose reference: {banana_pose}")
