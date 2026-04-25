from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaacsim.core.prims import XFormPrim
import isaacsim.core.utils.prims as prim_utils

from ur5_phase1_control import build_gripper_joint_target, build_reset_joint_target
from ur5_phase1_reset import reset_scene, step_scene

PHASE2_SCOPE_SENTENCE = "Phase 2 = 基于 canonical integrated UR5e + Robotiq + connector + GSmini embodiment，先校核挂载，再做 tactile output bring-up。"

CANONICAL_ROBOT_REFERENCE = "tactile_grasp/environment/ur5_robotiq_GSmini/urdf/ur5_robotiq_GSmini.urdf"
RETIRED_PHASE1_ROBOT_REFERENCE = "tactile_grasp/assets/ur5_usd/ur5_moveit.usd"

PHASE2_MOUNT_USD_DIR = Path(__file__).resolve().parent / "assets" / "phase2_mount_usd"
GELSIGHT_MINI_CASE_MESH = Path(__file__).resolve().parent / "assets" / "meshes" / "gelsight_mini" / "visual" / "base_link.STL"
GELSIGHT_MINI_GELPAD_MESH = Path(__file__).resolve().parent / "assets" / "meshes" / "gelsight_mini" / "visual" / "soft_link.STL"
GELSIGHT_CONNECTOR_MESH = Path(__file__).resolve().parent / "assets" / "meshes" / "gelsight_robotiq_connector" / "visual" / "gelsight_adaptor.STL"

LEFT_FINGER_LINK_PATH = "/World/Origin1/Robot/left_inner_finger_pad"
RIGHT_FINGER_LINK_PATH = "/World/Origin1/Robot/right_inner_finger_pad"

PHASE2_VISUALS_ROOT_PATH = "/World/Phase2Visuals"
LEFT_VISUAL_GROUP_PATH = f"{PHASE2_VISUALS_ROOT_PATH}/left"
RIGHT_VISUAL_GROUP_PATH = f"{PHASE2_VISUALS_ROOT_PATH}/right"
LEFT_CONNECTOR_PRIM_PATH = f"{LEFT_VISUAL_GROUP_PATH}/gelsight_connector_left"
LEFT_CASE_PRIM_PATH = f"{LEFT_VISUAL_GROUP_PATH}/gelsight_mini_case_left"
LEFT_GELPAD_PRIM_PATH = f"{LEFT_VISUAL_GROUP_PATH}/gelsight_mini_gelpad_left"
RIGHT_CONNECTOR_PRIM_PATH = f"{RIGHT_VISUAL_GROUP_PATH}/gelsight_connector_right"
RIGHT_CASE_PRIM_PATH = f"{RIGHT_VISUAL_GROUP_PATH}/gelsight_mini_case_right"
RIGHT_GELPAD_PRIM_PATH = f"{RIGHT_VISUAL_GROUP_PATH}/gelsight_mini_gelpad_right"

# Manual connector tuning values.
# Edit these two constants directly when you want to move or rotate the visual connector in simulation.
CONNECTOR_LOCAL_TRANSLATION = (-0.0155, -0.012, 0.0)
CONNECTOR_LOCAL_QUAT_WXYZ = (1.0, 0.0, 0.0, 0.0)
CASE_LOCAL_TRANSLATION = (0.0, 0.0, 0.0185)
CASE_LOCAL_QUAT_WXYZ = (1.0, 0.0, 0.0, 0.0)
GELPAD_LOCAL_TRANSLATION = (0.0, 0.0, 0.024)
GELPAD_LOCAL_QUAT_WXYZ = (1.0, 0.0, 0.0, 0.0)

RESET_TRANSLATION_TOLERANCE_M = 1.0e-4
FOLLOW_TRANSLATION_TOLERANCE_M = 2.0e-3
DEFAULT_RESET_TRIALS = 5
DEFAULT_FOLLOW_STEPS = 160
DEFAULT_SETTLE_STEPS = 45
MOUNT_IDENTITY_QUAT_WXYZ = (1.0, 0.0, 0.0, 0.0)

_XFORM_VIEW_CACHE: dict[str, XFormPrim] = {}
_LEFT_TIP_BODY_INDEX_CACHE: dict[int, int] = {}


@dataclass(frozen=True)
class MountAssetSpec:
    name: str
    mesh_path: Path
    usd_path: Path
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)


CASE_USD_SPEC = MountAssetSpec(
    name="gelsight_mini_case",
    mesh_path=GELSIGHT_MINI_CASE_MESH,
    usd_path=PHASE2_MOUNT_USD_DIR / "gelsight_mini_case.usd",
)
GELPAD_USD_SPEC = MountAssetSpec(
    name="gelsight_mini_gelpad",
    mesh_path=GELSIGHT_MINI_GELPAD_MESH,
    usd_path=PHASE2_MOUNT_USD_DIR / "gelsight_mini_gelpad.usd",
)
CONNECTOR_USD_SPEC = MountAssetSpec(
    name="gelsight_connector",
    mesh_path=GELSIGHT_CONNECTOR_MESH,
    usd_path=PHASE2_MOUNT_USD_DIR / "gelsight_connector.usd",
    scale=(0.001, 0.001, 0.001),
)


def source_of_truth_summary() -> dict[str, object]:
    return {
        "scope": PHASE2_SCOPE_SENTENCE,
        "primary_robot_reference": CANONICAL_ROBOT_REFERENCE,
        "retired_phase1_robot_reference": RETIRED_PHASE1_ROBOT_REFERENCE,
        "generated_robot_asset_policy": "Any generated USD is only a convenience derivative of the canonical URDF, not a new truth source.",
        "attachment_target_paths": {
            "left_fingertip": LEFT_FINGER_LINK_PATH,
            "right_fingertip": RIGHT_FINGER_LINK_PATH,
        },
        "visual_root_path": PHASE2_VISUALS_ROOT_PATH,
        "mesh_paths": {
            "gelsight_mini_case": str(GELSIGHT_MINI_CASE_MESH),
            "gelsight_mini_gelpad": str(GELSIGHT_MINI_GELPAD_MESH),
            "gelsight_connector": str(GELSIGHT_CONNECTOR_MESH),
        },
        "generated_usd_paths": {
            "gelsight_mini_case": str(CASE_USD_SPEC.usd_path),
            "gelsight_mini_gelpad": str(GELPAD_USD_SPEC.usd_path),
            "gelsight_connector": str(CONNECTOR_USD_SPEC.usd_path),
        },
        "single_side_debug_side": "left",
        "single_side_paths": {
            "connector": LEFT_CONNECTOR_PRIM_PATH,
            "case": LEFT_CASE_PRIM_PATH,
            "gelpad": LEFT_GELPAD_PRIM_PATH,
        },
        "future_right_side_paths": {
            "connector": RIGHT_CONNECTOR_PRIM_PATH,
            "case": RIGHT_CASE_PRIM_PATH,
            "gelpad": RIGHT_GELPAD_PRIM_PATH,
        },
    }


def _ensure_mesh_usd(asset: MountAssetSpec) -> Path:
    PHASE2_MOUNT_USD_DIR.mkdir(parents=True, exist_ok=True)
    if asset.usd_path.is_file():
        return asset.usd_path
    cfg = MeshConverterCfg(
        asset_path=asset.mesh_path.as_posix(),
        usd_dir=asset.usd_path.parent.as_posix(),
        usd_file_name=asset.usd_path.name,
        make_instanceable=False,
        collision_approximation="none",
        force_usd_conversion=True,
        scale=asset.scale,
    )
    MeshConverter(cfg)
    if not asset.usd_path.is_file():
        raise FileNotFoundError(f"Failed to generate USD for mesh: {asset.mesh_path}")
    return asset.usd_path


def ensure_phase2_mount_usds() -> dict[str, Path]:
    return {
        "connector": _ensure_mesh_usd(CONNECTOR_USD_SPEC),
        "case": _ensure_mesh_usd(CASE_USD_SPEC),
        "gelpad": _ensure_mesh_usd(GELPAD_USD_SPEC),
    }


def _clear_cached_prim_view(prim_path: str) -> None:
    _XFORM_VIEW_CACHE.pop(prim_path, None)


def _ensure_visual_group_paths() -> None:
    for path in (PHASE2_VISUALS_ROOT_PATH, LEFT_VISUAL_GROUP_PATH, RIGHT_VISUAL_GROUP_PATH):
        if not prim_utils.is_prim_path_valid(path):
            prim_utils.create_prim(path, "Xform")


def _spawn_mount_xform(prim_path: str, translation: tuple[float, float, float], orientation: tuple[float, float, float, float]) -> None:
    prim_utils.create_prim(prim_path, "Xform", translation=translation, orientation=orientation)
    _clear_cached_prim_view(prim_path)


def _spawn_referenced_asset(prim_path: str, usd_path: Path, child_name: str = "asset") -> str:
    asset_path = f"{prim_path}/{child_name}"
    prim_utils.create_prim(asset_path, usd_path=usd_path.as_posix())
    return asset_path


def _delete_prim_if_present(prim_path: str) -> None:
    if prim_utils.is_prim_path_valid(prim_path):
        prim_utils.delete_prim(prim_path)
    _clear_cached_prim_view(prim_path)


def _left_fingertip_body_index(robot: Articulation) -> int:
    cache_key = id(robot)
    if cache_key not in _LEFT_TIP_BODY_INDEX_CACHE:
        body_indices, _ = robot.find_bodies("left_inner_finger_pad", preserve_order=True)
        if not body_indices:
            raise ValueError("Failed to resolve left fingertip body index from articulation body names.")
        _LEFT_TIP_BODY_INDEX_CACHE[cache_key] = int(body_indices[0])
    return _LEFT_TIP_BODY_INDEX_CACHE[cache_key]


def mount_single_side_connector_only() -> dict[str, str]:
    usd_paths = ensure_phase2_mount_usds()
    _ensure_visual_group_paths()
    _delete_prim_if_present(LEFT_CONNECTOR_PRIM_PATH)
    _spawn_mount_xform(
        LEFT_CONNECTOR_PRIM_PATH,
        translation=(0.0, 0.0, 0.0),
        orientation=MOUNT_IDENTITY_QUAT_WXYZ,
    )
    _spawn_referenced_asset(LEFT_CONNECTOR_PRIM_PATH, usd_paths["connector"], child_name="connector_visual")
    return {"connector": LEFT_CONNECTOR_PRIM_PATH}


def mount_single_side_gsmini() -> dict[str, str]:
    usd_paths = ensure_phase2_mount_usds()
    _ensure_visual_group_paths()

    _delete_prim_if_present(LEFT_GELPAD_PRIM_PATH)
    _delete_prim_if_present(LEFT_CASE_PRIM_PATH)
    _delete_prim_if_present(LEFT_CONNECTOR_PRIM_PATH)
    _spawn_mount_xform(
        LEFT_CONNECTOR_PRIM_PATH,
        translation=(0.0, 0.0, 0.0),
        orientation=MOUNT_IDENTITY_QUAT_WXYZ,
    )
    _spawn_referenced_asset(LEFT_CONNECTOR_PRIM_PATH, usd_paths["connector"], child_name="connector_visual")

    _spawn_mount_xform(
        LEFT_CASE_PRIM_PATH,
        translation=(0.0, 0.0, 0.0),
        orientation=MOUNT_IDENTITY_QUAT_WXYZ,
    )
    _spawn_referenced_asset(LEFT_CASE_PRIM_PATH, usd_paths["case"], child_name="case_visual")

    _spawn_mount_xform(
        LEFT_GELPAD_PRIM_PATH,
        translation=(0.0, 0.0, 0.0),
        orientation=MOUNT_IDENTITY_QUAT_WXYZ,
    )
    _spawn_referenced_asset(LEFT_GELPAD_PRIM_PATH, usd_paths["gelpad"], child_name="gelpad_visual")

    return {
        "connector": LEFT_CONNECTOR_PRIM_PATH,
        "case": LEFT_CASE_PRIM_PATH,
        "gelpad": LEFT_GELPAD_PRIM_PATH,
    }


def _xform_prim(prim_path: str) -> XFormPrim:
    prim = _XFORM_VIEW_CACHE.get(prim_path)
    if prim is None:
        prim = XFormPrim(prim_paths_expr=[prim_path], name=prim_path.replace('/', '_'))
        prim.initialize()
        _XFORM_VIEW_CACHE[prim_path] = prim
    return prim


def _world_pose(prim_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    prim = _xform_prim(prim_path)
    pos, quat = prim.get_world_poses()
    return pos[0].clone(), quat[0].clone()


def _set_world_pose(prim_path: str, position: torch.Tensor, orientation: torch.Tensor) -> None:
    prim = _xform_prim(prim_path)
    prim.set_world_poses(positions=position.unsqueeze(0), orientations=orientation.unsqueeze(0))


def _translation_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.max(torch.abs(a - b)).item())


def read_mount_debug_world_positions(prim_paths: dict[str, str]) -> dict[str, list[float]]:
    positions: dict[str, list[float]] = {}
    for name, path in prim_paths.items():
        if not prim_utils.is_prim_path_valid(path):
            continue
        pos, _ = _world_pose(path)
        positions[name] = [float(v) for v in pos.tolist()]
    return positions


def read_left_fingertip_world_pose(robot: Articulation | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    if robot is None:
        return _world_pose(LEFT_FINGER_LINK_PATH)
    tip_body_idx = _left_fingertip_body_index(robot)
    tip_state = robot.data.body_state_w[:, tip_body_idx, :7]
    return tip_state[0, 0:3].clone(), tip_state[0, 3:7].clone()


def _compute_left_mount_pose_batches(
    tip_pos_batch: torch.Tensor,
    tip_quat_batch: torch.Tensor,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    device = tip_pos_batch.device
    dtype = tip_pos_batch.dtype
    connector_pos_batch, connector_quat_batch = math_utils.combine_frame_transforms(
        tip_pos_batch,
        tip_quat_batch,
        torch.tensor([CONNECTOR_LOCAL_TRANSLATION], device=device, dtype=dtype),
        torch.tensor([CONNECTOR_LOCAL_QUAT_WXYZ], device=device, dtype=dtype),
    )
    case_pos_batch, case_quat_batch = math_utils.combine_frame_transforms(
        connector_pos_batch,
        connector_quat_batch,
        torch.tensor([CASE_LOCAL_TRANSLATION], device=device, dtype=dtype),
        torch.tensor([CASE_LOCAL_QUAT_WXYZ], device=device, dtype=dtype),
    )
    gelpad_pos_batch, gelpad_quat_batch = math_utils.combine_frame_transforms(
        case_pos_batch,
        case_quat_batch,
        torch.tensor([GELPAD_LOCAL_TRANSLATION], device=device, dtype=dtype),
        torch.tensor([GELPAD_LOCAL_QUAT_WXYZ], device=device, dtype=dtype),
    )
    return {
        "connector": (connector_pos_batch, connector_quat_batch),
        "case": (case_pos_batch, case_quat_batch),
        "gelpad": (gelpad_pos_batch, gelpad_quat_batch),
    }


def sync_left_mount_to_runtime_fingertip(
    robot: Articulation | None = None,
    include_gsmini: bool = False,
) -> dict[str, list[float]]:
    fingertip_pos, fingertip_quat = read_left_fingertip_world_pose(robot)
    if robot is None:
        tip_pos_batch = fingertip_pos.unsqueeze(0)
        tip_quat_batch = fingertip_quat.unsqueeze(0)
    else:
        tip_body_idx = _left_fingertip_body_index(robot)
        tip_state = robot.data.body_state_w[:, tip_body_idx, :7]
        tip_pos_batch = tip_state[:, 0:3]
        tip_quat_batch = tip_state[:, 3:7]

    mount_poses = _compute_left_mount_pose_batches(tip_pos_batch, tip_quat_batch)
    connector_pos_batch, connector_quat_batch = mount_poses["connector"]
    connector_pos = connector_pos_batch[0]
    connector_quat = connector_quat_batch[0]
    if prim_utils.is_prim_path_valid(LEFT_CONNECTOR_PRIM_PATH):
        _set_world_pose(LEFT_CONNECTOR_PRIM_PATH, connector_pos, connector_quat)

    if include_gsmini:
        case_pos_batch, case_quat_batch = mount_poses["case"]
        gelpad_pos_batch, gelpad_quat_batch = mount_poses["gelpad"]
        if prim_utils.is_prim_path_valid(LEFT_CASE_PRIM_PATH):
            _set_world_pose(LEFT_CASE_PRIM_PATH, case_pos_batch[0], case_quat_batch[0])
        if prim_utils.is_prim_path_valid(LEFT_GELPAD_PRIM_PATH):
            _set_world_pose(LEFT_GELPAD_PRIM_PATH, gelpad_pos_batch[0], gelpad_quat_batch[0])

    visible_paths = {"connector": LEFT_CONNECTOR_PRIM_PATH}
    if include_gsmini:
        visible_paths.update(
            {
                "case": LEFT_CASE_PRIM_PATH,
                "gelpad": LEFT_GELPAD_PRIM_PATH,
            }
        )
    return read_mount_debug_world_positions(visible_paths)


def run_single_side_mount_validation(
    sim,
    robot: Articulation,
    banana: RigidObject,
    origin: torch.Tensor,
    reset_trials: int = DEFAULT_RESET_TRIALS,
    follow_steps: int = DEFAULT_FOLLOW_STEPS,
) -> dict[str, object]:
    results: dict[str, object] = {
        "source_of_truth": source_of_truth_summary(),
        "single_side_mount": {"passed": False},
    }
    mount_single_side_gsmini()

    reset_records = []
    reference_positions = None
    for trial in range(1, reset_trials + 1):
        reset_scene(sim, robot, banana, origin)
        step_scene(sim, robot, banana, joint_target=build_reset_joint_target(robot), steps=DEFAULT_SETTLE_STEPS)
        sync_left_mount_to_runtime_fingertip(robot, include_gsmini=True)
        positions = {
            name: _world_pose(path)[0]
            for name, path in {
                "connector": LEFT_CONNECTOR_PRIM_PATH,
                "case": LEFT_CASE_PRIM_PATH,
                "gelpad": LEFT_GELPAD_PRIM_PATH,
            }.items()
        }
        if reference_positions is None:
            reference_positions = {name: pos.clone() for name, pos in positions.items()}
        drift = {
            name: _translation_error(pos, reference_positions[name])
            for name, pos in positions.items()
        }
        reset_records.append(
            {
                "trial": trial,
                "positions": {name: [float(v) for v in pos.tolist()] for name, pos in positions.items()},
                "max_drift_m": max(drift.values()),
                "passed": max(drift.values()) <= RESET_TRANSLATION_TOLERANCE_M,
            }
        )
        if not reset_records[-1]["passed"]:
            results["single_side_mount"] = {
                "passed": False,
                "reset_records": reset_records,
                "reason": "mounted parts drift across resets",
            }
            return results

    def _follow_state_errors() -> dict[str, float]:
        tip_body_idx = _left_fingertip_body_index(robot)
        tip_state = robot.data.body_state_w[:, tip_body_idx, :7]
        expected_poses = _compute_left_mount_pose_batches(tip_state[:, 0:3], tip_state[:, 3:7])
        connector_error = _translation_error(_world_pose(LEFT_CONNECTOR_PRIM_PATH)[0], expected_poses["connector"][0][0])
        case_error = _translation_error(_world_pose(LEFT_CASE_PRIM_PATH)[0], expected_poses["case"][0][0])
        gelpad_error = _translation_error(_world_pose(LEFT_GELPAD_PRIM_PATH)[0], expected_poses["gelpad"][0][0])
        return {
            "connector_m": connector_error,
            "case_m": case_error,
            "gelpad_m": gelpad_error,
            "max_m": max(connector_error, case_error, gelpad_error),
        }

    reset_scene(sim, robot, banana, origin)
    step_scene(sim, robot, banana, joint_target=build_reset_joint_target(robot), steps=DEFAULT_SETTLE_STEPS)
    sync_left_mount_to_runtime_fingertip(robot, include_gsmini=True)
    follow_error_reset = _follow_state_errors()

    close_target = build_gripper_joint_target(robot, closed=True)
    step_scene(sim, robot, banana, joint_target=close_target, steps=follow_steps)
    sync_left_mount_to_runtime_fingertip(robot, include_gsmini=True)
    follow_error_close = _follow_state_errors()

    open_target = build_gripper_joint_target(robot, closed=False)
    step_scene(sim, robot, banana, joint_target=open_target, steps=follow_steps)
    sync_left_mount_to_runtime_fingertip(robot, include_gsmini=True)
    follow_error_open = _follow_state_errors()

    passed = max(
        follow_error_reset["max_m"],
        follow_error_close["max_m"],
        follow_error_open["max_m"],
    ) <= FOLLOW_TRANSLATION_TOLERANCE_M
    results["single_side_mount"] = {
        "passed": passed,
        "reset_records": reset_records,
        "follow_error_reset_m": follow_error_reset,
        "follow_error_close_m": follow_error_close,
        "follow_error_open_m": follow_error_open,
        "left_paths": {
            "finger": LEFT_FINGER_LINK_PATH,
            "connector": LEFT_CONNECTOR_PRIM_PATH,
            "case": LEFT_CASE_PRIM_PATH,
            "gelpad": LEFT_GELPAD_PRIM_PATH,
        },
        "relative_mount_vector_m": list(CONNECTOR_LOCAL_TRANSLATION),
    }
    return results
