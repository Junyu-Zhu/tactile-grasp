"""Runtime TacEx GelSight Mini configuration and live capture helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

from protac.tactile.gsmini_contract import (
    OFFICIAL_GELPAD_TO_CAMERA_MIN_DISTANCE_M,
    OFFICIAL_TACTILE_RESOLUTION,
    RUNTIME_CAMERA_CLIPPING_RANGE_M,
    RUNTIME_EFFECTIVE_GELPAD_HEIGHT_M,
)
from protac.tactile.mount import (
    LEFT_SENSOR_CASE_PRIM_PATH,
    RIGHT_SENSOR_CASE_PRIM_PATH,
    SENSOR_CAMERA_PRIM_PATH_APPENDIX,
    TACEX_GELSIGHT_CALIB_DIR,
    sensor_prim_paths,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACE_ROOT = PROJECT_ROOT.parent
_TACEX_SOURCE_DIRS = (
    _WORKSPACE_ROOT / "TacEx" / "source" / "tacex",
    _WORKSPACE_ROOT / "TacEx" / "source" / "tacex_assets",
    _WORKSPACE_ROOT / "TacEx" / "source" / "tacex_tasks",
)

DEFAULT_TACTILE_RESOLUTION = OFFICIAL_TACTILE_RESOLUTION
DEFAULT_SENSOR_CAMERA_CLIPPING_RANGE = RUNTIME_CAMERA_CLIPPING_RANGE_M


def configure_tacex_python_paths() -> list[str]:
    """Make TacEx source packages importable when this script is run from tactile_grasp."""

    added: list[str] = []
    for path in _TACEX_SOURCE_DIRS:
        if path.is_dir() and path.as_posix() not in sys.path:
            sys.path.insert(0, path.as_posix())
            added.append(path.as_posix())
    return added


def build_gsmini_cfg(
    side: str = "left",
    *,
    device: str = "cuda",
    resolution: tuple[int, int] = DEFAULT_TACTILE_RESOLUTION,
    clipping_range: tuple[float, float] = DEFAULT_SENSOR_CAMERA_CLIPPING_RANGE,
    include_camera_depth: bool = True,
    include_camera_rgb: bool = False,
    debug_vis: bool = False,
):
    """Build a TacEx GelSightMiniCfg for one runtime sensor prim.

    Start from TacEx's own GPU-Taxim preset and preserve its renderer,
    calibration images, camera frame, and nominal camera-to-gel distance.
    Runtime paths/outputs are selected here, while the far clipping plane and
    effective optical gel thickness use measured rigid-contact calibrations.
    Those two optical adjustments do not change the canonical URDF geometry.
    """

    configure_tacex_python_paths()
    from tacex_assets import GELSIGHT_MINI_TAXIM_CFG  # type: ignore

    prim_paths = {"left": LEFT_SENSOR_CASE_PRIM_PATH, "right": RIGHT_SENSOR_CASE_PRIM_PATH}
    if side not in prim_paths:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    camera_data_types = ["depth"]
    data_types = ["tactile_rgb"]
    if include_camera_depth:
        data_types.append("camera_depth")
    if include_camera_rgb:
        camera_data_types.append("rgb")
        data_types.append("camera_rgb")

    cfg = GELSIGHT_MINI_TAXIM_CFG.replace(
        prim_path=prim_paths[side],
        sensor_camera_cfg=GELSIGHT_MINI_TAXIM_CFG.sensor_camera_cfg.replace(
            prim_path_appendix=SENSOR_CAMERA_PRIM_PATH_APPENDIX,
            update_period=0,
            resolution=resolution,
            data_types=camera_data_types,
            clipping_range=clipping_range,
        ),
        update_period=0.01,
        data_types=data_types,
        marker_motion_sim_cfg=None,
        compute_indentation_depth_class="optical_sim",
        device=device,
        debug_vis=debug_vis,
    )
    cfg.optical_sim_cfg = cfg.optical_sim_cfg.replace(
        calib_folder_path=TACEX_GELSIGHT_CALIB_DIR.as_posix(),
        gelpad_height=RUNTIME_EFFECTIVE_GELPAD_HEIGHT_M,
        gelpad_to_camera_min_distance=OFFICIAL_GELPAD_TO_CAMERA_MIN_DISTANCE_M,
        with_shadow=False,
        tactile_img_res=resolution,
        device=device,
    )
    return cfg


def initialize_gsmini_sensor(cfg):
    configure_tacex_python_paths()
    from tacex import GelSightSensor  # type: ignore

    sensor = GelSightSensor(cfg)
    # The sensor is not owned by an InteractiveScene, so initialize it directly
    # after its prim exists.
    if not getattr(sensor, "_is_initialized", False):
        sensor._initialize_impl()  # noqa: SLF001 - IsaacLab standalone initialization hook.
        sensor._is_initialized = True  # noqa: SLF001
    return sensor


def position_tactile_debug_windows(
    tactile_sensors: list[tuple[str, Any]],
    *,
    anchor_x: int = 24,
    anchor_y: int = 96,
    horizontal_gap: int = 24,
) -> dict[str, dict[str, int | bool]]:
    """Place TacEx-created native Taxim windows side-by-side.

    The image providers and rendering stay entirely inside TacEx.  This helper
    only moves the existing ``omni.ui.Window`` objects after TacEx's post-update
    callback creates them, preventing the left and right windows from opening
    at the same default location.
    """

    result: dict[str, dict[str, int | bool]] = {}
    next_x = int(anchor_x)
    for side, sensor in tactile_sensors:
        simulator = getattr(sensor, "optical_simulator", None)
        windows = getattr(simulator, "_debug_windows", {})
        window = windows.get("0") if isinstance(windows, dict) else None
        if window is None:
            result[side] = {"positioned": False, "x": next_x, "y": int(anchor_y)}
            next_x += int(sensor.cfg.optical_sim_cfg.tactile_img_res[0]) + horizontal_gap
            continue
        window.position_x = next_x
        window.position_y = int(anchor_y)
        result[side] = {"positioned": True, "x": next_x, "y": int(anchor_y)}
        next_x += int(window.width) + int(horizontal_gap)
    return result


def enable_tactile_debug_windows(
    sides: tuple[str, ...],
    *,
    include_camera_depth: bool = False,
) -> dict[str, dict[str, bool]]:
    """Enable TacEx's built-in live windows for the selected sensor roots."""

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    paths = sensor_prim_paths()
    result: dict[str, dict[str, bool]] = {}
    for side in sides:
        sensor_prim = stage.GetPrimAtPath(paths[side]["sensor"])
        if not sensor_prim.IsValid():
            raise RuntimeError(f"Cannot enable TacEx debug output; missing sensor prim: {paths[side]['sensor']}")
        requested = ["debug_tactile_rgb"]
        if include_camera_depth:
            requested.append("debug_camera_depth")
        result[side] = {}
        for attr_name in requested:
            attr = sensor_prim.GetAttribute(attr_name)
            enabled = bool(attr)
            if attr:
                attr.Set(True)
            result[side][attr_name] = enabled
    return result


def update_tactile_sensor(
    sensor,
    sim,
    *,
    dt: float,
    render_ticks: int = 3,
) -> dict[str, torch.Tensor]:
    # Give RTX/TiledCamera a few render ticks after adding/removing validation
    # probes; otherwise a capture can reuse the previous frame in headless mode.
    for _ in range(max(0, render_ticks)):
        sim.render()
    sensor.update(dt=dt, force_recompute=True)
    return sensor.data.output
