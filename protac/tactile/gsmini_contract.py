"""Single source of truth for the canonical GSmini/TacEx integration.

The URDF owns robot, connector, case, gelpad and contact geometry.  TacEx's
``Sensor.usd`` is a detached render-only camera asset whose pose is synchronized
from each canonical case body.  This is required because a camera added below
an articulation after startup does not reliably inherit the live PhysX pose.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent

GELSIGHT_MINI_ASSET_DIR = (
    WORKSPACE_ROOT
    / "TacEx"
    / "source"
    / "tacex_assets"
    / "tacex_assets"
    / "data"
    / "Sensors"
    / "GelSight_Mini"
)
GELSIGHT_MINI_SENSOR_USD = GELSIGHT_MINI_ASSET_DIR / "Sensor.usd"
GELSIGHT_MINI_CASE_USD = GELSIGHT_MINI_ASSET_DIR / "Case.usd"
GELSIGHT_MINI_GELPAD_USD = GELSIGHT_MINI_ASSET_DIR / "Gelpad_low_res.usd"
GELSIGHT_MINI_CALIB_DIR = GELSIGHT_MINI_ASSET_DIR / "calibs" / "640x480"

OFFICIAL_TACTILE_RESOLUTION = (320, 240)
OFFICIAL_CAMERA_CLIPPING_RANGE_M = (0.024, 0.029)
# The rigid URDF contact surface and PhysX contact offset can leave the cube
# about 0.5--2 mm beyond TacEx's nominal 29 mm optical far plane.  Extend only
# the runtime far plane enough to observe that surface; all other Taxim
# geometry/calibration values remain the upstream defaults.
RUNTIME_CAMERA_CLIPPING_RANGE_M = (0.024, 0.032)
OFFICIAL_GELPAD_TO_CAMERA_MIN_DISTANCE_M = 0.024
# Camera-derived cube contact spans about 28.8--29.7 mm because the rigid URDF
# contact shell and PhysX contact offset sit beyond TacEx's nominal 28.5 mm gel
# surface.  This effective optical thickness moves the optical reference to
# 29.2 mm, so the nearer measured contact pixels become a small Taxim
# indentation without changing URDF geometry.
RUNTIME_EFFECTIVE_GELPAD_HEIGHT_M = 0.0052

# ``GSmini_base_link.stl`` is the local, baked-frame version of ``base_link.STL``.
# A rigid fit over all 35,634 STL vertices gives this registration with
# max absolute residual < 5e-9 m.  TacEx Sensor.usd uses the source CAD frame,
# so applying this transform makes its case/camera/gelpad coincide with the
# canonical URDF's visible GSmini frame.
SENSOR_ASSET_TO_CASE_LINK_TRANSLATION_M = (
    -0.001724479720,
    0.003719151132,
    -0.034950183704,
)
SENSOR_ASSET_TO_CASE_LINK_QUAT_WXYZ = (0.5, 0.5, 0.5, -0.5)
# Authored on ``Sensor.usd`` /Camera: 180 degrees around source-frame X,
# making the OpenGL -Z view direction point along source-frame +Z.
SENSOR_CAMERA_LOCAL_QUAT_WXYZ = (0.0, 1.0, 0.0, 0.0)
# ``CameraData.quat_w_world`` is converted by IsaacLab from the authored
# OpenGL/USD camera frame to its world-camera convention (+X forward, +Z up).
# Keep this distinct from the authored Camera prim's -Z forward axis.
ISAACLAB_CAMERA_DATA_FORWARD_AXIS = (1.0, 0.0, 0.0)

SENSOR_CHILD_NAME = "tacex_gelsight_sensor"
RUNTIME_SENSOR_SCOPE = "/World/Phase3TacExSensors"
SENSOR_CAMERA_PRIM_PATH_APPENDIX = "/Camera"
SENSOR_GELPAD_PRIM_PATH_APPENDIX = "/Gelpad_low_res"


def canonical_robot_prim_path(*, origin_index: int = 1) -> str:
    return f"/World/Origin{origin_index}/Robot"


def sensor_prim_paths(*, origin_index: int = 1) -> dict[str, dict[str, str]]:
    """Return canonical case links and detached TacEx prims for both fingers."""

    robot_path = canonical_robot_prim_path(origin_index=origin_index)
    sensor_scope = f"{RUNTIME_SENSOR_SCOPE}/Origin{origin_index}"
    result: dict[str, dict[str, str]] = {}
    for side in ("left", "right"):
        case_link = f"{robot_path}/{side}_gelsight_mini_case"
        gelpad_link = f"{robot_path}/{side}_gelsight_mini_gelpad"
        sensor_path = f"{sensor_scope}/{side}_{SENSOR_CHILD_NAME}"
        result[side] = {
            "canonical_case_link": case_link,
            "canonical_gelpad_link": gelpad_link,
            "runtime_sensor_scope": sensor_scope,
            "sensor": sensor_path,
            # Compatibility aliases retained in existing Phase3 v1 artifacts.
            "case": sensor_path,
            "gelpad": f"{sensor_path}{SENSOR_GELPAD_PRIM_PATH_APPENDIX}",
            "camera": f"{sensor_path}{SENSOR_CAMERA_PRIM_PATH_APPENDIX}",
        }
    return result
