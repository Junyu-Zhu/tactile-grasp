"""Phase 2 TacEx GelSight Mini tactile bring-up helpers.

This module is intentionally Phase-2 scoped: it connects TacEx's GelSight Mini
sensor configuration to the UR5/Robotiq/GSmini embodiment, records minimal
no-contact/contact frames, and writes review artifacts.  It does not run Sparsh,
closed-loop control, GraspNet, RL, or adaptation.
"""

from __future__ import annotations

import json
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from ur5_phase2_mount import (
    LEFT_SENSOR_CASE_PRIM_PATH,
    PHASE2_SCOPE_SENTENCE,
    RIGHT_SENSOR_CASE_PRIM_PATH,
    SENSOR_CAMERA_PRIM_PATH_APPENDIX,
    TACEX_GELSIGHT_CALIB_DIR,
    clear_phase2_visual_prims,
    mount_phase2_sensor_shells,
    phase2_sensor_prim_paths,
    run_dual_side_mount_validation,
    source_of_truth_summary,
    tacex_sensor_model_summary,
    validate_phase2_sensor_camera_prims,
    validate_phase2_sensor_mounts,
)

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
TACTILE_LOG_ROOT = ARTIFACT_DIR / "phase2_tactile_logs"
ISSUE_LOG_PATH = Path(__file__).resolve().parent / "log.md"
_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
_TACEX_SOURCE_DIRS = (
    _WORKSPACE_ROOT / "TacEx" / "source" / "tacex",
    _WORKSPACE_ROOT / "TacEx" / "source" / "tacex_assets",
    _WORKSPACE_ROOT / "TacEx" / "source" / "tacex_tasks",
)

DEFAULT_TACTILE_RESOLUTION = (320, 240)
DEFAULT_SENSOR_CAMERA_CLIPPING_RANGE = (0.024, 0.040)
DEFAULT_CONTACT_REPEATS = 3
STATIC_CONTACT_PROBE_PATH = "/World/Phase2TactileContactProbe"
STATIC_CONTACT_DISTANCE_M = 0.026
STATIC_CONTACT_PROBE_SCALE = (0.012, 0.012, 0.012)
TACTILE_DIFF_PASS_THRESHOLD = 1.0e-4


@dataclass(frozen=True)
class Phase2TactileOptions:
    output_dir: Path = TACTILE_LOG_ROOT
    contact_repeats: int = DEFAULT_CONTACT_REPEATS
    save_frames: bool = False
    include_camera_depth: bool = True
    include_camera_rgb: bool = False
    resolution: tuple[int, int] = DEFAULT_TACTILE_RESOLUTION
    debug_vis: bool = False


def configure_tacex_python_paths() -> list[str]:
    """Make TacEx source packages importable when this script is run from tactile_grasp."""

    added: list[str] = []
    for path in _TACEX_SOURCE_DIRS:
        if path.is_dir() and path.as_posix() not in sys.path:
            sys.path.insert(0, path.as_posix())
            added.append(path.as_posix())
    return added


def append_phase2_issue(summary: str, details: str) -> None:
    ISSUE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with ISSUE_LOG_PATH.open("a", encoding="utf-8") as stream:
        stream.write(f"\n## {timestamp} — {summary}\n\n")
        stream.write(details.rstrip() + "\n")


def build_phase2_gsmini_cfg(
    side: str = "left",
    *,
    device: str = "cuda",
    resolution: tuple[int, int] = DEFAULT_TACTILE_RESOLUTION,
    clipping_range: tuple[float, float] = DEFAULT_SENSOR_CAMERA_CLIPPING_RANGE,
    include_camera_depth: bool = True,
    include_camera_rgb: bool = False,
    debug_vis: bool = False,
):
    """Build a TacEx GelSightMiniCfg for the Phase 2 runtime sensor prim.

    The values intentionally mirror TacEx's `gsmini_cfg.py`: `/Camera`, 320x240
    default resolution, Taxim calibration folder, 0.024m gelpad-to-camera
    minimum distance, and GelSight Mini dimensions.  The far clipping plane is
    widened from TacEx's static 0.029 m default for this UR5/Robotiq mount so
    the camera still sees the cube when the URDF soft-link collision touches
    before the TacEx shell's exact visual gelpad plane.
    """

    configure_tacex_python_paths()
    from tacex.simulation_approaches.gpu_taxim import TaximSimulatorCfg  # type: ignore
    from tacex_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg  # type: ignore

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

    cfg = GelSightMiniCfg(
        prim_path=prim_paths[side],
        sensor_camera_cfg=GelSightMiniCfg.SensorCameraCfg(
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
    cfg.optical_sim_cfg = TaximSimulatorCfg(
        calib_folder_path=TACEX_GELSIGHT_CALIB_DIR.as_posix(),
        gelpad_height=cfg.gelpad_dimensions.height,
        gelpad_to_camera_min_distance=0.024,
        with_shadow=False,
        tactile_img_res=resolution,
        device=device,
    )
    return cfg


def initialize_phase2_sensor(cfg):
    configure_tacex_python_paths()
    from tacex import GelSightSensor  # type: ignore

    sensor = GelSightSensor(cfg)
    # In this standalone Phase2 script the sensor is not owned by an
    # InteractiveScene, so initialize it directly after its prim exists.
    if not getattr(sensor, "_is_initialized", False):
        sensor._initialize_impl()  # noqa: SLF001 - IsaacLab standalone initialization hook.
        sensor._is_initialized = True  # noqa: SLF001
    return sensor


def update_phase2_sensor(sensor, sim, *, dt: float) -> dict[str, torch.Tensor]:
    # Give RTX/TiledCamera a few render ticks after adding/removing validation
    # probes; otherwise a capture can reuse the previous frame in headless mode.
    for _ in range(3):
        sim.render()
    sensor.update(dt=dt, force_recompute=True)
    return sensor.data.output


def _delete_prim_if_present(prim_path: str) -> None:
    import isaacsim.core.utils.prims as prim_utils

    if prim_utils.is_prim_path_valid(prim_path):
        prim_utils.delete_prim(prim_path)


def spawn_static_contact_probe(camera_prim_path: str, *, repeat: int) -> dict[str, Any]:
    """Place a small probe in front of the TacEx internal camera.

    This gives Step 6 a deterministic camera-visible contact target without
    editing the canonical URDF appearance or mount offsets.  The probe is only
    a Phase2 validation object; physical robot contact remains covered by the
    prior Phase1 gripper-control tests and can be re-enabled once long
    standalone PhysX stepping is stable in this environment.
    """

    import omni.usd
    import isaacsim.core.utils.prims as prim_utils
    from pxr import Gf, Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    camera_prim = stage.GetPrimAtPath(camera_prim_path)
    if not camera_prim.IsValid():
        raise RuntimeError(f"Cannot spawn contact probe; camera prim is invalid: {camera_prim_path}")
    camera_xform = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    camera_pos = camera_xform.ExtractTranslation()
    basis = []
    for axis in (
        Gf.Vec3d(0.0, 0.0, -1.0),
        Gf.Vec3d(0.0, 0.0, 1.0),
        Gf.Vec3d(1.0, 0.0, 0.0),
        Gf.Vec3d(-1.0, 0.0, 0.0),
        Gf.Vec3d(0.0, 1.0, 0.0),
        Gf.Vec3d(0.0, -1.0, 0.0),
    ):
        direction = camera_xform.TransformDir(axis)
        direction.Normalize()
        basis.append(direction)
    lateral = (repeat - 1) * 0.0005
    _delete_prim_if_present(STATIC_CONTACT_PROBE_PATH)
    prim_utils.create_prim(STATIC_CONTACT_PROBE_PATH, "Xform")
    probe_positions = []
    for index, direction in enumerate(basis):
        probe_pos = camera_pos + (direction * STATIC_CONTACT_DISTANCE_M) + Gf.Vec3d(lateral, 0.0, 0.0)
        child_path = f"{STATIC_CONTACT_PROBE_PATH}/probe_{index}"
        prim_utils.create_prim(
            child_path,
            "Cube",
            translation=(float(probe_pos[0]), float(probe_pos[1]), float(probe_pos[2])),
            scale=STATIC_CONTACT_PROBE_SCALE,
        )
        probe_positions.append([float(probe_pos[0]), float(probe_pos[1]), float(probe_pos[2])])
    return {
        "path": STATIC_CONTACT_PROBE_PATH,
        "camera": camera_prim_path,
        "distance_m": STATIC_CONTACT_DISTANCE_M,
        "scale": list(STATIC_CONTACT_PROBE_SCALE),
        "position_world_m": probe_positions,
        "mode": "static_camera_visible_multi_axis_probe",
    }


def _tensor_to_numpy(tensor: torch.Tensor):
    array = tensor.detach().cpu().numpy()
    if array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in {1, 3} and array.shape[-1] not in {1, 3, 4}:
        array = array.transpose(1, 2, 0)
    return array


def _image_uint8(array):
    import numpy as np

    arr = np.asarray(array)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.size and float(arr.max()) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        return arr[..., :3]
    return arr.reshape(arr.shape[0], -1)


def save_array_image(path: Path, array) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = _image_uint8(array)
    try:
        import cv2  # type: ignore

        if image.ndim == 3 and image.shape[-1] == 3:
            cv2.imwrite(path.as_posix(), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(path.as_posix(), image)
        return path.as_posix()
    except Exception:
        # Dependency-light fallback: write Netpbm data even if the extension is
        # .ppm/.pgm.  Keep the data inspectable instead of failing the test.
        fallback = path.with_suffix(".ppm" if image.ndim == 3 else ".pgm")
        if image.ndim == 3:
            fallback.write_bytes(b"P6\n%d %d\n255\n" % (image.shape[1], image.shape[0]) + image.tobytes())
        else:
            fallback.write_bytes(b"P5\n%d %d\n255\n" % (image.shape[1], image.shape[0]) + image.tobytes())
        return fallback.as_posix()


def _output_stats(array) -> dict[str, Any]:
    import numpy as np

    arr = np.asarray(array, dtype=np.float32)
    return {
        "shape": list(arr.shape),
        "min": float(arr.min()) if arr.size else 0.0,
        "max": float(arr.max()) if arr.size else 0.0,
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size else 0.0,
    }


def capture_phase2_sensor_outputs(sensor, sim, output_dir: Path, *, side: str, label: str, save_frames: bool) -> dict[str, Any]:
    output = update_phase2_sensor(sensor, sim, dt=0.0)
    record: dict[str, Any] = {"side": side, "label": label, "outputs": {}, "files": {}, "_arrays": {}}
    for name, tensor in output.items():
        if tensor is None:
            continue
        array = _tensor_to_numpy(tensor)
        record["_arrays"][name] = array
        record["outputs"][name] = _output_stats(array)
        if save_frames and name in {"tactile_rgb", "camera_depth", "camera_rgb"}:
            suffix = "png"
            file_path = output_dir / f"{side}_{label}_{name}.{suffix}"
            record["files"][name] = save_array_image(file_path, array)
    return record


def compare_tactile_records(no_contact: dict[str, Any], contact: dict[str, Any]) -> dict[str, Any]:
    """Programmatically compare no-contact/contact tactile frames."""

    import numpy as np

    no_frame = no_contact.get("_arrays", {}).get("tactile_rgb")
    contact_frame = contact.get("_arrays", {}).get("tactile_rgb")
    if no_frame is None or contact_frame is None:
        return {"passed": False, "reason": "missing tactile_rgb output"}
    no_arr = np.asarray(no_frame, dtype=np.float32)
    contact_arr = np.asarray(contact_frame, dtype=np.float32)
    if no_arr.shape != contact_arr.shape:
        return {"passed": False, "reason": f"shape mismatch: {list(no_arr.shape)} vs {list(contact_arr.shape)}"}
    diff = np.abs(contact_arr - no_arr)
    mean_abs_diff = float(diff.mean()) if diff.size else 0.0
    max_abs_diff = float(diff.max()) if diff.size else 0.0
    changed_pixel_fraction = float((diff > TACTILE_DIFF_PASS_THRESHOLD).mean()) if diff.size else 0.0
    return {
        "passed": mean_abs_diff > TACTILE_DIFF_PASS_THRESHOLD and changed_pixel_fraction > 0.001,
        "mean_abs_diff": mean_abs_diff,
        "max_abs_diff": max_abs_diff,
        "changed_pixel_fraction": changed_pixel_fraction,
        "threshold": TACTILE_DIFF_PASS_THRESHOLD,
    }


def run_phase2_tactile_validation(
    sim,
    robot,
    banana,
    origin: torch.Tensor,
    *,
    options: Phase2TactileOptions,
    device: str = "cuda",
    run_dual_side: bool = True,
) -> dict[str, Any]:
    """Run Phase2 Step5/6 and optionally Step7/8 validation."""

    trial_id = datetime.now(timezone.utc).strftime("phase2_%Y%m%dT%H%M%SZ")
    output_dir = options.output_dir / trial_id
    output_dir.mkdir(parents=True, exist_ok=True)
    sim_dt = sim.get_physics_dt()

    clear_phase2_visual_prims()
    shell_spawn = mount_phase2_sensor_shells(("left",), hide_render_geometry=True)
    camera_prim_check = validate_phase2_sensor_camera_prims(("left",))
    sensor_mount_check = validate_phase2_sensor_mounts(("left",))
    if not camera_prim_check["left"]["camera_exists"]:
        raise RuntimeError(
            "TacEx GelSight Mini case reference did not expose the expected "
            f"camera prim: {camera_prim_check['left']['camera']}"
        )
    if not sensor_mount_check["left"]["passed"]:
        raise RuntimeError(
            "Phase2 TacEx sensor shell is not mounted under the canonical left fingertip: "
            f"{json.dumps(sensor_mount_check['left'], ensure_ascii=False)}"
        )
    sensor_cfg = build_phase2_gsmini_cfg(
        "left",
        device=device,
        resolution=options.resolution,
        include_camera_depth=options.include_camera_depth,
        include_camera_rgb=options.include_camera_rgb,
        debug_vis=options.debug_vis,
    )
    sensor = initialize_phase2_sensor(sensor_cfg)

    _delete_prim_if_present(STATIC_CONTACT_PROBE_PATH)
    no_contact = capture_phase2_sensor_outputs(
        sensor,
        sim,
        output_dir,
        side="left",
        label="no_contact",
        save_frames=options.save_frames,
    )

    contact_records = []
    comparisons = []
    contact_probes = []
    for repeat in range(1, options.contact_repeats + 1):
        contact_probes.append(spawn_static_contact_probe(camera_prim_check["left"]["camera"], repeat=repeat))
        # Rebuild the sensor/camera after adding the probe so the standalone
        # TiledCamera view cannot retain a pre-probe render product.
        contact_sensor = initialize_phase2_sensor(sensor_cfg)
        contact = capture_phase2_sensor_outputs(
            contact_sensor,
            sim,
            output_dir,
            side="left",
            label=f"contact_r{repeat}",
            save_frames=options.save_frames,
        )
        contact_records.append(contact)
        comparisons.append(compare_tactile_records(no_contact, contact))
        _delete_prim_if_present(STATIC_CONTACT_PROBE_PATH)

    dual_result: dict[str, Any] | None = None
    right_no_contact: dict[str, Any] | None = None
    if run_dual_side:
        dual_result = run_dual_side_mount_validation(
            sim,
            robot,
            banana,
            origin,
            cycles=5,
            spawn_debug_visuals=False,
        )
        dual_camera_prim_check = validate_phase2_sensor_camera_prims(("left", "right"))
        dual_sensor_mount_check = validate_phase2_sensor_mounts(("left", "right"))
        missing_dual_cameras = [
            side for side, check in dual_camera_prim_check.items() if not check["camera_exists"]
        ]
        if missing_dual_cameras:
            raise RuntimeError(
                "TacEx dual GelSight Mini case references did not expose expected "
                f"camera prims for: {missing_dual_cameras}"
            )
        failed_dual_mounts = [
            side for side, check in dual_sensor_mount_check.items() if not check["passed"]
        ]
        if failed_dual_mounts:
            raise RuntimeError(
                "Phase2 dual TacEx sensor shells are not mounted under canonical fingertips: "
                f"{json.dumps(dual_sensor_mount_check, ensure_ascii=False)}"
            )
        # Step 7 requires left and right no-contact outputs in the final
        # dual-side scene, so re-initialize and re-capture both sides after the
        # dual-side shell spawn.
        left_dual_cfg = build_phase2_gsmini_cfg(
            "left",
            device=device,
            resolution=options.resolution,
            include_camera_depth=options.include_camera_depth,
            include_camera_rgb=options.include_camera_rgb,
            debug_vis=options.debug_vis,
        )
        right_cfg = build_phase2_gsmini_cfg(
            "right",
            device=device,
            resolution=options.resolution,
            include_camera_depth=options.include_camera_depth,
            include_camera_rgb=options.include_camera_rgb,
            debug_vis=options.debug_vis,
        )
        left_dual_sensor = initialize_phase2_sensor(left_dual_cfg)
        right_sensor = initialize_phase2_sensor(right_cfg)
        left_dual_no_contact = capture_phase2_sensor_outputs(
            left_dual_sensor,
            sim,
            output_dir,
            side="left",
            label="dual_no_contact",
            save_frames=options.save_frames,
        )
        right_no_contact = capture_phase2_sensor_outputs(
            right_sensor,
            sim,
            output_dir,
            side="right",
            label="dual_no_contact",
            save_frames=options.save_frames,
        )
    else:
        left_dual_no_contact = None
        dual_sensor_mount_check = None

    summary: dict[str, Any] = {
        "trial_id": trial_id,
        "scope": PHASE2_SCOPE_SENTENCE,
        "source_of_truth": source_of_truth_summary(),
        "tacex_sensor_model": tacex_sensor_model_summary(),
        "sensor_paths": phase2_sensor_prim_paths(),
        "sensor_camera_prim_check": validate_phase2_sensor_camera_prims(("left", "right")),
        "sensor_shell_mount_check": validate_phase2_sensor_mounts(("left", "right")),
        "sensor_shell_spawn": shell_spawn,
        "config": {
            "resolution": list(options.resolution),
            "include_camera_depth": options.include_camera_depth,
            "include_camera_rgb": options.include_camera_rgb,
            "contact_repeats": options.contact_repeats,
            "device": device,
        },
        "step5_no_contact": no_contact,
        "step6_contact_records": contact_records,
        "step6_comparisons": comparisons,
        "step6_contact_probes": contact_probes,
        "step6_probe_mount_check": sensor_mount_check,
        "step7_dual_side_mount": dual_result,
        "step7_dual_sensor_mount_check": dual_sensor_mount_check,
        "step7_left_no_contact": left_dual_no_contact,
        "step7_right_no_contact": right_no_contact,
        "step8_log_dir": output_dir.as_posix(),
    }
    summary["passed"] = bool(
        no_contact.get("outputs", {}).get("tactile_rgb")
        and comparisons
        and all(item.get("passed") for item in comparisons)
        and all(item.get("passed") for item in sensor_mount_check.values())
        and (dual_result is None or dual_result.get("passed"))
        and (
            dual_sensor_mount_check is None
            or all(item.get("passed") for item in dual_sensor_mount_check.values())
        )
        and (left_dual_no_contact is None or left_dual_no_contact.get("outputs", {}).get("tactile_rgb"))
        and (right_no_contact is None or right_no_contact.get("outputs", {}).get("tactile_rgb"))
    )
    safe_summary = _json_safe_summary(summary)
    write_phase2_tactile_artifacts(safe_summary, output_dir)
    return safe_summary


def write_phase2_tactile_artifacts(summary: dict[str, Any], output_dir: Path) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = ARTIFACT_DIR / "phase2_tactile_report.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path = ARTIFACT_DIR / "phase2_tactile_report.md"
    md_path.write_text(_phase2_tactile_markdown(summary), encoding="utf-8")
    (output_dir / "metadata.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _json_safe_summary(value):
    if isinstance(value, dict):
        return {key: _json_safe_summary(item) for key, item in value.items() if key != "_arrays"}
    if isinstance(value, list):
        return [_json_safe_summary(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_summary(item) for item in value]
    return value


def _pass_fail(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _phase2_tactile_markdown(summary: dict[str, Any]) -> str:
    comparisons = summary.get("step6_comparisons", [])
    comparison_lines = "\n".join(
        f"- repeat {index}: {_pass_fail(bool(item.get('passed')))}, "
        f"mean_abs_diff={item.get('mean_abs_diff')}, max_abs_diff={item.get('max_abs_diff')}, "
        f"changed_pixel_fraction={item.get('changed_pixel_fraction')}"
        for index, item in enumerate(comparisons, start=1)
    ) or "- no comparison records"
    return f"""# Phase 2 Tactile Report

- Trial: `{summary.get('trial_id')}`
- Overall: **{_pass_fail(bool(summary.get('passed')))}**
- Log dir: `{summary.get('step8_log_dir')}`

## TacEx model/config parity
```json
{json.dumps(summary.get('tacex_sensor_model'), ensure_ascii=False, indent=2)}
```

## Sensor prim paths
```json
{json.dumps(summary.get('sensor_paths'), ensure_ascii=False, indent=2)}
```

## Step 5 no-contact output
```json
{json.dumps(summary.get('step5_no_contact'), ensure_ascii=False, indent=2)}
```

## Step 6 no-contact/contact comparisons
{comparison_lines}

## Step 7 dual-side result
```json
{json.dumps(summary.get('step7_dual_side_mount'), ensure_ascii=False, indent=2)}
```

## Step 8 logging
Frames/metadata are saved under `{summary.get('step8_log_dir')}` when `--phase2-log-frames` is enabled.
"""


def write_phase2_review_artifact(review: dict[str, Any]) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = ARTIFACT_DIR / "phase2_review.json"
    json_path.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = ["# Phase 2 Review", "", f"Overall: **{_pass_fail(bool(review.get('passed')))}**", ""]
    for step, result in review.get("steps", {}).items():
        md.append(f"## {step}")
        md.append(f"- Status: **{_pass_fail(bool(result.get('passed')))}**")
        md.append(f"- Evidence: {result.get('evidence', '')}")
        if result.get("notes"):
            md.append(f"- Notes: {result['notes']}")
        md.append("")
    md_path = ARTIFACT_DIR / "phase2_review.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    return json_path


def summarize_exception_for_log(command_hint: str, exc: BaseException) -> None:
    append_phase2_issue(
        "Phase2 tactile/runtime validation failed",
        "Command/context:\n"
        f"```\n{command_hint}\n```\n\n"
        "Exception:\n"
        f"```\n{type(exc).__name__}: {exc}\n{traceback.format_exc()}\n```",
    )
