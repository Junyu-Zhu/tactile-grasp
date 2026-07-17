"""Mount and synchronize detached TacEx render sensors to canonical GSmini links."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
import isaacsim.core.utils.prims as prim_utils

from protac.tactile.gsmini_contract import (
    GELSIGHT_MINI_CALIB_DIR,
    GELSIGHT_MINI_SENSOR_USD,
    RUNTIME_CAMERA_CLIPPING_RANGE_M,
    SENSOR_ASSET_TO_CASE_LINK_QUAT_WXYZ,
    SENSOR_ASSET_TO_CASE_LINK_TRANSLATION_M,
    SENSOR_CAMERA_LOCAL_QUAT_WXYZ,
    SENSOR_CAMERA_PRIM_PATH_APPENDIX,
    sensor_prim_paths as contract_sensor_prim_paths,
)

TACEX_GELSIGHT_SENSOR_USD = GELSIGHT_MINI_SENSOR_USD
TACEX_GELSIGHT_CALIB_DIR = GELSIGHT_MINI_CALIB_DIR

LEGACY_VISUALS_ROOT_PATH = "/World/Phase2Visuals"

_DEFAULT_SENSOR_PATHS = contract_sensor_prim_paths()
LEFT_SENSOR_CASE_PRIM_PATH = _DEFAULT_SENSOR_PATHS["left"]["sensor"]
RIGHT_SENSOR_CASE_PRIM_PATH = _DEFAULT_SENSOR_PATHS["right"]["sensor"]
SENSOR_CAMERA_CLIPPING_RANGE_M = RUNTIME_CAMERA_CLIPPING_RANGE_M
VIEWPORT_GUIDE_SETTING_PATH = "/persistent/app/hydra/displayPurpose/guide"
GELSIGHT_VIEWPORT_MATERIAL_PATH = "/World/Looks/GSminiGelViewportBlue"
GELSIGHT_VIEWPORT_SHADER_PATH = f"{GELSIGHT_VIEWPORT_MATERIAL_PATH}/PreviewSurface"
GELSIGHT_VIEWPORT_COLOR = (0.25, 0.60, 1.0)
GELSIGHT_VIEWPORT_OPACITY = 1.0
RTX_TRANSLUCENCY_SETTING_PATH = "/rtx/translucency/enabled"
RTX_INVISIBLE_TO_SECONDARY_RAYS_ATTR = "primvars:invisibleToSecondaryRays"
RTX_DO_NOT_CAST_SHADOWS_ATTR = "primvars:doNotCastShadows"
MOUNT_IDENTITY_QUAT_WXYZ = (1.0, 0.0, 0.0, 0.0)


def sensor_prim_paths() -> dict[str, dict[str, str]]:
    """Return runtime TacEx sensor prim paths for both fingers."""

    return contract_sensor_prim_paths()


def _clear_cached_prim_view(prim_path: str) -> None:
    """Compatibility hook for older XFormPrim-cache based helpers."""

    return None


def _validate_canonical_sensor_links(sides: tuple[str, ...]) -> None:
    paths_by_side = sensor_prim_paths()
    for side in sides:
        if side not in paths_by_side:
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        for key in ("canonical_case_link", "canonical_gelpad_link"):
            path = paths_by_side[side][key]
            if not prim_utils.is_prim_path_valid(path):
                raise RuntimeError(f"Cannot mount TacEx Sensor.usd; canonical URDF prim is missing: {path}")


def _ensure_runtime_sensor_scopes(sides: tuple[str, ...]) -> None:
    for side in sides:
        scope = sensor_prim_paths()[side]["runtime_sensor_scope"]
        current = ""
        for component in scope.strip("/").split("/"):
            current = f"{current}/{component}"
            if not prim_utils.is_prim_path_valid(current):
                prim_utils.create_prim(current, "Xform")


def validate_sensor_mounts(sides: tuple[str, ...] = ("left",)) -> dict[str, dict[str, object]]:
    """Validate canonical body mapping and detached TacEx runtime assets."""

    camera_checks = validate_sensor_camera_prims(sides)
    results: dict[str, dict[str, object]] = {}
    for side in sides:
        if side not in {"left", "right"}:
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        paths = sensor_prim_paths()[side]
        sensor_in_runtime_scope = paths["sensor"].startswith(f'{paths["runtime_sensor_scope"]}/')
        canonical_links_exist = all(
            prim_utils.is_prim_path_valid(paths[key])
            for key in ("canonical_case_link", "canonical_gelpad_link")
        )
        gelpad_exists = prim_utils.is_prim_path_valid(paths["gelpad"])
        results[side] = {
            "passed": bool(
                sensor_in_runtime_scope
                and canonical_links_exist
                and camera_checks[side]["camera_exists"]
                and gelpad_exists
            ),
            "mount_mode": "detached_render_sensor_synced_from_canonical_case_body",
            "canonical_case_link": paths["canonical_case_link"],
            "canonical_gelpad_link": paths["canonical_gelpad_link"],
            "runtime_sensor_scope": paths["runtime_sensor_scope"],
            "sensor": paths["sensor"],
            "case": paths["case"],
            "gelpad": paths["gelpad"],
            "camera": paths["camera"],
            "sensor_in_runtime_scope": sensor_in_runtime_scope,
            "canonical_links_exist": canonical_links_exist,
            "asset_to_case_registration": {
                "translation_m": list(SENSOR_ASSET_TO_CASE_LINK_TRANSLATION_M),
                "quat_wxyz": list(SENSOR_ASSET_TO_CASE_LINK_QUAT_WXYZ),
            },
            "camera_exists": camera_checks[side]["camera_exists"],
            "gelpad_exists": gelpad_exists,
        }
    return results


def _spawn_referenced_root(
    prim_path: str,
    usd_path: Path,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation: tuple[float, float, float, float] = MOUNT_IDENTITY_QUAT_WXYZ,
) -> None:
    if not usd_path.is_file():
        raise FileNotFoundError(f"TacEx GelSight Mini USD asset not found: {usd_path}")
    prim_utils.create_prim(
        prim_path,
        "Xform",
        usd_path=usd_path.as_posix(),
        translation=translation,
        orientation=orientation,
    )
    _clear_cached_prim_view(prim_path)


def _set_sensor_camera_clipping_range(camera_prim_path: str) -> dict[str, object]:
    """Apply the measured GSmini clipping range to the referenced USD camera.

    TacEx's ``GelSightSensor`` builds a ``TiledCameraCfg`` with ``spawn=None``
    because the camera already exists inside ``Sensor.usd``.  In that path the
    config clipping range is not authored onto the existing USD camera, so set
    the camera attribute here as part of shell mounting.
    """

    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(camera_prim_path)
    if not prim.IsValid():
        return {"camera": camera_prim_path, "applied": False, "reason": "missing camera prim"}
    camera = UsdGeom.Camera(prim)
    clipping_range = Gf.Vec2f(*SENSOR_CAMERA_CLIPPING_RANGE_M)
    camera.GetClippingRangeAttr().Set(clipping_range)
    return {
        "camera": camera_prim_path,
        "applied": True,
        "clipping_range_m": list(SENSOR_CAMERA_CLIPPING_RANGE_M),
    }


def _delete_prim_if_present(prim_path: str) -> None:
    if prim_utils.is_prim_path_valid(prim_path):
        prim_utils.delete_prim(prim_path)
    _clear_cached_prim_view(prim_path)


def clear_legacy_visual_prims() -> None:
    """Remove old debug-only visual prims before tactile rendering.

    The canonical URDF already contains the visible connector and GSmini
    assembly. Historical visual helper prims were useful for mount debugging, but
    tactile camera validation must not add duplicate appearance geometry.
    """

    _delete_prim_if_present(LEGACY_VISUALS_ROOT_PATH)


def _hide_sensor_shell_geometry(root_prim_path: str) -> int:
    """Hide mesh descendants while keeping cameras addressable for TacEx."""

    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(root_prim_path)
    hidden_count = 0
    if not root_prim.IsValid():
        return hidden_count
    for prim in Usd.PrimRange(root_prim):
        if prim.GetTypeName() == "Mesh":
            UsdGeom.Imageable(prim).MakeInvisible()
            hidden_count += 1
    return hidden_count


def set_viewport_guide_purpose_enabled(enabled: bool) -> bool:
    """Set global Hydra guide rendering and return its previous value.

    Isaac Sim 4.5 applies this setting to every Hydra render product, including
    TacEx's internal TiledCamera.  Tactile runs must therefore keep it disabled;
    it cannot be used as a main-viewport-only visibility channel.
    """

    import carb

    settings = carb.settings.get_settings()
    previous = settings.get_as_bool(VIEWPORT_GUIDE_SETTING_PATH)
    settings.set_bool(VIEWPORT_GUIDE_SETTING_PATH, enabled)
    return bool(previous)


def _hide_collision_guide_geometry(stage, robot_root_paths: set[str]) -> int:
    """Keep imported collider guides out of the interactive viewport.

    Isaac's URDF converter marks collision meshes as USD ``guide`` geometry.
    Drawing guide purpose globally renders the arm's colliders over its visual
    meshes, producing the striped/z-fighting appearance.  Author ordinary USD
    image visibility before ``sim.reset()`` so Hydra cannot draw that hierarchy
    even if guide purpose was left enabled in a persistent Kit setting.
    USD image visibility does not disable PhysX collision APIs.
    """

    from pxr import Usd, UsdGeom

    hidden_count = 0
    for robot_root_path in robot_root_paths:
        robot_root = stage.GetPrimAtPath(robot_root_path)
        if not robot_root.IsValid():
            continue
        for prim in Usd.PrimRange(robot_root):
            prim_path = str(prim.GetPath())
            if "/collisions" not in prim_path:
                continue
            imageable = UsdGeom.Imageable(prim)
            if not imageable:
                continue
            imageable.MakeInvisible()
            hidden_count += 1
    return hidden_count


def _ensure_gelpad_viewport_material(stage) -> Any:
    """Create a clearly visible blue material for the canonical gel surface."""

    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, GELSIGHT_VIEWPORT_MATERIAL_PATH)
    shader = UsdShade.Shader.Define(stage, GELSIGHT_VIEWPORT_SHADER_PATH)
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*GELSIGHT_VIEWPORT_COLOR)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.3)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(GELSIGHT_VIEWPORT_OPACITY)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def configure_canonical_gelpad_viewport_geometry(sides: tuple[str, ...]) -> dict[str, dict[str, object]]:
    """Prepare canonical render geometry for stable TacEx camera capture.

    TacEx documents two ways to keep sensor geometry out of the internal
    camera: translucency or making the obstructing meshes invisible to secondary
    rays.  TacEx's own GSmini plate authors
    ``primvars:invisibleToSecondaryRays=True`` in addition to a glass material.
    Mirror the ray-isolation contract, but keep the blue canonical gel opaque to
    the main viewport so it cannot look missing or black.  The embedded
    ``DistanceToImagePlane`` camera still observes the cube because RTX excludes
    the mesh from secondary rays.
    Imported collider guides remain hidden to prevent the striped robot
    appearance.  All USD visibility and material changes are authored before
    ``sim.reset()`` to avoid invalidating PhysX articulation views.
    """

    import carb
    import omni.usd
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    stage = omni.usd.get_context().get_stage()
    # This persistent setting may have been left on by a previous interactive
    # session.  It is global, not viewport-local, and would expose both robot
    # colliders and the gel mesh to TacEx's TiledCamera.
    previous_guide_enabled = set_viewport_guide_purpose_enabled(False)
    carb.settings.get_settings().set_bool(RTX_TRANSLUCENCY_SETTING_PATH, True)
    viewport_material = _ensure_gelpad_viewport_material(stage)
    paths = sensor_prim_paths()
    robot_roots = {
        paths[side]["canonical_gelpad_link"].rsplit("/", 1)[0]
        for side in sides
    }
    hidden_collision_guides = _hide_collision_guide_geometry(stage, robot_roots)
    configured: dict[str, dict[str, object]] = {}
    for side in sides:
        root_path = paths[side]["canonical_gelpad_link"]
        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            raise RuntimeError(f"Canonical gelpad prim is missing: {root_path}")
        visuals = stage.GetPrimAtPath(f"{root_path}/visuals")
        search_root = visuals if visuals.IsValid() else root
        # The URDF converter instances visual subtrees.  Instance proxies are
        # read-only, so expand these two tiny pad visuals before PhysX starts;
        # this lets us author the exact per-mesh RTX primvars used by TacEx.
        was_instance = search_root.IsInstance()
        if was_instance:
            search_root.SetInstanceable(False)
        meshes = [prim for prim in Usd.PrimRange(search_root) if prim.GetTypeName() == "Mesh"]
        if not meshes:
            raise RuntimeError(f"Canonical gelpad visual contains no editable mesh: {search_root.GetPath()}")
        targets = [search_root, *meshes]
        for prim in targets:
            imageable = UsdGeom.Imageable(prim)
            imageable.MakeVisible()
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                viewport_material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            )
        for mesh in meshes:
            for attribute_name in (
                RTX_INVISIBLE_TO_SECONDARY_RAYS_ATTR,
                RTX_DO_NOT_CAST_SHADOWS_ATTR,
            ):
                mesh.CreateAttribute(
                    attribute_name,
                    Sdf.ValueTypeNames.Bool,
                    custom=True,
                ).Set(True)
        visual_paths = [str(prim.GetPath()) for prim in targets]
        secondary_ray_hidden_paths = [str(mesh.GetPath()) for mesh in meshes]
        configured[side] = {
            "root": root_path,
            "visual_paths": visual_paths,
            "visibility": "visible_opaque_primary_rays_only",
            "material": GELSIGHT_VIEWPORT_MATERIAL_PATH,
            "opacity": GELSIGHT_VIEWPORT_OPACITY,
            "reason": "opaque blue gel excluded from TacEx secondary camera rays",
            "deinstanced_visual_root": was_instance,
            "secondary_ray_hidden_paths": secondary_ray_hidden_paths,
            "rtx_translucency_enabled": True,
            "global_guide_purpose_enabled": False,
            "previous_global_guide_purpose_enabled": previous_guide_enabled,
            "collision_guide_prims_hidden": hidden_collision_guides,
        }
    return configured


def _disable_sensor_shell_physics(root_prim_path: str) -> int:
    """Disable collision/rigid-body effects on hidden TacEx runtime shells.

    The canonical URDF owns the physical Robotiq pad / GSmini contact
    approximation. The runtime references TacEx ``Sensor.usd`` only to expose the
    internal render camera and optical gel prim required by ``GelSightMiniCfg``.
    If that asset brings collision or rigid-body metadata, leaving it enabled
    can create an invisible duplicate collider and block the gripper.  We
    therefore force physics off for the detached render hierarchy.
    """

    import omni.usd
    from pxr import Usd, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(root_prim_path)
    disabled_count = 0
    if not root_prim.IsValid():
        return disabled_count

    physics_attr_names = (
        "physics:collisionEnabled",
        "physics:rigidBodyEnabled",
        "physxCollision:collisionEnabled",
    )
    for prim in Usd.PrimRange(root_prim):
        for attr_name in physics_attr_names:
            attr = prim.GetAttribute(attr_name)
            if attr:
                attr.Set(False)
                disabled_count += 1
        if prim.GetTypeName() == "Mesh":
            try:
                collision_api = UsdPhysics.CollisionAPI.Apply(prim)
                collision_api.CreateCollisionEnabledAttr(False)
                disabled_count += 1
            except Exception:
                # Older USD/Isaac builds can expose schema differences.  The
                # direct attribute write above is the important path; keep
                # mounting robust if applying the API is unavailable.
                pass
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            try:
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
                disabled_count += 1
            except Exception:
                pass
    return disabled_count


def mount_sensor_shells(
    sides: tuple[str, ...] = ("left",),
    *,
    hide_render_geometry: bool = True,
    configure_canonical_gelpad_as_guide: bool = True,
    disable_physics_collisions: bool = True,
) -> dict[str, dict[str, object]]:
    """Spawn TacEx-compatible runtime sensor prims without editing the robot URDF.

    The canonical URDF owns visible and contact geometry.  Each render-only
    TacEx ``Sensor.usd`` lives in a detached scope and is synchronized from the
    corresponding canonical case body by
    :func:`sync_sensor_shells_to_robot`.  A measured CAD-frame
    registration aligns TacEx's source asset with the baked GSmini STL frame.
    """

    _validate_canonical_sensor_links(sides)
    _ensure_runtime_sensor_scopes(sides)
    canonical_gelpad_viewport = (
        configure_canonical_gelpad_viewport_geometry(sides)
        if configure_canonical_gelpad_as_guide
        else {side: {} for side in sides}
    )
    spawned: dict[str, dict[str, object]] = {}
    paths_by_side = sensor_prim_paths()
    for side in sides:
        if side not in {"left", "right"}:
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        paths = paths_by_side[side]
        _delete_prim_if_present(paths["sensor"])
        _spawn_referenced_root(
            paths["sensor"],
            TACEX_GELSIGHT_SENSOR_USD,
            translation=SENSOR_ASSET_TO_CASE_LINK_TRANSLATION_M,
            orientation=SENSOR_ASSET_TO_CASE_LINK_QUAT_WXYZ,
        )
        camera_clipping = _set_sensor_camera_clipping_range(paths["camera"])
        hidden_meshes = 0
        if hide_render_geometry:
            hidden_meshes += _hide_sensor_shell_geometry(paths["sensor"])
        disabled_physics_attrs = 0
        if disable_physics_collisions:
            disabled_physics_attrs += _disable_sensor_shell_physics(paths["sensor"])
        mount_check = validate_sensor_mounts((side,))[side]
        spawned[side] = {
            **paths,
            "hidden_render_meshes": hidden_meshes,
            "canonical_gelpad_viewport": canonical_gelpad_viewport[side],
            "disabled_physics_attrs": disabled_physics_attrs,
            "camera_clipping": camera_clipping,
            "mount_check": mount_check,
        }
    return spawned


def validate_sensor_camera_prims(sides: tuple[str, ...] = ("left",)) -> dict[str, dict[str, object]]:
    """Check the Camera prim embedded in each referenced TacEx Sensor.usd."""

    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    results: dict[str, dict[str, object]] = {}
    for side in sides:
        paths = sensor_prim_paths()[side]
        camera_exists = prim_utils.is_prim_path_valid(paths["camera"])
        clipping_range = None
        if camera_exists:
            camera_prim = stage.GetPrimAtPath(paths["camera"])
            camera = UsdGeom.Camera(camera_prim)
            clipping_attr = camera.GetClippingRangeAttr()
            value = clipping_attr.Get() if clipping_attr else None
            if value is not None:
                clipping_range = [float(value[0]), float(value[1])]
        results[side] = {
            "case": paths["case"],
            "camera": paths["camera"],
            "camera_exists": camera_exists,
            "expected_appendix": SENSOR_CAMERA_PRIM_PATH_APPENDIX,
            "clipping_range_m": clipping_range,
            "expected_clipping_range_m": list(SENSOR_CAMERA_CLIPPING_RANGE_M),
        }
    return results


def _set_world_pose(
    prim_path: str,
    position: torch.Tensor,
    orientation: torch.Tensor,
    *,
    reset_xform_stack: bool = False,
) -> None:
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim path is not valid: {prim_path}")
    pos = [float(v) for v in position.detach().cpu().tolist()]
    quat = [float(v) for v in orientation.detach().cpu().tolist()]
    transform = Gf.Matrix4d().SetRotate(Gf.Quatd(quat[0], Gf.Vec3d(quat[1], quat[2], quat[3])))
    transform.SetTranslate(Gf.Vec3d(*pos))
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(transform)
    xform.SetResetXformStack(reset_xform_stack)


def _set_local_pose_preserving_xform_ops(
    prim_path: str,
    position: torch.Tensor,
    orientation: torch.Tensor,
) -> None:
    """Set a local pose without replacing an existing camera's xform schema.

    ``TiledCamera`` caches the authored ``xformOp:translate`` and
    ``xformOp:orient`` attributes when its low-level view is initialized.  The
    generic :func:`_set_world_pose` helper clears that order and replaces it
    with ``xformOp:transform``; later camera-view updates then target attributes
    that no longer exist, so RTX keeps rendering from a stale pose.  Preserve
    those authored ops for cameras that move at runtime.
    """

    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim path is not valid: {prim_path}")
    pos = [float(value) for value in position.detach().cpu().tolist()]
    quat = [float(value) for value in orientation.detach().cpu().tolist()]
    xform = UsdGeom.Xformable(prim)
    translate_op = None
    orient_op = None
    for op in xform.GetOrderedXformOps():
        if op.IsInverseOp():
            continue
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate and translate_op is None:
            translate_op = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeOrient and orient_op is None:
            orient_op = op
    if translate_op is None:
        translate_op = xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
    if orient_op is None:
        orient_op = xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
    translate_op.Set(Gf.Vec3d(*pos))
    orient_op.Set(Gf.Quatd(quat[0], Gf.Vec3d(quat[1], quat[2], quat[3])))


def sync_sensor_shells_to_robot(
    robot: Articulation,
    sides: tuple[str, ...] = ("left", "right"),
    *,
    sensor_instances: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Synchronize render cameras from live PhysX articulation body poses.

    Isaac's render camera view does not automatically follow a USD child that
    is added below an articulation link after the simulation starts.  Read the
    canonical case body pose from the articulation and author the detached
    TacEx camera pose before each capture.
    """

    results: dict[str, dict[str, Any]] = {}
    paths = sensor_prim_paths()
    for side in sides:
        if side not in paths:
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        body_name = f"{side}_gelsight_mini_case"
        body_ids = robot.find_bodies([body_name], preserve_order=True)[0]
        if len(body_ids) != 1:
            raise RuntimeError(f"Expected one articulation body named {body_name!r}, got {body_ids}")
        case_pose = robot.data.body_pose_w[:, int(body_ids[0])]
        device, dtype = case_pose.device, case_pose.dtype
        sensor_position, sensor_orientation = math_utils.combine_frame_transforms(
            case_pose[:, 0:3],
            case_pose[:, 3:7],
            torch.tensor([SENSOR_ASSET_TO_CASE_LINK_TRANSLATION_M], device=device, dtype=dtype),
            torch.tensor([SENSOR_ASSET_TO_CASE_LINK_QUAT_WXYZ], device=device, dtype=dtype),
        )
        _, camera_orientation_opengl = math_utils.combine_frame_transforms(
            sensor_position,
            sensor_orientation,
            torch.zeros((1, 3), device=device, dtype=dtype),
            torch.tensor([SENSOR_CAMERA_LOCAL_QUAT_WXYZ], device=device, dtype=dtype),
        )
        # TiledCamera follows a post-start root translation, but it does not
        # reliably compose an orientation authored on that ancestor.  Keep the
        # detached hidden root axis-aligned and put the fully composed OpenGL
        # orientation directly on /Camera.
        _set_world_pose(
            paths[side]["sensor"],
            sensor_position[0],
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype),
            reset_xform_stack=False,
        )
        _set_local_pose_preserving_xform_ops(
            paths[side]["camera"],
            torch.zeros(3, device=device, dtype=dtype),
            camera_orientation_opengl[0],
        )
        camera_view_synced = False
        sensor = (sensor_instances or {}).get(side)
        camera_view = getattr(getattr(sensor, "camera", None), "_view", None)
        if camera_view is not None:
            # This is the low-level XFormPrimView, so its quaternion is the
            # authored OpenGL/USD camera convention (unlike Camera.set_world_poses,
            # whose default input convention is ROS).
            camera_view.set_world_poses(sensor_position, camera_orientation_opengl)
            # RTX/TiledCamera consumes Fabric transforms while simulation is
            # running; mirror the authored USD pose into Fabric as well.
            camera_view.set_world_poses(sensor_position, camera_orientation_opengl, usd=False)
            camera_view_synced = True
        results[side] = {
            "case_position_world_m": [float(value) for value in case_pose[0, 0:3].detach().cpu().tolist()],
            "sensor_position_world_m": [float(value) for value in sensor_position[0].detach().cpu().tolist()],
            "sensor_orientation_world_wxyz": [
                float(value) for value in sensor_orientation[0].detach().cpu().tolist()
            ],
            "camera_orientation_world_opengl_wxyz": [
                float(value) for value in camera_orientation_opengl[0].detach().cpu().tolist()
            ],
            "camera_view_synced": camera_view_synced,
        }
    return results

