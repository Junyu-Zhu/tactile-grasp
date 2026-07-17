from __future__ import annotations

import ast
import importlib.util
import struct
import sys
import types
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from protac.tactile.gsmini_contract import (
    GELSIGHT_MINI_CASE_USD,
    GELSIGHT_MINI_CALIB_DIR,
    GELSIGHT_MINI_GELPAD_USD,
    GELSIGHT_MINI_SENSOR_USD,
    ISAACLAB_CAMERA_DATA_FORWARD_AXIS,
    OFFICIAL_CAMERA_CLIPPING_RANGE_M,
    OFFICIAL_GELPAD_TO_CAMERA_MIN_DISTANCE_M,
    OFFICIAL_TACTILE_RESOLUTION,
    RUNTIME_CAMERA_CLIPPING_RANGE_M,
    RUNTIME_EFFECTIVE_GELPAD_HEIGHT_M,
    SENSOR_ASSET_TO_CASE_LINK_QUAT_WXYZ,
    SENSOR_ASSET_TO_CASE_LINK_TRANSLATION_M,
    sensor_prim_paths,
)
from protac.scene.object_profiles import get_object_profile
import protac.compat.phase3_v1_schema


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = REPO_ROOT / "environment" / "ur5_robotiq_GSmini"
URDF_PATH = ASSET_ROOT / "urdf" / "ur5_robotiq_GSmini_new.urdf"
MESH_ROOT = ASSET_ROOT / "meshes" / "gelsight_mini" / "visual"


def _binary_stl_vertices(path: Path) -> np.ndarray:
    data = path.read_bytes()
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + 50 * triangle_count:
        raise AssertionError(f"Expected binary STL: {path}")
    return np.asarray(
        [
            struct.unpack_from("<fff", data, 84 + triangle_index * 50 + 12 + vertex_index * 12)
            for triangle_index in range(triangle_count)
            for vertex_index in range(3)
        ],
        dtype=np.float64,
    )


def _quat_matrix_wxyz(quat: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = quat
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class GsminiTacexContractTest(unittest.TestCase):
    def test_official_tacex_sensor_and_calibration_assets_exist(self) -> None:
        for asset in (GELSIGHT_MINI_SENSOR_USD, GELSIGHT_MINI_CASE_USD, GELSIGHT_MINI_GELPAD_USD):
            self.assertTrue(asset.is_file(), asset)
        for name in ("params.json", "gelmap.npy", "polycalib.npz", "dataPack.npz"):
            self.assertTrue((GELSIGHT_MINI_CALIB_DIR / name).is_file(), name)

    def test_runtime_sensor_maps_to_canonical_links_from_detached_scope(self) -> None:
        paths = sensor_prim_paths()
        for side in ("left", "right"):
            canonical_case = f"/World/Origin1/Robot/{side}_gelsight_mini_case"
            self.assertEqual(paths[side]["canonical_case_link"], canonical_case)
            self.assertEqual(paths[side]["runtime_sensor_scope"], "/World/Phase3TacExSensors/Origin1")
            self.assertTrue(paths[side]["sensor"].startswith(f'{paths[side]["runtime_sensor_scope"]}/'))
            self.assertFalse(paths[side]["sensor"].startswith(f"{canonical_case}/"))
            self.assertEqual(paths[side]["camera"], f'{paths[side]["sensor"]}/Camera')
            self.assertEqual(paths[side]["gelpad"], f'{paths[side]["sensor"]}/Gelpad_low_res')

    def test_tacex_asset_registration_matches_baked_local_case_mesh(self) -> None:
        source = _binary_stl_vertices(MESH_ROOT / "base_link.STL")
        expected = _binary_stl_vertices(MESH_ROOT / "GSmini_base_link.stl")
        self.assertEqual(source.shape, expected.shape)
        rotation = _quat_matrix_wxyz(SENSOR_ASSET_TO_CASE_LINK_QUAT_WXYZ)
        translation = np.asarray(SENSOR_ASSET_TO_CASE_LINK_TRANSLATION_M)
        actual = source @ rotation.T + translation
        self.assertLess(float(np.max(np.abs(actual - expected))), 1.0e-7)

    def test_urdf_soft_colliders_match_visible_gelpad_bounds(self) -> None:
        root = ET.parse(URDF_PATH).getroot()
        vertices = _binary_stl_vertices(MESH_ROOT / "GSmini_soft_link.stl")
        expected_size = vertices.max(axis=0) - vertices.min(axis=0)
        expected_center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2.0
        for side in ("left", "right"):
            link = root.find(f"link[@name='{side}_gelsight_mini_gelpad']")
            self.assertIsNotNone(link)
            collision = link.find("collision")
            size = np.fromstring(collision.find("geometry/box").attrib["size"], sep=" ")
            center = np.fromstring(collision.find("origin").attrib["xyz"], sep=" ")
            np.testing.assert_allclose(size, expected_size, atol=1.0e-7)
            np.testing.assert_allclose(center, expected_center, atol=1.0e-7)

    def test_runtime_records_optical_adjustments_and_single_sensor_asset(self) -> None:
        self.assertEqual(OFFICIAL_CAMERA_CLIPPING_RANGE_M, (0.024, 0.029))
        self.assertEqual(OFFICIAL_TACTILE_RESOLUTION, (320, 240))
        self.assertEqual(OFFICIAL_GELPAD_TO_CAMERA_MIN_DISTANCE_M, 0.024)
        self.assertEqual(RUNTIME_CAMERA_CLIPPING_RANGE_M, (0.024, 0.032))
        self.assertEqual(RUNTIME_EFFECTIVE_GELPAD_HEIGHT_M, 0.0052)
        self.assertEqual(ISAACLAB_CAMERA_DATA_FORWARD_AXIS, (1.0, 0.0, 0.0))
        mount_text = (REPO_ROOT / "protac/tactile/mount.py").read_text(encoding="utf-8")
        self.assertIn("GELSIGHT_MINI_SENSOR_USD", mount_text)
        self.assertNotIn("_phase2_sensor_local_mount_offsets", mount_text)
        motion_text = (REPO_ROOT / "protac/grasp/nominal_controller.py").read_text(encoding="utf-8")
        self.assertIn("sync_sensor_shells_to_robot", motion_text)

    def test_streamlined_cube_entrypoint_requests_only_dual_tactile_rgb(self) -> None:
        entrypoint = (REPO_ROOT / "ur5_cube_tactile_grasp.py").read_text(encoding="utf-8")
        self.assertIn('TARGET_MODULE = "protac.baseline.cube_collect"', entrypoint)
        self.assertIn('"--tactile_sides", "both"', entrypoint)
        self.assertIn('"--tactile_debug_vis"', entrypoint)
        self.assertIn('"--no_camera_depth"', entrypoint)
        self.assertIn('"--sample_every_steps", "16"', entrypoint)
        self.assertIn('"--tactile_sample_every_steps", "4"', entrypoint)
        self.assertIn('"--tactile_live_every_steps", "4"', entrypoint)
        self.assertIn('"--close_steps", "180"', entrypoint)
        self.assertIn('"--pregrasp_move_steps", "90"', entrypoint)
        self.assertIn('"--max_joint_delta_per_step", "0.006"', entrypoint)
        self.assertIn('"--post_lift_hold_steps", "180"', entrypoint)

    def test_phase3_uses_camera_derived_rgb_without_synthetic_imprint(self) -> None:
        runner = (REPO_ROOT / "protac/baseline/trial_runner.py").read_text(encoding="utf-8")
        self.assertNotIn("tactile_contact_imprint_enabled", runner)
        motion = (REPO_ROOT / "protac/grasp/nominal_controller.py").read_text(encoding="utf-8")
        self.assertNotIn("_apply_continuous_taxim_imprint", motion)
        self.assertNotIn("_maybe_apply_continuous_tactile_imprint", motion)
        data_collection = (REPO_ROOT / "protac/baseline/cube_collect.py").read_text(encoding="utf-8")
        self.assertNotIn("enable_tactile_contact_imprint", data_collection)

    def test_phase3_contact_motion_never_kinematically_writes_mimic_joints(self) -> None:
        source = (REPO_ROOT / "protac/grasp/nominal_controller.py").read_text(encoding="utf-8")
        module = ast.parse(source)
        step_once = next(
            node
            for node in ast.walk(module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_step_once"
        )
        calls = {
            node.func.id if isinstance(node.func, ast.Name) else node.func.attr
            for node in ast.walk(step_once)
            if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))
        }
        self.assertNotIn("_stabilize_gripper_mimic_state", calls)
        self.assertNotIn("_align_gripper_before_contact", calls)
        self.assertNotIn("write_joint_state_to_sim", ast.get_source_segment(source, step_once))

        latch = next(
            node
            for node in ast.walk(module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_latch_arm"
        )
        self.assertNotIn("write_joint_state_to_sim", ast.get_source_segment(source, latch))

        control = (REPO_ROOT / "protac/robot/control.py").read_text(encoding="utf-8")
        scene = (REPO_ROOT / "protac/robot/asset.py").read_text(encoding="utf-8")
        self.assertIn(
            "GRIPPER_COMMAND_JOINT_NAMES = GRIPPER_CONTROL_JOINT_NAMES.copy()",
            control,
        )
        self.assertIn("joint_names_expr=GRIPPER_COMMAND_JOINT_NAMES", scene)
        self.assertIn("convert_mimic_joints_to_normal_joints=False", scene)

    def test_phase3_uses_tacex_style_ray_isolation_for_visible_canonical_gel(self) -> None:
        mount_text = (REPO_ROOT / "protac/tactile/mount.py").read_text(encoding="utf-8")
        self.assertIn("configure_canonical_gelpad_viewport_geometry", mount_text)
        self.assertIn("_hide_collision_guide_geometry", mount_text)
        self.assertIn('"/collisions"', mount_text)
        self.assertIn("blue canonical gel opaque", mount_text)
        self.assertIn("GELSIGHT_VIEWPORT_MATERIAL_PATH", mount_text)
        self.assertIn("GELSIGHT_VIEWPORT_OPACITY = 1.0", mount_text)
        self.assertIn("MaterialBindingAPI", mount_text)
        self.assertIn("/rtx/translucency/enabled", mount_text)
        # TacEx's own GSmini plate/gel assets author this primvar on the
        # render mesh.  Opacity alone still blocks DistanceToImagePlane, so a
        # visible gel must be excluded from the sensor's secondary rays.
        self.assertIn("primvars:invisibleToSecondaryRays", mount_text)

        mount_module = ast.parse(mount_text)
        hide_collision_guides = next(
            node
            for node in ast.walk(mount_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_hide_collision_guide_geometry"
        )
        hide_source = ast.get_source_segment(mount_text, hide_collision_guides)
        self.assertIn("imageable.MakeInvisible()", hide_source)

        configure_gel = next(
            node
            for node in ast.walk(mount_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "configure_canonical_gelpad_viewport_geometry"
        )
        configure_source = ast.get_source_segment(mount_text, configure_gel)
        self.assertNotIn("imageable.MakeInvisible()", configure_source)
        self.assertIn("imageable.MakeVisible()", configure_source)
        self.assertIn("_ensure_gelpad_viewport_material", configure_source)
        self.assertIn("SetInstanceable(False)", configure_source)
        self.assertIn("CreateAttribute", configure_source)
        self.assertNotIn("CreatePurposeAttr", configure_source)
        self.assertNotIn("GuideVisibility", configure_source)

        urdf = ET.parse(URDF_PATH).getroot()
        for side in ("left", "right"):
            color = urdf.find(
                f"link[@name='{side}_gelsight_mini_gelpad']/visual/material/color"
            )
            self.assertIsNotNone(color)
            self.assertEqual(color.attrib["rgba"], "0.25 0.60 1.0 1.0")

    def test_phase3_authors_robot_viewport_visibility_before_physics_reset(self) -> None:
        data_collection = (REPO_ROOT / "protac/baseline/cube_collect.py").read_text(encoding="utf-8")
        prepare_index = data_collection.index("configure_canonical_gelpad_viewport_geometry(")
        reset_index = data_collection.index("sim.reset()")
        self.assertLess(prepare_index, reset_index)
        preparation_source = data_collection[prepare_index:reset_index]
        self.assertIn("not args_cli.disable_tactile", preparation_source)
        self.assertNotIn("not args_cli.headless", preparation_source)

        runner = (REPO_ROOT / "protac/baseline/trial_runner.py").read_text(encoding="utf-8")
        self.assertIn("configure_canonical_gelpad_as_guide=False", runner)
        self.assertIn("viewport_geometry_setup", runner)

    def test_tactile_capture_uses_one_continuous_tacex_render_product(self) -> None:
        tactile_source = (REPO_ROOT / "protac/tactile/sensor.py").read_text(encoding="utf-8")
        self.assertNotIn("resolve_phase2_tactile_render_product", tactile_source)
        self.assertNotIn("set_phase2_tactile_render_products_enabled", tactile_source)
        self.assertNotIn("set_updates_enabled", tactile_source)

        motion_source = (REPO_ROOT / "protac/grasp/nominal_controller.py").read_text(encoding="utf-8")
        motion_module = ast.parse(motion_source)
        capture = next(
            node
            for node in ast.walk(motion_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_capture_tactile_outputs"
        )
        capture_source = ast.get_source_segment(motion_source, capture)
        self.assertNotIn("set_viewport_guide_purpose_enabled", capture_source)
        self.assertNotIn("_set_tactile_render_products_enabled", capture_source)
        self.assertNotIn("self.sim.RenderMode.PARTIAL_RENDERING", capture_source)
        self.assertIn("render_ticks=0", capture_source)
        self.assertNotIn("position_tactile_debug_windows", capture_source)
        self.assertNotIn("self._tactile_live_view", capture_source)

        advance = next(
            node
            for node in ast.walk(motion_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_advance_simulation_frame"
        )
        advance_source = ast.get_source_segment(motion_source, advance)
        self.assertIn("self.sim.step(render=False)", advance_source)
        self.assertIn("self._sync_tactile_sensors_and_render()", advance_source)

        synchronized_render = next(
            node
            for node in ast.walk(motion_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_sync_tactile_sensors_and_render"
        )
        synchronized_render_source = ast.get_source_segment(motion_source, synchronized_render)
        self.assertIn("sync_sensor_shells_to_robot", synchronized_render_source)
        self.assertIn("self._render_tactile_frame()", synchronized_render_source)
        self.assertNotIn("_set_tactile_render_products_enabled", synchronized_render_source)
        self.assertLess(
            synchronized_render_source.index("sync_sensor_shells_to_robot"),
            synchronized_render_source.index("self._render_tactile_frame()"),
        )

        render_frame = next(
            node
            for node in ast.walk(motion_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_render_tactile_frame"
        )
        render_frame_source = ast.get_source_segment(motion_source, render_frame)
        self.assertIn("self.sim.has_gui()", render_frame_source)
        self.assertIn("self.sim.RenderMode.FULL_RENDERING", render_frame_source)
        self.assertIn("self.sim.render()", render_frame_source)

        runner_source = (REPO_ROOT / "protac/baseline/trial_runner.py").read_text(encoding="utf-8")
        self.assertIn("set_viewport_guide_purpose_enabled(False)", runner_source)
        self.assertNotIn("resolve_phase2_tactile_render_product", runner_source)
        self.assertNotIn("camera_render_products_frozen_during_viewport_render", runner_source)
        self.assertNotIn("tactile_render_products", runner_source)
        self.assertIn("debug_vis=self.options.tactile_debug_vis", runner_source)
        self.assertIn("enable_tactile_debug_windows", runner_source)
        self.assertNotIn("create_phase2_tactile_live_view", runner_source)
        prime_index = runner_source.index("self._capture_tactile_outputs()")
        debug_attributes_index = runner_source.index("debug_attributes = (")
        self.assertLess(prime_index, debug_attributes_index)

    def test_tacex_native_windows_are_positioned_without_reimplementing_the_renderer(self) -> None:
        tactile_source = (REPO_ROOT / "protac/tactile/sensor.py").read_text(encoding="utf-8")
        self.assertIn("position_tactile_debug_windows", tactile_source)
        self.assertIn("optical_simulator", tactile_source)
        self.assertIn("_debug_windows", tactile_source)
        self.assertIn("window.position_x", tactile_source)
        self.assertIn("window.position_y", tactile_source)
        self.assertNotIn("omni.ui.ByteImageProvider", tactile_source)

        runner_source = (REPO_ROOT / "protac/baseline/trial_runner.py").read_text(encoding="utf-8")
        motion_source = (REPO_ROOT / "protac/grasp/nominal_controller.py").read_text(encoding="utf-8")
        self.assertIn('"user_draggable": True', runner_source)
        self.assertEqual(runner_source.count("position_tactile_debug_windows("), 1)
        self.assertNotIn("position_tactile_debug_windows", motion_source)

    def test_cube_profile_closes_far_enough_for_current_gsmini_mount(self) -> None:
        # The user-calibrated connector mount leaves a 61 mm open gap at
        # 0.25 rad.  Runtime evidence shows bilateral contact with a 40 mm cube
        # at 0.45 rad.  The 0.55 rad value is a force-seeking upper bound; the
        # controller stops earlier on stable bilateral force and remains well
        # inside the URDF's 0.8 rad finger limit.
        self.assertEqual(get_object_profile("cube").close_rad, 0.55)
        self.assertEqual(get_object_profile("cube").stable_force_threshold_n, 1.5)
        self.assertEqual(get_object_profile("cube").max_close_object_lift_m, 0.0035)
        self.assertGreater(get_object_profile("cube").approach_offset_world_m[2], 0.006)

    def test_cube_success_requires_gelpad_contact_without_adaptor_assistance(self) -> None:
        geometry = (REPO_ROOT / "protac/grasp/contact_geometry.py").read_text(encoding="utf-8")
        motion = (REPO_ROOT / "protac/grasp/nominal_controller.py").read_text(encoding="utf-8")
        review = (REPO_ROOT / "protac/eval/trial_validator.py").read_text(encoding="utf-8")

        self.assertIn("ADAPTOR_CONTACT_BODY_BY_SIDE", geometry)
        self.assertIn('"adaptor_force_by_side_n"', geometry)
        self.assertIn('"adaptor_contact_sides"', geometry)
        self.assertIn('"max_adaptor_force_seen_n"', motion)
        self.assertIn('failed_reasons.append("adaptor contacted cube during grasp")', motion)
        self.assertIn('"adaptor_clearance_passed"', review)

    def test_cube_profile_has_verified_fast_cartesian_grasp_pose(self) -> None:
        profile = get_object_profile("cube")
        self.assertIsNotNone(profile.grasp_ee_world_pos_m)
        self.assertEqual(len(profile.grasp_ee_world_pos_m), 3)
        self.assertIsNotNone(profile.grasp_ee_world_quat_wxyz)
        self.assertEqual(len(profile.grasp_ee_world_quat_wxyz), 4)
        self.assertAlmostEqual(
            sum(value * value for value in profile.grasp_ee_world_quat_wxyz),
            1.0,
            places=5,
        )
        self.assertGreaterEqual(profile.fast_grasp_move_steps, 180)

    def test_fast_cube_path_uses_one_bounded_soft_center_correction(self) -> None:
        runner = (REPO_ROOT / "protac/baseline/trial_runner.py").read_text(encoding="utf-8")
        self.assertIn("single_bounded_soft_center_correction", runner)
        self.assertIn("max_steps=min(80, self.object_profile.fast_grasp_move_steps)", runner)
        self.assertIn("desired_center = self._object_grasp_center_world() + approach_offset_w", runner)
        self.assertIn("approach_clearance_m", runner)

    def test_cube_success_requires_stable_bilateral_force_and_visible_lift_hold(self) -> None:
        motion = (REPO_ROOT / "protac/grasp/nominal_controller.py").read_text(encoding="utf-8")
        runner = (REPO_ROOT / "protac/baseline/trial_runner.py").read_text(encoding="utf-8")
        self.assertIn('failed_reasons.append("stable bilateral force grasp not established")', motion)
        self.assertIn('"bilateral_force_contact_detected"', motion)
        self.assertIn("post_lift_hold_steps", runner)
        self.assertIn("lift_arm_target", runner)
        self.assertIn("_run_release(logger, arm_joint_ids, lift_arm_target", runner)

    def test_fixed_pose_acceptance_cannot_ignore_orientation_failure(self) -> None:
        fixed_pose_failure = {
            "passed": False,
            "final_position_error_m": 0.001,
            "final_orientation_error_rad": 0.2,
            "orientation_tolerance_rad": 0.03,
        }
        position_only_fallback = {
            "passed": False,
            "final_position_error_m": 0.001,
        }

        self.assertFalse(
            protac.compat.phase3_v1_schema.move_result_accepted(
                fixed_pose_failure,
                position_fallback_tolerance_m=0.008,
            )
        )
        self.assertTrue(
            protac.compat.phase3_v1_schema.move_result_accepted(
                position_only_fallback,
                position_fallback_tolerance_m=0.008,
            )
        )

        motion_source = (REPO_ROOT / "protac/grasp/nominal_controller.py").read_text(encoding="utf-8")
        motion_module = ast.parse(motion_source)
        failure_reasons = next(
            node
            for node in ast.walk(motion_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_failure_reasons"
        )
        failure_source = ast.get_source_segment(motion_source, failure_reasons)
        self.assertIn("orientation_required", failure_source)
        self.assertIn(
            "orientation_required or not contact_onset.get(\"detected\")",
            failure_source,
        )

    def test_runtime_pose_sync_maps_both_case_bodies_and_updates_usd_and_fabric(self) -> None:
        """Exercise the high-risk detached-camera sync without launching Isaac Sim."""

        fake_modules = self._fake_isaac_modules()
        module_name = "_protac.tactile.mount_pose_sync_test"
        with mock.patch.dict(sys.modules, fake_modules):
            spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / "protac/tactile/mount.py")
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            mount = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = mount
            spec.loader.exec_module(mount)

            class FakeRobot:
                def __init__(self) -> None:
                    self.queries: list[str] = []
                    self.data = types.SimpleNamespace(
                        body_pose_w=torch.tensor(
                            [
                                [
                                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                    [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
                                ]
                            ],
                            dtype=torch.float64,
                        )
                    )

                def find_bodies(self, names, preserve_order=False):
                    self.assert_preserve_order = preserve_order
                    body_name = names[0]
                    self.queries.append(body_name)
                    return (torch.tensor([0 if body_name.startswith("left_") else 1]), None)

            class FakeView:
                def __init__(self) -> None:
                    self.calls: list[tuple[torch.Tensor, torch.Tensor, dict[str, object]]] = []

                def set_world_poses(self, positions, orientations, **kwargs) -> None:
                    self.calls.append((positions.detach().clone(), orientations.detach().clone(), dict(kwargs)))

            views = {side: FakeView() for side in ("left", "right")}
            sensors = {
                side: types.SimpleNamespace(camera=types.SimpleNamespace(_view=views[side]))
                for side in views
            }
            authored: list[tuple[str, torch.Tensor, torch.Tensor, bool]] = []
            authored_local: list[tuple[str, torch.Tensor, torch.Tensor]] = []

            def record_world_pose(path, position, orientation, *, reset_xform_stack=False) -> None:
                authored.append(
                    (
                        path,
                        position.detach().clone(),
                        orientation.detach().clone(),
                        reset_xform_stack,
                    )
                )

            def record_local_pose(path, position, orientation) -> None:
                authored_local.append(
                    (
                        path,
                        position.detach().clone(),
                        orientation.detach().clone(),
                    )
                )

            robot = FakeRobot()
            with (
                mock.patch.object(mount, "_set_world_pose", side_effect=record_world_pose),
                mock.patch.object(
                    mount,
                    "_set_local_pose_preserving_xform_ops",
                    side_effect=record_local_pose,
                ),
            ):
                result = mount.sync_sensor_shells_to_robot(
                    robot,
                    ("left", "right"),
                    sensor_instances=sensors,
                )

            self.assertEqual(robot.queries, ["left_gelsight_mini_case", "right_gelsight_mini_case"])
            self.assertTrue(robot.assert_preserve_order)
            self.assertEqual(len(authored), 2)
            self.assertEqual(len(authored_local), 2)
            paths = sensor_prim_paths()
            registration_q = torch.tensor(SENSOR_ASSET_TO_CASE_LINK_QUAT_WXYZ, dtype=torch.float64)
            camera_local_q = torch.tensor((0.0, 1.0, 0.0, 0.0), dtype=torch.float64)
            expected_camera_left = self._quat_multiply(registration_q, camera_local_q)
            expected_sensor_left = torch.tensor(SENSOR_ASSET_TO_CASE_LINK_TRANSLATION_M, dtype=torch.float64)
            right_case_position = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
            right_case_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
            expected_sensor_right, expected_sensor_right_q = self._combine_frame_transforms(
                right_case_position,
                right_case_quat,
                expected_sensor_left.reshape(1, 3),
                registration_q.reshape(1, 4),
            )
            _, expected_camera_right = self._combine_frame_transforms(
                expected_sensor_right,
                expected_sensor_right_q,
                torch.zeros((1, 3), dtype=torch.float64),
                camera_local_q.reshape(1, 4),
            )

            self.assertEqual(authored[0][0], paths["left"]["sensor"])
            torch.testing.assert_close(authored[0][1], expected_sensor_left)
            torch.testing.assert_close(authored[0][2], torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64))
            self.assertEqual(authored_local[0][0], paths["left"]["camera"])
            torch.testing.assert_close(authored_local[0][1], torch.zeros(3, dtype=torch.float64))
            torch.testing.assert_close(authored_local[0][2], expected_camera_left)
            self.assertEqual(authored[1][0], paths["right"]["sensor"])
            torch.testing.assert_close(authored[1][1], expected_sensor_right[0])
            torch.testing.assert_close(authored[1][2], torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64))
            self.assertEqual(authored_local[1][0], paths["right"]["camera"])
            torch.testing.assert_close(authored_local[1][1], torch.zeros(3, dtype=torch.float64))
            torch.testing.assert_close(authored_local[1][2], expected_camera_right[0])
            self.assertTrue(all(not item[3] for item in authored))

            for side in ("left", "right"):
                self.assertTrue(result[side]["camera_view_synced"])
                self.assertIn("camera_orientation_world_opengl_wxyz", result[side])
                self.assertEqual(len(views[side].calls), 2)
                self.assertEqual(views[side].calls[0][2], {})
                self.assertEqual(views[side].calls[1][2], {"usd": False})
                torch.testing.assert_close(views[side].calls[0][0], views[side].calls[1][0])
                torch.testing.assert_close(views[side].calls[0][1], views[side].calls[1][1])

            authored.clear()
            authored_local.clear()
            with (
                mock.patch.object(mount, "_set_world_pose", side_effect=record_world_pose),
                mock.patch.object(
                    mount,
                    "_set_local_pose_preserving_xform_ops",
                    side_effect=record_local_pose,
                ),
            ):
                no_view_result = mount.sync_sensor_shells_to_robot(robot, ("left",))
            self.assertEqual(len(authored), 1)
            self.assertEqual(len(authored_local), 1)
            self.assertFalse(no_view_result["left"]["camera_view_synced"])

    @staticmethod
    def _quat_multiply(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        w1, x1, y1, z1 = first.unbind(-1)
        w2, x2, y2, z2 = second.unbind(-1)
        return torch.stack(
            (
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ),
            dim=-1,
        )

    @classmethod
    def _combine_frame_transforms(cls, position_a, quat_a, position_b, quat_b):
        vector_quat = torch.cat((torch.zeros_like(position_b[..., :1]), position_b), dim=-1)
        quat_conjugate = quat_a.clone()
        quat_conjugate[..., 1:] *= -1
        rotated = cls._quat_multiply(cls._quat_multiply(quat_a, vector_quat), quat_conjugate)[..., 1:]
        return position_a + rotated, cls._quat_multiply(quat_a, quat_b)

    @classmethod
    def _fake_isaac_modules(cls) -> dict[str, types.ModuleType]:
        isaaclab = types.ModuleType("isaaclab")
        assets = types.ModuleType("isaaclab.assets")
        assets.Articulation = type("Articulation", (), {})
        assets.RigidObject = type("RigidObject", (), {})
        utils = types.ModuleType("isaaclab.utils")
        math_utils = types.ModuleType("isaaclab.utils.math")
        math_utils.combine_frame_transforms = cls._combine_frame_transforms
        sim = types.ModuleType("isaaclab.sim")
        converters = types.ModuleType("isaaclab.sim.converters")
        converters.MeshConverter = type("MeshConverter", (), {})
        converters.MeshConverterCfg = type("MeshConverterCfg", (), {})

        isaacsim = types.ModuleType("isaacsim")
        isaacsim_core = types.ModuleType("isaacsim.core")
        isaacsim_utils = types.ModuleType("isaacsim.core.utils")
        prims = types.ModuleType("isaacsim.core.utils.prims")
        return {
            "isaaclab": isaaclab,
            "isaaclab.assets": assets,
            "isaaclab.utils": utils,
            "isaaclab.utils.math": math_utils,
            "isaaclab.sim": sim,
            "isaaclab.sim.converters": converters,
            "isaacsim": isaacsim,
            "isaacsim.core": isaacsim_core,
            "isaacsim.core.utils": isaacsim_utils,
            "isaacsim.core.utils.prims": prims,
        }


if __name__ == "__main__":
    unittest.main()
