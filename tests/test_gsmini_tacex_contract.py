from __future__ import annotations

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

from ur5_gsmini_contract import (
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
from ur5_phase3_objects import get_object_profile


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
        mount_text = (REPO_ROOT / "ur5_phase2_mount.py").read_text(encoding="utf-8")
        self.assertIn("GELSIGHT_MINI_SENSOR_USD", mount_text)
        self.assertNotIn("_phase2_sensor_local_mount_offsets", mount_text)
        motion_text = (REPO_ROOT / "ur5_phase3_motion.py").read_text(encoding="utf-8")
        self.assertIn("sync_phase2_sensor_shells_to_robot", motion_text)

    def test_streamlined_cube_entrypoint_uses_modular_phase3_runner(self) -> None:
        entrypoint = (REPO_ROOT / "ur5_cube_tactile_grasp.py").read_text(encoding="utf-8")
        self.assertIn("ur5_phase3_data_collection.py", entrypoint)
        self.assertIn('"--tactile_sides", "both"', entrypoint)
        self.assertIn('"--tactile_debug_vis"', entrypoint)
        runner = (REPO_ROOT / "ur5_phase3_trial_runner.py").read_text(encoding="utf-8")
        self.assertIn("tactile_contact_imprint_enabled: bool = False", runner)
        motion = (REPO_ROOT / "ur5_phase3_motion.py").read_text(encoding="utf-8")
        self.assertIn('else "camera_derived_taxim_no_imprint"', motion)

    def test_cube_profile_closes_far_enough_for_current_gsmini_mount(self) -> None:
        # The user-calibrated connector mount leaves a 61 mm open gap at
        # 0.25 rad.  Runtime evidence shows bilateral contact with a 40 mm cube
        # at 0.45 rad, still well inside the URDF's 0.8 rad finger limit.
        self.assertEqual(get_object_profile("cube").close_rad, 0.45)

    def test_runtime_pose_sync_maps_both_case_bodies_and_updates_usd_and_fabric(self) -> None:
        """Exercise the high-risk detached-camera sync without launching Isaac Sim."""

        fake_modules = self._fake_isaac_modules()
        module_name = "_ur5_phase2_mount_pose_sync_test"
        with mock.patch.dict(sys.modules, fake_modules):
            spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / "ur5_phase2_mount.py")
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

            def record_world_pose(path, position, orientation, *, reset_xform_stack=False) -> None:
                authored.append(
                    (
                        path,
                        position.detach().clone(),
                        orientation.detach().clone(),
                        reset_xform_stack,
                    )
                )

            robot = FakeRobot()
            with mock.patch.object(mount, "_set_world_pose", side_effect=record_world_pose):
                result = mount.sync_phase2_sensor_shells_to_robot(
                    robot,
                    ("left", "right"),
                    sensor_instances=sensors,
                )

            self.assertEqual(robot.queries, ["left_gelsight_mini_case", "right_gelsight_mini_case"])
            self.assertTrue(robot.assert_preserve_order)
            self.assertEqual(len(authored), 4)
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
            self.assertEqual(authored[1][0], paths["left"]["camera"])
            torch.testing.assert_close(authored[1][1], torch.zeros(3, dtype=torch.float64))
            torch.testing.assert_close(authored[1][2], expected_camera_left)
            self.assertEqual(authored[2][0], paths["right"]["sensor"])
            torch.testing.assert_close(authored[2][1], expected_sensor_right[0])
            torch.testing.assert_close(authored[2][2], torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64))
            self.assertEqual(authored[3][0], paths["right"]["camera"])
            torch.testing.assert_close(authored[3][1], torch.zeros(3, dtype=torch.float64))
            torch.testing.assert_close(authored[3][2], expected_camera_right[0])
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
            with mock.patch.object(mount, "_set_world_pose", side_effect=record_world_pose):
                no_view_result = mount.sync_phase2_sensor_shells_to_robot(robot, ("left",))
            self.assertEqual(len(authored), 2)
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
