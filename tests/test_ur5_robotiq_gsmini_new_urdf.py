from __future__ import annotations

import struct
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = REPO_ROOT / "environment" / "ur5_robotiq_GSmini"
URDF_DIR = ASSET_ROOT / "urdf"
MESH_DIR = ASSET_ROOT / "meshes"
NEW_URDF = URDF_DIR / "ur5_robotiq_GSmini_new.urdf"
SOURCE_ADAPTOR_PATH = MESH_DIR / "gelsight_robotiq_connector" / "visual" / "adaptor-3.STL"
ADAPTOR_PATH = MESH_DIR / "gelsight_robotiq_connector" / "visual" / "adaptor_3.STL"
LEGACY_RAW_ADAPTOR_PATH = MESH_DIR / "gelsight_robotiq_connector" / "visual" / "gelsight_adaptor.STL"
LEGACY_METRE_CONNECTOR_PATH = MESH_DIR / "gelsight_robotiq_connector" / "visual" / "Connector.stl"
EXPECTED_MESH_ORIGIN = (-0.017104848723, -0.015687201598, -0.053443920870)
EXPECTED_MESH_SCALE = (0.001, 0.001, 0.001)


def _vector(value: str) -> tuple[float, float, float]:
    result = tuple(float(item) for item in value.split())
    if len(result) != 3:
        raise AssertionError(f"Expected a 3-vector, got {value!r}")
    return result


def _binary_stl_vertices(path: Path) -> list[tuple[float, float, float]]:
    data = path.read_bytes()
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    expected_size = 84 + 50 * triangle_count
    if len(data) != expected_size:
        raise AssertionError(f"Expected binary STL {path}, got {len(data)} bytes instead of {expected_size}")
    vertices: list[tuple[float, float, float]] = []
    for triangle_index in range(triangle_count):
        values = struct.unpack_from("<9f", data, 84 + triangle_index * 50 + 12)
        vertices.extend(tuple(values[offset : offset + 3]) for offset in (0, 3, 6))
    return vertices


def _bounds(path: Path) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    vertices = _binary_stl_vertices(path)
    minimum = tuple(min(vertex[axis] for vertex in vertices) for axis in range(3))
    maximum = tuple(max(vertex[axis] for vertex in vertices) for axis in range(3))
    return minimum, maximum


class NewGsminiUrdfTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.new_root = ET.parse(NEW_URDF).getroot()

    def test_all_mesh_references_exist(self) -> None:
        for mesh in self.new_root.findall(".//mesh"):
            mesh_path = (URDF_DIR / mesh.attrib["filename"]).resolve()
            self.assertTrue(mesh_path.is_file(), mesh_path)

    def test_connector_visual_and_collision_use_scaled_new_adaptor(self) -> None:
        expected_relative_path = "../meshes/gelsight_robotiq_connector/visual/adaptor_3.STL"
        for side in ("left", "right"):
            link = self.new_root.find(f"link[@name='{side}_gelsight_connector']")
            self.assertIsNotNone(link)
            for geometry_kind in ("visual", "collision"):
                geometry = link.find(geometry_kind)
                self.assertIsNotNone(geometry, f"{side} connector lacks {geometry_kind}")
                origin = geometry.find("origin")
                mesh = geometry.find("geometry/mesh")
                self.assertEqual(mesh.attrib["filename"], expected_relative_path)
                self.assertEqual(_vector(mesh.attrib["scale"]), EXPECTED_MESH_SCALE)
                for actual, expected in zip(_vector(origin.attrib["xyz"]), EXPECTED_MESH_ORIGIN):
                    self.assertAlmostEqual(actual, expected, places=11)

    def test_inner_finger_mimic_chain_is_retained_without_added_pad_visuals(self) -> None:
        for side in ("left", "right"):
            inner_joint = self.new_root.find(f"joint[@name='{side}_inner_finger_joint']")
            inner_link = self.new_root.find(f"link[@name='{side}_inner_finger']")
            self.assertIsNotNone(inner_joint)
            self.assertIsNotNone(inner_link)
            self.assertIsNone(inner_link.find(f"visual[@name='{side}_inner_finger_pad_visual']"))
            self.assertIsNotNone(inner_link.find(f"collision[@name='{side}_inner_finger_pad_collision']"))

    def test_connector_mount_remains_on_inner_finger(self) -> None:
        for side in ("left", "right"):
            new_connector_joint = self.new_root.find(
                f"joint[@name='{side}_gelsight_connector_fixed_attachment_joint']"
            )
            self.assertIsNotNone(new_connector_joint)
            self.assertEqual(new_connector_joint.find("parent").attrib["link"], f"{side}_inner_finger")
            self.assertEqual(new_connector_joint.find("child").attrib["link"], f"{side}_gelsight_connector")
            self.assertIsNotNone(new_connector_joint.find("origin"))

    def test_mesh_origin_is_derived_from_existing_cad_frame(self) -> None:
        raw_vertices = _binary_stl_vertices(LEGACY_RAW_ADAPTOR_PATH)
        metre_vertices = _binary_stl_vertices(LEGACY_METRE_CONNECTOR_PATH)
        self.assertEqual(len(raw_vertices), len(metre_vertices))
        translations = tuple(
            sum(metre[axis] - 0.001 * raw[axis] for raw, metre in zip(raw_vertices, metre_vertices))
            / len(raw_vertices)
            for axis in range(3)
        )
        for actual, expected in zip(translations, EXPECTED_MESH_ORIGIN):
            self.assertAlmostEqual(actual, expected, places=9)
        max_error = max(
            abs(metre[axis] - (0.001 * raw[axis] + translations[axis]))
            for raw, metre in zip(raw_vertices, metre_vertices)
            for axis in range(3)
        )
        self.assertLess(max_error, 5e-9)

    def test_isaac_compatible_mesh_copy_is_byte_identical_to_requested_source(self) -> None:
        self.assertEqual(ADAPTOR_PATH.read_bytes(), SOURCE_ADAPTOR_PATH.read_bytes())

    def test_mesh_basenames_are_valid_usd_identifiers(self) -> None:
        for mesh in self.new_root.findall(".//mesh"):
            stem = Path(mesh.attrib["filename"]).stem
            self.assertTrue(stem.isascii() and stem.isidentifier(), f"Isaac-incompatible mesh basename: {stem}")

    def test_new_adaptor_keeps_gelpad_as_forward_contact_surface(self) -> None:
        adaptor_min_mm, adaptor_max_mm = _bounds(ADAPTOR_PATH)
        adaptor_min_m = tuple(EXPECTED_MESH_ORIGIN[i] + adaptor_min_mm[i] * 0.001 for i in range(3))
        adaptor_max_m = tuple(EXPECTED_MESH_ORIGIN[i] + adaptor_max_mm[i] * 0.001 for i in range(3))
        self.assertAlmostEqual(adaptor_max_m[0] - adaptor_min_m[0], 0.031, places=8)
        self.assertAlmostEqual(adaptor_max_m[1] - adaptor_min_m[1], 0.030, places=8)
        self.assertAlmostEqual(adaptor_max_m[2] - adaptor_min_m[2], 0.075, places=8)

        gelpad_path = MESH_DIR / "gelsight_mini" / "visual" / "GSmini_soft_link.stl"
        gelpad_min, _ = _bounds(gelpad_path)
        # In the shared connector/case frame, negative Y points toward the
        # grasped object.  The soft gel remains in front of the rigid adaptor.
        self.assertLess(gelpad_min[1], adaptor_min_m[1])
        self.assertGreater(adaptor_min_m[1] - gelpad_min[1], 0.009)

    def test_runtime_code_references_new_source_asset(self) -> None:
        scene_text = (REPO_ROOT / "ur5_phase1_scene.py").read_text(encoding="utf-8")
        mount_text = (REPO_ROOT / "ur5_phase2_mount.py").read_text(encoding="utf-8")
        control_text = (REPO_ROOT / "ur5_phase1_control.py").read_text(encoding="utf-8")
        phase3_geometry_text = (REPO_ROOT / "ur5_phase3_geometry.py").read_text(encoding="utf-8")
        gsmini_contract_text = (REPO_ROOT / "ur5_gsmini_contract.py").read_text(encoding="utf-8")
        viewer_text = (REPO_ROOT / "environment" / "view_ur5_robotiq_GSmini_pybullet.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"ur5_robotiq_GSmini_new.urdf"', scene_text)
        self.assertIn('"ur5_robotiq_GSmini_new.usd"', scene_text)
        self.assertIn("ur5_robotiq_GSmini_new.urdf", mount_text)
        self.assertIn("ur5_robotiq_GSmini_new.urdf", viewer_text)
        self.assertIn("left_inner_finger", mount_text)
        self.assertIn("right_inner_finger", mount_text)
        self.assertIn("left_gelsight_mini_gelpad", phase3_geometry_text)
        self.assertIn("right_gelsight_mini_gelpad", phase3_geometry_text)
        self.assertIn('case_link = f"{robot_path}/{side}_gelsight_mini_case"', gsmini_contract_text)
        self.assertIn('"left_inner_finger_joint"', control_text)
        self.assertIn('"right_inner_finger_joint"', control_text)


if __name__ == "__main__":
    unittest.main()
