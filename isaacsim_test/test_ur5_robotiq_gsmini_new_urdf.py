from __future__ import annotations

import hashlib
import struct
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = REPO_ROOT / "environment" / "ur5_robotiq_GSmini"
URDF_DIR = ASSET_ROOT / "urdf"
MESH_DIR = ASSET_ROOT / "meshes"
NEW_URDF = URDF_DIR / "ur5_robotiq_GSmini_new.urdf"
ADAPTOR_RELATIVE_PATH = "../meshes/gelsight_robotiq_connector/visual/adaptor_4.STL"
ADAPTOR_PATH = MESH_DIR / "gelsight_robotiq_connector" / "visual" / "adaptor_4.STL"
LEGACY_RAW_ADAPTOR_PATH = MESH_DIR / "gelsight_robotiq_connector" / "visual" / "gelsight_adaptor.STL"
LEGACY_METRE_CONNECTOR_PATH = MESH_DIR / "gelsight_robotiq_connector" / "visual" / "Connector.stl"
EXPECTED_ADAPTOR_SHA256 = "ab10eaef867622f9107b43e7602e698c35191cf66376c3b2a6e3e70c277728e4"
EXPECTED_ADAPTOR_BOUNDS_MM = ((0.0, 2.0, 0.0), (31.0, 28.0, 75.0))
EXPECTED_ADAPTOR_SIZE_M = (0.031, 0.026, 0.075)
EXPECTED_MESH_ORIGIN = (-0.017104848723, -0.015687201598, -0.053443920870)
EXPECTED_MESH_SCALE = (0.001, 0.001, 0.001)


def _vector(value: str) -> tuple[float, float, float]:
    result = tuple(float(item) for item in value.split())
    if len(result) != 3:
        raise AssertionError(f"Expected a 3-vector, got {value!r}")
    return result


def _required(element: ET.Element | None, description: str) -> ET.Element:
    if element is None:
        raise AssertionError(f"Missing required URDF element: {description}")
    return element


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
    minimum = (
        min(vertex[0] for vertex in vertices),
        min(vertex[1] for vertex in vertices),
        min(vertex[2] for vertex in vertices),
    )
    maximum = (
        max(vertex[0] for vertex in vertices),
        max(vertex[1] for vertex in vertices),
        max(vertex[2] for vertex in vertices),
    )
    return minimum, maximum


class NewGsminiUrdfTest(unittest.TestCase):
    new_root: ET.Element

    @classmethod
    def setUpClass(cls) -> None:
        cls.new_root = ET.parse(NEW_URDF).getroot()

    def test_all_mesh_references_exist(self) -> None:
        for mesh in self.new_root.findall(".//mesh"):
            mesh_path = (URDF_DIR / mesh.attrib["filename"]).resolve()
            self.assertTrue(mesh_path.is_file(), mesh_path)

    def test_connector_visual_and_collision_use_locked_adaptor4(self) -> None:
        for side in ("left", "right"):
            link = _required(
                self.new_root.find(f"link[@name='{side}_gelsight_connector']"),
                f"{side} GSmini connector link",
            )
            for geometry_kind in ("visual", "collision"):
                geometry = _required(
                    link.find(geometry_kind),
                    f"{side} connector {geometry_kind}",
                )
                origin = _required(geometry.find("origin"), f"{side} connector {geometry_kind} origin")
                mesh = _required(geometry.find("geometry/mesh"), f"{side} connector {geometry_kind} mesh")
                self.assertEqual(mesh.attrib["filename"], ADAPTOR_RELATIVE_PATH)
                self.assertEqual(_vector(mesh.attrib["scale"]), EXPECTED_MESH_SCALE)
                for actual, expected in zip(_vector(origin.attrib["xyz"]), EXPECTED_MESH_ORIGIN):
                    self.assertAlmostEqual(actual, expected, places=11)

    def test_inner_finger_mimic_chain_is_retained_without_added_pad_visuals(self) -> None:
        for side in ("left", "right"):
            _required(
                self.new_root.find(f"joint[@name='{side}_inner_finger_joint']"),
                f"{side} inner-finger joint",
            )
            inner_link = _required(
                self.new_root.find(f"link[@name='{side}_inner_finger']"),
                f"{side} inner-finger link",
            )
            self.assertIsNone(inner_link.find(f"visual[@name='{side}_inner_finger_pad_visual']"))
            self.assertIsNotNone(inner_link.find(f"collision[@name='{side}_inner_finger_pad_collision']"))

    def test_connector_mount_remains_on_inner_finger(self) -> None:
        for side in ("left", "right"):
            new_connector_joint = _required(
                self.new_root.find(f"joint[@name='{side}_gelsight_connector_fixed_attachment_joint']"),
                f"{side} GSmini connector attachment joint",
            )
            parent = _required(new_connector_joint.find("parent"), f"{side} connector parent")
            child = _required(new_connector_joint.find("child"), f"{side} connector child")
            _required(new_connector_joint.find("origin"), f"{side} connector joint origin")
            self.assertEqual(parent.attrib["link"], f"{side}_inner_finger")
            self.assertEqual(child.attrib["link"], f"{side}_gelsight_connector")

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

    def test_adaptor4_mesh_matches_locked_binary_and_bounds(self) -> None:
        mesh_bytes = ADAPTOR_PATH.read_bytes()
        self.assertEqual(hashlib.sha256(mesh_bytes).hexdigest(), EXPECTED_ADAPTOR_SHA256)
        actual_bounds = _bounds(ADAPTOR_PATH)
        for actual_vector, expected_vector in zip(actual_bounds, EXPECTED_ADAPTOR_BOUNDS_MM):
            for actual, expected in zip(actual_vector, expected_vector):
                self.assertAlmostEqual(actual, expected, places=6)

    def test_mesh_basenames_are_valid_usd_identifiers(self) -> None:
        for mesh in self.new_root.findall(".//mesh"):
            stem = Path(mesh.attrib["filename"]).stem
            self.assertTrue(stem.isascii() and stem.isidentifier(), f"Isaac-incompatible mesh basename: {stem}")

    def test_adaptor4_keeps_gelpad_as_forward_contact_surface(self) -> None:
        adaptor_min_mm, adaptor_max_mm = _bounds(ADAPTOR_PATH)
        adaptor_min_m = tuple(EXPECTED_MESH_ORIGIN[i] + adaptor_min_mm[i] * 0.001 for i in range(3))
        adaptor_max_m = tuple(EXPECTED_MESH_ORIGIN[i] + adaptor_max_mm[i] * 0.001 for i in range(3))
        actual_size_m = tuple(adaptor_max_m[index] - adaptor_min_m[index] for index in range(3))
        for actual, expected in zip(actual_size_m, EXPECTED_ADAPTOR_SIZE_M):
            self.assertAlmostEqual(actual, expected, places=8)

        gelpad_path = MESH_DIR / "gelsight_mini" / "visual" / "GSmini_soft_link.stl"
        gelpad_min, _ = _bounds(gelpad_path)
        # In the shared connector/case frame, negative Y points toward the
        # grasped object.  The soft gel remains in front of the rigid adaptor.
        self.assertLess(gelpad_min[1], adaptor_min_m[1])
        self.assertGreater(adaptor_min_m[1] - gelpad_min[1], 0.009)

    def test_runtime_code_references_new_source_asset(self) -> None:
        scene_text = (REPO_ROOT / "protac/robot/asset.py").read_text(encoding="utf-8")
        control_text = (REPO_ROOT / "protac/robot/control.py").read_text(encoding="utf-8")
        phase3_geometry_text = (REPO_ROOT / "protac/grasp/contact_geometry.py").read_text(encoding="utf-8")
        gsmini_contract_text = (REPO_ROOT / "protac/tactile/gsmini_contract.py").read_text(encoding="utf-8")
        viewer_text = (
            REPO_ROOT
            / "isaacsim_test"
            / "viewers"
            / "view_ur5_robotiq_GSmini_pybullet.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"ur5_robotiq_GSmini_new.urdf"', scene_text)
        self.assertIn('"ur5_robotiq_GSmini_new.usd"', scene_text)
        self.assertIn("ur5_robotiq_GSmini_new.urdf", viewer_text)
        self.assertIn("left_gelsight_mini_gelpad", phase3_geometry_text)
        self.assertIn("right_gelsight_mini_gelpad", phase3_geometry_text)
        self.assertIn('case_link = f"{robot_path}/{side}_gelsight_mini_case"', gsmini_contract_text)
        self.assertIn('"left_inner_finger_joint"', control_text)
        self.assertIn('"right_inner_finger_joint"', control_text)


if __name__ == "__main__":
    unittest.main()
