from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from protac.eval.trial_validator import (
    _tactile_baseline_stability_summary,
    _tactile_optical_depth_summary,
    _tactile_response_summary,
)


class Phase3TactileReviewTest(unittest.TestCase):
    def test_accepts_stable_no_contact_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = np.full((10, 10, 3), 0.25, dtype=np.float32)
            rows = self._baseline_rows(root, "left", [baseline, baseline, baseline])

            result = _tactile_baseline_stability_summary(rows, sides=["left"])

            self.assertTrue(result["passed"])
            self.assertEqual(result["sides"]["left"]["max_mean_abs_diff"], 0.0)

    def test_rejects_irregular_no_contact_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = np.zeros((10, 10, 3), dtype=np.float32)
            irregular = baseline.copy()
            irregular[:3, :] = 0.05
            rows = self._baseline_rows(root, "left", [baseline, irregular])

            result = _tactile_baseline_stability_summary(rows, sides=["left"])

            self.assertFalse(result["passed"])
            self.assertFalse(result["sides"]["left"]["passed"])

    def test_rejects_stale_camera_height_map_at_far_plane(self) -> None:
        samples = self._depth_samples(
            left=(0.032, 0.032),
            right=(0.032, 0.032),
        )

        result = _tactile_optical_depth_summary(
            samples,
            sides=["left", "right"],
            far_plane_m=0.032,
        )

        self.assertFalse(result["passed"])
        self.assertFalse(result["sides"]["left"]["observed_surface"])

    def test_accepts_contact_depth_below_far_plane_with_spatial_shape(self) -> None:
        samples = self._depth_samples(
            left=(0.02888, 0.02950),
            right=(0.02890, 0.02944),
        )

        result = _tactile_optical_depth_summary(
            samples,
            sides=["left", "right"],
            far_plane_m=0.032,
        )

        self.assertTrue(result["passed"])

    def test_rejects_subtle_rgb_noise_as_not_visually_obvious(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = np.zeros((10, 10, 3), dtype=np.float32)
            subtle = baseline.copy()
            subtle[:2, :1] = 0.0025
            subtle_2 = baseline.copy()
            subtle_2[:2, :1] = 0.0035
            subtle_3 = baseline.copy()
            subtle_3[:2, :1] = 0.0045

            result = _tactile_response_summary(
                self._frame_rows(root, "left", baseline, [subtle, subtle_2, subtle_3])
            )

            self.assertFalse(result["passed"])
            self.assertFalse(result["sides"]["left"]["passed"])

    def test_accepts_sustained_spatially_obvious_rgb_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = np.zeros((10, 10, 3), dtype=np.float32)
            contact = baseline.copy()
            contact[:2, :] = 0.01

            result = _tactile_response_summary(
                self._frame_rows(root, "left", baseline, [contact, contact * 1.1, contact * 1.2])
            )

            self.assertTrue(result["passed"])
            self.assertTrue(result["sides"]["left"]["passed"])

    def test_accepts_stable_contact_patch_without_contact_to_contact_motion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = np.zeros((10, 10, 3), dtype=np.float32)
            contact = baseline.copy()
            contact[:2, :] = 0.01

            result = _tactile_response_summary(
                self._frame_rows(root, "left", baseline, [contact, contact, contact])
            )

            self.assertTrue(result["passed"])
            self.assertEqual(result["sides"]["left"]["consecutive_changed_frames"], 3)
            self.assertEqual(result["sides"]["left"]["max_contact_to_contact_mean_abs_diff"], 0.0)

    def test_uses_latest_precontact_frame_as_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reset_path = root / "reset.npy"
            pregrasp_path = root / "pregrasp.npy"
            contact_path = root / "contact.npy"
            np.save(reset_path, np.zeros((2, 2, 3), dtype=np.float32))
            np.save(pregrasp_path, np.full((2, 2, 3), 0.5, dtype=np.float32))
            np.save(contact_path, np.full((2, 2, 3), 0.6, dtype=np.float32))

            result = _tactile_response_summary(
                [
                    {"side": "left", "action_stage": "reset", "tactile_rgb_path": reset_path.as_posix()},
                    {
                        "side": "left",
                        "action_stage": "pre_grasp",
                        "tactile_rgb_path": pregrasp_path.as_posix(),
                        "contact_detected": "False",
                    },
                    {
                        "side": "left",
                        "action_stage": "contact_close",
                        "tactile_rgb_path": contact_path.as_posix(),
                        "contact_detected": "True",
                    },
                ]
            )

            self.assertAlmostEqual(result["sides"]["left"]["mean_abs_diff"], 0.1, places=6)

    def test_does_not_count_precontact_contact_close_images_as_tactile_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_path = root / "baseline.npy"
            near_object_path = root / "near_object.npy"
            physical_contact_path = root / "physical_contact.npy"
            np.save(baseline_path, np.zeros((4, 4, 3), dtype=np.float32))
            np.save(near_object_path, np.full((4, 4, 3), 0.5, dtype=np.float32))
            np.save(physical_contact_path, np.full((4, 4, 3), 0.5, dtype=np.float32))

            rows = [
                {
                    "side": "left",
                    "action_stage": "reset",
                    "tactile_rgb_path": baseline_path.as_posix(),
                    "contact_detected": "False",
                },
                {
                    "side": "left",
                    "action_stage": "contact_close",
                    "tactile_rgb_path": near_object_path.as_posix(),
                    "contact_detected": "False",
                },
                {
                    "side": "left",
                    "action_stage": "contact_close",
                    "tactile_rgb_path": physical_contact_path.as_posix(),
                    "contact_detected": "True",
                },
            ]

            result = _tactile_response_summary(rows)

            self.assertFalse(result["passed"])
            self.assertEqual(result["sides"]["left"]["mean_abs_diff"], 0.0)

    @staticmethod
    def _frame_rows(
        root: Path,
        side: str,
        baseline: np.ndarray,
        contact_frames: list[np.ndarray],
    ) -> list[dict[str, str]]:
        baseline_path = root / f"{side}_baseline.npy"
        np.save(baseline_path, baseline)
        rows = [
            {
                "side": side,
                "action_stage": "reset",
                "tactile_rgb_path": baseline_path.as_posix(),
                "contact_detected": "False",
            }
        ]
        for index, frame in enumerate(contact_frames):
            path = root / f"{side}_contact_{index}.npy"
            np.save(path, frame)
            rows.append(
                {
                    "side": side,
                    "action_stage": "contact_close",
                    "tactile_rgb_path": path.as_posix(),
                    "contact_detected": "True",
                }
            )
        return rows

    @staticmethod
    def _baseline_rows(
        root: Path,
        side: str,
        frames: list[np.ndarray],
    ) -> list[dict[str, str]]:
        rows = []
        for index, frame in enumerate(frames):
            path = root / f"{side}_baseline_{index}.npy"
            np.save(path, frame)
            rows.append(
                {
                    "side": side,
                    "action_stage": "reset" if index == 0 else "pre_grasp",
                    "tactile_rgb_path": path.as_posix(),
                    "contact_detected": "False",
                }
            )
        return rows

    @staticmethod
    def _depth_samples(
        *,
        left: tuple[float, float],
        right: tuple[float, float],
    ) -> list[dict[str, object]]:
        return [
            {
                "action_stage": "hold",
                "contact_state": {
                    "tactile_capture": {
                        "sides": {
                            "left": {
                                "camera_observation": {
                                    "taxim_height_map": {
                                        "available": True,
                                        "min_m": left[0],
                                        "max_m": left[1],
                                    }
                                }
                            },
                            "right": {
                                "camera_observation": {
                                    "taxim_height_map": {
                                        "available": True,
                                        "min_m": right[0],
                                        "max_m": right[1],
                                    }
                                }
                            },
                        }
                    }
                },
            }
        ]


if __name__ == "__main__":
    unittest.main()
