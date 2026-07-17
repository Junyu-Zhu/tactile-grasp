from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from protac.eval.trial_diagnostics import analyze_trial, compare_trials


class Phase3DiagnosticsTest(unittest.TestCase):
    def test_analyze_trial_returns_velocity_span_force_and_outcome_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trial_dir = self._write_trial(
                Path(tmp),
                "phase3_cube_0001",
                passed=True,
                lift_passed=True,
                object_lift_m=0.003,
                left_forces=(1.0, 3.0, 5.0),
                right_forces=(2.0, 4.0, 6.0),
                left_centers=((0.0, 0.0, 0.0), (0.2, 0.3, 0.4), (0.1, 0.1, 0.2)),
                right_centers=((1.0, 1.0, 1.0), (1.1, 1.2, 1.3), (0.9, 0.8, 0.7)),
                contact_close_finger_velocities=(0.1, -0.4),
                hold_finger_velocities=(-0.25,),
            )

            metrics = analyze_trial(trial_dir)

            self.assertEqual(metrics["total_steps"], 123)
            self.assertEqual(metrics["recorded_samples"], 3)
            self.assertTrue(metrics["trial"]["success"])
            self.assertTrue(metrics["trial"]["lift"]["passed"])
            self.assertAlmostEqual(metrics["trial"]["lift"]["object_lift_m"], 0.003)
            self.assertAlmostEqual(
                metrics["gripper_peak_velocity_rad_s"]["contact_close"]["finger_joint"],
                0.4,
            )
            self.assertAlmostEqual(
                metrics["gripper_peak_velocity_rad_s"]["hold"]["finger_joint"],
                0.25,
            )
            self.assertEqual(metrics["gel_aabb_center_span_m"]["left"], {"x": 0.2, "y": 0.3, "z": 0.4})
            self.assertAlmostEqual(metrics["contact_force_n"]["left"]["mean"], 3.0)
            self.assertAlmostEqual(metrics["contact_force_n"]["left"]["std"], 1.632993161855452)
            self.assertAlmostEqual(metrics["contact_force_n"]["right"]["peak"], 6.0)

    def test_compare_trials_reports_candidate_reduction_rates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_dir = self._write_trial(
                root / "baseline",
                "phase3_cube_0001",
                passed=False,
                lift_passed=False,
                object_lift_m=0.001,
                left_forces=(2.0, 4.0),
                right_forces=(1.0, 3.0),
                left_centers=((0.0, 0.0, 0.0), (0.4, 0.2, 0.1)),
                right_centers=((0.0, 0.0, 0.0), (0.2, 0.2, 0.2)),
                contact_close_finger_velocities=(1.0, 0.5),
                hold_finger_velocities=(0.8, 0.4),
            )
            candidate_dir = self._write_trial(
                root / "candidate",
                "phase3_cube_0001",
                passed=True,
                lift_passed=True,
                object_lift_m=0.004,
                left_forces=(1.0, 2.0),
                right_forces=(0.5, 1.5),
                left_centers=((0.0, 0.0, 0.0), (0.2, 0.1, 0.05)),
                right_centers=((0.0, 0.0, 0.0), (0.1, 0.1, 0.1)),
                contact_close_finger_velocities=(0.5, 0.25),
                hold_finger_velocities=(0.4, 0.2),
            )

            report = compare_trials(analyze_trial(baseline_dir), analyze_trial(candidate_dir))

            rates = report["candidate_vs_baseline"]["improvement_rates"]
            self.assertEqual(report["mode"], "compare")
            self.assertAlmostEqual(rates["gripper_peak_velocity_rad_s.contact_close.finger_joint"], 0.5)
            self.assertAlmostEqual(rates["gel_aabb_center_span_m.left.x"], 0.5)
            self.assertAlmostEqual(rates["contact_force_n.left.mean"], 0.5)
            self.assertEqual(report["candidate_vs_baseline"]["success_delta"], 1)
            self.assertEqual(report["candidate_vs_baseline"]["lift_passed_delta"], 1)

    def test_cli_outputs_single_trial_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trial_dir = self._write_trial(
                Path(tmp),
                "phase3_cube_0001",
                passed=True,
                lift_passed=False,
                object_lift_m=0.0,
                left_forces=(0.0,),
                right_forces=(0.0,),
                left_centers=((0.0, 0.0, 0.0),),
                right_centers=((0.0, 0.0, 0.0),),
                contact_close_finger_velocities=(0.2,),
                hold_finger_velocities=(0.1,),
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "protac.eval.trial_diagnostics",
                    str(trial_dir),
                    "--indent",
                    "0",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                text=True,
                capture_output=True,
            )

            payload = json.loads(completed.stdout)
            self.assertEqual(payload["mode"], "single")
            self.assertEqual(payload["trial"]["trial_id"], "phase3_cube_0001")

    def test_batch_summary_trial_id_selects_one_of_multiple_trials(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_trial(
                root,
                "phase3_cube_0001",
                passed=True,
                lift_passed=True,
                object_lift_m=0.004,
                left_forces=(1.0,),
                right_forces=(1.0,),
                left_centers=((0.0, 0.0, 0.0),),
                right_centers=((0.0, 0.0, 0.0),),
                contact_close_finger_velocities=(0.1,),
                hold_finger_velocities=(),
            )
            self._write_trial(
                root,
                "phase3_cube_0002",
                passed=False,
                lift_passed=False,
                object_lift_m=0.001,
                left_forces=(2.0,),
                right_forces=(3.0,),
                left_centers=((0.1, 0.2, 0.3),),
                right_centers=((0.4, 0.5, 0.6),),
                contact_close_finger_velocities=(0.2,),
                hold_finger_velocities=(),
            )
            summary_path = root / "phase3_batch_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "trial_id": "phase3_cube_0001",
                                "passed": True,
                                "step_budget_used": 111,
                            },
                            {
                                "trial_id": "phase3_cube_0002",
                                "passed": False,
                                "step_budget_used": 222,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            metrics = analyze_trial(summary_path, trial_id="phase3_cube_0002")

            self.assertEqual(metrics["trial_id"], "phase3_cube_0002")
            self.assertEqual(metrics["total_steps"], 222)
            self.assertFalse(metrics["trial"]["success"])
            self.assertEqual(
                Path(metrics["robot_state_path"]),
                root / "phase3_cube_0002" / "robot_state.json",
            )

    def _write_trial(
        self,
        root: Path,
        trial_id: str,
        *,
        passed: bool,
        lift_passed: bool,
        object_lift_m: float,
        left_forces: tuple[float, ...],
        right_forces: tuple[float, ...],
        left_centers: tuple[tuple[float, float, float], ...],
        right_centers: tuple[tuple[float, float, float], ...],
        contact_close_finger_velocities: tuple[float, ...],
        hold_finger_velocities: tuple[float, ...],
    ) -> Path:
        trial_dir = root / trial_id
        trial_dir.mkdir(parents=True)
        samples = []
        for index, left_force in enumerate(left_forces):
            stage = "contact_close" if index < len(contact_close_finger_velocities) else "hold"
            if stage == "contact_close":
                finger_velocity = contact_close_finger_velocities[index]
            else:
                finger_velocity = hold_finger_velocities[index - len(contact_close_finger_velocities)]
            samples.append(
                {
                    "index": index,
                    "action_stage": stage,
                    "joint_state": {
                        "velocity_rad_s": {
                            "finger_joint": finger_velocity,
                            "left_inner_finger_joint": finger_velocity / 2.0,
                            "left_inner_knuckle_joint": -finger_velocity / 3.0,
                            "right_outer_knuckle_joint": finger_velocity / 4.0,
                            "right_inner_finger_joint": -finger_velocity / 5.0,
                            "right_inner_knuckle_joint": finger_velocity / 6.0,
                        }
                    },
                    "contact_state": {
                        "contact_detected": True,
                        "force_by_side_n": {
                            "left": left_force,
                            "right": right_forces[index],
                        },
                        "geometry": {
                            "sides": {
                                "left": {"center_world_m": list(left_centers[index])},
                                "right": {"center_world_m": list(right_centers[index])},
                            }
                        },
                    },
                }
            )
        (trial_dir / "robot_state.json").write_text(
            json.dumps({"trial_id": trial_id, "samples": samples}),
            encoding="utf-8",
        )
        (root / "phase3_batch_summary.json").write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "trial_id": trial_id,
                            "passed": passed,
                            "step_budget_used": 123,
                            "micro_lift": {
                                "object_lift_passed": lift_passed,
                                "object_lift_m": object_lift_m,
                            },
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return trial_dir


if __name__ == "__main__":
    unittest.main()
