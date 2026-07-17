from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import ur5_cube_tactile_grasp as cube_entrypoint
from protac.data import force_slip_export


REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_MODULES = (
    "protac.baseline.cube_collect",
    "protac.baseline.trial_runner",
    "protac.robot.control",
    "protac.robot.reset",
    "protac.robot.asset",
    "protac.tactile.gsmini_contract",
    "protac.tactile.mount",
    "protac.tactile.sensor",
    "protac.grasp.contact_geometry",
    "protac.grasp.nominal_controller",
    "protac.scene.object_profiles",
    "protac.compat.phase3_v1_schema",
    "protac.compat.phase3_v1_logger",
    "protac.eval.trial_validator",
    "protac.eval.trial_diagnostics",
    "protac.data.sparsh_bridge",
    "protac.data.force_slip_export",
)


def test_repository_root_exposes_only_the_cube_user_entrypoint() -> None:
    root_python_files = sorted(path.name for path in REPO_ROOT.glob("*.py"))
    assert root_python_files == ["ur5_cube_tactile_grasp.py"]


def test_cube_entrypoint_targets_the_semantic_baseline_module() -> None:
    assert cube_entrypoint.TARGET_MODULE == "protac.baseline.cube_collect"
    assert find_spec(cube_entrypoint.TARGET_MODULE) is not None


def test_required_runtime_and_future_units_are_import_resolvable() -> None:
    missing = [module for module in REQUIRED_MODULES if find_spec(module) is None]
    assert missing == []


def test_historical_bringup_directories_are_removed() -> None:
    assert not (REPO_ROOT / "isaacsim_test" / "legacy").exists()
    assert not (REPO_ROOT / "isaacsim_test" / "smoke").exists()


def test_force_slip_export_preserves_external_dataset_compatibility() -> None:
    assert force_slip_export.DATASET_NAME == "cube_force_slip_v2"
    assert force_slip_export.DEFAULT_OUTPUT_ROOT == Path("sim_dataset/phase5_cube_force_slip")
    assert force_slip_export.DEFAULT_REPORT_ROOT == Path("artifacts/phase5_cube_force_slip/reports")
