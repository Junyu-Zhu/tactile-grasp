"""Run the canonical cube grasp with two live TacEx GelSight Mini streams.

This is the short user-facing entrypoint. The implementation is split into
semantic ProTac robot, tactile, grasp, scene and compatibility modules.

Run from the repository root after activating the Isaac/TacEx environment::

    python ur5_cube_tactile_grasp.py

Pass ``--headless`` for automated verification.  Any additional arguments are
forwarded to ``protac.baseline.cube_collect`` and override these defaults.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


TARGET_MODULE = "protac.baseline.cube_collect"
GUI_CAMERA_EXPERIENCE = (
    Path(__file__).resolve().parent.parent / "TacEx" / "IsaacLab" / "apps" / "isaaclab.python.kit"
)
COMMON_CAMERA_KIT_ARGS = (
    "--/isaaclab/cameras_enabled=true",
    # Tiled tactile cameras move with the gripper.  A temporal upscaler leaves
    # history trails in their depth products and on articulated viewport
    # geometry; FXAA is deterministic and does not reuse prior frames.
    "--/rtx/post/aa/op=2",
    # TacEx's GSmini asset uses translucency so the external gel/attachment
    # surface is visible without occluding the internal depth camera.
    "--/rtx/translucency/enabled=true",
    # Match the synchronization guarantees from IsaacLab's camera experience
    # while retaining the full interactive viewport experience.
    "--/app/renderer/waitIdle=true",
    "--/app/hydraEngine/waitIdle=true",
    "--/app/updateOrder/checkForHydraRenderComplete=1000",
)
GUI_CAMERA_KIT_ARGS = (
    *COMMON_CAMERA_KIT_ARGS,
    "--/exts/omni.renderer.core/present/enabled=true",
)
DEFAULT_ARGS = [
    "--object_id", "cube",
    "--protocol_variant", "contact_hold_micro_lift",
    "--trials", "1",
    "--tactile_sides", "both",
    "--tactile_debug_vis",
    "--no_camera_depth",
    "--no_preview_images",
    "--sample_every_steps", "16",
    "--tactile_sample_every_steps", "4",
    "--tactile_live_every_steps", "4",
    "--pregrasp_move_steps", "90",
    "--approach_steps", "240",
    "--max_joint_delta_per_step", "0.006",
    "--close_steps", "180",
    "--arm_hold_settle_steps", "30",
    "--stable_force_steps", "6",
    "--post_lift_hold_steps", "180",
    "--release_steps", "60",
    "--enable_cameras",
]


def _contains_option(arguments: list[str], option: str) -> bool:
    return option in arguments or any(argument.startswith(f"{option}=") for argument in arguments)


def _extract_option(arguments: list[str], option: str) -> tuple[list[str], str | None]:
    """Remove one argparse option and return its value, supporting both forms."""

    remaining: list[str] = []
    value: str | None = None
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument.startswith(f"{option}="):
            value = argument.split("=", 1)[1]
        elif argument == option and index + 1 < len(arguments):
            value = arguments[index + 1]
            index += 1
        else:
            remaining.append(argument)
        index += 1
    return remaining, value


def build_exec_arguments(forwarded: list[str]) -> list[str]:
    """Build Phase 3 arguments without sacrificing the interactive viewport.

    IsaacLab normally selects ``isaaclab.python.rendering.kit`` whenever
    ``--enable_cameras`` is present.  That experience is optimized for camera
    render products and disables renderer presentation, which leaves the main
    Isaac Sim viewport black even though TacEx images continue to update.
    Interactive runs therefore use IsaacLab's full GUI experience and enable
    camera support/presentation through Kit settings.  Headless runs retain
    the optimized headless rendering experience selected by AppLauncher.
    """

    forwarded = list(forwarded)
    defaults = list(DEFAULT_ARGS)
    headless = "--headless" in forwarded
    if headless:
        defaults.remove("--tactile_debug_vis")
    elif not _contains_option(forwarded, "--experience"):
        defaults.extend(("--experience", GUI_CAMERA_EXPERIENCE.as_posix()))
    forwarded, additional_kit_args = _extract_option(forwarded, "--kit_args")
    kit_args = [*(COMMON_CAMERA_KIT_ARGS if headless else GUI_CAMERA_KIT_ARGS)]
    if additional_kit_args:
        kit_args.append(additional_kit_args)
    defaults.append(f"--kit_args={' '.join(kit_args)}")
    return [*defaults, *forwarded]


def main() -> None:
    forwarded = sys.argv[1:]
    os.execv(sys.executable, [sys.executable, "-m", TARGET_MODULE, *build_exec_arguments(forwarded)])


if __name__ == "__main__":
    main()
