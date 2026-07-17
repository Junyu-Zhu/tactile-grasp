from __future__ import annotations

from pathlib import Path

import ur5_cube_tactile_grasp as entrypoint


def _option_value(arguments: list[str], option: str) -> str | None:
    for index, argument in enumerate(arguments):
        if argument == option:
            return arguments[index + 1]
        if argument.startswith(f"{option}="):
            return argument.split("=", 1)[1]
    return None


def test_interactive_entrypoint_uses_viewport_capable_camera_experience() -> None:
    arguments = entrypoint.build_exec_arguments([])

    experience = _option_value(arguments, "--experience")
    assert experience == entrypoint.GUI_CAMERA_EXPERIENCE.as_posix()
    assert Path(experience).is_file()
    assert Path(experience).name == "isaaclab.python.kit"

    kit_args = _option_value(arguments, "--kit_args")
    assert kit_args is not None
    assert "--/isaaclab/cameras_enabled=true" in kit_args
    assert "--/exts/omni.renderer.core/present/enabled=true" in kit_args
    assert "--/rtx/post/aa/op=2" in kit_args
    assert "--/rtx/translucency/enabled=true" in kit_args
    assert "--/app/renderer/waitIdle=true" in kit_args
    assert "--/app/hydraEngine/waitIdle=true" in kit_args
    assert "--/app/updateOrder/checkForHydraRenderComplete=1000" in kit_args


def test_headless_entrypoint_keeps_optimized_headless_camera_experience() -> None:
    arguments = entrypoint.build_exec_arguments(["--headless"])

    assert _option_value(arguments, "--experience") is None
    kit_args = _option_value(arguments, "--kit_args")
    assert kit_args is not None
    assert "--/rtx/post/aa/op=2" in kit_args
    assert "--/rtx/translucency/enabled=true" in kit_args
    assert "--/app/renderer/waitIdle=true" in kit_args
    assert "--/exts/omni.renderer.core/present/enabled=true" not in kit_args
    assert "--tactile_debug_vis" not in arguments
    assert "--enable_cameras" in arguments


def test_explicit_experience_overrides_interactive_default() -> None:
    custom_experience = "/tmp/custom.kit"
    arguments = entrypoint.build_exec_arguments([f"--experience={custom_experience}"])

    assert _option_value(arguments, "--experience") == custom_experience
    assert entrypoint.GUI_CAMERA_EXPERIENCE.as_posix() not in arguments


def test_interactive_entrypoint_preserves_additional_kit_arguments() -> None:
    arguments = entrypoint.build_exec_arguments(["--kit_args=--/app/printConfig=true"])

    kit_args = _option_value(arguments, "--kit_args")
    assert kit_args is not None
    assert "--/isaaclab/cameras_enabled=true" in kit_args
    assert "--/exts/omni.renderer.core/present/enabled=true" in kit_args
    assert "--/app/printConfig=true" in kit_args


def test_forwarded_arguments_follow_defaults_for_argparse_override() -> None:
    arguments = entrypoint.build_exec_arguments(["--trials", "7", "--close_steps=99"])

    assert arguments.index("--trials") < len(arguments) - 2
    assert arguments[-3:] == ["--trials", "7", "--close_steps=99"]
    assert arguments[-1] == "--close_steps=99"


def test_main_executes_cube_collector_module_with_current_python(monkeypatch) -> None:
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(entrypoint.sys, "argv", ["ur5_cube_tactile_grasp.py", "--headless", "--trials", "2"])
    monkeypatch.setattr(entrypoint.os, "execv", lambda executable, argv: calls.append((executable, argv)))

    entrypoint.main()

    assert len(calls) == 1
    executable, argv = calls[0]
    assert executable == entrypoint.sys.executable
    assert argv[:3] == [entrypoint.sys.executable, "-m", entrypoint.TARGET_MODULE]
    assert argv[-3:] == ["--headless", "--trials", "2"]
