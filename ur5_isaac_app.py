"""Local Isaac Sim startup compatibility helpers for UR5 entry points."""

from __future__ import annotations

import os
import sys


def patch_headless_viewport_wait() -> None:
    """Avoid an unbounded Isaac Sim 4.5 viewport wait in headless mode.

    On this pinned IsaacLab/Isaac Sim combination the headless experience can
    expose a viewport object whose handle never becomes valid.  SimulationApp's
    private startup helper otherwise loops forever before returning control to
    the UR5 script.  RTX cameras use their own render products, so the active
    GUI viewport is not required in headless runs.  GUI startup keeps the
    upstream behavior unchanged.
    """

    from isaacsim.simulation_app import SimulationApp

    if getattr(SimulationApp._wait_for_viewport, "_ur5_headless_guard", False):
        return

    original_wait = SimulationApp._wait_for_viewport

    def _wait_for_viewport(self) -> None:
        if self.config.get("headless", False):
            for _ in range(10):
                self._app.update()
            return
        original_wait(self)

    _wait_for_viewport._ur5_headless_guard = True
    SimulationApp._wait_for_viewport = _wait_for_viewport


def exit_headless_without_kit_shutdown(exit_code: int) -> None:
    """Exit after flushing reports when Isaac/Kit graceful shutdown would hang."""

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
