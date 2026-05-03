"""Compatibility entry point for the Phase 2 UR5/GSmini cube grasp simulation.

Historically this grasp demo lived in ``ur5_phase2_grasp_test.py``.  Keep this
thin wrapper so commands that launch ``ur5_phase2_grasp_sim.py`` exercise the
same fixed GSmini alignment, force-controlled close, and live TacEx tactile
display path.
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(Path(__file__).with_name("ur5_phase2_grasp_test.py").as_posix(), run_name="__main__")
