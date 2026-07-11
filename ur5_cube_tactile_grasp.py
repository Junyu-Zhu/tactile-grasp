"""Run the canonical cube grasp with two live TacEx GelSight Mini streams.

This is the short user-facing entrypoint.  The implementation stays in the
modular Phase3 scene/geometry/motion/logging/runner files instead of duplicating
the legacy monolithic Phase2 grasp script.

Run from the repository root after activating the Isaac/TacEx environment::

    python ur5_cube_tactile_grasp.py

Pass ``--headless`` for automated verification.  Any additional arguments are
forwarded to ``ur5_phase3_data_collection.py`` and override these defaults.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


TARGET = Path(__file__).with_name("ur5_phase3_data_collection.py")
DEFAULT_ARGS = [
    "--object_id", "cube",
    "--protocol_variant", "contact_hold_micro_lift",
    "--trials", "1",
    "--tactile_sides", "both",
    "--tactile_debug_vis",
    "--enable_cameras",
]


def main() -> None:
    forwarded = sys.argv[1:]
    defaults = list(DEFAULT_ARGS)
    if "--headless" in forwarded:
        defaults.remove("--tactile_debug_vis")
    os.execv(
        sys.executable,
        [sys.executable, TARGET.as_posix(), *defaults, *forwarded],
    )


if __name__ == "__main__":
    main()
