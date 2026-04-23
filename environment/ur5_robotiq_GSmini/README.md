# ur5_robotiq_GSmini asset pack

This folder is a local staging area for composing a UR5 + Robotiq 2F-85 + GelSight Mini assembly.

## Copied URDF bases
- `urdf/ur5.urdf`
- `urdf/robotiq_85.urdf`
- `urdf/ur5_robotiq_85.urdf`

## Copied mesh assets
- `meshes/ur5`
- `meshes/robotiq_85`
- `meshes/gelsight_robotiq_connector`
- `meshes/gelsight_mini`

## TacEx GSmini reference (for later tactile display integration)
- `tacex_reference/data/Sensors/GelSight_Mini`
- `tacex_reference/sensors/gelsight_mini`

## Notes
- The copied URDF files still use `../meshes/...` relative paths and this directory layout preserves that pattern.
- The copied connector mesh filename is `gelsight_adaptor.STL` under `meshes/gelsight_robotiq_connector/{visual,collision}/`.
- No standalone connector URDF existed in the copied sources; only the connector mesh assets were copied.
- For GSmini, both the local mesh copy (`meshes/gelsight_mini`) and the TacEx reference assets/config were copied so you can choose either pure mesh composition or TacEx-aligned sensor composition later.
