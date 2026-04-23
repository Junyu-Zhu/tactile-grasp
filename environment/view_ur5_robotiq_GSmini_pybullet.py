from __future__ import annotations

import argparse
from pathlib import Path
import time


FINGER_MIMIC = {
    'finger_joint': 1.0,
    'left_inner_finger_joint': -1.0,
    'left_inner_knuckle_joint': 1.0,
    'right_outer_knuckle_joint': 1.0,
    'right_inner_finger_joint': -1.0,
    'right_inner_knuckle_joint': 1.0,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Load and preview ur5_robotiq_GSmini.urdf in PyBullet.')
    parser.add_argument(
        '--urdf',
        type=Path,
        default=Path(__file__).resolve().parent / 'ur5_robotiq_GSmini' / 'urdf' / 'ur5_robotiq_GSmini.urdf',
        help='Path to the URDF to load.',
    )
    parser.add_argument('--direct', action='store_true', help='Use PyBullet DIRECT mode instead of GUI.')
    parser.add_argument('--steps', type=int, default=2400, help='Simulation steps to run in DIRECT mode.')
    parser.add_argument(
        '--enable-gravity',
        action='store_true',
        help='Enable gravity (default off so finger/unmimicked joints do not sag).',
    )
    return parser


def collect_joint_indices(pybullet_module, robot_id):
    name_to_index = {}
    revolute_indices = []
    for joint_index in range(pybullet_module.getNumJoints(robot_id)):
        joint_info = pybullet_module.getJointInfo(robot_id, joint_index)
        joint_name = joint_info[1].decode()
        joint_type = joint_info[2]
        name_to_index[joint_name] = joint_index
        if joint_type == pybullet_module.JOINT_REVOLUTE:
            revolute_indices.append(joint_index)
    return name_to_index, revolute_indices


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        import pybullet as p
        import pybullet_data
    except Exception as exc:
        raise SystemExit(f'PyBullet is required. Try the pybullet conda env. Import error: {exc}')

    urdf_path = args.urdf.resolve()
    if not urdf_path.is_file():
        raise SystemExit(f'URDF not found: {urdf_path}')

    mode = p.DIRECT if args.direct else p.GUI
    client = p.connect(mode)
    try:
        plane_path = Path(pybullet_data.getDataPath()) / 'plane.urdf'
        p.setAdditionalSearchPath(str(urdf_path.parent))
        if args.enable_gravity:
            p.setGravity(0, 0, -9.81)
        else:
            p.setGravity(0, 0, 0)
        if plane_path.is_file():
            p.loadURDF(str(plane_path))
        else:
            print(f'[WARN] plane.urdf not found in {plane_path.parent}, skipping ground plane.')

        robot_id = p.loadURDF(
            str(urdf_path),
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
        )
        print(f'[INFO] Loaded URDF: {urdf_path}')
        print(f'[INFO] bodyUniqueId: {robot_id}')
        print(f'[INFO] Num joints: {p.getNumJoints(robot_id)}')
        for joint_index in range(p.getNumJoints(robot_id)):
            joint_info = p.getJointInfo(robot_id, joint_index)
            print(f'[JOINT] {joint_index}: {joint_info[1].decode()} parentLinkIndex={joint_info[16]}')

        name_to_index, revolute_indices = collect_joint_indices(p, robot_id)
        mimic_indices = {}
        for name, multiplier in FINGER_MIMIC.items():
            if name in name_to_index:
                mimic_indices[name_to_index[name]] = multiplier
            else:
                print(f'[WARN] joint not found, skipping mimic for: {name}')

        for joint_index in revolute_indices:
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=0.0,
                force=80.0,
            )

        if mode == p.GUI:
            p.resetDebugVisualizerCamera(
                cameraDistance=1.1,
                cameraYaw=135,
                cameraPitch=-28,
                cameraTargetPosition=[0.0, 0.0, 0.85],
            )
            finger_slider = p.addUserDebugParameter('finger_joint', 0.0, 0.80, 0.0)
            print('[INFO] GUI mode active. Drag the finger_joint slider to open/close the gripper.')
            while p.isConnected():
                try:
                    target = p.readUserDebugParameter(finger_slider)
                except p.error:
                    break
                for idx, multiplier in mimic_indices.items():
                    p.setJointMotorControl2(
                        bodyUniqueId=robot_id,
                        jointIndex=idx,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=target * multiplier,
                        force=80.0,
                    )
                p.stepSimulation()
                time.sleep(1.0 / 240.0)
        else:
            for _ in range(args.steps):
                p.stepSimulation()
            print(f'[INFO] DIRECT mode finished after {args.steps} steps.')
    finally:
        if p.isConnected(client):
            p.disconnect(client)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
