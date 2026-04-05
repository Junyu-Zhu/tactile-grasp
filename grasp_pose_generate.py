import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import open3d as o3d
from PIL import Image

import torch
from graspnetAPI import GraspGroup


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
BASELINE_DIR = REPO_ROOT / "graspnet" / "graspnet-baseline"
CHECKPOINT_PATH = BASELINE_DIR / "checkpoint" / "checkpoint-rs.tar"
DEBUG_CAMERA_DIR = SCRIPT_DIR / "debug_camera"

# Isaac Sim camera intrinsics used by load_ur5_in_isaacsim.py
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FX = 480.0
CAMERA_FY = 480.0
CAMERA_CX = CAMERA_WIDTH / 2.0
CAMERA_CY = CAMERA_HEIGHT / 2.0


sys.path.append(str(BASELINE_DIR / "models"))
sys.path.append(str(BASELINE_DIR / "utils"))

from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and visualize 6D grasps for a specified saved RGB-D pair."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=CHECKPOINT_PATH,
        help="Path to checkpoint-rs.tar.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=DEBUG_CAMERA_DIR,
        help="Directory containing the saved RGB/depth frames.",
    )
    parser.add_argument(
        "--rgb_name",
        type=str,
        default="cam_1_frame_000180_rgb.png",
        help="RGB image filename, relative to --data_dir unless an absolute path is given.",
    )
    parser.add_argument(
        "--depth_name",
        type=str,
        default="cam_1_frame_000180_depth.npy",
        help="Depth filename, relative to --data_dir unless an absolute path is given.",
    )
    parser.add_argument("--num_point", type=int, default=20000, help="Point count fed to GraspNet.")
    parser.add_argument("--num_view", type=int, default=300, help="View count used by the model.")
    parser.add_argument("--collision_thresh", type=float, default=0.01, help="Collision threshold.")
    parser.add_argument("--voxel_size", type=float, default=0.01, help="Voxel size for collision detection.")
    parser.add_argument("--top_k", type=int, default=15, help="Number of top grasps to visualize.")
    parser.add_argument(
        "--foreground_depth_margin",
        type=float,
        default=0.008,
        help="Depth margin in meters below the table plane used to seed the foreground mask.",
    )
    parser.add_argument(
        "--roi_pad",
        type=int,
        default=80,
        help="Padding in pixels around the detected foreground bounding box.",
    )
    parser.add_argument(
        "--min_foreground_pixels",
        type=int,
        default=200,
        help="Minimum number of foreground pixels before falling back to the full valid-depth mask.",
    )
    parser.add_argument(
        "--save_grasp_path",
        type=Path,
        default=None,
        help="Optional .npy output path for the predicted grasp group.",
    )
    parser.add_argument(
        "--save_vis_path",
        type=Path,
        default=None,
        help="Optional screenshot path for the Open3D visualization.",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Skip interactive Open3D visualization.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Optional seed used when randomly selecting one grasp from the displayed candidates.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override, e.g. cuda:0 or cpu.",
    )
    return parser.parse_args()


def resolve_input_paths(data_dir: Path, rgb_name: str, depth_name: str) -> tuple[Path, Path]:
    rgb_path = Path(rgb_name)
    depth_path = Path(depth_name)
    if not rgb_path.is_absolute():
        rgb_path = data_dir / rgb_path
    if not depth_path.is_absolute():
        depth_path = data_dir / depth_path
    if not rgb_path.is_file():
        raise FileNotFoundError(f"RGB frame not found: {rgb_path}")
    if not depth_path.is_file():
        raise FileNotFoundError(f"Depth frame not found: {depth_path}")
    return rgb_path, depth_path


def output_stem_from_rgb(rgb_path: Path) -> str:
    stem = rgb_path.stem
    if stem.endswith("_rgb"):
        return stem[:-4]
    return stem


def get_device(device_override: str | None) -> torch.device:
    if device_override is not None:
        return torch.device(device_override)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def get_net(checkpoint_path: Path, num_view: int, device: torch.device) -> GraspNet:
    net = GraspNet(
        input_feature_dim=0,
        num_view=num_view,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.to(device)
    net.eval()
    print(f"[INFO] Loaded checkpoint: {checkpoint_path} (epoch {checkpoint['epoch']})")
    return net


def largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_label = 1 + int(np.argmax(areas))
    return labels == best_label


def build_roi_mask(depth: np.ndarray, margin: float, roi_pad: int, min_pixels: int) -> tuple[np.ndarray, np.ndarray]:
    valid_depth = np.isfinite(depth) & (depth > 0)
    table_depth = float(np.percentile(depth[valid_depth], 95))
    foreground_seed = valid_depth & (depth < (table_depth - margin))

    if int(foreground_seed.sum()) < min_pixels:
        print("[WARN] Foreground mask too small, falling back to all valid depth pixels.")
        return valid_depth, valid_depth

    foreground_seed = largest_component(foreground_seed)
    ys, xs = np.where(foreground_seed)
    x1 = max(0, int(xs.min()) - roi_pad)
    y1 = max(0, int(ys.min()) - roi_pad)
    x2 = min(depth.shape[1], int(xs.max()) + roi_pad + 1)
    y2 = min(depth.shape[0], int(ys.max()) + roi_pad + 1)

    roi_mask = np.zeros_like(valid_depth, dtype=bool)
    roi_mask[y1:y2, x1:x2] = True
    roi_mask &= valid_depth

    print(
        "[INFO] Foreground/table segmentation: "
        f"table_depth={table_depth:.4f} m, foreground_pixels={int(foreground_seed.sum())}, "
        f"roi_pixels={int(roi_mask.sum())}"
    )
    return foreground_seed, roi_mask


def sample_points(points: np.ndarray, colors: np.ndarray, num_point: int) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        raise ValueError("No points available for sampling.")
    if len(points) >= num_point:
        idxs = np.random.choice(len(points), num_point, replace=False)
    else:
        idxs1 = np.arange(len(points))
        idxs2 = np.random.choice(len(points), num_point - len(points), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    return points[idxs], colors[idxs]


def load_frame_data(rgb_path: Path, depth_path: Path) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.array(Image.open(rgb_path), dtype=np.uint8)
    if rgb.shape[-1] > 3:
        rgb = rgb[..., :3]
    depth = np.load(depth_path).astype(np.float32)
    if depth.shape != (CAMERA_HEIGHT, CAMERA_WIDTH):
        raise ValueError(f"Unexpected depth shape {depth.shape}, expected {(CAMERA_HEIGHT, CAMERA_WIDTH)}")
    return rgb, depth


def prepare_scene_cloud(
    rgb: np.ndarray,
    depth: np.ndarray,
    args,
    intrinsics: np.ndarray | None = None,
) -> tuple[dict, np.ndarray, np.ndarray, o3d.geometry.PointCloud, np.ndarray]:
    if intrinsics is None:
        fx = CAMERA_FX
        fy = CAMERA_FY
        cx = CAMERA_CX
        cy = CAMERA_CY
    else:
        intrinsics = np.asarray(intrinsics, dtype=np.float32)
        if intrinsics.shape != (3, 3):
            raise ValueError(f"Expected intrinsics with shape (3, 3), got {intrinsics.shape}")
        fx = float(intrinsics[0, 0])
        fy = float(intrinsics[1, 1])
        cx = float(intrinsics[0, 2])
        cy = float(intrinsics[1, 2])

    height, width = depth.shape
    camera = CameraInfo(
        width,
        height,
        fx,
        fy,
        cx,
        cy,
        scale=1.0,
    )
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    foreground_mask, roi_mask = build_roi_mask(
        depth,
        margin=args.foreground_depth_margin,
        roi_pad=args.roi_pad,
        min_pixels=args.min_foreground_pixels,
    )

    rgb_float = rgb.astype(np.float32) / 255.0
    roi_points = cloud[roi_mask].astype(np.float32)
    roi_colors = rgb_float[roi_mask].astype(np.float32)
    sampled_points, sampled_colors = sample_points(roi_points, roi_colors, args.num_point)

    vis_cloud = o3d.geometry.PointCloud()
    vis_cloud.points = o3d.utility.Vector3dVector(roi_points)
    vis_cloud.colors = o3d.utility.Vector3dVector(roi_colors)
    return (
        {
            "point_clouds": sampled_points,
            "cloud_colors": sampled_colors,
        },
        roi_points,
        roi_colors,
        vis_cloud,
        foreground_mask,
    )


def infer_grasps(net: GraspNet, prepared: dict, device: torch.device) -> GraspGroup:
    end_points = {
        "point_clouds": torch.from_numpy(prepared["point_clouds"][np.newaxis]).to(device),
        "cloud_colors": prepared["cloud_colors"],
    }
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    return GraspGroup(grasp_preds[0].detach().cpu().numpy())


def postprocess_grasps(
    grasp_group: GraspGroup,
    scene_points: np.ndarray,
    collision_thresh: float,
    voxel_size: float,
    top_k: int,
) -> GraspGroup:
    if collision_thresh > 0:
        detector = ModelFreeCollisionDetector(scene_points, voxel_size=voxel_size)
        collision_mask = detector.detect(grasp_group, approach_dist=0.05, collision_thresh=collision_thresh)
        grasp_group = grasp_group[~collision_mask]
        print(f"[INFO] Remaining grasps after collision filtering: {len(grasp_group)}")

    if len(grasp_group) == 0:
        raise RuntimeError("No valid grasps remain after post-processing.")

    grasp_group = grasp_group.nms()
    grasp_group = grasp_group.sort_by_score()
    top_k = min(top_k, len(grasp_group))
    print(f"[INFO] Visualizing top {top_k} grasp(s). Best score: {float(grasp_group[0].score):.4f}")
    return grasp_group[:top_k]


def save_foreground_debug(rgb: np.ndarray, foreground_mask: np.ndarray, out_path: Path):
    overlay = rgb.copy()
    overlay[foreground_mask] = np.array([255, 0, 0], dtype=np.uint8)
    blended = cv2.addWeighted(rgb, 0.6, overlay, 0.4, 0.0)
    Image.fromarray(blended).save(out_path)


def visualize_grasps(
    cloud: o3d.geometry.PointCloud,
    grasp_group: GraspGroup,
    save_vis_path: Path | None,
    show: bool,
):
    geometries = [cloud, *grasp_group.to_open3d_geometry_list()]
    if save_vis_path is not None:
        try:
            save_vis_path.parent.mkdir(parents=True, exist_ok=True)
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="grasp_pose_generate", width=1280, height=720, visible=False)
            for geometry in geometries:
                vis.add_geometry(geometry)
            ctr = vis.get_view_control()
            ctr.set_front([0.0, 0.0, -1.0])
            ctr.set_lookat(cloud.get_center())
            ctr.set_up([0.0, -1.0, 0.0])
            ctr.set_zoom(0.7)
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(str(save_vis_path), do_render=True)
            vis.destroy_window()
            print(f"[INFO] Saved Open3D screenshot to {save_vis_path}")
        except Exception as exc:
            print(f"[WARN] Failed to save Open3D screenshot: {exc!r}")

    if show:
        try:
            o3d.visualization.draw_geometries(geometries)
        except Exception as exc:
            print(f"[WARN] Failed to open interactive visualization window: {exc!r}")


def grasp_to_pose_data(selected_index: int, selected_grasp) -> dict[str, object]:
    translation = selected_grasp.translation.copy()
    rotation_matrix = selected_grasp.rotation_matrix.copy()
    transform_matrix = np.eye(4, dtype=np.float64)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = translation
    return {
        "selected_index": selected_index,
        "score": float(selected_grasp.score),
        "width": float(selected_grasp.width),
        "height": float(selected_grasp.height),
        "depth": float(selected_grasp.depth),
        "object_id": int(selected_grasp.object_id),
        "translation": translation,
        "rotation_matrix": rotation_matrix,
        "transform_matrix": transform_matrix,
        "frame": "camera",
        "grasp": selected_grasp,
    }


def generate_selected_grasp_pose(
    grasp_group: GraspGroup, random_seed: int | None = None
) -> dict[str, object]:
    rng = np.random.default_rng(random_seed)
    selected_index = int(rng.integers(len(grasp_group)))
    return grasp_to_pose_data(selected_index, grasp_group[selected_index])


def generate_grasp_pose_result(args) -> dict[str, object]:
    device = get_device(args.device)
    rgb_path, depth_path = resolve_input_paths(args.data_dir, args.rgb_name, args.depth_name)
    rgb, depth = load_frame_data(rgb_path, depth_path)
    output_stem = output_stem_from_rgb(rgb_path)

    print(f"[INFO] Using RGB frame: {rgb_path}")
    print(f"[INFO] Using depth frame: {depth_path}")
    print(f"[INFO] Torch device: {device}")

    net = get_net(args.checkpoint_path, args.num_view, device)
    prepared, scene_points, _, vis_cloud, foreground_mask = prepare_scene_cloud(rgb, depth, args)
    grasp_group = infer_grasps(net, prepared, device)
    print(f"[INFO] Raw predicted grasps: {len(grasp_group)}")
    grasp_group = postprocess_grasps(
        grasp_group,
        scene_points=scene_points,
        collision_thresh=args.collision_thresh,
        voxel_size=args.voxel_size,
        top_k=args.top_k,
    )
    selected_pose = generate_selected_grasp_pose(grasp_group, args.random_seed)
    return {
        "rgb_path": rgb_path,
        "depth_path": depth_path,
        "output_stem": output_stem,
        "rgb": rgb,
        "depth": depth,
        "foreground_mask": foreground_mask,
        "vis_cloud": vis_cloud,
        "grasp_group": grasp_group,
        "selected_pose": selected_pose,
    }


def build_runtime_grasp_args(**overrides):
    defaults = {
        "num_point": 20000,
        "num_view": 300,
        "collision_thresh": 0.01,
        "voxel_size": 0.01,
        "top_k": 15,
        "foreground_depth_margin": 0.008,
        "roi_pad": 80,
        "min_foreground_pixels": 200,
        "random_seed": None,
        "device": None,
        "checkpoint_path": CHECKPOINT_PATH,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def generate_grasp_pose_from_arrays(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    *,
    checkpoint_path: Path = CHECKPOINT_PATH,
    num_point: int = 20000,
    num_view: int = 300,
    collision_thresh: float = 0.01,
    voxel_size: float = 0.01,
    top_k: int = 15,
    foreground_depth_margin: float = 0.008,
    roi_pad: int = 80,
    min_foreground_pixels: int = 200,
    random_seed: int | None = None,
    device: str | None = None,
    net: GraspNet | None = None,
    net_device: torch.device | None = None,
) -> dict[str, object]:
    runtime_args = build_runtime_grasp_args(
        checkpoint_path=checkpoint_path,
        num_point=num_point,
        num_view=num_view,
        collision_thresh=collision_thresh,
        voxel_size=voxel_size,
        top_k=top_k,
        foreground_depth_margin=foreground_depth_margin,
        roi_pad=roi_pad,
        min_foreground_pixels=min_foreground_pixels,
        random_seed=random_seed,
        device=device,
    )
    inference_device = net_device if net_device is not None else get_device(runtime_args.device)
    runtime_net = net if net is not None else get_net(runtime_args.checkpoint_path, runtime_args.num_view, inference_device)

    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3+), got {rgb.shape}")
    if depth.ndim != 2:
        raise ValueError(f"Expected depth image with shape (H, W), got {depth.shape}")

    rgb = np.ascontiguousarray(rgb[..., :3].astype(np.uint8))
    depth = np.ascontiguousarray(depth.astype(np.float32))
    intrinsics = np.asarray(intrinsics, dtype=np.float32)

    prepared, scene_points, _, vis_cloud, foreground_mask = prepare_scene_cloud(
        rgb,
        depth,
        runtime_args,
        intrinsics=intrinsics,
    )
    grasp_group = infer_grasps(runtime_net, prepared, inference_device)
    grasp_group = postprocess_grasps(
        grasp_group,
        scene_points=scene_points,
        collision_thresh=runtime_args.collision_thresh,
        voxel_size=runtime_args.voxel_size,
        top_k=runtime_args.top_k,
    )
    selected_pose = generate_selected_grasp_pose(grasp_group, runtime_args.random_seed)
    return {
        "rgb": rgb,
        "depth": depth,
        "intrinsics": intrinsics,
        "foreground_mask": foreground_mask,
        "vis_cloud": vis_cloud,
        "scene_points": scene_points,
        "grasp_group": grasp_group,
        "selected_pose": selected_pose,
        "device": inference_device,
    }


def main():
    args = parse_args()
    result = generate_grasp_pose_result(args)
    rgb_path = result["rgb_path"]
    output_stem = result["output_stem"]
    rgb = result["rgb"]
    foreground_mask = result["foreground_mask"]
    vis_cloud = result["vis_cloud"]
    grasp_group = result["grasp_group"]
    selected_pose = result["selected_pose"]

    if args.save_grasp_path is None:
        args.save_grasp_path = args.data_dir / f"{output_stem}_grasps.npy"
    args.save_grasp_path.parent.mkdir(parents=True, exist_ok=True)
    grasp_group.save_npy(str(args.save_grasp_path))
    print(f"[INFO] Saved grasp predictions to {args.save_grasp_path}")

    print(
        "[INFO] Randomly selected one grasp from the displayed candidates "
        f"(index={selected_pose['selected_index']}, score={selected_pose['score']:.4f}, "
        f"frame={selected_pose['frame']})"
    )
    print(f"[INFO] Selected grasp translation:\n{selected_pose['translation']}")
    print(f"[INFO] Selected grasp rotation matrix:\n{selected_pose['rotation_matrix']}")
    print(f"[INFO] Selected grasp transform matrix:\n{selected_pose['transform_matrix']}")

    if args.save_vis_path is None:
        args.save_vis_path = args.data_dir / f"{output_stem}_grasp_vis.png"
    save_foreground_debug(
        rgb,
        foreground_mask,
        args.data_dir / f"{output_stem}_foreground_mask.png",
    )
    visualize_grasps(vis_cloud, grasp_group, args.save_vis_path, show=not args.no_show)


if __name__ == "__main__":
    main()
