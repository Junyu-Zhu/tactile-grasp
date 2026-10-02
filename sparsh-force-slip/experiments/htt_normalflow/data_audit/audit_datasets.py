#!/usr/bin/env python3
"""Reproducible, full-data audit for HTT GSmini and NormalFlow.

The script performs no training. It optionally extracts NormalFlow after checking
every ZIP member for path traversal, then writes per-episode JSON manifests and a
Markdown summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


SCHEMA_VERSION = "1.0"
HTT_ARRAY_KEYS = ("ref_frame", "ref_force", "tactile_img", "6d_force")
LABEL_KEYS = ("sliding_labels", "sliding_labels_signature", "sliding_labels_bracket")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def hash_arrays(arrays: Iterable[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode())
        digest.update(str(contiguous.dtype).encode())
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def scalar(value: np.ndarray) -> Any:
    result = value.item()
    return result.item() if isinstance(result, np.generic) else result


def array_stats(array: np.ndarray) -> dict[str, Any]:
    numeric = np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)
    finite_count = int(np.isfinite(array).sum()) if numeric else None
    nonfinite_count = int(array.size - finite_count) if numeric else None
    if numeric and finite_count:
        finite = array[np.isfinite(array)]
        minimum = float(finite.min())
        maximum = float(finite.max())
    else:
        minimum = maximum = None
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "size": int(array.size),
        "min": minimum,
        "max": maximum,
        "nonfinite_count": nonfinite_count,
    }


def update_extrema(accumulator: dict[str, Any], stats: dict[str, Any]) -> None:
    if stats["min"] is not None:
        accumulator["min"] = min(accumulator.get("min", math.inf), stats["min"])
        accumulator["max"] = max(accumulator.get("max", -math.inf), stats["max"])
    accumulator["nonfinite_count"] = accumulator.get("nonfinite_count", 0) + (stats["nonfinite_count"] or 0)
    accumulator["elements"] = accumulator.get("elements", 0) + stats["size"]


def canonical_json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
    temporary.replace(path)


def inspect_label_file(path: Path, expected_length: int) -> tuple[dict[str, Any], list[str]]:
    anomalies: list[str] = []
    with np.load(path, allow_pickle=False) as labels:
        result: dict[str, Any] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "keys": sorted(labels.files),
            "labels": {},
        }
        arrays: dict[str, np.ndarray] = {}
        for key in LABEL_KEYS:
            if key not in labels:
                anomalies.append(f"missing label key: {key}")
                continue
            array = labels[key]
            arrays[key] = array
            counts = Counter(int(value) for value in array.tolist())
            result["labels"][key] = {
                **array_stats(array),
                "counts": {str(k): v for k, v in sorted(counts.items())},
            }
            if len(array) != expected_length:
                anomalies.append(f"{key} length {len(array)} != tactile length {expected_length}")
            unexpected = sorted(set(counts).difference({0, 1, 2}))
            if unexpected:
                anomalies.append(f"{key} unexpected class ids: {unexpected}")
        pairwise: dict[str, int] = {}
        for index, left in enumerate(LABEL_KEYS):
            for right in LABEL_KEYS[index + 1 :]:
                if left in arrays and right in arrays and arrays[left].shape == arrays[right].shape:
                    pairwise[f"{left}__vs__{right}"] = int(np.count_nonzero(arrays[left] != arrays[right]))
        result["pairwise_difference_frames"] = pairwise
        for key in ("mus", "fz", "c_plus", "c_minus"):
            if key in labels:
                result[key] = array_stats(labels[key])
                if len(labels[key]) != expected_length:
                    anomalies.append(f"{key} length {len(labels[key])} != tactile length {expected_length}")
        for key in ("n0", "mu0", "sigma", "episode_path"):
            if key in labels:
                result[key] = scalar(labels[key])
        if "labeling_meta" in labels:
            raw_meta = scalar(labels["labeling_meta"])
            try:
                result["labeling_meta"] = json.loads(raw_meta)
            except json.JSONDecodeError:
                result["labeling_meta_raw"] = raw_meta
                anomalies.append("labeling_meta is not valid JSON")
    return result, anomalies


def audit_htt(root: Path) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    aggregate_arrays: dict[str, dict[str, Any]] = defaultdict(dict)
    aggregate_axes = {
        "6d_force": [{"min": math.inf, "max": -math.inf, "nonfinite_count": 0} for _ in range(6)],
        "ref_force": [{"min": math.inf, "max": -math.inf, "nonfinite_count": 0} for _ in range(6)],
    }
    label_totals = {key: Counter() for key in LABEL_KEYS}
    label_pairwise_totals: Counter[str] = Counter()
    label_base_parameters: Counter[str] = Counter()
    label_signature_parameters: Counter[str] = Counter()
    label_bracket_fixed_parameters: Counter[str] = Counter()
    bracket_starts: list[int] = []
    bracket_stops: list[int] = []
    physical_hash_paths: dict[str, list[str]] = defaultdict(list)

    raw_files = sorted((root / "force/gsmini/processed").glob("*/*.npz"))
    raw_files += sorted((root / "slip/gsmini/processed").glob("*/*.npz"))
    for index, path in enumerate(raw_files, 1):
        relative = path.relative_to(root)
        subset = relative.parts[0]
        anomalies: list[str] = []
        with np.load(path, allow_pickle=False) as archive:
            missing = sorted(set((*HTT_ARRAY_KEYS, "probe", "mode")).difference(archive.files))
            anomalies.extend(f"missing key: {key}" for key in missing)
            fields = {key: array_stats(archive[key]) for key in HTT_ARRAY_KEYS if key in archive}
            for key, stats in fields.items():
                update_extrema(aggregate_arrays[key], stats)
            tactile_length = int(archive["tactile_img"].shape[0]) if "tactile_img" in archive else -1
            force_length = int(archive["6d_force"].shape[0]) if "6d_force" in archive else -1
            if tactile_length != force_length:
                anomalies.append(f"tactile length {tactile_length} != force length {force_length}")
            for key in ("6d_force", "ref_force"):
                if key in archive and archive[key].shape[-1:] == (6,):
                    flat = archive[key].reshape(-1, 6)
                    for axis in range(6):
                        values = flat[:, axis]
                        finite = values[np.isfinite(values)]
                        aggregate_axes[key][axis]["nonfinite_count"] += int(values.size - finite.size)
                        if finite.size:
                            aggregate_axes[key][axis]["min"] = min(aggregate_axes[key][axis]["min"], float(finite.min()))
                            aggregate_axes[key][axis]["max"] = max(aggregate_axes[key][axis]["max"], float(finite.max()))
            probe = int(scalar(archive["probe"])) if "probe" in archive else None
            mode = str(scalar(archive["mode"])) if "mode" in archive else None
            path_probe_match = re.search(r"p(\d+)_", relative.as_posix())
            path_probe = int(path_probe_match.group(1)) if path_probe_match else None
            if probe != path_probe:
                anomalies.append(f"probe field {probe} != path probe {path_probe}")
            expected_mode = "static" if subset == "force" else "sliding"
            if mode != expected_mode:
                anomalies.append(f"mode field {mode!r} != expected {expected_mode!r}")
            physical_parts = [archive[key] for key in HTT_ARRAY_KEYS if key in archive]
            physical_hash = hash_arrays(physical_parts)
            physical_hash_paths[physical_hash].append(str(relative))

        label_info = None
        if subset == "slip":
            label_relative = Path("slip/gsmini/sliding_labeled") / relative.relative_to("slip/gsmini/processed")
            label_path = (root / label_relative).with_name(path.stem + ".labeled.npz")
            if not label_path.exists():
                anomalies.append(f"missing label file: {label_relative}")
            else:
                label_info, label_anomalies = inspect_label_file(label_path, tactile_length)
                label_info["path"] = str(label_path.relative_to(root))
                anomalies.extend(label_anomalies)
                for key, details in label_info["labels"].items():
                    label_totals[key].update({int(k): v for k, v in details["counts"].items()})
                label_pairwise_totals.update(label_info["pairwise_difference_frames"])
                meta = label_info.get("labeling_meta", {})
                if "params" in meta:
                    label_base_parameters[json.dumps(meta["params"], sort_keys=True)] += 1
                if "params_signature" in meta:
                    label_signature_parameters[json.dumps(meta["params_signature"], sort_keys=True)] += 1
                if "params_bracket" in meta:
                    bracket = dict(meta["params_bracket"])
                    if "t_start" in bracket:
                        bracket_starts.append(int(bracket.pop("t_start")))
                    if "t_stop" in bracket:
                        bracket_stops.append(int(bracket.pop("t_stop")))
                    label_bracket_fixed_parameters[json.dumps(bracket, sort_keys=True)] += 1
        episodes.append(
            {
                "episode_id": f"htt:{relative.as_posix()}",
                "relative_path": str(relative),
                "domain": "htt_gsmini",
                "subset": subset,
                "object": None,
                "probe": probe,
                "mode": mode,
                "length": tactile_length,
                "time": {"source": "unknown", "fps": None, "timestamps_available": False, "unit": "steps"},
                "force": {
                    "available": "6d_force" in fields,
                    "shape": fields.get("6d_force", {}).get("shape"),
                    "unit": "unknown_not_documented_in_dataset_files",
                    "axis_convention": "unknown_not_documented_in_dataset_files",
                },
                "supervision": {
                    "force_valid": "6d_force" in fields,
                    "slip_valid": label_info is not None,
                    "slip_label_source": "sliding_labels_bracket" if label_info else None,
                },
                "fields": fields,
                "file_sha256": sha256_file(path),
                "physical_content_sha256": physical_hash,
                "label_file": label_info,
                "anomalies": anomalies,
            }
        )
        if index % 20 == 0 or index == len(raw_files):
            print(f"HTT: audited {index}/{len(raw_files)} episodes", flush=True)

    duplicate_groups = [
        {"physical_content_sha256": digest, "paths": paths}
        for digest, paths in sorted(physical_hash_paths.items())
        if len(paths) > 1
    ]
    cross_subset_by_key: dict[tuple[int | None, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for episode in episodes:
        cross_subset_by_key[(episode["probe"], Path(episode["relative_path"]).name)][episode["subset"]] = episode
    basename_overlap_groups = []
    for (probe, basename), members in sorted(cross_subset_by_key.items()):
        if set(members) != {"force", "slip"}:
            continue
        frame_counters: dict[str, Counter[bytes]] = {}
        for subset, episode in members.items():
            with np.load(root / episode["relative_path"], allow_pickle=False) as archive:
                frame_counters[subset] = Counter(
                    hashlib.sha256(memoryview(np.ascontiguousarray(frame)).cast("B")).digest()
                    for frame in archive["tactile_img"]
                )
        exact_shared_frames = sum((frame_counters["force"] & frame_counters["slip"]).values())
        basename_overlap_groups.append(
            {
                "probe": probe,
                "basename": basename,
                "force_path": members["force"]["relative_path"],
                "slip_path": members["slip"]["relative_path"],
                "same_full_physical_content_sha256": members["force"]["physical_content_sha256"] == members["slip"]["physical_content_sha256"],
                "exact_shared_tactile_frames": exact_shared_frames,
            }
        )
    unmatched_labels = []
    for label_path in sorted((root / "slip/gsmini/sliding_labeled").glob("*/*.labeled.npz")):
        raw_name = label_path.name.replace(".labeled.npz", ".npz")
        raw_path = root / "slip/gsmini/processed" / label_path.parent.name / raw_name
        if not raw_path.exists():
            unmatched_labels.append(str(label_path.relative_to(root)))

    for values in aggregate_axes.values():
        for axis in values:
            if axis["min"] == math.inf:
                axis["min"] = axis["max"] = None
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "HTT GSmini force+slip",
        "root": str(root),
        "generated_unix_time": time.time(),
        "episode_count": len(episodes),
        "episodes": episodes,
        "summary": {
            "subset_episode_counts": dict(Counter(item["subset"] for item in episodes)),
            "subset_frame_counts": dict(Counter({subset: sum(e["length"] for e in episodes if e["subset"] == subset) for subset in {e["subset"] for e in episodes}})),
            "probe_episode_counts": {str(k): v for k, v in sorted(Counter(e["probe"] for e in episodes).items())},
            "array_global_ranges": aggregate_arrays,
            "force_axis_ranges_by_index": aggregate_axes,
            "label_class_totals": {key: {str(k): v for k, v in sorted(counts.items())} for key, counts in label_totals.items()},
            "label_pairwise_difference_frames": dict(label_pairwise_totals),
            "labeling_parameter_variants": {
                "params": [{"parameters": json.loads(key), "episodes": count} for key, count in sorted(label_base_parameters.items())],
                "params_signature": [{"parameters": json.loads(key), "episodes": count} for key, count in sorted(label_signature_parameters.items())],
                "params_bracket_excluding_episode_bounds": [{"parameters": json.loads(key), "episodes": count} for key, count in sorted(label_bracket_fixed_parameters.items())],
                "bracket_t_start_range": [min(bracket_starts), max(bracket_starts)] if bracket_starts else None,
                "bracket_t_stop_range": [min(bracket_stops), max(bracket_stops)] if bracket_stops else None,
            },
            "duplicate_physical_content_groups": duplicate_groups,
            "cross_subset_same_probe_basename_groups": basename_overlap_groups,
            "unmatched_label_files": unmatched_labels,
            "episodes_with_anomalies": sum(bool(e["anomalies"]) for e in episodes),
        },
        "documented_limits": {
            "time": "No per-frame timestamps or sampling rate are documented in README.md/FORMAT.md or stored in the audited NPZ files.",
            "force_units_and_axes": "The files describe aligned ATI 6D force-torque readings, but do not document component order, sign/frame convention, or units.",
            "primary_slip_label": "sliding_labels_bracket: 0 static, 1 incipient, 2 gross.",
            "duplicate_detection": "Full-content and exact tactile-frame hashes can establish byte-identical duplication, but different hashes cannot exclude different excerpts from the same physical collection trial.",
        },
    }


def safe_extract_zip(zip_path: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            target = (destination / info.filename).resolve()
            try:
                target.relative_to(destination_resolved)
            except ValueError as exc:
                raise RuntimeError(f"unsafe ZIP member: {info.filename}") from exc
        for info in archive.infolist():
            target = destination / info.filename
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif target.is_file() and target.stat().st_size == info.file_size:
                continue
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("wb") as sink:
                    shutil.copyfileobj(source, sink)
    extracted_root = destination / "normalflow_dataset"
    if not extracted_root.is_dir():
        raise RuntimeError(f"expected extracted root is missing: {extracted_root}")
    return extracted_root


def find_ffprobe(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    system = shutil.which("ffprobe")
    if system:
        return system
    candidates = sorted(Path("/home/zjy/miniconda3/pkgs").glob("ffmpeg-*/bin/ffprobe"), reverse=True)
    return str(candidates[0]) if candidates else None


def probe_video(path: Path, ffprobe: str | None) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "sha256": sha256_file(path)}
    if ffprobe:
        command = [
            ffprobe,
            "-v", "error",
            "-count_frames",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames,duration",
            "-of", "json",
            str(path),
        ]
        probe_env = os.environ.copy()
        # This server has ffprobe in Conda's package cache rather than a linked
        # environment. Supply the matching package-cache libraries without
        # changing the user's environment or installing anything.
        if "/miniconda3/pkgs/" in ffprobe:
            package_root = Path(ffprobe).parents[2]
            package_libs = sorted(str(path) for path in package_root.glob("*/lib") if path.is_dir())
            inherited = probe_env.get("LD_LIBRARY_PATH")
            probe_env["LD_LIBRARY_PATH"] = ":".join(package_libs + ([inherited] if inherited else []))
        completed = subprocess.run(command, check=True, capture_output=True, text=True, env=probe_env)
        streams = json.loads(completed.stdout).get("streams", [])
        result["ffprobe"] = streams[0] if streams else {}
    capture = cv2.VideoCapture(str(path))
    decoded = 0
    image_min = math.inf
    image_max = -math.inf
    image_shape = None
    decode_failed = False
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded += 1
        image_shape = list(frame.shape)
        image_min = min(image_min, int(frame.min()))
        image_max = max(image_max, int(frame.max()))
    declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    opencv_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if decoded == 0:
        decode_failed = True
    capture.release()
    result["decoded_frame_count"] = decoded
    result["opencv_declared_frame_count"] = declared
    result["opencv_fps"] = opencv_fps
    result["decoded_image_shape"] = image_shape
    result["decoded_pixel_range"] = [None, None] if decoded == 0 else [image_min, image_max]
    result["decode_failed"] = decode_failed
    return result


def audit_memmapped_array(path: Path, chunk: int = 16) -> dict[str, Any]:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    minimum = math.inf
    maximum = -math.inf
    nonfinite = 0
    if array.ndim == 0:
        blocks = [array]
    else:
        blocks = (array[index : index + chunk] for index in range(0, len(array), chunk))
    for block in blocks:
        numeric = np.issubdtype(block.dtype, np.number) or np.issubdtype(block.dtype, np.bool_)
        if not numeric:
            continue
        finite_mask = np.isfinite(block)
        nonfinite += int(block.size - finite_mask.sum())
        if finite_mask.any():
            finite = block[finite_mask]
            minimum = min(minimum, float(finite.min()))
            maximum = max(maximum, float(finite.max()))
    return {
        "path": str(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "size": int(array.size),
        "min": None if minimum == math.inf else minimum,
        "max": None if maximum == -math.inf else maximum,
        "nonfinite_count": nonfinite,
        "sha256": sha256_file(path),
    }


def parse_fraction(value: str | None) -> float | None:
    if not value or value in {"0/0", "N/A"}:
        return None
    numerator, denominator = value.split("/")
    return float(numerator) / float(denominator)


def audit_normalflow(root: Path, ffprobe: str | None) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    episode_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    for index, episode_dir in enumerate(episode_dirs, 1):
        match = re.fullmatch(r"(.+?)(\d+)", episode_dir.name)
        object_name = match.group(1) if match else episode_dir.name
        trial = int(match.group(2)) if match else None
        anomalies: list[str] = []
        files = {path.name: path for path in episode_dir.iterdir() if path.is_file()}
        expected = {"gelsight.avi", "webcam.avi", "true_start_T_currs.npy", "contact_masks.npy", "gradient_maps.npy"}
        missing = sorted(expected.difference(files))
        anomalies.extend(f"missing file: {name}" for name in missing)

        arrays = {name: audit_memmapped_array(files[name]) for name in sorted(expected) if name.endswith(".npy") and name in files}
        tactile_video = probe_video(files["gelsight.avi"], ffprobe) if "gelsight.avi" in files else None
        webcam_video = probe_video(files["webcam.avi"], ffprobe) if "webcam.avi" in files else None
        length = tactile_video["decoded_frame_count"] if tactile_video else -1
        for name, details in arrays.items():
            if details["shape"] and details["shape"][0] != length:
                anomalies.append(f"{name} length {details['shape'][0]} != decoded tactile frames {length}")
            if details["nonfinite_count"]:
                anomalies.append(f"{name} has {details['nonfinite_count']} non-finite values")
        if tactile_video:
            declared = tactile_video["opencv_declared_frame_count"]
            if declared != length:
                anomalies.append(f"OpenCV declared {declared} tactile frames but decoded {length}")
            ff_frames = tactile_video.get("ffprobe", {}).get("nb_read_frames")
            if ff_frames not in (None, "N/A") and int(ff_frames) != length:
                anomalies.append(f"ffprobe counted {ff_frames} tactile frames but OpenCV decoded {length}")
            if tactile_video["decode_failed"]:
                anomalies.append("tactile video failed to decode")
        pose = arrays.get("true_start_T_currs.npy")
        if pose and pose["shape"][1:] != [4, 4]:
            anomalies.append(f"pose shape is {pose['shape']}, expected [N,4,4]")
        if "true_start_T_currs.npy" in files:
            transforms = np.load(files["true_start_T_currs.npy"], mmap_mode="r", allow_pickle=False)
            if transforms.ndim == 3 and transforms.shape[1:] == (4, 4):
                last_row_error = float(np.max(np.abs(transforms[:, 3, :] - np.array([0, 0, 0, 1]))))
                rotations = transforms[:, :3, :3]
                identity = np.eye(3)
                orthogonality_error = float(np.max(np.abs(np.swapaxes(rotations, 1, 2) @ rotations - identity)))
                determinant_error = float(np.max(np.abs(np.linalg.det(rotations) - 1.0)))
            else:
                last_row_error = orthogonality_error = determinant_error = None
        else:
            last_row_error = orthogonality_error = determinant_error = None

        ff_stream = tactile_video.get("ffprobe", {}) if tactile_video else {}
        avg_fps = parse_fraction(ff_stream.get("avg_frame_rate")) or (tactile_video.get("opencv_fps") if tactile_video else None)
        episode_file_hashes = [bytes.fromhex(details["sha256"]) for details in arrays.values()]
        if tactile_video:
            episode_file_hashes.append(bytes.fromhex(tactile_video["sha256"]))
        episode_hash = hashlib.sha256(b"".join(episode_file_hashes)).hexdigest()
        episodes.append(
            {
                "episode_id": f"normalflow:{episode_dir.name}",
                "relative_path": episode_dir.name,
                "domain": "normalflow_gsmini_markerless",
                "object": object_name,
                "trial": trial,
                "probe": None,
                "length": length,
                "time": {
                    "source": "video_container_average_frame_rate",
                    "fps": avg_fps,
                    "timestamps_available": False,
                    "unit": "steps; nominal seconds only via container average fps",
                },
                "supervision": {
                    "pose_valid": pose is not None,
                    "slip_valid": False,
                    "slip_label_source": None,
                    "contact_mask_source": "derived_from_tactile_images",
                    "gradient_map_source": "derived_from_tactile_images",
                },
                "arrays": arrays,
                "tactile_video": tactile_video,
                "webcam_video": webcam_video,
                "pose_transform_checks": {
                    "last_row_max_abs_error": last_row_error,
                    "rotation_orthogonality_max_abs_error": orthogonality_error,
                    "rotation_determinant_max_abs_error": determinant_error,
                },
                "episode_content_sha256": episode_hash,
                "anomalies": anomalies,
            }
        )
        print(f"NormalFlow: audited {index}/{len(episode_dirs)} episodes", flush=True)
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "NormalFlow markerless GSmini",
        "root": str(root),
        "generated_unix_time": time.time(),
        "episode_count": len(episodes),
        "episodes": episodes,
        "summary": {
            "object_episode_counts": dict(sorted(Counter(e["object"] for e in episodes).items())),
            "total_decoded_tactile_frames": sum(e["length"] for e in episodes),
            "fps_min": min((e["time"]["fps"] for e in episodes if e["time"]["fps"] is not None), default=None),
            "fps_max": max((e["time"]["fps"] for e in episodes if e["time"]["fps"] is not None), default=None),
            "episodes_with_anomalies": sum(bool(e["anomalies"]) for e in episodes),
            "duplicate_episode_content_hashes": [
                {"sha256": digest, "episodes": names}
                for digest, names in sorted(
                    ((digest, [e["episode_id"] for e in episodes if e["episode_content_sha256"] == digest]) for digest in {e["episode_content_sha256"] for e in episodes}),
                    key=lambda item: item[0],
                )
                if len(names) > 1
            ],
        },
        "documented_limits": {
            "slip": "No slip labels are provided. Sensor/object relative motion must not be interpreted as slip supervision.",
            "time": "No independent timestamp arrays are provided. Container average frame rate is nominal timing metadata, so forecasting should be reported in steps unless independently calibrated.",
            "contact_geometry": "contact_masks and gradient_maps are derived from tactile images and are not independent ground truth.",
        },
    }


def write_report(htt: dict[str, Any], normalflow: dict[str, Any], output: Path) -> None:
    hs = htt["summary"]
    ns = normalflow["summary"]
    htt_anomalies = [(e["episode_id"], issue) for e in htt["episodes"] for issue in e["anomalies"]]
    nf_anomalies = [(e["episode_id"], issue) for e in normalflow["episodes"] for issue in e["anomalies"]]
    lines = [
        "# HTT and NormalFlow full-data audit",
        "",
        f"Generated: `{time.strftime('%Y-%m-%d %H:%M:%S %z')}`",
        "",
        "## Manifest schema",
        "",
        "Each manifest has top-level dataset metadata, `summary`, `documented_limits`, and an `episodes` list. Each episode records identity (`episode_id`, path, object/probe/domain), sequence length, timing source, supervision validity/source, file/content SHA-256 hashes, field shapes/ranges/non-finite counts, and `anomalies`.",
        "",
        "## HTT GSmini",
        "",
        f"- Episodes: {htt['episode_count']} ({json.dumps(hs['subset_episode_counts'], sort_keys=True)})",
        f"- Frames: {json.dumps(hs['subset_frame_counts'], sort_keys=True)}",
        f"- Probe episode counts: {json.dumps(hs['probe_episode_counts'], sort_keys=True)}",
        f"- `sliding_labels_bracket` totals: {json.dumps(hs['label_class_totals']['sliding_labels_bracket'], sort_keys=True)}",
        f"- Label-version differing frames: {json.dumps(hs['label_pairwise_difference_frames'], sort_keys=True)}",
        f"- Exact duplicate physical-content groups: {len(hs['duplicate_physical_content_groups'])}",
        f"- Same-probe/same-basename force-slip groups: {len(hs['cross_subset_same_probe_basename_groups'])}; exact shared tactile frames across those pairs: {sum(item['exact_shared_tactile_frames'] for item in hs['cross_subset_same_probe_basename_groups'])}",
        f"- Episodes with anomalies: {hs['episodes_with_anomalies']}",
        "- Image/reference and force extrema plus non-finite counts are in `htt_manifest.json`; force extrema are also recorded independently for each of the six stored component indices.",
        "- Evidence: dataset files call the signal aligned ATI 6D force-torque. Unknown: component order, sign/frame convention, force/torque units, timestamps, and sample rate are not documented in the supplied README/FORMAT or stored NPZ fields.",
        "- Primary slip supervision is `sliding_labels_bracket` (0 static, 1 incipient, 2 gross). The other two versions are retained and compared, never silently substituted.",
        "- All per-file labeling metadata is retained. The common algorithm parameters and the observed per-episode bracket start/stop ranges are summarized in the manifest.",
        "- Hash limit: absence of identical full sequences or identical frames does not prove that same-named files came from different physical trials; they may be different excerpts. Same-probe/same-basename pairs must remain conservatively bound during splitting.",
        "",
        "## NormalFlow",
        "",
        f"- Episodes: {normalflow['episode_count']} across {len(ns['object_episode_counts'])} objects",
        f"- Decoded tactile frames: {ns['total_decoded_tactile_frames']}",
        f"- Container/OpenCV average FPS range: {ns['fps_min']} to {ns['fps_max']}",
        f"- Episodes with anomalies: {ns['episodes_with_anomalies']}",
        "- All tactile videos are fully decoded; manifest entries compare decoded counts with ffprobe/container counts and each pose/mask/gradient array length.",
        "- Evidence: the dataset supplies motion-capture 6DoF pose and image-derived masks/gradients. It supplies no slip labels and no independent timestamp arrays.",
        "- Consequence: relative motion is not used as slip supervision. Timing is reported in steps; container FPS is retained only as nominal metadata.",
        "",
        "## Anomalies",
        "",
    ]
    if not htt_anomalies and not nf_anomalies:
        lines.append("No structural, alignment, decoding, or non-finite-value anomalies were detected.")
    else:
        for episode, issue in htt_anomalies + nf_anomalies:
            lines.append(f"- `{episode}`: {issue}")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `htt_manifest.json`: per-episode HTT audit and aggregate label/range/hash checks.",
            "- `normalflow_manifest.json`: per-episode video/array/timing/hash checks.",
            "- `audit_run.json`: invocation, Python/library versions, roots, and artifact hashes.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--htt-root", type=Path, required=True)
    parser.add_argument("--normalflow-zip", type=Path, required=True)
    parser.add_argument("--normalflow-extract-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffprobe")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ffprobe = find_ffprobe(args.ffprobe)
    extracted_root = safe_extract_zip(args.normalflow_zip, args.normalflow_extract_dir)
    htt = audit_htt(args.htt_root)
    normalflow = audit_normalflow(extracted_root, ffprobe)
    htt_path = args.output_dir / "htt_manifest.json"
    normalflow_path = args.output_dir / "normalflow_manifest.json"
    canonical_json_dump(htt, htt_path)
    canonical_json_dump(normalflow, normalflow_path)
    report_path = args.output_dir / "DATA_AUDIT.md"
    write_report(htt, normalflow, report_path)
    run = {
        "schema_version": SCHEMA_VERSION,
        "command": " ".join(sys.argv),
        "python": sys.version,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "ffprobe": ffprobe,
        "roots": {
            "htt": str(args.htt_root),
            "normalflow_zip": str(args.normalflow_zip),
            "normalflow_extracted": str(extracted_root),
        },
        "outputs": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in (htt_path, normalflow_path, report_path)
        },
    }
    canonical_json_dump(run, args.output_dir / "audit_run.json")
    print(f"Audit complete: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
