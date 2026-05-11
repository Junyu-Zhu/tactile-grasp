#!/usr/bin/env python3
"""Prepare GSmini force/slip Phase 1 artifacts without modifying raw data.

Outputs:
- data manifest with hashes for official dataset_gelsight_* files and org files
- derived trajectory-level train/val/test datasets with symlinked images and subset force/slip labels
- smoke test, preprocessing/domain stats, force axis/unit stats, slip alignment audit
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import random
import socket
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

SPARSH_REPO = Path("/home/zjy/document/sparsh")
if str(SPARSH_REPO) not in sys.path:
    sys.path.insert(0, str(SPARSH_REPO))

try:
    from omegaconf import OmegaConf
    from PIL import Image
    from tactile_ssl.data.vision_based_forces_slip_probes import VisionForceSlipDataset
    from tactile_ssl.data.digit.utils import load_sample_from_buf, get_resize_transform, load_bin_image
except Exception as exc:  # noqa: BLE001
    print(f"IMPORT_ERROR: {exc}", file=sys.stderr)
    raise

RAW_ROOT_DEFAULT = Path("/vla1/zjy/tactile_datasets/Gelsight-mini/gelsight-force-estimation")
OUT_BASE_DEFAULT = Path("/vla1/zjy/sparsh_runs/force_slip_phase1")
SOURCE_DATASETS = [
    "flat/batch_1",
    "flat/batch_2",
    "sharp/batch_1",
    "sharp/batch_2",
    "sphere/batch_1",
    "sphere/batch_2",
    "sphere/batch_3",
    "sphere/batch_4",
    "sphere/batch_5",
    "sphere/batch_6",
]


def json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def file_record(path: Path, root: Path, do_hash: bool = True) -> dict[str, Any]:
    st = path.stat()
    rec = {
        "path": str(path),
        "relative_path": str(path.relative_to(root)),
        "size": st.st_size,
        "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(),
    }
    if do_hash:
        rec["sha256"] = sha256_file(path)
    return rec


def load_force(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def load_frames_from_dataset_dir(path_data: Path) -> list[Any]:
    frames: list[Any] = []
    for p in sorted(path_data.glob("dataset_gelsight_*.pkl")):
        with p.open("rb") as f:
            frames.extend(pickle.load(f))
    return frames


def sample_counts(force_data: dict[str, Any]) -> dict[str, Any]:
    trajs = force_data["trajectories"]
    lengths = [len(t["indexes"]) for t in trajs.values()]
    eligible = [max(0, len(t["indexes"]) - 5) for t in trajs.values()]
    slips = []
    forces = []
    for t in trajs.values():
        if "slip_label" in t:
            slips.extend(np.asarray(t["slip_label"]).astype(int).tolist())
        if "forces" in t:
            forces.append(np.asarray(t["forces"], dtype=float))
    force_arr = np.concatenate(forces, axis=0) if forces else np.zeros((0, 3))
    slip_arr = np.asarray(slips, dtype=int) if slips else np.zeros((0,), dtype=int)
    return {
        "trajectories": len(trajs),
        "raw_samples": int(sum(lengths)),
        "eligible_dataset_samples_len_minus_5": int(sum(eligible)),
        "trajectory_len_min": int(np.min(lengths)) if lengths else 0,
        "trajectory_len_max": int(np.max(lengths)) if lengths else 0,
        "trajectory_len_mean": float(np.mean(lengths)) if lengths else 0.0,
        "slip_positive_frames": int(slip_arr.sum()) if slip_arr.size else 0,
        "slip_total_frames": int(slip_arr.size),
        "slip_positive_ratio": float(slip_arr.mean()) if slip_arr.size else 0.0,
        "force_min": force_arr.min(axis=0).tolist() if len(force_arr) else [],
        "force_max": force_arr.max(axis=0).tolist() if len(force_arr) else [],
        "force_mean": force_arr.mean(axis=0).tolist() if len(force_arr) else [],
        "force_std": force_arr.std(axis=0).tolist() if len(force_arr) else [],
    }


def deterministic_split(keys: list[Any], seed_text: str, ratios=(0.70, 0.15, 0.15)) -> dict[str, list[Any]]:
    keys = list(keys)
    rng = random.Random(hashlib.sha256(seed_text.encode("utf-8")).hexdigest())
    rng.shuffle(keys)
    n = len(keys)
    if n == 0:
        return {"train": [], "val": [], "test": []}
    n_train = max(1, int(round(n * ratios[0])))
    n_val = max(1, int(round(n * ratios[1]))) if n >= 3 else 0
    if n_train + n_val >= n and n >= 3:
        n_train = n - 2
        n_val = 1
    n_test = n - n_train - n_val
    if n_test == 0 and n >= 3:
        n_test = 1
        n_train = max(1, n_train - 1)
    return {
        "train": keys[:n_train],
        "val": keys[n_train:n_train + n_val],
        "test": keys[n_train + n_val:],
    }


def safe_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def subset_force_data(force_data: dict[str, Any], keys: list[Any]) -> dict[str, Any]:
    out = {k: copy.deepcopy(v) for k, v in force_data.items() if k != "trajectories"}
    out_traj = {}
    for new_i, key in enumerate(keys):
        out_traj[new_i] = copy.deepcopy(force_data["trajectories"][key])
        out_traj[new_i]["source_trajectory_id"] = key
    out["trajectories"] = out_traj
    return out


def make_derived_dataset(raw_root: Path, derived_root: Path, source: str, split_name: str, keys: list[Any], force_data: dict[str, Any]) -> str | None:
    if not keys:
        return None
    safe_name = source.replace("/", "_") + f"_{split_name}"
    ds_dir = derived_root / safe_name
    ds_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = raw_root / source
    for old in ds_dir.glob("dataset_gelsight_*.pkl"):
        old.unlink()
    for p in sorted(raw_dir.glob("dataset_gelsight_*.pkl")):
        safe_symlink(p, ds_dir / p.name)
    with (ds_dir / "dataset_slip_forces.pkl").open("wb") as f:
        pickle.dump(subset_force_data(force_data, keys), f, protocol=pickle.HIGHEST_PROTOCOL)
    source_record = {
        "source_dataset": source,
        "split": split_name,
        "source_trajectory_ids": [str(k) for k in keys],
        "reindexed_trajectories": True,
        "image_pkls_symlinked_from_raw": True,
    }
    write_json(ds_dir / "source_manifest.json", source_record)
    return safe_name


def force_axis_stats(force_datas: dict[str, dict[str, Any]]) -> dict[str, Any]:
    per_source = {}
    all_forces = []
    for source, force_data in force_datas.items():
        arrs = [np.asarray(t["forces"], dtype=float) for t in force_data["trajectories"].values()]
        arr = np.concatenate(arrs, axis=0) if arrs else np.zeros((0, 3))
        all_forces.append(arr)
        if len(arr):
            fn = np.abs(arr[:, 2])
            ft = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2)
            per_source[source] = {
                "count": len(arr),
                "axis_mean": arr.mean(axis=0).tolist(),
                "axis_std": arr.std(axis=0).tolist(),
                "axis_min": arr.min(axis=0).tolist(),
                "axis_max": arr.max(axis=0).tolist(),
                "median_abs_axis": np.median(np.abs(arr), axis=0).tolist(),
                "fn_abs_fz_mean": float(fn.mean()),
                "ft_xy_mean": float(ft.mean()),
                "ft_over_fn_p50": float(np.median(ft / (fn + 1e-8))),
                "ft_over_fn_p80": float(np.percentile(ft / (fn + 1e-8), 80)),
            }
    all_arr = np.concatenate([a for a in all_forces if len(a)], axis=0)
    med_abs = np.median(np.abs(all_arr), axis=0).tolist() if len(all_arr) else []
    normal_axis = int(np.argmax(med_abs)) if med_abs else None
    return {
        "axis_order": ["Fx", "Fy", "Fz"],
        "overall_count": int(len(all_arr)),
        "overall_median_abs_axis": med_abs,
        "inferred_normal_axis": "Fz" if normal_axis == 2 else ["Fx", "Fy", "Fz"][normal_axis] if normal_axis is not None else None,
        "normal_axis_confidence": "medium" if normal_axis == 2 else "low",
        "force_newton_assumption": "dataset trajectories['forces'] are treated as Newton labels; training sample force is normalized by force_scale and metrics must de-normalize with force_scale",
        "mapping_if_Fz_normal": {"Fn": "abs(Fz_N)", "Ft": "sqrt(Fx_N^2 + Fy_N^2)", "Fmag": "sqrt(Fx_N^2 + Fy_N^2 + Fz_N^2)"},
        "per_source": per_source,
    }


def preprocess_stats(raw_root: Path, sources: list[str], out_dir: Path, max_frames_per_kind: int = 16) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    img_out = out_dir / "preprocess_examples"
    img_out.mkdir(parents=True, exist_ok=True)
    transform = get_resize_transform([320, 240])
    for source in sources[:]:
        raw_dir = raw_root / source
        force_data = load_force(raw_dir / "dataset_slip_forces.pkl")
        official_frames = load_frames_from_dataset_dir(raw_dir)
        in_contact = np.asarray(force_data["in_contact"])
        bg = None
        if len(official_frames) and np.any(in_contact == 0):
            bg_idx = int(np.where(in_contact == 0)[0][0])
            if bg_idx < len(official_frames):
                bg = load_bin_image(official_frames[bg_idx])
        for kind, pattern in [("dataset_gelsight", "dataset_gelsight_*.pkl"), ("org_dataset_gelsight", "org_dataset_gelsight_*.pkl")]:
            files = sorted(raw_dir.glob(pattern))
            tensors = []
            frame_seen = 0
            for p in files:
                with p.open("rb") as f:
                    frames = pickle.load(f)
                for frame in frames[:max_frames_per_kind - frame_seen]:
                    try:
                        image = load_sample_from_buf(frame, bg)
                        t = transform(image)
                        tensors.append(t.numpy())
                        if frame_seen == 0:
                            arr = (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                            Image.fromarray(arr).save(img_out / f"{source.replace('/', '_')}_{kind}.png")
                        frame_seen += 1
                    except Exception as exc:  # noqa: BLE001
                        stats.setdefault(source, {}).setdefault(kind, {})["example_error"] = str(exc)
                    if frame_seen >= max_frames_per_kind:
                        break
                if frame_seen >= max_frames_per_kind:
                    break
            if tensors:
                arr = np.stack(tensors, axis=0)
                stats.setdefault(source, {})[kind] = {
                    "files": [str(p) for p in files],
                    "sampled_frames": int(arr.shape[0]),
                    "shape": list(arr.shape[1:]),
                    "dtype": str(arr.dtype),
                    "range_min": float(arr.min()),
                    "range_max": float(arr.max()),
                    "per_channel_mean": arr.mean(axis=(0, 2, 3)).tolist(),
                    "per_channel_std": arr.std(axis=(0, 2, 3)).tolist(),
                    "quantiles": np.quantile(arr, [0.01, 0.05, 0.5, 0.95, 0.99]).tolist(),
                    "preprocess_chain": "load_sample_from_buf(io_buf, bg) then get_resize_transform([320,240])",
                }
    return stats


def slip_alignment_audit(force_datas: dict[str, dict[str, Any]], frame_counts: dict[str, int], split_manifest: dict[str, Any]) -> dict[str, Any]:
    issues = []
    per_source = {}
    for source, force_data in force_datas.items():
        src_issues = []
        total = pos = eligible = 0
        frame_count = frame_counts[source]
        for key, traj in force_data["trajectories"].items():
            indexes = np.asarray(traj.get("indexes", []), dtype=int)
            forces = np.asarray(traj.get("forces", []))
            slip = np.asarray(traj.get("slip_label", []))
            n = len(indexes)
            # Some GSmini flat/sharp force files carry five extra frame indexes for temporal context.
            # The official dataset maps samples over indexes[5:], so len(indexes) == len(labels)+5 is acceptable.
            length_ok = (len(forces) == len(slip) == n) or (len(forces) == len(slip) and n == len(forces) + 5)
            if not length_ok:
                src_issues.append({"trajectory": str(key), "kind": "length_mismatch", "indexes": len(indexes), "forces": len(forces), "slip_label": len(slip)})
            if len(indexes) and (indexes.min() < 0 or indexes.max() >= frame_count):
                src_issues.append({"trajectory": str(key), "kind": "frame_index_out_of_bounds", "min": int(indexes.min()), "max": int(indexes.max()), "frame_count": frame_count})
            total += len(slip)
            pos += int(slip.astype(int).sum()) if len(slip) else 0
            eligible += max(0, n - 5)
        per_source[source] = {
            "frame_count": frame_count,
            "trajectory_count": len(force_data["trajectories"]),
            "slip_positive_frames": pos,
            "slip_total_frames": total,
            "slip_positive_ratio": float(pos / total) if total else 0.0,
            "eligible_dataset_samples_len_minus_5": eligible,
            "issues": src_issues[:20],
            "issue_count": len(src_issues),
        }
        issues.extend({"source": source, **issue} for issue in src_issues)
    split_overlap = []
    for source, by_split in split_manifest["source_splits"].items():
        sets = {k: set(v) for k, v in by_split.items()}
        for i, a in enumerate(sets):
            for b in list(sets)[i+1:]:
                overlap = sets[a] & sets[b]
                if overlap:
                    split_overlap.append({"source": source, "splits": [a, b], "overlap": sorted(map(str, overlap))[:20], "count": len(overlap)})
    return {
        "passed": len(issues) == 0 and len(split_overlap) == 0,
        "issue_count": len(issues),
        "issues_sample": issues[:50],
        "split_overlap": split_overlap,
        "per_source": per_source,
        "note": "Audit checks length equality, frame index bounds, and trajectory split overlap. It does not prove semantic slip_horizon optimality; slip_horizon remains train/val-only.",
    }


def smoke_dataset(derived_root: Path, dataset_names: list[str], sensor="gelsight") -> dict[str, Any]:
    cfg = OmegaConf.create({
        "sensor": sensor,
        "remove_bg": True,
        "out_format": "concat_ch_img",
        "num_frames": 2,
        "frame_stride": 5,
        "path_dataset": str(derived_root),
        "look_in_folder": False,
        "slip_horizon": 0,
        "max_abs_forceXYZ": [1.5, 1.5, 2.0],
        "max_delta_forceXYZ": [0.80, 0.80, 0.80],
        "transforms": {"resize": [320, 240]},
    })
    out = {"datasets_checked": [], "samples_checked": []}
    for name in dataset_names[: min(4, len(dataset_names))]:
        ds = VisionForceSlipDataset(cfg, name)
        out["datasets_checked"].append({"dataset_name": name, "len": len(ds), "traj_count": len(ds.traj2idx)})
        sample = ds[0]
        out["samples_checked"].append({
            "dataset_name": name,
            "image_shape": list(sample["image"].shape),
            "force_shape": list(sample["force"].shape),
            "delta_force_shape": list(sample["delta_force"].shape),
            "slip_label_shape": list(sample["slip_label"].shape),
            "slip_label_value": int(sample["slip_label"].item()),
            "force_scale": sample["force_scale"].tolist(),
            "delta_force_scale": sample["delta_force_scale"].tolist(),
            "image_dtype": str(sample["image"].dtype),
            "image_min": float(sample["image"].min()),
            "image_max": float(sample["image"].max()),
        })
    out["passed"] = all(s["image_shape"] == [6, 320, 240] and s["force_shape"] == [3] and s["delta_force_shape"] == [3] for s in out["samples_checked"])
    return out


def render_md(context: dict[str, Any], manifest: dict[str, Any], split_summary: dict[str, Any], axis: dict[str, Any], align: dict[str, Any], smoke: dict[str, Any]) -> str:
    return f"""# Force-Slip Phase 1 GSmini Preparation Report

Generated: {context['generated_at']} on {context['host']}

## Paths

- Raw dataset root: `{context['raw_root']}`
- Output root: `{context['output_root']}`
- Derived dataset root: `{context['derived_root']}`
- tactile-grasp repo: `/home/zjy/document/tactile-grasp`
- Sparsh repo: `/home/zjy/document/sparsh`

## Manifest

- Source datasets: {len(manifest['datasets'])}
- Official training files use `dataset_gelsight_*.pkl` plus `dataset_slip_forces.pkl`.
- `org_dataset_gelsight_*.pkl` is recorded for domain diagnostics only and is not included in derived training directories.

## Split

- Train derived datasets: {len(split_summary['train_datasets'])}
- Val derived datasets: {len(split_summary['val_datasets'])}
- Test derived datasets: {len(split_summary['test_datasets'])}
- Split overlap issues: {len(align['split_overlap'])}

## Smoke Test

- Passed: `{smoke['passed']}`
- Checked samples: {len(smoke['samples_checked'])}

## Force Axis / Unit

- Inferred normal axis: `{axis['inferred_normal_axis']}`
- Confidence: `{axis['normal_axis_confidence']}`
- Newton mapping if Fz normal: `{axis['mapping_if_Fz_normal']}`

## Slip Alignment Audit

- Passed: `{align['passed']}`
- Issue count: {align['issue_count']}

## Training Lists

Force task train/val/test lists are stored in `phase1_context.json` as `force_*_datasets`.
Slip task train/val/test lists are stored as `slip_*_datasets` and intentionally use `sphere_*` derived datasets.
Test lists are reserved for final evaluation, not hyperparameter tuning.

"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT_DEFAULT)
    parser.add_argument("--out-base", type=Path, default=OUT_BASE_DEFAULT)
    parser.add_argument("--run-id", default=datetime.now().strftime("phase1_gsmini_%Y%m%d_%H%M%S"))
    parser.add_argument("--sources", nargs="*", default=SOURCE_DATASETS)
    parser.add_argument("--seed", type=int, default=20260512)
    args = parser.parse_args()

    raw_root = args.raw_root.resolve()
    output_root = (args.out_base / args.run_id).resolve()
    derived_root = output_root / "derived_gsmini"
    reports_dir = output_root / "reports"
    derived_root.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    context = {
        "run_id": args.run_id,
        "generated_at": datetime.now().isoformat(),
        "host": socket.gethostname(),
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "derived_root": str(derived_root),
        "seed": args.seed,
        "sources": args.sources,
    }

    manifest = {"datasets": {}, "raw_root": str(raw_root), "hash_algorithm": "sha256"}
    force_datas: dict[str, dict[str, Any]] = {}
    frame_counts: dict[str, int] = {}

    for source in args.sources:
        raw_dir = raw_root / source
        if not raw_dir.exists():
            manifest["datasets"][source] = {"exists": False}
            continue
        force_path = raw_dir / "dataset_slip_forces.pkl"
        force_data = load_force(force_path)
        force_datas[source] = force_data
        official_files = sorted(raw_dir.glob("dataset_gelsight_*.pkl"))
        org_files = sorted(raw_dir.glob("org_dataset_gelsight_*.pkl"))
        frame_count = 0
        frame_file_counts = []
        for p in official_files:
            with p.open("rb") as f:
                frames = pickle.load(f)
            frame_count += len(frames)
            frame_file_counts.append({"file": p.name, "frames": len(frames)})
        frame_counts[source] = frame_count
        manifest["datasets"][source] = {
            "exists": True,
            "official_dataset_gelsight_files": [file_record(p, raw_root) for p in official_files],
            "org_dataset_gelsight_files_recorded_not_used_for_training": [file_record(p, raw_root) for p in org_files],
            "force_slip_file": file_record(force_path, raw_root),
            "frame_count_from_official_files": frame_count,
            "frame_file_counts": frame_file_counts,
            "summary": sample_counts(force_data),
        }

    write_json(reports_dir / "data_manifest.json", manifest)

    split_manifest: dict[str, Any] = {"seed": args.seed, "source_splits": {}, "derived_datasets": defaultdict(dict)}
    train_names: list[str] = []
    val_names: list[str] = []
    test_names: list[str] = []
    split_stats: dict[str, Any] = {"by_derived_dataset": {}}

    for source, force_data in force_datas.items():
        keys = list(force_data["trajectories"].keys())
        splits = deterministic_split(keys, f"{args.seed}:{source}")
        split_manifest["source_splits"][source] = {k: [str(x) for x in v] for k, v in splits.items()}
        for split_name, split_keys in splits.items():
            ds_name = make_derived_dataset(raw_root, derived_root, source, split_name, split_keys, force_data)
            if ds_name is None:
                continue
            split_manifest["derived_datasets"][source][split_name] = ds_name
            subset_data = subset_force_data(force_data, split_keys)
            split_stats["by_derived_dataset"][ds_name] = sample_counts(subset_data)
            if split_name == "train":
                train_names.append(ds_name)
            elif split_name == "val":
                val_names.append(ds_name)
            elif split_name == "test":
                test_names.append(ds_name)

    split_manifest["derived_datasets"] = dict(split_manifest["derived_datasets"])
    split_summary = {
        "derived_root": str(derived_root),
        "train_datasets": train_names,
        "val_datasets": val_names,
        "test_datasets": test_names,
        **split_stats,
    }
    write_json(reports_dir / "trajectory_split_manifest.json", split_manifest)
    write_json(reports_dir / "split_summary.json", split_summary)

    axis = force_axis_stats(force_datas)
    write_json(reports_dir / "axis_unit_report.json", axis)

    preprocess = preprocess_stats(raw_root, list(force_datas.keys()), reports_dir)
    write_json(reports_dir / "preprocess_domain_report.json", preprocess)

    alignment = slip_alignment_audit(force_datas, frame_counts, split_manifest)
    write_json(reports_dir / "slip_alignment_audit.json", alignment)

    smoke_names = train_names[:2] + val_names[:2]
    smoke = smoke_dataset(derived_root, smoke_names)
    write_json(reports_dir / "dataloader_smoke_test.json", smoke)

    slip_train_names = [name for name in train_names if name.startswith("sphere_")]
    slip_val_names = [name for name in val_names if name.startswith("sphere_")]
    slip_test_names = [name for name in test_names if name.startswith("sphere_")]
    phase_context = {
        **context,
        "train_datasets": train_names,
        "val_datasets": val_names,
        "test_datasets": test_names,
        "force_train_datasets": train_names,
        "force_val_datasets": val_names,
        "force_test_datasets": test_names,
        "slip_train_datasets": slip_train_names,
        "slip_val_datasets": slip_val_names,
        "slip_test_datasets": slip_test_names,
        "slip_dataset_policy": "Use sphere_* derived datasets for slip downstream tasks, matching prior Sparsh slip setup and avoiding flat/sharp temporal padding ambiguity.",
        "all_derived_datasets": train_names + val_names + test_names,
        "reports": {
            "data_manifest": str(reports_dir / "data_manifest.json"),
            "trajectory_split_manifest": str(reports_dir / "trajectory_split_manifest.json"),
            "split_summary": str(reports_dir / "split_summary.json"),
            "axis_unit_report": str(reports_dir / "axis_unit_report.json"),
            "preprocess_domain_report": str(reports_dir / "preprocess_domain_report.json"),
            "slip_alignment_audit": str(reports_dir / "slip_alignment_audit.json"),
            "dataloader_smoke_test": str(reports_dir / "dataloader_smoke_test.json"),
        },
    }
    write_json(output_root / "phase1_context.json", phase_context)
    (reports_dir / "phase1_preparation_report.md").write_text(render_md(context, manifest, split_summary, axis, alignment, smoke), encoding="utf-8")

    print(json.dumps(phase_context, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
