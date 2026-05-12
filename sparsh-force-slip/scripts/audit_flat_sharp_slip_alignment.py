#!/usr/bin/env python3
"""Audit whether flat/sharp GSmini data can be safely included in slip training.

This script is read-only with respect to the raw dataset. It checks the official
Sparsh sample-index semantics, derived split consistency, slip class balance,
and produces force/slip/image examples around slip onsets.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SPARSH_REPO = Path("/home/zjy/document/sparsh")
if str(SPARSH_REPO) not in sys.path:
    sys.path.insert(0, str(SPARSH_REPO))

from omegaconf import OmegaConf  # noqa: E402
from tactile_ssl.data.digit.utils import get_resize_transform, load_bin_image, load_sample_from_buf  # noqa: E402
from tactile_ssl.data.vision_based_forces_slip_probes import VisionForceSlipDataset  # noqa: E402

RAW_ROOT = Path("/vla1/zjy/tactile_datasets/Gelsight-mini/gelsight-force-estimation")
RUN_ID_FILE = Path("/home/zjy/document/tactile-grasp/sparsh-force-slip/phase1_run_id.txt")
REPORT_ROOT = Path("/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase1")
DERIVED_BASE = Path("/vla1/zjy/sparsh_runs/force_slip_phase1")

SOURCES = [
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
PRIMARY_AUDIT_SOURCES = ["flat/batch_1", "flat/batch_2", "sharp/batch_1", "sharp/batch_2"]


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def load_force(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def load_frames(raw_dir: Path) -> list[Any]:
    frames: list[Any] = []
    for p in sorted(raw_dir.glob("dataset_gelsight_*.pkl")):
        with p.open("rb") as f:
            frames.extend(pickle.load(f))
    return frames


def official_len(indexes: list[Any]) -> int:
    # This is exactly what VisionForceSlipDataset.get_map_idx2traj does:
    # t_idxs = indexes[5:], then sample ranges over len(t_idxs).
    return max(0, len(indexes) - 5)


def source_to_derived_prefix(source: str) -> str:
    return source.replace("/", "_")


def array_equal_or_close(a: Any, b: Any) -> bool:
    aa = np.asarray(a)
    bb = np.asarray(b)
    if aa.shape != bb.shape:
        return False
    if np.issubdtype(aa.dtype, np.number) and np.issubdtype(bb.dtype, np.number):
        return bool(np.allclose(aa, bb, equal_nan=True))
    return bool(np.array_equal(aa, bb))


def audit_source_structure(raw_root: Path, source: str) -> dict[str, Any]:
    raw_dir = raw_root / source
    force_data = load_force(raw_dir / "dataset_slip_forces.pkl")
    frame_count = 0
    frame_file_counts = []
    for p in sorted(raw_dir.glob("dataset_gelsight_*.pkl")):
        with p.open("rb") as f:
            n = len(pickle.load(f))
        frame_count += n
        frame_file_counts.append({"file": p.name, "frames": n})

    length_relations: dict[str, int] = defaultdict(int)
    issues = []
    used_labels = []
    full_labels = []
    delta_shear_pos = []
    delta_shear_neg = []
    delta_normal_pos = []
    delta_normal_neg = []
    transition_count = 0
    official_sample_count = 0
    label_count = 0
    raw_index_count = 0
    per_traj_samples = []

    for traj_id, traj in force_data["trajectories"].items():
        indexes = list(traj.get("indexes", []))
        labels = np.asarray(traj.get("slip_label", []), dtype=int)
        forces = np.asarray(traj.get("forces", []))
        n_idx, n_lab, n_forces = len(indexes), len(labels), len(forces)
        raw_index_count += n_idx
        label_count += n_lab
        full_labels.extend(labels.tolist())

        if n_idx == n_lab:
            relation = "len(indexes)==len(labels)"
        elif n_idx == n_lab + 5:
            relation = "len(indexes)==len(labels)+5"
        else:
            relation = f"other:indexes={n_idx},labels={n_lab}"
            issues.append({"trajectory": str(traj_id), "kind": "unexpected_index_label_length", "indexes": n_idx, "labels": n_lab})
        length_relations[relation] += 1

        if n_forces != n_lab:
            issues.append({"trajectory": str(traj_id), "kind": "force_label_length_mismatch", "forces": n_forces, "labels": n_lab})

        eff_len = official_len(indexes)
        official_sample_count += eff_len
        per_traj_samples.append(eff_len)
        if eff_len > n_lab:
            issues.append({"trajectory": str(traj_id), "kind": "official_len_exceeds_label_len", "official_len": eff_len, "labels": n_lab})
            eff_len = n_lab
        if indexes:
            arr_idx = np.asarray(indexes[:eff_len] if eff_len else indexes, dtype=int)
            if arr_idx.size and (arr_idx.min() < 0 or arr_idx.max() >= frame_count):
                issues.append({
                    "trajectory": str(traj_id),
                    "kind": "official_frame_index_out_of_bounds",
                    "min": int(arr_idx.min()),
                    "max": int(arr_idx.max()),
                    "frame_count": frame_count,
                })
        labels_eff = labels[:eff_len]
        used_labels.extend(labels_eff.tolist())
        if len(labels_eff) > 1:
            transition_count += int(np.sum((labels_eff[:-1] == 0) & (labels_eff[1:] == 1)))
        # Match VisionForceSlipDataset._get_force_slip_labels exactly for the
        # default two-frame setup: delta_force[s] = forces[s] - forces[max(s-5, 0)].
        if eff_len and len(forces) >= eff_len:
            prev = np.maximum(np.arange(eff_len) - 5, 0)
            delta_force_official = forces[:eff_len] - forces[prev]
            shear = np.linalg.norm(delta_force_official[:, :2], axis=1)
            normal = np.abs(delta_force_official[:, 2])
        else:
            shear = np.asarray([], dtype=float)
            normal = np.asarray([], dtype=float)
        if len(shear) == len(labels_eff):
            delta_shear_pos.extend(shear[labels_eff == 1].tolist())
            delta_shear_neg.extend(shear[labels_eff == 0].tolist())
        if len(normal) == len(labels_eff):
            delta_normal_pos.extend(normal[labels_eff == 1].tolist())
            delta_normal_neg.extend(normal[labels_eff == 0].tolist())

    used = np.asarray(used_labels, dtype=int)
    full = np.asarray(full_labels, dtype=int)

    def mean_or_none(xs: list[float]) -> float | None:
        return float(np.mean(xs)) if xs else None

    shear_pos_mean = mean_or_none(delta_shear_pos)
    shear_neg_mean = mean_or_none(delta_shear_neg)
    normal_pos_mean = mean_or_none(delta_normal_pos)
    normal_neg_mean = mean_or_none(delta_normal_neg)
    return {
        "source": source,
        "frame_count": frame_count,
        "frame_file_counts": frame_file_counts,
        "trajectory_count": len(force_data["trajectories"]),
        "raw_index_count": raw_index_count,
        "label_count": label_count,
        "official_sample_count": official_sample_count,
        "official_sample_min_per_traj": int(np.min(per_traj_samples)) if per_traj_samples else 0,
        "official_sample_max_per_traj": int(np.max(per_traj_samples)) if per_traj_samples else 0,
        "length_relations": dict(length_relations),
        "full_label_positive": int(full.sum()) if full.size else 0,
        "full_label_total": int(full.size),
        "full_label_positive_ratio": float(full.mean()) if full.size else 0.0,
        "official_label_positive": int(used.sum()) if used.size else 0,
        "official_label_total": int(used.size),
        "official_label_positive_ratio": float(used.mean()) if used.size else 0.0,
        "official_zero_to_one_transition_count": transition_count,
        "delta_mag_shear_mean_positive": shear_pos_mean,
        "delta_mag_shear_mean_negative": shear_neg_mean,
        "delta_mag_shear_pos_over_neg": float(shear_pos_mean / (shear_neg_mean + 1e-12)) if shear_pos_mean is not None and shear_neg_mean not in (None, 0.0) else None,
        "delta_mag_normal_mean_positive": normal_pos_mean,
        "delta_mag_normal_mean_negative": normal_neg_mean,
        "delta_mag_normal_pos_over_neg": float(normal_pos_mean / (normal_neg_mean + 1e-12)) if normal_pos_mean is not None and normal_neg_mean not in (None, 0.0) else None,
        "issues": issues[:50],
        "issue_count": len(issues),
    }


def audit_derived_consistency(raw_root: Path, derived_root: Path, source: str) -> dict[str, Any]:
    raw_force = load_force(raw_root / source / "dataset_slip_forces.pkl")
    raw_trajs = raw_force["trajectories"]
    prefix = source_to_derived_prefix(source)
    out: dict[str, Any] = {"source": source, "splits": {}, "issues": []}
    seen_source_ids: set[str] = set()
    for split in ["train", "val", "test"]:
        ds_name = f"{prefix}_{split}"
        ds_dir = derived_root / ds_name
        if not ds_dir.exists():
            out["issues"].append({"kind": "missing_derived_dir", "dataset": ds_name})
            continue
        data = load_force(ds_dir / "dataset_slip_forces.pkl")
        manifest = json.loads((ds_dir / "source_manifest.json").read_text())
        split_source_ids = [str(x) for x in manifest.get("source_trajectory_ids", [])]
        seen_source_ids.update(split_source_ids)
        mismatch = []
        for new_id, traj in data["trajectories"].items():
            src_id = str(traj.get("source_trajectory_id"))
            raw_key: Any = src_id
            if raw_key not in raw_trajs:
                try:
                    raw_key = int(src_id)
                except ValueError:
                    pass
            if raw_key not in raw_trajs:
                mismatch.append({"new_id": str(new_id), "source_trajectory_id": src_id, "kind": "missing_raw_source_traj"})
                continue
            raw_t = raw_trajs[raw_key]
            for field in ["indexes", "forces", "delta_forces", "slip_label", "delta_mag_shear", "delta_mag_normal"]:
                if field in raw_t and field in traj and not array_equal_or_close(raw_t[field], traj[field]):
                    mismatch.append({"new_id": str(new_id), "source_trajectory_id": src_id, "kind": "field_mismatch", "field": field})
        labels = []
        for traj in data["trajectories"].values():
            eff = min(official_len(list(traj.get("indexes", []))), len(traj.get("slip_label", [])))
            labels.extend(np.asarray(traj.get("slip_label", []), dtype=int)[:eff].tolist())
        arr = np.asarray(labels, dtype=int)
        out["splits"][split] = {
            "dataset_name": ds_name,
            "trajectory_count": len(data["trajectories"]),
            "source_manifest_trajectory_count": len(split_source_ids),
            "official_label_total": int(arr.size),
            "official_label_positive": int(arr.sum()) if arr.size else 0,
            "official_label_positive_ratio": float(arr.mean()) if arr.size else 0.0,
            "field_mismatch_count": len(mismatch),
            "field_mismatch_sample": mismatch[:20],
        }
        if mismatch:
            out["issues"].extend({"split": split, **m} for m in mismatch[:20])
    raw_ids = {str(k) for k in raw_trajs.keys()}
    missing = sorted(raw_ids - seen_source_ids)
    out["raw_trajectory_count"] = len(raw_ids)
    out["seen_source_trajectory_count"] = len(seen_source_ids)
    out["missing_source_trajectory_count"] = len(missing)
    out["missing_source_trajectory_sample"] = missing[:20]
    if missing:
        out["issues"].append({"kind": "source_trajectory_not_in_any_split", "count": len(missing), "sample": missing[:20]})
    out["passed"] = len(out["issues"]) == 0
    return out


def official_dataset_smoke(derived_root: Path, dataset_names: list[str]) -> dict[str, Any]:
    cfg = OmegaConf.create({
        "sensor": "gelsight",
        "remove_bg": True,
        "out_format": "concat_ch_img",
        "num_frames": 2,
        "frame_stride": 5,
        "path_dataset": str(derived_root),
        "look_in_folder": False,
        "slip_horizon": 0,
        "max_abs_forceXYZ": [1.5, 1.5, 2.0],
        "max_delta_forceXYZ": [0.80, 0.80, 0.40],
        "transforms": {"resize": [320, 240]},
    })
    out = {"datasets_checked": [], "issues": []}
    for name in dataset_names:
        try:
            ds = VisionForceSlipDataset(cfg, name)
            sample = ds[0]
            labels = getattr(ds, "slip_labels", np.asarray([], dtype=int))
            out["datasets_checked"].append({
                "dataset_name": name,
                "len": len(ds),
                "trajectory_count": len(ds.traj2idx),
                "slip_positive": int(labels.sum()) if len(labels) else 0,
                "slip_total": int(len(labels)),
                "slip_positive_ratio": float(labels.mean()) if len(labels) else 0.0,
                "sample_image_shape": list(sample["image"].shape),
                "sample_force_shape": list(sample["force"].shape),
                "sample_delta_force_shape": list(sample["delta_force"].shape),
                "sample_slip_label": int(sample["slip_label"].item()),
            })
        except Exception as exc:  # noqa: BLE001
            out["issues"].append({"dataset_name": name, "error": repr(exc)})
    out["passed"] = len(out["issues"]) == 0
    return out


def aggregate_split_balance(derived_root: Path, dataset_names: list[str]) -> dict[str, Any]:
    labels = []
    per_dataset = []
    for name in dataset_names:
        p = derived_root / name / "dataset_slip_forces.pkl"
        data = load_force(p)
        ds_labels = []
        for traj in data["trajectories"].values():
            eff = min(official_len(list(traj.get("indexes", []))), len(traj.get("slip_label", [])))
            ds_labels.extend(np.asarray(traj.get("slip_label", []), dtype=int)[:eff].tolist())
        arr = np.asarray(ds_labels, dtype=int)
        labels.extend(ds_labels)
        per_dataset.append({
            "dataset_name": name,
            "official_label_total": int(arr.size),
            "official_label_positive": int(arr.sum()) if arr.size else 0,
            "official_label_positive_ratio": float(arr.mean()) if arr.size else 0.0,
        })
    arr_all = np.asarray(labels, dtype=int)
    return {
        "dataset_count": len(dataset_names),
        "official_label_total": int(arr_all.size),
        "official_label_positive": int(arr_all.sum()) if arr_all.size else 0,
        "official_label_positive_ratio": float(arr_all.mean()) if arr_all.size else 0.0,
        "per_dataset": per_dataset,
    }


def find_example_event(force_data: dict[str, Any]) -> tuple[Any, int] | None:
    best = None
    best_len = -1
    for traj_id, traj in force_data["trajectories"].items():
        labels = np.asarray(traj.get("slip_label", []), dtype=int)
        eff = min(official_len(list(traj.get("indexes", []))), len(labels))
        labels = labels[:eff]
        if len(labels) < 2:
            continue
        onsets = np.where((labels[:-1] == 0) & (labels[1:] == 1))[0] + 1
        if len(onsets):
            sample = int(onsets[np.argmin(np.abs(onsets - len(labels) // 2))])
            return traj_id, sample
        positives = np.where(labels == 1)[0]
        if len(positives) and len(labels) > best_len:
            best = (traj_id, int(positives[len(positives) // 2]))
            best_len = len(labels)
    return best


def render_event_plot(raw_root: Path, source: str, out_dir: Path) -> dict[str, Any] | None:
    raw_dir = raw_root / source
    force_data = load_force(raw_dir / "dataset_slip_forces.pkl")
    event = find_example_event(force_data)
    if event is None:
        return None
    traj_id, sample = event
    traj = force_data["trajectories"][traj_id]
    indexes = list(traj["indexes"])
    labels = np.asarray(traj["slip_label"], dtype=int)
    forces = np.asarray(traj["forces"], dtype=float)
    eff = min(official_len(indexes), len(labels), len(forces))
    if eff:
        prev = np.maximum(np.arange(eff) - 5, 0)
        delta_force_official = forces[:eff] - forces[prev]
        delta_shear = np.linalg.norm(delta_force_official[:, :2], axis=1)
    else:
        delta_shear = np.asarray([], dtype=float)
    sample = int(np.clip(sample, 0, eff - 1))
    lo = max(0, sample - 25)
    hi = min(eff, sample + 26)
    xs = np.arange(lo, hi)

    frames = load_frames(raw_dir)
    bg = None
    in_contact = np.asarray(force_data.get("in_contact", []))
    if len(in_contact):
        no_contact = np.where(in_contact == 0)[0]
        if len(no_contact) and int(no_contact[0]) < len(frames):
            bg = load_bin_image(frames[int(no_contact[0])])
    transform = get_resize_transform([320, 240])

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 4, height_ratios=[2.1, 1.2])
    ax = fig.add_subplot(gs[0, :])
    ax.plot(xs, forces[lo:hi, 0], label="Fx")
    ax.plot(xs, forces[lo:hi, 1], label="Fy")
    ax.plot(xs, forces[lo:hi, 2], label="Fz")
    ax2 = ax.twinx()
    ax2.step(xs, labels[lo:hi], where="post", color="black", alpha=0.6, label="slip_label")
    ax2.plot(xs, delta_shear[lo:hi], color="magenta", alpha=0.55, label="official_delta_shear")
    ax.axvline(sample, color="red", linestyle="--", label="chosen sample")
    ax.set_title(f"{source} trajectory={traj_id} sample={sample} official-aligned force/slip window")
    ax.set_xlabel("official sample index")
    ax.set_ylabel("force (dataset units, treated as N)")
    ax2.set_ylabel("slip / delta shear")
    ax.grid(True, alpha=0.25)
    lines, labels_ = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels_ + labels2, loc="upper right")

    sample_points = [max(0, sample - 5), sample, min(eff - 1, sample + 5)]
    for col, s in enumerate(sample_points):
        ax_img = fig.add_subplot(gs[1, col])
        frame_idx = indexes[s]
        image = load_sample_from_buf(frames[frame_idx], bg)
        tensor = transform(image)
        arr = (tensor.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        ax_img.imshow(arr)
        ax_img.set_title(f"s={s}, frame={frame_idx}, y={int(labels[s])}")
        ax_img.axis("off")
    ax_blank = fig.add_subplot(gs[1, 3])
    ax_blank.axis("off")
    ax_blank.text(0.0, 0.8, "Official image rule", fontsize=10, weight="bold")
    ax_blank.text(0.0, 0.55, "current: indexes[s]\nprevious: indexes[max(s-5,0)]\nlabels: slip_label[s]", fontsize=9)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source.replace('/', '_')}_slip_alignment_event.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return {
        "source": source,
        "trajectory_id": str(traj_id),
        "sample": sample,
        "window": [lo, hi],
        "image_samples": sample_points,
        "output_png": str(out_path),
        "slip_label_at_sample": int(labels[sample]),
        "force_at_sample": forces[sample].tolist(),
        "official_delta_shear_at_sample": float(delta_shear[sample]) if len(delta_shear) > sample else None,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Flat/Sharp Slip Alignment Audit",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- run_id: `{report['run_id']}`",
        f"- raw_root: `{report['raw_root']}`",
        f"- derived_root: `{report['derived_root']}`",
        "",
        "## Conclusion",
        "",
        f"- structural_pass: `{report['conclusion']['structural_pass']}`",
        f"- derived_split_pass: `{report['conclusion']['derived_split_pass']}`",
        f"- dataloader_smoke_pass: `{report['conclusion']['dataloader_smoke_pass']}`",
        f"- official_delta_shear_positive_greater_than_negative: `{report['conclusion']['official_delta_shear_positive_greater_than_negative']}`",
        f"- recommendation: **{report['conclusion']['recommendation']}**",
        "",
        report['conclusion']['rationale'],
        "",
        "## Official sample-index semantics checked",
        "",
        "Sparsh `VisionForceSlipDataset` uses `indexes[5:]` only to set dataset length; the actual sample id remains `0..len(indexes[5:])-1`. For a sample `s`, labels use `slip_label[s]` and images use `indexes[s]` plus temporal context `indexes[max(s-5,0)]` for the default two-frame setup.",
        "",
        "## Source structure summary",
        "",
        "| source | trajectories | relation | official samples | slip ratio | transitions | shear pos/neg | issues |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for source in PRIMARY_AUDIT_SOURCES:
        rec = report["source_structure"][source]
        relation = "; ".join(f"{k}:{v}" for k, v in rec["length_relations"].items())
        shear_ratio = rec["delta_mag_shear_pos_over_neg"]
        shear_ratio_txt = f"{shear_ratio:.3f}" if shear_ratio is not None else "n/a"
        lines.append(
            f"| `{source}` | {rec['trajectory_count']} | {relation} | {rec['official_sample_count']} | "
            f"{rec['official_label_positive_ratio']:.4f} | {rec['official_zero_to_one_transition_count']} | "
            f"{shear_ratio_txt} | {rec['issue_count']} |"
        )
    lines.extend([
        "",
        "## Derived split consistency",
        "",
        "| source | train ratio | val ratio | test ratio | missing raw traj | passed |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for source in PRIMARY_AUDIT_SOURCES:
        rec = report["derived_consistency"][source]
        splits = rec["splits"]
        lines.append(
            f"| `{source}` | {splits['train']['official_label_positive_ratio']:.4f} | "
            f"{splits['val']['official_label_positive_ratio']:.4f} | {splits['test']['official_label_positive_ratio']:.4f} | "
            f"{rec['missing_source_trajectory_count']} | `{rec['passed']}` |"
        )
    lines.extend([
        "",
        "## Aggregate all-source slip balance",
        "",
        "| split | sphere-only ratio | all-source ratio | all-source positives/total |",
        "|---|---:|---:|---:|",
    ])
    for split in ["train", "val", "test"]:
        sphere = report["aggregate_balance"]["sphere_only"][split]
        allsrc = report["aggregate_balance"]["all_sources"][split]
        lines.append(
            f"| `{split}` | {sphere['official_label_positive_ratio']:.4f} | {allsrc['official_label_positive_ratio']:.4f} | "
            f"{allsrc['official_label_positive']}/{allsrc['official_label_total']} |"
        )
    lines.extend([
        "",
        "## Visual examples",
        "",
    ])
    for item in report["visual_examples"]:
        lines.append(f"- `{item['source']}` trajectory `{item['trajectory_id']}` sample `{item['sample']}`: `{item['output_png']}`")
    lines.extend([
        "",
        "## Notes",
        "",
        "- This audit does not modify raw data.",
        "- `org_dataset_gelsight_*` is not used by the official downstream loader and is not part of this audit's training compatibility check.",
        "- Passing this audit supports adding flat/sharp to a diagnostic all-source slip run. A full all-source training run is still recommended before upgrading it to the formal baseline.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=RUN_ID_FILE.read_text().strip())
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    args = parser.parse_args()

    run_id = args.run_id
    derived_root = DERIVED_BASE / run_id / "derived_gsmini"
    report_dir = args.report_root / run_id
    example_dir = report_dir / "flat_sharp_slip_alignment_examples"

    source_structure = {source: audit_source_structure(args.raw_root, source) for source in SOURCES}
    derived_consistency = {source: audit_derived_consistency(args.raw_root, derived_root, source) for source in SOURCES}

    flat_sharp_train = [f"{source_to_derived_prefix(s)}_train" for s in PRIMARY_AUDIT_SOURCES]
    flat_sharp_val = [f"{source_to_derived_prefix(s)}_val" for s in PRIMARY_AUDIT_SOURCES]
    flat_sharp_test = [f"{source_to_derived_prefix(s)}_test" for s in PRIMARY_AUDIT_SOURCES]
    sphere_sources = [s for s in SOURCES if s.startswith("sphere/")]
    sphere_train = [f"{source_to_derived_prefix(s)}_train" for s in sphere_sources]
    sphere_val = [f"{source_to_derived_prefix(s)}_val" for s in sphere_sources]
    sphere_test = [f"{source_to_derived_prefix(s)}_test" for s in sphere_sources]
    all_train = flat_sharp_train + sphere_train
    all_val = flat_sharp_val + sphere_val
    all_test = flat_sharp_test + sphere_test

    smoke = official_dataset_smoke(derived_root, flat_sharp_train + flat_sharp_val + flat_sharp_test)
    aggregate_balance = {
        "flat_sharp_only": {
            "train": aggregate_split_balance(derived_root, flat_sharp_train),
            "val": aggregate_split_balance(derived_root, flat_sharp_val),
            "test": aggregate_split_balance(derived_root, flat_sharp_test),
        },
        "sphere_only": {
            "train": aggregate_split_balance(derived_root, sphere_train),
            "val": aggregate_split_balance(derived_root, sphere_val),
            "test": aggregate_split_balance(derived_root, sphere_test),
        },
        "all_sources": {
            "train": aggregate_split_balance(derived_root, all_train),
            "val": aggregate_split_balance(derived_root, all_val),
            "test": aggregate_split_balance(derived_root, all_test),
        },
    }
    visual_examples = []
    for source in PRIMARY_AUDIT_SOURCES:
        item = render_event_plot(args.raw_root, source, example_dir)
        if item is not None:
            visual_examples.append(item)

    structural_pass = all(source_structure[s]["issue_count"] == 0 for s in PRIMARY_AUDIT_SOURCES)
    derived_split_pass = all(derived_consistency[s]["passed"] for s in PRIMARY_AUDIT_SOURCES)
    dataloader_smoke_pass = bool(smoke["passed"])
    shear_ratios = [source_structure[s].get("delta_mag_shear_pos_over_neg") for s in PRIMARY_AUDIT_SOURCES]
    shear_ok = all(r is not None and r > 1.0 for r in shear_ratios)
    if structural_pass and derived_split_pass and dataloader_smoke_pass and shear_ok:
        recommendation = "flat/sharp 可以进入下一步 all-source slip 诊断训练；正式 baseline 仍建议等 all-source 训练指标审计后再升级。"
        rationale = "flat/sharp 的 `len(indexes)==len(labels)+5` 与官方 `indexes[5:]` 采样长度规则结构一致，官方 dataloader 可读取派生 split，所有 source trajectory 都被 train/val/test 覆盖，且 slip 正样本的 official delta shear 平均值高于负样本。"
    else:
        recommendation = "暂不建议把 flat/sharp 纳入正式 slip 训练。"
        rationale = "至少一项结构、派生 split、dataloader 或 slip/delta-force 诊断未通过；请查看 JSON 中的 issue 列表。"

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "raw_root": str(args.raw_root),
        "derived_root": str(derived_root),
        "audited_sources": PRIMARY_AUDIT_SOURCES,
        "reference_sources": [s for s in SOURCES if s.startswith("sphere/")],
        "conclusion": {
            "structural_pass": structural_pass,
            "derived_split_pass": derived_split_pass,
            "dataloader_smoke_pass": dataloader_smoke_pass,
            "official_delta_shear_positive_greater_than_negative": shear_ok,
            "recommendation": recommendation,
            "rationale": rationale,
        },
        "source_structure": source_structure,
        "derived_consistency": derived_consistency,
        "dataloader_smoke": smoke,
        "aggregate_balance": aggregate_balance,
        "visual_examples": visual_examples,
    }
    write_json(report_dir / "flat_sharp_slip_alignment_audit.json", report)
    (report_dir / "flat_sharp_slip_alignment_audit.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["conclusion"], indent=2, ensure_ascii=False))
    print(report_dir / "flat_sharp_slip_alignment_audit.md")


if __name__ == "__main__":
    main()
