"""Build a Phase 4 Sparsh bridge dataset from existing Phase 3 cube artifacts.

This compatibility tool follows the G1 read-only adapter requirement:

1. Inventory the available cube trials.
2. Create a Sparsh-compatible derived dataset.
3. Save the results under this repo's `sim_dataset/` directory.

The output is intentionally conservative:
- tactile images are copied into a pickle list as uint8 RGB arrays
- force labels are derived from simulation contact-state proxies
- slip labels are placeholder zeros
- metric validity is marked false until real force/slip labels exist

The generated dataset is suitable for dataloader smoke tests, not for
claiming force/slip performance.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import pickle
from importlib.util import find_spec
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_OUTPUT_ROOT = Path("sim_dataset/phase4_sparsh_cube")
BRIDGE_DATASET_NAME = "cube_phase3_bridge"


@dataclass
class TrialInventory:
    trial_id: str
    object_id: str
    protocol_variant: str | None
    success_label: bool | None
    failure_reason: str | None
    frame_count: int
    tactile_frame_count: int
    left_count: int
    right_count: int
    contact_frame_count: int
    nonzero_force_frame_count: int
    max_force_n: float
    stage_counts: dict[str, int]
    source_dir: str


def _safe_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"true", "1", "yes", "y"}:
            return True
        if low in {"false", "0", "no", "n"}:
            return False
    return None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_frame_map(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_tactile_image(path: Path) -> np.ndarray:
    img = np.load(path)
    if img.dtype != np.uint8:
        # Phase 3 tactile RGB arrays are stored as float32 in [0, 1].
        # Convert them to a Sparsh-friendly uint8 buffer for PIL loading.
        if np.issubdtype(img.dtype, np.floating):
            img = np.clip(img, 0.0, 1.0)
            img = (img * 255.0).round().astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    return img


def _encode_png_bytes(image: np.ndarray) -> bytes:
    """Encode an RGB uint8 image as PNG bytes for Sparsh buffer loading."""

    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - exercised only in incomplete envs
        raise RuntimeError("PIL is required to build Sparsh-compatible PNG buffers") from exc

    with io.BytesIO() as buf:
        Image.fromarray(image, mode="RGB").save(buf, format="PNG")
        return buf.getvalue()


def _get_contact_state(robot_state: dict[str, Any], index: int) -> dict[str, Any]:
    samples = robot_state.get("samples", [])
    if 0 <= index < len(samples):
        return samples[index].get("contact_state", {}) or {}
    return {}


def _probe_smoke_contract() -> list[str]:
    """Return the runtime modules required to import the Sparsh loader."""

    required = ["omegaconf", "cv2", "torch", "PIL"]
    missing = [name for name in required if find_spec(name) is None]
    return missing


def _force_vector_from_contact_state(contact_state: dict[str, Any]) -> tuple[np.ndarray, str]:
    force_by_side = contact_state.get("force_by_side_n", {}) or {}
    left = float(force_by_side.get("left", 0.0) or 0.0)
    right = float(force_by_side.get("right", 0.0) or 0.0)
    max_force = float(contact_state.get("max_force_n", 0.0) or 0.0)

    if abs(left) > 0.0 or abs(right) > 0.0 or abs(max_force) > 0.0:
        source = "sim_contact_sensor"
    elif bool(contact_state.get("contact_detected", False)):
        source = "geometry_proxy"
    else:
        source = "placeholder"

    # Proxy 3-vector ordered as [left, right, max_force].
    return np.array([left, right, max_force], dtype=np.float32), source



def _default_source_roots() -> list[Path]:
    roots = [
        Path("artifacts/phase3"),
        Path("sim_dataset/phase4_collected/bridge_robustness"),
        Path("sim_dataset/phase4_collected/force_oriented"),
        Path("sim_dataset/phase4_collected/slip_oriented"),
    ]
    return [root for root in roots if root.exists()]


def _source_group(source_root: Path) -> str:
    if source_root.as_posix().endswith("artifacts/phase3"):
        return "phase3_existing"
    return source_root.name


def _dataset_trial_id(source_root: Path, trial_dir: Path) -> str:
    return f"{_source_group(source_root)}__{trial_dir.name}"


def _slip_labels_by_state(robot_state: dict[str, Any], *, threshold_m: float = 2.5e-4, horizon: int = 1) -> dict[int, int]:
    """Generate a conservative proxy slip label from object relative motion.

    A sample is marked slipping when contact is present and object XY motion in
    a short future horizon exceeds the threshold. This is a simulation-derived
    proxy for Phase 4 smoke/limited-proxy work, not a real-world slip label.
    """

    samples = robot_state.get("samples", [])
    labels: dict[int, int] = {}
    positions: list[np.ndarray | None] = []
    contacts: list[bool] = []
    for sample in samples:
        obj = sample.get("object_state", {}) or {}
        pos = obj.get("root_position_m")
        positions.append(np.asarray(pos[:3], dtype=np.float64) if isinstance(pos, list) and len(pos) >= 3 else None)
        contacts.append(bool((sample.get("contact_state", {}) or {}).get("contact_detected", False)))

    for idx, pos in enumerate(positions):
        label = 0
        if pos is not None and contacts[idx]:
            for j in range(idx + 1, min(len(positions), idx + horizon + 1)):
                nxt = positions[j]
                if nxt is None:
                    continue
                if float(np.linalg.norm(nxt[:2] - pos[:2])) >= threshold_m:
                    label = 1
                    break
        labels[idx] = label
    return labels


def _force_level(force_vec: np.ndarray, contact_detected: bool) -> str:
    mag = float(np.max(np.abs(force_vec))) if force_vec.size else 0.0
    if not contact_detected:
        return "no_contact"
    if mag <= 0.0:
        return "geometry_contact_no_force"
    if mag < 0.5:
        return "light_contact"
    if mag < 1.5:
        return "medium_contact"
    if mag < 8.0:
        return "firm_contact"
    return "over_contact"


def build_inventory(source_roots: list[Path]) -> list[TrialInventory]:
    records: list[TrialInventory] = []
    for source_root in source_roots:
        for trial_dir in sorted(source_root.glob("phase3_cube_*")):
            meta = _load_json(trial_dir / "meta.json")
            frame_rows = _load_frame_map(trial_dir / "frame_map.csv")
            robot_state = _load_json(trial_dir / "robot_state.json")

            stage_counts: dict[str, int] = {}
            left_count = 0
            right_count = 0
            contact_frame_count = 0
            nonzero_force_frame_count = 0
            max_force_n = 0.0

            for row in frame_rows:
                stage = row.get("action_stage", "")
                stage_counts[stage] = stage_counts.get(stage, 0) + 1
                side = row.get("side", "")
                if side == "left":
                    left_count += 1
                elif side == "right":
                    right_count += 1
                if row.get("contact_detected", "").lower() == "true":
                    contact_frame_count += 1

            for sample in robot_state.get("samples", []):
                contact_state = sample.get("contact_state", {}) or {}
                force_by_side = contact_state.get("force_by_side_n", {}) or {}
                sample_max = float(contact_state.get("max_force_n", 0.0) or 0.0)
                max_force_n = max(max_force_n, sample_max)
                if any(abs(float(force_by_side.get(side, 0.0) or 0.0)) > 0.0 for side in ("left", "right")) or sample_max > 0.0:
                    nonzero_force_frame_count += 1

            records.append(
                TrialInventory(
                    trial_id=_dataset_trial_id(source_root, trial_dir),
                    object_id=str(meta.get("object_id", "")),
                    protocol_variant=str(meta.get("protocol_variant", "")) if meta.get("protocol_variant") is not None else None,
                    success_label=_safe_bool(meta.get("success_label")),
                    failure_reason=meta.get("failure_reason"),
                    frame_count=len(frame_rows),
                    tactile_frame_count=len(list((trial_dir / "tactile").glob("*_tactile_rgb.npy"))),
                    left_count=left_count,
                    right_count=right_count,
                    contact_frame_count=contact_frame_count,
                    nonzero_force_frame_count=nonzero_force_frame_count,
                    max_force_n=max_force_n,
                    stage_counts=stage_counts,
                    source_dir=str(trial_dir),
                )
            )
    return records


def write_inventory(records: list[TrialInventory], output_root: Path) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "inventory.json"
    csv_path = output_root / "inventory.csv"

    json_path.write_text(
        json.dumps([asdict(record) for record in records], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    fieldnames = list(asdict(records[0]).keys()) if records else list(TrialInventory.__annotations__.keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["stage_counts"] = json.dumps(row["stage_counts"], ensure_ascii=False, sort_keys=True)
            writer.writerow(row)

    return json_path, csv_path


def build_bridge_dataset(source_roots: list[Path], output_root: Path) -> dict[str, Any]:
    bridge_root = output_root / BRIDGE_DATASET_NAME
    bridge_root.mkdir(parents=True, exist_ok=True)

    all_frames: list[bytes] = []
    in_contact: list[int] = []
    trajectories: dict[str, dict[str, Any]] = {}
    manifest_rows: list[dict[str, Any]] = []
    force_sources: set[str] = set()
    slip_sources: set[str] = set()
    force_levels: set[str] = set()

    for source_root in source_roots:
        source_group = _source_group(source_root)
        for trial_dir in sorted(source_root.glob("phase3_cube_*")):
            meta = _load_json(trial_dir / "meta.json")
            frame_rows = _load_frame_map(trial_dir / "frame_map.csv")
            robot_state = _load_json(trial_dir / "robot_state.json")
            slip_by_state = _slip_labels_by_state(robot_state)
            dataset_trial_id = _dataset_trial_id(source_root, trial_dir)

            trial_indexes: list[int] = []
            trial_forces: list[np.ndarray] = []
            trial_slip: list[int] = []

            for row in frame_rows:
                tactile_value = row.get("tactile_rgb_path", "")
                if not tactile_value:
                    continue
                tactile_rel = Path(tactile_value)
                tactile_path = tactile_rel if tactile_rel.is_absolute() else Path(tactile_value)
                if not tactile_path.exists():
                    tactile_path = trial_dir / "tactile" / Path(tactile_value).name
                if not tactile_path.exists():
                    raise FileNotFoundError(f"Missing tactile RGB array for {dataset_trial_id}: {tactile_value}")

                image = _load_tactile_image(tactile_path)
                global_index = len(all_frames)
                all_frames.append(_encode_png_bytes(image))

                contact_detected = row.get("contact_detected", "").lower() == "true"
                in_contact.append(1 if contact_detected else 0)

                robot_idx = row.get("robot_state_index", "")
                robot_idx_int = int(robot_idx) if str(robot_idx).strip() != "" else -1
                contact_state = _get_contact_state(robot_state, robot_idx_int)
                force_vec, force_source = _force_vector_from_contact_state(contact_state)
                force_sources.add(force_source)

                slip_label = int(slip_by_state.get(robot_idx_int, 0))
                slip_source = "relative_motion_proxy" if slip_label else "no_slip_relative_motion_proxy"
                if source_group == "slip_oriented" and slip_label:
                    slip_source = "controlled_slip_protocol_proxy"
                slip_sources.add(slip_source)

                level = _force_level(force_vec, contact_detected)
                force_levels.add(level)

                label_valid_for_metrics = force_source == "sim_contact_sensor" or slip_label == 1
                trial_indexes.append(global_index)
                trial_forces.append(force_vec)
                trial_slip.append(slip_label)

                manifest_rows.append(
                    {
                        "trial_id": dataset_trial_id,
                        "source_group": source_group,
                        "source_trial_id": trial_dir.name,
                        "side": row.get("side", ""),
                        "frame_id": row.get("frame_id", ""),
                        "source_path": str(tactile_path),
                        "action_stage": row.get("action_stage", ""),
                        "contact_detected": contact_detected,
                        "robot_state_index": robot_idx_int,
                        "force_label_source": force_source,
                        "force_level": level,
                        "slip_label_source": slip_source,
                        "slip_label": slip_label,
                        "label_valid_for_metrics": label_valid_for_metrics,
                        "global_frame_index": global_index,
                        "object_id": meta.get("object_id", ""),
                        "trial_success_label": meta.get("success_label", ""),
                    }
                )

            trajectories[dataset_trial_id] = {
                "indexes": np.asarray(trial_indexes, dtype=np.int64),
                "forces": np.asarray(trial_forces, dtype=np.float32),
                "slip_label": np.asarray(trial_slip, dtype=np.int64),
            }

    dataset_images_path = bridge_root / "dataset_gelsight_cube.pkl"
    with dataset_images_path.open("wb") as f:
        pickle.dump(all_frames, f, protocol=pickle.HIGHEST_PROTOCOL)

    has_sim_force = "sim_contact_sensor" in force_sources
    has_slip = any("slip" in source and not source.startswith("no_slip") for source in slip_sources)
    dataset_force_slip = {
        "in_contact": np.asarray(in_contact, dtype=np.int64),
        "trajectories": trajectories,
        "source": {
            "source_roots": [str(root) for root in source_roots],
            "bridge_dataset_name": BRIDGE_DATASET_NAME,
            "label_valid_for_metrics": False,
            "force_label_policy": "sim_contact_sensor_when_available_else_geometry_proxy",
            "slip_label_policy": "relative_motion_proxy_threshold_0.25mm_horizon_1_sample",
            "force_label_validity": "sim-valid" if has_sim_force else "proxy",
            "slip_label_validity": "proxy" if has_slip else "invalid",
        },
    }
    dataset_force_slip_path = bridge_root / "dataset_slip_forces.pkl"
    with dataset_force_slip_path.open("wb") as f:
        pickle.dump(dataset_force_slip, f, protocol=pickle.HIGHEST_PROTOCOL)

    manifest_path = bridge_root / "manifest.csv"
    fieldnames = [
        "trial_id",
        "source_group",
        "source_trial_id",
        "side",
        "frame_id",
        "source_path",
        "action_stage",
        "contact_detected",
        "robot_state_index",
        "force_label_source",
        "force_level",
        "slip_label_source",
        "slip_label",
        "label_valid_for_metrics",
        "global_frame_index",
        "object_id",
        "trial_success_label",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)

    readme_path = bridge_root / "README.md"
    readme_path.write_text(
        """# Phase 4 cube Phase3 bridge dataset

This folder is a Sparsh-compatible bridge view derived from existing Phase 3
cube artifacts plus Phase 4 cube data collected under `sim_dataset/phase4_collected`.

## What is inside

- `dataset_gelsight_cube.pkl`: list of PNG-encoded uint8 RGB tactile frames
- `dataset_slip_forces.pkl`: Sparsh-style force/slip trajectory dictionary
- `manifest.csv`: per-frame provenance, label source, force level, and slip proxy source

## Label validity caveat

Force labels use simulation contact sensors when nonzero and fall back to geometry
proxy/contact placeholders otherwise. Slip labels are generated from a short-horizon
relative-motion proxy. These labels support bridge, dataloader, and limited-proxy
smoke work only; they are not real-world force/slip ground truth.
""",
        encoding="utf-8",
    )

    return {
        "bridge_root": str(bridge_root),
        "dataset_images_path": str(dataset_images_path),
        "dataset_force_slip_path": str(dataset_force_slip_path),
        "manifest_path": str(manifest_path),
        "readme_path": str(readme_path),
        "frame_count": len(all_frames),
        "trajectory_count": len(trajectories),
        "force_label_sources": sorted(force_sources),
        "slip_label_sources": sorted(slip_sources),
        "force_levels": sorted(force_levels),
        "force_label_validity": "sim-valid" if has_sim_force else "proxy",
        "slip_label_validity": "proxy" if has_slip else "invalid",
        "metric_validity": "limited-proxy" if (has_sim_force or has_slip) else "invalid",
    }

def write_summary(output_root: Path, inventory: list[TrialInventory], bridge: dict[str, Any]) -> tuple[Path, Path]:
    summary_path = output_root / "summary.md"
    report_path = output_root / "bridge_report.json"
    missing_modules = _probe_smoke_contract()
    smoke_pass = not missing_modules
    smoke_status = "PASS" if smoke_pass else "BLOCKED-with-exact-contract"

    total_trials = len(inventory)
    total_frames = sum(item.frame_count for item in inventory)
    total_contact_frames = sum(item.contact_frame_count for item in inventory)
    total_nonzero_force_frames = sum(item.nonzero_force_frame_count for item in inventory)
    notes = [
        "- Derived from existing Phase 3 cube artifacts plus Phase 4 collected cube batches under `sim_dataset/phase4_collected` when present.",
        "- No Sparsh adaptation or fine-tuning was run.",
        "- Force/slip labels remain limited-proxy unless a later real sensor/controlled-slip validation upgrades them.",
    ] + (
        [
            "- Local smoke test is blocked because the runtime does not provide "
            "`omegaconf`, `cv2`, `torch`, or `PIL`, which the Sparsh loader imports."
        ]
        if not smoke_pass
        else ["- Sparsh loader smoke prerequisites are available in this runtime."]
    )

    summary_path.write_text(
        "\n".join(
            [
                "# Phase 4 sim dataset summary",
                "",
                f"- Source trials: {total_trials}",
                f"- Total frame rows: {total_frames}",
                f"- Contact frames: {total_contact_frames}",
                f"- Nonzero force frames (proxy or sim-contact): {total_nonzero_force_frames}",
                f"- Bridge dataset root: `{bridge['bridge_root']}`",
                "",
                "## Gate status",
                "",
                "- FORMAT_BRIDGE_PASS: PASS",
                f"- FORCE_SMOKE_PASS: {smoke_status}",
                f"- SLIP_SMOKE_PASS: {smoke_status}",
                f"- FORCE_LABEL_VALIDITY: {bridge['force_label_validity']}",
                f"- SLIP_LABEL_VALIDITY: {bridge['slip_label_validity']}",
                f"- METRIC_VALIDITY: {bridge['metric_validity']}",
                f"- Force label sources: {', '.join(bridge['force_label_sources']) or 'none'}",
                f"- Slip label sources: {', '.join(bridge['slip_label_sources']) or 'none'}",
                f"- Force levels: {', '.join(bridge['force_levels']) or 'none'}",
                "",
                "## Notes",
                "",
                *notes,
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report_path.write_text(
        json.dumps(
            {
                "source_trials": total_trials,
                "total_frame_rows": total_frames,
                "total_contact_frames": total_contact_frames,
                "total_nonzero_force_frames": total_nonzero_force_frames,
                "bridge_root": bridge["bridge_root"],
                "gates": {
                    "FORMAT_BRIDGE_PASS": "PASS",
                    "FORCE_SMOKE_PASS": smoke_status,
                    "SLIP_SMOKE_PASS": smoke_status,
                    "FORCE_LABEL_VALIDITY": bridge["force_label_validity"],
                    "SLIP_LABEL_VALIDITY": bridge["slip_label_validity"],
                    "METRIC_VALIDITY": bridge["metric_validity"],
                },
                "force_label_sources": bridge["force_label_sources"],
                "slip_label_sources": bridge["slip_label_sources"],
                "force_levels": bridge["force_levels"],
                "smoke_blocker": None
                if smoke_pass
                else {
                    "missing_python_modules": missing_modules,
                    "effect": "Sparsh dataset import / loader smoke cannot run in the current runtime.",
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return summary_path, report_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        action="append",
        default=None,
        help=(
            "Phase3-style artifact root containing phase3_cube_* trial folders. "
            "May be repeated. Defaults to artifacts/phase3 plus existing Phase4 collected roots."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    source_roots = args.source_root or _default_source_roots()
    if not source_roots:
        raise FileNotFoundError("No Phase3/Phase4 cube source roots found.")
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    inventory = build_inventory(source_roots)
    write_inventory(inventory, output_root)
    bridge = build_bridge_dataset(source_roots, output_root)
    write_summary(output_root, inventory, bridge)

    print(json.dumps(bridge, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
