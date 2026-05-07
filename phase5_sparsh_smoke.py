"""Run Phase 5 Sparsh dataloader and gate checks for cube force/slip v2."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from tactile_ssl.data.vision_based_forces_slip_probes import VisionForceSlipDataset

DEFAULT_DATASET_ROOT = Path("sim_dataset/phase5_cube_force_slip/sparsh_export_v2")
DEFAULT_DATASET_NAME = "cube_force_slip_v2"
DEFAULT_REPORT_ROOT = Path("artifacts/phase5_cube_force_slip/reports")


def _config(dataset_root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        sensor="gelsight",
        remove_bg=True,
        out_format="concat_ch_img",
        num_frames=2,
        frame_stride=5,
        path_dataset=dataset_root.as_posix(),
        slip_horizon=0,
        max_abs_forceXYZ=[1.5, 1.5, 2.0],
        max_delta_forceXYZ=[0.8, 0.8, 0.8],
        transforms=SimpleNamespace(resize=[320, 240]),
    )


def _load_export_manifest(dataset_root: Path, dataset_name: str) -> dict[str, Any]:
    path = dataset_root / dataset_name / "export_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_force_slip(dataset_root: Path, dataset_name: str) -> dict[str, Any]:
    with (dataset_root / dataset_name / "dataset_slip_forces.pkl").open("rb") as f:
        return pickle.load(f)


def _mask_counts(force_slip: dict[str, Any]) -> dict[str, int]:
    trajectories = force_slip.get("trajectories", {})
    valid_force = 0
    invalid_force = 0
    valid_slip = 0
    invalid_slip = 0
    positive_slip_valid = 0
    negative_slip_valid = 0
    for trajectory in trajectories.values():
        vf = np.asarray(trajectory.get("valid_force", []), dtype=bool)
        vs = np.asarray(trajectory.get("valid_slip", []), dtype=bool)
        labels = np.asarray(trajectory.get("slip_label", []), dtype=int)
        valid_force += int(vf.sum())
        invalid_force += int(vf.size - vf.sum())
        valid_slip += int(vs.sum())
        invalid_slip += int(vs.size - vs.sum())
        positive_slip_valid += int(((labels == 1) & vs).sum())
        negative_slip_valid += int(((labels == 0) & vs).sum())
    return {
        "valid_force": valid_force,
        "invalid_force": invalid_force,
        "valid_slip": valid_slip,
        "invalid_slip": invalid_slip,
        "positive_slip_valid": positive_slip_valid,
        "negative_slip_valid": negative_slip_valid,
    }


def _smoke_lane(dataset_root: Path, dataset_name: str, *, batch_size: int) -> dict[str, Any]:
    dataset = VisionForceSlipDataset(config=_config(dataset_root), dataset_name=dataset_name)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    first_item = dataset[0]
    first_batch = next(iter(dataloader))
    return {
        "status": "PASS_DATALOADER_ONLY",
        "dataset_len": len(dataset),
        "item_image_shape": list(first_item["image"].shape),
        "item_force_shape": list(first_item["force"].shape),
        "item_delta_force_shape": list(first_item["delta_force"].shape),
        "item_slip_label": int(first_item["slip_label"].item()),
        "batch_image_shape": list(first_batch["image"].shape),
        "batch_force_shape": list(first_batch["force"].shape),
        "batch_delta_force_shape": list(first_batch["delta_force"].shape),
        "batch_slip_shape": list(first_batch["slip_label"].shape),
    }


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _update_reports(
    *,
    report_root: Path,
    manifest_path: Path,
    smoke_report: dict[str, Any],
    manifest: dict[str, Any],
    mask_counts: dict[str, int],
    checkpoint: Path | None,
) -> None:
    format_status = smoke_report["force"]["status"]
    force_validity = manifest.get("gates", {}).get("FORCE_LABEL_VALIDITY", "invalid")
    slip_validity = manifest.get("gates", {}).get("SLIP_LABEL_VALIDITY", "invalid")
    label_gate_pass = (
        force_validity == "sim-valid"
        and slip_validity == "sim-valid"
        and mask_counts["valid_force"] > 0
        and mask_counts["positive_slip_valid"] > 0
        and mask_counts["negative_slip_valid"] > 0
    )
    if checkpoint is None:
        eval_gate = "BLOCKED_NO_CHECKPOINT_NO_TRUSTED_METRICS"
        eval_reason = "No Sparsh encoder/task checkpoint was supplied; no metrics were run."
    elif not checkpoint.exists():
        eval_gate = "BLOCKED_CHECKPOINT_PATH_MISSING"
        eval_reason = f"Checkpoint path does not exist: {checkpoint}"
    elif not label_gate_pass:
        eval_gate = "BLOCKED_LABEL_VALIDITY"
        eval_reason = "Label validity or valid-mask/class coverage did not pass."
    else:
        eval_gate = "READY_FOR_CHECKPOINT_BACKED_EVAL"
        eval_reason = "Checkpoint exists and gates pass, but this smoke script intentionally does not claim task metrics."

    manifest.setdefault("gates", {})["FORMAT_GATE"] = format_status
    manifest["gates"]["EVAL_GATE"] = eval_gate
    manifest.setdefault("sparsh_smoke", {})["report"] = (report_root / "phase5_sparsh_smoke_report.json").as_posix()
    manifest["sparsh_smoke"]["mask_counts"] = mask_counts
    manifest["sparsh_smoke"]["dataset_len"] = smoke_report["force"]["dataset_len"]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    _write(
        report_root / "format_gate.md",
        f"""# Format gate

FORMAT_GATE: **{format_status}**

## Sparsh dataloader evidence

- Dataset root: `{smoke_report['dataset_root']}`
- Dataset name: `{smoke_report['dataset_name']}`
- Dataset length: {smoke_report['force']['dataset_len']}
- Item image shape: `{smoke_report['force']['item_image_shape']}`
- Item force shape: `{smoke_report['force']['item_force_shape']}`
- Item delta-force shape: `{smoke_report['force']['item_delta_force_shape']}`
- Batch image shape: `{smoke_report['force']['batch_image_shape']}`
- Batch force shape: `{smoke_report['force']['batch_force_shape']}`
- Batch delta-force shape: `{smoke_report['force']['batch_delta_force_shape']}`
- Batch slip shape: `{smoke_report['force']['batch_slip_shape']}`
""",
    )
    _write(
        report_root / "label_validity_gate.md",
        f"""# Label validity gate

- FORCE_LABEL_VALIDITY: **{force_validity}**
- SLIP_LABEL_VALIDITY: **{slip_validity}**
- Label gate pass: **{label_gate_pass}**

## Mask/class coverage

- Valid force frames: {mask_counts['valid_force']}
- Invalid/masked force frames: {mask_counts['invalid_force']}
- Valid slip frames: {mask_counts['valid_slip']}
- Invalid/masked slip frames: {mask_counts['invalid_slip']}
- Valid positive slip frames: {mask_counts['positive_slip_valid']}
- Valid no-slip frames: {mask_counts['negative_slip_valid']}

Metric consumers must apply these masks; format-loaded rows are not all metric-valid.
""",
    )
    _write(
        report_root / "eval_gate.md",
        f"""# Eval gate

EVAL_GATE: **{eval_gate}**

Reason: {eval_reason}

No trusted force/slip metrics are reported by this Phase 5 smoke run.
""",
    )

    review_path = report_root / "phase5_review.md"
    if review_path.exists():
        text = review_path.read_text(encoding="utf-8")
        text = text.replace("| `FORMAT_GATE` | PASS_STRUCTURAL |", f"| `FORMAT_GATE` | {format_status} |")
        if "## Sparsh smoke evidence" not in text:
            text += (
                "\n## Sparsh smoke evidence\n\n"
                f"- FORMAT_GATE: {format_status}\n"
                f"- Label gate pass: {label_gate_pass}\n"
                f"- EVAL_GATE: {eval_gate}\n"
                "- Trusted metrics: not run.\n"
            )
        review_path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 5 Sparsh dataloader smoke/gate check.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    manifest = _load_export_manifest(args.dataset_root, args.dataset_name)
    manifest_path = args.dataset_root / args.dataset_name / "export_manifest.json"
    force_slip = _load_force_slip(args.dataset_root, args.dataset_name)
    mask_counts = _mask_counts(force_slip)
    smoke_report = {
        "environment": {"torch_version": torch.__version__, "device": "cpu"},
        "dataset_root": args.dataset_root.as_posix(),
        "dataset_name": args.dataset_name,
        "manifest_gates": manifest.get("gates", {}),
        "mask_counts": mask_counts,
        "force": _smoke_lane(args.dataset_root, args.dataset_name, batch_size=args.batch_size),
        "slip": _smoke_lane(args.dataset_root, args.dataset_name, batch_size=args.batch_size),
    }
    args.report_root.mkdir(parents=True, exist_ok=True)
    (args.report_root / "phase5_sparsh_smoke_report.json").write_text(
        json.dumps(smoke_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _update_reports(
        report_root=args.report_root,
        manifest_path=manifest_path,
        smoke_report=smoke_report,
        manifest=manifest,
        mask_counts=mask_counts,
        checkpoint=args.checkpoint,
    )
    print(json.dumps(smoke_report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
