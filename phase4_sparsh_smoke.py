"""Run Phase 4 Sparsh force/slip dataloader smoke checks.

The script intentionally stops at dataloader smoke unless a real Sparsh
checkpoint is supplied.  This keeps Phase 4 honest: generated labels can prove
format connectivity, but they are not real performance ground truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch.utils.data import DataLoader

from tactile_ssl.data.vision_based_forces_slip_probes import VisionForceSlipDataset


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
        "model_forward": {
            "status": "BLOCKED-with-exact-contract",
            "reason": "No Sparsh encoder/task checkpoint was supplied for Phase 4; metrics are intentionally not run.",
        },
    }


def _merge_report(report_path: Path, smoke_report: dict[str, Any]) -> None:
    if not report_path.exists():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["smoke"] = smoke_report
    report["gates"]["FORCE_SMOKE_PASS"] = smoke_report["force"]["status"]
    report["gates"]["SLIP_SMOKE_PASS"] = smoke_report["slip"]["status"]
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary_path = report_path.with_name("summary.md")
    if summary_path.exists():
        text = summary_path.read_text(encoding="utf-8")
        text = text.replace("- FORCE_SMOKE_PASS: PASS\n", f"- FORCE_SMOKE_PASS: {smoke_report['force']['status']}\n")
        text = text.replace("- SLIP_SMOKE_PASS: PASS\n", f"- SLIP_SMOKE_PASS: {smoke_report['slip']['status']}\n")
        if "Model forward blocker:" not in text:
            text += (
                "\n## Smoke detail\n\n"
                f"- Force dataloader: {smoke_report['force']['status']}\n"
                f"- Slip dataloader: {smoke_report['slip']['status']}\n"
                "- Model forward blocker: No Sparsh encoder/task checkpoint was supplied; no metrics were run.\n"
            )
        summary_path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4 Sparsh dataloader smoke check.")
    parser.add_argument("--dataset-root", type=Path, default=Path("sim_dataset/phase4_sparsh_cube"))
    parser.add_argument("--dataset-name", default="cube_phase3_bridge")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    smoke_report = {
        "environment": {
            "torch_version": torch.__version__,
            "device": "cpu",
        },
        "force": _smoke_lane(args.dataset_root, args.dataset_name, batch_size=args.batch_size),
        "slip": _smoke_lane(args.dataset_root, args.dataset_name, batch_size=args.batch_size),
    }
    report_path = args.dataset_root / "smoke_report.json"
    report_path.write_text(json.dumps(smoke_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _merge_report(args.dataset_root / "bridge_report.json", smoke_report)
    print(json.dumps(smoke_report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
