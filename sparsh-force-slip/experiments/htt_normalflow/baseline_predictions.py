"""Frozen-manifest B0 prediction entry point; metrics/calibration are round 2."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import torch

from adapters import collate, iter_split_windows
from prepare_splits import verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--fold", default="htt_leave_p1")
    parser.add_argument("--partition", choices=("train", "validation", "calibration", "test"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.max_windows is not None and args.max_windows < 1):
        raise ValueError("Batch and window limits must be positive")
    if not args.fold.startswith("htt_leave_"):
        raise ValueError("B0 slip prediction entry is for HTT; NormalFlow has no slip labels")
    output = Path(args.output)
    if output.suffix.lower() != ".csv":
        raise ValueError("Prediction output must use .csv to keep metadata separate")
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("Choose a new output to preserve previous predictions")
    manifest = json.loads(Path(args.manifest).read_text())
    verify(manifest)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import phase2_b_multitask as p2
    torch.set_num_threads(4)
    model, payload = p2.load_b_checkpoint(Path(args.checkpoint), torch.device(args.device))
    if model.encoder_name != "mae" or model.decoder_variant != "decoupled":
        raise ValueError("This entry fixes the selected two-frame MAE decoupled convention")
    model.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    pending = []
    fields = ["episode_id", "domain", "probe_id", "object_id", "label_source", "path", "t", "stage", "stage_valid", "p_slip", "force_x_normalized", "force_y_normalized", "force_z_normalized"]
    with output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()

        def flush():
            batch = collate(pending)
            if tuple(batch["inputs"]["image"].shape[1:]) != (6, 320, 240):
                raise ValueError("Unexpected MAE input shape")
            with torch.inference_mode():
                pred = model(batch["inputs"]["image"].to(args.device))
            probabilities = torch.softmax(pred["slip"], dim=1)[:, 1].cpu()
            forces = pred["force"].cpu()
            if tuple(forces.shape) != (len(pending), 3) or tuple(pred["slip"].shape) != (len(pending), 2):
                raise ValueError("Unexpected B0 output shape")
            if not torch.isfinite(probabilities).all() or not torch.isfinite(forces).all():
                raise RuntimeError("Nonfinite old-model predictions")
            for index, sample in enumerate(pending):
                meta, target = sample["metadata"], sample["targets"]
                writer.writerow({"episode_id": meta["episode_id"], "domain": meta["domain"], "t": meta["t"],
                                 **{k: meta[k] for k in ("probe_id", "object_id", "label_source", "path")},
                                 "stage": int(target["stage"]), "stage_valid": bool(target["stage_valid"]),
                                 "p_slip": float(probabilities[index]),
                                 **{f"force_{axis}_normalized": float(forces[index, j]) for j, axis in enumerate("xyz")}})
            stream.flush()
            pending.clear()

        for sample in iter_split_windows(manifest, args.fold, args.partition):
            pending.append(sample)
            count += 1
            if len(pending) == args.batch_size:
                flush()
            if args.max_windows is not None and count >= args.max_windows:
                break
        if pending:
            flush()
    row_map = {r["id"]: r for r in manifest["episodes"]}
    expected = sum(row_map[i]["frames"] for i in manifest["splits"][args.fold][args.partition])
    summary = {"status": "complete" if count == expected else "partial_smoke", "windows": count, "expected_windows": expected,
               "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
               "checkpoint_epoch": payload.get("epoch"), "checkpoint_format": payload.get("format"),
               "encoder": model.encoder_name, "decoder_variant": model.decoder_variant,
               "source_content_hashes_verified": True,
               "fold": args.fold, "partition": args.partition, "checkpoint": args.checkpoint,
               "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
               "output": str(output), "device": args.device,
               "scope": "raw predictions; no calibration, no force unit conversion, no accuracy claim"}
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
