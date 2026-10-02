#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import future_train as ft


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    rows = {}
    for group in ft.GROUPS:
        checkpoint = torch.load(arguments.formal / f"{group}_p1_s20260914" / "best.pth", map_location="cpu", weights_only=False)
        model = ft.init_model(group, 20260914)
        model.load_state_dict(checkpoint["model"])
        model.eval().to(arguments.device)
        value = torch.zeros(1, 9, 195, device=arguments.device)
        with torch.inference_mode():
            for _ in range(100):
                model(value)
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated(arguments.device)
            torch.cuda.reset_peak_memory_stats(arguments.device)
            model(value)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated(arguments.device)
            samples = []
            for _ in range(500):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(); model(value); end.record(); torch.cuda.synchronize()
                samples.append(start.elapsed_time(end))
        parameters = sum(parameter.numel() for parameter in model.parameters())
        rows[group] = {"parameters": parameters,
                       "parameter_memory_bytes_fp32": sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()),
                       "resident_cuda_allocated_bytes_after_warmup": int(baseline),
                       "batch1_forward_peak_incremental_allocated_bytes": int(max(0, peak - baseline)),
                       "head_latency_ms_median": float(np.median(samples)), "head_latency_ms_p95": float(np.quantile(samples, 0.95)),
                       "samples": len(samples), "effective_gru_steps": 1 if group == "C_current" else 9,
                       "required_raw_image_offsets": [-5, 0] if group == "C_current" else [-13, 0]}
    inherited = {"source": "R12 accepted F_class_trial_balanced deployment benchmark",
                 "sha256": "b7e98ac6be1366d61b75ef73c93bcc4c61db1c8f6355e037f3df966496e904ce",
                 "encoder_parameters": 86255616, "visual_parameters": 7459778, "force_parameters": 7459971,
                 "historical_cold_history_median_ms": 90.96007095649838, "historical_streaming_median_ms": 10.506300954148173}
    for group in ft.GROUPS:
        rows[group]["complete_absolute_pipeline_parameters"] = inherited["encoder_parameters"] + inherited["visual_parameters"] + inherited["force_parameters"] + rows[group]["parameters"]
        rows[group]["complete_parameter_memory_bytes_fp32_weights_only"] = 4 * rows[group]["complete_absolute_pipeline_parameters"]
    receipt = {"schema": "round15_cost_v1", "status": "complete", "device": "NVIDIA RTX 6000D GPU0",
               "future_heads": rows, "inherited_upstream_reference": inherited,
               "limitations": ["Head-only latency and CUDA memory exclude image preprocessing, H2D, MAE, visual/force branches, disk IO, capture, and control.",
                               "Complete parameter memory is FP32 weights-only arithmetic; full-pipeline runtime peak memory was not remeasured.",
                               "R12 full-pipeline latency is inherited, not a new end-to-end Round15 measurement.",
                               "C_current can require only raw t and t-5 while H_history requires raw t-13..t, but no new complete-pipeline latency claim is made from this dependency difference."]}
    ft.atomic_json(receipt, arguments.output)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
