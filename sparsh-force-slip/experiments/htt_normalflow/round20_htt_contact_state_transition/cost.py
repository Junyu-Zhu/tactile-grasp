#!/usr/bin/env python3
"""R20 added-module latency/memory; inherited full-path scope is explicit."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train import GROUPS, ChangeModel, atomic_json, sha

UPSTREAM_PARAMS=101175365  # R14 accepted F_concat complete params 101199326 minus 23961 head


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    torch.set_num_threads(4)
    results={}
    for group in GROUPS:
        model=ChangeModel(group).eval().to(a.device)
        x=torch.randn(1,9,195,device=a.device)
        with torch.inference_mode():
            for _ in range(100):model(x)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            resident=torch.cuda.memory_allocated()
            times=[]
            for _ in range(500):
                start=torch.cuda.Event(enable_timing=True);stop=torch.cuda.Event(enable_timing=True)
                start.record();model(x);stop.record();stop.synchronize()
                times.append(start.elapsed_time(stop))
            peak=torch.cuda.max_memory_allocated()
        params=sum(p.numel() for p in model.parameters())
        results[group]={'parameters':params,'fp32_weight_bytes':params*4,
                        'head_gpu_latency_median_ms':float(np.median(times)),
                        'head_gpu_latency_p95_ms':float(np.quantile(times,.95)),
                        'resident_allocated_bytes':resident,'peak_allocated_bytes':peak,
                        'incremental_peak_bytes':peak-resident,
                        'complete_parameter_count_with_common_visual_force_path':UPSTREAM_PARAMS+params}
    inherited=Path(__file__).resolve().parents[1]/'round14_htt_future_force_dual'/'COST.json'
    prior=json.loads(inherited.read_text())
    out={'schema':'round20_cost_v1','device':torch.cuda.get_device_name(a.device),
         'measurements':results,'samples':500,'historical_upstream_reference':prior['inherited_upstream_reference'],
         'historical_r14_cost_sha256':sha(inherited),
         'scope':'new-module measurements only; R14/R12 full-path reference shares accepted R10 force and visual input identities, but was not remeasured for R20',
         'full_path_latency_new_measurement':None,
         'limitations':['head-only excludes raw image decode, reference preprocessing, H2D, MAE, visual and force branches, IO and control',
                        'full parameter count is arithmetic for FP32 weights only; complete runtime memory peak not measured']}
    atomic_json(out,a.output);print(json.dumps({'status':'complete','groups':results}))

if __name__=='__main__':main()
