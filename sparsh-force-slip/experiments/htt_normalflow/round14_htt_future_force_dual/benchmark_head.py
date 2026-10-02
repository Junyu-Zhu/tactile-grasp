#!/usr/bin/env python3
import argparse,json,time
from pathlib import Path
import numpy as np,torch
import future_train as ft
def main():
 p=argparse.ArgumentParser();p.add_argument("--formal",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cuda:0");a=p.parse_args();rows={}
 for g in ft.GROUPS:
  ck=torch.load(a.formal/f"{g}_p1_s20260914/best.pth",map_location="cpu",weights_only=False);m=ft.init_model(g,20260914);m.load_state_dict(ck["model"]);m.eval().to(a.device);x=torch.zeros(1,9,195,device=a.device)
  with torch.inference_mode():
   for _ in range(100):m(x)
   torch.cuda.synchronize();base_alloc=torch.cuda.memory_allocated(a.device);torch.cuda.reset_peak_memory_stats(a.device);m(x);torch.cuda.synchronize();peak_alloc=torch.cuda.max_memory_allocated(a.device)
   torch.cuda.synchronize();samples=[]
   for _ in range(500):
    st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);st.record();m(x);en.record();torch.cuda.synchronize();samples.append(st.elapsed_time(en))
  params=sum(p.numel() for p in m.parameters());rows[g]={"parameters":params,"parameter_memory_bytes_fp32":sum(p.numel()*p.element_size() for p in m.parameters()),"resident_cuda_allocated_bytes_after_warmup":int(base_alloc),"batch1_forward_peak_incremental_allocated_bytes":int(max(0,peak_alloc-base_alloc)),"head_latency_ms_median":float(np.median(samples)),"head_latency_ms_p95":float(np.quantile(samples,.95)),"samples":len(samples)}
 inherited={"source":"R12 accepted F_class_trial_balanced deployment benchmark","sha256":"b7e98ac6be1366d61b75ef73c93bcc4c61db1c8f6355e037f3df966496e904ce","encoder_parameters":86255616,"visual_parameters":7459778,"force_parameters":7459971,"historical_cold_history_median_ms":90.96007095649838,"historical_streaming_median_ms":10.506300954148173}
 rows["V"]["complete_absolute_pipeline_parameters"]=inherited["encoder_parameters"]+inherited["visual_parameters"]+rows["V"]["parameters"]
 for g in ("F_concat","F_dual"):rows[g]["complete_absolute_pipeline_parameters"]=inherited["encoder_parameters"]+inherited["visual_parameters"]+inherited["force_parameters"]+rows[g]["parameters"]
 for g in ft.GROUPS:rows[g]["complete_parameter_memory_bytes_fp32_weights_only"]=4*rows[g]["complete_absolute_pipeline_parameters"]
 result={"schema":"round14_cost_v2","status":"complete","device":"NVIDIA RTX 6000D GPU0","future_heads":rows,"inherited_upstream_reference":inherited,"limitations":["Head-only latency and CUDA memory exclude image preprocessing, H2D, MAE, visual/force branches, disk IO, capture, and control.","Complete parameter memory is FP32 weights-only arithmetic; full-pipeline runtime peak memory was not remeasured.","R12 full-pipeline latency is an inherited reference, not a new end-to-end Round14 measurement.","V absolute prediction omits force branch; V change metric uses an external predicted-current anchor and therefore requires the force branch for that derived metric."]};ft.atomic_json(result,a.output);print(json.dumps(result))
if __name__=="__main__":main()
