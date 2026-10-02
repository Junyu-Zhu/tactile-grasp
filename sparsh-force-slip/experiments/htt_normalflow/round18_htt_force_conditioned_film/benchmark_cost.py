#!/usr/bin/env python3
"""Fresh Round-18 cached-head cost plus exact-identity end-to-end evidence applicability."""
from __future__ import annotations
import argparse,json,statistics,time
from pathlib import Path
import torch
import numpy as np
import train as r18
def med_repeats(source,key):return statistics.median(row["median_ms"] for row in source["measurements"][key])
def main():
 p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True);p.add_argument("--formal-root",type=Path,required=True);p.add_argument("--r9-v-cost",type=Path,required=True);p.add_argument("--r12-f-cost",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cuda:0");a=p.parse_args();data=torch.load(a.data,map_location="cpu",weights_only=False);r9=json.loads(a.r9_v_cost.read_text());r12=json.loads(a.r12_f_cost.read_text());up=data["provenance"]["immutable_upstream"]
 assert up["source_checkpoint"]["sha256"]==r9["upstream"]["source_checkpoint"]["sha256"]==r12["upstream"]["source_checkpoint"]["sha256"]
 assert up["visual_checkpoint"]["sha256"]==r9["upstream"]["visual_checkpoint"]["sha256"]==r12["upstream"]["visual_checkpoint"]["sha256"]
 assert up["force_checkpoint"]["sha256"]==r12["upstream"]["force_checkpoint"]["sha256"]
 result={"schema":"round18_cost_v1","status":"complete","device":torch.cuda.get_device_name(a.device),"cached_heads":{},"end_to_end_evidence":{},"limitations":["Fresh head timing excludes image decoding, H2D, MAE, visual/force branches, disk IO, capture, and robot control.","End-to-end timings are reused only because source/visual/force checkpoint identities and raw image dependency match exactly; shared-GPU timings remain descriptive."]}
 for group in r18.GROUPS:
  ck=torch.load(a.formal_root/group/"p1_s20260914"/"best.pth",map_location="cpu",weights_only=False);model=r18.init_model(group,20260914);model.load_state_dict(ck["model"]);model.to(a.device).eval();x=((data["roles"]["validation"]["x"][:1]-ck["normalizer"]["mean"])/ck["normalizer"]["std"]).to(a.device)
  with torch.inference_mode():
   for _ in range(50):model(x)
   torch.cuda.synchronize();baseline=int(torch.cuda.memory_allocated(a.device));torch.cuda.reset_peak_memory_stats(a.device);samples=[]
   for _ in range(500):
    began=time.perf_counter();model(x);torch.cuda.synchronize();samples.append((time.perf_counter()-began)*1000)
  params=sum(p.numel() for p in model.parameters());result["cached_heads"][group]={"parameters":params,"fp32_parameter_bytes":params*4,"latency_ms_median":float(np.median(samples)),"latency_ms_p95":float(np.quantile(samples,.95)),"baseline_allocated_bytes":baseline,"peak_allocated_bytes":int(torch.cuda.max_memory_allocated(a.device)),"incremental_peak_above_baseline_bytes":int(torch.cuda.max_memory_allocated(a.device))-baseline,"samples":len(samples),"checkpoint":str(a.formal_root/group/"p1_s20260914"/"best.pth"),"checkpoint_sha256":r18.sha(a.formal_root/group/"p1_s20260914"/"best.pth")}
  inherited=r9 if group=="V0" else r12;components=inherited["parameter_counts"];complete=components["encoder"]+components["visual"]+(0 if group=="V0" else components["force"])+params
  result["end_to_end_evidence"][group]={"applicable":True,"source":str(a.r9_v_cost if group=="V0" else a.r12_f_cost),"source_sha256":r18.sha(a.r9_v_cost if group=="V0" else a.r12_f_cost),"identity_match":{"source_checkpoint":True,"visual_checkpoint":True,"force_checkpoint":None if group=="V0" else True,"raw_image_dependency":"t-13..t same accepted path"},"complete_pipeline_parameters_with_round18_head":complete,"inherited_cold_history_median_ms":med_repeats(inherited,"cold_history"),"inherited_streaming_median_ms":med_repeats(inherited,"streaming"),"fresh_round18_head_latency_ms_median":result["cached_heads"][group]["latency_ms_median"],"interpretation":"inherited upstream path plus separately measured new head; not a new full-path speed claim"}
 r18.atomic_json(a.output,result);print(json.dumps(result,indent=2))
if __name__=="__main__":main()
