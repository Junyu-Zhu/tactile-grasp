#!/usr/bin/env python3
import argparse,json,time
from pathlib import Path
import numpy as np,torch
import multitask_train as mt
def forward(m,g,x):
 h=m.encode(x)
 if g=="S":return m.slip(h)
 if g=="F":return m.future(h)
 return m.slip(h),m.future(h)
def main():
 p=argparse.ArgumentParser();p.add_argument("--prepared",type=Path,required=True);p.add_argument("--formal-root",type=Path,required=True);p.add_argument("--r14-cost",type=Path,required=True);p.add_argument("--output",type=Path,default=Path(__file__).with_name("COST.json"));p.add_argument("--device",default="cuda:0");a=p.parse_args();d=torch.load(a.prepared,map_location="cpu",weights_only=False);x=((d["roles"]["validation"]["x"][:1]-d["normalizer"]["x_mean"])/d["normalizer"]["x_std"]).to(a.device);records={};lat={}
 for g in mt.GROUPS:
  ck=torch.load(a.formal_root/g/"p1_s20260914"/"best.pth",map_location="cpu",weights_only=False);m=mt.init_model(20260914);m.load_state_dict(ck["model"]);m.to(a.device).eval();torch.cuda.synchronize()
  with torch.no_grad():
   for _ in range(100):forward(m,g,x)
   torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();times=[]
   for _ in range(500):
    t=time.perf_counter();forward(m,g,x);torch.cuda.synchronize();times.append((time.perf_counter()-t)*1000)
  names=mt.active_names(g);n=sum(p.numel() for q in names for p in getattr(m,q).parameters());peak=torch.cuda.max_memory_allocated()-base;lat[g]=np.asarray(times);records[g]={"active_parameters":n,"canonical_parameters":sum(p.numel() for p in m.parameters()),"active_parameter_bytes_fp32":4*n,"latency_ms_median":float(np.median(times)),"latency_ms_p95":float(np.quantile(times,.95,method="linear")),"batch1_peak_incremental_allocated_bytes":int(peak),"samples":len(times),"best_checkpoint":str(a.formal_root/g/"p1_s20260914"/"best.pth")};del m;torch.cuda.empty_cache()
 r14=json.loads(a.r14_cost.read_text());up=r14["inherited_upstream_reference"];upn=up["encoder_parameters"]+up["force_parameters"]+up["visual_parameters"]
 for g in records:records[g]["complete_pipeline_parameters"]=upn+records[g]["active_parameters"];records[g]["complete_parameter_bytes_fp32_weights_only"]=4*records[g]["complete_pipeline_parameters"]
 out={"schema":"round17_cost_v1","status":"complete","device":torch.cuda.get_device_name(0),"cached_heads":records,"shared_savings":{"parameters_J_vs_separate_S_plus_F":records["S"]["active_parameters"]+records["F"]["active_parameters"]-records["J"]["active_parameters"],"parameter_fraction":1-records["J"]["active_parameters"]/(records["S"]["active_parameters"]+records["F"]["active_parameters"]),"latency_ms_median_sequential_S_plus_F":records["S"]["latency_ms_median"]+records["F"]["latency_ms_median"],"latency_ms_median_J":records["J"]["latency_ms_median"]},"inherited_upstream_reference":up,"inherited_r14_cost_sha256":mt.sha(a.r14_cost),"limitations":["Cached-head timing excludes preprocessing, H2D, MAE, visual/force upstream, disk IO, capture and control.","Full pipeline parameters are arithmetic from unchanged accepted upstream identities; full runtime peak was not remeasured.","Shared-GPU timing is descriptive and not an architecture speed guarantee."]};mt.atomic_json(out,a.output);print(json.dumps(out,indent=2))
if __name__=="__main__":main()
