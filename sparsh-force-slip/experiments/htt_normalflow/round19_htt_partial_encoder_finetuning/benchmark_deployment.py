#!/usr/bin/env python3
"""Measure raw-image full-window and incremental shared-prefix deployment cost."""
from __future__ import annotations
import argparse,json,time,sys
from pathlib import Path
import numpy as np,torch
import train as tr
sys.path.insert(0,str(tr.HERE.parent/"round5_force_conditioned_slip"/"current"));from models import ForceAdapter
sys.path.insert(0,str(tr.HERE.parent));import adapters
from sources import load_decoupled_decoder
def percentile(x,q):return float(np.quantile(np.asarray(x),q))
def main():
 p=argparse.ArgumentParser();p.add_argument("--prepared",type=Path,required=True);p.add_argument("--prefix-index",type=Path,required=True);p.add_argument("--source",type=Path,required=True);p.add_argument("--visual-checkpoint",type=Path,required=True);p.add_argument("--force-checkpoint",type=Path,required=True);p.add_argument("--run",type=Path,required=True);p.add_argument("--group",choices=tr.GROUPS,required=True);p.add_argument("--fold",type=int,required=True);p.add_argument("--seed",type=int,required=True);p.add_argument("--device",default="cuda:0");p.add_argument("--repeats",type=int,default=50);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True);device=torch.device(a.device)
 manifest=json.loads((a.run/"COMMIT.json").read_text());best=a.run/manifest["best"]["path"];ck=torch.load(best,map_location="cpu",weights_only=False);model=tr.Model(a.group,a.seed,a.source,a.visual_checkpoint);model.load_state_dict(ck["model"]);model.to(device).eval();norm=ck["normalizer"]
 base,_=tr.p2.load_b_checkpoint(a.source,device);enc=base.encoder.eval().requires_grad_(False);decoder,_=load_decoupled_decoder(a.source);force=ForceAdapter(decoder).to(device);fck=torch.load(a.force_checkpoint,map_location="cpu",weights_only=False);force.load_state_dict(fck["model_state"]);force.eval().requires_grad_(False);fmean=torch.as_tensor(fck["normalization"]["mean"],device=device);fstd=torch.as_tensor(fck["normalization"]["std"],device=device)
 data=torch.load(a.prepared,map_location="cpu",weights_only=False);d=data["roles"]["validation"];eid=d["episode_id"][0];t=int(d["t"][0]);idx=json.loads(a.prefix_index.read_text());entry=next(x for x in idx["entries"] if x["episode_id"]==eid);ep=adapters.load_htt(entry["source_path"]);images=torch.stack([adapters.window(ep,s)["inputs"]["image"] for s in range(t-8,t+1)]).to(device);current=images[-1:]
 def prefix(x):
  z=enc.prepare_tokens_with_masks(x)
  for block in enc.blocks[:10]:z=block(z)
  return z
 def force_suffix(z):
  for block in enc.blocks[10:]:z=block(z)
  return enc.norm(z)[:,enc.num_register_tokens:]
 def branches(z):
  visual=model.visual(z);pred=force(force_suffix(z))*fstd+fmean;return visual,(pred-norm["mean"].to(device))/norm["std"].to(device)
 def head(v,f):return torch.sigmoid(model.head.components(torch.cat((v,f),-1))["logit"])
 with torch.inference_mode():
  torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();z=prefix(images);v,f=branches(z);score=head(v[None] if v.ndim==2 and len(v)==9 else v,f[None] if f.ndim==2 and len(f)==9 else f);torch.cuda.synchronize();cold=time.perf_counter()-start;cold_peak=torch.cuda.max_memory_allocated()
  # Warm full-window endpoint and streaming new-step latency. Histories stay device resident.
  for _ in range(5):z=prefix(images);vv,ff=branches(z);head(vv.unsqueeze(0),ff.unsqueeze(0))
  full=[]
  for _ in range(a.repeats):
   torch.cuda.synchronize();start=time.perf_counter();z=prefix(images);vv,ff=branches(z);head(vv.unsqueeze(0),ff.unsqueeze(0));torch.cuda.synchronize();full.append(1000*(time.perf_counter()-start))
  vh=vv.clone();fh=ff.clone()
  for _ in range(5):z=prefix(current);nv,nf=branches(z);vh=torch.cat((vh[1:],nv));fh=torch.cat((fh[1:],nf));head(vh.unsqueeze(0),fh.unsqueeze(0))
  stream=[];torch.cuda.reset_peak_memory_stats()
  for _ in range(a.repeats):
   torch.cuda.synchronize();start=time.perf_counter();z=prefix(current);nv,nf=branches(z);vh=torch.cat((vh[1:],nv));fh=torch.cat((fh[1:],nf));head(vh.unsqueeze(0),fh.unsqueeze(0));torch.cuda.synchronize();stream.append(1000*(time.perf_counter()-start))
  stream_peak=torch.cuda.max_memory_allocated()
  head_only=[]
  for _ in range(a.repeats):
   torch.cuda.synchronize();start=time.perf_counter();head(vh.unsqueeze(0),fh.unsqueeze(0));torch.cuda.synchronize();head_only.append(1000*(time.perf_counter()-start))
 modules=[enc.patch_embed,enc.blocks[:10],enc.blocks[10:],enc.norm,force,model.blocks,model.norm,model.pooler,model.trunk,model.head];unique={id(p):p for m in modules for p in m.parameters()}
 result={"schema":"round19_deployment_cost_v1","status":"complete","group":a.group,"fold":a.fold,"seed":a.seed,"scope":"GPU compute from nine preprocessed raw image-pair tensors through shared frozen MAE blocks0..9, separate frozen source force suffix and fine-tuned visual suffix, R3 branch and temporal head; excludes disk decode/window construction","history_steps":9,"cold_first_full_window_ms":1000*cold,"warm_full_window_ms":{"median":percentile(full,.5),"p95":percentile(full,.95)},"warm_streaming_new_step_ms":{"median":percentile(stream,.5),"p95":percentile(stream,.95)},"cached_9step_head_only_ms":{"median":percentile(head_only,.5),"p95":percentile(head_only,.95)},"cold_peak_cuda_bytes":cold_peak,"stream_peak_cuda_bytes":stream_peak,"unique_full_path_parameters":sum(x.numel() for x in unique.values()),"trainable_round19_parameters":sum(x.numel() for x in model.parameters() if x.requires_grad),"best_commit_sha256":tr.sha(best),"force_checkpoint_sha256":tr.sha(a.force_checkpoint),"source_sha256":tr.sha(a.source),"test_consumed":False}
 tr.atomic_json(a.output,result);print(json.dumps(result,indent=2))
if __name__=="__main__":main()
