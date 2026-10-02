#!/usr/bin/env python3
"""Supplemental R19 raw-episode decode/preprocess/H2D/dual-path latency measurement."""
from __future__ import annotations
import argparse,json,subprocess,time,sys
from pathlib import Path
import numpy as np,torch
import train as tr
sys.path.insert(0,str(tr.HERE.parent/"round5_force_conditioned_slip"/"current"));from models import ForceAdapter
sys.path.insert(0,str(tr.HERE.parent));import adapters
from sources import load_decoupled_decoder

def distribution(values):
 return {"median":float(np.median(values)),"p95":float(np.quantile(values,.95)),"min":float(np.min(values)),"max":float(np.max(values)),"repeats":len(values)}

def main():
 p=argparse.ArgumentParser();p.add_argument("--prepared",type=Path,required=True);p.add_argument("--prefix-index",type=Path,required=True);p.add_argument("--source",type=Path,required=True);p.add_argument("--visual-checkpoint",type=Path,required=True);p.add_argument("--force-checkpoint",type=Path,required=True);p.add_argument("--run",type=Path,required=True);p.add_argument("--group",choices=tr.GROUPS,required=True);p.add_argument("--fold",type=int,required=True);p.add_argument("--seed",type=int,required=True);p.add_argument("--device",default="cuda:0");p.add_argument("--repeats",type=int,default=12);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 if a.repeats<3:raise ValueError("at least three repeats required")
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True);device=torch.device(a.device)
 manifest=json.loads((a.run/"COMMIT.json").read_text());best=a.run/manifest["best"]["path"]
 if tr.sha(best)!=manifest["best"]["sha256"]:raise ValueError("best commit hash")
 setup_start=time.perf_counter();ck=torch.load(best,map_location="cpu",weights_only=False);model=tr.Model(a.group,a.seed,a.source,a.visual_checkpoint);model.load_state_dict(ck["model"]);model.to(device).eval();norm=ck["normalizer"]
 base,_=tr.p2.load_b_checkpoint(a.source,device);enc=base.encoder.eval().requires_grad_(False);decoder,_=load_decoupled_decoder(a.source);force=ForceAdapter(decoder).to(device);fck=torch.load(a.force_checkpoint,map_location="cpu",weights_only=False);force.load_state_dict(fck["model_state"]);force.eval().requires_grad_(False);fmean=torch.as_tensor(fck["normalization"]["mean"],device=device);fstd=torch.as_tensor(fck["normalization"]["std"],device=device);torch.cuda.synchronize();model_setup_ms=1000*(time.perf_counter()-setup_start)
 data=torch.load(a.prepared,map_location="cpu",weights_only=False);role=data["roles"]["validation"];episode_id=role["episode_id"][0];t=int(role["t"][0]);entries=json.loads(a.prefix_index.read_text())["entries"];entry=next(x for x in entries if x["episode_id"]==episode_id);source_path=entry["source_path"]
 def pair_image(ep,s):return torch.cat((adapters.preprocess(ep.images[s],ep.reference),adapters.preprocess(ep.images[s-5],ep.reference)))
 def raw_nine(ep,endpoint):return torch.stack([pair_image(ep,s) for s in range(endpoint-8,endpoint+1)])
 def prefix(x):
  z=enc.prepare_tokens_with_masks(x)
  for block in enc.blocks[:10]:z=block(z)
  return z
 def branches(z):
  visual=model.visual(z)
  force_z=z
  for block in enc.blocks[10:]:force_z=block(force_z)
  force_z=enc.norm(force_z)[:,enc.num_register_tokens:]
  predicted=force(force_z)*fstd+fmean
  return visual,(predicted-norm["mean"].to(device))/norm["std"].to(device)
 def head(v,f):return torch.sigmoid(model.head.components(torch.cat((v,f),-1))["logit"])
 def compute(images):
  z=prefix(images);v,f=branches(z);score=head(v.unsqueeze(0),f.unsqueeze(0));return score,v,f
 ep=adapters.load_htt(source_path)
 t=min(t,len(ep.images)-a.repeats-1)
 if t<13 or t>=len(ep.images):raise ValueError("selected endpoint cannot form nine base image pairs")
 direct=raw_nine(ep,t);independent=torch.stack([adapters.window(ep,s)["inputs"]["image"] for s in range(t-8,t+1)])
 parity=float((direct-independent).abs().max());
 if parity!=0.:raise ValueError(f"raw pair construction mismatch: {parity}")
 device_name=torch.cuda.get_device_name(0);torch.cuda.reset_peak_memory_stats()
 with torch.inference_mode():
  # Models are resident. Cold input loads/decompresses the episode, preprocesses all nine
  # image pairs with the same reference, copies them to GPU, then runs both suffixes/head.
  torch.cuda.synchronize();start=time.perf_counter();cold_episode=adapters.load_htt(source_path);cold_images=raw_nine(cold_episode,t).to(device);cold_score,_,_=compute(cold_images);torch.cuda.synchronize();cold_ms=1000*(time.perf_counter()-start);cold_peak=torch.cuda.max_memory_allocated()
  warm=[]
  for _ in range(a.repeats):
   torch.cuda.synchronize();start=time.perf_counter();images=raw_nine(ep,t).to(device);compute(images);torch.cuda.synchronize();warm.append(1000*(time.perf_counter()-start))
  # Streaming retains the decoded episode and eight prior model outputs. Each timed
  # step constructs the new raw image/reference pair, copies it, runs prefix and both
  # suffixes, updates the nine-step history, and executes the task head.
  init=raw_nine(ep,t).to(device);_,visual_history,force_history=compute(init);stream=[];torch.cuda.reset_peak_memory_stats()
  parity_scores=[]
  for j in range(a.repeats):
   step_t=t+j+1;torch.cuda.synchronize();start=time.perf_counter();current=pair_image(ep,step_t).unsqueeze(0).to(device);z=prefix(current);new_visual,new_force=branches(z);visual_history=torch.cat((visual_history[1:],new_visual));force_history=torch.cat((force_history[1:],new_force));stream_score=head(visual_history.unsqueeze(0),force_history.unsqueeze(0));torch.cuda.synchronize();stream.append(1000*(time.perf_counter()-start))
   direct_score,_,_=compute(raw_nine(ep,step_t).to(device));parity_scores.append(float((stream_score-direct_score).abs().max()))
  if max(parity_scores)>1e-5:raise ValueError(f"streaming/direct score mismatch: {max(parity_scores)}")
  streaming_peak=torch.cuda.max_memory_allocated()
 smi=subprocess.run(["nvidia-smi","--query-gpu=index,uuid,pci.bus_id,memory.used","--format=csv,noheader,nounits"],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 result={"schema":"round19_end_to_end_cost_supplement_v1","status":"complete","group":a.group,"fold":a.fold,"seed":a.seed,"scope":"Validated raw HTT NPZ episode decode/reference transform and nine (s,s-5) pairs, CPU-to-GPU copy, shared frozen MAE prefix, separate frozen force and fine-tuned visual suffixes, temporal head in one synchronized timing interval. Model resident; file was previously loaded, so cold first window includes decode from OS page cache and is not a disk-cold measurement.","cold_first_full_window_with_episode_decode_ms":cold_ms,"warm_full_window_episode_resident_ms":distribution(warm),"warm_streaming_new_step_episode_and_history_resident_ms":distribution(stream),"model_setup_before_latency_ms":model_setup_ms,"cold_peak_cuda_bytes":cold_peak,"streaming_peak_cuda_bytes":streaming_peak,"preprocess_pair_max_abs_vs_locked_adapter":parity,"streaming_direct_score_max_abs":max(parity_scores),"history_steps":9,"episode_id":episode_id,"endpoint_t":t,"source_path":source_path,"source_sha256":tr.sha(source_path),"best_commit_sha256":tr.sha(best),"force_checkpoint_sha256":tr.sha(a.force_checkpoint),"gpu_device_name":device_name,"cuda_visible_devices":__import__("os").environ.get("CUDA_VISIBLE_DEVICES"),"nvidia_smi":smi.stdout,"shared_gpu_variability":"Measured on shared GPU; distributions include concurrent-load variation.","test_consumed":False};tr.atomic_json(a.output,result);print(json.dumps(result,indent=2))
if __name__=="__main__":main()
