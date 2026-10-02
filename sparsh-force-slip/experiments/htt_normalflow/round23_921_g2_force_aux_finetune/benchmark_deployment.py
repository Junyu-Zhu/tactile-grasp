#!/usr/bin/env python3
"""Benchmark an actual E3 deployment path; auxiliary head is removed."""
from __future__ import annotations
import argparse,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent;R22=HERE.parent/"round22_921_g1_joint_frozen"
def load_train():
 s=importlib.util.spec_from_file_location("r23_cost_train",R22/"train_e3.py");m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m
tr=load_train();sys.path.insert(0,str(HERE.parent/"round5_force_conditioned_slip"/"current"));from models import ForceAdapter
sys.path.insert(0,str(HERE.parent));import adapters
from sources import load_decoupled_decoder
def pct(x,q):return float(np.quantile(np.asarray(x),q))
def main():
 p=argparse.ArgumentParser();p.add_argument("--inventory",type=Path,required=True);p.add_argument("--run",required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cuda:0");p.add_argument("--repeats",type=int,default=12);a=p.parse_args();rr=next(x for x in json.loads(a.inventory.read_text())["runs"] if x["run"]==a.run);run=Path(rr["output"]);cm=json.loads((run/"COMMIT.json").read_text());best=run/cm["best"]["path"];ck=torch.load(best,map_location="cpu",weights_only=False);device=torch.device(a.device)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True);model=tr.Model(rr["group"],rr["seed"],rr["source"],rr["visual_checkpoint"]);model.load_state_dict(ck["model"],strict=True);training=sum(p.numel() for p in model.parameters() if p.requires_grad);aux=0 if model.aux is None else sum(p.numel() for p in model.aux.parameters());model.aux=None;model.to(device).eval();norm=ck["normalizer"]
 prepared=torch.load(rr["data"],map_location="cpu",weights_only=False);upstream=prepared["provenance"]["immutable_upstream"];force_record=upstream["force_checkpoint"];d=prepared["roles"]["validation"];eid=d["episode_id"][0];t=int(d["t"][0]);idx=json.loads(Path(rr["prefix_index"]).read_text());entry=next(x for x in idx["entries"] if x["episode_id"]==eid);source_path=entry["source_path"];ep=adapters.load_htt(source_path);t=min(t,len(ep.images)-a.repeats-1)
 base,_=tr.p2.load_b_checkpoint(rr["source"],device);enc=base.encoder.eval().requires_grad_(False);force=None;fmean=fstd=None
 if rr["group"] in tr.FORCE_INPUT_GROUPS:
  decoder,_=load_decoupled_decoder(rr["source"]);force=ForceAdapter(decoder).to(device);fck=torch.load(force_record["path"],map_location="cpu",weights_only=False);force.load_state_dict(fck["model_state"]);force.eval().requires_grad_(False);fmean=torch.as_tensor(fck["normalization"]["mean"],device=device);fstd=torch.as_tensor(fck["normalization"]["std"],device=device)
 def prefix(x):
  z=enc.prepare_tokens_with_masks(x)
  for block in enc.blocks[:10]:z=block(z)
  return z
 def suffix(z):
  v=model.visual(z)
  if force is None:f=torch.zeros((len(z),3),device=device)
  else:
   q=z
   for block in enc.blocks[10:]:q=block(q)
   q=enc.norm(q)[:,enc.num_register_tokens:];pred=force(q)*fstd+fmean;f=(pred-norm["force_mean"].to(device))/norm["force_std"].to(device)
  return v,f
 def head(v,f):return torch.sigmoid(model.head.components(torch.cat((v,f),-1))["logit"])
 def pair_image(e,s):return torch.cat((adapters.preprocess(e.images[s],e.reference),adapters.preprocess(e.images[s-5],e.reference)))
 def raw_nine(e,endpoint):return torch.stack([pair_image(e,s) for s in range(endpoint-8,endpoint+1)])
 def compute(images):z=prefix(images);v,f=suffix(z);return head(v.unsqueeze(0),f.unsqueeze(0)),v,f
 locked=torch.stack([adapters.window(ep,s)["inputs"]["image"] for s in range(t-8,t+1)]);preprocess_parity=float((raw_nine(ep,t)-locked).abs().max())
 if preprocess_parity!=0:raise ValueError("preprocess parity")
 with torch.inference_mode():
  torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tic=time.perf_counter();cold_ep=adapters.load_htt(source_path);cold_score,_,_=compute(raw_nine(cold_ep,t).to(device));torch.cuda.synchronize();cold=1000*(time.perf_counter()-tic);coldmem=torch.cuda.max_memory_allocated();full=[]
  for _ in range(a.repeats):
   torch.cuda.synchronize();tic=time.perf_counter();compute(raw_nine(ep,t).to(device));torch.cuda.synchronize();full.append(1000*(time.perf_counter()-tic))
  _,vh,fh=compute(raw_nine(ep,t).to(device));stream=[];parity=[];torch.cuda.reset_peak_memory_stats()
  for _ in range(a.repeats):
   step=t+len(stream)+1;torch.cuda.synchronize();tic=time.perf_counter();nv,nf=suffix(prefix(pair_image(ep,step).unsqueeze(0).to(device)));vh=torch.cat((vh[1:],nv));fh=torch.cat((fh[1:],nf));stream_score=head(vh.unsqueeze(0),fh.unsqueeze(0));torch.cuda.synchronize();stream.append(1000*(time.perf_counter()-tic));direct,_,_=compute(raw_nine(ep,step).to(device));parity.append(float((stream_score-direct).abs().max()))
  if max(parity)>1e-5:raise ValueError("streaming/direct score mismatch")
  streammem=torch.cuda.max_memory_allocated()
 enc_used=list(enc.parameters()) if force is not None else [p for n,p in enc.named_parameters() if not n.startswith("blocks.10") and not n.startswith("blocks.11") and not n.startswith("norm.")];deployed_params=enc_used+[p for m in (model.blocks,model.norm,model.pooler,model.trunk,model.head) for p in m.parameters()]+([] if force is None else list(force.parameters()));unique={id(p):p for p in deployed_params};smi=subprocess.run(["nvidia-smi","--query-gpu=index,uuid,pci.bus_id,memory.used","--format=csv,noheader,nounits"],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);out={"schema":"round23_e3_deployment_cost_v2","status":"complete","run":a.run,"group":rr["group"],"scope":"resident models; real HTT NPZ decode for cold window (OS page cache), reference preprocessing and (s,s-5) pairs, H2D, shared blocks0..9, fine-tuned visual suffix and slip head; C/D also execute frozen R10 force suffix","auxiliary_head_deployed":False,"auxiliary_module_removed_before_device_residency":True,"cold_first_full_window_with_episode_decode_ms":cold,"warm_full_window_episode_resident_ms":{"median":pct(full,.5),"p95":pct(full,.95),"repeats":len(full)},"warm_streaming_new_step_episode_and_history_resident_ms":{"median":pct(stream,.5),"p95":pct(stream,.95),"repeats":len(stream)},"cold_peak_cuda_bytes":coldmem,"streaming_peak_cuda_bytes":streammem,"preprocess_pair_max_abs_vs_locked_adapter":preprocess_parity,"streaming_direct_score_max_abs":max(parity),"complete_deployed_parameters":sum(p.numel() for p in unique.values()),"parameter_count_scope":"all actually executed encoder parameters including direct token/position/register parameters, fine-tuned visual branch and temporal head; C/D add source force suffix/adapter; identity-deduplicated","training_trainable_parameters":training,"training_only_auxiliary_parameters":aux,"best_commit_sha256":cm["best"]["sha256"],"force_checkpoint_sha256":force_record["sha256"] if force is not None else None,"gpu_device_name":torch.cuda.get_device_name(0),"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"nvidia_smi":smi.stdout,"shared_gpu_variability":"Measured on shared GPU; distributions include concurrent-load variation.","source_path":source_path,"source_sha256":tr.sha(source_path),"test_consumed":False};tr.atomic_json(a.output,out);print(json.dumps(out))
if __name__=="__main__":main()
