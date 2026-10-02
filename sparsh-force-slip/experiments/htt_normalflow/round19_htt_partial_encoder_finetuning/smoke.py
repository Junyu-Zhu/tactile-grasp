#!/usr/bin/env python3
"""Gradient, initialization, freezing, force parity, and real nine-step cost smoke."""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np,torch
from torch import nn
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import train as tr
sys.path.insert(0,str(HERE.parent/"round5_force_conditioned_slip"/"current"));from models import ForceAdapter;from sources import load_decoupled_decoder
sys.path.insert(0,str(HERE.parent));import adapters

def th(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def ph(mod):return {n:th(p) for n,p in mod.named_parameters()}
def main():
 p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True);p.add_argument("--prefix-index",type=Path,required=True);p.add_argument("--source",type=Path,required=True);p.add_argument("--visual-checkpoint",type=Path,required=True);p.add_argument("--force-checkpoint",type=Path,required=True);p.add_argument("--force-predictions",type=Path,required=True);p.add_argument("--split",type=Path,required=True);p.add_argument("--device",default="cuda:0");p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True);torch.manual_seed(20260914);torch.cuda.manual_seed_all(20260914)
 d=tr.Data(a.data,a.prefix_index);primary,norm,weights,_=tr.fit_assets(d.d);ids=primary[:64];z,f=d.batch("fit",ids);f=(f-norm["mean"])/norm["std"]
 v=tr.Model("V2",20260914,a.source,a.visual_checkpoint).to(a.device);m=tr.Model("M2",20260914,a.source,a.visual_checkpoint).to(a.device)
 init_equal={name:all(torch.equal(x,y) for x,y in zip(getattr(v,name).state_dict().values(),getattr(m,name).state_dict().values())) for name in ("blocks","norm","pooler","trunk")}
 init_equal.update({"head_"+name:all(torch.equal(x,y) for x,y in zip(getattr(v.head,name).state_dict().values(),getattr(m.head,name).state_dict().values())) for name in ("visual_ln","gru","risk")})
 with torch.no_grad():
  vx=v.visual(z[:2].to(a.device).reshape(-1,*z.shape[2:])).reshape(2,9,192);conditioned,gamma,beta=m.head.conditioned_visual(torch.cat((vx,f[:2].to(a.device)),-1));film_identity=float((conditioned-m.head.visual_ln(vx)).abs().max())
 before={"blocks":ph(m.blocks),"norm":ph(m.norm),"pooler":ph(m.pooler),"trunk":ph(m.trunk),"head":ph(m.head)};opt=torch.optim.AdamW([{"params":m.blocks.parameters(),"lr":1e-5},{"params":m.head.parameters(),"lr":1e-3}],weight_decay=1e-4);target=d.d["roles"]["fit"]["stage"].eq(2).float()[ids].to(a.device)
 step=[]
 for k in range(3):
  zz=z.detach().clone().to(a.device).requires_grad_(True);opt.zero_grad(set_to_none=True);logit=m(zz,f.to(a.device));loss=(nn.functional.binary_cross_entropy_with_logits(logit,target,reduction="none")*weights[ids].to(a.device)).mean();loss.backward();step.append({"step":k,"loss":float(loss),"prefix_input_grad_absmax":float(zz.grad.abs().max()),"film_hidden_grad_absmax":float(m.head.film_hidden.weight.grad.abs().max()),"film_out_grad_absmax":float(m.head.film_out.weight.grad.abs().max())});opt.step()
 after={"blocks":ph(m.blocks),"norm":ph(m.norm),"pooler":ph(m.pooler),"trunk":ph(m.trunk),"head":ph(m.head)}
 changed={k:sum(before[k][n]!=after[k][n] for n in before[k]) for k in before}
 # Accepted frozen force raw replay against the cached prediction used by Round 17.
 fp=json.loads(a.force_predictions.read_text());entry=fp["entries"][0];eid=entry["episode_id"];split=json.loads(a.split.read_text());row=next(x for x in split["episodes"] if x["id"]==eid);ep=adapters.load_htt(row["path"]);t=13;image=adapters.window(ep,t)["inputs"]["image"].unsqueeze(0).to(a.device)
 base,_=tr.p2.load_b_checkpoint(a.source,torch.device(a.device));base.encoder.eval().requires_grad_(False);decoder,_=load_decoupled_decoder(a.source);force=ForceAdapter(decoder).to(a.device);fck=torch.load(a.force_checkpoint,map_location="cpu",weights_only=False);force.load_state_dict(fck["model_state"]);force.eval().requires_grad_(False);mean=torch.as_tensor(fck["normalization"]["mean"],device=a.device);std=torch.as_tensor(fck["normalization"]["std"],device=a.device)
 with torch.inference_mode():raw_force=(force(base.encoder(image))*std+mean).cpu().numpy()[0]
 cached=np.load(entry["prediction_path"],allow_pickle=False)[t];force_err=float(np.max(np.abs(raw_force-cached)))
 # Real batch timing/memory includes nine cached prefixes, live suffix, frozen visual branch and new head.
 probe=tr.Model("M2",20260914,a.source,a.visual_checkpoint).to(a.device);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();beg=time.perf_counter();probe.zero_grad(set_to_none=True);loss=probe(z.to(a.device),f.to(a.device)).square().mean();loss.backward();torch.cuda.synchronize();elapsed=time.perf_counter()-beg;peak=torch.cuda.max_memory_allocated()
 result={"schema":"round19_gpu_smoke_v1","status":"pass","batch":64,"history_steps":9,"finite_three_step":all(np.isfinite(x["loss"]) for x in step),"steps":step,"initial_shared_module_equality":init_equal,"film_identity_max_abs":film_identity,"film_gamma_max_abs":float(gamma.abs().max()),"film_beta_max_abs":float(beta.abs().max()),"changed_parameter_tensor_counts":changed,"allowed_update_only":changed["blocks"]>0 and changed["head"]>0 and changed["norm"]==changed["pooler"]==changed["trunk"]==0,"frozen_norm_pooler_trunk_with_nonzero_input_gradient":changed["norm"]==changed["pooler"]==changed["trunk"]==0 and all(x["prefix_input_grad_absmax"]>0 for x in step),"force_raw_cache_max_abs_n":force_err,"force_path_frozen_detached":True,"real_batch_wall_seconds":elapsed,"examples_per_second":64/elapsed,"peak_cuda_allocated_bytes":peak,"source_encoder_forward":"prepare_tokens + blocks0..11 + norm, patch tokens; cache cut after block9 includes register","source_sha256":tr.sha(a.source),"visual_checkpoint_sha256":tr.sha(a.visual_checkpoint),"force_checkpoint_sha256":tr.sha(a.force_checkpoint),"data_sha256":tr.sha(a.data),"prefix_index_sha256":tr.sha(a.prefix_index)}
 if not all(init_equal.values()) or film_identity!=0 or not result["allowed_update_only"] or force_err>2e-5 or step[0]["film_hidden_grad_absmax"]!=0 or not any(x["film_hidden_grad_absmax"]>0 for x in step[1:]):result["status"]="fail"
 tr.atomic_json(a.output,result);print(json.dumps(result,indent=2))
 if result["status"]!="pass":raise SystemExit(1)
if __name__=="__main__":main()
