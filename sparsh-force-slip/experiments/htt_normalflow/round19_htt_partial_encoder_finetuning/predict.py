#!/usr/bin/env python3
"""Export locked-role predictions and M2 diagnostics from one accepted best commit."""
from __future__ import annotations
import argparse,csv,json,math,tempfile,os
from pathlib import Path
import numpy as np
import torch
import train as tr

def write_csv(path,rows):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fields=list(dict.fromkeys(k for r in rows for k in r));fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=path.name+".",suffix=".tmp");os.close(fd)
 try:
  with open(tmp,"w",newline="") as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)

def main():
 p=argparse.ArgumentParser();p.add_argument("--prepared",type=Path,required=True);p.add_argument("--prefix-index",type=Path,required=True);p.add_argument("--source",type=Path,required=True);p.add_argument("--visual-checkpoint",type=Path,required=True);p.add_argument("--run",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--group",choices=tr.GROUPS,required=True);p.add_argument("--fold",type=int,required=True);p.add_argument("--seed",type=int,required=True);p.add_argument("--device",default="cuda:0");a=p.parse_args()
 summary=json.loads((a.run/"summary.json").read_text());manifest=json.loads((a.run/"COMMIT.json").read_text());best=a.run/manifest["best"]["path"]
 if summary.get("status")!="complete" or tr.sha(best)!=manifest["best"]["sha256"]:raise ValueError("unaccepted formal run")
 ck=torch.load(best,map_location="cpu",weights_only=False);identity=ck["identity"]
 expected={"group":a.group,"fold":a.fold,"seed":a.seed,"data_sha256":tr.sha(a.prepared),"prefix_index_sha256":tr.sha(a.prefix_index),"source_sha256":tr.sha(a.source),"visual_checkpoint_sha256":tr.sha(a.visual_checkpoint),"source_sha256_code":tr.sha(Path(tr.__file__)),"protocol_sha256":tr.sha(Path(tr.__file__).resolve().parent/"PROTOCOL.md"),"formal":True}
 if any(identity.get(k)!=v for k,v in expected.items()):raise ValueError("prediction identity mismatch")
 data=tr.Data(a.prepared,a.prefix_index);model=tr.Model(a.group,a.seed,a.source,a.visual_checkpoint);model.load_state_dict(ck["model"]);model.to(a.device).eval();norm=ck["normalizer"];arrays={};film_rows=[];anomaly={};force_error={}
 with torch.inference_mode():
  for role in tr.ROLES:
   normal=[];mean=[];lag=[];off=[];gammas=[];betas=[];ratios=[]
   n=len(data.d["roles"][role]["t"])
   for ids in torch.arange(n).split(tr.BATCH):
    z,f=data.batch(role,ids);f=((f-norm["mean"])/norm["std"]).to(a.device);v=model.visual(z.reshape(-1,*z.shape[2:]).to(a.device)).reshape(len(ids),9,192);comp=model.head.components(torch.cat((v,f),-1));normal.append(torch.sigmoid(comp["logit"]).cpu())
    if role=="validation" and a.group=="M2":
     fm=torch.zeros_like(f);fl=torch.zeros_like(f);fl[:,1:]=f[:,:-1]
     mean.append(torch.sigmoid(model.head.components(torch.cat((v,fm),-1))["logit"]).cpu());lag.append(torch.sigmoid(model.head.components(torch.cat((v,fl),-1))["logit"]).cpu());base=model.head.visual_ln(v);off.append(torch.sigmoid(model.head.risk(model.head.gru(base)[0][:,-1]).squeeze(-1)).cpu())
    if a.group=="M2":
     gamma,beta=comp["gamma"].cpu(),comp["beta"].cpu();base=model.head.visual_ln(v).cpu();mod=(1+gamma)*base+beta;gammas.append(gamma);betas.append(beta);ratios.append(mod.norm(dim=-1)/(base.norm(dim=-1)+1e-12))
   arrays[f"{role}_score"]=torch.cat(normal).numpy()
   if role=="validation" and a.group=="M2":arrays["validation_force_mean"]=torch.cat(mean).numpy();arrays["validation_force_lag1"]=torch.cat(lag).numpy();arrays["validation_film_off"]=torch.cat(off).numpy()
   if a.group=="M2":
    g=torch.cat(gammas).numpy();b=torch.cat(betas).numpy();ratio=torch.cat(ratios).numpy();flag=((np.abs(g)>=.5)|(1+g<.5)|(1+g>1.5)|(np.abs(b)>=3)).any((1,2))|(ratio>2).any(1);anomaly[role]=flag
    d=data.d["roles"][role];force_error[role]=np.abs(d["x"][:,-1,192:195].numpy()-d["y_current"].numpy()).mean(1)
    for ti in range(9):
     for feature in range(192):
      gg=g[:,ti,feature];bb=b[:,ti,feature];film_rows.append({"group":a.group,"fold":a.fold,"seed":a.seed,"role":role,"time_index":ti,"feature":feature,"endpoints":len(gg),"gamma_mean":float(gg.mean()),"gamma_abs_q95":float(np.quantile(np.abs(gg),.95)),"gamma_abs_max":float(np.abs(gg).max()),"beta_mean":float(bb.mean()),"beta_abs_q95":float(np.quantile(np.abs(bb),.95)),"beta_abs_max":float(np.abs(bb).max()),"scale_min":float((1+gg).min()),"scale_max":float((1+gg).max())})
    arrays[f"{role}_anomaly"]=flag;arrays[f"{role}_norm_ratio"]=ratio;arrays[f"{role}_force_error_mae_n"]=force_error[role]
 a.output.mkdir(parents=True,exist_ok=True);np.savez(a.output/"PREDICTIONS.npz",**arrays)
 if film_rows:write_csv(a.output/"FILM_FEATURE_TIME_ROLE.csv",film_rows)
 output={"schema":"round19_prediction_export_v1","status":"complete","group":a.group,"fold":a.fold,"seed":a.seed,"best_commit":str(best),"best_commit_sha256":tr.sha(best),"formal_summary_sha256":tr.sha(a.run/"summary.json"),"prediction_sha256":tr.sha(a.output/"PREDICTIONS.npz"),"film_stats_sha256":tr.sha(a.output/"FILM_FEATURE_TIME_ROLE.csv") if film_rows else None,"rows":{r:len(data.d["roles"][r]["t"]) for r in tr.ROLES},"test_consumed":False}
 tr.atomic_json(a.output/"SUMMARY.json",output);print(json.dumps(output,indent=2))
if __name__=="__main__":main()
