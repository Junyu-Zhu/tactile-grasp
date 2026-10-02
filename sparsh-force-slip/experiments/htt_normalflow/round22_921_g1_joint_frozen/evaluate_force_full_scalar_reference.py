#!/usr/bin/env python3
"""Scalar bootstrap reference snapshot for equivalence testing only."""
from __future__ import annotations
import argparse,csv,hashlib,json,math
from pathlib import Path
from collections import defaultdict
import numpy as np,torch
import train_f1 as T
import evaluate_f1_f2 as E
STRATA=("all","stable","transition","changing");PAIRS=(("K-VF","K-V"),("K-F","K-V"),("F2-half","K-VF"),("K-VF","hold"),("K-VF","ridge-K-VF"))
def sha(p):h=hashlib.sha256();h.update(Path(p).read_bytes());return h.hexdigest()
def write_csv(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if not rows:p.write_text("");return
 fields=list(dict.fromkeys(k for r in rows for k in r));
 with p.open("w",newline="") as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def masks(r):
 q=T.delta(r)[:,2].abs().amax(1);return {"all":torch.ones(len(q),dtype=torch.bool),"stable":q<=.25,"transition":(q>.25)&(q<1),"changing":q>=1}
def metric_rows(meta,role,r,name,pred):
 truth=T.delta(r);anchor=r["x"][:,-1,192:195];anchor_err=anchor-r["y_current"];delta_err=pred-truth;absolute_err=anchor_err[:,None,:]+delta_err;rows=[]
 for stratum,mask in masks(r).items():
  for hi,h in enumerate(T.HORIZONS):
   for ai,axis in enumerate(("x","y","z","all")):
    de=delta_err[mask,hi] if axis=="all" else delta_err[mask,hi,ai];ae=absolute_err[mask,hi] if axis=="all" else absolute_err[mask,hi,ai];anc=anchor_err[mask] if axis=="all" else anchor_err[mask,ai]
    if not mask.any():continue
    am=float((anc.square()).mean());dm=float(de.square().mean());cross=float(2*(anc*de).mean());total=float(ae.square().mean());rows.append({**meta,"role":role,"variant":name,"stratum":stratum,"horizon":h,"axis":axis,"endpoints":int(mask.sum()),"delta_mae_n":float(de.abs().mean()),"delta_rmse_n":float(de.square().mean().sqrt()),"absolute_future_mae_n":float(ae.abs().mean()),"absolute_future_rmse_n":float(ae.square().mean().sqrt()),"anchor_mse":am,"delta_mse":dm,"cross_2mean_anchor_delta":cross,"absolute_mse":total,"mse_decomposition_residual":total-am-dm-cross})
 return rows
def bootstrap(store,draws_n):
 rows=[]
 for fold in sorted({k[0] for k in store}):
  seeds=sorted({k[1] for k in store if k[0]==fold});groups=sorted(set(store[fold,seeds[0],"K-VF"]["leakage"]));rng=np.random.default_rng(221000+fold);draws=rng.integers(0,len(groups),size=(draws_n,len(groups)))
  for left,right in PAIRS:
   if any((fold,s,left) not in store or (fold,s,right) not in store for s in seeds):continue
   for hi,h in enumerate(T.HORIZONS):
    for st in STRATA:
     for kind in ("delta","absolute_future"):
      vals=[]
      for ids in draws:
       draw_weights=np.bincount(ids,minlength=len(groups));sv=[]
       for seed in seeds:
        l=store[fold,seed,left];r=store[fold,seed,right];mask=np.ones(len(l["stratum"]),bool) if st=="all" else l["stratum"]==st;target=l["truth"] if kind=="delta" else l["future"];lp=l["pred_delta"] if kind=="delta" else l["pred_future"];rp=r["pred_delta"] if kind=="delta" else r["pred_future"];err=np.abs(lp[:,hi]-target[:,hi]).mean(1)-np.abs(rp[:,hi]-target[:,hi]).mean(1);sums=np.asarray([err[mask&(l["leakage"]==g)].sum() for g in groups]);counts=np.asarray([(mask&(l["leakage"]==g)).sum() for g in groups]);den=(counts*draw_weights).sum()
        if not den:break
        sv.append(float((sums*draw_weights).sum()/den))
       if len(sv)==len(seeds):vals.append(float(np.mean(sv)))
      rows.append({"fold":fold,"left":left,"right":right,"horizon":h,"stratum":st,"metric":kind+"_mae","draws":draws_n,"valid":len(vals),"invalid":draws_n-len(vals),"mean":float(np.mean(vals)) if vals else None,"q025":float(np.quantile(vals,.025)) if vals else None,"q975":float(np.quantile(vals,.975)) if vals else None})
 return rows
def main():
 p=argparse.ArgumentParser();p.add_argument("--inventory",type=Path);p.add_argument("--smoke-root",type=Path);p.add_argument("--smoke-data",type=Path);p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cpu");p.add_argument("--bootstrap-draws",type=int,default=2000);a=p.parse_args();runs=[]
 if a.smoke_root:
  for g in T.GROUPS:runs.append({"package":"F1","group":g,"fold":1,"seed":20260914,"data":str(a.smoke_data),"output":str(a.smoke_root/"f1"/g)})
 else:runs=[r for r in json.loads(a.inventory.read_text())["runs"] if r["package"]=="F1"]
 metrics=[];trials=[];store={};bygrid=defaultdict(dict);index=[]
 for rr in runs:
  out=Path(rr["output"]);s=json.loads((out/"summary.json").read_text());cm=json.loads((out/"COMMIT.json").read_text());bp=out/cm["best"]["path"];ck=torch.load(bp,map_location="cpu",weights_only=False)
  ident=s["identity"];expected={"group":rr["group"],"fold":rr["fold"],"seed":rr["seed"],"data_sha256":sha(rr["data"]),"source_sha256":rr.get("trainer_sha256",sha(T.__file__)),**rr.get("dependencies",{})}
  if not a.smoke_root:expected.update({"protocol_sha256":rr["protocol_sha256"],"formal":True})
  else:expected["formal"]=False
  if s["status"]!="complete" or ck["identity"]!=ident or sha(bp)!=cm["best"]["sha256"] or any(ident.get(k)!=v for k,v in expected.items()):raise ValueError("run acceptance")
  d=torch.load(rr["data"],map_location="cpu",weights_only=False);T.validate(d,rr["data"],rr["fold"],rr["seed"],Path(T.__file__).with_name("G1_PREPARE.json"));m=T.init_model(rr["group"],rr["seed"]);m.load_state_dict(ck["model"]);m.to(a.device);n=ck["normalizer"];ridge=E.ridge_fit(d["roles"]["fit"],rr["group"]);meta={k:rr[k] for k in ("group","fold","seed")};index.append({**meta,"data_sha256":sha(rr["data"]),"best_sha256":sha(bp),"summary_sha256":sha(out/"summary.json")})
  for role,r in d["roles"].items():
   raw=T.predict(m,T.normalize_x(r["x"],n,rr["group"]),n,a.device);variants={rr["group"]:raw,"hold":torch.zeros_like(raw),f"ridge-{rr['group']}":E.ridge_predict(r,rr["group"],ridge)}
   if rr["group"]=="K-VF":variants["F2-half"]=.5*raw
   for name,pred in variants.items():metrics.extend(metric_rows(meta,role,r,name,pred))
   if role=="validation":
    truth=T.delta(r).numpy();anchor=r["x"][:,-1,192:195].numpy();q=np.max(np.abs(truth[:,2]),1);strata=np.where(q<=.25,"stable",np.where(q>=1,"changing","transition"));leak=np.asarray(r["leakage_group"],object)
    for name,pred in variants.items():
     pn=pred.numpy();store[rr["fold"],rr["seed"],name]={"leakage":leak,"stratum":strata,"truth":truth,"future":r["y"].numpy(),"pred_delta":pn,"pred_future":anchor[:,None,:]+pn}
     for ep in sorted(set(r["episode_id"])):
      mask=np.asarray(r["episode_id"],object)==ep;trials.append({**meta,"variant":name,"episode":ep,"leakage_group":leak[mask][0],"endpoints":int(mask.sum()),"delta_mae_n":float(np.abs(pn[mask]-truth[mask]).mean()),"absolute_future_mae_n":float(np.abs(anchor[mask,None,:]+pn[mask]-r["y"][mask].numpy()).mean())})
 ci=bootstrap(store,a.bootstrap_draws);cases=[]
 for (fold,seed,name),z in store.items():
  if name not in ("K-VF","F2-half"):continue
  base=store.get((fold,seed,"K-VF"));
  if name=="F2-half":
    for g in sorted(set(z["leakage"])):
     mask=(z["leakage"]==g)&(z["stratum"]=="stable")
     if mask.any():cases.append({"fold":fold,"seed":seed,"leakage_group":g,"rule_family":"F2-half_vs_K-VF","stable_endpoints":int(mask.sum()),"stable_delta_mae_difference":float(np.abs(z["pred_delta"][mask]-z["truth"][mask]).mean()-np.abs(base["pred_delta"][mask]-base["truth"][mask]).mean())})
 selected=[]
 if cases:
  selected=[{"rule":"largest_F2_stable_improvement","record":sorted(cases,key=lambda x:(x["stable_delta_mae_difference"],x["fold"],x["seed"],x["leakage_group"]))[0]},{"rule":"largest_F2_stable_worsening","record":sorted(cases,key=lambda x:(-x["stable_delta_mae_difference"],x["fold"],x["seed"],x["leakage_group"]))[0]}]
 max_res=max(abs(r["mse_decomposition_residual"]) for r in metrics);a.output.mkdir(parents=True,exist_ok=False);write_csv(a.output/"METRICS.csv",metrics);write_csv(a.output/"TRIAL_METRICS.csv",trials);write_csv(a.output/"PAIRED_CI.csv",ci);write_csv(a.output/"CASE_CANDIDATES.csv",cases);(a.output/"SELECTED_CASES.json").write_text(json.dumps({"ties":"fold,seed,leakage_group lexical","selected":selected,"source_images":"full reporting step only"},indent=2)+"\n");(a.output/"INPUT_INDEX.json").write_text(json.dumps(index,indent=2)+"\n");summary={"schema":"round22_force_evaluation_v1","status":"entrypoint_smoke_complete" if a.smoke_root else "core_tables_complete_full_reporting_pending","runs":len(runs),"metric_rows":len(metrics),"trial_rows":len(trials),"ci_rows":len(ci),"bootstrap_draws":a.bootstrap_draws,"max_mse_decomposition_residual":max_res,"mse_identity":"e_abs=e_anchor+e_delta; MSE=anchor_MSE+delta_MSE+2mean(anchor_error*delta_error)","mae_additive_decomposition_claimed":False,"strata":"10-frame max-axis <=0.25N stable, >=1N changing, otherwise transition","test_consumed":False};(a.output/"SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary))
if __name__=="__main__":main()
