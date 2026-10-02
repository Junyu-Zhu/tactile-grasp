#!/usr/bin/env python3
"""Round-17 frozen-checkpoint prediction, full R13 working points, future metrics, paired CIs and cases."""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,math,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import Ridge
import multitask_train as mt

ROLES=("fit","selection","calibration","validation");SLIP_GROUPS=("S","J");FUTURE_GROUPS=("F","J");ALPHAS=(.01,.05,.10);KS=(1,2,4);FAMILIES=("trial_macro_static_FPR","trial_any_static_alarm_rate")
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.generic):x=x.item()
 if isinstance(x,float) and not math.isfinite(x):return None
 return x
def write_csv(path,rows):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if not rows:path.write_text("");return
 fields=list(dict.fromkeys(k for r in rows for k in r));tmp=path.with_suffix(path.suffix+".tmp")
 with tmp.open("w",newline="") as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(clean(rows))
 tmp.replace(path)
def load_r13(path):
 spec=importlib.util.spec_from_file_location("r17_r13",path);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
def episodes(r13,d,scores):
 out=[]
 for eid in sorted(set(d["episode_id"])):
  ids=np.asarray([i for i,e in enumerate(d["episode_id"]) if e==eid]);order=np.argsort(d["t"][ids].numpy());ids=ids[order];group=d["leakage_group"][int(ids[0])];probe=d["probe"][int(ids[0])]
  out.append(r13.Episode(eid,group,probe,d["t"][ids].numpy(),d["stage"][ids].numpy(),scores[ids]))
 return out
def infer(model,d,norm,device):
 x=(d["x"]-norm["x_mean"])/norm["x_std"];ss=[];ff=[];model.eval()
 with torch.no_grad():
  for ids in torch.arange(len(x)).split(1024):
   s,f=model(x[ids].to(device));ss.append(torch.sigmoid(s).cpu());ff.append((f.cpu()*norm["y_std"]+norm["y_mean"]))
 return torch.cat(ss).numpy(),torch.cat(ff).numpy()
def maxba(r13,cal):
 best=None
 for th in r13.candidate_thresholds(cal,1):
  _,a=r13.evaluate_policy(cal,float(th),1);key=(a["balanced_accuracy"],-a["frame_static_FPR"],float(th))
  if best is None or key>best[0]:best=(key,float(th),a)
 return {"status":"fit","threshold":best[1],"never_alarm":bool(best[1]>1),"constraint_value":best[2]["frame_static_FPR"],"calibration_gross_recall":best[2]["gross_recall"]}
def point_rows(r13,eps_by_role,meta):
 cal=eps_by_role["calibration"];rows=[];trials=[];thresholds=[]
 defs=[("fixed_0.5",1,.5,"fixed"),("maxBA",1,maxba(r13,cal)["threshold"],"calibration_maxBA")]
 cache1=r13.build_fit_cache(cal,1)
 for a in ALPHAS:
  ft=r13.fit_cached(cache1,FAMILIES[0],a,constraint_override="frame_static_FPR");defs.append((f"FPR{a:.2f}",1,ft["threshold"],"calibration_frame_FPR"))
 for name,k,th,selection in defs:
  thresholds.append({**meta,"policy":name,"family":"historical","k":k,"threshold":th,"selection":selection,"never_alarm":th>1})
  useks=(1,2) if name in ("fixed_0.5","maxBA") or name.startswith("FPR") else (k,)
  for uk in useks:
   rule="raw" if uk==1 else "confirm2";pname=f"{name}|{rule}"
   for role,eps in eps_by_role.items():
    ts,agg=r13.evaluate_policy(eps,th,uk);rank=r13.rank_metrics(eps);rows.append({**meta,"role":role,"policy":pname,"family":"historical","k":uk,"threshold":th,"never_alarm":th>1,**agg,**rank});trials.extend({**meta,"role":role,"policy":pname,"threshold":th,"k":uk,**x} for x in ts)
 for family in FAMILIES:
  for a in ALPHAS:
   for k in KS:
    fit=r13.fit_cached(r13.build_fit_cache(cal,k),family,a);pname=r13.policy_name(family,a,k);thresholds.append({**meta,"policy":pname,"family":family,"alpha":a,"k":k,**fit})
    if fit["status"]!="fit":continue
    for role,eps in eps_by_role.items():
     ts,agg=r13.evaluate_policy(eps,fit["threshold"],k);rank=r13.rank_metrics(eps);rows.append({**meta,"role":role,"policy":pname,"family":family,"alpha":a,"k":k,"threshold":fit["threshold"],"never_alarm":fit["never_alarm"],**agg,**rank});trials.extend({**meta,"role":role,"policy":pname,"threshold":fit["threshold"],"k":k,**x} for x in ts)
 for a in ALPHAS:
  th=next(x[2] for x in defs if x[0]==f"FPR{a:.2f}");pname=f"FPR{a:.2f}|confirm4_reference";thresholds.append({**meta,"policy":pname,"family":"confirm4_reference","alpha":a,"k":4,"threshold":th,"never_alarm":th>1})
  for role,eps in eps_by_role.items():
   ts,agg=r13.evaluate_policy(eps,th,4);rank=r13.rank_metrics(eps);rows.append({**meta,"role":role,"policy":pname,"family":"confirm4_reference","alpha":a,"k":4,"threshold":th,"never_alarm":th>1,**agg,**rank});trials.extend({**meta,"role":role,"policy":pname,"threshold":th,"k":4,**x} for x in ts)
 return rows,trials,thresholds
def strata(data,role):
 q=data["joint_weight"]["fit_change_strata"];m=(role["y"][:,2]-role["y_current"]).abs().amax(1).numpy();stable=m<=q["stable_le_n"];changing=(m>=q["changing_ge_n"])&(~stable);out=np.full(len(m),"transition",object);out[stable]="stable";out[changing]="changing";return out
def metric_cell(pred,y,ycur,pcur,mask):
 p=pred[mask];g=y[mask];gc=ycur[mask];pc=pcur[mask];e=p-g;deployed=(p-pc)-(g-gc);anchor=pc-gc;reschg=deployed
 return {"n":int(mask.sum()),"future_mae":float(np.abs(e).mean()),"future_rmse":float(np.sqrt((e*e).mean())),"future_signed_error":float(e.mean()),"ideal_true_change_mae":float(np.abs((p-gc)-(g-gc)).mean()),"deployed_change_mae":float(np.abs(deployed).mean()),"deployed_change_signed_error":float(deployed.mean()),"anchor_error_mae":float(np.abs(anchor).mean()),"anchor_sq":float((anchor*anchor).mean()),"residual_minus_change_sq":float((reschg*reschg).mean()),"cross_2_anchor_residual":float((2*anchor*reschg).mean()),"future_error_sq":float((e*e).mean()),"prediction_variance":float(np.var(p)),"target_variance":float(np.var(g)),"copy_fraction_0p01n":float((np.abs(p-pc)<=.01).mean()),"target_at_clip_fraction":float((np.abs(g)>=20).mean())}
def future_rows(data,preds,ridge_preds,meta):
 rows=[];trials=[]
 for role,d in data["roles"].items():
  st=strata(data,d);pcur=d["x"][:,-1,192:195].numpy();y=d["y"].numpy();ycur=d["y_current"].numpy();base={"predicted_current_persistence":np.repeat(pcur[:,None,:],3,axis=1),"ground_truth_current_persistence_ideal":np.repeat(ycur[:,None,:],3,axis=1),"linear_ridge":ridge_preds[role]};base.update(preds.get(role,{}))
  for name,pred in base.items():
   for hi,h in enumerate(mt.HORIZONS):
    for ai,axis in enumerate(("x","y","z")):
     for sn in ("all","stable","transition","changing"):
      m=np.ones(len(y),bool) if sn=="all" else st==sn
      if m.any():rows.append({**meta,"role":role,"predictor":name,"horizon":h,"axis":axis,"stratum":sn,**metric_cell(pred[:,hi,ai],y[:,hi,ai],ycur[:,ai],pcur[:,ai],m)})
   if role=="validation" and name in ("F","J"):
    for ep in sorted(set(d["episode_id"])):
     ids=np.asarray([i for i,e in enumerate(d["episode_id"]) if e==ep]);trials.append({**meta,"role":role,"predictor":name,"episode":ep,"leakage_group":d["leakage_group"][int(ids[0])],"n":len(ids),"future_mae":float(np.abs(pred[ids]-y[ids]).mean()),"deployed_change_mae":float(np.abs((pred[ids]-pcur[ids,None,:])-(y[ids]-ycur[ids,None,:])).mean())})
 return rows,trials
def weighted_pauc(r13,eps,weights):
 expanded=[]
 for ep in eps:
  expanded.extend([ep]*weights.get(ep.group,0))
 return r13.rank_metrics(expanded)["pAUC"] if expanded else float("nan")
def ci_primary(r13,store,slip_trials,future_trial_rows):
 out=[];draw_rows=[];sidx=defaultdict(list);fidx=defaultdict(list)
 for x in slip_trials:
  if x["role"]=="validation" and x["policy"]=="FPR0.05|raw":sidx[(x["group"],x["fold"],int(x["seed"]))].append(x)
 for x in future_trial_rows:fidx[(x["group"],x["fold"],int(x["seed"]))].append(x)
 for fold in range(1,5):
  groups=sorted({e.group for e in store[("S",fold,mt.SEEDS[0],"validation")]});rng=np.random.default_rng(170000+fold);draws=[]
  for _ in range(2000):
   z=rng.integers(0,len(groups),len(groups));draws.append(dict(zip(groups,np.bincount(z,minlength=len(groups)))))
  vals=defaultdict(list)
  for di,w in enumerate(draws):
   seedvals=defaultdict(list)
   for seed in mt.SEEDS:
    for g in SLIP_GROUPS:
     eps=store[(g,fold,seed,"validation")];seedvals[f"{g}_pauc"].append(weighted_pauc(r13,eps,w))
     stats=sidx[(g,fold,seed)]
     a=r13.aggregate(stats,w);seedvals[f"{g}_fpr"].append(a["frame_static_FPR"]);seedvals[f"{g}_recall"].append(a["gross_recall"])
    for g in FUTURE_GROUPS:
     rr=fidx[(g,fold,seed)];num=sum(w.get(x["leakage_group"],0)*x["future_mae"]*x["n"] for x in rr);den=sum(w.get(x["leakage_group"],0)*x["n"] for x in rr);seedvals[f"{g}_future"].append(num/den if den else np.nan)
   metrics={"J_minus_S_pAUC":np.mean(seedvals["J_pauc"])-np.mean(seedvals["S_pauc"]),"J_minus_S_FPR_at_cal_FPR05":np.mean(seedvals["J_fpr"])-np.mean(seedvals["S_fpr"]),"J_minus_S_recall_at_cal_FPR05":np.mean(seedvals["J_recall"])-np.mean(seedvals["S_recall"]),"J_minus_F_future_MAE":np.mean(seedvals["J_future"])-np.mean(seedvals["F_future"])}
   for k,v in metrics.items():vals[k].append(v);draw_rows.append({"fold":fold,"draw":di,"metric":k,"difference":v})
  for k,v in vals.items():out.append({"fold":fold,"metric":k,"mean":float(np.mean(v)),"q025":float(np.quantile(v,.025,method="linear")),"median":float(np.quantile(v,.5,method="linear")),"q975":float(np.quantile(v,.975,method="linear")),"draws":len(v)})
 return out,draw_rows
def main():
 p=argparse.ArgumentParser();p.add_argument("--prepared-root",type=Path,required=True);p.add_argument("--formal-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--r13-source",type=Path,required=True);p.add_argument("--device",default="cuda:0");a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);r13=load_r13(a.r13_source);assert r13.smoke()["status"]=="pass"
 slip_rows=[];slip_trials=[];threshold_rows=[];future_metrics=[];future_trials=[];store={};input_index=[]
 for fold in range(1,5):
  for seed in mt.SEEDS:
   dp=a.prepared_root/f"p{fold}_s{seed}"/"prepared.pt";data=torch.load(dp,map_location="cpu",weights_only=False);norm=data["normalizer"]
   ridge=Ridge(alpha=1e-6,fit_intercept=True);xf=((data["roles"]["fit"]["x"]-norm["x_mean"])/norm["x_std"]).reshape(len(data["roles"]["fit"]["x"]),-1).numpy();ridge.fit(xf,data["roles"]["fit"]["y"].reshape(len(xf),-1).numpy());ridge_preds={role:ridge.predict(((d["x"]-norm["x_mean"])/norm["x_std"]).reshape(len(d["x"]),-1).numpy()).reshape(-1,3,3) for role,d in data["roles"].items()}
   future_pred_by_role={role:{} for role in ROLES}
   for group in mt.GROUPS:
    ckpath=a.formal_root/group/f"p{fold}_s{seed}"/"best.pth";ck=torch.load(ckpath,map_location="cpu",weights_only=False);model=mt.init_model(seed);model.load_state_dict(ck["model"]);model.to(a.device);meta={"group":group,"fold":fold,"seed":seed};role_eps={}
    pred_dir=a.output/"predictions"/group/f"p{fold}_s{seed}";pred_dir.mkdir(parents=True,exist_ok=True)
    for role,d in data["roles"].items():
     score,future=infer(model,d,norm,a.device);kwargs={"t":d["t"].numpy(),"stage":d["stage"].numpy(),"episode_id":np.asarray(d["episode_id"]),"leakage_group":np.asarray(d["leakage_group"])}
     if group in SLIP_GROUPS:kwargs["score"]=score;role_eps[role]=episodes(r13,d,score);store[(group,fold,seed,role)]=role_eps[role]
     if group in FUTURE_GROUPS:kwargs["future"]=future;future_pred_by_role[role][group]=future
     np.savez_compressed(pred_dir/f"{role}.npz",**kwargs)
    if group in SLIP_GROUPS:
     rr,tt,hh=point_rows(r13,role_eps,meta);slip_rows+=rr;slip_trials+=tt;threshold_rows+=hh
    input_index.append({**meta,"prepared":str(dp),"prepared_sha256":mt.sha(dp),"best":str(ckpath),"best_sha256":mt.sha(ckpath),"best_epoch":ck["best_epoch"],"best_metric":ck["best_metric"]})
   for group in FUTURE_GROUPS:
    preds={role:{group:future_pred_by_role[role][group]} for role in ROLES};rr,tt=future_rows(data,preds,ridge_preds,{"group":group,"fold":fold,"seed":seed});future_metrics+=rr;future_trials+=tt
   print(json.dumps({"evaluated":f"p{fold}_s{seed}"}),flush=True)
 write_csv(a.output/"slip/metrics.csv",slip_rows);write_csv(a.output/"slip/trials.csv",slip_trials);write_csv(a.output/"slip/thresholds.csv",threshold_rows);write_csv(a.output/"future/metrics.csv",future_metrics);write_csv(a.output/"future/trials.csv",future_trials);mt.atomic_json({"schema":"round17_evaluation_input_index_v1","runs":input_index},a.output/"INPUT_INDEX.json")
 ci,draws=ci_primary(r13,store,slip_trials,future_trials);write_csv(a.output/"bootstrap/PAIRED_CI.csv",ci);write_csv(a.output/"bootstrap/PAIRED_DRAWS.csv",draws)
 # Pre-fixed failure/improvement case candidates.
 sc=[]
 for fold in range(1,5):
  for seed in mt.SEEDS:
   for direction in ("J_minus_S","S_minus_J"):
    a1="J" if direction.startswith("J") else "S";b1="S" if a1=="J" else "J";aa={(x["episode"],x["leakage_group"]):x for x in slip_trials if x["group"]==a1 and x["fold"]==fold and int(x["seed"])==seed and x["role"]=="validation" and x["policy"]=="FPR0.05|raw"};bb={(x["episode"],x["leakage_group"]):x for x in slip_trials if x["group"]==b1 and x["fold"]==fold and int(x["seed"])==seed and x["role"]=="validation" and x["policy"]=="FPR0.05|raw"}
    for k in sorted(aa.keys()&bb.keys()):sc.append({"kind":direction,"fold":fold,"seed":seed,"episode":k[0],"leakage_group":k[1],"difference":float(aa[k]["static_fp"])-float(bb[k]["static_fp"])})
 fc=[]
 for fold in range(1,5):
  for seed in mt.SEEDS:
   aa={x["episode"]:x for x in future_trials if x["group"]=="J" and x["fold"]==fold and int(x["seed"])==seed};bb={x["episode"]:x for x in future_trials if x["group"]=="F" and x["fold"]==fold and int(x["seed"])==seed}
   for e in sorted(aa.keys()&bb.keys()):fc.append({"kind":"J_minus_F_future_MAE","fold":fold,"seed":seed,"episode":e,"leakage_group":aa[e]["leakage_group"],"difference":aa[e]["future_mae"]-bb[e]["future_mae"]})
 candidates=sc+fc;write_csv(a.output/"cases/CANDIDATES.csv",candidates);selected=[]
 for kind in ("J_minus_S","S_minus_J","J_minus_F_future_MAE"):
  rr=[x for x in candidates if x["kind"]==kind];selected.append(sorted(rr,key=lambda x:(-x["difference"],x["fold"],x["seed"],x["episode"]))[0])
 mt.atomic_json({"schema":"round17_cases_v1","rules":"pre-registered maximum adverse/improvement differences with lexical tie","selected":selected},a.output/"cases/SELECTED.json")
 summary={"schema":"round17_evaluation_summary_v1","status":"complete","models":36,"slip_metric_rows":len(slip_rows),"slip_trial_rows":len(slip_trials),"future_metric_rows":len(future_metrics),"future_trial_rows":len(future_trials),"paired_ci_rows":len(ci),"test_consumed":False,"hashes":{str(x.relative_to(a.output)):mt.sha(x) for x in a.output.rglob("*.csv")}};mt.atomic_json(summary,a.output/"SUMMARY.json");print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
