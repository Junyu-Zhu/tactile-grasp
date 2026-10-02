#!/usr/bin/env python3
"""Locked Round-18 calibration, validation, perturbation, FiLM and paired-CI evaluation."""
from __future__ import annotations
import argparse,csv,importlib.util,json,math,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import train as r18

ROLES=r18.ROLES;GROUPS=r18.GROUPS;ALPHAS=(.01,.05,.10);PAIRS=(("M0","C0"),("M0","V0"),("C0","V0"))
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.generic):x=x.item()
 if isinstance(x,float) and not math.isfinite(x):return None
 return x
def write_csv(path,rows):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if not rows:path.write_text("");return
 fields=list(dict.fromkeys(k for row in rows for k in row));tmp=path.with_suffix(path.suffix+".tmp")
 with tmp.open("w",newline="") as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(clean(rows))
 tmp.replace(path)
def load_r13(path):
 spec=importlib.util.spec_from_file_location("r18_r13",path);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
def episodes(r13,d,scores):
 out=[]
 for eid in sorted(set(d["episode_id"])):
  ids=np.asarray([i for i,e in enumerate(d["episode_id"]) if e==eid]);ids=ids[np.argsort(d["t"][ids].numpy())]
  out.append(r13.Episode(eid,d["leakage_group"][int(ids[0])],d["probe"][int(ids[0])],d["t"][ids].numpy(),d["stage"][ids].numpy(),scores[ids]))
 return out
def normal_x(d,norm):return (d["x"]-norm["mean"])/norm["std"]
def infer(model,x,device,mode="normal"):
 x=x.clone()
 if mode=="force_mean":x[...,192:195]=0
 elif mode=="force_lag1":x[...,1:,192:195]=x[...,:-1,192:195].clone();x[...,0,192:195]=0
 scores=[];gammas=[];betas=[];ratios=[];model.eval()
 with torch.no_grad():
  for ids in torch.arange(len(x)).split(1024):
   xx=x[ids].to(device)
   if mode=="film_off":
    visual=model.visual_ln(xx[...,:192]);hidden=model.gru(visual)[0][:,-1];logit=model.risk(hidden).squeeze(-1);comp=None
   else:comp=model.components(xx);logit=comp["logit"]
   scores.append(torch.sigmoid(logit).cpu())
   if model.group=="M0" and comp is not None:
    gamma=comp["gamma"];beta=comp["beta"];base=model.visual_ln(xx[...,:192]);mod=(1+gamma)*base+beta
    gammas.append(gamma.cpu());betas.append(beta.cpu());ratios.append((mod.norm(dim=-1)/(base.norm(dim=-1)+1e-12)).cpu())
 return torch.cat(scores).numpy(),({"gamma":torch.cat(gammas).numpy(),"beta":torch.cat(betas).numpy(),"ratio":torch.cat(ratios).numpy()} if gammas else None)
def maxba(r13,cal):
 best=None
 for th in r13.candidate_thresholds(cal,1):
  _,agg=r13.evaluate_policy(cal,float(th),1);key=(agg["balanced_accuracy"],-agg["frame_static_FPR"],float(th))
  if best is None or key>best[0]:best=(key,float(th),agg)
 return best[1]
def thresholds(r13,cal):
 out={"fixed_0.5":.5,"maxBA":maxba(r13,cal)};cache=r13.build_fit_cache(cal,1)
 for alpha in ALPHAS:out[f"FPR{int(alpha*100)}"]=r13.fit_cached(cache,"trial_macro_static_FPR",alpha,constraint_override="frame_static_FPR")["threshold"]
 return out
def rank_weighted(r13,eps,weights):
 expanded=[]
 for ep in eps:expanded.extend([ep]*weights.get(ep.group,0))
 return r13.rank_metrics(expanded) if expanded else {"AP":float("nan"),"pAUC":float("nan"),"positive_prevalence":float("nan")}
def paired_ci(r13,store,trial_rows):
 trial=defaultdict(list)
 for row in trial_rows:
  if row["role"]=="validation" and row["policy"]=="FPR5|raw":trial[(row["group"],row["fold"],row["seed"])].append(row)
 summaries=[];draw_rows=[]
 for fold in range(1,5):
  groups=sorted({ep.group for ep in store[("V0",fold,r18.SEEDS[0],"validation")]});rng=np.random.default_rng(180000+fold);draws=[dict(zip(groups,np.bincount(rng.integers(0,len(groups),len(groups)),minlength=len(groups)))) for _ in range(2000)]
  values=defaultdict(list)
  for draw_id,weights in enumerate(draws):
   by_group=defaultdict(lambda:defaultdict(list))
   for seed in r18.SEEDS:
    for group in GROUPS:
     rank=rank_weighted(r13,store[(group,fold,seed,"validation")],weights);agg=r13.aggregate(trial[(group,fold,seed)],weights)
     for metric in ("AP","pAUC"):by_group[group][metric].append(rank[metric])
     for metric in ("frame_static_FPR","gross_recall","balanced_accuracy","macro_f1"):by_group[group][metric].append(agg[metric])
   for left,right in PAIRS:
    for metric in ("AP","pAUC","frame_static_FPR","gross_recall","balanced_accuracy","macro_f1"):
     lv=float(np.mean(by_group[left][metric]));rv=float(np.mean(by_group[right][metric]));difference=lv-rv;valid=math.isfinite(difference);key=(left,right,metric)
     if valid:values[key].append(difference)
     draw_rows.append({"fold":fold,"draw":draw_id,"left":left,"right":right,"metric":metric,"difference":difference if valid else None,"valid":valid,"invalid_reason":None if valid else "resample_lacks_required_class_support"})
  for left,right in PAIRS:
   for metric in ("AP","pAUC","frame_static_FPR","gross_recall","balanced_accuracy","macro_f1"):
    vals=values[(left,right,metric)];summaries.append({"fold":fold,"left":left,"right":right,"metric":metric,"attempted_draws":2000,"valid_draws":len(vals),"invalid_draws":2000-len(vals),"mean":float(np.mean(vals)) if vals else None,"q025":float(np.quantile(vals,.025)) if vals else None,"median":float(np.quantile(vals,.5)) if vals else None,"q975":float(np.quantile(vals,.975)) if vals else None})
 return summaries,draw_rows
def film_rows(meta,role,d,film,threshold,fit_force_q90):
 gamma,beta,ratio=film["gamma"],film["beta"],film["ratio"];anomaly=((np.abs(gamma)>=.5)|(1+gamma<.5)|(1+gamma>1.5)|(np.abs(beta)>=3)).any((1,2))|(ratio>2).any(1)
 force_error=np.abs(d["x"][:,-1,192:195].numpy()-d["y_current"].numpy()).mean(1);large=force_error>=fit_force_q90
 return {**meta,"role":role,"endpoints":len(anomaly),"gamma_mean":float(gamma.mean()),"gamma_abs_q95":float(np.quantile(np.abs(gamma),.95)),"gamma_abs_max":float(np.abs(gamma).max()),"beta_mean":float(beta.mean()),"beta_abs_q95":float(np.quantile(np.abs(beta),.95)),"beta_abs_max":float(np.abs(beta).max()),"scale_min":float((1+gamma).min()),"scale_max":float((1+gamma).max()),"norm_ratio_q95":float(np.quantile(ratio,.95)),"norm_ratio_max":float(ratio.max()),"anomalous_endpoints":int(anomaly.sum()),"anomalous_fraction":float(anomaly.mean()),"force_error_q90_fit_n":fit_force_q90,"large_force_error_endpoints":int(large.sum()),"anomaly_fraction_large_force_error":float(anomaly[large].mean()) if large.any() else None,"anomaly_fraction_other":float(anomaly[~large].mean()) if (~large).any() else None,"original_fpr5_threshold":threshold}
def main():
 p=argparse.ArgumentParser();p.add_argument("--prepared-root",type=Path,required=True);p.add_argument("--formal-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--r13-source",type=Path,required=True);p.add_argument("--device",default="cuda:0");a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);r13=load_r13(a.r13_source);assert r13.smoke()["status"]=="pass"
 metric_rows=[];trial_rows=[];threshold_rows=[];score_rows=[];incipient_rows=[];curve_rows=[];perturb_rows=[];film_stats=[];store={};input_index=[]
 for fold in range(1,5):
  for seed in r18.SEEDS:
   dp=a.prepared_root/f"p{fold}_s{seed}"/"prepared.pt";data=torch.load(dp,map_location="cpu",weights_only=False);assert data["fold"]==fold and data["seed"]==seed
   for group in GROUPS:
    ckpath=a.formal_root/group/f"p{fold}_s{seed}"/"best.pth";ck=torch.load(ckpath,map_location="cpu",weights_only=False);identity=ck["identity"]
    if identity["input_sha256"]!=r18.sha(dp) or identity["group"]!=group:raise ValueError("evaluation input identity")
    model=r18.init_model(group,seed);model.load_state_dict(ck["model"]);model.to(a.device);norm=ck["normalizer"];meta={"group":group,"fold":fold,"seed":seed};role_eps={};role_scores={};role_films={}
    fit_error=np.abs(data["roles"]["fit"]["x"][:,-1,192:195].numpy()-data["roles"]["fit"]["y_current"].numpy()).mean(1);force_q90=float(np.quantile(fit_error,.9))
    for role in ROLES:
     d=data["roles"][role];score,film=infer(model,normal_x(d,norm),a.device);role_scores[role]=score;role_films[role]=film;role_eps[role]=episodes(r13,d,score);store[(group,fold,seed,role)]=role_eps[role]
     for i,(ep,lg,t,stage,probe,value) in enumerate(zip(d["episode_id"],d["leakage_group"],d["t"].tolist(),d["stage"].tolist(),d["probe"],score.tolist())):score_rows.append({**meta,"role":role,"episode":ep,"leakage_group":lg,"probe":probe,"t":t,"stage":stage,"score":value,"predicted_force_error_mae_n":float(np.abs(d["x"][i,-1,192:195].numpy()-d["y_current"][i].numpy()).mean())})
     inc=score[d["stage"].eq(1).numpy()];incipient_rows.append({**meta,"role":role,"frames":len(inc),"episodes":len(set(e for e,s in zip(d["episode_id"],d["stage"].tolist()) if s==1)),"mean":float(inc.mean()) if len(inc) else None,"std":float(inc.std()) if len(inc) else None,"q10":float(np.quantile(inc,.1)) if len(inc) else None,"median":float(np.median(inc)) if len(inc) else None,"q90":float(np.quantile(inc,.9)) if len(inc) else None})
    th=thresholds(r13,role_eps["calibration"])
    for name,value in th.items():threshold_rows.append({**meta,"policy":name,"threshold":value,"fit_role":"calibration","never_alarm":bool(value>1)})
    for name,value in th.items():
     for k,rule in ((1,"raw"),(2,"confirm2_release1")):
      for role in ROLES:
       trials,agg=r13.evaluate_policy(role_eps[role],value,k);rank=r13.rank_metrics(role_eps[role]);policy=f"{name}|{'raw' if k==1 else 'confirm2'}";metric_rows.append({**meta,"role":role,"policy":policy,"threshold":value,"k":k,"never_alarm":value>1,**agg,**rank});trial_rows.extend({**meta,"role":role,"policy":policy,"threshold":value,"k":k,**row} for row in trials)
    for q in range(1001):
     value=q/1000;_,agg=r13.evaluate_policy(role_eps["validation"],value,1);curve_rows.append({**meta,"role":"validation","threshold":value,"frame_static_FPR":agg["frame_static_FPR"],"gross_recall":agg["gross_recall"]})
    fpr5=th["FPR5"]
    if group=="M0":
     for role in ROLES:film_stats.append(film_rows(meta,role,data["roles"][role],role_films[role],fpr5,force_q90))
    for mode in (("force_mean","force_lag1") if group=="C0" else ("force_mean","force_lag1","film_off") if group=="M0" else ("force_mean","force_lag1")):
     score,_=infer(model,normal_x(data["roles"]["validation"],norm),a.device,mode);eps=episodes(r13,data["roles"]["validation"],score);_,agg=r13.evaluate_policy(eps,fpr5,1);perturb_rows.append({**meta,"role":"validation","mode":mode,"threshold_source":"original_calibration_FPR5_raw","threshold":fpr5,**agg,**r13.rank_metrics(eps),"max_abs_score_change":float(np.max(np.abs(score-role_scores["validation"])))})
     if group=="V0" and not np.array_equal(score,role_scores["validation"]):raise ValueError("V0 force perturbation changed output")
    input_index.append({**meta,"prepared":str(dp),"prepared_sha256":r18.sha(dp),"best":str(ckpath),"best_sha256":r18.sha(ckpath),"best_epoch":ck["best_epoch"],"best_metric":ck["best_metric"],"source_sha256":identity["source_sha256"],"protocol_sha256":identity["protocol_sha256"]})
    print(json.dumps({"evaluated":f"{group}/p{fold}_s{seed}"}),flush=True)
 write_csv(a.output/"scores/RAW_SCORES.csv",score_rows);write_csv(a.output/"metrics/WORKPOINT_METRICS.csv",metric_rows);write_csv(a.output/"metrics/TRIAL_METRICS.csv",trial_rows);write_csv(a.output/"metrics/CALIBRATION_THRESHOLDS.csv",threshold_rows);write_csv(a.output/"metrics/INCIPIENT_DISTRIBUTION.csv",incipient_rows);write_csv(a.output/"metrics/DESCRIPTIVE_CURVES.csv",curve_rows);write_csv(a.output/"diagnostics/FORCE_PERTURBATIONS.csv",perturb_rows);write_csv(a.output/"diagnostics/FILM_STATS.csv",film_stats)
 ci,draws=paired_ci(r13,store,trial_rows);write_csv(a.output/"bootstrap/PAIRED_CI.csv",ci);write_csv(a.output/"bootstrap/PAIRED_DRAWS.csv",draws)
 # Locked cases: historical exact episode plus two deterministic adverse M0-C0 cases.
 candidates=[]
 for fold in range(1,5):
  for seed in r18.SEEDS:
   mm={x["episode"]:x for x in trial_rows if x["group"]=="M0" and x["fold"]==fold and x["seed"]==seed and x["role"]=="validation" and x["policy"]=="FPR5|raw"};cc={x["episode"]:x for x in trial_rows if x["group"]=="C0" and x["fold"]==fold and x["seed"]==seed and x["role"]=="validation" and x["policy"]=="FPR5|raw"}
   for ep in sorted(mm.keys()&cc.keys()):candidates.append({"fold":fold,"seed":seed,"episode":ep,"leakage_group":mm[ep]["leakage_group"],"m_minus_c_static_alarm_frames":float(mm[ep]["static_fp"])-float(cc[ep]["static_fp"]),"m_minus_c_gross_missed_frames":float(mm[ep]["fn"])-float(cc[ep]["fn"])})
 selected=[{"kind":"historical_fixed_failure","fold":4,"seed":seed,"episode":"htt/p3_sliding/0_press_13"} for seed in r18.SEEDS]
 selected.append({"kind":"largest_M0_minus_C0_static_alarm_increase",**sorted(candidates,key=lambda x:(-x["m_minus_c_static_alarm_frames"],x["fold"],x["seed"],x["episode"]))[0]})
 selected.append({"kind":"largest_M0_minus_C0_gross_miss_increase",**sorted(candidates,key=lambda x:(-x["m_minus_c_gross_missed_frames"],x["fold"],x["seed"],x["episode"]))[0]})
 write_csv(a.output/"cases/CANDIDATES.csv",candidates);r18.atomic_json(a.output/"cases/SELECTED.json",{"schema":"round18_locked_cases_v1","selected":selected,"ties":"fold,seed,episode lexical","source_images":"generated by report.py from accepted parent source_path; missing is reported"})
 r18.atomic_json(a.output/"INPUT_INDEX.json",{"schema":"round18_evaluation_input_index_v1","runs":input_index})
 summary={"schema":"round18_evaluation_summary_v1","status":"complete","runs":len(input_index),"workpoint_metric_rows":len(metric_rows),"trial_metric_rows":len(trial_rows),"raw_score_rows":len(score_rows),"paired_ci_rows":len(ci),"bootstrap_draw_rows":len(draws),"perturbation_rows":len(perturb_rows),"film_rows":len(film_stats),"test_consumed":False,"support_boundary":"Round-17 common future-complete endpoints only","hashes":{str(path.relative_to(a.output)):r18.sha(path) for path in a.output.rglob("*.csv")}}
 r18.atomic_json(a.output/"SUMMARY.json",summary);print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
