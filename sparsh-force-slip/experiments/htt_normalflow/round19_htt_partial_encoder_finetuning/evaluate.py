#!/usr/bin/env python3
"""Unified locked evaluation for accepted V0/M0 baselines and Round-19 V2/M2."""
from __future__ import annotations
import argparse,csv,importlib.util,json,math,sys,tempfile,os
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import train as tr

GROUPS=("V0","M0","V2","M2");ALPHAS=(.01,.05,.10)
CONTRASTS={"V2-V0":{"V2":1,"V0":-1},"M2-M0":{"M2":1,"M0":-1},"M2-V2":{"M2":1,"V2":-1},"interaction_(M2-V2)-(M0-V0)":{"M2":1,"V2":-1,"M0":-1,"V0":1}}
BOOT_METRICS=("AP","pAUC","frame_static_FPR","trial_macro_static_FPR","trial_any_static_alarm_rate","gross_recall","balanced_accuracy","macro_f1","false_starts_per_trial","static_alarming_frames_per_static_support_trial","event_recall","mean_detected_event_delay","preexisting_alarm_event_rate","segment_coverage")
def load_r13(path):
 spec=importlib.util.spec_from_file_location("r19_r13",path);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.generic):x=x.item()
 return None if isinstance(x,float) and not math.isfinite(x) else x
def write_csv(path,rows):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if not rows:path.write_text("");return
 fields=list(dict.fromkeys(k for r in rows for k in r));fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=path.name+".",suffix=".tmp");os.close(fd)
 try:
  with open(tmp,"w",newline="") as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(clean(rows))
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def read_csv(path):
 with open(path,newline="") as f:return list(csv.DictReader(f))
def episodes(r13,d,scores):
 out=[]
 for eid in sorted(set(d["episode_id"])):
  ids=np.asarray([i for i,e in enumerate(d["episode_id"]) if e==eid]);ids=ids[np.argsort(d["t"][ids].numpy())];out.append(r13.Episode(eid,d["leakage_group"][int(ids[0])],d["probe"][int(ids[0])],d["t"][ids].numpy(),d["stage"][ids].numpy(),scores[ids]))
 return out
def rank_metrics(eps,weights=None):
 ss=[];ll=[]
 for ep in eps:
  w=1 if weights is None else weights.get(ep.group,0)
  if not w:continue
  mask=ep.stage!=1
  for _ in range(w):ss.append(ep.score[mask]);ll.append(ep.stage[mask]==2)
 if not ss:return {"AP":float("nan"),"pAUC":float("nan"),"positive_prevalence":float("nan")}
 scores=np.concatenate(ss);labels=np.concatenate(ll).astype(np.int64)
 if set(labels.tolist())!={0,1}:return {"AP":float("nan"),"pAUC":float("nan"),"positive_prevalence":float("nan")}
 order=np.argsort(-scores,kind="stable");scores=scores[order];labels=labels[order];ends=np.r_[np.flatnonzero(scores[1:]!=scores[:-1]),len(scores)-1];tp=np.cumsum(labels)[ends];fp=np.cumsum(1-labels)[ends];rec=tp/labels.sum();precision=tp/(tp+fp);ap=float(np.sum(np.diff(np.r_[0.,rec])*precision));fpr=np.r_[0.,fp/(len(labels)-labels.sum())];tpr=np.r_[0.,rec];limit=.1
 if np.any(fpr==limit):keep=fpr<=limit;xx=fpr[keep];yy=tpr[keep]
 else:
  i=np.flatnonzero(fpr<limit)[-1];q=(limit-fpr[i])/(fpr[i+1]-fpr[i]);xx=np.r_[fpr[:i+1],limit];yy=np.r_[tpr[:i+1],tpr[i]+q*(tpr[i+1]-tpr[i])]
 return {"AP":ap,"pAUC":float(np.trapezoid(yy,xx)/limit),"positive_prevalence":float(labels.mean())}
def maxba(r13,cal):
 best=None
 for th in r13.candidate_thresholds(cal,1):
  _,agg=r13.evaluate_policy(cal,float(th),1);key=(agg["balanced_accuracy"],-agg["frame_static_FPR"],float(th))
  if best is None or key>best[0]:best=(key,float(th))
 return best[1]
def thresholds(r13,cal):
 out={"fixed_0.5":.5,"maxBA":maxba(r13,cal)};cache=r13.build_fit_cache(cal,1)
 for alpha in ALPHAS:out[f"FPR{int(alpha*100)}"]=r13.fit_cached(cache,"trial_macro_static_FPR",alpha,constraint_override="frame_static_FPR")["threshold"]
 return out
def bootstrap(r13,store,trial_rows):
 trial=defaultdict(list)
 for x in trial_rows:
  if x["role"]=="validation" and x["policy"]=="FPR5|raw":trial[(x["group"],x["fold"],x["seed"])].append(x)
 ci=[];draw_rows=[]
 for fold in range(1,5):
  expected=None
  for group in GROUPS:
   for seed in tr.SEEDS:
    got={e.group for e in store[(group,fold,seed,"validation")]};expected=got if expected is None else expected
    if got!=expected:raise ValueError("bootstrap leakage-group support mismatch")
  groups=sorted(expected);rng=np.random.default_rng(190000+fold);values=defaultdict(list)
  for draw in range(2000):
   weights=dict(zip(groups,np.bincount(rng.integers(0,len(groups),len(groups)),minlength=len(groups))));gm=defaultdict(lambda:defaultdict(list))
   for group in GROUPS:
    for seed in tr.SEEDS:
     rank=rank_metrics(store[(group,fold,seed,"validation")],weights);agg=r13.aggregate(trial[(group,fold,seed)],weights)
     for metric in BOOT_METRICS:gm[group][metric].append(rank[metric] if metric in rank else agg[metric])
   means={g:{m:float(np.mean(gm[g][m])) for m in BOOT_METRICS} for g in GROUPS}
   for contrast,coef in CONTRASTS.items():
    for metric in BOOT_METRICS:
     value=sum(c*means[g][metric] for g,c in coef.items());valid=math.isfinite(value)
     if valid:values[(contrast,metric)].append(value)
     draw_rows.append({"fold":fold,"draw":draw,"contrast":contrast,"metric":metric,"difference":value if valid else None,"valid":valid,"invalid_reason":None if valid else "resample_lacks_required_support"})
  for contrast in CONTRASTS:
   for metric in BOOT_METRICS:
    v=values[(contrast,metric)];ci.append({"fold":fold,"contrast":contrast,"metric":metric,"attempted_draws":2000,"valid_draws":len(v),"invalid_draws":2000-len(v),"mean":float(np.mean(v)) if v else None,"q025":float(np.quantile(v,.025)) if v else None,"median":float(np.quantile(v,.5)) if v else None,"q975":float(np.quantile(v,.975)) if v else None})
 return ci,draw_rows
def main():
 p=argparse.ArgumentParser();p.add_argument("--prepared-root",type=Path,required=True);p.add_argument("--prediction-root",type=Path,required=True);p.add_argument("--r18-evaluation",type=Path,required=True);p.add_argument("--r18-final-status",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--r13-source",type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);r13=load_r13(a.r13_source);assert r13.smoke()["status"]=="pass"
 r18_summary=json.loads((a.r18_evaluation/"SUMMARY.json").read_text());r18_final=json.loads(a.r18_final_status.read_text())
 if r18_summary.get("status")!="complete" or r18_summary.get("test_consumed") or r18_final.get("status")!="complete":raise ValueError("R18 acceptance missing")
 raw=a.r18_evaluation/"scores/RAW_SCORES.csv"
 if tr.sha(raw)!=r18_summary["hashes"]["scores/RAW_SCORES.csv"]:raise ValueError("R18 score hash")
 baseline={}
 for x in read_csv(raw):
  if x["group"] in ("V0","M0"):baseline[(x["group"],int(x["fold"]),int(x["seed"]),x["role"],x["episode"],int(x["t"]))]=float(x["score"])
 metric_rows=[];trial_rows=[];threshold_rows=[];score_rows=[];incipient_rows=[];curve_rows=[];perturb_rows=[];film_summary=[];film_feature=[];transfer=[];store={};index=[]
 for fold in range(1,5):
  for seed in tr.SEEDS:
   dp=a.prepared_root/f"p{fold}_s{seed}"/"prepared.pt";data=torch.load(dp,map_location="cpu",weights_only=False)
   for group in GROUPS:
    meta={"group":group,"fold":fold,"seed":seed};pred_path=None;pred=None
    if group in ("V2","M2"):
     po=a.prediction_root/group/f"p{fold}_s{seed}";ps=json.loads((po/"SUMMARY.json").read_text());pred_path=po/"PREDICTIONS.npz"
     if ps.get("status")!="complete" or tr.sha(pred_path)!=ps["prediction_sha256"]:raise ValueError("prediction identity")
     pred=np.load(pred_path,allow_pickle=False);index.append({**meta,"kind":"round19_prediction","path":str(pred_path),"sha256":tr.sha(pred_path),"summary_sha256":tr.sha(po/"SUMMARY.json")})
     if group=="M2":film_feature.extend(read_csv(po/"FILM_FEATURE_TIME_ROLE.csv"))
    else:index.append({**meta,"kind":"round18_accepted_scores","path":str(raw),"sha256":tr.sha(raw),"round18_summary_sha256":tr.sha(a.r18_evaluation/"SUMMARY.json")})
    role_eps={};role_scores={}
    for role in tr.ROLES:
     d=data["roles"][role]
     if pred is not None:score=np.asarray(pred[f"{role}_score"],dtype=np.float64)
     else:score=np.asarray([baseline[(group,fold,seed,role,e,int(t))] for e,t in zip(d["episode_id"],d["t"].tolist())])
     if len(score)!=len(d["t"]) or not np.isfinite(score).all():raise ValueError("score support")
     role_scores[role]=score;role_eps[role]=episodes(r13,d,score);store[(group,fold,seed,role)]=role_eps[role]
     for e,lg,probe,t,stage,value in zip(d["episode_id"],d["leakage_group"],d["probe"],d["t"].tolist(),d["stage"].tolist(),score.tolist()):score_rows.append({**meta,"role":role,"episode":e,"leakage_group":lg,"probe":probe,"t":t,"stage":stage,"score":value})
     inc=score[d["stage"].eq(1).numpy()];incipient_rows.append({**meta,"role":role,"frames":len(inc),"mean":float(inc.mean()) if len(inc) else None,"std":float(inc.std()) if len(inc) else None,"q10":float(np.quantile(inc,.1)) if len(inc) else None,"median":float(np.median(inc)) if len(inc) else None,"q90":float(np.quantile(inc,.9)) if len(inc) else None})
    th=thresholds(r13,role_eps["calibration"])
    for name,value in th.items():threshold_rows.append({**meta,"policy":name,"threshold":value,"fit_role":"calibration","never_alarm":bool(value>1)})
    for name,value in th.items():
     for k in (1,2):
      policy=f"{name}|{'raw' if k==1 else 'confirm2'}"
      for role in tr.ROLES:
       trials,agg=r13.evaluate_policy(role_eps[role],value,k);rank=rank_metrics(role_eps[role]);metric_rows.append({**meta,"role":role,"policy":policy,"threshold":value,"k":k,"never_alarm":value>1,**agg,**rank});trial_rows.extend({**meta,"role":role,"policy":policy,"threshold":value,"k":k,**row} for row in trials)
    for q in range(1001):
     value=q/1000;_,agg=r13.evaluate_policy(role_eps["validation"],value,1);curve_rows.append({**meta,"role":"validation","threshold":value,"frame_static_FPR":agg["frame_static_FPR"],"gross_recall":agg["gross_recall"]})
    fpr5=th["FPR5"];_,calagg=r13.evaluate_policy(role_eps["calibration"],fpr5,1);_,valagg=r13.evaluate_policy(role_eps["validation"],fpr5,1);transfer.append({**meta,"policy":"FPR5|raw","threshold":fpr5,"calibration_frame_static_FPR":calagg["frame_static_FPR"],"validation_frame_static_FPR":valagg["frame_static_FPR"],"validation_minus_calibration":valagg["frame_static_FPR"]-calagg["frame_static_FPR"],"absolute_transfer_error":abs(valagg["frame_static_FPR"]-calagg["frame_static_FPR"]),"never_alarm":fpr5>1})
    modes=("force_mean","force_lag1","film_off") if group=="M2" else ("force_mean","force_lag1") if group=="V2" else ()
    for mode in modes:
     score=role_scores["validation"] if group=="V2" else np.asarray(pred[f"validation_{mode}"]);eps=episodes(r13,data["roles"]["validation"],score);_,agg=r13.evaluate_policy(eps,fpr5,1);perturb_rows.append({**meta,"role":"validation","mode":mode,"threshold_source":"original_calibration_FPR5_raw","threshold":fpr5,**agg,**rank_metrics(eps),"max_abs_score_change":float(np.max(np.abs(score-role_scores["validation"])))})
    if group=="M2":
     fitq=float(np.quantile(pred["fit_force_error_mae_n"],.9))
     for role in tr.ROLES:
      flag=pred[f"{role}_anomaly"].astype(bool);err=pred[f"{role}_force_error_mae_n"];ratio=pred[f"{role}_norm_ratio"];large=err>=fitq;film_summary.append({**meta,"role":role,"endpoints":len(flag),"anomalous_endpoints":int(flag.sum()),"anomalous_fraction":float(flag.mean()),"norm_ratio_q95":float(np.quantile(ratio,.95)),"norm_ratio_max":float(ratio.max()),"force_error_q90_fit_n":fitq,"large_force_error_endpoints":int(large.sum()),"anomaly_fraction_large_force_error":float(flag[large].mean()) if large.any() else None,"anomaly_fraction_other":float(flag[~large].mean()) if (~large).any() else None,"original_fpr5_threshold":fpr5})
    print(json.dumps({"evaluated":f"{group}/p{fold}_s{seed}"}),flush=True)
 ci,draws=bootstrap(r13,store,trial_rows)
 candidates=[]
 for fold in range(1,5):
  for seed in tr.SEEDS:
   by={}
   for group in ("V2","M2"):
    by[group]={x["episode"]:x for x in trial_rows if x["group"]==group and x["fold"]==fold and x["seed"]==seed and x["role"]=="validation" and x["policy"]=="FPR5|raw"}
   for ep in sorted(by["V2"].keys()&by["M2"].keys()):candidates.append({"fold":fold,"seed":seed,"episode":ep,"leakage_group":by["M2"][ep]["leakage_group"],"M2_minus_V2_static_alarm_frames":float(by["M2"][ep]["static_fp"])-float(by["V2"][ep]["static_fp"]),"M2_minus_V2_gross_missed_frames":float(by["M2"][ep]["fn"])-float(by["V2"][ep]["fn"])})
 selected=[{"kind":"historical_fixed_failure","fold":4,"seed":s,"episode":"htt/p3_sliding/0_press_13"} for s in tr.SEEDS];selected.append({"kind":"largest_M2_minus_V2_static_alarm_increase",**sorted(candidates,key=lambda x:(-x["M2_minus_V2_static_alarm_frames"],x["fold"],x["seed"],x["episode"]))[0]});selected.append({"kind":"largest_M2_minus_V2_gross_miss_increase",**sorted(candidates,key=lambda x:(-x["M2_minus_V2_gross_missed_frames"],x["fold"],x["seed"],x["episode"]))[0]});worst=sorted(transfer,key=lambda x:(-x["absolute_transfer_error"],x["group"],x["fold"],x["seed"]))[0];trial_candidates=[x for x in trial_rows if x["group"]==worst["group"] and x["fold"]==worst["fold"] and x["seed"]==worst["seed"] and x["role"]=="validation" and x["policy"]=="FPR5|raw" and x["static_frames"]>0];case=sorted(trial_candidates,key=lambda x:(-abs(x["static_fp"]/x["static_frames"]-worst["calibration_frame_static_FPR"]),x["episode"]))[0];selected.append({"kind":"largest_validation_minus_calibration_FPR_transfer_failure","group":worst["group"],"fold":worst["fold"],"seed":worst["seed"],"episode":case["episode"],"run_absolute_transfer_error":worst["absolute_transfer_error"]})
 outputs={"scores/RAW_SCORES.csv":score_rows,"metrics/WORKPOINT_METRICS.csv":metric_rows,"metrics/TRIAL_METRICS.csv":trial_rows,"metrics/CALIBRATION_THRESHOLDS.csv":threshold_rows,"metrics/INCIPIENT_DISTRIBUTION.csv":incipient_rows,"metrics/DESCRIPTIVE_CURVES.csv":curve_rows,"metrics/THRESHOLD_TRANSFER.csv":transfer,"diagnostics/FORCE_PERTURBATIONS.csv":perturb_rows,"diagnostics/FILM_SUMMARY.csv":film_summary,"diagnostics/FILM_FEATURE_TIME_ROLE.csv":film_feature,"bootstrap/PAIRED_CI.csv":ci,"bootstrap/PAIRED_DRAWS.csv":draws,"cases/CANDIDATES.csv":candidates}
 for name,rows in outputs.items():write_csv(a.output/name,rows)
 tr.atomic_json(a.output/"cases/SELECTED.json",{"schema":"round19_locked_cases_v1","selected":selected,"ties":"fold,seed,episode lexical","test_consumed":False});tr.atomic_json(a.output/"INPUT_INDEX.json",{"schema":"round19_evaluation_input_index_v1","runs":index,"r18_summary_sha256":tr.sha(a.r18_evaluation/"SUMMARY.json"),"r18_final_status_sha256":tr.sha(a.r18_final_status),"r13_source_sha256":tr.sha(a.r13_source)})
 summary={"schema":"round19_evaluation_summary_v1","status":"complete","runs":len(index),"groups":list(GROUPS),"bootstrap_draws_per_fold":2000,"primary_contrasts":list(CONTRASTS),"test_consumed":False,"support_boundary":"Round-17 common future-complete endpoints only","counts":{k:len(v) for k,v in outputs.items()},"hashes":{str(x.relative_to(a.output)):tr.sha(x) for x in a.output.rglob("*.csv")}}
 tr.atomic_json(a.output/"SUMMARY.json",summary);print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
