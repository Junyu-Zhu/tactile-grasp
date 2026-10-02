#!/usr/bin/env python3
"""Locked E3 evaluation over 48 fine-tuned runs; calibration fits, validation reports."""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,math,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;R22=HERE.parent/"round22_921_g1_joint_frozen";sys.path.insert(0,str(R22))
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
core=load("r23_r22_detection_core",R22/"evaluate_frozen_detection.py")
tr=load("r23_e3_train_eval",R22/"train_e3.py")
GROUPS=("A","B","C","D");ROLES=tr.ROLES;ALPHAS=(.01,.05,.10)
CONTRASTS={"B-A":{"B":1,"A":-1},"D-C":{"D":1,"C":-1},"C-A":{"C":1,"A":-1},"D-B":{"D":1,"B":-1},"interaction_(D-B)-(C-A)":{"D":1,"B":-1,"C":-1,"A":1}}
BOOT_METRICS=("AP","pAUC","frame_static_FPR","trial_macro_static_FPR","trial_any_static_alarm_rate","gross_recall","balanced_accuracy","macro_f1","false_starts_per_trial","static_alarming_frames_per_static_support_trial","event_recall","mean_detected_event_delay","preexisting_alarm_event_rate","segment_coverage")

def threshold_values(r13,eps):
 cache=r13.build_fit_cache(eps,1);fp=cache.static_fp.sum(0);tp=cache.gross_tp.sum(0);sn=cache.static_n.sum();gn=cache.gross_n.sum()
 if sn<=0 or gn<=0:raise ValueError("calibration maxBA lacks class support")
 fpr=fp/sn;rec=tp/gn;ba=((1-fpr)+rec)/2;best=max(range(len(cache.thresholds)),key=lambda i:(ba[i],-fpr[i],cache.thresholds[i]));out={"fixed_0.5":.5,"maxBA":float(cache.thresholds[best])}
 for alpha in ALPHAS:out[f"FPR{int(alpha*100)}"]=r13.fit_cached(cache,"trial_macro_static_FPR",alpha,constraint_override="frame_static_FPR")["threshold"]
 return out

def sha(p):
 h=hashlib.sha256();
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def write_csv(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if not rows:p.write_text("");return
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with p.open("w",newline="") as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def weighted_eps(eps,weights):return [e for e in eps for _ in range(weights.get(e.group,0))]
def paired(r13,store,trials,n):
 by=defaultdict(list)
 for x in trials:
  if x["role"]=="validation" and x["policy"]=="FPR5|raw":by[x["group"],x["fold"],x["seed"]].append(x)
 ci=[];draw_rows=[]
 for fold in range(1,5):
  seeds=sorted({k[2] for k in store if k[1]==fold});support=None
  for g in GROUPS:
   for s in seeds:
    got={e.group for e in store[g,fold,s,"validation"]};support=got if support is None else support
    if got!=support:raise ValueError("paired leakage-group support differs")
  units=sorted(support);rng=np.random.default_rng(230000+fold);vals=defaultdict(list)
  for draw in range(n):
   ids=rng.integers(0,len(units),len(units));weights=dict(zip(units,np.bincount(ids,minlength=len(units))));gm=defaultdict(lambda:defaultdict(list))
   for g in GROUPS:
    for s in seeds:
     rank=core.rank(r13,weighted_eps(store[g,fold,s,"validation"],weights));agg=r13.aggregate(by[g,fold,s],weights)
     for metric in BOOT_METRICS:gm[g][metric].append(rank[metric] if metric in rank else agg[metric])
   means={g:{m:float(np.mean(gm[g][m])) for m in BOOT_METRICS} for g in GROUPS}
   for name,coef in CONTRASTS.items():
    for metric in BOOT_METRICS:
     value=sum(c*means[g][metric] for g,c in coef.items());valid=math.isfinite(value)
     if valid:vals[name,metric].append(value)
     draw_rows.append({"fold":fold,"draw":draw,"contrast":name,"metric":metric,"difference":value if valid else "","valid":int(valid),"invalid_reason":"" if valid else "resample_lacks_required_support"})
  for name in CONTRASTS:
   for metric in BOOT_METRICS:
    v=vals[name,metric];ci.append({"fold":fold,"contrast":name,"metric":metric,"attempted_draws":n,"valid_draws":len(v),"invalid_draws":n-len(v),"mean":float(np.mean(v)) if v else "","q025":float(np.quantile(v,.025)) if v else "","median":float(np.quantile(v,.5)) if v else "","q975":float(np.quantile(v,.975)) if v else ""})
 return ci,draw_rows
def aux_rows(meta,role,d,pred):
 if f"{role}_aux_force_pred_n" not in pred:return []
 y=np.asarray(pred[f"{role}_aux_force_gt_n"],float);q=np.asarray(pred[f"{role}_aux_force_pred_n"],float);rows=[]
 for axis,name in enumerate("xyz"):
  e=q[:,axis]-y[:,axis];rows.append({**meta,"role":role,"axis":name,"endpoints":len(e),"MAE_N":float(np.abs(e).mean()),"RMSE_N":float(np.sqrt(np.mean(e*e))),"bias_N":float(e.mean())})
 e=q-y;rows.append({**meta,"role":role,"axis":"all","endpoints":len(e),"MAE_N":float(np.abs(e).mean()),"RMSE_N":float(np.sqrt(np.mean(e*e))),"bias_N":float(e.mean())})
 return rows
def corr(x,y):
 x=np.asarray(x,float);y=np.asarray(y,float);return float(np.corrcoef(x,y)[0,1]) if len(x)>1 and x.std()>0 and y.std()>0 else ""
def main():
 p=argparse.ArgumentParser();p.add_argument("--inventory",type=Path,required=True);p.add_argument("--prediction-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--r13",type=Path,required=True);p.add_argument("--bootstrap-draws",type=int,default=2000);a=p.parse_args();r13=core.load_r13(a.r13);assert r13.smoke()["status"]=="pass"
 inv=json.loads(a.inventory.read_text());runs=inv["runs"]
 if len(runs)!=48 or {(x["group"],x["fold"],x["seed"]) for x in runs}!={(g,f,s) for g in GROUPS for f in range(1,5) for s in tr.SEEDS}:raise ValueError("48-run grid")
 metrics=[];trials=[];thresholds=[];incipient=[];sensitivity=[];auxiliary=[];aux_trials=[];film=[];scores=[];store={};input_index=[]
 for rr in sorted(runs,key=lambda x:(x["fold"],x["seed"],x["group"])):
  po=a.prediction_root/rr["group"]/f"p{rr['fold']}_s{rr['seed']}";ps=json.loads((po/"SUMMARY.json").read_text());pp=po/"PREDICTIONS.npz"
  if ps.get("status")!="complete" or ps["run"]!=rr["run"] or sha(pp)!=ps["prediction_sha256"]:raise ValueError(f"prediction acceptance {rr['run']}")
  pred=np.load(pp,allow_pickle=False);data=torch.load(rr["data"],map_location="cpu",weights_only=False);meta={"group":rr["group"],"fold":rr["fold"],"seed":rr["seed"]};roleeps={};normal={}
  input_index.append({**meta,"run":rr["run"],"prediction":str(pp),"prediction_sha256":ps["prediction_sha256"],"best_commit_sha256":ps["best_commit_sha256"],"fine_tuned_suffix_state_sha256":ps["fine_tuned_suffix_state_sha256"]})
  for role in ROLES:
   d=data["roles"][role];score=np.asarray(pred[f"{role}_score"],float)
   if len(score)!=len(d["t"]) or not np.isfinite(score).all():raise ValueError("score support/nonfinite")
   normal[role]=score;roleeps[role]=core.episodes(r13,d,score);store[rr["group"],rr["fold"],rr["seed"],role]=roleeps[role]
   for e,lg,probe,t,stage,value in zip(d["episode_id"],d["leakage_group"],d["probe"],d["t"].tolist(),d["stage"].tolist(),score.tolist()):scores.append({**meta,"role":role,"episode":e,"leakage_group":lg,"probe":probe,"t":t,"stage":stage,"score":value})
   inc=score[d["stage"].eq(1).numpy()];incipient.append({**meta,"role":role,"frames":len(inc),"mean":float(inc.mean()) if len(inc) else "","std":float(inc.std()) if len(inc) else "","q10":float(np.quantile(inc,.1)) if len(inc) else "","median":float(np.median(inc)) if len(inc) else "","q90":float(np.quantile(inc,.9)) if len(inc) else ""})
   auxiliary.extend(aux_rows(meta,role,d,pred))
   if f"{role}_aux_force_pred_n" in pred:
    ae=np.abs(np.asarray(pred[f"{role}_aux_force_pred_n"],float)-np.asarray(pred[f"{role}_aux_force_gt_n"],float)).mean(1)
    for ep in sorted(set(d["episode_id"])):
     ids=np.asarray([i for i,e in enumerate(d["episode_id"]) if e==ep]);aux_trials.append({**meta,"role":role,"episode":ep,"leakage_group":d["leakage_group"][int(ids[0])],"endpoints":len(ids),"aux_force_MAE_N":float(ae[ids].mean())})
   if f"{role}_film_gamma" in pred:
    g=np.asarray(pred[f"{role}_film_gamma"],float);b=np.asarray(pred[f"{role}_film_beta"],float);ratio=np.asarray(pred[f"{role}_film_norm_ratio"],float);film.append({**meta,"role":role,"endpoints":len(g),"gamma_mean":float(g.mean()),"gamma_abs_q95":float(np.quantile(np.abs(g),.95)),"gamma_abs_max":float(np.abs(g).max()),"beta_mean":float(b.mean()),"beta_abs_q95":float(np.quantile(np.abs(b),.95)),"beta_abs_max":float(np.abs(b).max()),"scale_min":float((1+g).min()),"scale_max":float((1+g).max()),"norm_ratio_q95":float(np.quantile(ratio,.95)),"norm_ratio_max":float(ratio.max()),"anomalous_fraction":float((((np.abs(g)>=.5)|(1+g<.5)|(1+g>1.5)|(np.abs(b)>=3)).any((1,2))|(ratio>2).any(1)).mean())})
  th=threshold_values(r13,roleeps["calibration"])
  for name,v in th.items():thresholds.append({**meta,"policy":name,"threshold":v,"fit_role":"calibration","never_alarm":int(v>1)})
  for name,v in th.items():
   for k,label in ((1,"raw"),(2,"confirm2_release1")):
    policy=f"{name}|{label}"
    for role in ROLES:
     ts,agg=r13.evaluate_policy(roleeps[role],v,k);metrics.append({**meta,"role":role,"policy":policy,"threshold":v,"k":k,"never_alarm":int(v>1),**agg,**core.rank(r13,roleeps[role])});trials.extend({**meta,"role":role,"policy":policy,"threshold":v,"k":k,**x} for x in ts)
  fpr5=th["FPR5"]
  for mode in ("force_mean","force_lag1","film_off"):
   key=f"validation_{mode}_score"
   if key not in pred:continue
   score=np.asarray(pred[key],float);eps=core.episodes(r13,data["roles"]["validation"],score);_,agg=r13.evaluate_policy(eps,fpr5,1);sensitivity.append({**meta,"role":"validation","mode":mode,"threshold_source":"calibration_FPR5_raw","threshold":fpr5,"max_abs_score_change":float(np.max(np.abs(score-normal["validation"]))),**agg,**core.rank(r13,eps)})
 # Persist the expensive prediction-derived core before resampling; a bootstrap interruption
 # never destroys the accepted core evidence.
 a.output.mkdir(parents=True,exist_ok=True);core_outputs={"scores/RAW_SCORES.csv":scores,"metrics/WORKPOINT_METRICS.csv":metrics,"metrics/TRIAL_METRICS.csv":trials,"metrics/CALIBRATION_THRESHOLDS.csv":thresholds,"metrics/INCIPIENT_DISTRIBUTION.csv":incipient,"metrics/AUXILIARY_FORCE_ERROR.csv":auxiliary,"metrics/AUXILIARY_FORCE_TRIAL.csv":aux_trials,"diagnostics/FIXED_SENSITIVITY.csv":sensitivity,"diagnostics/FILM_SUMMARY.csv":film}
 for path,rows in core_outputs.items():write_csv(a.output/path,rows)
 (a.output/"STAGE_STATUS.json").write_text(json.dumps({"schema":"round23_e3_evaluation_stage_v1","stage":"core_complete_bootstrap_pending","runs":48,"test_consumed":False},indent=2)+"\n")
 ci,draws=paired(r13,store,trials,a.bootstrap_draws)
 primary=[x for x in trials if x["role"]=="validation" and x["policy"]=="FPR5|raw"];candidates=[];selected=[]
 for fold in range(1,5):
  for seed in tr.SEEDS:
   by={g:{x["episode"]:x for x in primary if x["fold"]==fold and x["seed"]==seed and x["group"]==g} for g in GROUPS}
   for name,coef in CONTRASTS.items():
    if name.startswith("interaction"):continue
    left,right=name.split("-")
    for ep in sorted(by[left].keys()&by[right].keys()):candidates.append({"fold":fold,"seed":seed,"comparison":name,"episode":ep,"leakage_group":by[left][ep]["leakage_group"],"static_alarm_difference":float(by[left][ep]["static_fp"])-float(by[right][ep]["static_fp"]),"gross_miss_difference":float(by[left][ep]["fn"])-float(by[right][ep]["fn"])})
    pool=[x for x in candidates if x["fold"]==fold and x["seed"]==seed and x["comparison"]==name]
    for metric in ("static_alarm_difference","gross_miss_difference"):
     if pool:
      selected.append({"rule":f"largest_{metric}",**sorted(pool,key=lambda x:(-x[metric],x["episode"]))[0]});selected.append({"rule":f"smallest_{metric}_improvement",**sorted(pool,key=lambda x:(x[metric],x["episode"]))[0]})
   # One preregistered threshold-migration failure per fold/seed.
   shifts=[]
   for g in GROUPS:
    cal=next(x for x in metrics if x["group"]==g and x["fold"]==fold and x["seed"]==seed and x["role"]=="calibration" and x["policy"]=="FPR5|raw");val=next(x for x in metrics if x["group"]==g and x["fold"]==fold and x["seed"]==seed and x["role"]=="validation" and x["policy"]=="FPR5|raw");shifts.append({"group":g,"absolute_FPR_shift":abs(float(val["frame_static_FPR"])-float(cal["frame_static_FPR"])),"calibration_FPR":cal["frame_static_FPR"],"validation_FPR":val["frame_static_FPR"]})
   selected.append({"rule":"largest_calibration_to_validation_FPR_shift","fold":fold,"seed":seed,**sorted(shifts,key=lambda x:(-x["absolute_FPR_shift"],x["group"]))[0]})
 associations=[];auxmap={(x["group"],x["fold"],x["seed"],x["role"],x["episode"]):x for x in aux_trials}
 for g in ("B","D"):
  for fold in range(1,5):
   for seed in tr.SEEDS:
    rr=[x for x in primary if x["group"]==g and x["fold"]==fold and x["seed"]==seed];joined=[(auxmap[g,fold,seed,"validation",x["episode"]],x) for x in rr if (g,fold,seed,"validation",x["episode"]) in auxmap];force=[float(x[0]["aux_force_MAE_N"]) for x in joined];static=[float(x[1]["static_fp"])/float(x[1]["static_frames"]) for x in joined if float(x[1]["static_frames"])>0];static_force=[float(x[0]["aux_force_MAE_N"]) for x in joined if float(x[1]["static_frames"])>0];gross=[float(x[1]["fn"])/float(x[1]["gross_frames"]) for x in joined if float(x[1]["gross_frames"])>0];gross_force=[float(x[0]["aux_force_MAE_N"]) for x in joined if float(x[1]["gross_frames"])>0];associations.append({"group":g,"fold":fold,"seed":seed,"role":"validation","trials":len(joined),"aux_MAE_vs_trial_static_FPR_pearson":corr(static_force,static),"aux_MAE_vs_trial_gross_miss_rate_pearson":corr(gross_force,gross)})
  
 outputs={**core_outputs,"diagnostics/AUXILIARY_FORCE_ASSOCIATION.csv":associations,"bootstrap/PAIRED_CI.csv":ci,"bootstrap/PAIRED_DRAWS.csv":draws,"cases/CANDIDATES.csv":candidates,"cases/PER_FOLD_SEED_SELECTED.csv":selected}
 for path,rows in outputs.items():write_csv(a.output/path,rows)
 (a.output/"INPUT_INDEX.json").write_text(json.dumps({"schema":"round23_e3_evaluation_inputs_v1","runs":input_index,"inventory_sha256":sha(a.inventory),"test_consumed":False},indent=2)+"\n")
 summary={"schema":"round23_e3_evaluation_v1","status":"complete","runs":48,"groups":list(GROUPS),"contrasts":list(CONTRASTS),"bootstrap_draws_per_fold":a.bootstrap_draws,"bootstrap_unit":"complete leakage_group, shared across groups and seeds; three seeds averaged inside each draw","roles":{"fit":"fitting only","selection":"checkpoint choice only","calibration":"threshold fitting only","validation":"evaluation only"},"continuous_rule":"R13 raw and confirm2/release1; gaps reset; incipient remains in timeline but is excluded from binary fitting","never_alarm_and_observed_no_alarm":"reported separately in threshold/workpoint columns","censoring":"left_censored_events and right_boundary_events retained from accepted R13 event evaluator","test_consumed":False,"counts":{k:len(v) for k,v in outputs.items()},"hashes":{str(p.relative_to(a.output)):sha(p) for p in a.output.rglob("*.csv")}}
 (a.output/"SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary))
if __name__=="__main__":main()
