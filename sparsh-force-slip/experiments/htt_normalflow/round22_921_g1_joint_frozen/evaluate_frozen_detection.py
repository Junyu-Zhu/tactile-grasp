#!/usr/bin/env python3
"""Locked E1/E2 detection evaluation, runnable on smoke or accepted formal outputs."""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,math,sys
from collections import defaultdict
from pathlib import Path
import numpy as np,torch
import train_frozen as T
ALPHAS=(.01,.05,.10);PAIRS=(("M","V"),("C","V"),("M","C"),("MB","M"),("MB","V"),("MB","C"));ROLES=T.ROLES
def sha(p):
 h=hashlib.sha256();h.update(Path(p).read_bytes());return h.hexdigest()
def write_csv(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if not rows:p.write_text("");return
 fields=list(dict.fromkeys(k for r in rows for k in r));
 with p.open("w",newline="") as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def load_r13(p):
 s=importlib.util.spec_from_file_location("r22_r13",p);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m
def episodes(r13,r,scores):
 out=[]
 for eid in sorted(set(r["episode_id"])):
  ids=np.asarray([i for i,e in enumerate(r["episode_id"]) if e==eid]);ids=ids[np.argsort(r["t"][ids].numpy())];out.append(r13.Episode(eid,r["leakage_group"][int(ids[0])],r["probe"][int(ids[0])],r["t"][ids].numpy(),r["stage"][ids].numpy(),scores[ids]))
 return out
def infer(m,x,device,mode="normal"):
 x=x.clone()
 if mode=="force_mean":x[...,192:195]=0
 elif mode=="force_lag1":x[...,1:,192:195]=x[...,:-1,192:195].clone();x[...,0,192:195]=0
 out=[];films=[];m.eval()
 with torch.inference_mode():
  for z in x.split(1024):
   z=z.to(device)
   if mode=="film_off":v=m.visual_ln(z[...,:192]);logit=m.risk(m.gru(v)[0][:,-1]).squeeze(-1);comp=None
   else:comp=m.components(z);logit=comp["logit"]
   out.append(torch.sigmoid(logit).cpu())
   if comp is not None and m.group in ("M","MB"):films.append((comp["gamma"].cpu(),comp["beta"].cpu()))
 return torch.cat(out).numpy(),films
def rank(r13,eps):
 if not eps:return {"AP":float("nan"),"pAUC":float("nan"),"positive_prevalence":float("nan")}
 stage=torch.from_numpy(np.concatenate([e.stage[e.stage!=1] for e in eps]));score=torch.from_numpy(np.concatenate([e.score[e.stage!=1] for e in eps]));labels=stage.eq(2)
 if not labels.any() or labels.all():return {"AP":float("nan"),"pAUC":float("nan"),"positive_prevalence":float("nan")}
 base=r13.rank_metrics(eps);base["pAUC"]=T.low_fpr_auc(stage,score);return base
def maxba(r13,eps):
 best=None
 for th in r13.candidate_thresholds(eps,1):
  _,a=r13.evaluate_policy(eps,float(th),1);key=(a["balanced_accuracy"],-a["frame_static_FPR"],float(th));best=(key,float(th)) if best is None or key>best[0] else best
 return best[1]
def thresholds(r13,eps):
 out={"fixed_0.5":.5,"maxBA":maxba(r13,eps)};cache=r13.build_fit_cache(eps,1)
 for alpha in ALPHAS:out[f"FPR{int(alpha*100)}"]=r13.fit_cached(cache,"trial_macro_static_FPR",alpha,constraint_override="frame_static_FPR")["threshold"]
 return out
def paired_ci(r13,store,trials,draws_n):
 bytrial=defaultdict(list)
 for r in trials:
  if r["role"]=="validation" and r["policy"]=="FPR5|raw":bytrial[r["group"],r["fold"],r["seed"]].append(r)
 rows=[]
 for fold in sorted({k[1] for k in store}):
  available=sorted({k[0] for k in store if k[1]==fold});seeds=sorted({k[2] for k in store if k[1]==fold});groups=sorted({e.group for e in store[available[0],fold,seeds[0],"validation"]});rng=np.random.default_rng(220000+fold);draws=rng.integers(0,len(groups),size=(draws_n,len(groups)))
  for left,right in PAIRS:
   if left not in available or right not in available:continue
   for metric in ("AP","pAUC","frame_static_FPR","gross_recall","balanced_accuracy","macro_f1"):
    vals=[]
    for ids in draws:
     weights=dict(zip(groups,np.bincount(ids,minlength=len(groups))));seedvals=[]
     for seed in seeds:
      if metric in ("AP","pAUC"):
       def weighted(g):return [e for e in store[g,fold,seed,"validation"] for _ in range(weights.get(e.group,0))]
       lv=rank(r13,weighted(left))[metric];rv=rank(r13,weighted(right))[metric]
      else:lv=r13.aggregate(bytrial[left,fold,seed],weights)[metric];rv=r13.aggregate(bytrial[right,fold,seed],weights)[metric]
      seedvals.append(lv-rv)
     v=float(np.mean(seedvals))
     if math.isfinite(v):vals.append(v)
    rows.append({"fold":fold,"left":left,"right":right,"metric":metric,"draws":draws_n,"valid":len(vals),"invalid":draws_n-len(vals),"mean":float(np.mean(vals)) if vals else None,"q025":float(np.quantile(vals,.025)) if vals else None,"q975":float(np.quantile(vals,.975)) if vals else None})
 return rows
def main():
 p=argparse.ArgumentParser();p.add_argument("--inventory",type=Path);p.add_argument("--smoke-root",type=Path);p.add_argument("--smoke-data",type=Path);p.add_argument("--output",type=Path,required=True);p.add_argument("--r13",type=Path,required=True);p.add_argument("--device",default="cpu");p.add_argument("--bootstrap-draws",type=int,default=2000);a=p.parse_args();r13=load_r13(a.r13);assert r13.smoke()["status"]=="pass";runs=[]
 if a.smoke_root:
  for g in T.GROUPS:runs.append({"package":"E2" if g=="MB" else "E1","group":g,"fold":1,"seed":20260914,"data":str(a.smoke_data),"output":str(a.smoke_root/"frozen"/g)})
 else:runs=[r for r in json.loads(a.inventory.read_text())["runs"] if r["package"] in ("E1","E2")]
 metrics=[];trials=[];threshold_rows=[];diagnostics=[];store={};candidates=[]
 for rr in runs:
  out=Path(rr["output"]);summary=json.loads((out/"summary.json").read_text());cm=json.loads((out/"COMMIT.json").read_text());bp=out/cm["best"]["path"];ck=torch.load(bp,map_location="cpu",weights_only=False)
  ident=summary["identity"];expected={"group":rr["group"],"fold":rr["fold"],"seed":rr["seed"],"data_sha256":sha(rr["data"]),"source_sha256":rr.get("trainer_sha256",sha(T.__file__)),**rr.get("dependencies",{})}
  if not a.smoke_root:expected.update({"protocol_sha256":rr["protocol_sha256"],"formal":True})
  else:expected["formal"]=False
  if summary["status"]!="complete" or ck["identity"]!=ident or sha(bp)!=cm["best"]["sha256"] or any(ident.get(k)!=v for k,v in expected.items()):raise ValueError("run acceptance")
  d=torch.load(rr["data"],map_location="cpu",weights_only=False);T.validate(d,rr["data"],rr["fold"],rr["seed"],Path(T.__file__).with_name("PREPARE_CURRENT_SUPPORT.json"));m=T.init_model(rr["group"],rr["seed"]);m.load_state_dict(ck["model"]);m.to(a.device);norm=ck["normalizer"];meta={k:rr[k] for k in ("package","group","fold","seed")};roleeps={};normal_scores={}
  for role in ROLES:
   x=(d["roles"][role]["x"]-norm["mean"])/norm["std"];score,film=infer(m,x,a.device);normal_scores[role]=score;roleeps[role]=episodes(r13,d["roles"][role],score);store[rr["group"],rr["fold"],rr["seed"],role]=roleeps[role]
   if film:
    gamma=np.concatenate([x[0].numpy() for x in film]);beta=np.concatenate([x[1].numpy() for x in film]);diagnostics.append({**meta,"role":role,"diagnostic":"film","gamma_abs_max":float(np.abs(gamma).max()),"beta_abs_max":float(np.abs(beta).max()),"scale_min":float((1+gamma).min()),"scale_max":float((1+gamma).max())})
  th=thresholds(r13,roleeps["calibration"])
  for name,v in th.items():threshold_rows.append({**meta,"policy":name,"threshold":v,"fit_role":"calibration","never_alarm":v>1})
  for name,v in th.items():
   for k,rulename in ((1,"raw"),(2,"confirm2_release1")):
    for role in ROLES:
     ts,agg=r13.evaluate_policy(roleeps[role],v,k);policy=f"{name}|{'raw' if k==1 else 'confirm2_release1'}";metrics.append({**meta,"role":role,"policy":policy,"threshold":v,**agg,**rank(r13,roleeps[role])});trials.extend({**meta,"role":role,"policy":policy,"threshold":v,**x} for x in ts)
  fpr5=th["FPR5"]
  for mode in (("force_mean","force_lag1") if rr["group"] in ("V","C") else ("force_mean","force_lag1","film_off")):
   x=(d["roles"]["validation"]["x"]-norm["mean"])/norm["std"];score,_=infer(m,x,a.device,mode);eps=episodes(r13,d["roles"]["validation"],score);_,agg=r13.evaluate_policy(eps,fpr5,1);diagnostics.append({**meta,"role":"validation","diagnostic":mode,"max_abs_score_change":float(np.max(np.abs(score-normal_scores["validation"]))),**agg})
 ci=paired_ci(r13,store,trials,a.bootstrap_draws);fpr5tr=[x for x in trials if x["role"]=="validation" and x["policy"]=="FPR5|raw"]
 for fold in sorted({x["fold"] for x in fpr5tr}):
  for seed in sorted({x["seed"] for x in fpr5tr if x["fold"]==fold}):
   by={g:{x["episode"]:x for x in fpr5tr if x["fold"]==fold and x["seed"]==seed and x["group"]==g} for g in T.GROUPS}
   for left,right in (("MB","M"),("M","V")):
    if not by[left] or not by[right]:continue
    for ep in sorted(by[left].keys()&by[right].keys()):candidates.append({"fold":fold,"seed":seed,"episode":ep,"comparison":f"{left}-{right}","static_alarm_difference":by[left][ep]["static_fp"]-by[right][ep]["static_fp"],"gross_miss_difference":by[left][ep]["fn"]-by[right][ep]["fn"]})
 selected=[]
 for metric in ("static_alarm_difference","gross_miss_difference"):
  if candidates:selected.append({"rule":f"largest_{metric}",**sorted(candidates,key=lambda x:(-x[metric],x["fold"],x["seed"],x["episode"],x["comparison"]))[0]})
 a.output.mkdir(parents=True,exist_ok=False);write_csv(a.output/"WORKPOINT_METRICS.csv",metrics);write_csv(a.output/"TRIAL_METRICS.csv",trials);write_csv(a.output/"CALIBRATION_THRESHOLDS.csv",threshold_rows);write_csv(a.output/"PAIRED_CI.csv",ci);write_csv(a.output/"DIAGNOSTICS.csv",diagnostics);write_csv(a.output/"CASE_CANDIDATES.csv",candidates);(a.output/"SELECTED_CASES.json").write_text(json.dumps({"rules":"fixed before formal results; ties fold,seed,episode,comparison lexical","selected":selected,"source_images":"not generated by this core evaluator; full reporting step resolves accepted source_path or records missing"},indent=2)+"\n");summary={"schema":"round22_detection_evaluation_v1","status":"entrypoint_smoke_complete" if a.smoke_root else "core_tables_complete_full_reporting_pending","runs":len(runs),"metrics":len(metrics),"trials":len(trials),"ci":len(ci),"ci_scope":"complete-leakage-group paired bootstrap at calibration-FPR5 raw only","bootstrap_draws":a.bootstrap_draws,"continuous_rule":"historical R13 state machine k=2 confirm, release after first below-threshold frame, gaps reset; verified by r13.smoke","force_lag1_rule":"normalized force history shifted one position; first position set to fit mean (normalized zero)","test_consumed":False,"pAUC":"corrected current-round implementation"};(a.output/"SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary))
if __name__=="__main__":main()
