#!/usr/bin/env python3
"""Round-6 re-evaluation of frozen Round-5 future heads and simple causal baselines."""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,math,os
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

HERE=Path(__file__).resolve().parent
R5_CODE=HERE.parents[1]/"round5_force_conditioned_slip"

def _load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
r5=_load("round6_r5_future",R5_CODE/"future/future_pipeline.py")
r5eval=_load("round6_r5_eval",R5_CODE/"analysis_future/evaluate.py")
SEEDS=(20260914,20260915,20260916)
_SHA_CACHE={}

def sha(path:Path)->str:
 path=Path(path);st=path.stat();identity=(str(path.resolve()),st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
 if identity in _SHA_CACHE:return _SHA_CACHE[identity]
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 digest=h.hexdigest();_SHA_CACHE[identity]=digest;return digest

r5eval.sha256=sha

def atomic_json(path,payload):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f".tmp.{os.getpid()}");tmp.write_text(json.dumps(payload,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+"\n");os.replace(tmp,path)

def confusion(y,s,t):
 y=np.asarray(y,dtype=np.int8);p=np.asarray(s)>=t;tp=int(np.sum(p&(y==1)));fp=int(np.sum(p&(y==0)));fn=int(np.sum(~p&(y==1)));tn=int(np.sum(~p&(y==0)));rec=tp/max(1,tp+fn);fpr=fp/max(1,fp+tn);spec=tn/max(1,tn+fp)
 return {"tn":tn,"fp":fp,"fn":fn,"tp":tp,"recall":rec,"fpr":fpr,"balanced_accuracy":(rec+spec)/2,"macro_f1":((2*tp/max(1,2*tp+fp+fn))+(2*tn/max(1,2*tn+fp+fn)))/2}

def metrics(rows,key,t):
 y=np.asarray([r["target"] for r in rows]);s=np.asarray([r[key] for r in rows],dtype=np.float64)
 if len(rows)==0 or set(y.tolist())!={0,1}:
  return {"status":"unavailable","reason":f"{key} cohort lacks both classes","frames":len(rows),"threshold":float(t)}
 if not np.isfinite(s).all() or np.any((s<0)|(s>1)):
  raise ValueError(f"invalid score for {key}")
 return {"status":"available",**confusion(y,s,t),"average_precision":float(average_precision_score(y,s)),"brier":float(np.mean((s-y)**2)),"prevalence":float(y.mean()),"frames":len(y),"threshold":float(t),"never_alarm":bool(np.all(s<t))}

def calibrate(rows,key):
 y=np.asarray([r["target"] for r in rows]);s=np.asarray([r[key] for r in rows]);
 if set(y.tolist())!={0,1}:return None
 if not np.isfinite(s).all() or np.any((s<0)|(s>1)):raise ValueError(f"invalid calibration score for {key}")
 candidates=np.r_[np.unique(s),math.nextafter(1.,math.inf)];order=np.argsort(s,kind="mergesort");ss=s[order];yy=y[order];prefix_pos=np.r_[0,np.cumsum(yy==1)];prefix_neg=np.r_[0,np.cumsum(yy==0)];total_pos=prefix_pos[-1];total_neg=prefix_neg[-1];scored=[]
 for t,i in zip(candidates,np.searchsorted(ss,candidates,side="left")):
  tp=int(total_pos-prefix_pos[i]);fp=int(total_neg-prefix_neg[i]);fn=int(total_pos-tp);tn=int(total_neg-fp);rec=tp/total_pos;fpr=fp/total_neg;spec=tn/total_neg
  scored.append((float(t),{"balanced_accuracy":(rec+spec)/2,"fpr":fpr}))
 return max(scored,key=lambda x:(x[1]["balanced_accuracy"],-x[1]["fpr"],x[0]))[0]

def cluster_bootstrap(rows,key,t,seed=20260915,reps=200):
 groups=defaultdict(list)
 for row in rows:groups[row["cluster"]].append(row)
 names=sorted(groups)
 if not names:return {"status":"unavailable","reason":"no complete clusters"}
 rng=np.random.default_rng(seed);values=defaultdict(list)
 for _ in range(reps):
  sample=[]
  for name in rng.choice(names,len(names),replace=True):sample.extend(groups[name])
  if set(r["target"] for r in sample)!={0,1}:continue
  m=metrics(sample,key,t)
  if m["status"]!="available":continue
  for metric in ("average_precision","balanced_accuracy","macro_f1","brier","fpr","recall"):values[metric].append(m[metric])
 if not values:return {"status":"unavailable","reason":"no bootstrap replicate contained both classes"}
 return {"status":"available",**{k:{"lower":float(np.quantile(v,.025)),"upper":float(np.quantile(v,.975)),"valid_replicates":len(v)} for k,v in values.items()}}

def runs_count(values,times=None):
 values=np.asarray(values,dtype=bool)
 if times is None:continuous=np.ones(len(values),dtype=bool)
 else:
  times=np.asarray(times,dtype=np.int64);continuous=np.r_[False,np.diff(times)==1]
 previous=np.r_[False,values[:-1]] & continuous
 return int(np.sum(values & ~previous))

def per_trial(rows,key,t):
 out=[]
 for episode in sorted(set(r["episode_id"] for r in rows)):
  x=sorted((r for r in rows if r["episode_id"]==episode),key=lambda r:r["t"]);alarm=np.asarray([r[key]>=t for r in x]);negative=np.asarray([r["target"]==0 for r in x])
  times=[r["t"] for r in x];out.append({"episode_id":episode,"cluster":x[0]["cluster"],"frames":len(x),"positive_frames":sum(r["target"] for r in x),"alarm_frames":int(alarm.sum()),"alarm_runs":runs_count(alarm,times),"false_alarm_frames":int(np.sum(alarm&negative)),"false_alarm_runs":runs_count(alarm&negative,times)})
 return out

def event_records(all_rows,key,t,horizon):
 out=[]
 for episode in sorted(set(r["episode_id"] for r in all_rows)):
  x=sorted((r for r in all_rows if r["episode_id"]==episode),key=lambda r:r["t"]);onset=x[0].get("onset");eligible=[r for r in x if r.get("eligible")]
  if onset is None:
   negative=[r for r in eligible if r["target"]==0];alarm=[r[key]>=t for r in negative];out.append({"episode_id":episode,"cluster":x[0]["cluster"],"status":"negative_trial","onset":None,"event_detected":None,"lead_frames":None,"false_alarm_frames":int(sum(alarm)),"false_alarm_runs":runs_count(alarm,[r["t"] for r in negative])});continue
  positives=[r for r in eligible if r["target"]==1]
  negative=[r for r in eligible if r["target"]==0];false=[r[key]>=t for r in negative]
  if not positives:
   out.append({"episode_id":episode,"cluster":x[0]["cluster"],"status":"censored_no_complete_pre_onset_positive","onset":onset,"event_detected":None,"lead_frames":None,"false_alarm_frames":int(sum(false)),"false_alarm_runs":runs_count(false,[r["t"] for r in negative])});continue
  early=sorted(r["t"] for r in positives if r[key]>=t);late=sorted(r["t"] for r in x if onset<=r["t"]<=onset+horizon and r.get(key,0)>=t)
  status="detected_early" if early else ("late" if late else "miss");first=early[0] if early else (late[0] if late else None)
  out.append({"episode_id":episode,"cluster":x[0]["cluster"],"status":status,"onset":onset,"event_detected":bool(early),"first_in_window_alarm":first,"lead_frames_from_first_in_window_alarm":onset-early[0] if early else None,"lead_frames":onset-early[0] if early else None,"false_alarm_frames":int(sum(false)),"false_alarm_runs":runs_count(false,[r["t"] for r in negative])})
 return out

def event_bootstrap(records,seed,reps=200):
 groups=defaultdict(list)
 for row in records:
  if row["event_detected"] is not None:groups[row["cluster"]].append(row)
 names=sorted(groups)
 if not names:return {"status":"unavailable","reason":"no uncensored eligible event clusters"}
 rng=np.random.default_rng(seed);recalls=[];leads=[]
 for _ in range(reps):
  sample=[]
  for name in rng.choice(names,len(names),replace=True):sample.extend(groups[name])
  recalls.append(float(np.mean([r["event_detected"] for r in sample])))
  hit_leads=[r["lead_frames"] for r in sample if r["lead_frames"] is not None]
  if hit_leads:leads.append(float(np.mean(hit_leads)))
 interval=lambda v:None if not v else {"lower":float(np.quantile(v,.025)),"upper":float(np.quantile(v,.975)),"valid_replicates":len(v)}
 return {"status":"available","event_recall":interval(recalls),"lead_frames_among_early_hits":interval(leads),"clusters":len(names)}

def group_roles(domain,groups):
 ordered=sorted(set(groups),key=lambda g:hashlib.sha256(f"round6|{domain}|{g}".encode()).hexdigest());roles={}
 for rank,g in enumerate(ordered):
  if domain=="source":roles[g]=("selection" if rank%5==0 else ("calibration" if rank%5==1 else "fit_train"))
  else:roles[g]="selection" if rank%5==0 else "fit_train"
 return roles

def position_fit(train_rows,target_rows):
 max_t=max(r["t"] for r in train_rows)
 if max_t<=0:raise ValueError("position fit requires positive train time scale")
 feat=lambda rows:np.asarray([[min(r["t"]/max_t,1),min(r["t"]/max_t,1)**2] for r in rows]);y=np.asarray([r["target"] for r in train_rows])
 if set(y.tolist())!={0,1}:raise ValueError("position fit lacks both classes")
 model=LogisticRegression(C=1,class_weight="balanced",solver="lbfgs",random_state=20260915,max_iter=1000).fit(feat(train_rows),y);return model.predict_proba(feat(target_rows))[:,1]

def force_rule_fit(train_rows,target_rows):
 values=np.asarray([r["force_change_magnitude"] for r in train_rows]);center=float(np.median(values));scale=max(float(np.quantile(values,.75)-np.quantile(values,.25)),1e-6)
 scores=1/(1+np.exp(-np.clip((np.asarray([r["force_change_magnitude"] for r in target_rows])-center)/scale,-60,60)))
 return scores,{"train_median":center,"train_iqr_floor":scale}

def add_simple_scores(train_rows,*cohorts):
 all_cohorts=(train_rows,)+cohorts
 for rows in all_cohorts:
  for r in rows:r["history_mean4"]=float(np.mean(r["history"]));r["history_trend"]=float(np.clip(r["history"][-1]+r["history"][-1]-r["history"][0],0,1));r["current_slip"]=float(r["history"][-1])
 all_rows=[r for rows in all_cohorts for r in rows];pos=position_fit(train_rows,all_rows);force,norm=force_rule_fit(train_rows,all_rows)
 for r,p,f in zip(all_rows,pos,force):r["sequence_position"]=float(p);r["force_delta_rule"]=float(f)
 return norm

def canonical_source_episode(meta):
 return f"source/{meta['dataset']}/{meta['trajectory']}"

def support_endpoints(manifest,domain,horizon,fold=None):
 if manifest.get("status")!="complete":raise ValueError(f"{domain} support manifest incomplete")
 root=manifest["horizons"][str(horizon)] if domain=="source" else manifest["folds"][fold]["horizons"][str(horizon)]
 endpoints={};onsets={}
 for role,block in root["roles"].items():
  for trial in block["trials"]:
   eid=str(trial["episode_id"]);onset=trial.get("first_current_slip_t") if domain=="source" else trial.get("first_gross_proxy_t")
   if eid in onsets and onsets[eid]!=onset:raise ValueError("support onset drift")
   onsets[eid]=onset
   for target,name in ((1,"positive_endpoints"),(0,"negative_endpoints")):
    for t in trial[name]:
     key=(eid,int(t))
     if key in endpoints:raise ValueError(f"duplicate support endpoint {key}")
     endpoints[key]={"role":role,"target":target,"onset":onset}
  summary=block["summary"]
  selected=[v for v in endpoints.values() if v["role"]==role]
  if sum(v["target"] for v in selected)!=int(summary["positive_endpoints"]) or sum(1-v["target"] for v in selected)!=int(summary["negative_endpoints"]):
   raise ValueError(f"support endpoint count mismatch {domain}/{horizon}/{role}")
 return endpoints,onsets

def source_rows(cache_path,raw_path,horizon,support_map):
 cache=torch.load(cache_path,map_location="cpu",weights_only=False);raw=torch.load(raw_path,map_location="cpu",weights_only=False);hidx=cache["horizons"].index(horizon);lookup={}
 for i,m in enumerate(raw["metadata"]):lookup[(str(m["dataset"]),str(m["trajectory"]),int(m["sample"]))]=i
 rows=[]
 for i,m in enumerate(cache["metadata"]):
  group=canonical_source_episode(m);t=int(m["sample"]);identity=(group,t)
  if identity not in support_map:continue
  idx=[lookup.get((str(m["dataset"]),str(m["trajectory"]),t-j)) for j in range(3,-1,-1)]
  if any(x is None for x in idx):continue
  current=lookup[(str(m["dataset"]),str(m["trajectory"]),t)];previous=lookup.get((str(m["dataset"]),str(m["trajectory"]),t-5))
  if previous is None:continue
  f=raw["force_pred_n"][current].float();fp=raw["force_pred_n"][previous].float();dfn=float(torch.abs(f[2])-torch.abs(fp[2]));dft=float(torch.linalg.vector_norm(f[:2])-torch.linalg.vector_norm(fp[:2]))
  spec=support_map[identity];cache_target=int(cache["future_slip"][i,hidx])
  if cache_target!=spec["target"]:raise ValueError(f"source support/cache target mismatch {identity}")
  current_label=int(raw["current_slip"][current])
  if current_label!=0:raise ValueError(f"source support endpoint is not current-static {identity}")
  rows.append({"episode_id":group,"cluster":group,"t":t,"role":spec["role"],"target":cache_target,"onset":spec["onset"],"history":[float(raw["slip_probs"][j,1]) for j in idx],"force_change_magnitude":float(math.hypot(dfn,dft)),"cache_index":i,"raw_index":current,"current_slip_label":current_label})
 return cache,rows

def eval_method(domain,variant,seed,horizon,method,cal_rows,val_rows,contaminated,events=None):
 thresholds=[("fixed_0.5",.5)];selected=calibrate(cal_rows,method) if cal_rows else None
 if selected is not None:thresholds.append(("calibration_max_ba",selected))
 results=[]
 for opname,t in thresholds:
  result={"domain":domain,"variant":variant,"seed":seed,"horizon":horizon,"method":method,"operating_point":opname,"calibration_status":"legacy_seen_contaminated" if contaminated and opname!="fixed_0.5" else ("independent_role" if opname!="fixed_0.5" else "fixed"),"validation":metrics(val_rows,method,t),"cluster_bootstrap_95ci":cluster_bootstrap(val_rows,method,t,20260915+(seed or 0)+horizon),"per_trial":per_trial(val_rows,method,t)}
  if events is not None:
   records=event_records(events,method,t,horizon);valid=[r for r in records if r["event_detected"] is not None];positive=[r for r in valid if r["onset"] is not None]
   result["events"]={"definition":"first-gross HTT label proxy; not independent physical truth","all_onset_events":sum(r["onset"] is not None for r in records),"eligible_event_denominator":len(positive),"event_recall":None if not positive else sum(r["event_detected"] for r in positive)/len(positive),"misses":sum(r["status"]=="miss" for r in records),"late_diagnostic":sum(r["status"]=="late" for r in records),"late_scope":"post-onset full-trace diagnostic, excluded from strict primary eligible cohort","censored":sum(r["status"].startswith("censored") for r in records),"lead_frames_among_early_hits":[r["lead_frames"] for r in records if r.get("lead_frames") is not None],"cluster_bootstrap_95ci":event_bootstrap(records,20260915+(seed or 0)+horizon),"records":records}
  results.append(result)
 return results

def source_evaluation(args,inventory,support_manifest):
 protocol_sha=sha(args.r5_future_protocol);train_raw=args.source_train_raw;val_raw=args.source_val_raw
 results=[];all_trial=[];counts={};unavailable=[];predictions={}
 for horizon in (1,3,5):
  endpoint_map,_=support_endpoints(support_manifest,"source",horizon)
  train_cache,train_rows=source_rows(args.source_train_cache,train_raw,horizon,endpoint_map);val_cache,val_rows=source_rows(args.source_val_cache,val_raw,horizon,endpoint_map)
  fit=[r for r in train_rows if r["role"]=="fit_train"];cal=[r for r in train_rows if r["role"]=="calibration"];selection=[r for r in train_rows if r["role"]=="selection"]
  if any(r["role"]!="outer" for r in val_rows):raise ValueError("source validation endpoint role drift")
  norm=add_simple_scores(fit,cal,selection,val_rows);counts[str(horizon)]={role:len([r for r in train_rows if r["role"]==role]) for role in ("fit_train","selection","calibration")}|{"outer":len(val_rows),"groups":{role:len(set(r["cluster"] for r in train_rows if r["role"]==role)) for role in ("fit_train","selection","calibration")}|{"outer":len(set(r["cluster"] for r in val_rows))}}
  for method in ("current_slip","history_mean4","history_trend","force_delta_rule","sequence_position"):
   out=eval_method("source","shared_baseline",None,horizon,method,cal,val_rows,True);results.extend(out);all_trial.extend(({"domain":"source",**{k:v for k,v in row.items() if k!="records"}}) for item in out for row in item["per_trial"])
  for variant in r5.SOURCE_VARIANTS:
   for seed in SEEDS:
    if (variant,seed) not in predictions:
     summary,ckpt=r5eval.verified_run(args.r5_source_runs/variant/f"seed_{seed}","source",variant,seed,protocol_sha,False,inventory)
     if summary["run_config"].get("train_cache_sha256")!=sha(args.source_train_cache) or summary["run_config"].get("val_cache_sha256")!=sha(args.source_val_cache):raise ValueError("source run/cache drift")
     _,ptr=r5eval.infer_source(args.source_train_cache,ckpt,variant);_,pva=r5eval.infer_source(args.source_val_cache,ckpt,variant)
     predictions[(variant,seed)]=(summary,ptr,pva)
    summary,ptr,pva=predictions[(variant,seed)]
    for row in train_rows:row["raw_future"]=float(ptr[row["cache_index"],train_cache["horizons"].index(horizon)]);row["raw_mul_current_slip"]=row["raw_future"]*float(row["history"][-1])
    for row in val_rows:row["raw_future"]=float(pva[row["cache_index"],val_cache["horizons"].index(horizon)]);row["raw_mul_current_slip"]=row["raw_future"]*float(row["history"][-1])
    for method in ("raw_future","raw_mul_current_slip"):
     out=eval_method("source",variant,seed,horizon,method,cal,val_rows,True);out[0]["run_provenance"]=summary["_verified_provenance"]
     if len(out)>1:out[1]["run_provenance"]=summary["_verified_provenance"]
     results.extend(out);all_trial.extend(({"domain":"source",**row} for item in out for row in item["per_trial"]))
  counts[str(horizon)]["force_rule_train_normalization"]=norm
 unavailable.append({"metric":"physical first-onset event recall/lead/miss/late","reason":"support derives a dataset-label current-slip onset from a contiguous timeline, but it is not independently verified physical slip truth; this lane conservatively reports frame/trial future-any-slip only"})
 return {"scope":"source H1/H3/H5 on the support-listed current-static, pre-first-current-slip subset intersected with the Round-5 exact-lag cache; original validation outer","counts":counts,"results":results,"trial_rows":all_trial,"event_metrics_unavailable":unavailable,"calibration_limit":"derived calibration is from original train already seen by frozen legacy heads; calibrated values are contaminated development diagnostics","support_intersection_limit":"support permits endpoints from t>=3, while the compatible Round-5 predicted-delta cache requires t>=10 and exact t-5/t-10; only the intersection is evaluated"}

def htt_rows(payload,horizon,split_groups,support_map):
 all_rows=[]
 for role in ("train","calibration","validation"):
  for e in r5eval.htt_examples(payload,horizon,role):
   cluster=split_groups[e["episode_id"]];base=e["base_x"][-1];spec=support_map.get((e["episode_id"],int(e["t"])))
   if spec is not None:
    if not e["eligible"] or int(e["target"])!=spec["target"] or e["onset"]!=spec["onset"]:raise ValueError(f"HTT support/R5 endpoint mismatch {e['episode_id']}/{e['t']}")
   row={"episode_id":e["episode_id"],"cluster":cluster,"t":e["t"],"source_role":role,"role":spec["role"] if spec else role,"eligible":spec is not None,"target":spec["target"] if spec else e["target"],"onset":spec["onset"] if spec else e["onset"],"history":[float(x[-1]) for x in e["risk_x"]],"force_change_magnitude":float(torch.linalg.vector_norm(base[-2:])) ,"example":e}
   all_rows.append(row)
 found={(r["episode_id"],r["t"]) for r in all_rows if r["eligible"]}
 missing=set(support_map)-found
 if missing:raise ValueError(f"HTT support endpoints missing from R5 compatible examples: {len(missing)}")
 return all_rows

def htt_evaluation(args,inventory,support_manifest):
 payload=r5.canonical_htt_payload(args.htt_upstream,require_formal=True);manifest=json.loads(args.split_manifest.read_text());groups={r["id"]:r["leakage_group"] for r in manifest["episodes"]}
 support_map,_=support_endpoints(support_manifest,"htt",8,"htt_leave_p1")
 rows=htt_rows(payload,8,groups,support_map);fit=[r for r in rows if r["role"]=="fit_train" and r["eligible"]];selection=[r for r in rows if r["role"]=="selection" and r["eligible"]];cal=[r for r in rows if r["role"]=="calibration" and r["eligible"]];val=[r for r in rows if r["role"]=="outer" and r["eligible"]];val_all=[r for r in rows if r["source_role"]=="validation"]
 norm=add_simple_scores(fit,cal,selection,val_all);results=[];all_trial=[];events=[]
 for method in ("current_slip","history_mean4","history_trend","force_delta_rule","sequence_position"):
  out=eval_method("htt","shared_baseline",None,8,method,cal,val,False,val_all);results.extend(out);all_trial.extend(({"domain":"htt",**row} for item in out for row in item["per_trial"]));events.extend(({"domain":"htt","variant":"shared_baseline","seed":None,"horizon":8,"method":method,"operating_point":item["operating_point"],**row} for item in out for row in item["events"]["records"]))
 protocol_sha=sha(args.r5_future_protocol)
 for variant in r5.HTT_VARIANTS:
  for seed in SEEDS:
   summary,ckpt=r5eval.verified_run(args.r5_htt_runs/variant/f"seed_{seed}","htt",variant,seed,protocol_sha,False,inventory)
   if int(summary.get("horizon",-1))!=8:raise ValueError("legacy HTT run is not fixed H8")
   probs,_=r5eval.infer_htt([r["example"] for r in rows],ckpt,variant)
   for r,p in zip(rows,probs):r["raw_future"]=float(p);r["raw_mul_current_slip"]=float(p)*float(r["history"][-1])
   for method in ("raw_future","raw_mul_current_slip"):
    out=eval_method("htt",variant,seed,8,method,cal,val,False,val_all)
    for item in out:item["run_provenance"]=summary["_verified_provenance"]
    results.extend(out);all_trial.extend(({"domain":"htt",**row} for item in out for row in item["per_trial"]));events.extend(({"domain":"htt","variant":variant,"seed":seed,"horizon":8,"method":method,"operating_point":item["operating_point"],**row} for item in out for row in item["events"]["records"]))
 counts={role:{"frames":len(x),"groups":len(set(r["cluster"] for r in x)),"positive_frames":sum(r["target"] for r in x)} for role,x in (("fit_train",fit),("selection",selection),("calibration",cal),("outer_validation",val))}
 return {"scope":"HTT fixed old-head H8, strict pre-first-gross current-static endpoints; first-gross is label proxy","counts":counts,"force_rule_train_normalization":norm,"results":results,"trial_rows":all_trial,"event_rows":events,"legacy_model_selection_limit":"Round-5 checkpoints were selected on this validation role; results are development re-analysis, not independent confirmation"}

def write_csv(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 keys=sorted(set().union(*(r.keys() for r in rows))) if rows else []
 tmp=path.with_name(path.name+f".tmp.{os.getpid()}")
 with tmp.open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
 os.replace(tmp,path)

def charts(results,out):
 import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
 rows=[r for r in results if r["operating_point"]=="fixed_0.5"]
 methods=sorted(set(r["method"] for r in rows));domains=("source","htt");fig,axes=plt.subplots(1,2,figsize=(13,4.5))
 for ax,domain in zip(axes,domains):
  vals=[]
  for method in methods:
   x=[r["validation"]["average_precision"] for r in rows if r["domain"]==domain and r["method"]==method and r["validation"].get("status")=="available"];vals.append(float(np.mean(x)) if x else np.nan)
  ax.bar(range(len(methods)),vals);ax.set_xticks(range(len(methods)),methods,rotation=35,ha="right");ax.set_ylabel("mean AP at fixed score set");ax.set_title(domain);ax.set_ylim(0,1)
 fig.tight_layout();path=out/"fixed_0p5_ap.png";tmp=path.with_name(path.stem+f".tmp.{os.getpid()}"+path.suffix);fig.savefig(tmp,dpi=160);plt.close(fig);os.replace(tmp,path);return path

def main():
 p=argparse.ArgumentParser();p.add_argument("--r5-inventory",type=Path,required=True);p.add_argument("--r5-future-protocol",type=Path,required=True);p.add_argument("--source-train-cache",type=Path,required=True);p.add_argument("--source-val-cache",type=Path,required=True);p.add_argument("--source-train-raw",type=Path,required=True);p.add_argument("--source-val-raw",type=Path,required=True);p.add_argument("--r5-source-runs",type=Path,required=True);p.add_argument("--htt-upstream",type=Path,required=True);p.add_argument("--split-manifest",type=Path,required=True);p.add_argument("--r5-htt-runs",type=Path,required=True);p.add_argument("--support-decision",type=Path,required=True);p.add_argument("--support-manifests",type=Path,required=True);p.add_argument("--test-evidence",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();out=a.output.resolve();inventory=json.loads(a.r5_inventory.read_text());support=json.loads(a.support_decision.read_text());tests=json.loads(a.test_evidence.read_text())
 if tests.get("status")!="pass" or tests.get("source_hashes",{}).get(str(Path(__file__).resolve()))!=sha(Path(__file__).resolve()):raise ValueError("test evidence does not cover current evaluator")
 if support.get("format")!="round6_future_support_decision_v1" or support.get("status")!="complete" or support.get("test_content_read") is not False:raise ValueError("support decision incomplete or consumed test")
 support_paths={name:a.support_manifests/name for name in ("source_support_manifest.json","htt_support_manifest.json")}
 support_payloads={name:json.loads(path.read_text()) for name,path in support_paths.items()}
 for domain,name in (("source","source_support_manifest.json"),("htt","htt_support_manifest.json")):
  declared=support["role_manifests"][domain]
  if Path(declared["path"]).resolve()!=support_paths[name].resolve() or sha(support_paths[name])!=declared["sha256"]:raise ValueError(f"support role-manifest identity drift: {domain}")
 source=source_evaluation(a,inventory,support_payloads["source_support_manifest.json"]);htt=htt_evaluation(a,inventory,support_payloads["htt_support_manifest.json"]);results=source["results"]+htt["results"]
 frame=[]
 for r in results:frame.append({"domain":r["domain"],"variant":r["variant"],"seed":r["seed"],"horizon":r["horizon"],"method":r["method"],"operating_point":r["operating_point"],"calibration_status":r["calibration_status"],**r["validation"]})
 write_csv(out/"frame_metrics.csv",frame);write_csv(out/"trial_metrics.csv",source["trial_rows"]+htt["trial_rows"]);write_csv(out/"event_metrics.csv",htt["event_rows"]);plot=charts(results,out)
 inputs=[a.r5_inventory,a.r5_future_protocol,a.source_train_cache,a.source_val_cache,a.source_train_raw,a.source_val_raw,a.htt_upstream,a.split_manifest,a.support_decision,a.test_evidence,HERE/"protocol.json",HERE/"PROTOCOL.md",HERE/"PROTOCOL_CLARIFICATION.json",Path(__file__)]
 support_files=sorted(a.support_manifests.rglob("*.json"));inputs+=support_files
 summary={"status":"complete","format":"round6_legacy_future_reevaluation_v1","source":source,"htt":htt,"frame_result_rows":len(frame),"trial_result_rows":len(source["trial_rows"])+len(htt["trial_rows"]),"event_result_rows":len(htt["event_rows"]),"tests":{"path":str(a.test_evidence.resolve()),"sha256":sha(a.test_evidence),**tests},"source_hashes":{str(x.resolve()):sha(x.resolve()) for x in inputs},"output_hashes":{"frame_metrics.csv":sha(out/"frame_metrics.csv"),"trial_metrics.csv":sha(out/"trial_metrics.csv"),"event_metrics.csv":sha(out/"event_metrics.csv"),"fixed_0p5_ap.png":sha(plot)},"support_decision_status":support.get("status"),"limitations":["source support provides a contiguous dataset-label current-slip onset, but no independently verified physical onset; physical event lead-time is not claimed","HTT onset is a partly force-rule-generated label proxy","legacy model selection used validation; outer re-evaluation is development evidence","source derived calibration was already seen by legacy models","manual-rule Brier is descriptive score error, not a calibration claim","no test role consumed"]}
 atomic_json(out/"summary.json",summary);print(json.dumps({"status":"complete","frame_rows":len(frame),"trial_rows":summary["trial_result_rows"],"event_rows":summary["event_result_rows"],"summary":str(out/"summary.json")},indent=2))
if __name__=="__main__":main()
