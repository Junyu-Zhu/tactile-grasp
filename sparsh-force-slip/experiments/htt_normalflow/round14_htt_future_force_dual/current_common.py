#!/usr/bin/env python3
"""State-first R13 current-slip reevaluation on Round-14 common future endpoints."""
import argparse,csv,importlib.util,json,math,sys
from pathlib import Path
import torch,numpy as np
import future_train as ft
def readcsv(p):return list(csv.DictReader(open(p)))
def writecsv(p,rows):
  with p.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def stats_from_state(ep,alarm,starts):
 static=ep.stage==0;gross=ep.stage==2;incipient=ep.stage==1;breaks=np.r_[True,np.diff(ep.t)!=1];events=hits=left=right=covered=prealarm=prealarm0=delay_sum=delay_n=0
 for i in range(len(ep.t)):
  if not gross[i] or (i and not breaks[i] and gross[i-1]):continue
  j=i+1
  while j<len(ep.t) and not breaks[j] and gross[j]:j+=1
  hit=np.flatnonzero(alarm[i:j]);covered+=int(bool(len(hit)));is_left=bool(breaks[i]);left+=int(is_left);right+=int(j==len(ep.t) or (j<len(ep.t) and breaks[j]))
  if not is_left:
   events+=1;prior=bool(i>0 and not breaks[i] and alarm[i-1]);prealarm+=int(prior)
   if len(hit):hits+=1;delay_sum+=int(hit[0]);delay_n+=1;prealarm0+=int(prior and int(hit[0])==0)
 longest=current=0
 for i in range(len(ep.t)):
  if static[i] and alarm[i] and (i==0 or (not breaks[i] and static[i-1] and alarm[i-1])):current+=1
  elif static[i] and alarm[i]:current=1
  else:current=0
  longest=max(longest,current)
 sn=int(static.sum());fp=int((static&alarm).sum());tp=int((gross&alarm).sum())
 return {"episode":ep.episode,"leakage_group":ep.group,"probe":ep.probe,"static_frames":sn,"static_fp":fp,"static_any_alarm":int(fp>0),"gross_frames":int(gross.sum()),"gross_tp":tp,"tn":sn-fp,"fp":fp,"fn":int(gross.sum())-tp,"tp":tp,"false_starts":int((starts&static).sum()),"alarm_starts":int(starts.sum()),"longest_static_alarm_segment":longest,"events":events,"event_hits":hits,"left_censored_events":left,"right_boundary_events":right,"gross_segments":events+left,"segments_covered":covered,"preexisting_alarm_events":prealarm,"preexisting_alarm_delay0_events":prealarm0,"delay_sum":delay_sum,"delay_n":delay_n,"trial_count":1,"incipient_frames":int(incipient.sum()),"observed_alarm":int(alarm.any())}
def main():
 p=argparse.ArgumentParser();p.add_argument("--r13-index",type=Path,required=True);p.add_argument("--r13-source",type=Path,required=True);p.add_argument("--r13-historical",type=Path,required=True);p.add_argument("--r13-new-metrics",type=Path,required=True);p.add_argument("--r13-confirm4",type=Path,required=True);p.add_argument("--prepared-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 spec=importlib.util.spec_from_file_location("r13",a.r13_source);r13=importlib.util.module_from_spec(spec);sys.modules["r13"]=r13;spec.loader.exec_module(r13)
 index=json.loads(a.r13_index.read_text());r13.validate_index(index);historical=readcsv(a.r13_historical);hist={(x["group"],x["fold"],int(x["seed"]),x["role"],x["point"],x["rule"]):x for x in historical}
 common={}; endpoint_audit=[]
 for f in range(1,5):
  for s in r13.SEEDS:
   d=torch.load(a.prepared_root/f"p{f}_s{s}/prepared.pt",map_location="cpu",weights_only=False)
   common[(f,s,"train")]={(e,int(t)) for role in ("fit","selection") for e,t in zip(d["roles"][role]["episode_id"],d["roles"][role]["t"])}
   for role in ("calibration","validation"):common[(f,s,role)]={(e,int(t)) for e,t in zip(d["roles"][role]["episode_id"],d["roles"][role]["t"])}
 out=[];full=[];state_checks=0;episode_cache={}
 for run in index["selected_runs"]:
  g,fold,s=run["group"],run["fold"],int(run["seed"]);f=int(fold[-1])
  for role in r13.ROLES:
   eps=r13.load_episodes(run,role);keys=common[(f,s,role)];masked=[];used=set();removed=0
   masks={}
   for ep in eps:
    mask=[(ep.episode,int(t)) in keys for t in ep.t];removed+=len(mask)-sum(mask);used|={(ep.episode,int(t)) for t,m in zip(ep.t,mask) if m}
    if any(mask):
     m=np.asarray(mask,bool);mep=r13.Episode(ep.episode,ep.group,ep.probe,ep.t[m],ep.stage[m],ep.score[m]);masked.append(mep);masks[ep.episode]=m
   assert used==keys
   episode_cache[(g,fold,s,role)]=(eps,masked,masks)
   endpoint_audit.append({"group":g,"fold":fold,"seed":s,"role":role,"full_endpoints":sum(len(x.t) for x in eps),"common_endpoints":len(keys),"tail_removed":removed,"episodes":len(eps),"state_first_parity_checks":2*len(masked)})
   for point in ("fixed_0.5","maxBA","FPR0.01","FPR0.05","FPR0.10"):
    for rule,k in (("raw",1),("confirm2",2)):
     h=hist[(g,fold,s,role,point,rule)];threshold=float(h["threshold"]);fs=[];cs=[]
     for ep,mep in zip(eps,masked):
      alarm,starts=r13.state_alarm(ep,threshold,k);m=masks[ep.episode];masked_alarm,masked_starts=alarm[m],starts[m];recomputed_alarm,recomputed_starts=r13.state_alarm(mep,threshold,k);assert np.array_equal(masked_alarm,recomputed_alarm) and np.array_equal(masked_starts,recomputed_starts);state_checks+=1
      fs.append(r13.episode_stats(ep,threshold,k));cs.append(stats_from_state(mep,masked_alarm,masked_starts))
     fa=r13.aggregate(fs);ca=r13.aggregate(cs)
     for hkey,fkey in (("static_fpr","frame_static_FPR"),("gross_recall","gross_recall"),("balanced_accuracy","balanced_accuracy"),("macro_f1","macro_f1"),("event_recall","event_recall"),("false_starts_per_trial","false_starts_per_trial")):
      hv=float(h[hkey]);fv=float(fa[fkey]);assert (math.isnan(hv) and math.isnan(fv)) or math.isclose(hv,fv,rel_tol=1e-12,abs_tol=1e-12)
     meta={"group":g,"fold":fold,"seed":s,"role":role,"point":point,"rule":rule,"threshold":threshold}
     fields=("frame_static_FPR","gross_recall","balanced_accuracy","macro_f1","event_recall","false_starts_per_trial","mean_detected_event_delay","static_frames","static_alarming_frames","gross_frames","gross_tp","trial_count","events","event_hits","left_censored_events","right_boundary_events","gross_segments","segments_covered","delay_sum","delay_n")
     full.append({**meta,**{q:fa[q] for q in fields}})
     out.append({**meta,**{q:ca[q] for q in fields}})
 writecsv(a.output/"full_endpoints.csv",full);writecsv(a.output/"common_endpoints.csv",out);writecsv(a.output/"endpoint_audit.csv",endpoint_audit)
 metric_fields=("static_frames","static_fp","gross_frames","gross_tp","tn","fp","fn","tp","false_starts","alarm_starts","events","event_hits","left_censored_events","right_boundary_events","gross_segments","segments_covered","preexisting_alarm_events","preexisting_alarm_delay0_events","delay_sum","delay_n","trial_count","incipient_frames","observed_alarm","static_support_trials","static_alarming_frames","static_support_trial_denominator","static_alarming_frames_per_static_support_trial","frame_static_FPR","trial_macro_static_FPR","trial_any_static_alarm_rate","gross_recall","balanced_accuracy","macro_f1","false_starts_per_trial","event_recall","mean_detected_event_delay","preexisting_alarm_event_rate","preexisting_alarm_delay0_rate","segment_coverage","longest_static_alarm_segment")
 def process_saved(rows,kind):
  nonlocal state_checks
  saved_full=[];saved_common=[]
  for h in rows:
   g,fold,s,role=h["group"],h["fold"],int(h["seed"]),h["role"];eps,masked,masks=episode_cache[(g,fold,s,role)];threshold=float(h["threshold"]);k=int(h["k"]);fs=[];cs=[]
   for ep,mep in zip(eps,masked):
    alarm,starts=r13.state_alarm(ep,threshold,k);m=masks[ep.episode];ma,ms=alarm[m],starts[m];ra,rs=r13.state_alarm(mep,threshold,k);assert np.array_equal(ma,ra) and np.array_equal(ms,rs);state_checks+=1;fs.append(r13.episode_stats(ep,threshold,k));cs.append(stats_from_state(mep,ma,ms))
   fa,ca=r13.aggregate(fs),r13.aggregate(cs)
   for key in metric_fields:
    if key in h and h[key] not in ("",None):
     hv=float(h[key]);fv=float(fa[key]);assert (math.isnan(hv) and math.isnan(fv)) or math.isclose(hv,fv,rel_tol=1e-12,abs_tol=1e-12),(kind,g,fold,s,role,h.get("policy"),key,hv,fv)
   meta={q:h[q] for q in ("group","fold","seed","role","policy","alpha","k","threshold") if q in h};meta["family"]=h.get("family",kind);saved_full.append({**meta,**{q:fa[q] for q in metric_fields}});saved_common.append({**meta,**{q:ca[q] for q in metric_fields}})
  return saved_full,saved_common
 np_full,np_common=process_saved(readcsv(a.r13_new_metrics),"new_policy");c4_full,c4_common=process_saved(readcsv(a.r13_confirm4),"confirm4_reference")
 writecsv(a.output/"r13_new_policy_full_endpoints.csv",np_full);writecsv(a.output/"r13_new_policy_common_endpoints.csv",np_common);writecsv(a.output/"r13_confirm4_full_endpoints.csv",c4_full);writecsv(a.output/"r13_confirm4_common_endpoints.csv",c4_common)
 result={"schema":"round14_current_common_v3","status":"pass","models":48,"historical_full_metric_rows":len(full),"historical_common_metric_rows":len(out),"r13_new_policy_rows_each_scope":len(np_full),"r13_confirm4_rows_each_scope":len(c4_full),"state_first_parity_checks":state_checks,"state_first_scope":"all 48 models x 3 roles; historical 5 points raw/confirm2 plus all saved R13 trial-macro/trial-any alpha/k policies and confirm4 references; every episode","historical_reproduction":"exact for six core metrics on all 1440 historical rows","r13_policy_reproduction":"all saved aggregate numeric fields reproduced for 2592 new-policy and 432 confirm4 rows","rule":"reuse saved thresholds; generate alarm and starts on full original timeline, then mask both to common endpoint prefix; no refit, search, or state restart","tail_censoring":"last 10 endpoints removed per episode; all static/gross/event denominators and censoring fields reported","hashes":{p.name:ft.sha(p) for p in a.output.glob("*.csv")}};ft.atomic_json(result,a.output/"SUMMARY.json");print(json.dumps(result))
if __name__=="__main__":main()
