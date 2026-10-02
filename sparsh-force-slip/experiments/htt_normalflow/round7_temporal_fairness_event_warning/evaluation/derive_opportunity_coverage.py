#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,os
from collections import Counter,defaultdict
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):s=importlib.util.spec_from_file_location("r7coverage",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def atomic_json(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f".tmp.{os.getpid()}");t.write_text(json.dumps(v,indent=2,sort_keys=True)+"\n");os.replace(t,p)
def main():
 p=argparse.ArgumentParser();p.add_argument("--evaluator",type=Path,required=True);p.add_argument("--formal-root",type=Path,required=True);p.add_argument("--prepared-sha",required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();ev=load(a.evaluator);summary,roles,sources=ev.verify_run(a.formal_root/"future_A_visual_20260914","A_visual",20260914,a.prepared_sha,[1,3,5]);rows=roles["outer"];by=defaultdict(list)
 for r in rows:by[r["episode_id"]].append(r)
 records=[]
 for episode,seq in sorted(by.items()):
  seq=sorted(seq,key=lambda r:r["t"]);onset=seq[0]["onset"]
  if onset is None:continue
  lookup={r["t"]:r for r in seq}
  for h in (1,3,5):
   expected=list(range(onset-h,onset))
   for pop,mask in (("primary","common_population"),("per_h_max",f"eligible_H{h}")):
    timeline=sum(t in lookup for t in expected);eligible=sum(t in lookup and lookup[t][mask] and lookup[t][f"target_H{h}"]==1 for t in expected)
    coverage="full_H" if eligible==h else ("partial_H" if eligible else "none")
    records.append({"episode_id":episode,"leakage_group":seq[0]["leakage_group"],"horizon":h,"population":pop,"onset_t":onset,"timeline_first_t":seq[0]["t"],"expected_opportunities":h,"timeline_present_opportunities":timeline,"eligible_positive_opportunities":eligible,"coverage":coverage,"missing_before_timeline":sum(t<seq[0]["t"] for t in expected),"present_but_population_excluded":timeline-eligible})
 a.output.mkdir(parents=True,exist_ok=True);ev.atomic_csv(a.output/"event_opportunity_coverage_trials.csv",records);aggregates=[]
 for h in (1,3,5):
  for pop in ("primary","per_h_max"):
   rr=[r for r in records if r["horizon"]==h and r["population"]==pop];c=Counter(r["coverage"] for r in rr)
   aggregates.append({"horizon":h,"population":pop,"observed_onset_trials":len(rr),"full_H":c["full_H"],"partial_H":c["partial_H"],"none":c["none"],"full_H_fraction":c["full_H"]/len(rr) if rr else ""})
 ev.atomic_csv(a.output/"event_opportunity_coverage_summary.csv",aggregates);atomic_json(a.output/"OPPORTUNITY_COVERAGE_PROVENANCE.json",{"schema":"round7_event_opportunity_coverage_v1","status":"complete","post_score_supplement":True,"core_metrics_changed":False,"sources":{**sources,str(a.evaluator.resolve()):sha(a.evaluator),str(Path(__file__).resolve()):sha(Path(__file__).resolve())},"outputs":{str((a.output/n).resolve()):sha(a.output/n) for n in ("event_opportunity_coverage_trials.csv","event_opportunity_coverage_summary.csv")}})
 print(json.dumps({"status":"complete","records":len(records),"summary":aggregates}))
if __name__=="__main__":main()
