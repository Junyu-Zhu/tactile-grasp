#!/usr/bin/env python3
"""Render deterministic global E3 failures with source tactile frames and score traces."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
def read(p):
 with open(p,newline="") as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--inventory",type=Path,required=True);p.add_argument("--evaluation",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);cands=read(a.evaluation/"cases/CANDIDATES.csv");scores=read(a.evaluation/"scores/RAW_SCORES.csv");ths={(x["group"],x["fold"],x["seed"]):float(x["threshold"]) for x in read(a.evaluation/"metrics/CALIBRATION_THRESHOLDS.csv") if x["policy"]=="FPR5"};inv=json.loads(a.inventory.read_text())["runs"];selected=[]
 for comp in ("B-A","D-C","C-A","D-B"):
  pool=[x for x in cands if x["comparison"]==comp]
  for metric in ("static_alarm_difference","gross_miss_difference"):
   selected.append({"rule":f"global_largest_{metric}",**sorted(pool,key=lambda x:(-float(x[metric]),int(x["fold"]),int(x["seed"]),x["episode"]))[0]})
 figures=[]
 for i,item in enumerate(selected,1):
  fold=item["fold"];seed=item["seed"];ep=item["episode"];left,right=item["comparison"].split("-");z={}
  for g in (left,right):
   rr=[x for x in scores if x["group"]==g and x["fold"]==fold and x["seed"]==seed and x["role"]=="validation" and x["episode"]==ep];rr=sorted(rr,key=lambda x:int(x["t"]));z[g]=(np.asarray([int(x["t"]) for x in rr]),np.asarray([int(x["stage"]) for x in rr]),np.asarray([float(x["score"]) for x in rr]))
  run=next(x for x in inv if x["fold"]==int(fold) and x["seed"]==int(seed));idx=json.loads(Path(run["prefix_index"]).read_text());source=Path(next(x["source_path"] for x in idx["entries"] if x["episode_id"]==ep));raw=np.load(source,allow_pickle=False);imgs=raw["tactile_img"];t,stage,_=z[left];bad=((stage==0)&((z[left][2]>=ths[left,fold,seed])|(z[right][2]>=ths[right,fold,seed])))|((stage==2)&((z[left][2]<ths[left,fold,seed])|(z[right][2]<ths[right,fold,seed])));pick=t[bad] if bad.any() else t;ids=np.unique(np.clip(np.quantile(pick,[0,.5,1]).astype(int),0,len(imgs)-1))
  while len(ids)<3:ids=np.unique(np.r_[ids,np.linspace(0,len(imgs)-1,3,dtype=int)])[:3]
  fig=plt.figure(figsize=(11,7));gs=fig.add_gridspec(2,3,height_ratios=[1.2,1]);ax=fig.add_subplot(gs[0,:])
  for g,ls in ((left,"-"),(right,"--")):ax.plot(z[g][0],z[g][2],ls,label=f"{g} score");ax.axhline(ths[g,fold,seed],ls=ls,label=f"{g} calibration-FPR5")
  for label,color,name in ((0,"#d9f0d3","static"),(1,"#fee08b","incipient"),(2,"#f4a582","gross")):
   q=np.where(stage==label)[0]
   if len(q):ax.scatter(t[q],np.full(len(q),-.03),s=8,c=color,label=name)
  ax.set(xlabel="native frame t",ylabel="slip probability",ylim=(-.08,1.03),title=f"{item['rule']} | {item['comparison']} | {ep}");ax.legend(ncol=4,fontsize=7)
  for j,k in enumerate(ids[:3]):q=fig.add_subplot(gs[1,j]);q.imshow(imgs[k]);q.set_title(f"source t={k}");q.axis("off")
  fig.tight_layout();name=f"e3_failure_{i}.png";fig.savefig(a.output/name,dpi=180);plt.close(fig);figures.append({"figure":name,"rule":item["rule"],"comparison":item["comparison"],"fold":fold,"seed":seed,"episode":ep,"source_path":str(source),"source_sha256":sha(source)})
 out={"schema":"round23_e3_failure_figures_v1","status":"complete","figures":figures,"count":len(figures),"selection":"global largest per-trial static alarm and gross miss increments for each preregistered non-interaction comparison; lexical ties","test_consumed":False};(a.output/"SUMMARY.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out))
if __name__=="__main__":main()
