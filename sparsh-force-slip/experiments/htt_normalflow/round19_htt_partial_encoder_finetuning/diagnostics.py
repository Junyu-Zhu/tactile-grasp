#!/usr/bin/env python3
"""Materialize preregistered Round-19 failure cases with aligned scores and source images."""
import argparse,csv,json
from pathlib import Path
from collections import defaultdict
import numpy as np,torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import train as tr
import sys
sys.path.insert(0,str(tr.REPO/"scripts"));import adapters
def rows(p):
 with open(p,newline="") as f:return list(csv.DictReader(f))
def write(path,rr):
 path.parent.mkdir(parents=True,exist_ok=True)
 with open(path,"w",newline="") as f:w=csv.DictWriter(f,rr[0]);w.writeheader();w.writerows(rr)
def main():
 p=argparse.ArgumentParser();p.add_argument("--evaluation",type=Path,required=True);p.add_argument("--prefix-index",type=Path,required=True);p.add_argument("--prepared-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 selected=json.loads((a.evaluation/"cases/SELECTED.json").read_text())["selected"];scores=rows(a.evaluation/"scores/RAW_SCORES.csv");ths=rows(a.evaluation/"metrics/CALIBRATION_THRESHOLDS.csv");index={x["episode_id"]:x for x in json.loads(a.prefix_index.read_text())["entries"]};allrows=[];artifacts=[]
 prepared_cache={}
 for ci,case in enumerate(selected,1):
  fold,seed,ep=int(case["fold"]),int(case["seed"]),case["episode"];rr=[]
  key=(fold,seed)
  if key not in prepared_cache:
   role=torch.load(a.prepared_root/f"p{fold}_s{seed}"/"prepared.pt",map_location="cpu",weights_only=False)["roles"]["validation"]
   prepared_cache[key]={(eid,int(t)):(role["x"][i,-1,192:195].tolist(),role["y_current"][i].tolist()) for i,(eid,t) in enumerate(zip(role["episode_id"],role["t"]))}
  for group in ("V0","M0","V2","M2"):
   th=float(next(x["threshold"] for x in ths if x["group"]==group and int(x["fold"])==fold and int(x["seed"])==seed and x["policy"]=="FPR5"));g=sorted([x for x in scores if x["group"]==group and int(x["fold"])==fold and int(x["seed"])==seed and x["role"]=="validation" and x["episode"]==ep],key=lambda x:int(x["t"]));
   active=False;count=0
   for j,x in enumerate(g):
    t=int(x["t"]);gap=j==0 or t!=int(g[j-1]["t"])+1
    if gap:active=False;count=0
    raw=float(x["score"])>=th;start=False
    if active:
     if not raw:active=False;count=0
    else:
     count=count+1 if raw else 0
     if count>=2:active=True;start=True
    predicted,observed=prepared_cache[key][(ep,t)]
    rr.append({"case_index":ci,"kind":case["kind"],"group":group,"fold":fold,"seed":seed,"episode":ep,"t":t,"stage":int(x["stage"]),"score":float(x["score"]),"calibration_FPR5_threshold":th,"raw_alarm":int(raw),"k2_alarm":int(active),"k2_start":int(start),"native_gap_or_left_reset":int(gap),"observed_segment_right_boundary":int(j==len(g)-1 or int(g[j+1]["t"])!=t+1),"observed_gross_left_censored":int(int(x["stage"])==2 and gap),"observed_gross_right_boundary":int(int(x["stage"])==2 and (j==len(g)-1 or int(g[j+1]["t"])!=t+1)),"predicted_force_x":predicted[0],"predicted_force_y":predicted[1],"predicted_force_z":predicted[2],"observed_force_x":observed[0],"observed_force_y":observed[1],"observed_force_z":observed[2]})
  if not rr:raise ValueError(f"empty case {case}")
  allrows.extend(rr);write(a.output/f"case_{ci}_aligned.csv",rr)
  fig,ax=plt.subplots(figsize=(9,4.8))
  for group,color in zip(("V0","M0","V2","M2"),("#777777","#d95f02","#1b9e77","#7570b3")):
   q=[x for x in rr if x["group"]==group];ax.plot([x["t"] for x in q],[x["score"] for x in q],label=group,color=color);ax.axhline(q[0]["calibration_FPR5_threshold"],color=color,ls="--",alpha=.35)
  stage={int(x["t"]):int(x["stage"]) for x in rr};ax.fill_between(sorted(stage),0,1,where=np.asarray([stage[t]==2 for t in sorted(stage)]),color="red",alpha=.08,transform=ax.get_xaxis_transform());ax.set(xlabel="native frame t",ylabel="slip score",title=f"{case['kind']} | p{fold} s{seed} | {ep}",ylim=(0,1));ax.legend(ncol=4);fig.tight_layout();plot=a.output/f"case_{ci}_timeline.svg";fig.savefig(plot);plt.close(fig)
  source=None
  if ep in index:
   source=index[ep]["source_path"];obj=adapters.load_htt(source);t=int(sorted({x["t"] for x in rr})[len({x["t"] for x in rr})//2]);im=adapters.window(obj,t)["inputs"]["image"].detach().cpu();panels=[]
   if im.ndim==3 and im.shape[0]>=6:panels=[im[:3],im[3:6]]
   elif im.ndim==3:panels=[im[:3]]
   fig,axs=plt.subplots(1,len(panels),figsize=(4*len(panels),4));axs=np.atleast_1d(axs)
   for j,(ax,panel) in enumerate(zip(axs,panels)):
    z=panel.permute(1,2,0).numpy();z=(z-z.min())/(z.max()-z.min()+1e-12);ax.imshow(z);ax.axis("off");ax.set_title(f"input panel {j+1}, t={t}")
   fig.tight_layout();image=a.output/f"case_{ci}_source.png";fig.savefig(image,dpi=160);plt.close(fig)
   fig,axs=plt.subplots(1,2,figsize=(8,4))
   axs[0].imshow(obj.images[t]);axs[0].set_title(f"raw tactile RGB t={t}")
   axs[1].imshow(obj.reference.astype(np.uint8));axs[1].set_title("raw reference frame")
   for ax in axs:ax.axis("off")
   fig.tight_layout();fig.savefig(a.output/f"case_{ci}_raw_source_reference.png",dpi=160);plt.close(fig)
  artifacts.append({"case_index":ci,"kind":case["kind"],"fold":fold,"seed":seed,"episode":ep,"source_path":source,"aligned_csv":f"case_{ci}_aligned.csv","timeline":f"case_{ci}_timeline.svg","model_input_image":f"case_{ci}_source.png" if source else None,"raw_source_reference_image":f"case_{ci}_raw_source_reference.png" if source else None})
 write(a.output/"ALL_CASE_ALIGNED.csv",allrows);summary={"schema":"round19_failure_diagnostics_v1","status":"complete","cases":len(selected),"aligned_rows":len(allrows),"artifacts":artifacts,"test_consumed":False,"hashes":{str(x.relative_to(a.output)):tr.sha(x) for x in a.output.rglob("*") if x.is_file()}};tr.atomic_json(a.output/"SUMMARY.json",summary);print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
