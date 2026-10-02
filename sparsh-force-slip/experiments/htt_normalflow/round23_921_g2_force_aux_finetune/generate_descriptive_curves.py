#!/usr/bin/env python3
"""Validation-only ROC/PR and matched-FPR/recall diagnostics; never deployable."""
from __future__ import annotations
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
CONTRASTS=("B-A","D-C","C-A","D-B")
def read(p):
 with open(p,newline="") as f:return list(csv.DictReader(f))
def write(p,r):
 with open(p,"w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)
def curve(stage,score):
 mask=stage!=1;y=(stage[mask]==2).astype(int);s=score[mask];order=np.argsort(-s,kind="stable");y=y[order];s=s[order];ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1];tp=np.cumsum(y)[ends];fp=np.cumsum(1-y)[ends];P=y.sum();N=len(y)-P;return np.r_[0,fp/N],np.r_[0,tp/P],np.r_[1,tp/np.maximum(tp+fp,1)],np.r_[0,tp/P],np.r_[np.inf,s[ends]]
def main():
 p=argparse.ArgumentParser();p.add_argument("--evaluation",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);raw=[x for x in read(a.evaluation/"scores/RAW_SCORES.csv") if x["role"]=="validation"];work=[x for x in read(a.evaluation/"metrics/WORKPOINT_METRICS.csv") if x["role"]=="validation" and x["policy"]=="FPR5|raw"];by=defaultdict(list)
 for x in raw:by[x["group"],int(x["fold"]),int(x["seed"])].append(x)
 points=[];curves={}
 for key,rr in by.items():
  stage=np.asarray([int(x["stage"]) for x in rr]);score=np.asarray([float(x["score"]) for x in rr]);fpr,tpr,precision,recall,threshold=curve(stage,score);curves[key]=(fpr,tpr,precision,recall,threshold)
  for i in range(len(fpr)):points.append({"group":key[0],"fold":key[1],"seed":key[2],"point":i,"threshold":threshold[i],"FPR":fpr[i],"recall":tpr[i],"precision":precision[i]})
 lookup={(x["group"],int(x["fold"]),int(x["seed"])):x for x in work};matched=[]
 for comp in CONTRASTS:
  left,right=comp.split("-")
  for fold in range(1,5):
   for seed in (20260914,20260915,20260916):
    target_fpr=float(lookup[right,fold,seed]["frame_static_FPR"]);target_recall=float(lookup[right,fold,seed]["gross_recall"]);fpr,tpr,_,_,th=curves[left,fold,seed];ok=np.flatnonzero(fpr<=target_fpr+1e-15);i=max(ok,key=lambda q:(tpr[q],-fpr[q],th[q]));ok2=np.flatnonzero(tpr>=target_recall-1e-15);j=min(ok2,key=lambda q:(fpr[q],-tpr[q],-th[q]));matched.append({"comparison":comp,"fold":fold,"seed":seed,"reference_group":right,"candidate_group":left,"reference_FPR5_validation_FPR":target_fpr,"candidate_recall_at_no_greater_validation_FPR":tpr[i],"reference_FPR5_validation_recall":target_recall,"same_FPR_recall_difference":tpr[i]-target_recall,"candidate_FPR_at_no_lower_validation_recall":fpr[j],"same_recall_FPR_difference":fpr[j]-target_fpr,"candidate_same_FPR_threshold_from_validation":th[i],"candidate_same_recall_threshold_from_validation":th[j],"deployable":False})
 write(a.output/"ROC_PR_POINTS.csv",points);write(a.output/"MATCHED_VALIDATION_ONLY.csv",matched);grid=np.linspace(0,1,201);fig,ax=plt.subplots(figsize=(6,5));figp,axp=plt.subplots(figsize=(6,5));figz,axz=plt.subplots(figsize=(6,5))
 for g in "ABCD":
  roc=[];pr=[]
  for fold in range(1,5):
   for seed in (20260914,20260915,20260916):
    fpr,tpr,precision,recall,_=curves[g,fold,seed];roc.append(np.interp(grid,fpr,tpr));ix=np.argsort(recall);pr.append(np.interp(grid,recall[ix],precision[ix]))
  mr=np.mean(roc,0);mp=np.mean(pr,0);ax.plot(grid,mr,label=g);axz.plot(grid[grid<=.2],mr[grid<=.2],label=g);axp.plot(grid,mp,label=g)
 for q,title,xlabel,ylabel in ((ax,"Validation ROC","FPR","Recall"),(axz,"Validation low-FPR ROC (descriptive)","FPR","Recall"),(axp,"Validation PR","Recall","Precision")):q.set(title=title,xlabel=xlabel,ylabel=ylabel);q.grid(alpha=.2);q.legend()
 for fig,name in ((fig,"ROC.png"),(figz,"LOW_FPR_ROC.png"),(figp,"PR.png")):fig.tight_layout();fig.savefig(a.output/name,dpi=180);plt.close(fig)
 out={"schema":"round23_e3_descriptive_curves_v1","status":"complete","runs":48,"roc_pr_points":len(points),"matched_rows":len(matched),"scope":"validation-only descriptive; matched thresholds are derived from validation and are not deployable or used for selection","test_consumed":False};(a.output/"SUMMARY.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out))
if __name__=="__main__":main()
