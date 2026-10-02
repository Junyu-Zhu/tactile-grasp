#!/usr/bin/env python3
"""Post-score deterministic derivation of preregistered frame ROC/PR envelopes."""
from __future__ import annotations
import argparse,csv,gzip,hashlib,importlib.util,json,math,os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
GROUPS=("A_visual","B_force","C_force_delta","D_visual_delta");SEEDS=(20260914,20260915,20260916);FPRS=(.01,.05,.10);RECALLS=(.8,.9,.95)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(path):
 s=importlib.util.spec_from_file_location("bound_r7_eval_curves",path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def atomic_json(path,v):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f".tmp.{os.getpid()}");tmp.write_text(json.dumps(v,indent=2,sort_keys=True)+"\n");os.replace(tmp,path)
def atomic_gzip_csv(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f".tmp.{os.getpid()}")
 with gzip.open(tmp,"wt",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 os.replace(tmp,path)
def curve(rows,key,mask):
 cohort=[r for r in rows if r[mask]];y=np.asarray([r["target"] for r in cohort],np.int8);score=np.asarray([r[key] for r in cohort],float)
 if set(y.tolist())!={0,1}:raise ValueError("single-class curve")
 order=np.argsort(-score,kind="mergesort");ys,ss=y[order],score[order];ends=np.r_[np.flatnonzero(ss[:-1]!=ss[1:]),len(ss)-1];tp=np.cumsum(ys)[ends];fp=ends+1-tp;pos=int(y.sum());neg=len(y)-pos
 out=[{"threshold":math.nextafter(1.,math.inf),"recall":0.,"fpr":0.,"precision":1.}]
 out += [{"threshold":float(ss[e]),"recall":float(t/pos),"fpr":float(f/neg),"precision":float(t/(t+f))} for e,t,f in zip(ends,tp,fp)]
 return out,y,score
def pauc(points,limit):
 x=np.asarray([r["fpr"] for r in points]);y=np.asarray([r["recall"] for r in points]);keep=x<limit;xx=x[keep].tolist();yy=y[keep].tolist();
 if not xx or xx[-1]<limit:
  j=int(np.searchsorted(x,limit));ylo=y[j-1] if j else y[0];xlo=x[j-1] if j else x[0];yhi=y[j] if j<len(x) else y[-1];xhi=x[j] if j<len(x) else x[-1];yi=ylo if xhi==xlo else ylo+(limit-xlo)*(yhi-ylo)/(xhi-xlo);xx.append(limit);yy.append(float(yi))
 return float(np.trapz(yy,xx)/limit)
def main():
 p=argparse.ArgumentParser();p.add_argument("--evaluator",type=Path,required=True);p.add_argument("--formal-root",type=Path,required=True);p.add_argument("--analysis",type=Path,required=True);p.add_argument("--prepared-sha",required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();ev=load(a.evaluator);sels=list(csv.DictReader(open(a.analysis/"rule_selection.csv")));sources={str(a.evaluator.resolve()):sha(a.evaluator),str((a.analysis/"rule_selection.csv").resolve()):sha(a.analysis/"rule_selection.csv")};summaries=[];artifacts=[];plot_data={}
 for group in GROUPS:
  shards=[]
  for seed in SEEDS:
   _,roles,src=ev.verify_run(a.formal_root/f"future_{group}_{seed}",group,seed,a.prepared_sha,[1,3,5]);sources.update(src)
   for rows in roles.values():
    for h in (1,3,5):ev.add_rules(rows,f"p_H{h}",f"H{h}_")
   outer=roles["outer"]
   for h in (1,3,5):
    selected=next(r["rule"] for r in sels if r["group"]==group and r["seed"]==str(seed) and r["horizon"]==str(h) and r["population"]=="primary" and r["selected"]=="True")
    for pop in ("primary","per_h_max"):
     rows=ev.role_rows(outer,h,pop)
     for method,key in (("raw",f"H{h}_raw"),("selected_rule",f"H{h}_{selected}")):
      points,y,score=curve(rows,key,"metric_mask");prefix={"group":group,"seed":seed,"horizon":h,"population":pop,"method":method,"selected_rule":selected}
      for q in points:shards.append({**prefix,**q})
      summary={**prefix,"n":len(y),"positive":int(y.sum()),"prevalence":float(y.mean()),"average_precision":ev.average_precision(y,score),"roc_auc":float(np.trapz([q["recall"] for q in points],[q["fpr"] for q in points]))}
      for f in FPRS:summary[f"normalized_pauc_fpr_{f:.2f}"]=pauc(points,f);valid=[q for q in points if q["fpr"]<=f+1e-15];summary[f"descriptive_recall_at_fpr_{f:.2f}"]=max(q["recall"] for q in valid)
      for target in RECALLS:
       valid=[q for q in points if q["recall"]>=target-1e-15];summary[f"descriptive_fpr_at_recall_{target:.2f}"]=min(q["fpr"] for q in valid) if valid else ""
      summaries.append(summary)
      if pop=="primary" and method=="selected_rule":plot_data[(group,seed,h)]=points
  path=a.output/f"curves_{group}.csv.gz";atomic_gzip_csv(path,shards);artifacts.append(path)
 a.output.mkdir(parents=True,exist_ok=True);ev.atomic_csv(a.output/"frame_curve_summary.csv",summaries);artifacts.append(a.output/"frame_curve_summary.csv")
 colors=dict(zip(GROUPS,("tab:blue","tab:orange","tab:green","tab:red")));styles=("-","--",":");fig,axes=plt.subplots(1,3,figsize=(15,4.5))
 for ax,h in zip(axes,(1,3,5)):
  for g in GROUPS:
   for i,s in enumerate(SEEDS):
    q=plot_data[(g,s,h)];ax.plot([r["fpr"] for r in q],[r["recall"] for r in q],color=colors[g],ls=styles[i],label=f"{g} s{str(s)[-2:]}" if h==1 else None)
  ax.set(title=f"H{h} primary outer",xlabel="frame FPR",ylabel="frame recall",xlim=(0,.1),ylim=(0,1));ax.grid(alpha=.25)
 axes[0].legend(fontsize=6,ncol=2);fig.tight_layout();path=a.output/"low_fpr_roc_all_runs.png";fig.savefig(path,dpi=180);plt.close(fig);artifacts.append(path)
 atomic_json(a.output/"CURVE_DERIVATION_PROVENANCE.json",{"schema":"round7_preregistered_frame_curve_derivation_v1","status":"complete","timing":"implemented after core scores were available; fixed FPR/recall grids were preregistered; no selection, threshold, metric JSON, or training artifact changed","fpr_grid":list(FPRS),"recall_grid":list(RECALLS),"sources":sources,"outputs":{str(x.resolve()):sha(x) for x in artifacts}})
 print(json.dumps({"status":"complete","summaries":len(summaries),"curve_shards":4}))
if __name__=="__main__":main()
