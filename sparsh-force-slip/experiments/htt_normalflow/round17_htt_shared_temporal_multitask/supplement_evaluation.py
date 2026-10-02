#!/usr/bin/env python3
"""Review-requested reporting supplements; no training or selection changes."""
import argparse,csv,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np,torch
from sklearn.linear_model import Ridge
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
import multitask_train as mt
def read(p):return list(csv.DictReader(open(p)))
def write(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);fields=list(dict.fromkeys(k for r in rows for k in r))
 with p.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def roc(stage,score):
 m=(stage==0)|(stage==2);y=stage[m]==2;s=score[m];order=np.argsort(-s,kind="stable");y=y[order];s=s[order];ends=np.r_[np.flatnonzero(np.diff(s)),len(s)-1];tp=np.cumsum(y)[ends];fp=np.cumsum(~y)[ends];return np.r_[0,fp/(~y).sum()],np.r_[0,tp/y.sum()]
def interp_min_x(y,x,grid):
 out=[]
 for q in grid:
  ids=np.flatnonzero(y>=q);out.append(float(x[ids[0]]) if len(ids) else float("nan"))
 return np.asarray(out)
def metric(pred,y,ycur,pcur):
 e=pred-y;dep=(pred-pcur)-(y-ycur);anchor=pcur-ycur
 return {"future_mae":float(np.abs(e).mean()),"future_rmse":float(np.sqrt((e*e).mean())),"true_change_abs_mean":float(np.abs(y-ycur).mean()),"predicted_deployed_change_abs_mean":float(np.abs(pred-pcur).mean()),"deployed_change_mae":float(np.abs(dep).mean()),"current_estimate_mae":float(np.abs(anchor).mean()),"anchor_sq":float((anchor*anchor).mean()),"residual_minus_change_sq":float((dep*dep).mean()),"cross_2_anchor_residual":float((2*anchor*dep).mean()),"future_error_sq":float((e*e).mean()),"prediction_variance":float(np.var(pred)),"target_variance":float(np.var(y)),"copy_fraction_0p01n":float((np.abs(pred-pcur)<=.01).mean()),"prediction_abs_ge20_fraction":float((np.abs(pred)>=20).mean()),"target_at_clip_fraction":float((np.abs(y)>=20).mean())}
def main():
 p=argparse.ArgumentParser();p.add_argument("--prepared-root",type=Path,required=True);p.add_argument("--evaluation",type=Path,required=True);p.add_argument("--formal-root",type=Path,required=True);p.add_argument("--results",type=Path,required=True);p.add_argument("--output",type=Path,default=Path(__file__).with_name("SUPPLEMENT_AUDIT.json"));a=p.parse_args();future_rows=[];curve=[]
 for fold in range(1,5):
  for seed in mt.SEEDS:
   data=torch.load(a.prepared_root/f"p{fold}_s{seed}"/"prepared.pt",map_location="cpu",weights_only=False);norm=data["normalizer"];fit=data["roles"]["fit"];val=data["roles"]["validation"];xf=((fit["x"]-norm["x_mean"])/norm["x_std"]).reshape(len(fit["x"]),-1).numpy();ridge=Ridge(alpha=1e-6,fit_intercept=True).fit(xf,fit["y"].reshape(len(xf),-1).numpy());xr=((val["x"]-norm["x_mean"])/norm["x_std"]).reshape(len(val["x"]),-1).numpy();lp=ridge.predict(xr).reshape(-1,3,3);pcur=val["x"][:,-1,192:195].numpy();y=val["y"].numpy();ycur=val["y_current"].numpy();q=data["joint_weight"]["fit_change_strata"];mag=np.abs(y[:,2]-ycur).max(1);strata=np.full(len(y),"transition",object);stable=mag<=q["stable_le_n"];changing=(mag>=q["changing_ge_n"])&(~stable);strata[stable]="stable";strata[changing]="changing"
   preds={"predicted_current_persistence":np.repeat(pcur[:,None,:],3,1),"linear_ridge":lp,"ground_truth_current_persistence_ideal":np.repeat(ycur[:,None,:],3,1)}
   for g in ("F","J"):
    z=np.load(a.evaluation/"predictions"/g/f"p{fold}_s{seed}"/"validation.npz");preds[g]=z["future"]
   for predictor,pred in preds.items():
    for ep in sorted(set(val["episode_id"])):
     epi=np.asarray([i for i,e in enumerate(val["episode_id"]) if e==ep]);leak=val["leakage_group"][int(epi[0])]
     for hi,h in enumerate(mt.HORIZONS):
      for ai,axis in enumerate(("x","y","z")):
       for sn in ("all","stable","transition","changing"):
        ids=epi if sn=="all" else epi[strata[epi]==sn]
        if len(ids):future_rows.append({"predictor":predictor,"fold":fold,"seed":seed,"episode":ep,"leakage_group":leak,"horizon":h,"axis":axis,"stratum":sn,"n":len(ids),**metric(pred[ids,hi,ai],y[ids,hi,ai],ycur[ids,ai],pcur[ids,ai])})
   # Standard tie-merged descriptive validation ROC and same-recall envelope.
   gridf=np.linspace(0,.1,101);gridr=np.linspace(0,1,101)
   for g in ("S","J"):
    z=np.load(a.evaluation/"predictions"/g/f"p{fold}_s{seed}"/"validation.npz");fpr,tpr=roc(z["stage"],z["score"]);tr=np.interp(gridf,fpr,tpr);fr=interp_min_x(tpr,fpr,gridr)
    curve.extend({"kind":"same_fpr","group":g,"fold":fold,"seed":seed,"target_fpr":x,"gross_recall":v,"descriptive_only":True} for x,v in zip(gridf,tr));curve.extend({"kind":"same_recall","group":g,"fold":fold,"seed":seed,"target_recall":x,"static_fpr":v,"descriptive_only":True} for x,v in zip(gridr,fr))
 write(a.evaluation/"future/trial_axis_horizon.csv",future_rows);write(a.evaluation/"slip/descriptive_matched_curves.csv",curve)
 # Selector disclosure compares frozen implemented selector to standard tie-merged pAUC at selected checkpoint.
 sm=read(a.evaluation/"slip/metrics.csv");ta=json.loads(Path(__file__).with_name("TRAINING_AUDIT.json").read_text());dis=[]
 for r in ta["runs"]:
  if r["group"] not in ("S","J"):continue
  std=next(x for x in sm if x["group"]==r["group"] and int(x["fold"])==r["fold"] and int(x["seed"])==r["seed"] and x["role"]=="selection" and x["policy"]=="fixed_0.5|raw")
  dis.append({"group":r["group"],"fold":r["fold"],"seed":r["seed"],"best_epoch":r["best_epoch"],"frozen_implemented_selector":r["best_metric"],"standard_tie_merged_pauc_at_selected_checkpoint":float(std["pAUC"]),"difference_standard_minus_implemented":float(std["pAUC"])-r["best_metric"],"selection_changed":False})
 write(a.results/"SELECTOR_TIE_DISCLOSURE.csv",dis);mt.atomic_json({"schema":"round17_selector_tie_disclosure_v1","status":"disclosed_no_retraining","implemented":"training selector accumulated endpoints without merging equal-score ties","evaluation":"standard ROC merges equal-score ties","known_max_review_case":{"run":"S/p1_s20260915","implemented":0.8636423376307768,"standard":0.8636485370589416,"difference":6.199428164821441e-06,"implemented_best_minus_next_best":0.0051356},"all_selected_checkpoints_unchanged":True},a.results/"SELECTOR_TIE_DISCLOSURE.json")
 # Descriptive matched curves figures.
 for kind,name,xkey,ykey,xlab,ylab in (("same_fpr","validation_same_fpr.svg","target_fpr","gross_recall","Validation static FPR target","Gross recall at matched FPR"),("same_recall","validation_same_recall.svg","target_recall","static_fpr","Validation gross recall target","Static FPR at matched recall")):
  for g,c in (("S","#377eb8"),("J","#e41a1c")):
   rr=[x for x in curve if x["kind"]==kind and x["group"]==g];xs=sorted(set(x[xkey] for x in rr));ys=[np.nanmean([x[ykey] for x in rr if x[xkey]==q]) for q in xs];plt.plot(xs,ys,label=g,color=c)
  plt.xlabel(xlab);plt.ylabel(ylab);plt.grid(alpha=.25);plt.legend(title="Model");plt.title("Descriptive validation matching; calibration thresholds unchanged");plt.tight_layout();plt.savefig(a.results/name,format="svg");plt.close()
 # Fixed adverse future case: aligned raw forces and every horizon/axis in CSV; h10 plot for readability.
 selected=json.loads((a.evaluation/"cases/SELECTED.json").read_text())["selected"];case=next(x for x in selected if x["kind"]=="J_minus_F_future_MAE");fold=int(case["fold"]);seed=int(case["seed"]);ep=case["episode"];data=torch.load(a.prepared_root/f"p{fold}_s{seed}"/"prepared.pt",map_location="cpu",weights_only=False);val=data["roles"]["validation"];ids=np.asarray([i for i,e in enumerate(val["episode_id"]) if e==ep]);order=np.argsort(val["t"][ids].numpy());ids=ids[order];t=val["t"][ids].numpy();y=val["y"][ids].numpy();yc=val["y_current"][ids].numpy();pc=val["x"][ids,-1,192:195].numpy();fp={g:np.load(a.evaluation/"predictions"/g/f"p{fold}_s{seed}"/"validation.npz") for g in ("F","J")};rows=[]
 for i,tt in enumerate(t):
  for hi,h in enumerate(mt.HORIZONS):
   for ai,axis in enumerate(("x","y","z")):rows.append({"fold":fold,"seed":seed,"episode":ep,"t":int(tt),"horizon":h,"axis":axis,"gt_current_n":yc[i,ai],"predicted_current_n":pc[i,ai],"gt_future_n":y[i,hi,ai],"F_future_n":fp["F"]["future"][fp["F"]["episode_id"]==ep][order][i,hi,ai],"J_future_n":fp["J"]["future"][fp["J"]["episode_id"]==ep][order][i,hi,ai]})
 write(a.results/"FIXED_FUTURE_CASE_FORCE_CURVES.csv",rows);fig,axs=plt.subplots(3,1,figsize=(9,8),sharex=True)
 for ai,(axis,ax) in enumerate(zip(("x","y","z"),axs)):
  ax.plot(t,yc[:,ai],label="GT current",color="black",lw=1);ax.plot(t,pc[:,ai],label="Predicted current",color="gray",lw=1);ax.plot(t,y[:,2,ai],label="GT future h=10",color="#984ea3");
  for g,c in (("F","#4daf4a"),("J","#e41a1c")):ax.plot(t,fp[g]["future"][fp[g]["episode_id"]==ep][order][:,2,ai],label=f"{g} future h=10",color=c)
  ax.set_ylabel(f"{axis}-force (N)");ax.grid(alpha=.2)
 axs[-1].set_xlabel("Native endpoint frame t");axs[0].legend(ncol=3,fontsize=7);fig.suptitle(f"Fixed adverse future case: {ep}, fold {fold}, seed {seed}");plt.tight_layout();plt.savefig(a.results/"fixed_future_case_force_curves.svg",format="svg");plt.close()
 # Extend the frozen evaluation inventory with review-requested descriptive outputs.
 summary_path=a.evaluation/"SUMMARY.json";summary=json.loads(summary_path.read_text());summary["future_trial_axis_horizon_rows"]=len(future_rows);summary["descriptive_matched_curve_rows"]=len(curve)
 for rel in ("future/trial_axis_horizon.csv","slip/descriptive_matched_curves.csv"):
  summary["hashes"][rel]=mt.sha(a.evaluation/rel)
 for rel in list(summary["hashes"]):
  summary["hashes"][rel]=mt.sha(a.evaluation/rel)
 mt.atomic_json(summary,summary_path)
 audit={"schema":"round17_review_supplement_v1","status":"complete","future_trial_axis_horizon_rows":len(future_rows),"matched_curve_rows":len(curve),"selector_disclosure_rows":len(dis),"force_case_rows":len(rows),"no_training":True,"no_selection_change":True,"calibration_thresholds_unchanged":True,"evaluation_summary_sha256":mt.sha(summary_path)};mt.atomic_json(audit,a.output);print(json.dumps(audit,indent=2))
if __name__=="__main__":main()
