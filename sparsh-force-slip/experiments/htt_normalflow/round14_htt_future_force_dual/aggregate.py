#!/usr/bin/env python3
"""Aggregate all runs and grouped trial bootstrap comparisons."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np,torch
import future_train as ft
def readcsv(p):return list(csv.DictReader(open(p)))
def writecsv(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def loadpred(root,g,f,s,role="validation"):return torch.load(root/f"{g}_p{f}_s{s}"/f"predictions_{role}.pt",map_location="cpu",weights_only=False)
def main():
 p=argparse.ArgumentParser();p.add_argument("--evaluation",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 allm=[];diags=[];temporal=[];trials=[]
 for g in ft.GROUPS:
  for f in range(1,5):
   for s in (20260914,20260915,20260916):
    run=f"{g}_p{f}_s{s}"
    for r in readcsv(a.evaluation/run/"metrics.csv"):allm.append({"group":g,"fold":f,"seed":s,**r})
    d=loadpred(a.evaluation,g,f,s);gt=d["y"].numpy();pc=d["predictions"]["predicted_current_persistence"].numpy();groups=np.array(d["leakage_group"]);ts=d["t"].numpy()
    for method,predt in d["predictions"].items():
     pred=predt.numpy();err=np.abs(pred-gt).mean((1,2)); std_ratio=float(pred.std()/max(gt.std(),1e-12)); copy=float(np.mean(np.abs(pred-pc)<.05)); corr=float(np.corrcoef(err,ts)[0,1]) if np.std(err)>0 else 0.
     diags.append({"group":g,"fold":f,"seed":s,"method":method,"prediction_std":float(pred.std()),"gt_std":float(gt.std()),"std_ratio":std_ratio,"fraction_within_0p05N_of_pred_current":copy,"abs_error_vs_t_corr":corr,"finite":bool(np.isfinite(pred).all())})
     for ep in sorted(set(d["episode_id"])):
      em=np.array(d["episode_id"])==ep
      for hi,h in enumerate(ft.HORIZONS):
       for ai,axis in enumerate(("fx","fy","fz")):
        ps=float(pred[em,hi,ai].std());gs=float(gt[em,hi,ai].std());cp=float(np.mean(np.abs(pred[em,hi,ai]-pc[em,hi,ai])<.05));temporal.append({"group":g,"fold":f,"seed":s,"method":method,"episode_id":ep,"horizon":h,"axis":axis,"n":int(em.sum()),"prediction_temporal_std_n":ps,"gt_temporal_std_n":gs,"temporal_std_ratio":ps/gs if gs>=1e-6 else "","gt_near_constant_lt_1e_6":gs<1e-6,"collapse_flag_pred_le_0p05_gt_ge_0p25":ps<=.05 and gs>=.25,"copy_fraction_within_0p05n":cp})
     for lg in sorted(set(groups)):
      m=groups==lg; trials.append({"group":g,"fold":f,"seed":s,"method":method,"leakage_group":lg,"n":int(m.sum()),"future_mae":float(np.abs(pred[m]-gt[m]).mean())})
 writecsv(a.output/"all_metrics.csv",allm);writecsv(a.output/"diagnostics.csv",diags);writecsv(a.output/"temporal_diagnostics.csv",temporal);writecsv(a.output/"trial_metrics.csv",trials)
 collapse=[]
 for g in ft.GROUPS:
  for method in ("neural","predicted_current_persistence","fit_ridge_linear","gt_current_persistence_ideal_only"):
   z=[r for r in temporal if r["group"]==g and r["method"]==method];valid=[r for r in z if not r["gt_near_constant_lt_1e_6"]];collapse.append({"group":g,"method":method,"episode_axis_horizon_cells":len(z),"gt_near_constant_cells":sum(r["gt_near_constant_lt_1e_6"] for r in z),"collapse_flag_cells":sum(r["collapse_flag_pred_le_0p05_gt_ge_0p25"] for r in z),"collapse_flag_fraction_of_gt_variable":sum(r["collapse_flag_pred_le_0p05_gt_ge_0p25"] for r in valid)/len(valid) if valid else "","median_temporal_std_ratio_gt_nonconstant":float(np.median([r["temporal_std_ratio"] for r in valid])) if valid else "","mean_copy_fraction":float(np.mean([r["copy_fraction_within_0p05n"] for r in z]))})
 writecsv(a.output/"temporal_collapse_summary.csv",collapse)
 summary=[]
 for g in ft.GROUPS:
  for method in ("neural","predicted_current_persistence","fit_ridge_linear","gt_current_persistence_ideal_only"):
   z=[r for r in allm if r["group"]==g and r["method"]==method and r["role"]=="validation" and r["stratum"]=="all"]
   for metric in ("future_mae","future_rmse","change_mae","change_rmse"):
    vals=np.array([float(r[metric]) for r in z]);summary.append({"group":g,"method":method,"metric":metric,"mean":float(vals.mean()),"std_across_run_axis_horizon_cells":float(vals.std()),"cells":len(vals)})
 # Complete leakage-group bootstrap, shared draws across all three seeds within each fold.
 ci=[];pairs=(("F_concat","V"),("F_dual","F_concat"));B=1000
 for ga,gb in pairs:
  for f in range(1,5):
   pa={s:loadpred(a.evaluation,ga,f,s) for s in (20260914,20260915,20260916)};pb={s:loadpred(a.evaluation,gb,f,s) for s in (20260914,20260915,20260916)}
   groups=sorted(set(pa[20260914]["leakage_group"]));rng=np.random.default_rng(140000+f);vals=[]
   for _ in range(B):
    draw=rng.choice(groups,len(groups),replace=True);sd=[]
    for s in (20260914,20260915,20260916):
     A=pa[s];Bv=pb[s]; assert A["episode_id"]==Bv["episode_id"] and torch.equal(A["t"],Bv["t"])
     ga_arr=np.array(A["leakage_group"]);per=[]
     for lg in draw:
      m=ga_arr==lg;per.append(float(np.abs(A["predictions"]["neural"][m].numpy()-A["y"][m].numpy()).mean()-np.abs(Bv["predictions"]["neural"][m].numpy()-Bv["y"][m].numpy()).mean()))
     sd.append(np.mean(per))
    vals.append(np.mean(sd))
   point=np.mean(vals);lo,hi=np.quantile(vals,[.025,.975]);ci.append({"comparison":f"{ga}-{gb}","fold":f,"metric":"validation_future_mae_N","point":float(point),"ci_low":float(lo),"ci_high":float(hi),"bootstrap_draws":1000,"unit":"complete_leakage_group_shared_across_seeds"})
 writecsv(a.output/"summary.csv",summary);writecsv(a.output/"paired_group_ci.csv",ci)
 result={"schema":"round14_future_aggregate_v2","status":"complete","runs":36,"metric_rows":len(allm),"trial_rows":len(trials),"diagnostic_rows":len(diags),"temporal_diagnostic_rows":len(temporal),"all_finite":all(r["finite"] for r in diags),"collapse_definition":"per validation episode x axis x horizon: prediction temporal std <=0.05 N while GT temporal std >=0.25 N; post-training diagnostic only","hashes":{p.name:ft.sha(p) for p in a.output.iterdir() if p.is_file() and p.name!="SUMMARY.json"}};ft.atomic_json(result,a.output/"SUMMARY.json");print(json.dumps(result))
if __name__=="__main__":main()
