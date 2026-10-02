#!/usr/bin/env python3
"""Evaluate one accepted future run and fixed fit-only baselines; never reads test."""
import argparse, csv, json
from pathlib import Path
import numpy as np, torch
import future_train as ft

AXES=("fx","fy","fz")
def design(x,group):
 z=x[...,:192] if group=="V" else x[...,:195]
 return z.reshape(len(z),-1).double()
def ridge_fit(x,y,lam=1e-6):
 x=torch.cat((x,torch.ones(len(x),1,dtype=x.dtype)),1); y=y.reshape(len(y),-1).double()
 reg=torch.eye(x.shape[1],dtype=x.dtype)*lam; reg[-1,-1]=0
 return torch.linalg.solve(x.T@x+reg,x.T@y)
def ridge_predict(x,w): return (torch.cat((x,torch.ones(len(x),1,dtype=x.dtype)),1)@w).float().reshape(-1,3,3)
def rows_for(role,r,preds):
 out=[]; change_mag=(r["y"][:,2]-r["y_current"]).abs().amax(1)
 strata=np.where(change_mag.numpy()<=.25,"stable",np.where(change_mag.numpy()>=1.,"changing","transitional"))
 for method,p in preds.items():
  for h_i,h in enumerate(ft.HORIZONS):
   for a,axis in enumerate(AXES):
    e=p[:,h_i,a]-r["y"][:,h_i,a]; de=(p[:,h_i,a]-r["x"][:,-1,192+a])-(r["y"][:,h_i,a]-r["y_current"][:,a])
    for stratum in ("all","stable","transitional","changing"):
     mask=np.ones(len(e),bool) if stratum=="all" else strata==stratum
     if not mask.any(): continue
     ee=e.numpy()[mask]; dd=de.numpy()[mask]
     out.append({"role":role,"method":method,"horizon":h,"axis":axis,"stratum":stratum,"n":int(mask.sum()),"future_mae":float(np.abs(ee).mean()),"future_rmse":float(np.sqrt((ee**2).mean())),"change_mae":float(np.abs(dd).mean()),"change_rmse":float(np.sqrt((dd**2).mean()))})
 return out
def main():
 p=argparse.ArgumentParser(); p.add_argument("--data",type=Path,required=True); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--group",choices=ft.GROUPS,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--device",default="cpu"); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
 d=torch.load(a.data,map_location="cpu",weights_only=False); assert set(d["roles"])=={"fit","selection","calibration","validation"}; ft.assert_role_isolation(d["roles"])
 ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False); assert ck["identity"]["group"]==a.group and ck["identity"]["round14_data_sha256"]==ft.sha(a.data)
 m=ft.init_model(a.group,ck["identity"]["seed"]); m.load_state_dict(ck["model"]); m.eval().to(a.device); n=ck["normalizer"]
 fit_x=(d["roles"]["fit"]["x"]-n["x_mean"])/n["x_std"]
 fit_y=(d["roles"]["fit"]["y"]-n["y_mean"])/n["y_std"]
 w=ridge_fit(design(fit_x,a.group),fit_y)
 metrics=[]; prediction_hashes={}
 for role,r in d["roles"].items():
  xn=(r["x"]-n["x_mean"])/n["x_std"]
  with torch.inference_mode(): neural=m(xn.to(a.device)).cpu()*n["y_std"]+n["y_mean"]
  current_pred=r["x"][:,-1,192:195]; persistence=current_pred[:,None,:].expand(-1,3,-1).clone(); ideal=r["y_current"][:,None,:].expand(-1,3,-1).clone()
  linear_norm=ridge_predict(design(xn,a.group),w); linear=linear_norm*n["y_std"]+n["y_mean"]
  preds={"neural":neural,"predicted_current_persistence":persistence,"fit_ridge_linear":linear,"gt_current_persistence_ideal_only":ideal}
  metrics += rows_for(role,r,preds)
  pp=a.output/f"predictions_{role}.pt"; ft.atomic_save({"episode_id":r["episode_id"],"leakage_group":r["leakage_group"],"t":r["t"],"y_current":r["y_current"],"y":r["y"],"predictions":preds},pp); prediction_hashes[role]=ft.sha(pp)
 with (a.output/"metrics.csv").open("w",newline="") as f:
  wri=csv.DictWriter(f,fieldnames=list(metrics[0])); wri.writeheader(); wri.writerows(metrics)
 result={"schema":"round14_future_evaluation_v1","status":"complete","group":a.group,"checkpoint":str(a.checkpoint),"checkpoint_sha256":ft.sha(a.checkpoint),"data_sha256":ft.sha(a.data),"fit_only_linear_ridge":1e-6,"prediction_hashes":prediction_hashes,"metrics_sha256":ft.sha(a.output/"metrics.csv"),"test_consumed":False,"note":"Change error uses common R10 predicted-current anchor; ideal GT-current persistence is non-deployable."}; ft.atomic_json(result,a.output/"SUMMARY.json"); print(json.dumps(result))
if __name__=="__main__": main()
