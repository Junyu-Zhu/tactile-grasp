#!/usr/bin/env python3
"""Direct repeated-window repair for the single preregistered parity exception."""
from __future__ import annotations
import argparse,importlib.util,json,random,sys
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
base=load("r23_predict_base",HERE/"predict_e3.py");tr=base.tr
ALLOWED_RUN="D/p4_s20260915"
def main():
 p=argparse.ArgumentParser();p.add_argument("--inventory",type=Path,required=True);p.add_argument("--run",choices=[ALLOWED_RUN],required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--diagnostic",type=Path,required=True);p.add_argument("--device",default="cuda:0");a=p.parse_args();diag=json.loads(a.diagnostic.read_text())
 if not diag.get("direct_required_for_this_run") or diag.get("run")!=a.run:raise ValueError("repair not justified by bounded diagnostic")
 item=next(x for x in json.loads(a.inventory.read_text())["runs"] if x["run"]==a.run);run=Path(item["output"]);summary=json.loads((run/"summary.json").read_text());cm=json.loads((run/"COMMIT.json").read_text());best=run/cm["best"]["path"];ck=torch.load(best,map_location="cpu",weights_only=False);norm=ck["normalizer"]
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True);random.seed(item["seed"]);np.random.seed(item["seed"]);torch.manual_seed(item["seed"]);torch.cuda.manual_seed_all(item["seed"])
 data=tr.Data(item["data"],item["prefix_index"]);model=tr.Model(item["group"],item["seed"],item["source"],item["visual_checkpoint"]);model.load_state_dict(ck["model"],strict=True);model.to(a.device).eval();arrays={};counts={}
 with torch.inference_mode():
  for role in tr.ROLES:
   normal=[];fmean=[];flag=[];foff=[];aux=[];aux_gt=[];physical=[];gammas=[];betas=[];ratios=[];rd=data.d["roles"][role]
   for ids in torch.arange(len(rd["t"])).split(tr.BATCH):
    z,f,gt=data.batch(role,ids);z=z.to(a.device);fn=((f-norm["force_mean"])/norm["force_std"]).to(a.device);c=model.components(z,fn);v=c["visual"];normal.append(torch.sigmoid(c["logit"]).cpu());zero=torch.zeros_like(fn);lag=torch.zeros_like(fn);lag[:,1:]=fn[:,:-1];fmean.append(torch.sigmoid(model.head.components(torch.cat((v,zero),-1))["logit"]).cpu());flag.append(torch.sigmoid(model.head.components(torch.cat((v,lag),-1))["logit"]).cpu());basev=model.head.visual_ln(v);foff.append(torch.sigmoid(model.head.risk(model.head.gru(basev)[0][:,-1]).squeeze(-1)).cpu());gammas.append(c["gamma"].cpu());betas.append(c["beta"].cpu());ratios.append((((1+c["gamma"])*basev+c["beta"]).norm(dim=-1)/(basev.norm(dim=-1)+1e-12)).cpu());pred=c["aux_force"].cpu()*norm["aux_std"]+norm["aux_mean"];aux.append(pred);aux_gt.append(gt);physical.append((f[:,-1]-gt).abs().mean(1))
   arrays[f"{role}_score"]=torch.cat(normal).numpy();arrays[f"{role}_force_mean_score"]=torch.cat(fmean).numpy();arrays[f"{role}_force_lag1_score"]=torch.cat(flag).numpy();arrays[f"{role}_film_off_score"]=torch.cat(foff).numpy();arrays[f"{role}_aux_force_pred_n"]=torch.cat(aux).numpy();arrays[f"{role}_aux_force_gt_n"]=torch.cat(aux_gt).numpy();arrays[f"{role}_physical_force_mae_n"]=torch.cat(physical).numpy();arrays[f"{role}_film_gamma"]=torch.cat(gammas).numpy();arrays[f"{role}_film_beta"]=torch.cat(betas).numpy();arrays[f"{role}_film_norm_ratio"]=torch.cat(ratios).numpy();counts[role]={"endpoints":len(rd["t"]),"endpoint_sha256":base.endpoint_hash(rd)}
 mask=data.d["roles"]["selection"]["stage"].eq(0)|data.d["roles"]["selection"]["stage"].eq(2);replayed=tr.low_fpr_auc(data.d["roles"]["selection"]["stage"][mask],torch.from_numpy(arrays["selection_score"])[mask]);err=abs(replayed-float(summary["best_metric"]))
 if err>2e-6:raise ValueError(f"direct repair replay mismatch {err}")
 base.atomic_npz(a.output/"PREDICTIONS.npz",arrays);suffix={k:v for k,v in ck["model"].items() if k.startswith("blocks.")};out={"schema":"round23_e3_prediction_v1_direct_repair","status":"complete","run":a.run,"group":item["group"],"fold":item["fold"],"seed":item["seed"],"best_commit":str(best),"best_commit_sha256":cm["best"]["sha256"],"fine_tuned_suffix_state_sha256":tr.state_hash(suffix),"checkpoint_model_loaded_strict":True,"prediction_method":"direct_repeated_window_exception","repair_diagnostic_sha256":tr.sha(a.diagnostic),"selection_pAUC_replayed":replayed,"selection_best_metric":float(summary["best_metric"]),"selection_replay_abs_error":err,"prediction_path":str(a.output/"PREDICTIONS.npz"),"prediction_sha256":tr.sha(a.output/"PREDICTIONS.npz"),"counts":counts,"source_hashes":{"inventory":tr.sha(a.inventory),"predictor":tr.sha(HERE/"predict_e3.py"),"direct_repair":tr.sha(__file__),"trainer":tr.sha(base.R22/"train_e3.py")},"test_consumed":False};tr.atomic_json(a.output/"SUMMARY.json",out);print(json.dumps(out))
if __name__=="__main__":main()
