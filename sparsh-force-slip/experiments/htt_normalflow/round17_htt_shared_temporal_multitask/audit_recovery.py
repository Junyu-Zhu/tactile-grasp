#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import torch
import multitask_train as mt
def exact(a,b):
 if torch.is_tensor(a) or torch.is_tensor(b):return torch.is_tensor(a) and torch.is_tensor(b) and torch.equal(a,b)
 if isinstance(a,dict) or isinstance(b,dict):return isinstance(a,dict) and isinstance(b,dict) and a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)) or isinstance(b,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
 return a==b
def main():
 p=argparse.ArgumentParser();p.add_argument("--full",type=Path,required=True);p.add_argument("--resumed",type=Path,required=True);p.add_argument("--output",type=Path,default=Path(__file__).with_name("RECOVERY_AUDIT.json"));a=p.parse_args();fs=json.loads((a.full/"summary.json").read_text());rs=json.loads((a.resumed/"summary.json").read_text());fc=torch.load(a.full/"latest.pth",map_location="cpu",weights_only=False);rc=torch.load(a.resumed/"latest.pth",map_location="cpu",weights_only=False);fb=torch.load(a.full/"best.pth",map_location="cpu",weights_only=False);rb=torch.load(a.resumed/"best.pth",map_location="cpu",weights_only=False)
 checks={"identity":fc["identity"]==rc["identity"],"latest_model_all_tensors":exact(fc["model"],rc["model"]),"latest_optimizer_full_state_and_param_groups":exact(fc["optimizer"],rc["optimizer"]),"optimizer_steps":fc["optimizer_steps"]==rc["optimizer_steps"],"sampler_rng":torch.equal(fc["generator"],rc["generator"]),"best_epoch":fs["best_epoch"]==rs["best_epoch"],"best_metric":fs["best_metric"]==rs["best_metric"],"history":fc["history"]==rc["history"],"normalizer":exact(fc["normalizer"],rc["normalizer"]),"best_checkpoint_identity":fb["identity"]==rb["identity"],"best_checkpoint_model_all_tensors":exact(fb["model"],rb["model"]),"best_checkpoint_optimizer_full_state_and_param_groups":exact(fb["optimizer"],rb["optimizer"]),"best_checkpoint_normalizer":exact(fb["normalizer"],rb["normalizer"]),"best_checkpoint_history":fb["history"]==rb["history"]}
 if not all(checks.values()):raise AssertionError(checks)
 out={"schema":"round17_real_recovery_audit_v2","status":"pass","scope":"real p1 seed20260914 common cache, J, CUDA GPU0, four epochs; interruption committed after epoch 1 and resumed in a new process","checks":checks,"full_latest_sha256":mt.sha(a.full/"latest.pth"),"resumed_latest_sha256":mt.sha(a.resumed/"latest.pth"),"full_best_sha256":mt.sha(a.full/"best.pth"),"resumed_best_sha256":mt.sha(a.resumed/"best.pth"),"latest_model_state_sha256":mt.state_hash(fc["model"]),"best_model_state_sha256":mt.state_hash(fb["model"]),"note":"Container bytes may differ, while latest and best model tensors, complete optimizer state including moments and param_groups, RNG, normalizers, history, selection, and counters are exact."};mt.atomic_json(out,a.output);print(json.dumps(out,indent=2))
if __name__=="__main__":main()
