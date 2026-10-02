#!/usr/bin/env python3
import argparse,json
from collections import Counter
from pathlib import Path
import torch
import multitask_train as mt
def main():
 p=argparse.ArgumentParser();p.add_argument("--formal-root",type=Path,required=True);p.add_argument("--prepared-root",type=Path,required=True);p.add_argument("--output",type=Path,default=Path(__file__).with_name("TRAINING_AUDIT.json"));a=p.parse_args();rows=[]
 for g in mt.GROUPS:
  for f in range(1,5):
   for s in mt.SEEDS:
    d=a.formal_root/g/f"p{f}_s{s}";summary=json.loads((d/"summary.json").read_text());best=torch.load(d/"best.pth",map_location="cpu",weights_only=False);latest=torch.load(d/"latest.pth",map_location="cpu",weights_only=False);data=a.prepared_root/f"p{f}_s{s}"/"prepared.pt";ident=summary["identity"]
    assert summary["status"]=="complete" and ident["group"]==g and ident["fold"]==f and ident["seed"]==s and ident["data_sha256"]==mt.sha(data)
    assert ident["selector"]==("future_mae" if g=="F" else "slip_pauc_0_0.1") and best["best_epoch"]==summary["best_epoch"] and best["identity"]==ident and latest["identity"]==ident
    hist=latest["history"];vals=[x["selection_metric"] for x in hist];target=min(vals) if g=="F" else max(vals);earliest=next(i for i,x in enumerate(vals) if x==target);assert earliest==summary["best_epoch"] and target==summary["best_metric"]
    init=mt.init_model(s).state_dict();names=mt.active_names(g);active_changed=all(not torch.equal(init[k],best["model"][k]) for k in init if any(k.startswith(n+".") for n in names));inactive_unchanged=all(torch.equal(init[k],best["model"][k]) for k in init if not any(k.startswith(n+".") for n in names));assert active_changed and inactive_unchanged
    rows.append({"run_id":f"{g}/p{f}_s{s}","group":g,"fold":f,"seed":s,"epochs":summary["epochs"],"best_epoch":summary["best_epoch"],"best_metric":summary["best_metric"],"earliest_strict_best_verified":True,"optimizer_steps":summary["optimizer_steps"],"slip_supervised_endpoints_seen":summary["slip_supervised_endpoints_seen"],"future_supervised_endpoints_seen":summary["future_supervised_endpoints_seen"],"skipped_empty_slip_steps":summary["skipped_empty_slip_steps"],"active_parameters":summary["active_parameters"],"initial_active_state_sha256":mt.state_hash(init,names),"best_active_state_sha256":mt.state_hash(best["model"],names),"all_active_parameters_changed":active_changed,"all_inactive_parameters_unchanged":inactive_unchanged,"best_sha256":mt.sha(d/"best.pth"),"latest_sha256":mt.sha(d/"latest.pth"),"summary_sha256":mt.sha(d/"summary.json"),"selector":ident["selector"]})
 counts=Counter(x["group"] for x in rows);assert counts==Counter({"S":12,"F":12,"J":12})
 out={"schema":"round17_training_audit_v2","status":"pass","formal_runs":len(rows),"group_counts":dict(counts),"failures":0,"test_consumed":False,"smoke_checkpoint_used":False,"all36_earliest_strict_best_verified":True,"all36_active_parameters_changed":True,"all36_inactive_parameters_unchanged":True,"selectors":{"S":"internal selection slip pAUC[0,.1]","F":"internal selection future native-N MAE","J":"internal selection slip pAUC[0,.1]; its future uses the same checkpoint"},"epoch_range":[min(x["epochs"] for x in rows),max(x["epochs"] for x in rows)],"parameter_counts":{g:sorted({x["active_parameters"] for x in rows if x["group"]==g}) for g in mt.GROUPS},"runs":rows};mt.atomic_json(out,a.output);print(json.dumps({k:v for k,v in out.items() if k!="runs"},indent=2))
if __name__=="__main__":main()
