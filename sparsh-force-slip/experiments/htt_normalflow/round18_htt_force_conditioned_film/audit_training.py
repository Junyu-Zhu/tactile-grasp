#!/usr/bin/env python3
"""Audit all Round-18 formal training artifacts before evaluation."""
from __future__ import annotations
import argparse,json
from collections import Counter
from pathlib import Path
import torch
import train as r18

def main():
 p=argparse.ArgumentParser();p.add_argument("--inventory",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();inventory=json.loads(a.inventory.read_text());rows=[]
 for run in inventory["runs"]:
  out=Path(run["output"]);summary=json.loads((out/"summary.json").read_text());best=torch.load(out/"best.pth",map_location="cpu",weights_only=False);latest=torch.load(out/"latest.pth",map_location="cpu",weights_only=False)
  identity=summary["identity"];assert summary["status"]=="complete" and identity==best["identity"]==latest["identity"];assert identity["group"]==run["group"] and identity["fold"]==run["fold"] and identity["seed"]==run["seed"] and identity["input_sha256"]==run["input_sha256"]
  values=[item["selection_pAUC_0_0.1"] for item in latest["history"]];target=max(values);earliest=values.index(target);assert earliest==summary["best_epoch"] and target==summary["best_metric"]
  assert summary["input_unchanged"] and summary["upstream_frozen_by_cached_input_boundary"] and summary["all_parameters_structurally_active"] and all(summary["parameter_changes_at_best"].values())
  assert "smoke" not in str(out) and identity["formal"] is True and identity["selector"]=="internal_selection_pAUC_0_0.1_earliest_strict_max"
  rows.append({"run_id":run["run_id"],"group":run["group"],"fold":run["fold"],"seed":run["seed"],"epochs":summary["epochs"],"best_epoch":summary["best_epoch"],"best_metric":summary["best_metric"],"parameters":summary["parameters"],"best_sha256":r18.sha(out/"best.pth"),"latest_sha256":r18.sha(out/"latest.pth"),"summary_sha256":r18.sha(out/"summary.json")})
 counts=Counter(row["group"] for row in rows);result={"schema":"round18_training_audit_v1","status":"pass","formal_runs":len(rows),"group_counts":dict(counts),"all_earliest_strict_selection":True,"all_parameters_updated":True,"all_inputs_unchanged":True,"upstream_frozen_by_cache_boundary":True,"smoke_checkpoint_used":False,"test_consumed":False,"epoch_range":[min(row["epochs"] for row in rows),max(row["epochs"] for row in rows)],"parameter_counts":{g:sorted({row["parameters"] for row in rows if row["group"]==g}) for g in r18.GROUPS},"runs":rows};r18.atomic_json(a.output,result);print(json.dumps({k:v for k,v in result.items() if k!="runs"},indent=2))
if __name__=="__main__":main()
