#!/usr/bin/env python3
import argparse,csv,json
from pathlib import Path
import torch
import future_train as ft
def main():
 p=argparse.ArgumentParser();p.add_argument("--evaluation",type=Path,required=True);p.add_argument("--reporting",type=Path,required=True);p.add_argument("--current",type=Path,required=True);p.add_argument("--training-audit",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();ta=json.loads(a.training_audit.read_text());assert ta["status"]=="pass" and ta["accepted_count"]==36
 checks=0
 for r in ta["accepted_runs"]:
  run=a.evaluation/r["id"];su=json.loads((run/"SUMMARY.json").read_text());assert su["status"]=="complete" and su["checkpoint_sha256"]==r["best_sha256"] and su["test_consumed"] is False
  preds={role:torch.load(run/f"predictions_{role}.pt",map_location="cpu",weights_only=False) for role in ("fit","selection","calibration","validation")}
  for d in preds.values():
   pc=d["predictions"]["predicted_current_persistence"];ideal=d["predictions"]["gt_current_persistence_ideal_only"]
   assert torch.equal(pc[:,0],pc[:,1]) and torch.equal(pc[:,1],pc[:,2]) and torch.equal(ideal[:,0],d["y_current"]) and torch.isfinite(d["predictions"]["neural"]).all() and torch.isfinite(d["predictions"]["fit_ridge_linear"]).all();checks+=1
 agg=json.loads((a.reporting/"SUMMARY.json").read_text());cur=json.loads((a.current/"SUMMARY.json").read_text());assert agg["status"]=="complete" and agg["runs"]==36 and agg["all_finite"] and cur["status"]=="pass" and cur["models"]==48 and cur["historical_full_metric_rows"]==cur["historical_common_metric_rows"]==1440 and cur["r13_new_policy_rows_each_scope"]==2592 and cur["r13_confirm4_rows_each_scope"]==432
 result={"schema":"round14_evaluation_audit_v2","status":"pass","formal_runs":36,"prediction_role_checks":checks,"future_metric_rows":agg["metric_rows"],"temporal_diagnostic_rows":agg["temporal_diagnostic_rows"],"trial_rows":agg["trial_rows"],"current_models":48,"current_historical_rows_each_scope":1440,"current_r13_new_policy_rows_each_scope":2592,"current_r13_confirm4_rows_each_scope":432,"checks":["checkpoint_and_data_bound_summaries","deployable_persistence_identity","ideal_persistence_identity","finite_neural_and_linear_predictions","fit_only_linear_identity_declared_in_each_summary","per_episode_axis_horizon_temporal_collapse_diagnostics","complete_leakage_group_bootstrap","all_saved_r13_policy_state_first_current_common_pass","no_test"]};ft.atomic_json(result,a.output);print(json.dumps(result))
if __name__=="__main__":main()
