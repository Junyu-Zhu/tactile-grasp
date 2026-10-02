#!/usr/bin/env python3
"""Verify and consolidate the fixed force-analysis queue."""
from __future__ import annotations
import argparse,csv,json,statistics
from pathlib import Path
from analysis_common import atomic_json,sha256
from run_analysis import accepted,code_bundle

def mean(rows,key):return statistics.fmean(row[key] for row in rows)

def main():
    p=argparse.ArgumentParser();p.add_argument("--manifest",type=Path,required=True);p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    manifest=json.loads(args.manifest.read_text());manifest_sha=sha256(args.manifest.resolve());bundle=code_bundle();invalid=[];payloads={}
    for job in manifest["jobs"]:
        if not accepted(job,manifest_sha,bundle["sha256"]):invalid.append(job["id"]);continue
        payloads[job["id"]]=json.loads(Path(job["acceptance_path"]).read_text())
    if invalid:raise ValueError(f"analysis queue incomplete or identity-drifted ({len(invalid)}): {invalid[:8]}")
    force=[]
    for key,value in payloads.items():
        if not key.startswith("force_"):continue
        force.append({"fold":value["fold"],"seed":value["seed"],"role":value["role"],"axis_rmse_n":value["supervised_regression"]["axis"]["rmse_n"],"Fn_rmse_n":value["supervised_regression"]["invariant"]["Fn"]["rmse_n"],"Ft_rmse_n":value["supervised_regression"]["invariant"]["Ft"]["rmse_n"],"Fmag_rmse_n":value["supervised_regression"]["invariant"]["Fmag"]["rmse_n"],"target_any_saturation_fraction":value["target_saturation"]["any_axis_fraction"],"collapse_pass":value["collapse_check"]["pass"],"trainmean_axis_improvement_n":value["improvement_over_train_mean"]["axis_rmse_n"]})
    sensitivity=[]
    for key,value in payloads.items():
        if not key.startswith("sensitivity_"):continue
        sensitivity.append({"variant":value["variant"],"fold":value["fold"],"seed":value["seed"],"role":value["role"],"original_partial_tpr_auc_0_0p1":value["metrics"]["original"]["partial_tpr_auc_0_0p1"],"masked_mean_abs_dp":value["probability_change"]["masked_zero_train_standardized"]["mean_absolute"],"mismatched_mean_abs_dp":value["probability_change"]["mismatched_half_cycle"]["mean_absolute"],"masked_flip_fraction":value["probability_change"]["masked_zero_train_standardized"]["class_flip_at_0p5"],"mismatched_flip_fraction":value["probability_change"]["mismatched_half_cycle"]["class_flip_at_0p5"]})
    aggregates={}
    for role in ("calibration","validation"):
        rows=[r for r in force if r["role"]==role]
        aggregates[role]={"runs":len(rows),"all_collapse_checks_pass":all(r["collapse_pass"] for r in rows),"mean_axis_rmse_n":[statistics.fmean(r["axis_rmse_n"][i] for r in rows) for i in range(3)],"mean_Fn_rmse_n":mean(rows,"Fn_rmse_n"),"mean_Ft_rmse_n":mean(rows,"Ft_rmse_n"),"mean_Fmag_rmse_n":mean(rows,"Fmag_rmse_n"),"mean_target_any_saturation_fraction":mean(rows,"target_any_saturation_fraction"),"mean_trainmean_axis_improvement_n":[statistics.fmean(r["trainmean_axis_improvement_n"][i] for r in rows) for i in range(3)]}
    sensitivity_aggregates={}
    for variant in ("F-old","F-adapt"):
        sensitivity_aggregates[variant]={}
        for role in ("calibration","validation"):
            rows=[r for r in sensitivity if r["variant"]==variant and r["role"]==role]
            sensitivity_aggregates[variant][role]={"runs":len(rows),"mean_original_partial_tpr_auc_0_0p1":mean(rows,"original_partial_tpr_auc_0_0p1"),"mean_masked_absolute_probability_change":mean(rows,"masked_mean_abs_dp"),"mean_mismatched_absolute_probability_change":mean(rows,"mismatched_mean_abs_dp"),"mean_masked_class_flip_fraction":mean(rows,"masked_flip_fraction"),"mean_mismatched_class_flip_fraction":mean(rows,"mismatched_flip_fraction")}
    e2e={name.removeprefix("e2e_"):value for name,value in payloads.items() if name.startswith("e2e_")}
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    with (output/"force_runs.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=force[0].keys());writer.writeheader();writer.writerows(force)
    with (output/"sensitivity_runs.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=sensitivity[0].keys());writer.writeheader();writer.writerows(sensitivity)
    result={"status":"complete","format":"round5_force_analysis_summary_v1","manifest_sha256":manifest_sha,"code_bundle_sha256":bundle["sha256"],"verified_jobs":len(payloads),"force_runs":force,"force_aggregate":aggregates,"sensitivity_runs":sensitivity,"sensitivity_aggregate":sensitivity_aggregates,"source_historical":payloads["source_old"],"source_current_raw_evaluation":payloads["source_adapt_all"],"e2e":e2e,"failure_exports":{k:v for k,v in payloads.items() if k.startswith("failures_")},"boundaries":["HTT forces are reference-relative and clipped; Fn/Ft/Fmag and ratios are not calibrated friction coefficients or physical stability margins.","HTT-adapted signed axes are not mapped to source-domain signed axes; adapted source results are descriptive only.","Sensitivity interventions measure model reliance and do not prove physical causality.","Latency starts with resident raw arrays and excludes sensor capture and NPZ I/O."]}
    atomic_json(output/"summary.json",result)
    (output/"SUMMARY_ZH.md").write_text("# 第五轮力与条件输入分析\n\n所有82项分析产物已通过固定manifest、命令、输入SHA和代码bundle校验。定量结果见 `summary.json`、`force_runs.csv` 与 `sensitivity_runs.csv`。\n\n边界：HTT力是参考相对且裁剪的目标，Fn/Ft/Fmag及比值不是经过标定的摩擦系数或物理稳定裕度；跨源域adapt输出仅作分布描述；输入屏蔽/错配仅说明模型依赖；延迟从驻内存原始数组开始，不包含传感器采集与NPZ I/O。\n")
    print(json.dumps({"status":"complete","verified_jobs":len(payloads),"output":str(output/"summary.json")},indent=2))

if __name__=="__main__":main()
