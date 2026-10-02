#!/usr/bin/env python3
"""Structural and numerical audit of Round-18 evaluation outputs."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import train as r18
def rows(path):return list(csv.DictReader(Path(path).open()))
def main():
 p=argparse.ArgumentParser();p.add_argument("--evaluation",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();s=json.loads((a.evaluation/"SUMMARY.json").read_text());work=rows(a.evaluation/"metrics/WORKPOINT_METRICS.csv");trials=rows(a.evaluation/"metrics/TRIAL_METRICS.csv");th=rows(a.evaluation/"metrics/CALIBRATION_THRESHOLDS.csv");ci=rows(a.evaluation/"bootstrap/PAIRED_CI.csv");draws=rows(a.evaluation/"bootstrap/PAIRED_DRAWS.csv");pert=rows(a.evaluation/"diagnostics/FORCE_PERTURBATIONS.csv");film=rows(a.evaluation/"diagnostics/FILM_STATS.csv");index=json.loads((a.evaluation/"INPUT_INDEX.json").read_text());sel=json.loads((a.evaluation/"cases/SELECTED.json").read_text())
 checks={"summary_complete":s["status"]=="complete","runs_36":len(index["runs"])==36,"unique_grid":len({(x["group"],x["fold"],x["seed"]) for x in index["runs"]})==36,"workpoints_1440":len(work)==1440,"thresholds_180":len(th)==180,"ci_72":len(ci)==72,"draw_rows_144000":len(draws)==144000,"perturbations_84":len(pert)==84,"film_rows_48":len(film)==48,"cases_5":len(sel["selected"])==5,"test_not_consumed":s["test_consumed"] is False,"v0_force_invariant":all(float(x["max_abs_score_change"])==0 for x in pert if x["group"]=="V0"),"ci_support_accounted":all(int(x["valid_draws"])+int(x["invalid_draws"])==2000 and int(x["valid_draws"])>0 for x in ci),"all_workpoint_numeric":all(all(value!="" and math.isfinite(float(value)) for value in (x["frame_static_FPR"],x["gross_recall"],x["balanced_accuracy"],x["macro_f1"],x["AP"],x["pAUC"])) for x in work),"hashes_match":all(r18.sha(a.evaluation/path)==digest for path,digest in s["hashes"].items())}
 result={"schema":"round18_evaluation_audit_v1","status":"pass" if all(checks.values()) else "fail","checks":checks,"counts":{"workpoints":len(work),"trials":len(trials),"thresholds":len(th),"ci":len(ci),"draws":len(draws),"perturbations":len(pert),"film":len(film)},"support_boundary":s["support_boundary"],"test_consumed":False};r18.atomic_json(a.output,result);print(json.dumps(result,indent=2))
 if result["status"]!="pass":raise SystemExit(1)
if __name__=="__main__":main()

