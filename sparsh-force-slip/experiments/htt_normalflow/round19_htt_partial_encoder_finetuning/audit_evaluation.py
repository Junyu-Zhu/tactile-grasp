#!/usr/bin/env python3
"""Read-only structural and hash audit for unified Round-19 evaluation."""
import argparse,csv,json,math
from pathlib import Path
import train as tr
def rows(p):
 with open(p,newline="") as f:return list(csv.DictReader(f))
def main():
 p=argparse.ArgumentParser();p.add_argument("--evaluation",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();s=json.loads((a.evaluation/"SUMMARY.json").read_text());idx=json.loads((a.evaluation/"INPUT_INDEX.json").read_text());work=rows(a.evaluation/"metrics/WORKPOINT_METRICS.csv");th=rows(a.evaluation/"metrics/CALIBRATION_THRESHOLDS.csv");ci=rows(a.evaluation/"bootstrap/PAIRED_CI.csv");draws=rows(a.evaluation/"bootstrap/PAIRED_DRAWS.csv");pert=rows(a.evaluation/"diagnostics/FORCE_PERTURBATIONS.csv");film=rows(a.evaluation/"diagnostics/FILM_SUMMARY.csv");feature=rows(a.evaluation/"diagnostics/FILM_FEATURE_TIME_ROLE.csv");selected=json.loads((a.evaluation/"cases/SELECTED.json").read_text())
 checks={"summary_complete":s["status"]=="complete","runs_48":len(idx["runs"])==48,"unique_grid":len({(x["group"],x["fold"],x["seed"]) for x in idx["runs"]})==48,"workpoints_1920":len(work)==1920,"thresholds_240":len(th)==240,"ci_224":len(ci)==224,"draw_rows_448000":len(draws)==448000,"perturbations_60":len(pert)==60,"film_summary_48":len(film)==48,"film_feature_82944":len(feature)==82944,"cases_6":len(selected["selected"])==6,"test_not_consumed":s["test_consumed"] is False,"v2_force_invariant":all(float(x["max_abs_score_change"])==0 for x in pert if x["group"]=="V2"),"ci_support_accounted":all(int(x["valid_draws"])+int(x["invalid_draws"])==2000 for x in ci),"contrasts_exact":{x["contrast"] for x in ci}==set(s["primary_contrasts"]),"all_workpoint_numeric":all(all(v!="" and math.isfinite(float(v)) for v in (x["frame_static_FPR"],x["gross_recall"],x["balanced_accuracy"],x["macro_f1"],x["AP"],x["pAUC"])) for x in work),"hashes_match":all(tr.sha(a.evaluation/path)==digest for path,digest in s["hashes"].items())}
 result={"schema":"round19_evaluation_audit_v1","status":"pass" if all(checks.values()) else "fail","checks":checks,"test_consumed":False};tr.atomic_json(a.output,result);print(json.dumps(result,indent=2));assert result["status"]=="pass"
if __name__=="__main__":main()
