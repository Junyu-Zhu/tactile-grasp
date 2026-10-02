#!/usr/bin/env python3
"""Audit complete work-package-C outputs and provenance."""
import argparse, hashlib, json, os
from pathlib import Path

MODELS=("mae","dino","ijepa","mae_letterbox"); SEEDS=(20260914,20260915,20260916)
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--code",type=Path,required=True);a=p.parse_args()
    expected={(m,f"htt_leave_p{i}",s) for m in MODELS for i in range(1,5) for s in SEEDS}; actual=set(); errors=[]
    evaluator_sha=sha(a.code/"evaluate_alarms.py"); protocol_sha=sha(a.code.parent/"PROTOCOL.md")
    for path in sorted((a.root/"runs").glob("*/*/seed_*/metrics.json")):
        data=json.loads(path.read_text()); identity=(data["model"],data["fold"],data["seed"]); actual.add(identity)
        if data.get("status")!="complete":errors.append(f"incomplete {identity}")
        provenance=data.get("provenance",{})
        if provenance.get("evaluator_sha256")!=evaluator_sha or provenance.get("master_protocol_sha256")!=protocol_sha:errors.append(f"code/protocol mismatch {identity}")
        for role in ("calibration","validation"):
            source=Path(data["source"])/f"predictions/{role}.csv"
            if sha(source)!=provenance.get(f"{role}_predictions_sha256"):errors.append(f"prediction hash mismatch {identity}/{role}")
        for name,item in data["operating_points"].items():
            for mode in ("raw","sequential"):
                if mode not in item:continue
                for role in ("calibration","validation"):
                    metrics=item[mode][role]
                    if not all(key in metrics for key in ("never_alarm","observed_no_alarm","observed_no_primary_alarm","average_precision","positive_prevalence")):errors.append(f"missing explicit fields {identity}/{name}/{mode}/{role}")
                if name.startswith("fpr_") and item[mode]["calibration"]["static_fpr"]>item.get("sequential",{}).get("budget",float(name.split('_')[1]))+1e-12:errors.append(f"calibration budget violation {identity}/{name}/{mode}")
        if data.get("nonunique_leakage_groups"):errors.append(f"unexpected nonunique leakage groups {identity}")
        if not (path.parent/"trials.csv").is_file():errors.append(f"missing trial CSV {identity}")
    summary=json.loads((a.root/"summary.json").read_text())
    if actual!=expected:errors.append(f"identity mismatch missing={sorted(expected-actual)} unexpected={sorted(actual-expected)}")
    if summary.get("status")!="complete" or not summary.get("expected_identities_present"):errors.append("summary incomplete")
    run_csv=a.root/"run_metrics.csv"
    if not run_csv.is_file() or sum(1 for _ in run_csv.open())!=385:errors.append("run_metrics.csv expected header+384 rows")
    payload={"status":"pass" if not errors else "fail","errors":errors,"runs":len(actual),"expected_runs":48,
             "evaluator_sha256":evaluator_sha,"master_protocol_sha256":protocol_sha,"summary_sha256":sha(a.root/"summary.json"),
             "run_metrics_sha256":sha(run_csv) if run_csv.is_file() else None,"unit_tests":"5 tests required separately"}
    target=a.root/"FINAL_AUDIT.json";tmp=target.with_name(target.name+f".tmp.{os.getpid()}");tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");os.replace(tmp,target)
    print(json.dumps(payload,indent=2));raise SystemExit(bool(errors))
if __name__=="__main__":main()
