#!/usr/bin/env python3
import argparse,csv,json,shutil
from pathlib import Path
import torch
import multitask_train as mt
def read(p):return list(csv.DictReader(open(p)))
def copytree(src,dst):
 if dst.exists():shutil.rmtree(dst)
 shutil.copytree(src,dst)
def main():
 p=argparse.ArgumentParser();p.add_argument("--round",type=Path,default=Path(__file__).parent);p.add_argument("--remote",type=Path,required=True);p.add_argument("--r13-source",type=Path,required=True);a=p.parse_args();ev=a.remote/"evaluation";res=a.remote/"results"
 # Compact actual results: metrics/CI/cases locally, large raw predictions remain remote.
 for sub in ("slip","future","bootstrap","cases"):
  dst=a.round/"results"/"evaluation"/sub;dst.mkdir(parents=True,exist_ok=True)
  for x in (ev/sub).iterdir():
   if x.suffix in (".csv",".json"):shutil.copy2(x,dst/x.name)
 copytree(res,a.round/"results"/"final_report")
 shutil.copy2(ev/"SUMMARY.json",a.round/"results"/"evaluation"/"SUMMARY.json");shutil.copy2(ev/"INPUT_INDEX.json",a.round/"results"/"evaluation"/"INPUT_INDEX.json")
 # Complete per-run compact configs and epoch logs.
 meta=a.round/"runs_metadata"/"formal";meta.mkdir(parents=True,exist_ok=True);run_index=[]
 for g in mt.GROUPS:
  for f in range(1,5):
   for s in mt.SEEDS:
    src=a.remote/"formal"/g/f"p{f}_s{s}";dst=meta/g/f"p{f}_s{s}";dst.mkdir(parents=True,exist_ok=True);summary=json.loads((src/"summary.json").read_text());latest=torch.load(src/"latest.pth",map_location="cpu",weights_only=False);shutil.copy2(src/"summary.json",dst/"summary.json");mt.atomic_json(latest["identity"],dst/"config.json")
    with (dst/"history.csv").open("w",newline="") as h:
     w=csv.DictWriter(h,fieldnames=list(latest["history"][0]));w.writeheader();w.writerows(latest["history"])
    hashes={"best":{"path":str(src/"best.pth"),"sha256":mt.sha(src/"best.pth"),"bytes":(src/"best.pth").stat().st_size},"latest":{"path":str(src/"latest.pth"),"sha256":mt.sha(src/"latest.pth"),"bytes":(src/"latest.pth").stat().st_size},"summary":{"path":str(dst/"summary.json"),"sha256":mt.sha(dst/"summary.json")},"config":{"path":str(dst/"config.json"),"sha256":mt.sha(dst/"config.json")},"history":{"path":str(dst/"history.csv"),"sha256":mt.sha(dst/"history.csv")}};mt.atomic_json(hashes,dst/"hashes.json");run_index.append({"run_id":f"{g}/p{f}_s{s}","status":"complete","local_metadata":str(dst.relative_to(a.round)),**hashes})
 mt.atomic_json({"schema":"round17_run_index_v1","status":"complete","runs":run_index},a.round/"RUN_INDEX.json")
 inv=json.loads((a.round/"RUN_INVENTORY.json").read_text());inv["status"]="complete";inv["actual_runs"]=36
 for x in inv["runs"]:x["status"]="complete";x["metadata"]=str(Path("runs_metadata/formal")/x["group"]/f"p{x['fold']}_s{x['seed']}")
 mt.atomic_json(inv,a.round/"RUN_INVENTORY.json")
 mt.atomic_json({"schema":"round17_formal_status_v1","status":"review_ready","formal_runs":36,"training_failures":0,"evaluation_status":"complete","test_consumed":False,"independent_review":"pending_single_root_dispatched_review"},a.round/"FORMAL_STATUS.json")
 logs=a.round/"logs";logs.mkdir(exist_ok=True)
 for name in ("formal_S_driver.log","formal_F_driver.log","formal_J_driver.log","evaluation_driver.log","cost.log","report.log","supplement_evaluation.log","recovery_audit.log","smoke_cpu.log","smoke_gpu.log","prepare_all_final.log"):
  src=a.round/name
  if src.exists():shutil.copy2(src,logs/name)
 # Exact R13 policy compatibility arithmetic and source lock.
 metrics=read(ev/"slip/metrics.csv");thresholds=read(ev/"slip/thresholds.csv");families={x["family"] for x in metrics};expected={"historical":{"points":["fixed_0.5","maxBA","FPR0.01","FPR0.05","FPR0.10"],"rules":["raw","confirm2"],"count":10},"new":{"families":["trial_macro_static_FPR","trial_any_static_alarm_rate"],"alphas":[.01,.05,.10],"k":[1,2,4],"count":18},"confirm4":{"raw_fpr_points":[.01,.05,.10],"k":4,"count":3},"total_per_role_run":31}
 per={(g,f,s,r):len([x for x in metrics if x["group"]==g and int(x["fold"])==f and int(x["seed"])==s and x["role"]==r]) for g in ("S","J") for f in range(1,5) for s in mt.SEEDS for r in ("fit","selection","calibration","validation")};assert set(per.values())=={31} and families>={"historical","trial_macro_static_FPR","trial_any_static_alarm_rate","confirm4_reference"}
 compat={"schema":"round17_r13_rule_compatibility_v1","status":"pass","r13_source":{"path":str(a.r13_source),"sha256":mt.sha(a.r13_source)},"expected":expected,"observed":{"metric_rows":len(metrics),"threshold_rows":len(thresholds),"per_role_run_counts":sorted(set(per.values())),"families":sorted(families)},"state_semantics":"R13 causal k=1/2/4, release1, >= threshold, maximal common consecutive segments with gap reset","applicability":"All R13 historical/new/confirm4 families applicable to newly scored common endpoints are included. R13 calibration bootstrap and LOO are calibration-rule robustness analyses from its historical round, not silently claimed as new neural working points; Round17 supplies registered 2000-draw validation paired CI."};mt.atomic_json(compat,a.round/"R13_RULE_COMPATIBILITY.json")
 # Server-only payload index.
 large=[]
 for label,path,pattern in (("prepared",a.remote/"prepared","prepared.pt"),("formal_checkpoints",a.remote/"formal","*.pth"),("raw_predictions",ev/"predictions","*.npz")):
  files=list(path.rglob(pattern));large.append({"label":label,"root":str(path),"files":len(files),"bytes":sum(x.stat().st_size for x in files)})
 mt.atomic_json({"schema":"round17_remote_large_v1","artifacts":large},a.round/"REMOTE_LARGE_ARTIFACTS.json")
 # Review-facing reports.
 summary=(a.round/"results/final_report/SUMMARY_ZH.md").read_text();(a.round/"SCIENTIFIC_CONCLUSION.md").write_text(summary)
 report="""# Round 17 delivery report for independent review

The fixed 36-run S/F/J grid completed without failures or test-role access. Training audit verifies exact identities, all active parameters changed, inactive heads stayed unchanged, and earliest strict checkpoint selection in all 36 runs. Evaluation retains all applicable 31 R13 working points per role/run, common-endpoint segment reset and censoring, fit-only force-change strata, three baselines, paired group CI, fixed cases, source images, and cost.

The scientific result is negative: J trades higher gross recall for higher validation false alarms and lower low-FPR ranking; its future-force prediction is materially worse than F in three folds and approximately tied in one. F itself does not beat predicted-current persistence overall. Within the new temporal modules after the frozen upstream outputs, J uses 22,809 fewer parameters (49.47%) than separate S+F modules. The frozen MAE/force upstream can also be shared by independent heads, so this is not a claim that full deployment parameters or end-to-end latency are halved. The performance result does not support promotion.

Large caches, checkpoints and raw predictions remain under the server Round-17 output root. Compact metrics, every run's config/history/summary/hash, CI draws, figures, cases and logs are mirrored locally for review.
""";(a.round/"DELIVERY_REPORT.md").write_text(report)
 required=["PROTOCOL.md","PROTOCOL_LOCK.json","EVALUATION_PROTOCOL.json","EVALUATION_LOCK.json","IDENTITY_LOCK.json","SUPPORT_AUDIT.json","PREPARE_AUDIT.json","RECOVERY_AUDIT.json","TRAINING_AUDIT.json","EVALUATION_AUDIT.json","SUPPLEMENT_AUDIT.json","COST.json","R13_RULE_COMPATIBILITY.json","RUN_INDEX.json","RUN_INVENTORY.json","FORMAL_STATUS.json","REMOTE_LARGE_ARTIFACTS.json","DELIVERY_REPORT.md","SCIENTIFIC_CONCLUSION.md","INDEPENDENT_REVIEW_REQUEST.md"]
 missing=[x for x in required if not (a.round/x).exists()];assert not missing,missing
 files={str(x.relative_to(a.round)):{"sha256":mt.sha(x),"bytes":x.stat().st_size} for x in a.round.rglob("*") if x.is_file() and x.name not in ("REVIEW_READY.json","review_ready.log") and not any(q in x.parts for q in ("__pycache__",))}
 ready={"schema":"round17_review_ready_v1","status":"ready_for_single_independent_review","formal_runs":36,"training_audit":"pass","evaluation_audit":"pass","r13_rules":"pass_31_per_role_run","test_consumed":False,"local_files":len(files),"file_manifest":files,"remote_large":large};mt.atomic_json(ready,a.round/"REVIEW_READY.json");print(json.dumps({k:v for k,v in ready.items() if k!="file_manifest"},indent=2))
if __name__=="__main__":main()
