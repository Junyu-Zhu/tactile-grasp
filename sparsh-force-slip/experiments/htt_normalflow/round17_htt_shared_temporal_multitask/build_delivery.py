#!/usr/bin/env python3
"""Build the final compact delivery only after review pass and root metadata freeze."""
import argparse,json
from pathlib import Path
import multitask_train as mt
EXCLUDE={"DELIVERY_MANIFEST.json","FINAL_STATUS.json","FINAL_SYNC_PROOF.json"}
def main():
 p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path(__file__).parent);a=p.parse_args();review=json.loads((a.root/"reviews/INDEPENDENT_FINAL_REVIEW.json").read_text());control=json.loads((a.root/"ROOT_CONTROL.json").read_text())
 open_findings=review.get("open_findings",0)
 if review.get("status")!="pass" or (isinstance(open_findings,(list,dict,str)) and len(open_findings)!=0) or (not isinstance(open_findings,(list,dict,str)) and open_findings!=0):raise SystemExit("independent review has not passed")
 if control.get("status") not in ("metadata_frozen_for_final_sync","complete"):raise SystemExit("root metadata not frozen")
 if not (a.root/"ROOT_PROVENANCE.json").is_file():raise SystemExit("ROOT_PROVENANCE.json missing")
 run_logs=list((a.root/"runs_metadata/formal").glob("*/*/history.csv"));run_configs=list((a.root/"runs_metadata/formal").glob("*/*/config.json"));run_summaries=list((a.root/"runs_metadata/formal").glob("*/*/summary.json"));run_hashes=list((a.root/"runs_metadata/formal").glob("*/*/hashes.json"))
 if not all(len(x)==36 for x in (run_logs,run_configs,run_summaries,run_hashes)):raise SystemExit(f"incomplete 36-run metadata: {[len(x) for x in (run_logs,run_configs,run_summaries,run_hashes)]}")
 files={}
 for x in sorted(a.root.rglob("*")):
  if x.is_file() and x.name not in EXCLUDE and "__pycache__" not in x.parts:files[str(x.relative_to(a.root))]={"sha256":mt.sha(x),"bytes":x.stat().st_size}
 manifest={"schema":"round17_delivery_manifest_v2","status":"ready_for_sync","self_reference_exclusions":sorted(EXCLUDE),"run_log_semantics":"runs_metadata/formal/<group>/<run>/history.csv is the actual per-epoch training log for each of all 36 runs; group driver logs retain dispatch/stdout summaries","run_histories":len(run_logs),"run_configs":len(run_configs),"run_summaries":len(run_summaries),"run_hash_records":len(run_hashes),"files":files,"file_count":len(files),"bytes":sum(x["bytes"] for x in files.values())};mt.atomic_json(manifest,a.root/"DELIVERY_MANIFEST.json");print(json.dumps({k:v for k,v in manifest.items() if k!="files"},indent=2))
if __name__=="__main__":main()
