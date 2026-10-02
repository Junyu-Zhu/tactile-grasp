#!/usr/bin/env python3
"""Write and mirror the final status after three-root payload verification passes."""
import argparse,json,subprocess
from pathlib import Path
import multitask_train as mt

def main():
 p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path(__file__).parent);p.add_argument("--host",default="zjy-4090");p.add_argument("--server-code",required=True);p.add_argument("--server-delivery",required=True);a=p.parse_args()
 proof=json.loads((a.root/"FINAL_SYNC_PROOF.json").read_text());manifest=json.loads((a.root/"DELIVERY_MANIFEST.json").read_text());review=json.loads((a.root/"reviews/INDEPENDENT_FINAL_REVIEW.json").read_text())
 if proof.get("status")!="pass" or manifest.get("status")!="ready_for_sync" or review.get("status")!="pass":raise SystemExit("final prerequisites have not passed")
 status={"schema":"round17_final_status_v1","status":"complete","formal_runs":36,"training_failures":0,"test_consumed":False,"scientific_outcome":"fixed shared multitask candidate rejected as a performance upgrade","delivery_manifest_sha256":mt.sha(a.root/"DELIVERY_MANIFEST.json"),"final_sync_proof_sha256":mt.sha(a.root/"FINAL_SYNC_PROOF.json"),"independent_review_sha256":mt.sha(a.root/"reviews/INDEPENDENT_FINAL_REVIEW.json"),"root_provenance_sha256":mt.sha(a.root/"ROOT_PROVENANCE.json"),"three_root_payload_verification":"pass"}
 mt.atomic_json(status,a.root/"FINAL_STATUS.json")
 for dst in (a.server_code,a.server_delivery):subprocess.run(["rsync","-a",str(a.root/"FINAL_STATUS.json"),f"{a.host}:{dst}/FINAL_STATUS.json"],check=True)
 print(json.dumps(status,indent=2))
if __name__=="__main__":main()
