#!/usr/bin/env python3
"""Freeze and verify the pre-recovery Round-19 hardware-failure evidence."""
import argparse,json,subprocess
from pathlib import Path
import torch
import run_all as q

def main():
 p=argparse.ArgumentParser();p.add_argument("--local",type=Path,default=Path(__file__).resolve().parent);p.add_argument("--output",type=Path);a=p.parse_args();a.output=a.output or a.local/"HARDWARE_FAILURE_AUDIT.json"
 status_path=a.local/"FORMAL_STATUS.json";inventory_path=a.local/"RUN_INVENTORY.json";status=json.loads(status_path.read_text());rows=[];failures=[]
 for run in status["runs"]:
  row={"run":run["run"],"preserved_status":run["status"],"gpu_index":run.get("gpu"),"attempts":run.get("attempts"),"log":run.get("log"),"log_sha256":q.tr.sha(run["log"]) if run.get("log") and Path(run["log"]).exists() else None}
  commit_path=Path(run["output"])/"COMMIT.json"
  if commit_path.exists():
   manifest=json.loads(commit_path.read_text());latest=Path(run["output"])/manifest["latest"]["path"];row.update(commit_sha256=q.tr.sha(commit_path),latest_epoch=manifest["latest"]["epoch"],latest_sha256=q.tr.sha(latest),latest_hash_matches=q.tr.sha(latest)==manifest["latest"]["sha256"])
   checkpoint=torch.load(latest,map_location="cpu",weights_only=False);row["checkpoint_identity_matches"]=(checkpoint["identity"]["group"],checkpoint["identity"]["fold"],checkpoint["identity"]["seed"])==(run["group"],run["fold"],run["seed"])
  else:row.update(commit_sha256=None,latest_epoch=None,latest_sha256=None,latest_hash_matches=None,checkpoint_identity_matches=None)
  if run["status"]=="complete":
   row["accepted_complete_hash_commit_identity"]=q.accepted_summary(run) is not None
   if not row["accepted_complete_hash_commit_identity"]:failures.append(run["run"]+": completed acceptance failed")
  elif run["status"]=="failed_terminal":
   row["verified_hardware_failure"]=q.verified_hardware_terminal(run)
   row["attempt_failure_classes"]=[x.get("class") for x in run.get("recovery_history",[]) if x.get("event")=="attempt_failed"]
   row["original_terminal_reason"]=run.get("terminal_reason")
   if not row["verified_hardware_failure"]:failures.append(run["run"]+": terminal failure lacks exclusive CUDA hardware evidence")
  else:failures.append(run["run"]+": unexpected status "+run["status"])
  rows.append(row)
 smi=subprocess.run(["nvidia-smi","--query-gpu=index,uuid,name,pci.bus_id","--format=csv,noheader"],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 pci=subprocess.run(["lspci","-s","e1:00.0","-vv"],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 counts={}
 for r in rows:counts[r["preserved_status"]]=counts.get(r["preserved_status"],0)+1
 if counts!={"complete":9,"failed_terminal":15}:failures.append("unexpected preserved counts: "+repr(counts))
 result={"schema":"round19_hardware_failure_audit_v1","status":"pass" if not failures else "fail","created_at":q.now().isoformat(),"source_state":{"formal_status_sha256":q.tr.sha(status_path),"run_inventory_sha256":q.tr.sha(inventory_path),"event":status.get("event"),"updated_at":status.get("updated_at"),"counts":counts},"hardware":{"nvidia_smi_returncode":smi.returncode,"nvidia_smi":smi.stdout,"failed_pci_function":"e1:00.0","failed_pci_evidence":pci.stdout[:4000],"action_boundary":"No GPU reset, driver reload, or host reboot performed."},"recovery_scope":{"eligible_terminal_runs":15,"one_time_only":True,"completed_runs_skipped_after_summary_identity_and_best_latest_commit_hash_verification":9},"rows":rows,"failures":failures,"test_consumed":False}
 q.tr.atomic_json(a.output,result);print(json.dumps(result,indent=2));assert result["status"]=="pass"
if __name__=="__main__":main()
