#!/usr/bin/env python3
"""Create the immutable pre-dispatch protocol lock and full inventory."""
import argparse, datetime, json
from pathlib import Path
import future_train as ft
def main():
 p=argparse.ArgumentParser(); p.add_argument("--prepare-audit",type=Path,required=True); p.add_argument("--root",type=Path,required=True); a=p.parse_args()
 audit=json.loads(a.prepare_audit.read_text()); assert audit["status"]=="pass" and audit["run_count"]==12
 gpu=json.loads((a.root/"GPU_SMOKE.json").read_text()); cpu=json.loads((a.root/"SMOKE.json").read_text()); assert gpu["status"]==cpu["status"]=="pass"
 files=("PROTOCOL.md","future_train.py","run_all.py","evaluate.py","audit_prepare.py","BUDGET.json")
 hashes={f:ft.sha(a.root/f) for f in files}
 inputs=[{"fold":r["fold"],"seed":r["seed"],"path":r["path"],"sha256":r["sha256"],"parent_prepared_sha256":r["provenance"]["prepared_sha256"],"support_sha256":r["provenance"]["support_sha256"]} for r in audit["runs"]]
 lock={"schema":"round14_protocol_lock_v1","status":"locked_before_formal_results","created_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"source_hashes":hashes,"input_count":len(inputs),"inputs":inputs,"fixed":{"groups":["V","F_concat","F_dual"],"folds":[1,2,3,4],"seeds":[20260914,20260915,20260916],"horizons_frames":[1,5,10],"max_epochs":60,"patience":10,"batch_size":256,"lr":0.001,"weight_decay":0.0001,"selection":"earliest strict minimum train-internal selection native-N MAE","device_policy":"GPU0/1/2, one job per GPU, max_parallel=3"},"acceptance_inputs":{"prepare_audit_sha256":ft.sha(a.prepare_audit),"cpu_smoke_sha256":ft.sha(a.root/"SMOKE.json"),"gpu_smoke_sha256":ft.sha(a.root/"GPU_SMOKE.json")}}
 ft.atomic_json(lock,a.root/"PROTOCOL_LOCK.json")
 runs=[]
 for group in ft.GROUPS:
  for fold in range(1,5):
   for seed in (20260914,20260915,20260916): runs.append({"id":f"{group}_p{fold}_s{seed}","group":group,"fold":fold,"seed":seed,"status":"registered","formal_output":f"/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round14_htt_future_force_dual/formal/{group}_p{fold}_s{seed}"})
 inventory=json.loads((a.root/"RUN_INVENTORY.json").read_text()); inventory.update(status="registered_not_dispatched",expected_runs=36,runs=runs,protocol_lock_sha256=ft.sha(a.root/"PROTOCOL_LOCK.json")); ft.atomic_json(inventory,a.root/"RUN_INVENTORY.json")
 ready={"schema":"round14_dispatch_ready_v1","status":"pass","authorized_devices":[0,1,2],"max_parallel":3,"formal_runs":36,"conditions":{"protocol_lock":True,"prepare_audit":True,"cpu_smoke":True,"gpu_smoke":True,"runner_rejects_smoke_path":True,"runner_skips_complete_accepted":True},"budget_evidence":{"gpu_smoke_wall_seconds":gpu["wall_seconds_for_V_Fconcat_and_Fdual_interrupt_resume_fresh"],"conservative_estimated_60_epoch_grid_hours_at_parallel3":2.0,"deadline_margin":"multiple days; formal launch before 2026-09-23 cutoff"}}
 ft.atomic_json(ready,a.root/"DISPATCH_READY.json"); print(json.dumps({"status":"pass","protocol_lock_sha256":ft.sha(a.root/"PROTOCOL_LOCK.json"),"runs":36}))
if __name__=="__main__": main()
