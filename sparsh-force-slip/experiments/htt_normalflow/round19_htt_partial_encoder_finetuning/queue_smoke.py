#!/usr/bin/env python3
"""No-training smoke for the durable Round-19 queue coordinator."""
from __future__ import annotations
import datetime as dt,json,subprocess,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
import run_all as q

def main():
 checks={}
 with tempfile.TemporaryDirectory() as raw:
  root=Path(raw);lock=q.acquire_lock(root/"queue.lock")
  try:
   try:q.acquire_lock(root/"queue.lock")
   except RuntimeError:checks["single_instance_lock"]="pass"
   else:raise AssertionError("second queue lock unexpectedly acquired")
  finally:lock.close()
  again=q.acquire_lock(root/"queue.lock");again.close();checks["stale_lock_os_release"]="pass"
  before=q.STOP_DISPATCH-dt.timedelta(microseconds=1);at=q.STOP_DISPATCH
  assert q.dispatch_open(before) and not q.dispatch_open(at);checks["dispatch_deadline_boundary"]="pass"
  out=root/"formal/V2/p1_s20260914";out.mkdir(parents=True)
  run={"run":"V2/p1_s20260914","group":"V2","fold":1,"seed":20260914,"output":str(out),"data_sha256":"d","prefix_index_sha256":"p","source_sha256":"s","visual_checkpoint_sha256":"v","source_sha256_code":"c","protocol_sha256":"r","status":"running","pid":999999}
  identity={k:run[k] for k in ("group","fold","seed","data_sha256","prefix_index_sha256","source_sha256","visual_checkpoint_sha256","source_sha256_code","protocol_sha256")};identity.update(formal=True,max_epochs=q.tr.MAX_EPOCHS,patience=q.tr.PATIENCE)
  (out/"summary.json").write_text(json.dumps({"status":"complete","identity":identity}))
  commit=out/"commits/epoch-0000.pth";commit.parent.mkdir();commit.write_bytes(b"checkpoint")
  digest=q.tr.sha(commit);(out/"COMMIT.json").write_text(json.dumps({"best":{"path":"commits/epoch-0000.pth","sha256":digest},"latest":{"path":"commits/epoch-0000.pth","sha256":digest}}))
  assert q.accepted_summary(run)==out/"summary.json";checks["complete_identity_hash_acceptance"]="pass"
  bad=json.loads((out/"summary.json").read_text());bad["identity"]["data_sha256"]="wrong";(out/"summary.json").write_text(json.dumps(bad));assert q.accepted_summary(run) is None;checks["bad_hash_rejected"]="pass"
  stale={**run,"output":str(root/"stale"),"status":"running","pid":999999};pending,live=q.reconcile([stale]);assert len(pending)==1 and not live and pending[0]["status"]=="registered_not_dispatched";checks["stale_pid_recovered"]="pass"
  live_run={**stale,"pid":None};cmd=[sys.executable,"-c","import time;time.sleep(10)","train.py",live_run["output"],live_run["group"],str(live_run["fold"]),str(live_run["seed"]),"--formal"]
  proc=subprocess.Popen(cmd)
  try:live_run["pid"]=proc.pid;assert q.pid_matches(live_run);checks["live_pid_identity_monitored"]="pass"
  finally:proc.terminate();proc.wait()
  oom=root/"oom.log";oom.write_text("RuntimeError: CUDA out of memory")
  assert q.classify_failure({"log":str(oom)})[0]=="oom";checks["oom_classification"]="pass"
  hardware=root/"hardware.log";hardware.write_text("RuntimeError: CUDA unknown error")
  assert q.classify_failure({"log":str(hardware)})[0]=="hardware_cuda_unavailable";checks["hardware_classification"]="pass"
  mixed=root/"mixed.log";mixed.write_text("RuntimeError: CUDA unknown error\n");offset=mixed.stat().st_size
  with mixed.open("a") as h:h.write("FloatingPointError: nonfinite loss\n")
  assert q.classify_failure({"log":str(mixed),"log_start_offset":offset})[0]=="terminal_scientific_or_identity_error";checks["old_hardware_new_numeric_is_scientific"]="pass"
  offset=mixed.stat().st_size
  with mixed.open("a") as h:h.write("new attempt exited without CUDA or numerical error\n")
  assert q.classify_failure({"log":str(mixed),"log_start_offset":offset})[0]=="recoverable_process_error";checks["old_error_excluded_from_new_attempt"]="pass"
  terminal={**stale,"status":"failed_terminal","attempts":2,"recovery_history":[{"event":"attempt_failed","log_tail":"CUDA error: unknown error"},{"event":"attempt_failed","log_tail":"CUDA unknown error"}]}
  pending,live=q.reconcile([terminal],True);assert len(pending)==1 and not live and terminal["hardware_terminal_requeued"] is True
  terminal["status"]="failed_terminal";pending2,live2=q.reconcile([terminal],True);assert not pending2 and not live2;checks["verified_hardware_terminal_requeued_exactly_once"]="pass"
  contaminated={**terminal,"hardware_terminal_requeued":False,"recovery_history":[{"event":"attempt_failed","log_tail":"CUDA unknown error; FloatingPointError: nonfinite loss"}]}
  assert not q.verified_hardware_terminal(contaminated);checks["scientific_failure_excludes_hardware_requeue"]="pass"
  state_run={**stale,"status":"registered_for_recovery"};args=SimpleNamespace(local=root);status=q.save_state(args,[state_run],2,"smoke_change");assert status["event"]=="smoke_change" and (root/"FORMAL_STATUS.json").exists() and (root/"RUN_INVENTORY.json").exists();checks["per_change_atomic_state"]="pass"
 result={"schema":"round19_queue_smoke_v3","status":"pass","checks":checks,"stop_dispatch":q.STOP_DISPATCH.isoformat(),"final_deadline":q.FINAL_DEADLINE.isoformat(),"run_all_sha256":q.tr.sha(Path(q.__file__))}
 (Path(__file__).resolve().parent/"QUEUE_SMOKE_V3.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps(result,indent=2))
if __name__=="__main__":main()
