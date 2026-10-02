#!/usr/bin/env python3
"""Recoverable bounded Round-18 formal queue with stale-process reconciliation."""
from __future__ import annotations
import argparse, datetime as dt, json, os, subprocess, sys, time
from pathlib import Path
import train as r18

STOP_DISPATCH=dt.datetime.fromisoformat("2026-09-23T01:55:53+08:00")
FINAL_DEADLINE=dt.datetime.fromisoformat("2026-09-25T01:55:53+08:00")
NUMERIC_TERMINAL=("nonfinite","missing/nonfinite gradient","identity mismatch","role leakage","upstream hash mismatch","bad selection")
OOM_MARKERS=("CUDA out of memory","CUBLAS_STATUS_ALLOC_FAILED","cudaErrorMemoryAllocation")
def now():return dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))
def load(path):return json.loads(Path(path).read_text())

def summary_complete(run):
 path=Path(run["output"])/"summary.json"
 if not path.exists():return False
 try: summary=load(path)
 except (OSError,json.JSONDecodeError):return False
 identity=summary.get("identity",{})
 return summary.get("status")=="complete" and all(identity.get(k)==run[k] for k in ("group","fold","seed")) and identity.get("input_sha256")==run["input_sha256"]

def pid_matches(run):
 pid=run.get("pid")
 if not isinstance(pid,int) or pid<=1:return False
 try: command=(Path("/proc")/str(pid)/"cmdline").read_bytes().replace(b"\0",b" ").decode(errors="replace")
 except OSError:return False
 required=("train.py",str(run["output"]),run["group"],str(run["fold"]),str(run["seed"]),"--execute-formal")
 return all(token in command for token in required)

def log_tail(path,limit=20000):
 path=Path(path)
 if not path.exists():return ""
 with path.open("rb") as handle:
  handle.seek(0,os.SEEK_END);size=handle.tell();handle.seek(max(0,size-limit));return handle.read().decode(errors="replace")

def classify_failure(run):
 text=log_tail(run.get("log",""))
 if any(marker.lower() in text.lower() for marker in OOM_MARKERS):return "oom",text[-2000:]
 if any(marker.lower() in text.lower() for marker in NUMERIC_TERMINAL):return "terminal_scientific_or_identity_error",text[-2000:]
 return "recoverable_process_error",text[-2000:]

def reconcile(inventory):
 pending=[];monitored=[]
 for run in inventory["runs"]:
  if summary_complete(run):
   path=Path(run["output"])/"summary.json";run.update(status="complete",summary_sha256=r18.sha(path),pid=None);continue
  if run.get("status") in ("running","running_reconciled") and pid_matches(run):
   run["status"]="running_reconciled";monitored.append(run);continue
  if run.get("status") in ("failed_terminal","blocked_after_cutoff"):continue
  if run.get("status") in ("running","running_reconciled"):
   run.setdefault("recovery_history",[]).append({"at":now().isoformat(),"event":"stale_running_pid_or_identity","old_pid":run.get("pid")});run["pid"]=None
  run["status"]="registered_for_recovery" if Path(run["output"],"latest.pth").exists() else "registered_not_dispatched";pending.append(run)
 return pending,monitored

def main():
 p=argparse.ArgumentParser();p.add_argument("--local",type=Path,default=Path(__file__).resolve().parent);p.add_argument("--max-parallel",type=int,default=3);p.add_argument("--gpus",default="0,1,2");p.add_argument("--reconcile-only",action="store_true");a=p.parse_args()
 if not 1<=a.max_parallel<=3:raise ValueError("max three Round-18 jobs")
 control_path=a.local/"ROOT_CONTROL.json"
 if not control_path.exists():raise ValueError("formal dispatch blocked: ROOT_CONTROL.json absent")
 control=load(control_path)
 if control.get("formal_authorized") is not True or control.get("protocol_sha256")!=r18.sha(a.local/"PROTOCOL.md"):raise ValueError("formal dispatch blocked: root authorization/identity mismatch")
 smoke=load(a.local/"SMOKE_GPU1.json")
 if smoke.get("status")!="pass" or smoke.get("device")!="cuda:0" or smoke.get("CUDA_VISIBLE_DEVICES")!="1":raise ValueError("accepted GPU1 smoke missing")
 inventory_path=a.local/"RUN_INVENTORY.json";inventory=load(inventory_path);logs=a.local/"logs";logs.mkdir(exist_ok=True)
 pending,reconciled=reconcile(inventory);r18.atomic_json(inventory_path,inventory)
 if a.reconcile_only:
  print(json.dumps({"pending":len(pending),"live_reconciled":len(reconciled),"complete":sum(r["status"]=="complete" for r in inventory["runs"])},indent=2));return
 if now()>=STOP_DISPATCH and pending:
  for run in pending:run["status"]="blocked_after_cutoff"
  r18.atomic_json(inventory_path,inventory);raise ValueError("new formal dispatch cutoff reached")
 gpus=[int(v) for v in a.gpus.split(",")];active_limit=a.max_parallel
 active=[{"process":None,"run":run,"handle":None,"gpu":int(run["gpu"]),"pid":int(run["pid"])} for run in reconciled]
 while pending or active:
  if now()>=FINAL_DEADLINE:raise RuntimeError("final deadline reached")
  while pending and len(active)<active_limit and now()<STOP_DISPATCH:
   used={item["gpu"] for item in active};gpu=next((value for value in gpus if value not in used),None)
   if gpu is None:break
   run=pending.pop(0);attempts=int(run.get("attempts",0))+1
   command=[sys.executable,str(a.local/"train.py"),"--data",run["input_path"],"--output",run["output"],"--group",run["group"],"--fold",str(run["fold"]),"--seed",str(run["seed"]),"--device","cuda:0","--execute-formal"]
   log_path=logs/(run["run_id"].replace("/","_")+".log");handle=log_path.open("a");env={**os.environ,"CUDA_VISIBLE_DEVICES":str(gpu),"XFORMERS_DISABLED":"1"};process=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,env=env,text=True)
   run.update(status="running",gpu=gpu,pid=process.pid,started_at=now().isoformat(),command=command,attempts=attempts,log=str(log_path));active.append({"process":process,"run":run,"handle":handle,"gpu":gpu,"pid":process.pid});r18.atomic_json(inventory_path,inventory)
  if not active:
   if pending and now()>=STOP_DISPATCH:
    for run in pending:run["status"]="blocked_after_cutoff"
    r18.atomic_json(inventory_path,inventory);break
   time.sleep(1);continue
  time.sleep(2);survivors=[]
  for item in active:
   process,run,handle=item["process"],item["run"],item["handle"];alive=process.poll() is None if process is not None else pid_matches(run)
   if alive:survivors.append(item);continue
   returncode=process.returncode if process is not None else None
   if handle is not None:handle.close()
   run.update(returncode=returncode,finished_at=now().isoformat(),pid=None)
   if summary_complete(run):
    path=Path(run["output"])/"summary.json";run.update(status="complete",summary_sha256=r18.sha(path))
   else:
    failure,excerpt=classify_failure(run);run.setdefault("recovery_history",[]).append({"at":now().isoformat(),"event":"attempt_failed","attempt":run.get("attempts",0),"class":failure,"returncode":returncode,"log_tail":excerpt})
    if failure=="oom":
     active_limit=max(1,active_limit-1);run["queue_parallel_limit_after_oom"]=active_limit;retry=run.get("attempts",0)<3 and not (active_limit==1 and run.get("attempts",0)>=2);run["status"]="registered_for_recovery" if retry else "failed_terminal"
     if not retry:run["terminal_reason"]="OOM persists at one Round-18 job; requires preregistered group-wide batch revision"
    elif failure=="terminal_scientific_or_identity_error":retry=False;run["status"]="failed_terminal";run["terminal_reason"]=failure
    else:
     retry=run.get("attempts",0)<2;run["status"]="registered_for_recovery" if retry else "failed_terminal"
     if not retry:run["terminal_reason"]="recoverable process error repeated twice"
    if retry and now()<STOP_DISPATCH:pending.append(run)
    elif retry:run["status"]="blocked_after_cutoff"
   r18.atomic_json(inventory_path,inventory)
  active=survivors
 counts={}
 for run in inventory["runs"]:counts[run["status"]]=counts.get(run["status"],0)+1
 status={"schema":"round18_formal_status_v1","status":"complete" if counts=={"complete":36} else "incomplete","status_counts":counts,"active_parallel_limit_final":active_limit,"new_dispatch_closed":now()>=STOP_DISPATCH,"updated_at":now().isoformat(),"test_consumed":False};r18.atomic_json(a.local/"FORMAL_STATUS.json",status);print(json.dumps(status,indent=2))
 if status["status"]!="complete":raise SystemExit(1)
if __name__=="__main__":main()
