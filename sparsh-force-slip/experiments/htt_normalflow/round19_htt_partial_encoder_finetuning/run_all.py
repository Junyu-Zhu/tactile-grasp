#!/usr/bin/env python3
"""Recoverable Round-19 formal queue with durable per-run state."""
from __future__ import annotations
import argparse,datetime as dt,fcntl,json,os,subprocess,time
from pathlib import Path
import train as tr

STOP_DISPATCH=dt.datetime.fromisoformat("2026-09-23T01:55:53+08:00")
FINAL_DEADLINE=dt.datetime.fromisoformat("2026-09-25T01:55:53+08:00")
NUMERIC_TERMINAL=("identity mismatch","hash mismatch","nonfinite","floatingpointerror","test forbidden")
OOM_MARKERS=("cuda out of memory","cublas_status_alloc_failed","cudaerrormemoryallocation")
HARDWARE_MARKERS=("cuda error: unknown error","cuda unknown error","no cuda gpus are available","cuda driver initialization failed")
def now():return dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))
def load(path):return json.loads(Path(path).read_text())
def dispatch_open(at=None):return (at or now())<STOP_DISPATCH

def visible_gpus(python,requested):
 command=["nvidia-smi","--query-gpu=index,uuid,name,pci.bus_id","--format=csv,noheader,nounits"]
 found={}
 for line in subprocess.check_output(command,text=True,timeout=15).splitlines():
  index,uuid,name,bus=(x.strip() for x in line.split(",",3));found[index]={"index":int(index),"uuid":uuid,"name":name,"pci_bus_id":bus}
 result=[]
 for token in requested:
  if token not in found:continue
  slot=found[token];env={**os.environ,"CUDA_VISIBLE_DEVICES":slot["uuid"],"XFORMERS_DISABLED":"1"}
  smoke="import torch;x=torch.arange(4096,device='cuda',dtype=torch.float32);y=(x*x).sum();torch.cuda.synchronize();assert y.isfinite().item();print(torch.cuda.get_device_name(0))"
  try:probe=subprocess.run([python,"-c",smoke],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=45);returncode=probe.returncode;output=probe.stdout
  except subprocess.TimeoutExpired as error:returncode=124;output=(error.stdout or "")+(error.stderr or "")+"\nCUDA smoke timed out"
  slot={**slot,"probe_returncode":returncode,"probe_output":output[-2000:],"checked_at":now().isoformat()}
  if returncode==0:result.append(slot)
 return result

def acquire_lock(path):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);handle=path.open("a+")
 try:fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:
  handle.seek(0);owner=handle.read().strip();handle.close();raise RuntimeError(f"formal queue already owned: {owner}")
 handle.seek(0);handle.truncate();handle.write(json.dumps({"pid":os.getpid(),"acquired_at":now().isoformat()}));handle.flush();os.fsync(handle.fileno());return handle

def expected_runs(a):
 prefix=a.large/"prefix_cache/PREFIX_INDEX.json"
 common={"prefix_index":str(prefix),"prefix_index_sha256":tr.sha(prefix),"source":str(a.source),"source_sha256":tr.sha(a.source),"source_sha256_code":tr.sha(a.local/"train.py"),"protocol_sha256":tr.sha(a.local/"PROTOCOL.md")};rows=[]
 for fold in range(1,5):
  for seed in tr.SEEDS:
   data=a.r17/f"p{fold}_s{seed}"/"prepared.pt";visual=a.r3/f"fold_p{fold}"/f"seed_{seed}"/"best.pth"
   for group in tr.GROUPS:
    rid=f"{group}/p{fold}_s{seed}";rows.append({**common,"run":rid,"group":group,"fold":fold,"seed":seed,"data":str(data),"data_sha256":tr.sha(data),"visual_checkpoint":str(visual),"visual_checkpoint_sha256":tr.sha(visual),"output":str(a.large/"formal"/group/f"p{fold}_s{seed}"),"status":"registered_not_dispatched","attempts":0,"pid":None})
 return rows

def accepted_summary(run):
 path=Path(run["output"])/"summary.json"
 if not path.exists():return None
 try:summary=load(path)
 except (OSError,json.JSONDecodeError):return None
 identity=summary.get("identity",{});expected={"group":run["group"],"fold":run["fold"],"seed":run["seed"],"data_sha256":run["data_sha256"],"prefix_index_sha256":run["prefix_index_sha256"],"source_sha256":run["source_sha256"],"visual_checkpoint_sha256":run["visual_checkpoint_sha256"],"source_sha256_code":run["source_sha256_code"],"protocol_sha256":run["protocol_sha256"],"formal":True,"max_epochs":tr.MAX_EPOCHS,"patience":tr.PATIENCE}
 if summary.get("status")!="complete" or not all(identity.get(k)==v for k,v in expected.items()):return None
 try:
  manifest=load(Path(run["output"])/"COMMIT.json")
  for key in ("best","latest"):
   item=manifest[key];checkpoint=Path(run["output"])/item["path"]
   if tr.sha(checkpoint)!=item["sha256"]:return None
 except (OSError,KeyError,TypeError,ValueError,json.JSONDecodeError):return None
 return path

def pid_matches(run):
 pid=run.get("pid")
 if not isinstance(pid,int) or pid<=1:return False
 try:command=(Path("/proc")/str(pid)/"cmdline").read_bytes().replace(b"\0",b" ").decode(errors="replace")
 except OSError:return False
 return all(token in command for token in ("train.py",run["output"],run["group"],str(run["fold"]),str(run["seed"]),"--formal"))

def log_tail(path,limit=20000,start_offset=0):
 path=Path(path)
 if not path.exists():return ""
 with path.open("rb") as handle:
  handle.seek(0,os.SEEK_END);size=handle.tell()
  if not isinstance(start_offset,int) or start_offset<0 or start_offset>size:return ""
  handle.seek(max(start_offset,size-limit));return handle.read().decode(errors="replace")
def classify_failure(run):
 text=log_tail(run.get("log",""),start_offset=run.get("log_start_offset",0));lower=text.lower()
 if any(x in lower for x in NUMERIC_TERMINAL):return "terminal_scientific_or_identity_error",text[-2000:]
 if any(x in lower for x in OOM_MARKERS):return "oom",text[-2000:]
 if any(x in lower for x in HARDWARE_MARKERS):return "hardware_cuda_unavailable",text[-2000:]
 return "recoverable_process_error",text[-2000:]

def verified_hardware_terminal(run):
 failures=[x for x in run.get("recovery_history",[]) if x.get("event")=="attempt_failed"]
 return bool(failures) and all(any(marker in x.get("log_tail","").lower() for marker in HARDWARE_MARKERS) and not any(marker in x.get("log_tail","").lower() for marker in NUMERIC_TERMINAL) for x in failures)

def reconcile(runs,recover_verified_hardware_terminal=False):
 pending=[];monitored=[]
 for run in runs:
  summary=accepted_summary(run)
  if summary:run.update(status="complete",summary_sha256=tr.sha(summary),pid=None);continue
  if run.get("status") in ("running","running_reconciled") and pid_matches(run):run["status"]="running_reconciled";monitored.append(run);continue
  if run.get("status")=="failed_terminal":
   if recover_verified_hardware_terminal and not run.get("hardware_terminal_requeued") and verified_hardware_terminal(run):
    run["hardware_terminal_requeued"]=True;run.setdefault("recovery_history",[]).append({"at":now().isoformat(),"event":"verified_hardware_terminal_requeued_once","preserved_terminal_reason":run.get("terminal_reason"),"preserved_attempts":run.get("attempts")})
    run["status"]="registered_for_recovery" if Path(run["output"],"COMMIT.json").exists() else "registered_not_dispatched";run["pid"]=None;pending.append(run)
   continue
  if run.get("status")=="blocked_after_cutoff":continue
  if run.get("status") in ("running","running_reconciled"):run.setdefault("recovery_history",[]).append({"at":now().isoformat(),"event":"stale_running_pid_or_identity","old_pid":run.get("pid")})
  run["pid"]=None;run["status"]="registered_for_recovery" if Path(run["output"],"COMMIT.json").exists() else "registered_not_dispatched";pending.append(run)
 return pending,monitored

def save_state(a,runs,active_limit,event,gpu_health=None):
 counts={}
 for run in runs:counts[run["status"]]=counts.get(run["status"],0)+1
 status={"schema":"round19_formal_status_v3","status":"complete" if counts=={"complete":24} else "incomplete","event":event,"updated_at":now().isoformat(),"stop_dispatch":STOP_DISPATCH.isoformat(),"final_deadline":FINAL_DEADLINE.isoformat(),"new_dispatch_closed":now()>=STOP_DISPATCH,"active_parallel_limit":active_limit,"gpu_health":gpu_health or [],"status_counts":counts,"test_consumed":False,"runs":runs}
 tr.atomic_json(a.local/"RUN_INVENTORY.json",{"schema":"round19_run_inventory_v2","runs":runs});tr.atomic_json(a.local/"FORMAL_STATUS.json",status);return status

def command_for(a,run):
 return [a.python,str(a.local/"train.py"),"--data",run["data"],"--prefix-index",run["prefix_index"],"--source",run["source"],"--visual-checkpoint",run["visual_checkpoint"],"--output",run["output"],"--group",run["group"],"--fold",str(run["fold"]),"--seed",str(run["seed"]),"--device","cuda:0","--formal"]

def main():
 p=argparse.ArgumentParser();p.add_argument("--local",type=Path,default=Path(__file__).resolve().parent);p.add_argument("--large",type=Path,required=True);p.add_argument("--r17",type=Path,required=True);p.add_argument("--r3",type=Path,required=True);p.add_argument("--source",type=Path,required=True);p.add_argument("--python",default="/home/zjy/miniconda3/envs/sparsh/bin/python");p.add_argument("--gpus",default="0,1,2");p.add_argument("--max-parallel",type=int,default=3);p.add_argument("--reconcile-only",action="store_true");p.add_argument("--recover-verified-hardware-terminal",action="store_true");p.add_argument("--execute-formal",action="store_true");a=p.parse_args()
 if not a.execute_formal:raise SystemExit("formal queue requires root internal release and --execute-formal")
 requested_gpus=[x.strip() for x in a.gpus.split(",") if x.strip()]
 if not 1<=a.max_parallel<=len(requested_gpus)<=3:raise ValueError("invalid Round-19 GPU/parallel configuration")
 queue_lock=acquire_lock(a.local/"FORMAL_QUEUE.lock")
 try:
  inventory_path=a.local/"RUN_INVENTORY.json";expected=expected_runs(a)
  if inventory_path.exists():
   runs=load(inventory_path)["runs"]
   keys=lambda rows:[(r["run"],r["data_sha256"],r["prefix_index_sha256"],r["source_sha256"],r["visual_checkpoint_sha256"],r["source_sha256_code"],r["protocol_sha256"]) for r in rows]
   if keys(runs)!=keys(expected):raise ValueError("existing inventory identity mismatch")
  else:runs=expected
  gpu_health=visible_gpus(a.python,requested_gpus)
  if not gpu_health:raise RuntimeError("no requested GPU passed nvidia-smi UUID discovery and CUDA compute smoke")
  active_limit=min(a.max_parallel,len(gpu_health));pending,monitored=reconcile(runs,a.recover_verified_hardware_terminal);save_state(a,runs,active_limit,"reconciled",gpu_health)
  if a.reconcile_only:print(json.dumps({"pending":len(pending),"live_reconciled":len(monitored)},indent=2));return
  if not dispatch_open() and pending:
   for run in pending:run["status"]="blocked_after_cutoff"
   save_state(a,runs,active_limit,"dispatch_cutoff_applied");raise ValueError("new formal dispatch cutoff reached")
  by_uuid={x["uuid"]:x for x in gpu_health};active=[{"process":None,"run":run,"handle":None,"gpu":by_uuid[run["gpu_uuid"]]} for run in monitored if run.get("gpu_uuid") in by_uuid];logs=a.local/"logs";logs.mkdir(exist_ok=True);quarantined=set()
  while pending or active:
   if now()>=FINAL_DEADLINE:save_state(a,runs,active_limit,"final_deadline_reached");raise RuntimeError("final deadline reached")
   while pending and len(active)<active_limit and dispatch_open():
    used={x["gpu"]["uuid"] for x in active};candidates=[x for x in visible_gpus(a.python,requested_gpus) if x["uuid"] not in quarantined];gpu=next((x for x in candidates if x["uuid"] not in used),None)
    if gpu is None:break
    run=pending.pop(0);attempt=int(run.get("attempts",0))+1;command=command_for(a,run);log_path=logs/(run["run"].replace("/","_")+".log");handle=log_path.open("a");handle.seek(0,os.SEEK_END);log_start_offset=handle.tell();handle.write(json.dumps({"dispatch":run["run"],"attempt":attempt,"gpu_index":gpu["index"],"gpu_uuid":gpu["uuid"],"at":now().isoformat()})+"\n");handle.flush();env={**os.environ,"CUDA_VISIBLE_DEVICES":gpu["uuid"],"XFORMERS_DISABLED":"1"};process=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,env=env,text=True);run.update(status="running",gpu=gpu["index"],gpu_uuid=gpu["uuid"],gpu_pci_bus_id=gpu["pci_bus_id"],pid=process.pid,started_at=now().isoformat(),command=command,attempts=attempt,log=str(log_path),log_start_offset=log_start_offset);active.append({"process":process,"run":run,"handle":handle,"gpu":gpu});save_state(a,runs,active_limit,f"dispatched:{run['run']}:attempt{attempt}",gpu_health)
   if not active:
    if pending and not dispatch_open():
     for run in pending:run["status"]="blocked_after_cutoff"
     save_state(a,runs,active_limit,"dispatch_cutoff_applied");break
    time.sleep(1);continue
   time.sleep(2);survivors=[]
   for item in active:
    process,run,handle=item["process"],item["run"],item["handle"];alive=process.poll() is None if process is not None else pid_matches(run)
    if alive:survivors.append(item);continue
    returncode=process.returncode if process is not None else None
    if handle is not None:handle.close()
    run.update(returncode=returncode,finished_at=now().isoformat(),pid=None);summary=accepted_summary(run)
    if summary:run.update(status="complete",summary_sha256=tr.sha(summary));save_state(a,runs,active_limit,f"completed:{run['run']}");continue
    failure,excerpt=classify_failure(run);run.setdefault("recovery_history",[]).append({"at":now().isoformat(),"event":"attempt_failed","attempt":run["attempts"],"class":failure,"returncode":returncode,"log_tail":excerpt})
    if failure=="oom":
     old_limit=active_limit;active_limit=max(1,active_limit-1);retry=run["attempts"]<3 and not (old_limit==1 and run["attempts"]>=2);run["queue_parallel_limit_after_oom"]=active_limit;run["status"]="registered_for_recovery" if retry else "failed_terminal"
     if not retry:run["terminal_reason"]="OOM persists with one job; any batch change requires a registered group-wide revision"
    elif failure=="terminal_scientific_or_identity_error":retry=False;run.update(status="failed_terminal",terminal_reason=failure)
    elif failure=="hardware_cuda_unavailable":
     quarantined.add(item["gpu"]["uuid"]);used_retries=int(run.get("runtime_hardware_requeues",0));retry=not run.get("hardware_terminal_requeued") and used_retries<1 and bool([x for x in gpu_health if x["uuid"] not in quarantined]);run["runtime_hardware_requeues"]=used_retries+1;run["status"]="registered_for_recovery" if retry else "failed_terminal"
     if not retry:run["terminal_reason"]="CUDA hardware unavailable and no healthy alternate GPU remained"
    else:
     retry=run["attempts"]<2;run["status"]="registered_for_recovery" if retry else "failed_terminal"
     if not retry:run["terminal_reason"]="recoverable process error repeated twice"
    if retry and dispatch_open():pending.insert(0,run)
    elif retry:run["status"]="blocked_after_cutoff"
    save_state(a,runs,active_limit,f"failed:{run['run']}:{failure}",gpu_health)
   active=survivors
  status=save_state(a,runs,active_limit,"queue_finished");print(json.dumps({k:status[k] for k in ("status","status_counts","active_parallel_limit")},indent=2))
  if status["status"]!="complete":raise SystemExit(1)
 finally:queue_lock.close()
if __name__=="__main__":main()
