#!/usr/bin/env python3
"""Root-gated persistent E3 queue with reusable, smoke-tested queue mechanics."""
from __future__ import annotations
import argparse,datetime as dt,fcntl,hashlib,json,os,subprocess,tempfile,time
from pathlib import Path
GROUPS=("A","B","C","D");SEEDS=(20260914,20260915,20260916)
STOP=dt.datetime.fromisoformat("2026-09-28T01:55:53+08:00")
MAX_ATTEMPTS=2
TERMINAL_MARKERS=("identity mismatch","hash mismatch","floatingpointerror","test forbidden","formal e3 blocked")
OOM_MARKERS=("cuda out of memory","cublas_status_alloc_failed","cudaerrormemoryallocation")
HARDWARE_MARKERS=("cuda error: unknown error","no cuda gpus are available","cuda driver initialization failed","simulated_gpu_fault")
def now():return dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def atomic(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(dir=p.parent);os.close(fd)
 try:Path(t).write_text(json.dumps(x,indent=2,sort_keys=True)+"\n");os.replace(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)
def acquire(p):
 h=Path(p).open("a+")
 try:fcntl.flock(h.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:h.close();raise RuntimeError("queue lock already owned")
 h.seek(0);h.truncate();h.write(json.dumps({"pid":os.getpid(),"at":now().isoformat()}));h.flush();os.fsync(h.fileno());return h
def pid_matches(pid,tokens):
 if not isinstance(pid,int) or pid<=1:return False
 try:cmd=(Path("/proc")/str(pid)/"cmdline").read_bytes().replace(b"\0",b" ").decode(errors="replace")
 except OSError:return False
 return all(str(x) in cmd for x in tokens)
def accepted_e3(r):
 out=Path(r["output"]);p=out/"summary.json"
 try:
  d=json.loads(p.read_text());i=d["identity"];c=json.loads((out/"COMMIT.json").read_text())
  committed=all(sha(out/c[k]["path"])==c[k]["sha256"] for k in ("latest","best"))
 except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError):return False
 expected=r.get("expected_identity",{});identity_ok=all(i.get(k)==v for k,v in expected.items())
 return d.get("status")=="complete" and committed and identity_ok and (d.get("group"),d.get("fold"),d.get("seed"),i.get("formal"))==(r["group"],r["fold"],r["seed"],True)
def log_tail(path,n=8000):
 try:
  with Path(path).open("rb") as f:f.seek(0,2);size=f.tell();f.seek(max(0,size-n));return f.read().decode(errors="replace")
 except OSError:return ""
def reconcile(state,accepted,pid_check,max_attempts=MAX_ATTEMPTS,launch_discover=None):
 for r in state["runs"]:
  if accepted(r):r.update(status="complete",pid=None,accepted_at=now().isoformat());continue
  if r["status"]=="running" and not pid_check(r):
   tail=log_tail(r.get("log",""));lower=tail.lower();terminal=any(x in lower for x in TERMINAL_MARKERS);oom=any(x in lower for x in OOM_MARKERS);hardware=any(x in lower for x in HARDWARE_MARKERS)
   if oom:state["concurrency_limit"]=max(1,int(state.get("concurrency_limit",1))-1);state.setdefault("events",[]).append({"at":now().isoformat(),"event":"oom_reduce_concurrency","run":r["run"],"new_limit":state["concurrency_limit"]})
   if hardware and r.get("device"):state.setdefault("quarantined_devices",[]).append(r["device"]);state["quarantined_devices"]=sorted(set(state["quarantined_devices"]));state.setdefault("events",[]).append({"at":now().isoformat(),"event":"hardware_quarantine","run":r["run"],"device":r["device"]})
   retry=(not terminal and r["attempts"]<max_attempts);failure="oom" if oom else "hardware" if hardware else "terminal_identity_or_numeric" if terminal else "process_error"
   r.update(status="retryable" if retry else "failed_terminal",pid=None,finished_at=now().isoformat(),failure_class=failure,last_log_tail=tail[-2000:])
  elif r["status"]=="launching":
   found=(launch_discover(r) if launch_discover else discover_pids(("train_e3.py",r["output"],"--formal")))
   if len(found)==1:r.update(status="running",pid=found[0],adopted_after_launch_crash=True)
   elif len(found)>1:r.update(status="failed_terminal",pid=None,failure_class="duplicate_processes",discovered_pids=found)
   else:r.update(status="retryable" if r["attempts"]<max_attempts else "failed_terminal",pid=None,failure_class="launch_intent_without_process")
def discover_pids(tokens):
 out=[]
 for p in Path("/proc").iterdir():
  if not p.name.isdigit():continue
  try:cmd=(p/"cmdline").read_bytes().replace(b"\0",b" ").decode(errors="replace")
  except OSError:continue
  if all(str(x) in cmd for x in tokens):out.append(int(p.name))
 return out
def run_queue(state_path,lock_path,devices,stop,accepted,pid_check,command_builder,env_builder=lambda d:{},poll_seconds=10,max_attempts=MAX_ATTEMPTS,launch_discover=None,owner=None):
 owner=owner or acquire(lock_path);state=json.loads(Path(state_path).read_text())
 while True:
  reconcile(state,accepted,pid_check,max_attempts,launch_discover)
  occupied={r.get("device") for r in state["runs"] if r["status"]=="running" and pid_check(r)};quarantined=set(state.get("quarantined_devices",[]));limit=int(state.get("concurrency_limit",len(devices)));active=sum(r["status"]=="running" and pid_check(r) for r in state["runs"]);free=[d for d in devices if d not in occupied and d not in quarantined][:max(0,limit-active)];pending=[r for r in state["runs"] if r["status"] in ("registered_not_dispatched","retryable")]
  if pending and active==0 and not [d for d in devices if d not in quarantined]:
   blocked_at=now().isoformat()
   for r in pending:r.update(status="blocked_no_healthy_devices",pid=None,blocked_at=blocked_at)
   state.setdefault("events",[]).append({"at":blocked_at,"event":"blocked_no_healthy_devices","runs":len(pending)});state["updated_at"]=blocked_at;atomic(state_path,state);break
  for d,r in zip(free,pending):
   dispatch_checked_at=now()
   if dispatch_checked_at>=stop:r["dispatch_blocked_at_cutoff"]=dispatch_checked_at.isoformat();continue
   cmd,log=command_builder(r,d);Path(log).parent.mkdir(parents=True,exist_ok=True);launch_token=f"{os.getpid()}-{time.time_ns()}";r.update(status="launching",attempts=r["attempts"]+1,pid=None,device=d,log=str(log),started_at=dispatch_checked_at.isoformat(),command=[str(x) for x in cmd],launch_token=launch_token);state["updated_at"]=now().isoformat();atomic(state_path,state);fh=Path(log).open("ab");proc=subprocess.Popen(cmd,stdout=fh,stderr=subprocess.STDOUT,env={**os.environ,**env_builder(d)},start_new_session=True);fh.close();r.update(status="running",pid=proc.pid)
  state["updated_at"]=now().isoformat();atomic(state_path,state);statuses={r["status"] for r in state["runs"]}
  if statuses<={"complete","failed_terminal"}:break
  if now()>=stop and "running" not in statuses:break
  time.sleep(poll_seconds)
 return state
def validate_existing_state(existing,inventory,inventory_hash):
 expected_runs=[(r["run"],r["output"]) for r in inventory["runs"]];actual_runs=[(r.get("run"),r.get("output")) for r in existing.get("runs",[])]
 return existing.get("schema")=="round22_e3_queue_v3" and existing.get("inventory_sha256")==inventory_hash and len(actual_runs)==48 and len(actual_runs)==len(set(actual_runs)) and set(actual_runs)==set(expected_runs)
def gpu_inventory(tokens):
 rows={}
 for line in subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,name","--format=csv,noheader"],text=True).splitlines():
  i,u,n=(x.strip() for x in line.split(",",2));rows[i]={"index":int(i),"uuid":u,"name":n}
 if any(x not in rows for x in tokens):raise ValueError("requested GPU missing")
 return [rows[x] for x in tokens]
def load_inventory(path,train_source=None):
 inv=json.loads(Path(path).read_text());runs=inv.get("runs",[])
 if inv.get("schema")!="round22_g2_run_inventory_v1" or len(runs)!=48:raise ValueError("invalid G2 run inventory schema/count")
 required=("run","group","fold","seed","data","data_sha256","data_schema","support_scope","prefix_index","prefix_index_sha256","source","source_sha256","visual_checkpoint","visual_checkpoint_sha256","protocol","protocol_sha256","train_source_sha256","output")
 issues=[];expected={(g,f,s) for f in range(1,5) for s in SEEDS for g in GROUPS};seen=[];outputs=[];hash_cache={}
 def checked_sha(p):
  p=Path(p);st=p.stat();key=(str(p.resolve()),st.st_size,st.st_mtime_ns)
  if key not in hash_cache:hash_cache[key]=sha(p)
  return hash_cache[key]
 actual_train_sha=checked_sha(train_source) if train_source is not None else None
 for r in runs:
  missing=[k for k in required if not r.get(k)]
  if missing:issues.append({"run":r.get("run"),"missing":missing});continue
  seen.append((r["group"],r["fold"],r["seed"]));outputs.append(r["output"])
  if r["run"]!=f"{r['group']}/p{r['fold']}_s{r['seed']}":issues.append({"run":r["run"],"error":"run_key_mismatch"});continue
  if train_source is not None and r["train_source_sha256"]!=actual_train_sha:issues.append({"run":r["run"],"error":"train_source_sha256_mismatch"});continue
  if r["support_scope"]!="round22_current_common_gt_exact":issues.append({"run":r["run"],"error":"wrong_support_scope"});continue
  for key,hashkey in (("data","data_sha256"),("prefix_index","prefix_index_sha256"),("source","source_sha256"),("visual_checkpoint","visual_checkpoint_sha256"),("protocol","protocol_sha256")):
   p=Path(r[key])
   if not p.is_file() or checked_sha(p)!=r[hashkey]:issues.append({"run":r["run"],"error":f"{key}_missing_or_hash_mismatch"});break
 if set(seen)!=expected or len(seen)!=len(set(seen)):issues.append({"error":"grid_not_unique_complete"})
 if len(outputs)!=len(set(outputs)):issues.append({"error":"duplicate_output_paths"})
 return inv,issues
def main():
 p=argparse.ArgumentParser();p.add_argument("--authorization",type=Path,required=True);p.add_argument("--run-inventory",type=Path,required=True);p.add_argument("--state",type=Path,required=True);p.add_argument("--lock",type=Path,required=True);p.add_argument("--python",required=True);p.add_argument("--repo",type=Path,required=True);p.add_argument("--run-root",type=Path,required=True);p.add_argument("--source-root",type=Path,required=True);p.add_argument("--gpu",nargs="+",default=["0","1","2"]);p.add_argument("--execute",action="store_true");a=p.parse_args();here=Path(__file__).resolve().parent;auth=json.loads(a.authorization.read_text());inventory,inventory_issues=load_inventory(a.run_inventory,here/"train_e3.py")
 if a.execute and auth.get("e3_formal_authorized") is not True:raise SystemExit("E3 formal dispatch not authorized by root")
 if a.execute and (auth.get("budget_pass") is not True or inventory_issues):raise SystemExit("E3 formal dispatch blocked by budget or run inventory")
 devices=gpu_inventory(a.gpu);owner=acquire(a.lock);inventory_hash=sha(a.run_inventory)
 if not a.state.exists():
  runs=[]
  for item in inventory["runs"]:
   expected={"group":item["group"],"fold":item["fold"],"seed":item["seed"],"data_sha256":item.get("data_sha256"),"prefix_index_sha256":item.get("prefix_index_sha256"),"source_sha256":item.get("source_sha256"),"visual_checkpoint_sha256":item.get("visual_checkpoint_sha256"),"protocol_sha256":item.get("protocol_sha256"),"source_sha256_code":item.get("train_source_sha256"),"authorization_sha256":sha(a.authorization),"formal":True};runs.append({**item,"expected_identity":expected,"status":"registered_not_dispatched","attempts":0,"pid":None})
  atomic(a.state,{"schema":"round22_e3_queue_v3","created_at":now().isoformat(),"stop_dispatch":STOP.isoformat(),"concurrency_limit":len(devices),"quarantined_devices":[],"events":[],"inventory":str(a.run_inventory),"inventory_sha256":inventory_hash,"runs":runs})
 else:
  existing=json.loads(a.state.read_text())
  if not validate_existing_state(existing,inventory,inventory_hash):raise SystemExit("existing queue state does not match exact v3 inventory")
 if not a.execute:print(json.dumps({"schema":"round22_g2_queue_machine_v2","status":"prepared_not_dispatched_inventory_blocked" if inventory_issues else "prepared_not_dispatched","g2_ready":not inventory_issues and auth.get("e3_formal_authorized") is True and auth.get("budget_pass") is True,"runs":48,"inventory_issues":inventory_issues,"gpus":devices,"g1_formal_authorized":auth.get("g1_formal_authorized") is True,"e3_formal_authorized":auth.get("e3_formal_authorized") is True,"budget_pass":auth.get("budget_pass") is True,"authorization":str(a.authorization),"authorization_sha256":sha(a.authorization),"python":a.python,"repo":str(a.repo),"run_root":str(a.run_root),"source_root":str(a.source_root),"stop_dispatch":STOP.isoformat(),"max_attempts":MAX_ATTEMPTS,"train_source":str(here/"train_e3.py"),"train_source_sha256":sha(here/"train_e3.py"),"queue_source_sha256":sha(__file__)}));return
 by_uuid={x["uuid"]:x for x in devices}
 def build(r,uuid):
  log=here/"logs"/f"formal_{r['group']}_p{r['fold']}_s{r['seed']}.log";cmd=[a.python,str(here/"train_e3.py"),"--data",r["data"],"--prefix-index",r["prefix_index"],"--source",r["source"],"--visual-checkpoint",r["visual_checkpoint"],"--output",r["output"],"--group",r["group"],"--fold",str(r["fold"]),"--seed",str(r["seed"]),"--formal","--authorization",str(a.authorization),"--protocol",r["protocol"]];return cmd,log
 state=run_queue(a.state,a.lock,[x["uuid"] for x in devices],STOP,accepted_e3,lambda r:pid_matches(r.get("pid"),("train_e3.py",r["output"],"--formal")),build,lambda uuid:{"CUDA_VISIBLE_DEVICES":uuid,"XFORMERS_DISABLED":"1"},launch_discover=lambda r:discover_pids(("train_e3.py",r["output"],"--formal")),owner=owner)
 print(json.dumps({s:sum(r["status"]==s for r in state["runs"]) for s in sorted({r["status"] for r in state["runs"]})}))
if __name__=="__main__":main()
