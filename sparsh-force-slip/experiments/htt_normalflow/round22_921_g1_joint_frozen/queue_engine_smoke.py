#!/usr/bin/env python3
"""CPU-only end-to-end verification of the durable queue control plane."""
from __future__ import annotations
import argparse,datetime as dt,json,os,subprocess,sys,time
from pathlib import Path
import formal_queue as q
def accepted(r):
 try:d=json.loads((Path(r["output"])/"summary.json").read_text())
 except Exception:return False
 return d.get("status")=="complete" and d.get("run")==r["run"]
def pid_check(r):return q.pid_matches(r.get("pid"),("queue_smoke_worker.py",r["output"],r["run"]))
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--python",default=sys.executable);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);here=Path(__file__).resolve().parent
 started={"pid":os.getpid(),"ppid_at_start":os.getppid(),"started_at":q.now().isoformat()};q.atomic(a.output/"STARTED.json",started)
 runs=[]
 for name,behavior in (("success","success"),("retry","fail_once"),("bounded_failure","always_fail"),("oom","oom_once"),("hardware","hardware_once")):
  runs.append({"run":name,"behavior":behavior,"output":str(a.output/"runs"/name),"status":"registered_not_dispatched","attempts":0,"pid":None})
 state=a.output/"STATE.json";lock=a.output/"QUEUE.lock";q.atomic(state,{"schema":"round22_queue_engine_smoke_v2","concurrency_limit":3,"quarantined_devices":[],"events":[],"runs":runs})
 def build(r,device):return [a.python,str(here/"queue_smoke_worker.py"),"--output",r["output"],"--run",r["run"],"--behavior",r["behavior"],"--sleep","0.3"],a.output/"logs"/(r["run"]+".log")
 stop=q.now()+dt.timedelta(minutes=2);result=q.run_queue(state,lock,["cpu0","cpu1","cpu2"],stop,accepted,pid_check,build,poll_seconds=.1,max_attempts=2)
 attempts_before={r["run"]:r["attempts"] for r in result["runs"]};resume_lock=a.output/"RESUME.lock";resumed=q.run_queue(state,resume_lock,["cpu0"],stop,accepted,pid_check,build,poll_seconds=.1,max_attempts=2);attempts_after={r["run"]:r["attempts"] for r in resumed["runs"]}
 cutoff_state=a.output/"CUTOFF_STATE.json";cutoff_lock=a.output/"CUTOFF.lock";q.atomic(cutoff_state,{"runs":[{"run":"cutoff","behavior":"success","output":str(a.output/"runs"/"cutoff"),"status":"registered_not_dispatched","attempts":0,"pid":None}]});cut=q.run_queue(cutoff_state,cutoff_lock,["cpu0"],q.now()-dt.timedelta(seconds=1),accepted,pid_check,build,poll_seconds=.1,max_attempts=2)["runs"][0]
 orphan={"run":"launch_adopt","behavior":"success","output":str(a.output/"runs"/"launch_adopt"),"status":"launching","attempts":1,"pid":None,"device":"cpu0"};orphan_cmd,orphan_log=build(orphan,"cpu0");Path(orphan_log).parent.mkdir(parents=True,exist_ok=True);fh=Path(orphan_log).open("ab");subprocess.Popen(orphan_cmd,stdout=fh,stderr=subprocess.STDOUT,start_new_session=True);fh.close();orphan_state=a.output/"ORPHAN_STATE.json";q.atomic(orphan_state,{"concurrency_limit":1,"quarantined_devices":[],"events":[],"runs":[orphan]})
 adopted_state=q.run_queue(orphan_state,a.output/"ORPHAN.lock",["cpu0"],stop,accepted,pid_check,build,poll_seconds=.1,max_attempts=2,launch_discover=lambda r:q.discover_pids(("queue_smoke_worker.py",r["output"],r["run"])))
 adopted=adopted_state["runs"][0]
 nohealthy_state=a.output/"NOHEALTHY_STATE.json";q.atomic(nohealthy_state,{"concurrency_limit":1,"quarantined_devices":["cpu0"],"events":[],"runs":[{"run":"nohealthy","behavior":"success","output":str(a.output/"runs"/"nohealthy"),"status":"registered_not_dispatched","attempts":0,"pid":None}]});nohealthy=q.run_queue(nohealthy_state,a.output/"NOHEALTHY.lock",["cpu0"],stop,accepted,pid_check,build,poll_seconds=.1,max_attempts=2)
 inventory={"runs":[{"run":str(i),"output":str(i)} for i in range(48)]};valid_state={"schema":"round22_e3_queue_v3","inventory_sha256":"x","runs":[dict(x) for x in inventory["runs"]]};duplicate_state={**valid_state,"runs":[*valid_state["runs"][:-1],dict(valid_state["runs"][0])]}
 held=q.acquire(a.output/"DUPLICATE.lock");probe="import sys;from pathlib import Path;import formal_queue as q\ntry:q.acquire(Path(sys.argv[1]))\nexcept RuntimeError:sys.exit(23)\nsys.exit(0)";dup=subprocess.run([a.python,"-c",probe,str(a.output/"DUPLICATE.lock")],cwd=here);held.close()
 by={r["run"]:r for r in result["runs"]};checks={"success_completed":by["success"]["status"]=="complete" and by["success"]["attempts"]==1,"failure_retried_once":by["retry"]["status"]=="complete" and by["retry"]["attempts"]==2,"failure_retry_bounded":by["bounded_failure"]["status"]=="failed_terminal" and by["bounded_failure"]["attempts"]==2,"oom_retried_and_reduced_concurrency":by["oom"]["status"]=="complete" and by["oom"]["attempts"]==2 and result["concurrency_limit"]==2,"hardware_retried_and_device_quarantined":by["hardware"]["status"]=="complete" and by["hardware"]["attempts"]==2 and len(result["quarantined_devices"])==1,"all_devices_quarantined_persists_block_and_exits":nohealthy["runs"][0]["status"]=="blocked_no_healthy_devices" and nohealthy["runs"][0]["attempts"]==0 and nohealthy["events"][-1]["event"]=="blocked_no_healthy_devices","exact_state_accepts_unique_48":q.validate_existing_state(valid_state,inventory,"x"),"duplicate_state_rows_rejected":not q.validate_existing_state(duplicate_state,inventory,"x"),"launch_intent_adopted_without_duplicate":adopted["status"]=="complete" and adopted["attempts"]==1 and adopted.get("adopted_after_launch_crash") is True,"resume_skips_accepted":attempts_before==attempts_after,"cutoff_checked_before_dispatch":cut["attempts"]==0 and "dispatch_blocked_at_cutoff" in cut,"duplicate_lock_rejected":dup.returncode==23,"detached_parent_released":os.getppid()==1}
 receipt={"schema":"round22_queue_engine_smoke_receipt_v3","status":"pass" if all(checks.values()) else "fail","checks":checks,"attempts_before_resume":attempts_before,"attempts_after_resume":attempts_after,"concurrency_limit":result["concurrency_limit"],"quarantined_devices":result["quarantined_devices"],"events":result["events"],"cutoff_run":cut,"launch_adopt_run":adopted,"started":started,"finished_pid":os.getpid(),"finished_ppid":os.getppid(),"formal_training_started":False};q.atomic(a.output/"RECEIPT.json",receipt);print(json.dumps(receipt))
if __name__=="__main__":main()
