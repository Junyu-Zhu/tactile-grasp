#!/usr/bin/env python3
"""Wait for 24/24 formal completion, then execute the locked Round-19 delivery pipeline."""
from __future__ import annotations
import argparse,fcntl,json,os,subprocess,time,traceback
from pathlib import Path
import train as tr
def read(p):return json.loads(Path(p).read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument("--local",type=Path,default=Path(__file__).resolve().parent);p.add_argument("--large",type=Path,required=True);p.add_argument("--r17",type=Path,required=True);p.add_argument("--r3",type=Path,required=True);p.add_argument("--source",type=Path,required=True);p.add_argument("--r18-evaluation",type=Path,required=True);p.add_argument("--r18-final-status",type=Path,required=True);p.add_argument("--r13-source",type=Path,required=True);p.add_argument("--python",default="/home/zjy/miniconda3/envs/sparsh/bin/python");a=p.parse_args()
 lock=(a.local/"POSTPROCESS.lock").open("a+")
 try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise RuntimeError("postprocess already active")
 status_path=a.local/"POSTPROCESS_STATUS.json";logs=a.local/"logs";logs.mkdir(exist_ok=True)
 state=read(status_path) if status_path.exists() else {"schema":"round19_postprocess_status_v1","status":"waiting_formal","steps":{},"test_consumed":False}
 if state.get("status")=="failed":
  history=a.local/"POSTPROCESS_FAILURE_20260919.json"
  if not history.exists():tr.atomic_json(history,state)
  state.setdefault("recovery",{})["prior_failure_sha256"]=tr.sha(history)
  state["recovery"]["reason"]="Completed training status had no gpu_health UUID; cost GPU is freshly probed without altering training identity."
 if state.get("status")=="review_ready" and "supplement" not in state.get("steps",{}):
  revision=a.local/"POSTPROCESS_REVISION_20260920.json"
  if not revision.exists():tr.atomic_json(revision,{"schema":"round19_postprocess_revision_v1","reason":"Independent review requested descriptive matched operating points and force-error trial performance; Q3 readiness wording clarified.","prior_status":state,"training_repeated":False,"locked_evaluation_changed":False,"test_consumed":False})
  state.setdefault("recovery",{})["supplement_revision_sha256"]=tr.sha(revision)
  for step in ("report","finalize"):
   if step in state["steps"]:state["steps"][step]["status"]="pending_supplement_revision"
 tr.atomic_json(status_path,state)
 def mark(status,step=None,extra=None):
  state["status"]=status;state["updated_at"]=time.time()
  if step:state["steps"][step]={"status":status,**(extra or {})}
  tr.atomic_json(status_path,state)
 def run(step,cmd,cuda_uuid=None):
  if state.get("steps",{}).get(step,{}).get("status")=="complete":return
  mark("running",step,{"command":cmd,"started_at":time.time(),"cuda_visible_devices_uuid":cuda_uuid});log=logs/f"post_{step}.log"
  env={**os.environ,"XFORMERS_DISABLED":"1"}
  if cuda_uuid is not None:env["CUDA_VISIBLE_DEVICES"]=cuda_uuid
  with log.open("a") as h:rc=subprocess.run(cmd,stdout=h,stderr=subprocess.STDOUT,env=env).returncode
  if rc:mark("failed",step,{"returncode":rc,"log":str(log)});raise RuntimeError(f"{step} failed: {rc}")
  mark("complete",step,{"returncode":0,"log":str(log),"finished_at":time.time(),"cuda_visible_devices_uuid":cuda_uuid})
 try:
  while True:
   formal=read(a.local/"FORMAL_STATUS.json")
   if formal.get("status_counts",{}).get("failed_terminal",0):raise RuntimeError("formal terminal failure")
   if formal.get("status")=="complete" and formal.get("status_counts")=={"complete":24}:break
   time.sleep(60)
  mark("formal_complete")
  eval_lock=read(a.local/"EVALUATION_LOCK.json")
  for name,digest in eval_lock["sources"].items():
   if tr.sha(a.local/name)!=digest:raise RuntimeError(f"evaluation source changed: {name}")
  cost_lock=read(a.local/"COST_LOCK.json")
  if tr.sha(a.local/"benchmark_deployment.py")!=cost_lock["source_sha256"]:raise RuntimeError("cost source changed")
  evaluation=a.large/"results/evaluation";diagnostics=a.large/"results/diagnostics";report=a.large/"results/report";costdir=a.large/"results/cost";costdir.mkdir(parents=True,exist_ok=True)
  run("training_audit",[a.python,str(a.local/"audit_training.py"),"--local",str(a.local),"--large",str(a.large),"--r17",str(a.r17),"--r3",str(a.r3),"--source",str(a.source),"--output",str(a.local/"TRAINING_AUDIT.json")])
  run("predictions",[a.python,str(a.local/"predict_all.py"),"--local",str(a.local),"--large",str(a.large),"--r17",str(a.r17),"--r3",str(a.r3),"--source",str(a.source)])
  run("evaluation",[a.python,str(a.local/"evaluate.py"),"--prepared-root",str(a.r17),"--prediction-root",str(a.large/"predictions"),"--r18-evaluation",str(a.r18_evaluation),"--r18-final-status",str(a.r18_final_status),"--output",str(evaluation),"--r13-source",str(a.r13_source)])
  run("evaluation_audit",[a.python,str(a.local/"audit_evaluation.py"),"--evaluation",str(evaluation),"--output",str(a.local/"EVALUATION_AUDIT.json")])
  run("diagnostics",[a.python,str(a.local/"diagnostics.py"),"--evaluation",str(evaluation),"--prefix-index",str(a.large/"prefix_cache/PREFIX_INDEX.json"),"--prepared-root",str(a.r17),"--output",str(diagnostics)])
  supplement=a.large/"results/supplement"
  run("supplement",[a.python,str(a.local/"supplement_analysis.py"),"--evaluation",str(evaluation),"--predictions",str(a.large/"predictions"),"--output",str(supplement)])
  force=a.large.parent/"round10_htt_force_supervision_adaptation/formal/force/p1_s20260915/best.pth"
  probe=subprocess.run(["nvidia-smi","--query-gpu=uuid,memory.used,utilization.gpu","--format=csv,noheader,nounits"],capture_output=True,text=True,check=True)
  healthy=[]
  for line in probe.stdout.splitlines():
   uuid,mem,util=[x.strip() for x in line.split(",")]
   check=subprocess.run([a.python,"-c","import torch; assert torch.cuda.is_available(); x=torch.ones(1,device='cuda'); assert x.item()==1"],env={**os.environ,"CUDA_VISIBLE_DEVICES":uuid,"XFORMERS_DISABLED":"1"},capture_output=True,text=True)
   healthy.append({"uuid":uuid,"memory_used_mib":int(mem),"utilization_pct":int(util),"torch_cuda_check_returncode":check.returncode,"torch_cuda_check_stderr":check.stderr[-1000:]})
  tr.atomic_json(a.local/"COST_GPU_PROBE.json",{"schema":"round19_cost_gpu_probe_v1","timestamp":time.time(),"source":"fresh nvidia-smi plus actual torch CUDA allocation","devices":healthy,"test_consumed":False})
  healthy=sorted((x for x in healthy if x["torch_cuda_check_returncode"]==0),key=lambda x:(x["memory_used_mib"],x["utilization_pct"]))
  if not healthy:raise RuntimeError("fresh cost GPU probe found no working CUDA device")
  for group in tr.GROUPS:
   run(f"cost_{group}",[a.python,str(a.local/"benchmark_deployment.py"),"--prepared",str(a.r17/"p1_s20260915/prepared.pt"),"--prefix-index",str(a.large/"prefix_cache/PREFIX_INDEX.json"),"--source",str(a.source),"--visual-checkpoint",str(a.r3/"fold_p1/seed_20260915/best.pth"),"--force-checkpoint",str(force),"--run",str(a.large/"formal"/group/"p1_s20260915"),"--group",group,"--fold","1","--seed","20260915","--device","cuda:0","--output",str(costdir/f"{group}.json")],cuda_uuid=healthy[0]["uuid"])
   run(f"cost_end_to_end_{group}",[a.python,str(a.local/"benchmark_end_to_end_supplement.py"),"--prepared",str(a.r17/"p1_s20260915/prepared.pt"),"--prefix-index",str(a.large/"prefix_cache/PREFIX_INDEX.json"),"--source",str(a.source),"--visual-checkpoint",str(a.r3/"fold_p1/seed_20260915/best.pth"),"--force-checkpoint",str(force),"--run",str(a.large/"formal"/group/"p1_s20260915"),"--group",group,"--fold","1","--seed","20260915","--device","cuda:0","--output",str(costdir/f"{group}_END_TO_END.json")],cuda_uuid=healthy[0]["uuid"])
  run("report",[a.python,str(a.local/"report.py"),"--evaluation",str(evaluation),"--training-audit",str(a.local/"TRAINING_AUDIT.json"),"--evaluation-audit",str(a.local/"EVALUATION_AUDIT.json"),"--cost-v2",str(costdir/"V2.json"),"--cost-m2",str(costdir/"M2.json"),"--cost-end-to-end-v2",str(costdir/"V2_END_TO_END.json"),"--cost-end-to-end-m2",str(costdir/"M2_END_TO_END.json"),"--supplement",str(supplement),"--output",str(report)])
  run("finalize",[a.python,str(a.local/"finalize_delivery.py"),"--local",str(a.local),"--large",str(a.large),"--evaluation",str(evaluation),"--diagnostics",str(diagnostics),"--report",str(report),"--supplement",str(supplement),"--cost-v2",str(costdir/"V2.json"),"--cost-m2",str(costdir/"M2.json"),"--cost-end-to-end-v2",str(costdir/"V2_END_TO_END.json"),"--cost-end-to-end-m2",str(costdir/"M2_END_TO_END.json")])
  state.pop("error",None);state.pop("traceback",None);mark("review_ready")
 except Exception as e:
  state["error"]=str(e);state["traceback"]=traceback.format_exc();mark("failed");raise
 finally:lock.close()
if __name__=="__main__":main()
