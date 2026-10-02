#!/usr/bin/env python3
"""Non-formal D-group interruption/resume equivalence smoke on a fixed subset."""
from __future__ import annotations
import argparse,copy,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
import torch

def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def dump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")
def subset_role(r,per_class):
 ids=torch.cat([torch.nonzero(r["stage"].eq(c),as_tuple=False).flatten()[:per_class] for c in (0,2)]).sort().values
 if any(int(r["stage"][ids].eq(c).sum())!=per_class for c in (0,2)):raise ValueError("insufficient class support")
 out={}
 for k,v in r.items():
  if torch.is_tensor(v):out[k]=v[ids].clone()
  elif isinstance(v,list):out[k]=[v[i] for i in ids.tolist()]
  else:out[k]=copy.deepcopy(v)
 return out
def prepare_subset(source,out):
 d=torch.load(source,map_location="cpu",weights_only=False);roles={}
 for role,r in d["roles"].items():roles[role]=subset_role(r,32 if role=="fit" else 16)
 groups={role:set(r["leakage_group"]) for role,r in roles.items()}
 if groups["fit"]&groups["selection"]:raise ValueError("fit/selection leakage")
 result={**{k:copy.deepcopy(v) for k,v in d.items() if k!="roles"},"schema":"round22_e3_recovery_smoke_subset_v1","roles":roles,"smoke_subset":{"source":str(source),"source_sha256":sha(source),"selection":"stable first per class within original role","formal":False}}
 torch.save(result,out)
 return {r:{"rows":len(x["t"]),"stage_counts":{str(c):int(x["stage"].eq(c).sum()) for c in (0,1,2)},"leakage_groups":len(set(x["leakage_group"]))} for r,x in roles.items()}
def wait_for_line(proc,log,event,epoch,timeout):
 deadline=time.time()+timeout
 while time.time()<deadline:
  text=Path(log).read_text(errors="replace") if Path(log).exists() else ""
  for line in text.splitlines():
   try:d=json.loads(line)
   except json.JSONDecodeError:continue
   if d.get("event")==event and d.get("epoch")==epoch:return d
  if proc.poll() is not None:raise RuntimeError(f"process exited before {event} epoch {epoch}: {proc.returncode}")
  time.sleep(.2)
 raise TimeoutError(f"timeout waiting for {event} epoch {epoch}")
def launch(cmd,log,env):
 fh=Path(log).open("ab");p=subprocess.Popen(cmd,stdout=fh,stderr=subprocess.STDOUT,env=env,start_new_session=True);fh.close();return p
def nested_equal(a,b,path="root",differences=None):
 differences=[] if differences is None else differences
 if torch.is_tensor(a) and torch.is_tensor(b):
  if not torch.equal(a,b):differences.append(path)
 elif isinstance(a,dict) and isinstance(b,dict):
  if set(a)!=set(b):differences.append(path+".keys")
  else:
   for k in sorted(a,key=str):nested_equal(a[k],b[k],path+f".{k}",differences)
 elif isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
  if len(a)!=len(b):differences.append(path+".length")
  else:
   for i,(x,y) in enumerate(zip(a,b)):nested_equal(x,y,path+f"[{i}]",differences)
 elif hasattr(a,"shape") and hasattr(b,"shape"):
  import numpy as np
  if not np.array_equal(a,b):differences.append(path)
 elif a!=b:differences.append(path)
 return differences
def comparable(ck):
 x=copy.deepcopy(ck)
 for row in x["history"]:
  row.pop("train_seconds",None);row.pop("selection_seconds",None)
 return x
def main():
 p=argparse.ArgumentParser();p.add_argument("--source-data",type=Path,required=True);p.add_argument("--prefix-index",type=Path,required=True);p.add_argument("--source",type=Path,required=True);p.add_argument("--visual-checkpoint",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--python",default=sys.executable);p.add_argument("--device",default="cuda:0");a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 here=Path(__file__).resolve().parent;data=a.output/"subset.pt";roles=prepare_subset(a.source_data,data);continuous=a.output/"continuous";resumed=a.output/"interrupted_resumed";logs=a.output/"logs";logs.mkdir()
 base=[a.python,str(here/"train_e3.py"),"--data",str(data),"--prefix-index",str(a.prefix_index),"--source",str(a.source),"--visual-checkpoint",str(a.visual_checkpoint),"--group","D","--fold","1","--seed","20260914","--device",a.device,"--max-epochs","3","--patience","10"]
 env={**os.environ,"XFORMERS_DISABLED":"1","OMP_NUM_THREADS":"4","MKL_NUM_THREADS":"4"}
 ccmd=base+["--output",str(continuous)];cp=launch(ccmd,logs/"continuous.log",env);crc=cp.wait()
 if crc:raise RuntimeError(f"continuous failed: {crc}")
 icmd=base+["--output",str(resumed)];ip=launch(icmd,logs/"interrupted.log",env);observed=wait_for_line(ip,logs/"interrupted.log","first_optimizer_step",1,900)
 actual_cmd=(Path("/proc")/str(ip.pid)/"cmdline").read_bytes().replace(b"\0",b" ").decode(errors="replace")
 if str(here/"train_e3.py") not in actual_cmd or str(resumed) not in actual_cmd:raise RuntimeError("refusing to terminate unmatched process")
 terminated={"pid":ip.pid,"command":actual_cmd,"observed":observed,"signal":"SIGTERM","at":time.time()};os.kill(ip.pid,signal.SIGTERM);irc=ip.wait(timeout=30);terminated["commit_after_termination"]=json.loads((resumed/"COMMIT.json").read_text())
 if irc==0:raise RuntimeError("interruption unexpectedly completed normally")
 rp=launch(icmd,logs/"resumed.log",env);rrc=rp.wait()
 if rrc:raise RuntimeError(f"resume failed: {rrc}")
 cm=json.loads((continuous/"COMMIT.json").read_text());rm=json.loads((resumed/"COMMIT.json").read_text());cc=torch.load(continuous/cm["latest"]["path"],map_location="cpu",weights_only=False);rc=torch.load(resumed/rm["latest"]["path"],map_location="cpu",weights_only=False)
 differences=nested_equal(comparable(cc),comparable(rc));checks={"continuous_complete":json.loads((continuous/"summary.json").read_text())["status"]=="complete","interrupted_after_next_epoch_started":terminated["observed"]["epoch"]==1,"authoritative_commit_still_epoch0_after_termination":terminated["commit_after_termination"]["epoch"]==0,"interrupted_nonzero_exit":irc!=0,"resumed_complete":json.loads((resumed/"summary.json").read_text())["status"]=="complete","final_epoch_equal":cc["epoch"]==rc["epoch"]==2,"model_optimizer_rng_earlystop_exact":not differences,"same_best_commit_epoch":cm["best"]["epoch"]==rm["best"]["epoch"]}
 receipt={"schema":"round22_e3_real_recovery_smoke_v1","status":"pass" if all(checks.values()) else "fail","formal":False,"group":"D","epochs":3,"roles":roles,"subset_sha256":sha(data),"continuous_command":ccmd,"interrupted_command":icmd,"terminated_process":terminated,"interrupted_returncode":irc,"checks":checks,"differences_excluding_walltime_fields":differences,"continuous_commit":cm,"resumed_commit":rm}
 dump(a.output/"RECEIPT.json",receipt);print(json.dumps(receipt))
 if receipt["status"]!="pass":raise SystemExit(1)
if __name__=="__main__":main()
