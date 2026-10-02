#!/usr/bin/env python3
"""Short CPU-only worker used exclusively to verify queue mechanics."""
import argparse,json,os,tempfile,time
from pathlib import Path
def atomic(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(dir=p.parent);os.close(fd)
 try:Path(t).write_text(json.dumps(x,sort_keys=True)+"\n");os.replace(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--run",required=True);p.add_argument("--behavior",choices=("success","fail_once","always_fail","oom_once","hardware_once"),required=True);p.add_argument("--sleep",type=float,default=.2);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);counter=a.output/"attempt_counter.json"
 try:n=json.loads(counter.read_text())["attempts"]+1
 except Exception:n=1
 atomic(counter,{"attempts":n});time.sleep(a.sleep)
 if a.behavior=="oom_once" and n==1:raise RuntimeError("CUDA out of memory: simulated queue smoke")
 if a.behavior=="hardware_once" and n==1:raise RuntimeError("SIMULATED_GPU_FAULT")
 if a.behavior=="always_fail" or (a.behavior=="fail_once" and n==1):raise SystemExit(17)
 atomic(a.output/"summary.json",{"schema":"round22_queue_smoke_worker_v1","status":"complete","run":a.run,"attempt":n,"pid":os.getpid()})
if __name__=="__main__":main()
