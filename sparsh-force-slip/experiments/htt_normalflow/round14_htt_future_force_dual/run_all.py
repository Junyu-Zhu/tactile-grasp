#!/usr/bin/env python3
"""Root-controlled resumable runner for the pre-registered 36-run grid."""
import argparse, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import future_train as ft

def main():
 p=argparse.ArgumentParser(); p.add_argument("--data-root",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--devices",default="cuda:0"); p.add_argument("--max-parallel",type=int,default=1); p.add_argument("--only",choices=ft.GROUPS); p.add_argument("--smoke",action="store_true"); p.add_argument("--execute-formal",action="store_true"); a=p.parse_args()
 if not a.smoke and not a.execute_formal: raise SystemExit("formal execution requires --execute-formal")
 if a.execute_formal and (a.output_root.name!="formal" or "smoke" in str(a.output_root).lower()): raise SystemExit("formal output root identity rejected")
 devices=tuple(x.strip() for x in a.devices.split(",") if x.strip()); assert devices and a.max_parallel==len(devices)
 groups=(a.only,) if a.only else ft.GROUPS; records=[]; jobs=[]
 for group in groups:
  for fold in range(1,5):
   for seed in (20260914,20260915,20260916):
    run=f"{group}_p{fold}_s{seed}"; data=a.data_root/f"p{fold}_s{seed}/prepared.pt"; out=a.output_root/run
    jobs.append((run,data,out,group,fold,seed))
 def execute(job,device):
    run,data,out,group,fold,seed=job; cmd=[sys.executable,str(Path(__file__).with_name("future_train.py")),"train","--data",str(data),"--output",str(out),"--group",group,"--fold",str(fold),"--seed",str(seed),"--device",device]
    if "smoke" in str(data).lower(): return {"run":run,"device":device,"returncode":2,"stdout_tail":"","stderr_tail":"smoke input rejected"}
    if (out/"summary.json").exists() and (out/"best.pth").exists() and (out/"latest.pth").exists():
     old=json.loads((out/"summary.json").read_text())
     if old.get("status")=="complete": return {"run":run,"device":device,"returncode":0,"stdout_tail":"accepted complete; not rerun","stderr_tail":"","reused":True}
    if a.smoke: cmd += ["--max-epochs","2","--patience","99"]
    cp=subprocess.run(cmd,text=True,capture_output=True); return {"run":run,"device":device,"returncode":cp.returncode,"stdout_tail":cp.stdout[-1000:],"stderr_tail":cp.stderr[-1000:]}
 for start in range(0,len(jobs),len(devices)):
  batch=jobs[start:start+len(devices)]
  with ThreadPoolExecutor(max_workers=len(batch)) as ex: batch_records=list(ex.map(lambda z:execute(*z),zip(batch,devices)))
  records.extend(batch_records); ft.atomic_json({"status":"running","records":records},a.output_root/"RUN_STATE.json")
  if any(r["returncode"] for r in batch_records):
   ft.atomic_json({"status":"failed","records":records},a.output_root/"RUN_STATE.json"); raise SystemExit(1)
 ft.atomic_json({"status":"complete","records":records},a.output_root/"RUN_STATE.json"); print(json.dumps({"status":"complete","runs":len(records)}))
if __name__=="__main__": main()
