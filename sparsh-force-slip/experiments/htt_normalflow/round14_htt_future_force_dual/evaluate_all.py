#!/usr/bin/env python3
import argparse, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import future_train as ft
def main():
 p=argparse.ArgumentParser();p.add_argument("--formal",type=Path,required=True);p.add_argument("--data-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--max-parallel",type=int,default=3);a=p.parse_args();jobs=[]
 for g in ft.GROUPS:
  for f in range(1,5):
   for s in (20260914,20260915,20260916):jobs.append((g,f,s))
 def one(z):
  g,f,s=z;run=f"{g}_p{f}_s{s}";cmd=[sys.executable,str(Path(__file__).with_name("evaluate.py")),"--data",str(a.data_root/f"p{f}_s{s}/prepared.pt"),"--checkpoint",str(a.formal/run/"best.pth"),"--group",g,"--output",str(a.output/run)];cp=subprocess.run(cmd,text=True,capture_output=True);return {"run":run,"returncode":cp.returncode,"stdout_tail":cp.stdout[-500:],"stderr_tail":cp.stderr[-500:]}
 with ThreadPoolExecutor(max_workers=a.max_parallel) as ex: records=list(ex.map(one,jobs))
 status="complete" if all(x["returncode"]==0 for x in records) else "failed";ft.atomic_json({"status":status,"records":records},a.output/"EVALUATION_STATE.json");print(json.dumps({"status":status,"runs":len(records)}));raise SystemExit(0 if status=="complete" else 1)
if __name__=="__main__":main()
