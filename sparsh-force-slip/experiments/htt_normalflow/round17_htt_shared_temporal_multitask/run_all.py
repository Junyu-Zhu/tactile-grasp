#!/usr/bin/env python3
import argparse,json,subprocess,sys
from pathlib import Path
import multitask_train as mt
def main():
 p=argparse.ArgumentParser();p.add_argument("--control",type=Path,required=True);p.add_argument("--prepared-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cuda:0");p.add_argument("--group",choices=mt.GROUPS);p.add_argument("--fold",type=int);p.add_argument("--seed",type=int);a=p.parse_args();ctl=json.loads(a.control.read_text())
 if not ctl.get("formal_authorized"):raise SystemExit("formal training not authorized by ROOT_CONTROL.json")
 groups=(a.group,) if a.group else mt.GROUPS;folds=(a.fold,) if a.fold else range(1,5);seeds=(a.seed,) if a.seed else mt.SEEDS
 for g in groups:
  for f in folds:
   for s in seeds:
    data=a.prepared_root/f"p{f}_s{s}"/"prepared.pt";out=a.output/"formal"/g/f"p{f}_s{s}";cmd=[sys.executable,str(Path(__file__).with_name("multitask_train.py")),"train","--data",str(data),"--output",str(out),"--group",g,"--fold",str(f),"--seed",str(s),"--device",a.device];print(json.dumps({"dispatch":f"{g}/p{f}_s{s}","cmd":cmd}),flush=True);subprocess.run(cmd,check=True)
if __name__=="__main__":main()
