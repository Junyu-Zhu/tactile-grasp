#!/usr/bin/env python3
"""Three-GPU prediction/diagnostic exporter for completed Round-19 runs."""
from __future__ import annotations
import argparse,concurrent.futures,json,os,subprocess,time
from pathlib import Path
import train as tr

def accepted(out,group,fold,seed):
 p=out/"SUMMARY.json"
 if not p.exists():return False
 try:x=json.loads(p.read_text())
 except Exception:return False
 return x.get("status")=="complete" and (x.get("group"),x.get("fold"),x.get("seed"))==(group,fold,seed) and tr.sha(out/"PREDICTIONS.npz")==x.get("prediction_sha256")
def main():
 p=argparse.ArgumentParser();p.add_argument("--local",type=Path,default=Path(__file__).resolve().parent);p.add_argument("--large",type=Path,required=True);p.add_argument("--r17",type=Path,required=True);p.add_argument("--r3",type=Path,required=True);p.add_argument("--source",type=Path,required=True);p.add_argument("--python",default="/home/zjy/miniconda3/envs/sparsh/bin/python");a=p.parse_args()
 formal=json.loads((a.local/"FORMAL_STATUS.json").read_text())
 if formal.get("status")!="complete" or formal.get("status_counts")!={"complete":24}:raise ValueError("formal grid incomplete")
 tasks=[(g,f,s) for f in range(1,5) for s in tr.SEEDS for g in tr.GROUPS];state={"schema":"round19_prediction_queue_v1","status":"running","runs":{}};tr.atomic_json(a.local/"PREDICTION_STATUS.json",state)
 def one(task,gpu):
  group,fold,seed=task;rid=f"{group}/p{fold}_s{seed}";out=a.large/"predictions"/group/f"p{fold}_s{seed}"
  if accepted(out,group,fold,seed):return {"run":rid,"status":"complete_reused","gpu":gpu}
  cmd=[a.python,str(a.local/"predict.py"),"--prepared",str(a.r17/f"p{fold}_s{seed}"/"prepared.pt"),"--prefix-index",str(a.large/"prefix_cache/PREFIX_INDEX.json"),"--source",str(a.source),"--visual-checkpoint",str(a.r3/f"fold_p{fold}"/f"seed_{seed}"/"best.pth"),"--run",str(a.large/"formal"/group/f"p{fold}_s{seed}"),"--output",str(out),"--group",group,"--fold",str(fold),"--seed",str(seed),"--device","cuda:0"]
  log=a.local/"logs"/f"predict_{group}_p{fold}_s{seed}.log";beg=time.time()
  with log.open("a") as h:rc=subprocess.run(cmd,stdout=h,stderr=subprocess.STDOUT,env={**os.environ,"CUDA_VISIBLE_DEVICES":str(gpu),"XFORMERS_DISABLED":"1"}).returncode
  return {"run":rid,"status":"complete" if rc==0 and accepted(out,group,fold,seed) else "failed","gpu":gpu,"returncode":rc,"wall_seconds":time.time()-beg}
 def lane(gpu,items):return [one(x,gpu) for x in items]
 lanes=[tasks[i::3] for i in range(3)];rows=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
  for part in ex.map(lambda x:lane(*x),enumerate(lanes)):
   rows.extend(part);state["runs"].update({r["run"]:r for r in part});tr.atomic_json(a.local/"PREDICTION_STATUS.json",state)
 state.update(status="complete" if all(x["status"].startswith("complete") for x in rows) else "failed",complete=sum(x["status"].startswith("complete") for x in rows));tr.atomic_json(a.local/"PREDICTION_STATUS.json",state);print(json.dumps(state,indent=2))
 if state["status"]!="complete":raise SystemExit(1)
if __name__=="__main__":main()
