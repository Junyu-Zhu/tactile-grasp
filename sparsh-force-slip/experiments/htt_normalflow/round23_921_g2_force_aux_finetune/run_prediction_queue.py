#!/usr/bin/env python3
"""Resume-safe, one-process-per-GPU prediction queue for the accepted E3 grid."""
from __future__ import annotations
import argparse, concurrent.futures, datetime as dt, fcntl, hashlib, json, os, subprocess, threading
from pathlib import Path

HERE=Path(__file__).resolve().parent
LOCK=threading.Lock()
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def accepted(out,run,inventory_sha,predictor_sha):
 try:
  s=json.loads((out/"SUMMARY.json").read_text());return s.get("status")=="complete" and s.get("run")==run and s.get("source_hashes",{}).get("inventory")==inventory_sha and s.get("source_hashes",{}).get("predictor")==predictor_sha and sha(out/"PREDICTIONS.npz")==s.get("prediction_sha256")
 except Exception:return False
def main():
 p=argparse.ArgumentParser();p.add_argument("--inventory",type=Path,required=True);p.add_argument("--output-root",type=Path,required=True);p.add_argument("--python",default="/home/zjy/miniconda3/envs/sparsh/bin/python");p.add_argument("--gpu",nargs="+",default=["0","1","2"]);a=p.parse_args()
 lock=(HERE/"PREDICTION_QUEUE.lock").open("a+")
 try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise RuntimeError("prediction queue already active")
 inv=json.loads(a.inventory.read_text());inventory_sha=sha(a.inventory);predictor_sha=sha(HERE/"predict_e3.py");runs=sorted(inv["runs"],key=lambda x:(x["fold"],x["seed"],x["group"]));a.output_root.mkdir(parents=True,exist_ok=True);logs=HERE/"logs";logs.mkdir(exist_ok=True);state=HERE/"PREDICTION_QUEUE_STATE.json"
 pending=[]
 for r in runs:
  out=a.output_root/"predictions"/r["group"]/f"p{r['fold']}_s{r['seed']}"
  if not accepted(out,r["run"],inventory_sha,predictor_sha):pending.append((r,out))
 def worker(gpu,tasks):
  results=[]
  for r,out in tasks:
   out.mkdir(parents=True,exist_ok=True);log=logs/f"predict_{r['group']}_p{r['fold']}_s{r['seed']}.log";cmd=[a.python,str(HERE/"predict_e3.py"),"--inventory",str(a.inventory),"--run",r["run"],"--output",str(out),"--device","cuda:0"]
   env={**os.environ,"CUDA_VISIBLE_DEVICES":str(gpu),"XFORMERS_DISABLED":"1"}
   with log.open("a") as f:
    f.write(json.dumps({"event":"launch","at":dt.datetime.now(dt.timezone.utc).isoformat(),"gpu":gpu,"cmd":cmd})+"\n");f.flush();q=subprocess.run(cmd,cwd=HERE.parents[2],env=env,stdout=f,stderr=subprocess.STDOUT)
   ok=q.returncode==0 and accepted(out,r["run"],inventory_sha,predictor_sha);results.append({"run":r["run"],"gpu":gpu,"returncode":q.returncode,"accepted":ok,"log":str(log)})
   with LOCK:
    current=json.loads(state.read_text()) if state.exists() else {"schema":"round23_e3_prediction_queue_v1","results":[]};current["results"].append(results[-1]);current["updated_at"]=dt.datetime.now(dt.timezone.utc).isoformat();state.write_text(json.dumps(current,indent=2)+"\n")
   if not ok:break
  return results
 buckets=[[] for _ in a.gpu]
 for i,x in enumerate(pending):buckets[i%len(buckets)].append(x)
 with concurrent.futures.ThreadPoolExecutor(max_workers=len(a.gpu)) as ex:all_results=sum((f.result() for f in [ex.submit(worker,g,b) for g,b in zip(a.gpu,buckets)]),[])
 complete=sum(accepted(a.output_root/"predictions"/r["group"]/f"p{r['fold']}_s{r['seed']}",r["run"],inventory_sha,predictor_sha) for r in runs);summary={"schema":"round23_e3_prediction_queue_summary_v1","status":"complete" if complete==48 else "incomplete","complete":complete,"runs":48,"launched":len(all_results),"inventory_sha256":inventory_sha,"predictor_sha256":predictor_sha,"test_consumed":False};(HERE/"PREDICTION_STATUS.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary));raise SystemExit(0 if complete==48 else 1)
if __name__=="__main__":main()
