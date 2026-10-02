#!/usr/bin/env python3
"""Resume-safe preregistered gradient diagnostic queue (B/D only)."""
from __future__ import annotations
import argparse,concurrent.futures,fcntl,hashlib,json,os,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256(Path(p).read_bytes());return h.hexdigest()
def ok(path,run,source_sha,ids_sha):
 try:
  x=json.loads(path.read_text());return x.get("status")=="complete" and x.get("run")==run and x.get("source_sha256")==source_sha and x.get("ids_sha256")==ids_sha
 except Exception:return False
def main():
 p=argparse.ArgumentParser();p.add_argument("--inventory",type=Path,required=True);p.add_argument("--ids",type=Path,required=True);p.add_argument("--output-root",type=Path,required=True);p.add_argument("--python",default="/home/zjy/miniconda3/envs/sparsh/bin/python");p.add_argument("--gpu",nargs="+",default=["0","1","2"]);a=p.parse_args();lock=(HERE/"GRADIENT_QUEUE.lock").open("a+")
 try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise RuntimeError("gradient queue already active")
 source_sha=sha(HERE/"gradient_diagnostics.py");ids_sha=sha(a.ids);runs=[r for r in json.loads(a.inventory.read_text())["runs"] if r["group"] in ("B","D")];tasks=[]
 for r in runs:
  out=a.output_root/"gradients"/r["group"]/f"p{r['fold']}_s{r['seed']}.json"
  if not ok(out,r["run"],source_sha,ids_sha):tasks.append((r,out))
 buckets=[[] for _ in a.gpu]
 for i,x in enumerate(tasks):buckets[i%len(buckets)].append(x)
 def worker(gpu,bucket):
  ans=[]
  for r,out in bucket:
   out.parent.mkdir(parents=True,exist_ok=True);log=HERE/"logs"/f"gradient_{r['group']}_p{r['fold']}_s{r['seed']}.log";log.parent.mkdir(exist_ok=True);cmd=[a.python,str(HERE/"gradient_diagnostics.py"),"--inventory",str(a.inventory),"--ids",str(a.ids),"--run",r["run"],"--output",str(out),"--device","cuda:0"]
   with log.open("a") as f:q=subprocess.run(cmd,cwd=HERE.parents[2],env={**os.environ,"CUDA_VISIBLE_DEVICES":str(gpu),"XFORMERS_DISABLED":"1"},stdout=f,stderr=subprocess.STDOUT)
   if q.returncode==0:
    x=json.loads(out.read_text());x["source_sha256"]=source_sha;x["ids_sha256"]=ids_sha;out.write_text(json.dumps(x,indent=2)+"\n")
   good=q.returncode==0 and ok(out,r["run"],source_sha,ids_sha);ans.append({"run":r["run"],"gpu":gpu,"accepted":good,"log":str(log)});
   if not good:break
  return ans
 with concurrent.futures.ThreadPoolExecutor(max_workers=len(a.gpu)) as ex:launched=sum((f.result() for f in [ex.submit(worker,g,b) for g,b in zip(a.gpu,buckets)]),[])
 complete=sum(ok(a.output_root/"gradients"/r["group"]/f"p{r['fold']}_s{r['seed']}.json",r["run"],source_sha,ids_sha) for r in runs);summary={"schema":"round23_e3_gradient_queue_v1","status":"complete" if complete==24 else "incomplete","complete":complete,"runs":24,"launched":launched,"source_sha256":source_sha,"ids_sha256":ids_sha,"test_consumed":False};(HERE/"GRADIENT_STATUS.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary));raise SystemExit(0 if complete==24 else 1)
if __name__=="__main__":main()
