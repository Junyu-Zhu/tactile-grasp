#!/usr/bin/env python3
"""Real CUDA compute smoke for the enumerated Round-19 recovery GPUs."""
import argparse,datetime as dt,hashlib,json,os,subprocess
from pathlib import Path

TZ=dt.timezone(dt.timedelta(hours=8))
def atomic_json(path,value):
 path=Path(path);tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n");os.replace(tmp,path)
def main():
 p=argparse.ArgumentParser();p.add_argument("--python",default="/home/zjy/miniconda3/envs/sparsh/bin/python");p.add_argument("--gpus",default="0,1");p.add_argument("--output",type=Path,default=Path(__file__).resolve().parent/"GPU_RECOVERY_SMOKE.json");p.add_argument("--require-pass",action="store_true");a=p.parse_args();requested=[x.strip() for x in a.gpus.split(",") if x.strip()]
 query=subprocess.run(["nvidia-smi","--query-gpu=index,uuid,name,pci.bus_id","--format=csv,noheader,nounits"],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 found={}
 if query.returncode==0:
  for line in query.stdout.splitlines():
   index,uuid,name,bus=(x.strip() for x in line.split(",",3));found[index]={"index":int(index),"uuid":uuid,"name":name,"pci_bus_id":bus}
 rows=[];program="import json,torch;x=torch.arange(1048576,device='cuda',dtype=torch.float32);y=(x.sin()*x.cos()).sum();torch.cuda.synchronize();print(json.dumps({'device_name':torch.cuda.get_device_name(0),'value':float(y),'allocated':torch.cuda.memory_allocated(0)}))"
 for token in requested:
  base=found.get(token,{"requested_index":token});env={**os.environ,"CUDA_VISIBLE_DEVICES":base.get("uuid",token),"XFORMERS_DISABLED":"1"}
  try:r=subprocess.run([a.python,"-c",program],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60);row={**base,"returncode":r.returncode,"output":r.stdout[-4000:],"compute_pass":r.returncode==0}
  except subprocess.TimeoutExpired as error:row={**base,"returncode":124,"output":((error.stdout or "")+(error.stderr or ""))[-4000:],"compute_pass":False}
  rows.append(row)
 passed=len(rows)==len(requested) and all(x["compute_pass"] for x in rows)
 result={"schema":"round19_gpu_recovery_smoke_v2","status":"pass" if passed else "blocked_cuda_unavailable","created_at":dt.datetime.now(TZ).isoformat(),"requested_indices":[int(x) for x in requested],"nvidia_smi":{"returncode":query.returncode,"output":query.stdout},"probes":rows,"required":"Every requested GPU must complete a real CUDA allocation, kernel, reduction, and synchronize before queue release.","forbidden_recovery_actions_observed":False,"test_consumed":False}
 atomic_json(a.output,result);print(json.dumps(result,indent=2));
 if a.require_pass and not passed:raise SystemExit(2)
if __name__=="__main__":main()
