#!/usr/bin/env python3
"""Root-dispatched bounded GPU queue for immutable R16 inventories."""
import argparse,fcntl,hashlib,json,os,subprocess,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');os.replace(t,p)
def execute(job,gpu,logs):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,XFORMERS_DISABLED='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4');log=logs/(job['id']+'.log');began=time.time()
 with log.open('a') as stream:rc=subprocess.run(job['command'],env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
 ok=rc==0 and Path(job['receipt']).exists()
 if ok:
  try:ok=json.loads(Path(job['receipt']).read_text()).get('status')=='complete'
  except Exception:ok=False
 return {'id':job['id'],'gpu':gpu,'returncode':rc,'ok':ok,'seconds':time.time()-began,'receipt':job['receipt']}
def main():
 p=argparse.ArgumentParser();p.add_argument('--inventory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpus',default='0,1,2');a=p.parse_args();spec=json.loads(a.inventory.read_text());assert spec['status']=='locked'
 if spec.get('requires'):
  prerequisite=Path(spec['requires'])
  if not prerequisite.is_file() or json.loads(prerequisite.read_text()).get('status')!='complete':raise RuntimeError('incomplete prerequisite '+str(prerequisite))
 for path,h in spec['source_hashes'].items():
  if sha(path)!=h:raise RuntimeError('source drift '+path)
 a.output.mkdir(parents=True,exist_ok=True);lock=(a.output/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);logs=a.output/'logs';logs.mkdir(exist_ok=True);state_path=a.output/'status.json';state=json.loads(state_path.read_text()) if state_path.exists() else {'jobs':{}}
 todo=[j for j in spec['jobs'] if state['jobs'].get(j['id'],{}).get('status')!='complete'];gpus=[x.strip() for x in a.gpus.split(',') if x.strip()]
 while todo:
  batch=todo[:len(gpus)];todo=todo[len(batch):]
  with ThreadPoolExecutor(max_workers=len(batch)) as pool:results=list(pool.map(lambda z:execute(*z,logs),zip(batch,gpus)))
  for result in results:state['jobs'][result['id']]={'status':'complete' if result['ok'] else 'failed',**result}
  state['status']='running';state['inventory_sha256']=sha(a.inventory);save(state_path,state)
  if not all(r['ok'] for r in results):state['status']='failed';save(state_path,state);raise SystemExit(1)
 state['status']='complete';save(state_path,state);print(json.dumps({'status':'complete','jobs':len(spec['jobs'])}))
if __name__=='__main__':main()
