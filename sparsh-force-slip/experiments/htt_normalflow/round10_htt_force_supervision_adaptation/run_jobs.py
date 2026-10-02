#!/usr/bin/env python3
"""Root-owned bounded GPU queue. Never inspects/stops unrelated processes."""
import argparse,datetime,json,os,subprocess,time,fcntl,hashlib
from pathlib import Path

def save(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)

def main(a):
 spec=json.loads(a.inventory.read_text());out=a.output;out.mkdir(parents=True,exist_ok=True)
 for filename,expected in spec.get('source_hashes',{}).items():
  if hashlib.sha256(Path(filename).read_bytes()).hexdigest()!=expected:raise ValueError('locked source changed: '+filename)
 for check in spec.get('required_checks',[]):
  path=Path(check['path'])
  if hashlib.sha256(path.read_bytes()).hexdigest()!=check['sha256'] or json.loads(path.read_text()).get('status') not in ('pass','complete'):raise ValueError('required check failed')
 lock=(out/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state_path=out/'status.json';state=json.loads(state_path.read_text()) if state_path.exists() else {'jobs':{}}
 previous=state.get('inventory')
 if previous is not None and previous!=spec:raise ValueError('queue inventory changed')
 state['inventory']=spec;state['status']='running';save(state_path,state)
 todo=[j for j in spec['jobs'] if state['jobs'].get(j['id'],{}).get('status')!='complete']
 active={};gpus=a.gpus.split(',');failed=[]
 while todo or active:
  for gpu in gpus:
   if gpu in active or not todo:continue
   if spec.get('stop_new_dispatch') and datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat(spec['stop_new_dispatch']):raise RuntimeError('training dispatch deadline reached; no extension')
   job=todo.pop(0);env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,XFORMERS_DISABLED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
   log=(out/(job['id']+'.log')).open('a');proc=subprocess.Popen(job['command'],env=env,stdout=log,stderr=subprocess.STDOUT)
   active[gpu]=(job,proc,log,time.time());state['jobs'][job['id']]={'status':'running','pid':proc.pid,'gpu':gpu,'started_at':time.time()};save(state_path,state)
   print(json.dumps({'started':job['id'],'gpu':gpu,'pid':proc.pid}),flush=True)
  for gpu,(job,proc,log,start) in list(active.items()):
   rc=proc.poll()
   if rc is None:continue
   log.close();ok=rc==0
   if ok:
    try:ok=json.loads(Path(job['receipt']).read_text()).get('status') in ('pass','complete')
    except (OSError,ValueError):ok=False
   state['jobs'][job['id']].update(status='complete' if ok else 'failed',returncode=rc,seconds=time.time()-start)
   if not ok:failed.append(job['id'])
   del active[gpu];save(state_path,state);print(json.dumps({'finished':job['id'],'ok':ok}),flush=True)
  if active:time.sleep(.5)
 state['status']='complete' if not failed else 'failed';state['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save(state_path,state)
 if failed:raise RuntimeError('failed jobs: '+','.join(failed))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--inventory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpus',default='0,1,2');main(p.parse_args())
