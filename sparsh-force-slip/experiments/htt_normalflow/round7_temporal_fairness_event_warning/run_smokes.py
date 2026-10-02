#!/usr/bin/env python3
"""Root-managed bounded smoke jobs, including a real interruption/resume."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--code',type=Path,required=True);a=p.parse_args();r=a.root;c=a.code;log=r/'smoke_logs';log.mkdir(parents=True,exist_ok=True)
 jobs=[('A_visual','A_visual','0',[]),('B_force','B_force','1',[]),('C_force_delta','C_force_delta','2',[]),('D_visual_delta','D_visual_delta','0',[]),('C_resume','C_force_delta','2',['--interrupt-after-epoch','1']),('C_resume','C_force_delta','2',['--resume'])];receipts=[]
 def spawn(job):
  name,g,gpu,extra=job;stage='resume' if '--resume' in extra else 'initial';out=r/'smoke'/name;env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,XFORMERS_DISABLED='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4');cmd=[sys.executable,str(c/'training/train.py'),'--data',str(r/'prepare/prepared.pt'),'--group',g,'--seed','20260914','--output',str(out),'--device','cuda:0','--smoke',*extra];f=(log/f'{name}.{stage}.log').open('w');return {'name':name,'stage':stage,'command':cmd,'gpu':gpu,'started_unix':time.time(),'proc':subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env),'file':f}
 def finish(x):
  rc=x['proc'].wait();x['file'].close();d={k:v for k,v in x.items() if k not in ['proc','file']};d.update(exit_code=rc,finished_unix=time.time());receipts.append(d);(r/'SMOKE_LAUNCH_RECEIPTS.json').write_text(json.dumps(receipts,indent=2)+'\n')
  if rc:raise RuntimeError('Smoke failed '+x['name']+' '+x['stage'])
 batch=[spawn(j) for j in jobs[:3]]
 for x in batch:finish(x)
 batch=[spawn(j) for j in jobs[3:5]]
 for x in batch:finish(x)
 finish(spawn(jobs[5]));print(json.dumps({'status':'complete','smoke_processes':len(receipts)}))
if __name__=='__main__':main()
