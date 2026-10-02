#!/usr/bin/env python3
"""Three model families: real subprocess interrupted checkpoint recovery."""
import json,os,subprocess,time,hashlib,importlib.util
from pathlib import Path
import torch
C=Path(__file__).resolve().parent
O=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round11_stable_negative_force_residual')
R10=O.parent/'round10_htt_force_supervision_adaptation'
PY='/home/zjy/miniconda3/envs/sparsh/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def same(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return a==b

def main():
 out=O/'smoke';out.mkdir(parents=True,exist_ok=True);start=time.time();results={}
 env=dict(os.environ,XFORMERS_DISABLED='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
 def run(name,args):
  with (out/(name+'.log')).open('w') as f:p=subprocess.run([PY,*map(str,args)],env=env,stdout=f,stderr=subprocess.STDOUT)
  if p.returncode:raise RuntimeError(name+' failed; see log')
 data=R10/'prepare/p1_s20260914/prepared.pt';inputsha=sha(data)
 for group in ['V_balanced','F_history_balanced','F_residual_balanced']:
  prefix=[C/'training/train.py','--data',data,'--group',group,'--fold','htt_leave_p1','--seed','20260914','--smoke','--epochs','2']
  continuous=out/(group+'_continuous');recovery=out/(group+'_recovery')
  run(group+'_continuous',prefix+['--output',continuous])
  run(group+'_interrupted',prefix+['--output',recovery,'--stop-after-epochs','1'])
  if (recovery/'summary.json').exists() or not (recovery/'latest.pth').exists():raise ValueError('process did not stop at committed checkpoint')
  interrupted=torch.load(recovery/'latest.pth',map_location='cpu',weights_only=False)
  if interrupted['epoch']!=1:raise ValueError('wrong interruption epoch')
  run(group+'_resumed',prefix+['--output',recovery]);checks={}
  for name in ['best.pth','latest.pth']:
   a=torch.load(continuous/name,map_location='cpu',weights_only=False);b=torch.load(recovery/name,map_location='cpu',weights_only=False)
   checks[name]={k:same(a[k],b[k]) for k in ['model_state','optimizer_state','normalizer','history','gradient_audit','initial_state_sha256']}
   if not all(checks[name].values()):raise ValueError('nonexact interrupted recovery')
  summary=json.loads((continuous/'summary.json').read_text())
  if not summary['optimizer_only_new_head'] or not summary['prepared_unchanged'] or not all(summary['parameter_tensors_updated'].values()):raise ValueError('freeze/update failure')
  results[group]={'status':'pass','recovery':checks,'summary':{'path':str(continuous/'summary.json'),'sha256':sha(continuous/'summary.json')},'gradient_audit':summary['gradient_audit'],'parameters':summary['parameters'],'effective_parameters':summary['effective_parameters']}
  print(group+' pass',flush=True)
 if sha(data)!=inputsha:raise ValueError('frozen prepared changed')
 audit={'status':'pass','groups':results,'seconds':time.time()-start,'real_process_recovery':True,'determinism':'no stochastic layers; fixed seed+epoch shuffle; model/optimizer/history/normalizer exact','prepared':{'path':str(data),'sha256':inputsha},'sources':{str(p):sha(p) for p in [C/'training/train.py',C/'training/protocol.json',Path(__file__)]}}
 (O/'SMOKE_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps({'status':'pass','seconds':audit['seconds']}))
if __name__=='__main__':main()
