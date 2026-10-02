#!/usr/bin/env python3
"""Root-owned real process interruption and complete two-stage smoke."""
import argparse,json,os,subprocess,time
from pathlib import Path
import numpy as np
import torch
C=Path(__file__).resolve().parent
O=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round10_htt_force_supervision_adaptation')
R9=O.parent/'round9_htt_temporal_force_fusion'
PY='/home/zjy/miniconda3/envs/sparsh/bin/python'
def same(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return a==b

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=O/'smoke');a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True);began=time.time()
 env=dict(os.environ,XFORMERS_DISABLED='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
 def run(name,args,interrupt=False):
  with (out/(name+'.log')).open('w') as log:rc=subprocess.run([PY,*map(str,args)],env=env,stdout=log,stderr=subprocess.STDOUT).returncode
  if interrupt:
   if rc==0 or 'intentional interruption after committed epoch 1' not in (out/(name+'.log')).read_text():raise ValueError('intentional interruption not observed')
  elif rc:raise RuntimeError(name+' failed, see log')
  print(name+' pass',flush=True)
 key='p1_s20260914';audit=json.loads((R9/f'prepare/{key}/audit.json').read_text());prov=audit['provenance']
 forcebase=[C/'force/train.py','--manifest',O/'force_support/fold_p1.json','--r5-checkpoint',prov['force_checkpoint']['path'],'--source-checkpoint',prov['source_checkpoint']['path'],'--fold','htt_leave_p1','--seed','20260914','--workers','0','--smoke']
 run('force_continuous',forcebase+['--output',out/'force_continuous'])
 run('force_interrupted',forcebase+['--output',out/'force_recovery','--interrupt-after-epoch','1'],True)
 run('force_resumed',forcebase+['--output',out/'force_recovery'])
 forcecheck={}
 for file in ['best.pth','latest.pth']:
  x=torch.load(out/'force_continuous'/file,map_location='cpu',weights_only=False);y=torch.load(out/'force_recovery'/file,map_location='cpu',weights_only=False)
  forcecheck[file]={k:same(x[k],y[k]) for k in ['model_state','optimizer_state','history','normalization','rng_state']}
  if not all(forcecheck[file].values()):raise ValueError('force interrupted recovery not exact')
 run('force_export',[C/'force/export.py','--run',out/'force_continuous','--manifest',O/'force_support/fold_p1.json','--output',out/'force_predictions'])
 run('force_replace',[C/'fusion/replace_force.py','--r9-data',R9/f'prepare/{key}/prepared.pt','--predictions',out/'force_predictions/prediction_manifest.json','--output',out/'prepared','--smoke'])
 fusionbase=[C/'fusion/train.py','--data',out/'prepared/prepared.pt','--group','F_history','--fold','htt_leave_p1','--seed','20260914','--smoke','--epochs','2']
 run('fusion_continuous',fusionbase+['--output',out/'fusion_continuous'])
 run('fusion_interrupted',fusionbase+['--output',out/'fusion_recovery','--stop-after-epochs','1'])
 if (out/'fusion_recovery/summary.json').exists():raise ValueError('not actually interrupted')
 run('fusion_resumed',fusionbase+['--output',out/'fusion_recovery'])
 fusioncheck={}
 for file in ['best.pth','latest.pth']:
  x=torch.load(out/'fusion_continuous'/file,map_location='cpu',weights_only=False);y=torch.load(out/'fusion_recovery'/file,map_location='cpu',weights_only=False)
  fusioncheck[file]={k:same(x[k],y[k]) for k in ['model_state','optimizer_state','history','normalizer']}
  if not all(fusioncheck[file].values()):raise ValueError('fusion interrupted recovery not exact')
 import hashlib
 def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
 result={'status':'pass','force_real_recovery':forcecheck,'fusion_real_recovery':fusioncheck,'seconds':time.time()-began,'small_force_smoke_not_full_budget':True,'force_subset_frames':64,'sources':{str(f):sha(f) for f in [Path(__file__),C/'force/train.py',C/'force/export.py',C/'fusion/train.py',C/'fusion/replace_force.py']},'summaries':{str(f):sha(f) for f in [out/'force_continuous/training_summary.json',out/'force_recovery/training_summary.json',out/'fusion_continuous/summary.json',out/'fusion_recovery/summary.json']}}
 (out/'SMOKE_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'pass','seconds':result['seconds']}))
if __name__=='__main__':main()
