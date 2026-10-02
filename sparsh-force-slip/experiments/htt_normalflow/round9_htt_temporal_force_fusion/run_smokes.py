#!/usr/bin/env python3
"""Real-cache two-epoch smoke and interrupted-vs-continuous recovery."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
import torch

def equal(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b

def main(a):
 c=Path(__file__).resolve().parent;out=a.output;out.mkdir(parents=True,exist_ok=True);started=time.time()
 common=[a.python,str(c/'training/train.py'),'--data',str(a.data),'--fold','htt_leave_p1','--seed','20260914','--device','cuda:0','--smoke','--epochs','2']
 for group in ('V_temporal','F_history','F_delta'):
  with (out/(group+'.log')).open('a') as log:
   subprocess.run(common+['--group',group,'--output',str(out/group)],stdout=log,stderr=subprocess.STDOUT,check=True)
  print(group+' smoke pass',flush=True)
 dest=out/'F_delta_recovery'
 with (out/'recovery.log').open('a') as log:
  if not (dest/'latest.pth').exists():subprocess.run(common+['--group','F_delta','--output',str(dest),'--stop-after-epochs','1'],stdout=log,stderr=subprocess.STDOUT,check=True)
  first=torch.load(dest/'latest.pth',map_location='cpu',weights_only=False)
  if first['epoch']!=1 and not (dest/'summary.json').exists():raise ValueError('not the intended interrupt boundary')
  subprocess.run(common+['--group','F_delta','--output',str(dest)],stdout=log,stderr=subprocess.STDOUT,check=True)
 proofs={}
 for name in ('best.pth','latest.pth'):
  original=torch.load(out/'F_delta'/name,map_location='cpu',weights_only=False);resumed=torch.load(dest/name,map_location='cpu',weights_only=False)
  proofs[name]=equal(original,resumed)
 if not all(proofs.values()):raise ValueError('recovery checkpoint differs')
 summaries={g:json.loads((out/g/'summary.json').read_text()) for g in ('V_temporal','F_history','F_delta')}
 if summaries['F_history']['initial_state_sha256']!=summaries['F_delta']['initial_state_sha256']:raise ValueError('initialization differs')
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 audit={'status':'pass','schema':'round9_real_cache_smoke_v1','all_groups':list(summaries),'real_process_interruption_resume_exact':proofs,'epochs':2,'data':str(a.data),'data_sha256':sha(a.data),'same_force_group_initialization':True,'gradient_and_update_checks':{g:{'gradient_audit':s['gradient_audit'],'all_parameter_tensors_updated':all(s['parameter_tensors_updated'].values()),'checkpoint_roundtrip_exact':s['checkpoint_roundtrip_exact'],'prepared_unchanged':s['prepared_unchanged']} for g,s in summaries.items()},'parameters':{g:{'total':s['parameters'],'effective':s['effective_parameters']} for g,s in summaries.items()},'sources':{str(c/'training/train.py'):sha(c/'training/train.py'),str(c/'training/protocol.json'):sha(c/'training/protocol.json'),str(c/'run_smokes.py'):sha(c/'run_smokes.py')},'seconds':time.time()-started}
 (out/'SMOKE_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps({'status':'pass','seconds':audit['seconds']}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--python',required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
