#!/usr/bin/env python3
"""Freeze the complete R12 job grid and required preflight receipts."""
import json,hashlib,datetime
from pathlib import Path
C=Path(__file__).resolve().parent
O=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round12_class_preserving_trial_balance')
R10=O.parent/'round10_htt_force_supervision_adaptation'
PY='/home/zjy/miniconda3/envs/sparsh/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
 if p.exists() and json.loads(p.read_text())!=d:raise ValueError('refusing inventory mutation '+str(p))
 p.write_text(json.dumps(d,indent=2)+'\n')
def main():
 protocol=json.loads((C/'training/protocol.json').read_text());O.mkdir(parents=True,exist_ok=True)
 sources={str(p):sha(p) for p in [C/'training/train.py',C/'training/protocol.json',C/'audit_training.py',C/'run_smoke.py',C/'run_jobs.py',C/'PROTOCOL.md',Path(__file__).resolve()]}
 checks=[]
 for p in [O/'SMOKE_AUDIT.json',C/'reviews/REMOTE_PRECHECK.json',C/'reviews/INDEPENDENT_TRAINING_CODE_REVIEW.json',C/'reviews/INDEPENDENT_SMOKE_REVIEW.json',C/'reviews/INDEPENDENT_ACCEPTANCE_CODE_REVIEW.json']:
  d=json.loads(p.read_text())
  if d.get('status')!='pass':raise ValueError('preflight not pass '+str(p))
  checks.append({'path':str(p),'sha256':sha(p)})
 smoke=json.loads((O/'SMOKE_AUDIT.json').read_text())
 for p,h in smoke['sources'].items():
  if sha(p)!=h:raise ValueError('smoke source drift')
 jobs=[]
 for f in range(1,5):
  for seed in protocol['seeds']:
   for g in protocol['groups']:
    name=f'{g}_p{f}_s{seed}';data=R10/f'prepare/p{f}_s{seed}/prepared.pt';out=O/'formal'/name
    jobs.append({'id':name,'group':g,'fold':f'htt_leave_p{f}','seed':seed,'data':str(data),'output':str(out),'receipt':str(out/'summary.json'),'command':[PY,str(C/'training/train.py'),'--data',str(data),'--group',g,'--fold',f'htt_leave_p{f}','--seed',str(seed),'--output',str(out),'--execute-formal']})
 common={'schema':'round12_formal_inventory_v1','source_hashes':sources,'required_checks':checks,'stop_new_dispatch':'2026-09-23T01:55:53+08:00'}
 save(O/'FORMAL_INVENTORY.json',{**common,'jobs':jobs});save(O/'FIRST_FOLD_INVENTORY.json',{**common,'jobs':[x for x in jobs if x['fold']=='htt_leave_p1']});save(O/'REMAINING_INVENTORY.json',{**common,'jobs':[x for x in jobs if x['fold']!='htt_leave_p1']})
 print(json.dumps({'status':'pass','runs':len(jobs)}))
if __name__=='__main__':main()
