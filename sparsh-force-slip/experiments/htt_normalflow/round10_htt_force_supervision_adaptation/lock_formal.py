#!/usr/bin/env python3
import json,hashlib,datetime
from pathlib import Path
C=Path(__file__).resolve().parent;O=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round10_htt_force_supervision_adaptation');PY='/home/zjy/miniconda3/envs/sparsh/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 text=json.dumps(x,indent=2)+'\n'
 if p.exists():
  if json.loads(p.read_text())!=x:raise ValueError('immutable inventory exists '+str(p))
 else:p.write_text(text)
checks=[]
for p in [O/'smoke/SMOKE_AUDIT.json',O/'smoke/OLD_FORCE_PARITY.json',O/'force_support/PREPARE_AUDIT.json',C/'audit/R9_IDENTITY_AUDIT.json',C/'reviews/INDEPENDENT_FORCE_CODE_REVIEW.json',C/'reviews/INDEPENDENT_FUSION_CODE_REVIEW.json',C/'reviews/INDEPENDENT_FORCE_ACCEPTANCE_REVIEW.json']:
 j=json.loads(p.read_text());assert j['status']=='pass';checks.append({'path':str(p),'sha256':sha(p)})
smoke=json.loads((O/'smoke/SMOKE_AUDIT.json').read_text())
for p,h in smoke['sources'].items():assert sha(p)==h,('smoke source drift',p)
paths=[C/x for x in ['PROTOCOL.md','force/train.py','force/prepare.py','force/export.py','fusion/train.py','fusion/protocol.json','fusion/replace_force.py','audit_fusion.py','audit/audit_force.py','run_pair.py','run_jobs.py']]
paths += [O/f'force_support/fold_p{f}.json' for f in range(1,5)]
sources={str(p):sha(p) for p in paths};jobs=[];fusion=[];logical=[]
for f in range(1,5):
 for s in [20260914,20260915,20260916]:
  key=f'p{f}_s{s}';fold=f'htt_leave_p{f}';jobs.append({'id':key,'fold':fold,'seed':s,'command':[PY,str(C/'run_pair.py'),'--fold',str(f),'--seed',str(s)],'receipt':str(O/'pairs'/key/'summary.json')})
  fusion.append({'id':'fusion_'+key,'group':'F_history','fold':fold,'seed':s,'data':str(O/'prepare'/key/'prepared.pt'),'output':str(O/'formal/fusion'/key)})
  logical.extend([{'id':'force_'+key,'kind':'force','fold':fold,'seed':s,'output':str(O/'formal/force'/key)},{'id':'fusion_'+key,'kind':'fusion','fold':fold,'seed':s,'dependency':'force_'+key,'output':str(O/'formal/fusion'/key)}])
base={'source_hashes':sources,'required_checks':checks,'stop_new_dispatch':'2026-09-22T17:55:53+00:00'}
write(O/'FIRST_FOLD_INVENTORY.json',{**base,'jobs':jobs[:3]})
write(O/'REMAINING_INVENTORY.json',{**base,'jobs':jobs[3:]})
write(O/'FUSION_INVENTORY.json',{**base,'jobs':fusion})
write(O/'FORMAL_INVENTORY.json',{**base,'logical_runs':logical,'max_training':24,'run_pairs':jobs,'deadline':'2026-09-25T01:55:53+08:00'})
write(O/'PROTOCOL_LOCK.json',{'status':'pass','source_hashes':sources,'required_checks':checks,'selection_not_based_on_validation_performance':True})
print(json.dumps({'status':'pass','formal_runs':24,'pairs':12}))
