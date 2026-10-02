"""Small read-only verification of F3 support and bound target identities."""
import hashlib,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
audit=json.loads((HERE/'E0_F3_SUPPORT.json').read_text());rows=[];targets={};labels={};fail=[]
for fold in range(1,5):
 support=ROOT/f'round10_htt_force_supervision_adaptation/force_support/fold_p{fold}.json'
 data=json.loads(support.read_text());role_groups={}
 for role in ('fit','selection','calibration','validation'):
  entries=[e for e in data['entries'] if e['role']==role];role_groups[role]={e['leakage_group'] for e in entries}
  for e in entries:
   p=e['force_native_n_path']
   if p not in targets:targets[p]=sha(p)
   if targets[p]!=e['force_native_n_sha256']:fail.append({'target_hash':p})
   p=e['label_path']
   if p not in labels:labels[p]=sha(p)
  # Fixed outer roles suffice to reject F3; no test or new split.
  if role not in ('calibration','validation'):continue
  for h in (1,5,10):
   pos=set();neg=set()
   for e in entries:
    y=np.load(e['label_path']);gross=np.flatnonzero(y==2);first=int(gross[0]) if len(gross) else len(y)
    for t in range(13,min(first,len(y)-h)):
     if y[t]!=0:continue
     future=y[t+1:t+h+1]
     if not np.isin(future,[0,1,2]).all():continue
     (pos if np.any(future==2) else neg).add(e['episode_id'])
   expected=next(r for r in audit['f3'] if r['fold']==fold and r['role']==role)['windows'][str(h)]
   match=pos==set(expected['positive_trial_ids']) and neg==set(expected['negative_trial_ids'])
   if not match:fail.append({'counts':[fold,role,h]})
   rows.append({'fold':fold,'role':role,'horizon':h,'positive_trial_upper_bound':len(pos),'negative_trial_upper_bound':len(neg),'minimum_each':5,'support_pass':len(pos)>=5 and len(neg)>=5,'matches_author':match})
 for a,ga in role_groups.items():
  for b,gb in role_groups.items():
   if a<b and ga&gb:fail.append({'role_overlap':[fold,a,b]})
all_windows_fail=all(any(not r['support_pass'] for r in rows if r['fold']==f and r['horizon']==h) for f in range(1,5) for h in (1,5,10))
result={'scope':'author-stage root read-only checks; not G3 independent review','status':'pass' if not fail and all_windows_fail else 'fail','failures':fail,'all_F3_fold_windows_rejected_by_fixed_outer_roles':all_windows_fail,'outer_counts':rows,'target_hashes_checked':targets,'label_hashes_recorded':labels,'limitations':['label-available upper bounds, not complete neural endpoint validation','force-rule target is not independent physical slip onset','no new independent trials or held-out tests'], 'source_script_sha256':sha(__file__),'support_audit_sha256':sha(HERE/'E0_F3_SUPPORT.json')}
(HERE/'root_support_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['status'],'F3_not_triggered':all_windows_fail,'targets':len(targets),'labels':len(labels),'failures':fail}))
