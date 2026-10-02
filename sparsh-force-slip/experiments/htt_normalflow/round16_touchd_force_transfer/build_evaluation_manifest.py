#!/usr/bin/env python3
"""Build the frozen V/H/T_H slip-evaluation index after all formal runs complete."""
import argparse,hashlib,json
from pathlib import Path

HERE=Path(__file__).resolve().parent
R=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow')
ROLES=('train','calibration','validation')

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()

def record(path):path=Path(path);return {'path':str(path),'sha256':sha(path)}

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 old=json.loads((HERE.parent/'round13_trial_level_alarm_calibration/ARTIFACT_INDEX.json').read_text())
 runs=[]
 for source in old['selected_runs']:
  if source['group']=='V_class_trial_balanced':runs.append({**source,'group':'V'})
 for group in ('H','T_H'):
  for fold in range(1,5):
   for seed in (20260914,20260915,20260916):
    key=f'{group}_p{fold}_s{seed}';slip=R/f'round16_touchd_force_transfer/formal/slip/{key}';prepared=R/f'round16_touchd_force_transfer/formal/prepared/{key}'
    summary=json.loads((slip/'summary.json').read_text())
    if summary['status']!='complete' or summary['smoke']:raise RuntimeError('incomplete '+key)
    runs.append({'group':group,'fold':f'htt_leave_p{fold}','seed':seed,'source_round':'round16','source_group':'F_class_trial_balanced','training_summary':record(slip/'summary.json'),'checkpoint':record(slip/'best.pth'),'predictions':{role:record(slip/f'predictions_{role}.csv') for role in ROLES},'endpoints':{role:record(prepared/f'endpoints_{role}.csv') for role in ROLES}})
 expected={(g,f'htt_leave_p{i}',s) for g in ('V','H','T_H') for i in range(1,5) for s in (20260914,20260915,20260916)}
 if len(runs)!=36 or {(x['group'],x['fold'],x['seed']) for x in runs}!=expected:raise RuntimeError('grid')
 out={'schema':'round16_evaluation_manifest_v1','status':'locked','runs':runs,'test_consumed':False}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':'locked','runs':len(runs),'sha256':sha(a.output)}))
if __name__=='__main__':main()
