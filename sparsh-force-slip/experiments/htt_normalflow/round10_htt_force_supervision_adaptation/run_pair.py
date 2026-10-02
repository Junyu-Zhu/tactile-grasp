#!/usr/bin/env python3
"""One accepted force adaptation followed by its strictly dependent fusion."""
import argparse,json,os,subprocess,sys,time,hashlib
from pathlib import Path
C=Path(__file__).resolve().parent
O=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round10_htt_force_supervision_adaptation');R9=O.parent/'round9_htt_temporal_force_fusion'
PY='/home/zjy/miniconda3/envs/sparsh/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--fold',type=int,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args();key=f'p{a.fold}_s{a.seed}';fold=f'htt_leave_p{a.fold}';out=O/'pairs'/key;out.mkdir(parents=True,exist_ok=True);began=time.time()
 prov=json.loads((R9/f'prepare/{key}/audit.json').read_text())['provenance'];fr=O/'formal/force'/key;fu=O/'formal/fusion'/key;prep=O/'prepare'/key;pred=O/'force_predictions'/key
 commands=[('force',[C/'force/train.py','--manifest',O/f'force_support/fold_p{a.fold}.json','--r5-checkpoint',prov['force_checkpoint']['path'],'--source-checkpoint',prov['source_checkpoint']['path'],'--fold',fold,'--seed',a.seed,'--workers',2,'--output',fr]),('force_acceptance',[C/'audit/audit_force.py','--run',fr,'--output',fr/'ACCEPTANCE.json']),('force_predictions',[C/'force/export.py','--run',fr,'--manifest',O/f'force_support/fold_p{a.fold}.json','--output',pred]),('replace_force',[C/'fusion/replace_force.py','--r9-data',R9/f'prepare/{key}/prepared.pt','--predictions',pred/'prediction_manifest.json','--output',prep]),('fusion',[C/'fusion/train.py','--data',prep/'prepared.pt','--group','F_history','--fold',fold,'--seed',a.seed,'--output',fu,'--execute-formal'])]
 completed=[]
 for name,cmd in commands:
  if name=='force' and (fr/'ACCEPTANCE.json').exists():
   acceptance=json.loads((fr/'ACCEPTANCE.json').read_text())
   if acceptance['status']=='pass' and all(sha(p)==h for p,h in acceptance['checkpoints'].items()) and sha(acceptance['summary']['path'])==acceptance['summary']['sha256']:
    completed.append({'stage':'force','reused_accepted':True});continue
  log=out/(name+'.log');start=time.time()
  with log.open('a') as stream:rc=subprocess.run([PY,*map(str,cmd)],stdout=stream,stderr=subprocess.STDOUT).returncode
  if rc:raise RuntimeError(f'{key} {name} failed; {log}')
  completed.append({'stage':name,'seconds':time.time()-start,'command':list(map(str,cmd))});print(key+' '+name+' complete',flush=True)
 summary=json.loads((fu/'summary.json').read_text())
 if summary['status']!='complete' or summary['smoke'] or summary['fold']!=fold or summary['seed']!=a.seed:raise ValueError('fusion receipt identity')
 for name,h in summary['output_hashes'].items():
  if sha(fu/name)!=h:raise ValueError('fusion receipt hash')
 result={'status':'complete','fold':fold,'seed':a.seed,'stages':completed,'seconds':time.time()-began,'force_acceptance':{'path':str(fr/'ACCEPTANCE.json'),'sha256':sha(fr/'ACCEPTANCE.json')},'fusion_summary':{'path':str(fu/'summary.json'),'sha256':sha(fu/'summary.json')},'final_grid_acceptance_still_required':True}
 (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
