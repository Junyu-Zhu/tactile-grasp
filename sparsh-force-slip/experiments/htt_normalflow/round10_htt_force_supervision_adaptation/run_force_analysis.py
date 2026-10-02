#!/usr/bin/env python3
"""Post-training native-force and original-task regression diagnostics."""
import argparse,json,subprocess,time,hashlib
from pathlib import Path
C=Path(__file__).resolve().parent;O=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round10_htt_force_supervision_adaptation');R9=O.parent/'round9_htt_temporal_force_fusion';PY='/home/zjy/miniconda3/envs/sparsh/bin/python'
def main():
 p=argparse.ArgumentParser();p.add_argument('--fold',type=int,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args();key=f'p{a.fold}_s{a.seed}';sup=O/f'force_support/fold_p{a.fold}.json';run=O/'formal/force'/key
 acceptance=json.loads((run/'ACCEPTANCE.json').read_text());assert acceptance['status']=='pass'
 for path,h in acceptance['checkpoints'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
 old=json.loads((R9/f'prepare/{key}/audit.json').read_text())['provenance']['force_prediction_manifest'];assert hashlib.sha256(Path(old['path']).read_bytes()).hexdigest()==old['sha256']
 out=O/'force_analysis_jobs'/key;out.mkdir(parents=True,exist_ok=True);newreg=O/'regression_predictions/new'/key;oldreg=O/'regression_predictions/old'/key
 commands=[('new_regression',[C/'force/export.py','--run',run,'--manifest',sup,'--output',newreg,'--regression']),('old_regression',[C/'force/export_regression_old.py','--run',run,'--manifest',sup,'--output',oldreg,'--old-r5','--regression']),('slip_force_evaluation',[C/'force/evaluate.py','--new',O/'force_predictions'/key/'prediction_manifest.json','--old',old['path'],'--support',sup,'--output',O/'force_evaluation'/key]),('regression_evaluation',[C/'force/evaluate.py','--new',newreg/'prediction_manifest.json','--old',oldreg/'prediction_manifest.json','--support',sup,'--output',O/'force_evaluation'/key/'old_force_regression','--regression'])]
 for name,cmd in commands:
  with (out/(name+'.log')).open('a') as f:subprocess.run([PY,*map(str,cmd)],stdout=f,stderr=subprocess.STDOUT,check=True)
  print(key+' '+name+' complete',flush=True)
 outputs={}
 for p in [O/'force_evaluation'/key/'AUDIT.json',O/'force_evaluation'/key/'old_force_regression/AUDIT.json']:
  assert json.loads(p.read_text())['status']=='pass';outputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 (out/'summary.json').write_text(json.dumps({'status':'complete','fold':a.fold,'seed':a.seed,'outputs':outputs},indent=2)+'\n')
if __name__=='__main__':main()
