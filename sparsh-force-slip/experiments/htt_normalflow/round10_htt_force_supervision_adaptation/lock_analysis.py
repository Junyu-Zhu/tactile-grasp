import json,hashlib
from pathlib import Path
C=Path(__file__).resolve().parent;O=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round10_htt_force_supervision_adaptation');PY='/home/zjy/miniconda3/envs/sparsh/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
fa=O/'formal_delivery/FORCE_TRAINING_AUDIT.json';assert json.loads(fa.read_text())['status']=='pass'
paths=[C/x for x in ['run_force_analysis.py','force/export.py','force/export_regression_old.py','force/evaluate.py','force/aggregate.py','force/report.py']]
jobs=[]
for f in range(1,5):
 for s in [20260914,20260915,20260916]:
  key=f'p{f}_s{s}';receipt=O/'force_analysis_jobs'/key/'summary.json'
  if receipt.exists():
   r=json.loads(receipt.read_text());assert r['status']=='complete' and all(sha(p)==h for p,h in r['outputs'].items());continue
  jobs.append({'id':key,'command':[PY,str(C/'run_force_analysis.py'),'--fold',str(f),'--seed',str(s)],'receipt':str(receipt)})
obj={'jobs':jobs,'source_hashes':{str(p):sha(p) for p in paths},'required_checks':[{'path':str(fa),'sha256':sha(fa)}]}
p=O/'FORCE_ANALYSIS_INVENTORY.json'
if p.exists():assert json.loads(p.read_text())==obj
else:p.write_text(json.dumps(obj,indent=2)+'\n')
print(json.dumps({'status':'pass','jobs':len(jobs)}))
