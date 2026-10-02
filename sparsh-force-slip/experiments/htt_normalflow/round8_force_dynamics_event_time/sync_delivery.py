#!/usr/bin/env python3
"""Copy small accepted artifacts locally and verify every mirrored file SHA256."""
import hashlib,json,shlex,subprocess,sys
from pathlib import Path
LOCAL=Path(__file__).resolve().parent
HOST='zjy-4090'
CODE='/home/zjy/document/tactile-grasp/sparsh-force-slip/experiments/htt_normalflow/round8_force_dynamics_event_time'
OUTPUT='/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round8_force_dynamics_event_time'
ORIGINS={}
PYTHON='/home/zjy/miniconda3/envs/sparsh/bin/python'

def remote_script(code):
 return subprocess.check_output(['ssh',HOST,PYTHON+' - <<\'R8_SYNC_PY\'\n'+code+'\nR8_SYNC_PY'],text=True)

def copy_tree(source,dest,filters=()):
 dest.mkdir(parents=True,exist_ok=True)
 # Only output-only destinations are exact mirrors. Mixed code/audit folders
 # retain locally authored review and preparation sources.
 exact = dest.is_relative_to(LOCAL/'results') or dest.is_relative_to(LOCAL/'run_records')
 subprocess.run(['rsync','-a',*(['--delete'] if exact else []),*filters,f'{HOST}:{source}/',str(dest)+'/'],check=True)
 for f in dest.rglob('*'):
  if f.is_file() and '__pycache__' not in f.parts:
   ORIGINS[str(f.relative_to(LOCAL))]=source+'/'+str(f.relative_to(dest))

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 for name in ('FORMAL_INVENTORY.json','PROTOCOL_LOCK.json','SMOKE_AUDIT.json','SUPPORT_DECISION.json','BUDGET_UPDATE.json'):
  subprocess.run(['rsync','-a',f'{HOST}:{OUTPUT}/{name}',str(LOCAL/name)],check=True)
  ORIGINS[name]=OUTPUT+'/'+name
 for name in ('formal_delivery','formal_evaluation','benchmark','formal_queue'):
  copy_tree(OUTPUT+'/'+name,LOCAL/'results'/name,('--exclude','queue.lock','--exclude','*.tmp'))
 # Keep the preparation-era convenience index aligned with its canonical
 # results counterpart, including the final intervention-aware manifest.
 copy_tree(OUTPUT+'/formal_delivery',LOCAL/'formal_delivery')
 copy_tree(OUTPUT+'/amendments',LOCAL/'results/amendments',('--include','*/','--include','*.json','--include','*.md','--include','*.py','--exclude','*'))
 copy_tree(OUTPUT+'/benchmark_multigpu_v1',LOCAL/'results/benchmark_multigpu_v1')
 copy_tree(OUTPUT+'/reporting',LOCAL/'results/reporting')
 copy_tree(OUTPUT+'/state_diagnostics',LOCAL/'results/state_diagnostics')
 copy_tree(OUTPUT+'/reviews',LOCAL/'reviews',('--exclude','__pycache__'))
 copy_tree(OUTPUT+'/interventions',LOCAL/'results/interventions',('--include','*.json','--exclude','*'))
 copy_tree(OUTPUT+'/prepare',LOCAL/'prepare',('--include','*.json','--exclude','*'))
 inventory=json.loads((LOCAL/'FORMAL_INVENTORY.json').read_text())
 for run in inventory['runs']:
  copy_tree(run['output'],LOCAL/'run_records'/run['id'],('--include','config.json','--include','summary.json','--exclude','*'))
 # Canonical code mirror also contains the small delivery bundle; no checkpoint/token cache enters it.
 subprocess.run(['rsync','-a','--exclude','__pycache__','--exclude','*.pyc','--exclude','LOCAL_SYNC_PROOF.json','--exclude','LOCAL_SYNC_PROOF_SHA256.txt',str(LOCAL)+'/',f'{HOST}:{CODE}/'],check=True)
 excluded={'LOCAL_SYNC_PROOF.json','LOCAL_SYNC_PROOF_SHA256.txt'}
 files=[p for p in LOCAL.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and p.name not in excluded]
 local={str(p.relative_to(LOCAL)):digest(p) for p in files}
 script='from pathlib import Path\nimport hashlib,json\nroot=Path('+repr(CODE)+')\npaths='+repr(list(local))+'\nprint(json.dumps({p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}))'
 remote=json.loads(remote_script(script))
 checks={p:h==remote.get(p) for p,h in local.items()}
 origin_code='from pathlib import Path\nimport hashlib,json\npaths='+repr(ORIGINS)+'\nprint(json.dumps({k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in paths.items() if Path(v).is_file()}))'
 origins=json.loads(remote_script(origin_code))
 origin_checks={p:local[p]==h for p,h in origins.items()}
 if not all(origin_checks.values()):raise RuntimeError('Original output vs local SHA mismatch')
 proof={'status':'pass' if all(checks.values()) else 'fail','local_root':str(LOCAL),'server_code_root':CODE,'large_artifact_root':OUTPUT,'file_count':len(files),'bytes':sum(p.stat().st_size for p in files),'files':[{'relative_path':p,'local_sha256':local[p],'server_sha256':remote.get(p),'equal':checks[p]} for p in sorted(local)],'exclusions':['__pycache__','*.pyc','self-referential proof files'],'original_server_output_checks':[{'relative_path':p,'server_source':ORIGINS[p],'sha256':origins[p],'equal':origin_checks[p]} for p in sorted(origins)],'large_artifacts_copied':False}
 dest=LOCAL/'LOCAL_SYNC_PROOF.json';dest.write_text(json.dumps(proof,indent=2)+'\n')
 if not all(checks.values()):raise RuntimeError('Local/server SHA mismatch')
 subprocess.run(['rsync','-a',str(dest),f'{HOST}:{CODE}/LOCAL_SYNC_PROOF.json'],check=True)
 proof_sha=digest(dest)
 remote_sha=remote_script('from pathlib import Path\nimport hashlib\nprint(hashlib.sha256(Path('+repr(CODE+'/LOCAL_SYNC_PROOF.json')+').read_bytes()).hexdigest())').strip()
 if proof_sha!=remote_sha:raise RuntimeError('Sync proof itself differs on server')
 (LOCAL/'LOCAL_SYNC_PROOF_SHA256.txt').write_text(proof_sha+'  LOCAL_SYNC_PROOF.json\n')
 subprocess.run(['rsync','-a',str(LOCAL/'LOCAL_SYNC_PROOF_SHA256.txt'),f'{HOST}:{CODE}/'],check=True)
 print(json.dumps({'status':'pass','files':len(files),'proof_sha256':proof_sha}))
if __name__=='__main__':main()
