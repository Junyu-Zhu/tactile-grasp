#!/usr/bin/env python3
"""Copy R12 deliverables and prove source/local/code-mirror SHA256 equality."""
from pathlib import Path
import subprocess,json,hashlib,datetime
L=Path(__file__).resolve().parent
C='/home/zjy/document/tactile-grasp/sparsh-force-slip/experiments/htt_normalflow/round12_class_preserving_trial_balance'
O='/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round12_class_preserving_trial_balance'
def run(args):return subprocess.check_output(args,text=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def remote_hash(root,rels):
 code="import pathlib,hashlib,json;root=pathlib.Path("+repr(root)+");rels="+repr(rels)+";print(json.dumps({r:hashlib.sha256((root/r).read_bytes()).hexdigest() for r in rels}))"
 return json.loads(subprocess.check_output(['ssh','zjy-4090','/home/zjy/miniconda3/envs/sparsh/bin/python -'],input=code,text=True))
exclude=['formal/','*.pt','*.pth','*.npy','__pycache__/','*.pyc','predictions/','frame_errors.csv']
run(['rsync','-a',*sum((['--exclude',x] for x in exclude),[]),'zjy-4090:'+O+'/',str(L/'results')+'/'])
run(['rsync','-a','--include','*/','--include','config.json','--include','summary.json','--include','training_summary.json','--include','ACCEPTANCE.json','--include','TRAIN_WEIGHTS.json','--exclude','*','zjy-4090:'+O+'/formal/',str(L/'run_records')+'/'])
# Reports/audits inside server-only reviews supplement, never delete local author records.
subprocess.run(['ssh','zjy-4090','mkdir -p '+O+'/reviews'],check=True)
run(['rsync','-a','zjy-4090:'+O+'/reviews/',str(L/'reviews')+'/'])
records=[]
for local,remote in [(L/'results',O),(L/'run_records',O+'/formal')]:
 files=sorted(p for p in local.rglob('*') if p.is_file());rels=[str(p.relative_to(local)) for p in files]
 remote_values=remote_hash(remote,rels)
 for p,r in zip(files,rels):
  h=sha(p)
  if h!=remote_values[r]:raise RuntimeError('origin mismatch '+str(p))
  records.append({'local':str(p.relative_to(L)),'origin':remote+'/'+r,'sha256':h})
# Every small code/report/review artifact is also mirrored back to C12; proof excludes itself.
skip={'LOCAL_SYNC_PROOF.json','LOCAL_SYNC_PROOF.sha256'}
run(['rsync','-a','--exclude','__pycache__/','--exclude','*.pyc',*sum((['--exclude',s] for s in skip),[]),str(L)+'/', 'zjy-4090:'+C+'/'])
files=sorted(p for p in L.rglob('*') if p.is_file() and p.name not in skip and '__pycache__' not in p.parts and p.suffix!='.pyc')
rels=[str(p.relative_to(L)) for p in files];rv=remote_hash(C,rels)
mirrors=[]
for p,r in zip(files,rels):
 h=sha(p)
 if h!=rv[r]:raise RuntimeError('mirror mismatch '+r)
 mirrors.append({'local':r,'server':C+'/'+r,'sha256':h})
proof={'status':'pass','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'origin_files':records,'mirror_files':mirrors,'origin_count':len(records),'mirror_count':len(mirrors),'large_artifacts_retained_on_server':True,'excluded_self':['LOCAL_SYNC_PROOF.json','LOCAL_SYNC_PROOF.sha256']}
p=L/'LOCAL_SYNC_PROOF.json';p.write_text(json.dumps(proof,indent=2)+'\n');(L/'LOCAL_SYNC_PROOF.sha256').write_text(sha(p)+'  LOCAL_SYNC_PROOF.json\n')
run(['rsync','-a',str(p),str(L/'LOCAL_SYNC_PROOF.sha256'),'zjy-4090:'+C+'/'])
if remote_hash(C,['LOCAL_SYNC_PROOF.json'])['LOCAL_SYNC_PROOF.json']!=sha(p):raise RuntimeError('proof mismatch')
print(json.dumps({'status':'pass','origin_count':len(records),'mirror_count':len(mirrors),'proof_sha256':sha(p)}))
