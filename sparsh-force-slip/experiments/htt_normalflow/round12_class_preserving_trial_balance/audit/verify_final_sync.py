"""Read-only final cross-host sync validation; writes stdout only."""
import sys,json,hashlib,subprocess
from pathlib import Path
L=Path('/home/zjy/Documents/grasp/tactile_grasp/sparsh-force-slip/experiments/htt_normalflow/round12_class_preserving_trial_balance')
C='/home/zjy/document/tactile-grasp/sparsh-force-slip/experiments/htt_normalflow/round12_class_preserving_trial_balance'
O='/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round12_class_preserving_trial_balance'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
expected=sys.argv[1];p=L/'LOCAL_SYNC_PROOF.json';assert sha(p)==expected,'unexpected final proof';proof=json.loads(p.read_text());assert proof['status']=='pass'
assert (L/'LOCAL_SYNC_PROOF.sha256').read_text()==expected+'  LOCAL_SYNC_PROOF.json\n'
orig=proof['origin_files'];mir=proof['mirror_files'];assert len(orig)==proof['origin_count'] and len(mir)==proof['mirror_count'];assert len({r['local'] for r in orig})==len(orig) and len({r['local'] for r in mir})==len(mir)
skip={'LOCAL_SYNC_PROOF.json','LOCAL_SYNC_PROOF.sha256'}
local_files={str(f.relative_to(L)) for f in L.rglob('*') if f.is_file() and f.name not in skip and '__pycache__' not in f.parts and f.suffix!='.pyc'}
assert local_files=={r['local'] for r in mir},'local coverage mismatch'
for r in orig+mir:assert sha(L/r['local'])==r['sha256'],r['local']
for r in mir:assert r['server']==C+'/'+r['local']
assert {r['local'] for r in orig}=={r for r in local_files if r.startswith(('results/','run_records/'))},'origin coverage mismatch'
remote_code='''import pathlib,hashlib,json
payload=json.loads(PAYLOAD)
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
for r in payload['origin']:assert sha(r['origin'])==r['sha256'],r['origin']
for r in payload['mirror']:assert sha(r['server'])==r['sha256'],r['server']
C=pathlib.Path(payload['C']);O=pathlib.Path(payload['O']);expected=payload['proof']
assert sha(C/'LOCAL_SYNC_PROOF.json')==expected
assert (C/'LOCAL_SYNC_PROOF.sha256').read_text()==expected+'  LOCAL_SYNC_PROOF.json\\n'
cm={str(f.relative_to(C)) for f in C.rglob('*') if f.is_file() and f.name not in ('LOCAL_SYNC_PROOF.json','LOCAL_SYNC_PROOF.sha256') and '__pycache__' not in f.parts and f.suffix!='.pyc'}
assert cm=={r['local'] for r in payload['mirror']},'C12 mirror coverage mismatch'
eligible=set()
for f in O.rglob('*'):
 if not f.is_file():continue
 rel=f.relative_to(O)
 if 'formal' in rel.parts:
  if rel.parts[0]=='formal' and f.name in ('config.json','summary.json','training_summary.json','ACCEPTANCE.json','TRAIN_WEIGHTS.json'):eligible.add(str(f))
 elif not any(x in rel.parts for x in ('predictions','__pycache__')) and f.suffix not in ('.pt','.pth','.npy','.pyc') and f.name!='frame_errors.csv':eligible.add(str(f))
assert eligible=={r['origin'] for r in payload['origin']},'O12 eligible origin coverage mismatch'
print(json.dumps({'status':'pass','origin_count':len(payload['origin']),'mirror_count':len(payload['mirror']),'proof_sha256':expected,'local_origin_mirror_hashes':True,'coverage_exact':True,'proof_and_sidecar_both_hosts':True,'read_only':True}))
'''
payload=json.dumps(dict(C=C,O=O,proof=expected,origin=orig,mirror=mir));code=remote_code.replace('PAYLOAD',repr(payload));result=subprocess.check_output(['ssh','zjy-4090','/home/zjy/miniconda3/envs/sparsh/bin/python -'],input=code,text=True)
assert sha(p)==expected,'proof changed during verification'
print(result,end='')
