import json,pathlib,hashlib,math,numpy as np
R=pathlib.Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round10_htt_force_supervision_adaptation')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
p=R/'force_support/PREPARE_AUDIT.json';a=json.loads(p.read_text());assert a['status']=='pass'
records=[]
for run in a['runs']:
 m=json.loads(pathlib.Path(run['manifest']).read_text());assert sha(run['manifest'])==run['sha256'];c=json.loads(pathlib.Path(m['provenance']['contract']).read_text());ce={e['episode_id']:e for e in c['entries']};selection=set();gs={}
 for e in m['entries']:
  assert ce[e['episode_id']]['roles_by_fold'][m['fold']]==e['outer_role']
  assert e['role'] in ('fit','selection') if e['outer_role']=='train' else e['role']==e['outer_role']
  if e['outer_role']=='train':gs.setdefault(e['probe'],set()).add(e['leakage_group'])
 for probe,groups in gs.items():
  order=sorted(groups,key=lambda g:hashlib.sha256(f'round10-force-internal-v1|{m["fold"]}|{g}'.encode()).hexdigest());k=min(len(groups)-2,max(1,math.ceil(.2*len(groups))));selection.update(order[:k])
 assert selection=={e['leakage_group'] for e in m['entries'] if e['role']=='selection'}
 for role in ('fit','selection','validation','calibration'):
  es=[e for e in m['entries'] if e['role']==role];n=0
  for e in es:
   y=np.load(e['force_native_n_path']);assert y.shape==(e['frames'],3) and np.isfinite(y).all() and (np.abs(y)<=20).all();assert sha(e['force_native_n_path'])==e['force_native_n_sha256'];n+=len(y)-5
  assert n==m['coverage'][role]['frames']; assert len(es)==m['coverage'][role]['trials']
 records.append({'fold':m['fold'],'counts':m['group_counts'],'coverage':m['coverage'],'manifest_sha256':sha(run['manifest'])})
e=m['entries'][0]
with np.load(e['source_path']) as z:y=np.clip((z['6d_force']-z['ref_force'])[:,:3],-20,20).astype(np.float32)
assert np.array_equal(y,np.load(e['force_native_n_path']))
print(json.dumps({'status':'pass','folds':records,'same_npz_gt_exact_example':e['episode_id'],'prepare_audit_sha256':sha(p)}))
