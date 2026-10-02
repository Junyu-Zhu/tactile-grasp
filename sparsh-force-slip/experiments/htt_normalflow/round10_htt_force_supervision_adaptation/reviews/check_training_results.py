import pathlib,json,hashlib,torch,numpy as np,datetime
R=pathlib.Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round10_htt_force_supervision_adaptation');C=pathlib.Path('/home/zjy/document/tactile-grasp/sparsh-force-slip/experiments/htt_normalflow/round10_htt_force_supervision_adaptation');checked={}
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def check(p,h):
 if str(p) not in checked:checked[str(p)]=sha(p)
 assert checked[str(p)]==h,(str(p),h,checked[str(p)])
def load(p):return json.loads(pathlib.Path(p).read_text())
def walk(o):
 if isinstance(o,dict):
  if 'path' in o and 'sha256' in o:check(o['path'],o['sha256'])
  for k,v in o.items():
   if isinstance(k,str) and k.startswith('/') and isinstance(v,str) and len(v)==64:check(k,v)
   walk(v)
 elif isinstance(o,list):
  for v in o:walk(v)
def equal(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
lock=load(R/'PROTOCOL_LOCK.json');assert lock['status']=='pass';walk(lock)
fa=load(R/'formal_delivery/FORCE_TRAINING_AUDIT.json');fu=load(R/'formal_delivery/TRAINING_AUDIT.json');grid={(f'htt_leave_p{f}',s) for f in range(1,5) for s in (20260914,20260915,20260916)}
for audit in [fa,fu]:
 assert audit['status']=='pass' and audit['expected_count']==12 and audit['test_role_consumed'] is False
 assert len(audit['accepted_runs'])==12 and {(x['fold'],x['seed']) for x in audit['accepted_runs']}==grid;walk(audit)
for row in fa['accepted_runs']:
 assert row['status']=='pass' and row['smoke'] is False;conf=load(pathlib.Path(row['run'])/'config.json');assert not conf['smoke'];assert conf['source_bundle']['files'][str(C/'force/train.py')]==lock['source_hashes'][str(C/'force/train.py')]
for row in fu['accepted_runs']:
 assert row['group']=='F_history_new';run=pathlib.Path(row['training_summary']['path']).parent;conf=load(run/'config.json');assert conf['smoke'] is False;walk(conf);su=load(row['training_summary']['path']);assert not su['smoke'] and su['status']=='complete';walk(su['output_hashes'])
 for n,h in su['output_hashes'].items():check(run/n,h)
 assert conf['sources'][str(C/'fusion/train.py')]==lock['source_hashes'][str(C/'fusion/train.py')]
 prepared=torch.load(conf['prepared']['path'],map_location='cpu',weights_only=False);proof=prepared['force_replacement_audit'];assert proof['pass'] is True and proof['smoke'] is False;walk(proof)
 assert any(z['path']==str(R/f'formal/force/p{row["fold"][-1]}_s{row["seed"]}/best.pth') for z in proof['proofs'])
sa=load(R/'smoke/SMOKE_AUDIT.json');assert sa['status']=='pass';walk(sa);recovery=[]
for prefix,norm in [('force','normalization'),('fusion','normalizer')]:
 for name in ('best.pth','latest.pth'):
  a=torch.load(R/f'smoke/{prefix}_continuous'/name,map_location='cpu',weights_only=False);b=torch.load(R/f'smoke/{prefix}_recovery'/name,map_location='cpu',weights_only=False)
  keys=['model_state','optimizer_state','history',norm]+(['rng_state'] if prefix=='force' else [])
  assert all(equal(a[k],b[k]) for k in keys);recovery.append({'branch':prefix,'checkpoint':name,'keys_exact':keys})
pa=load(R/'smoke/OLD_FORCE_PARITY.json');assert pa['status']=='pass' and len(pa['rows'])==12 and {(f'htt_leave_p{x["fold"]}',x['seed']) for x in pa['rows']}==grid;walk(pa)
assert all(np.isfinite(x['max_abs_N']) and x['max_abs_N']<pa['atol'] for x in pa['rows'])
review=load(C/'reviews/INDEPENDENT_FORCE_ACCEPTANCE_REVIEW.json');assert review['status']=='pass';assert fa['auditor_sha256']==sha(C/'audit/audit_force.py')
print(json.dumps({'status':'pass','scope':'all24 formal artifact chains and source lock; no new training/evaluation','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'force_runs':12,'fusion_runs':12,'full_grid':True,'all_formal_no_smoke':True,'formal_fusion_uses_matching_accepted_force':True,'all_locked_sources_and_required_checks_unchanged':True,'exact_smoke_recovery':recovery,'old_force_parity':{'runs':12,'max_abs_N':max(x['max_abs_N'] for x in pa['rows']),'atol':pa['atol'],'scope':pa['scope'],'checkpoint_identity_checked':True,'inference_not_repeated_in_this_review':True},'checked_hash_count':len(checked),'checked_hashes':checked,'force_auditor_independently_reviewed_by':'r10_eval; reviews/INDEPENDENT_FORCE_ACCEPTANCE_REVIEW.json','review_separation':'this reviewer authored audit_force; current checks validate artifact chain and reference r10_eval independent source/mutation review, not self-approval of audit_force','test_role_consumed':False,'audits':{str(R/'formal_delivery'/n):sha(R/'formal_delivery'/n) for n in ['FORCE_TRAINING_AUDIT.json','TRAINING_AUDIT.json']}}))
