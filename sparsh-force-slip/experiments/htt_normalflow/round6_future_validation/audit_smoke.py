#!/usr/bin/env python3
import argparse,hashlib,json
from pathlib import Path
import torch,numpy as np

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def equal(a,b):
 if type(a)!=type(b):return False
 if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--code',type=Path,required=True);a=p.parse_args();r=a.root;c=a.code;checks={};records={};summaries=[]
 for g in ['A_visual_v2','B_force','C_force_delta','C_resume']:
  f=r/'smoke'/g/'summary.json';d=json.loads(f.read_text());summaries.append(d);records[str(f)]=sha(f)
  checks[g]=d['status']=='complete' and d['smoke'] and not d['formal'] and all(d['checks'][k] is True for k in ['finite_loss_and_all_trainable_gradients','all_parameter_tensors_updated','checkpoint_roundtrip_exact']) and all(d['checks']['parameter_updates_by_tensor'].values())
 current_bundle=hashlib.sha256(json.dumps({p.name:sha(p) for p in [c/'training/train.py',c/'training/protocol.json']},sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
 checks['common_code_bundle']=all(d['run_config']['code_bundle_sha256']==current_bundle for d in summaries)
 a1=torch.load(r/'smoke/C_force_delta/latest.pth',map_location='cpu',weights_only=False);b1=torch.load(r/'smoke/C_resume/latest.pth',map_location='cpu',weights_only=False)
 for key in ['model_state','optimizer_state','history','best_model_state','best_selection_loss','best_epoch','stale','rng_state']:
  checks['actual_resume_exact:'+key]=equal(a1[key],b1[key])
 data=r/'training/prepared.pt';prep=json.loads((r/'training/PREPARE_AUDIT.json').read_text());checks['prepared_unchanged']=sha(data)==prep['output']['sha256'];payload=torch.load(data,map_location='cpu',weights_only=False)
 checks['all_input_tensors_detached']=all(not x['x_C'].requires_grad and not x['y'].requires_grad for x in payload['roles'].values())
 groups={k:set(x['leakage_group']) for k,x in payload['roles'].items()};checks['roles_disjoint']=all(not groups[x]&groups[y] for x in groups for y in groups if x<y)
 checks['causal_history_and_target_identity']=all(payload['role_identities'][k]['endpoint_identity_sha256']==prep['role_identities'][k]['endpoint_identity_sha256'] for k in groups)
 for label,spec in payload['provenance'].items():
  records[spec['path']]=sha(Path(spec['path']));checks['input_unchanged:'+label]=records[spec['path']]==spec['sha256']
 checks['only_future_parameters']=set(a1['model_state'])=={'gru.weight_ih_l0','gru.weight_hh_l0','gru.bias_ih_l0','gru.bias_hh_l0','risk.weight','risk.bias'}
 result={'status':'pass' if all(checks.values()) else 'fail','checks':checks,'source_sha256':records,'upstream_freeze_evidence':'Frozen R5 MAE/force/slip cached outputs are immutable detached inputs; only six future GRU/risk parameter tensors are instantiated and optimized. R5 encoder eval/no-grad identity audit is reused; no new upstream forward/update occurs.','prepared_sha256':sha(data),'smoke_not_formal_parent':True}
 (r/'SMOKE_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'checks':len(checks),'failed':[k for k,v in checks.items() if not v]}));return 0 if all(checks.values()) else 1
if __name__=='__main__':raise SystemExit(main())
