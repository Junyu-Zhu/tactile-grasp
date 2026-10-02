#!/usr/bin/env python3
"""Read-only audit of all formal G2 runs; writes only an R23 receipt."""
import json, pathlib, hashlib, datetime, math, collections
import torch, numpy as np
def equal(a,b):
 if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and torch.equal(a,b)
 if isinstance(a,np.ndarray): return isinstance(b,np.ndarray) and np.array_equal(a,b)
 if isinstance(a,dict): return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)): return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
P=pathlib.Path(__file__).resolve().parent
sha=lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
s=json.loads((P/'G2_QUEUE_STATE.json').read_text()); rows=[]
for r in s['runs']:
 o=pathlib.Path(r['output']); d=json.loads((o/'summary.json').read_text()); c=json.loads((o/'COMMIT.json').read_text()); issues=[]
 if d['status']!='complete' or r['status']!='complete':issues.append('not_complete')
 for k,v in r['expected_identity'].items():
  if d['identity'].get(k)!=v:issues.append('identity:'+k)
 for k in ['best','latest']:
  if sha(o/c[k]['path'])!=c[k]['sha256']:issues.append('commit_hash:'+k)
  if not equal(torch.load(o/c[k]['path'],map_location='cpu',weights_only=False),torch.load(o/(k+'.pth'),map_location='cpu',weights_only=False)):issues.append('alias_content:'+k)
 for k,v in d['changes_at_best'].items():
  if k.startswith(('norm.','pooler.','trunk.')) and v:issues.append('frozen_changed:'+k)
 if not math.isfinite(d['best_metric']):issues.append('nonfinite_selection')
 rows.append({'run':r['run'],'issues':issues,'epochs':d['epochs'],'best_epoch':d['best_epoch'],'best_selection_metric':d['best_metric'],'checkpoint_root':str(o),'wall_seconds':d['wall_seconds'],'attempts':r['attempts']})
result={'checked_at':datetime.datetime.now().astimezone().isoformat(),'passed':len(rows)==48 and all(not r['issues'] for r in rows),'runs':rows,'scope':'Formal identities, exact committed best/latest SHA and semantic alias equality (independent torch serialization need not have equal file SHA), recorded frozen boundary and selection finiteness. Evaluation and final acceptance separate.','group_counts':dict(collections.Counter(r['run'].split('/')[0] for r in rows))}
(P/'ROOT_TRAINING_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':result['passed'],'count':len(rows),'group_counts':result['group_counts'],'issues':[r for r in rows if r['issues']]}))
