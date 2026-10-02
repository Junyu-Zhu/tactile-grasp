import json,hashlib,sys,collections
from pathlib import Path
import numpy as np,torch
O=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round16_touchd_force_transfer');C=Path('/home/zjy/document/tactile-grasp/sparsh-force-slip/experiments/htt_normalflow/round16_touchd_force_transfer');S=(20260914,20260915,20260916)
sys.path.insert(0,str(C));from touchd_common import fresh_adapter,state_sha256,SOURCE_CHECKPOINT

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def equal(a,b):
 if isinstance(a,torch.Tensor):return torch.equal(a,b)
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
report={'checks':{},'replay':{}}
idx=json.loads((O/'CHECKPOINT_INDEX.json').read_text()); assert len(idx['entries'])==75
counts=collections.Counter(); norms={}; hist=0; hashes=0
for ent in idx['entries']:
 stage=ent['stage'];counts[stage]+=1
 su=json.loads(Path(ent['summary']).read_text()); assert su['status']=='complete'; assert not su.get('smoke',False);assert sha(ent['summary'])==ent['summary_sha256']
 best=torch.load(ent['best'],map_location='cpu',weights_only=False);latest=torch.load(ent['latest'],map_location='cpu',weights_only=False)
 for k in ('best','latest'):assert sha(ent[k])==ent[k+'_sha256'];hashes+=1
 history=latest['history']; field='validation_low_fpr_auc' if stage=='slip' else 'selection_mae' if stage=='future' else 'selection_mean_rmse'
 vals=[x[field] for x in history];assert np.isfinite(vals).all(); assert all(np.isfinite([v for k,v in row.items() if isinstance(v,(int,float))]).all() for row in history)
 pick=int(np.argmax(vals) if stage=='slip' else np.argmin(vals));assert best['epoch']==history[pick]['epoch']==su['best_epoch'];assert best['history']==history[:pick+1]; hist+=1
 if stage in ('touchd','htt_force'):
  conf=best['config'];assert conf['source_checkpoint_sha256']=='850e4a74e6d9bd60e8ed22efc8b70d343c5d21bb5f60b3053b0a800674dffcb4' and not conf['smoke']
  if stage=='htt_force':
   fold=int(conf['fold'][-1]);seed=conf['seed'];route=conf['route'];norms[route,fold,seed]=best['normalization']; assert sha(conf['manifest_path'])==conf['manifest_sha256']
   if route=='H':assert conf['t_private'] is None
   else:
    assert conf['t_private']['path']==str(O/f'formal/touchd/s{seed}/transferable_private_state.pth');assert sha(conf['t_private']['path'])==conf['t_private']['sha256']
for f in range(1,5):
 sup=json.loads((O.parent/f'round10_htt_force_supervision_adaptation/force_support/fold_p{f}.json').read_text());y=np.concatenate([np.load(e['force_native_n_path'])[5:] for e in sup['entries'] if e['role']=='fit']).astype(np.float32)
 for s in S:
  assert equal(norms['H',f,s],norms['T_H',f,s]);assert np.array_equal(norms['H',f,s]['mean'],y.mean(0));assert np.array_equal(norms['H',f,s]['std'],y.std(0).clip(1e-6))
report['checks'].update(grid=dict(counts),checkpoint_sha256_recomputed=hashes,earliest_best_from_latest_history=hist,htt_fit_only_normalization_pairs=12)
assert sha(SOURCE_CHECKPOINT)=='850e4a74e6d9bd60e8ed22efc8b70d343c5d21bb5f60b3053b0a800674dffcb4'
# Independent source initialization and transferred states; shared initial hashes across folds.
for s in S:
 tpath=O/f'formal/touchd/s{s}/transferable_private_state.pth';t=torch.load(tpath,map_location='cpu',weights_only=False);assert set(t)=={'force_pooler','force_trunk'}
 tb=torch.load(O/f'formal/touchd/s{s}/best.pth',map_location='cpu',weights_only=False)['model_state']
 for part,state in t.items():assert all(torch.equal(v,tb[part+'.'+k]) for k,v in state.items())
 heads=[]
 for route,private in (('H',None),('T_H',t)):
  m=fresh_adapter(s,trunk_state=private);initial=state_sha256(m.state_dict());head=state_sha256(m.force_head.state_dict());heads.append(head)
  for f in range(1,5):
   su=json.loads((O/f'formal/htt_force/{route}_p{f}_s{s}/training_summary.json').read_text());assert su['initial_state_sha256']==initial;assert su['reset_head_sha256']==head
  if route=='H':assert json.loads((O/f'formal/touchd/s{s}/summary.json').read_text())['initial_state_sha256']==initial
 assert heads[0]==heads[1]
report['checks']['source_init_and_matching_seed_private_and_symmetric_reset']=True
# T cache metadata/fit targets: no full dataset rehash.
cm=json.loads((O/'cache/cache_manifest.json').read_text());assert len(cm['entries'])==142 and cm['samples']==80783
obj={}; total=0;fit=[]
for e in cm['entries']:
 i=e['identity'];total+=i['samples'];obj.setdefault(i['object'],set()).add(i['role']); assert np.load(e['token_path'],mmap_mode='r').shape==(i['samples'],300,768);assert np.load(e['token_path'],mmap_mode='r').dtype==np.float16
 if i['role']=='fit':fit.append(np.load(e['target_path']))
assert all(len(x)==1 for x in obj.values());assert collections.Counter(next(iter(v)) for v in obj.values())=={'fit':57,'selection':14}
order=sorted(obj,key=lambda n:hashlib.sha256(f'round16-touchd-object-role-v1|obj{n:03d}'.encode()).hexdigest());assert {n for n in obj if obj[n]=={'selection'}}==set(order[:14])
y=torch.tensor(np.concatenate(fit));n={'mean':y.mean(0),'std':y.std(0,unbiased=False).clamp_min(1e-6)}
for s in S:
 ck=torch.load(O/f'formal/touchd/s{s}/best.pth',map_location='cpu',weights_only=False);assert all(np.array_equal(np.asarray(ck['normalization'][k]),v.numpy()) for k,v in n.items())
for e in (cm['entries'][0],cm['entries'][-1]):
 assert sha(e['token_path'])==e['token_sha256'] and sha(e['target_path'])==e['target_sha256']
report['checks']['touchd_cache_shapes_all142_split_norm_and_two_shard_sha']=True
# Readonly CPU force replay of two cached tokens in each route.
sup=json.loads((O.parent/'round10_htt_force_supervision_adaptation/force_support/fold_p1.json').read_text());e=next(e for e in sup['entries'] if e['role']=='validation');x=torch.from_numpy(np.array(np.load(e['token_path'],mmap_mode='r')[[13,100]],copy=True)).float()
for route in ('H','T_H'):
 key=f'{route}_p1_s{S[0]}';ck=torch.load(O/f'formal/htt_force/{key}/best.pth',map_location='cpu',weights_only=False);m.load_state_dict(ck['model_state']);m.eval();before=state_sha256(m.state_dict());n=ck['normalization']
 with torch.inference_mode():pred=(m(x)*torch.tensor(n['std'])+torch.tensor(n['mean'])).numpy()
 pm=json.loads((O/f'formal/predictions/{key}/prediction_manifest.json').read_text());entry=next(z for z in pm['entries'] if z['episode_id']==e['episode_id']);ref=np.load(entry['prediction_path'])[[13,100]];err=float(np.max(np.abs(pred-ref)));assert err<1e-4 and state_sha256(m.state_dict())==before;report['replay'][route+'_cpu_max_abs_native_n']=err
# Real interrupted-resume artifacts (identical fields, excluding run identity).
paths=list((O/'smoke').glob('sharded*/latest.pth'));report['checks']['sharded_resume_paths']=[str(p) for p in paths]
if len(paths)==2:
 a,b=[torch.load(p,map_location='cpu',weights_only=False) for p in paths]
 for k in ('model_state','optimizer_state','rng_state','history','best_metric','best_epoch','wait','normalization'):assert equal(a[k],b[k])
 report['checks']['sharded_resume_exact_from_artifacts']=True
print(json.dumps(report,indent=2))
