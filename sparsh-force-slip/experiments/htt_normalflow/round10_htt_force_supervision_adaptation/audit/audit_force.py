#!/usr/bin/env python3
"""Read-only acceptance of R10 force outputs before prediction export."""
import argparse,csv,hashlib,json,math,os
from pathlib import Path
import numpy as np
import torch
SEEDS=(20260914,20260915,20260916)
ROLES=('fit','selection','validation','calibration')
PROTOCOL={'optimizer':'AdamW','learning_rate':1e-4,'weight_decay':1e-4,'batch_size':128,'max_epochs':30,'patience':7,'loss':'SmoothL1 on inherited R5-standardized native [shear_x,shear_y,normal] N; beta=1','selection':'earliest strict minimum internal-selection mean per-axis native-N RMSE','target':'clip((6d_force-ref_force)[:3],-20,20) N; shear_x,shear_y,normal'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def state_sha(d):
 h=hashlib.sha256()
 for name,value in sorted(d.items()):
  v=value.detach().cpu().contiguous();h.update(name.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.numpy().tobytes())
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def atomic(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def finite(x):
 if torch.is_tensor(x):assert torch.isfinite(x).all()
 elif isinstance(x,np.ndarray):assert np.isfinite(x).all()
 elif isinstance(x,float):assert math.isfinite(x)
 elif isinstance(x,dict):
  for v in x.values():finite(v)
 elif isinstance(x,(tuple,list)):
  for v in x:finite(v)
def audit_run(run,allow_smoke=False):
 run=Path(run).resolve();su=load(run/'training_summary.json');conf=load(run/'config.json')
 assert su['format']=='round10_force_training_summary_v1' and su['status']=='complete'
 assert conf['smoke'] is False or allow_smoke
 assert su['smoke']==conf['smoke'] and conf['seed'] in SEEDS and conf['fold'] in [f'htt_leave_p{i}' for i in range(1,5)]
 for key,value in PROTOCOL.items():assert conf[key]==value,(key,conf[key],value)
 for k,v in conf.items():assert su[k]==v,('summary/config',k)
 expected={str(run/n):sha(run/n) for n in ('best.pth','latest.pth','config.json')};assert su['output_hashes']==expected
 for name in ('best','latest'):
  assert su[name+'_checkpoint']==str(run/(name+'.pth')) and su[name+'_checkpoint_sha256']==expected[str(run/(name+'.pth'))]
 assert su['config_path']==str(run/'config.json') and su['config_sha256']==expected[str(run/'config.json')]
 for path,h in conf['source_bundle']['files'].items():assert sha(path)==h,('source',path)
 assert sha(conf['manifest_path'])==conf['manifest_sha256'];m=load(conf['manifest_path']);assert m['status']=='pass' and m['fold']==conf['fold'];pv=m['provenance'];assert sha(pv['contract'])==pv['contract_sha256']
 contract=load(pv['contract']);ce={e['episode_id']:e for e in contract['entries']}
 assert sha(pv['r9_prepared'])==pv['r9_prepared_sha256'];r9path=Path(pv['r9_prepared']);assert sha(r9path.with_name('audit.json'))==pv['r9_audit_sha256']
 r9=torch.load(r9path,map_location='cpu',weights_only=False);re={e['episode_id']:e for e in r9['episodes']};assert len(m['entries'])==len(re) and {e['episode_id'] for e in m['entries']}==set(re)
 groups={r:set() for r in ROLES};targets=[];probe_groups={}
 for e in m['entries']:
  src=ce[e['episode_id']];old=re[e['episode_id']];assert src['task']=='slip' and src['roles_by_fold'][conf['fold']]==e['outer_role']==old['role']
  assert e['role'] in (('fit','selection') if e['outer_role']=='train' else (e['outer_role'],))
  assert e['leakage_group']==old['leakage_group'] and e['frames']==src['frames'] and e['token_path']==src['token_path'];groups[e['role']].add(e['leakage_group'])
  if e['outer_role']=='train':probe_groups.setdefault(e['probe'],set()).add(e['leakage_group'])
  assert sha(e['force_native_n_path'])==e['force_native_n_sha256'];y=np.load(e['force_native_n_path'],mmap_mode='r');assert y.shape==(e['frames'],3) and np.isfinite(y).all() and (abs(y)<=20).all()
  tok=np.load(e['token_path'],mmap_mode='r');assert tok.shape==(e['frames'],300,768) and tok.dtype==np.float32
 assert all(groups.values()) and all(groups[a].isdisjoint(groups[b]) for a in ROLES for b in ROLES if a!=b)
 selected=set()
 for gs in probe_groups.values():
  assert len(gs)>=3
  order=sorted(gs,key=lambda g:hashlib.sha256(f'round10-force-internal-v1|{conf["fold"]}|{g}'.encode()).hexdigest());selected.update(order[:min(len(gs)-2,max(1,math.ceil(.2*len(gs))))])
 assert groups['selection']==selected
 oldaudit=load(r9path.parent.parent/f'p{conf["fold"][-1]}_s{conf["seed"]}/audit.json');identity=oldaudit['provenance']['force_checkpoint'];assert conf['r5_checkpoint']==identity['path'] and sha(conf['r5_checkpoint'])==conf['r5_checkpoint_sha256']==identity['sha256']
 assert sha(conf['source_checkpoint'])==conf['source_checkpoint_sha256']==oldaudit['provenance']['source_checkpoint']['sha256']
 old=torch.load(conf['r5_checkpoint'],map_location='cpu',weights_only=False);assert old['format']=='round5_htt_native_force_adapter_v1' and not old['config']['smoke'] and old['config']['fold']==conf['fold'] and old['config']['seed']==conf['seed']
 best=torch.load(run/'best.pth',map_location='cpu',weights_only=False);last=torch.load(run/'latest.pth',map_location='cpu',weights_only=False)
 history=su['history'];finite(history);n=len(history);assert [r['epoch'] for r in history]==list(range(1,n+1)) and su['epochs_completed']==n
 limit=2 if conf['smoke'] else 30;patience=2 if conf['smoke'] else 7;assert 1<=n<=limit
 scores=[r['mean_rmse_xyz_native_n'] for r in history]
 for row in history:assert row['finite_gradients'] is True and math.isclose(row['mean_rmse_xyz_native_n'],np.mean(row['rmse_xyz_native_n']),rel_tol=1e-10,abs_tol=1e-10) and row['train_loss']>=0
 best_idx=min(range(n),key=lambda i:scores[i]);be=best_idx+1;wait=0;running=float('inf')
 for i,score in enumerate(scores):
  if score<running:running=score;wait=0
  else:wait+=1
  assert wait<patience or i==n-1
 assert n==limit or wait>=patience
 assert su['best_epoch']==be and su['best_mean_rmse_xyz_native_n']==scores[best_idx]
 assert su['initial_state_sha256']==state_sha(old['model_state']) and su['selected_state_sha256']==state_sha(best['model_state'])
 keys=list(old['model_state']);assert all(k.startswith(('force_pooler.','force_trunk.','force_head.')) for k in keys)
 boundary=su['provenance']['parameter_boundary'];assert boundary['trainable_names']==keys and boundary['trainable_tensors']==len(keys) and boundary['trainable_parameters']==sum(x.numel() for x in old['model_state'].values()) and boundary['frozen_parameters']==0
 batches=1 if conf['smoke'] else math.ceil(sum(e['frames']-5 for e in m['entries'] if e['role']=='fit')/conf['batch_size'])
 for cp,epoch in ((best,be),(last,n)):
  assert cp['format']=='round10_htt_native_force_adapter_v1' and cp['config']==conf and cp['provenance']==su['provenance'];assert cp['epoch']==epoch and cp['history']==history[:epoch]
  assert cp['best_epoch']==be and cp['best_metric']==scores[best_idx]
  assert cp['patience_wait']==(0 if cp is best else wait)
  assert list(cp['model_state'])==keys;finite(cp['model_state']);finite(cp['optimizer_state'])
  for k in keys:assert cp['model_state'][k].shape==old['model_state'][k].shape
  for axis in ('mean','std'):
   v=np.asarray(cp['normalization'][axis]);assert v.shape==(3,) and np.isfinite(v).all() and np.array_equal(v,np.asarray(old['normalization'][axis])) and np.array_equal(v,np.asarray(su['normalization'][axis],dtype=v.dtype))
   if axis=='std':assert (v>0).all()
  opt=cp['optimizer_state'];assert len(opt['param_groups'])==1;pg=opt['param_groups'][0];assert pg['lr']==conf['learning_rate'] and pg['weight_decay']==conf['weight_decay'] and len(pg['params'])==len(keys) and set(pg['params'])==set(opt['state'])
  for pid,k in zip(pg['params'],keys):
   st=opt['state'][pid];assert st['exp_avg'].shape==old['model_state'][k].shape and st['exp_avg_sq'].shape==old['model_state'][k].shape and float(st['step'])==epoch*batches
  assert {'python','numpy','torch','loader_generator','cuda'}==set(cp['rng_state'])
 for comp in ('force_pooler','force_trunk','force_head'):
  assert any(not torch.equal(old['model_state'][k],best['model_state'][k]) for k in keys if k.startswith(comp+'.')) and su['trainable_component_changed'][comp] is True
 for key in ('checkpoint_restore_exact','force_parameters_changed','finite_gradients_all_epochs','upstream_unchanged','encoder_not_instantiated_cached_tokens_only'):assert su[key] is True
 for key in ('encoder_in_optimizer','slip_in_optimizer','force_output_head_reinitialized'):assert su['provenance'][key] is False
 assert su['provenance']['r5_full_state_loaded'] is True and su['provenance']['internal_roles_disjoint'] is True
 for key in ('fit','selection'):assert su['provenance'][key+'_groups']==sorted(groups[key])
 return {'status':'smoke_review_pass' if conf['smoke'] else 'pass','fold':conf['fold'],'seed':conf['seed'],'smoke':conf['smoke'],'run':str(run),'epochs':n,'best_epoch':be,'role_groups':{k:len(v) for k,v in groups.items()},'checkpoints':expected,'summary':{'path':str(run/'training_summary.json'),'sha256':sha(run/'training_summary.json')},'input_manifest':{'path':conf['manifest_path'],'sha256':conf['manifest_sha256']},'source_bundle':conf['source_bundle'],'test_role_consumed':False,'encoder_cached_no_new_update':True,'inherited_normalization_exact':True,'initial_R5_state_exact':True,'best_selection_recomputed':True,'optimizer_steps_exact':True,'auditor_sha256':sha(__file__)}
def main():
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--run',type=Path);g.add_argument('--root',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
 if a.run:
  result=audit_run(a.run);atomic(a.output or a.run/'ACCEPTANCE.json',result);print(json.dumps({'status':result['status'],'run':str(a.run)}));return
 paths=sorted((a.root/'formal/force').glob('*/training_summary.json'));rows=[audit_run(p.parent) for p in paths];grid={(f'htt_leave_p{f}',s) for f in range(1,5) for s in SEEDS};assert len(rows)==12 and {(r['fold'],r['seed']) for r in rows}==grid
 out=a.output or a.root/'formal_delivery/FORCE_TRAINING_AUDIT.json';atomic(out,{'status':'pass','expected_count':12,'accepted_runs':rows,'test_role_consumed':False,'auditor_sha256':sha(__file__)})
 index=out.with_name('FORCE_CHECKPOINT_INDEX.csv');index.parent.mkdir(parents=True,exist_ok=True)
 with index.open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['fold','seed','artifact','path','sha256'])
  for row in rows:
   for path,h in row['checkpoints'].items():w.writerow([row['fold'],row['seed'],Path(path).name,path,h])
 print(json.dumps({'status':'pass','runs':len(rows),'output':str(out)}))
if __name__=='__main__':main()
