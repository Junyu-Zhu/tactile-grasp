#!/usr/bin/env python3
"""Export frozen R3-B visual trunk + role-matched R5 force; no MAE recompute."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,sys
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
EXP=HERE.parents[1]
sys.path.insert(0,str(EXP/'round5_force_conditioned_slip/current'))
from sources import load_decoupled_decoder,load_r3_slip_branch
from models import FrozenSlipBranch
from common import configure_determinism
import sources,models,common
ROOT=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow')
R5=ROOT/'round5_force_conditioned_slip'
ROLES=('train','validation','calibration')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def atomic_json(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(obj,indent=2,ensure_ascii=False));os.replace(q,p)
def windows(v,f):
 """No padding, no future input, common full signed force history."""
 if len(v)!=len(f) or v.shape[1]!=192 or f.shape[1]!=3:raise ValueError('base shape')
 ts=np.arange(13,len(v));ix=ts[:,None]+np.arange(-8,1)[None,:]
 x=np.zeros((len(ts),9,199),np.float32);x[:,:,:192]=v[ix];x[:,:,192:195]=f[ix]
 x[:,5:,195:198]=f[ix[:,5:]]-f[ix[:,5:]-5];x[:,5:,198]=1
 return ts,x

def run(a):
 configure_determinism();fold=f'htt_leave_p{a.fold}';key=f'p{a.fold}_s{a.seed}'
 out=a.output/key;out.mkdir(parents=True,exist_ok=True)
 cp=R5/'data/contract/contract.json';ca=R5/'data/CACHE_HASH_AUDIT.json'
 c=json.loads(cp.read_text());audit=json.loads(ca.read_text());assert audit['status']=='pass' and audit['cache_manifest_sha256']==sha(cp)
 splitp=Path(c['split_manifest']);assert sha(splitp)==c['split_manifest_sha256'];split=json.loads(splitp.read_text());assert split['schema_version']==2
 rows={e['id']:e for e in split['episodes']}
 mp=R5/f'formal/force_predictions/adapt/{key}/prediction_manifest.json';m=json.loads(mp.read_text())
 assert m['status']=='complete' and m['formal'] and m['variant']=='adapt' and m['fold']==fold and m['seed']==a.seed
 assert m['cache_manifest_sha256']==sha(cp)
 fp=Path(m['force_checkpoint']);assert sha(fp)==m['force_checkpoint_sha256']
 accepted_path=R5/'formal_delivery/TRAINING_AUDIT.json';accepted=json.loads(accepted_path.read_text());assert accepted['status']=='pass'
 assert any(r['id']==f'force_{key}' and r['status']=='pass' and r['best_checkpoint']==str(fp) for r in accepted['runs'])
 force_payload=torch.load(fp,map_location='cpu',weights_only=False);fc=force_payload['config']
 assert not fc['smoke'] and fc['fold']==fold and fc['seed']==a.seed and fc['target']==c['force_target_semantics']
 source=Path(m['source_checkpoint']);assert sha(source)==m['source_checkpoint_sha256']
 bp=ROOT/f'round3_mae_slip_adaptation/runs/B/fold_p{a.fold}/seed_{a.seed}/best.pth'
 bc=torch.load(bp,map_location='cpu',weights_only=False);assert bc['config']['fold']==fold and bc['config']['seed']==a.seed and not bc['config']['smoke']
 assert bc['config']['source_checkpoint_sha256']==sha(source)
 provenance={'accepted_training_audit':{'path':str(accepted_path),'sha256':sha(accepted_path)},'contract':{'path':str(cp),'sha256':sha(cp)},'cache_audit':{'path':str(ca),'sha256':sha(ca)},'split':{'path':str(splitp),'sha256':sha(splitp)},'force_prediction_manifest':{'path':str(mp),'sha256':sha(mp)},'force_checkpoint':{'path':str(fp),'sha256':sha(fp)},'visual_checkpoint':{'path':str(bp),'sha256':sha(bp)},'source_checkpoint':{'path':str(source),'sha256':sha(source)},'source_code_sha256':sha(__file__),'dependency_hashes':{str(Path(m.__file__).resolve()):sha(m.__file__) for m in (sources,models,common,sources.p2)}}
 done=out/'audit.json'
 if done.exists():
  d=json.loads(done.read_text())
  if d.get('provenance')!=provenance:raise RuntimeError('existing cache identity mismatch')
  for p,h in d['output_hashes'].items():assert sha(p)==h
  print(json.dumps({'status':'reused','path':str(done)}));return
 decoder,_=load_decoupled_decoder(source);model=FrozenSlipBranch(load_r3_slip_branch(decoder,bp)).eval().requires_grad_(False).to(a.device)
 before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 pred={e['episode_id']:e for e in m['entries']};bases=[];buf={r:[] for r in ROLES};inputs=[]
 selected=[e for e in c['entries'] if e['task']=='slip' and e['roles_by_fold'][fold] in ROLES]
 for e in sorted(selected,key=lambda e:e['episode_id']):
  eid=e['episode_id'];role=e['roles_by_fold'][fold];meta=rows[eid]
  assert eid in split['splits'][fold][role] and meta['task']=='slip'
  p=pred[eid];assert p['roles_by_fold']==e['roles_by_fold'] and p['frames']==e['frames']
  fpath=Path(p['prediction_path']);assert sha(fpath)==p['prediction_sha256'];f=np.load(fpath,allow_pickle=False)
  lp=Path(e['label_path']);assert sha(lp)==e['label_sha256'];labels=np.load(lp,allow_pickle=False)
  tok=np.load(e['token_path'],mmap_mode='r',allow_pickle=False);assert tok.shape==(e['frames'],300,768) and tok.dtype==np.float32
  assert f.shape==(len(tok),3) and np.isfinite(f).all() and np.isin(labels,[0,1,2]).all()
  chunks=[]
  with torch.inference_mode():
   for start in range(0,len(tok),a.batch_size):
    chunks.append(model.forward_features(torch.from_numpy(np.array(tok[start:start+a.batch_size],copy=True)).to(a.device)).cpu().numpy())
  v=np.concatenate(chunks).astype(np.float32);assert np.isfinite(v).all()
  ts,x=windows(v,f);stage=labels[ts].astype(np.int64)
  bases.append({'episode_id':eid,'leakage_group':meta['leakage_group'],'probe':meta['group'],'role':role,'visual':torch.from_numpy(v),'force':torch.from_numpy(f.copy()),'stage':torch.from_numpy(labels.astype(np.int64)),'token_path':e['token_path'],'source_path':meta['path']})
  buf[role].append((x,ts,stage,eid,meta['leakage_group'],meta['group']))
  inputs.append({'episode_id':eid,'role':role,'token_path':e['token_path'],'token_sha256_previously_audited':e['token_sha256'],'token_shape':list(tok.shape),'token_size':Path(e['token_path']).stat().st_size,'label_sha256':sha(lp),'force_prediction_sha256':sha(fpath)})
  print(f'{key} {eid} {len(ts)}',flush=True)
 roles={};hashes={};counts={}
 for role,items in buf.items():
  x=np.concatenate([z[0] for z in items]);ts=np.concatenate([z[1] for z in items]);stage=np.concatenate([z[2] for z in items])
  ids=[z[3] for z in items for _ in z[1]];groups=[z[4] for z in items for _ in z[1]];probes=[z[5] for z in items for _ in z[1]]
  roles[role]={'x':torch.from_numpy(x),'t':torch.from_numpy(ts),'stage':torch.from_numpy(stage),'episode_id':ids,'leakage_group':groups,'probe':probes}
  ep=out/f'endpoints_{role}.csv'
  with ep.open('w') as f:
   w=csv.writer(f);w.writerow(['episode_id','t','stage','leakage_group','fold','role','probe']);w.writerows(zip(ids,ts,stage,groups,[fold]*len(ts),[role]*len(ts),probes))
  hashes[str(ep)]=sha(ep);counts[role]={'rows':len(ts),'trials':len(items),'stage_counts':{str(i):int((stage==i).sum()) for i in (0,1,2)}}
 for r in ROLES:
  for s in ROLES:
   if r!=s:assert not(set(roles[r]['leakage_group'])&set(roles[s]['leakage_group']))
 assert all(torch.equal(before[k],v.cpu()) for k,v in model.state_dict().items())
 payload={'schema':'round9_htt_temporal_prepared_v1','format':'round9_htt_temporal_prepared_v1','fold':fold,'seed':a.seed,'roles':roles,'episodes':bases,'provenance':provenance,'input_layout':{'visual':[0,192],'force':[192,195],'delta':[195,198],'valid':198},'force_target_normalization':force_payload['normalization'],'upstream_audit':{'frozen_force':True,'no_optimizer':True,'frozen_visual':True,'no_force_visual_path':True,'role_group_disjoint':True,'base_start':5,'endpoint_start':13,'gt_force_not_input':True}}
 dst=out/'prepared.pt';tmp=out/'prepared.tmp';torch.save(payload,tmp);os.replace(tmp,dst);hashes[str(dst)]=sha(dst)
 atomic_json(done,{'status':'pass','fold':fold,'seed':a.seed,'provenance':provenance,'counts':counts,'inputs':inputs,'output_hashes':hashes,'checks':{'frozen_force':True,'no_optimizer':True,'frozen_visual':True,'no_force_visual_path':True,'role_group_disjoint':True,'current_and_past_only':True,'base_start':5,'endpoint_start':13,'raw_image_union':'t-13..t','gt_force_not_input':True},'gt_force_diagnostic':'slip episodes have no paired force ground truth in audited contract; no basename joins','token_hash_policy':'Reuse R5 full hash audit, verify current shape; current size recorded, no historical size comparison or full token rehash'})
 print(json.dumps({'status':'pass','path':str(done),'counts':counts}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=range(1,5),required=True);p.add_argument('--seed',type=int,choices=[20260914,20260915,20260916],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--batch-size',type=int,default=128);run(p.parse_args())
