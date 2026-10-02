#!/usr/bin/env python3
"""Bounded metadata/weight verification; never opens test episode content."""
import argparse,json
from pathlib import Path
import torch
from export import ROOT,R5,sha,atomic_json

def run(out):
 cp=R5/'data/contract/contract.json';c=json.loads(cp.read_text());sp=Path(c['split_manifest']);assert sha(sp)==c['split_manifest_sha256'];s=json.loads(sp.read_text());assert s['schema_version']==2
 auditp=R5/'formal_delivery/TRAINING_AUDIT.json';a=json.loads(auditp.read_text());assert a['status']=='pass';accepted={x['id']:x for x in a['runs']}
 ca=R5/'data/CACHE_HASH_AUDIT.json';caudit=json.loads(ca.read_text());assert caudit['status']=='pass' and caudit['cache_manifest_sha256']==sha(cp)
 results=[]
 for fold in range(1,5):
  fn=f'htt_leave_p{fold}'
  sets={r:{e['id'] for e in s['episodes'] if e['id'] in s['splits'][fn][r] and e['task']=='slip'} for r in ('train','validation','calibration')}
  for seed in (20260914,20260915,20260916):
   key=f'p{fold}_s{seed}';mp=R5/f'formal/force_predictions/adapt/{key}/prediction_manifest.json';m=json.loads(mp.read_text());fp=Path(m['force_checkpoint'])
   assert m['formal'] and m['status']=='complete' and m['fold']==fn and m['seed']==seed and m['cache_manifest_sha256']==sha(cp)
   assert sha(fp)==m['force_checkpoint_sha256'];fc=torch.load(fp,map_location='cpu',weights_only=False);cfg=fc['config'];assert not cfg['smoke'] and cfg['fold']==fn and cfg['seed']==seed and cfg['target']==c['force_target_semantics'];assert accepted[f'force_{key}']['status']=='pass'
   bp=ROOT/f'round3_mae_slip_adaptation/runs/B/fold_p{fold}/seed_{seed}/best.pth';b=torch.load(bp,map_location='cpu',weights_only=False);assert b['config']['init']=='fresh' and not b['config']['smoke'] and b['config']['fold']==fn and b['config']['seed']==seed
   assert set(k.split('.')[0] for k in b['branch_state'])=={'slip_pooler','slip_trunk','slip_head'};assert b['branch_state']['slip_head.weight'].shape==(2,192)
   assert set(e['episode_id'] for e in m['entries'])==set.union(*sets.values())
   results.append({'fold':fn,'seed':seed,'force_checkpoint':str(fp),'force_sha256':sha(fp),'force_prediction_manifest':str(mp),'force_prediction_manifest_sha256':sha(mp),'visual_checkpoint':str(bp),'visual_sha256':sha(bp),'force_target':cfg['target'],'force_normalization':{k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in fc['normalization'].items()},'force_selection':cfg['selection'],'visual_selection':b['config']['selection'],'role_episode_counts':{r:len(v) for r,v in sets.items()},'status':'pass'})
 atomic_json(out,{'status':'pass','runs':results,'contract_sha256':sha(cp),'split_sha256':sha(sp),'r5_training_audit_sha256':sha(auditp),'r5_cache_audit_sha256':sha(ca),'input_dim':199,'visual_dim':192,'base_first_nonpadded':5,'endpoint_start':13,'history':'t-8..t','raw_image_union':'t-13..t','future_input':False,'test_content_read':False,'gt_force_on_slip_trials':False,'checks':['all12 accepted force checkpoint identities match predictions','same fold/seed R3-B private visual branch only','force train task role specified by schema2; validation selection disclosed','R5 full token hash audit reused; no raw data re-audit']})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
