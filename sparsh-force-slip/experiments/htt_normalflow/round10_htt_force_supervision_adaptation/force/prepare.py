#!/usr/bin/env python3
"""Train-only grouped force support preparation; small GT arrays, existing complete tokens."""
import argparse,json,hashlib,sys,os
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
R5CODE=HERE.parents[1]/'round5_force_conditioned_slip/current'
sys.path.insert(0,str(R5CODE))
from common import sha256_file as _sha,atomic_json
def sha(p):return _sha(Path(p))
ROOT=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow')
POLICY={'selection_fraction':0.2,'selection':'per probe SHA256(round10-force-internal-v1|fold|leakage_group) ascending; first ceil(0.2*n), at least one and retain at least two fit groups','min_frame':5,'seed_independent':True}
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 cp=ROOT/'round5_force_conditioned_slip/data/contract/contract.json';c=json.loads(cp.read_text());split=json.loads(Path(c['split_manifest']).read_text());assert sha(c['split_manifest'])==c['split_manifest_sha256'];meta={e['id']:e for e in split['episodes']};entries={e['episode_id']:e for e in c['entries']}
 ca=ROOT/'round5_force_conditioned_slip/data/CACHE_HASH_AUDIT.json';audit=json.loads(ca.read_text());assert audit['status']=='pass' and audit['cache_manifest_sha256']==sha(cp)
 gt={};runs=[]
 for f in range(1,5):
  fold=f'htt_leave_p{f}';pp=ROOT/f'round9_htt_temporal_force_fusion/prepare/p{f}_s20260914/prepared.pt';ap=pp.with_name('audit.json');au=json.loads(ap.read_text());assert au['status']=='pass' and sha(pp)==au['output_hashes'][str(pp)];pr=torch.load(pp,map_location='cpu',weights_only=False)
  eps=pr['episodes'];groups={}
  for e in eps:
   if e['role']=='train':groups.setdefault(e['probe'],set()).add(e['leakage_group'])
  sel=set()
  for probe,gs in groups.items():
   assert len(gs)>=3
   order=sorted(gs,key=lambda g:hashlib.sha256(f'round10-force-internal-v1|{fold}|{g}'.encode()).hexdigest());k=min(len(gs)-2,max(1,int(np.ceil(.2*len(gs)))));sel.update(order[:k])
  out=[]
  for e in eps:
   eid=e['episode_id'];ce=entries[eid];m=meta[eid];assert ce['task']=='slip' and ce['roles_by_fold'][fold]==e['role'];assert e['role'] in ('train','validation','calibration')
   if eid not in gt:
    with np.load(e['source_path'],allow_pickle=False) as z:force=np.asarray(z['6d_force'],np.float64);ref=np.asarray(z['ref_force'],np.float64)
    assert force.shape==(ce['frames'],6) and ref.shape==(6,) and np.isfinite(force).all() and np.isfinite(ref).all()
    raw=force[:,:3]-ref[None,:3];y=np.clip(raw,-20,20).astype(np.float32);dst=a.output/'targets'/f'{hashlib.sha256(eid.encode()).hexdigest()}.npy';dst.parent.mkdir(exist_ok=True);np.save(dst,y)
    gt[eid]={'path':str(dst),'sha256':sha(dst),'source_path':e['source_path'],'force_array_sha256':hashlib.sha256(force.tobytes()).hexdigest(),'reference_sha256':hashlib.sha256(ref.tobytes()).hexdigest(),'raw_source_prior_sha256':m['source_files'][e['source_path']],'clip_fraction':(np.abs(raw[5:])>20).mean(axis=0).tolist()}
   tok=np.load(ce['token_path'],mmap_mode='r');assert tok.shape==(ce['frames'],300,768) and tok.dtype==np.float32
   role=('selection' if e['leakage_group'] in sel else 'fit') if e['role']=='train' else e['role']
   out.append({'episode_id':eid,'leakage_group':e['leakage_group'],'probe':e['probe'],'role':role,'outer_role':e['role'],'frames':ce['frames'],'token_path':ce['token_path'],'token_sha256_previously_audited':ce['token_sha256'],'token_size':Path(ce['token_path']).stat().st_size,'force_native_n_path':gt[eid]['path'],'force_native_n_sha256':gt[eid]['sha256'],'label_path':ce['label_path'],'source_path':e['source_path']})
  sets={r:{e['leakage_group'] for e in out if e['role']==r} for r in ('fit','selection','validation','calibration')}
  assert all(sets[r] for r in sets) and all(not(sets[r]&sets[s]) for r in sets for s in sets if r!=s)
  coverage={}
  for role in sets:
   es=[e for e in out if e['role']==role];ys=np.concatenate([np.load(e['force_native_n_path'])[5:] for e in es]);stages=np.concatenate([np.load(e['label_path'])[5:] for e in es]);assert ys.shape==(len(stages),3) and np.isfinite(ys).all();coverage[role]={'frames':len(ys),'trials':len(es),'stage_counts':{str(i):int((stages==i).sum()) for i in (0,1,2)},'target_mean':ys.mean(0).tolist(),'target_std':ys.std(0).tolist(),'target_abs_max':np.abs(ys).max(0).tolist()}
  norm=np.concatenate([np.load(e['force_native_n_path'])[5:] for e in out if e['role']=='fit']);fn=a.output/f'fold_p{f}.json';obj={'status':'pass','fold':fold,'policy':POLICY,'entries':out,'group_counts':{k:len(v) for k,v in sets.items()},'coverage':coverage,'fit_normalization_diagnostic_only':{'mean':norm.mean(0).tolist(),'std':np.maximum(norm.std(0),1e-6).tolist()},'provenance':{'contract':str(cp),'contract_sha256':sha(cp),'cache_audit_sha256':sha(ca),'r9_prepared':str(pp),'r9_prepared_sha256':sha(pp),'r9_audit_sha256':sha(ap)}};atomic_json(fn,obj);runs.append({'fold':fold,'manifest':str(fn),'sha256':sha(fn),'group_counts':obj['group_counts']})
 atomic_json(a.output/'PREPARE_AUDIT.json',{'status':'pass','source_sha256':sha(__file__),'policy':POLICY,'runs':runs,'targets':gt,'full_tokens_reused':True,'raw_images_read':False,'token_full_hash_reused':True,'test_consumed':False});print(json.dumps(runs))
if __name__=='__main__':main()
