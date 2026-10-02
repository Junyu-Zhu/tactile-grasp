#!/usr/bin/env python3
"""Paired native-force diagnostics. No model selection, training or test consumption."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def writecsv(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 p=argparse.ArgumentParser();p.add_argument('--new',type=Path,required=True);p.add_argument('--old',type=Path,required=True);p.add_argument('--support',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--regression',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 nm=json.loads(a.new.read_text());om=json.loads(a.old.read_text());support=json.loads(a.support.read_text());assert nm['status']=='complete' and om['status']=='complete' and nm['fold']==om['fold']==support['fold'] and nm['seed']==om['seed'];assert not nm['smoke'];fold=nm['fold'];seed=nm['seed']; assert nm.get('regression',False)==a.regression
 for key in ('force_checkpoint','force_summary','force_config'):
  assert sha(nm[key]['path'])==nm[key]['sha256']
 config=json.loads(Path(nm['force_config']['path']).read_text());summary=json.loads(Path(nm['force_summary']['path']).read_text());assert config['fold']==summary['fold']==fold and config['seed']==summary['seed']==seed and not config['smoke'] and not summary['smoke'] and summary['status']=='complete'
 assert summary['config_sha256']==nm['force_config']['sha256']
 assert nm['force_checkpoint']['sha256']==summary['best_checkpoint_sha256'] and config['manifest_sha256']==sha(a.support)
 if isinstance(om['force_checkpoint'],dict):assert sha(om['force_checkpoint']['path'])==om['force_checkpoint']['sha256']
 else:assert sha(om['force_checkpoint'])==om['force_checkpoint_sha256']
 oldcp=om['force_checkpoint']['path'] if isinstance(om['force_checkpoint'],dict) else om['force_checkpoint']
 assert str(Path(oldcp))==str(Path(config['r5_checkpoint'])) and sha(oldcp)==config['r5_checkpoint_sha256']
 assert om.get('regression',False)==a.regression
 target_sources={}
 support_entries={e['episode_id']:e for e in support['entries']}
 cp=Path(support['provenance']['contract']);assert sha(cp)==support['provenance']['contract_sha256'];contract=json.loads(cp.read_text());entry={e['episode_id']:e for e in contract['entries']};assert sha(contract['split_manifest'])==contract['split_manifest_sha256'];split=json.loads(Path(contract['split_manifest']).read_text());meta={e['id']:e for e in split['episodes']}
 old={e['episode_id']:e for e in om['entries']};new={e['episode_id']:e for e in nm['entries']};assert len(new)==len(nm['entries'])
 expected={e['episode_id'] for e in contract['entries'] if e['task']==('force' if a.regression else 'slip') and e['roles_by_fold'][fold] in ('train','validation','calibration')};assert set(new)==expected and expected<=set(old)
 rows=[];frame_rows=[];excluded=[]
 for eid in sorted(new):
  ce=entry[eid];role=ce['roles_by_fold'][fold];assert role!='test';ne=new[eid];oe=old[eid];assert sha(ne['prediction_path'])==ne['prediction_sha256'] and sha(oe['prediction_path'])==oe['prediction_sha256'];pn=np.load(ne['prediction_path']);po=np.load(oe['prediction_path']);m=meta[eid]
  if a.regression:
   assert sha(ce['force_native_n_path'])==ce['force_native_n_sha256'];target=np.load(ce['force_native_n_path']);raw=target;stage=np.full(len(target),-1)
  else:
   with np.load(m['path'],allow_pickle=False) as z:raw=np.asarray(z['6d_force'],np.float64)[:,:3]-np.asarray(z['ref_force'],np.float64)[None,:3]
   target=np.clip(raw,-20,20);se=support_entries[eid];assert sha(se['force_native_n_path'])==se['force_native_n_sha256'];locked=np.load(se['force_native_n_path']);assert np.array_equal(target.astype(np.float32),locked);target_sources[eid]={'target_path':se['force_native_n_path'],'target_sha256':se['force_native_n_sha256'],'label_path':ce['label_path'],'label_sha256':ce['label_sha256']};assert sha(ce['label_path'])==ce['label_sha256'];stage=np.load(ce['label_path'])
  assert pn.shape==po.shape==target.shape==(ce['frames'],3) and all(np.isfinite(x).all() for x in (pn,po,target))
  if len(target)<=13:
   excluded.append({'episode_id':eid,'role':role,'frames':len(target),'reason':'no_common_endpoint_t_ge13'});continue
  t=np.arange(13,len(target));populations=[('all',np.ones(len(t),bool))] if a.regression else [('all',np.ones(len(t),bool)),('primary',np.isin(stage[t],[0,2])),('static',stage[t]==0),('incipient',stage[t]==1),('gross',stage[t]==2)]
  for name,pr in [('old',po),('new',pn)]:
   for pop,mask in populations:
    if not mask.any():continue
    ix=t[mask];dy=target[ix]-target[ix-5];dp=pr[ix]-pr[ix-5]
    for j,axis in enumerate(('shear_x','shear_y','normal')):
     y=target[ix,j];pred=pr[ix,j];er=pred-y;de=dp[:,j]-dy[:,j]
     rows.append(dict(fold=fold,seed=seed,variant=name,task='old_force_regression' if a.regression else 'slip_force',role=role,episode_id=eid,leakage_group=m['leakage_group'],probe=m['group'],population=pop,axis=axis,n=len(ix),mae=float(np.abs(er).mean()),rmse=float(np.sqrt(np.mean(er**2))),bias=float(er.mean()),prediction_std=float(pred.std()),target_std=float(y.std()),prediction_mean_abs=float(np.abs(pred).mean()),target_mean_abs=float(np.abs(y).mean()),delta5_mae=float(np.abs(de).mean()),delta5_bias=float(de.mean()),prediction_delta5_std=float(dp[:,j].std()),target_delta5_std=float(dy[:,j].std()),target_clip_fraction=float((np.abs(raw[ix,j])>20).mean()) if not a.regression else None,prediction_outside_clip_fraction=float((np.abs(pred)>20).mean())))
   for ti in t:frame_rows.append(dict(fold=fold,seed=seed,variant=name,role=role,episode_id=eid,t=int(ti),stage=int(stage[ti]),force_mae=float(np.abs(pr[ti]-target[ti]).mean()),delta5_mae=float(np.abs((pr[ti]-pr[ti-5])-(target[ti]-target[ti-5])).mean()),target_xyz=json.dumps(target[ti].tolist()),prediction_xyz=json.dumps(pr[ti].tolist())))
 writecsv(a.output/'per_trial_axis.csv',rows);writecsv(a.output/'frame_errors.csv',frame_rows)
 out={'status':'pass','fold':fold,'seed':seed,'regression':a.regression,'source_sha256':sha(__file__),'new_manifest_sha256':sha(a.new),'old_manifest_sha256':sha(a.old),'support_sha256':sha(a.support),'contract_sha256':sha(cp),'target_sources':target_sources,'same_acquisition':True,'test_consumed':False,'frame_start':13,'base_lag':5,'runs':2,'trials':len(new)-len(excluded),'source_trials':len(new),'evaluated_trials':len(new)-len(excluded),'excluded_no_common_endpoint':excluded,'rows':len(rows),'output_hashes':{n:sha(a.output/n) for n in ('per_trial_axis.csv','frame_errors.csv')},'limits':['HTT slip labels partly force-rule-derived; association is not causality','Old dedicated-force regression targets already clipped; unclipped clipping fraction unavailable','No cross-coordinate pooled physical errors; native HTT axes only']};(a.output/'AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':'pass','rows':len(rows)}))
if __name__=='__main__':main()
