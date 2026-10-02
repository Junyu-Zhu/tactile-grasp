#!/usr/bin/env python3
import argparse,csv,json,hashlib,zipfile
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
def arrsha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def writecsv(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))));w.writeheader();w.writerows(rows)
def header(path,name):
 with zipfile.ZipFile(path) as z:
  with z.open(name+'.npy') as f:
   version=np.lib.format.read_magic(f)
   return np.lib.format._read_array_header(f,version)[0]
def main():
 p=argparse.ArgumentParser();p.add_argument('--r5',type=Path,required=True);p.add_argument('--precheck',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 cpath=a.r5/'data/contract/contract.json';c=json.loads(cpath.read_text());sfile=Path(c['split_manifest']);assert sha(sfile)==c['split_manifest_sha256'];spl=json.loads(sfile.read_text());meta={r['id']:r for r in spl['episodes']}
 pre=json.loads(a.precheck.read_text());assert pre['status']=='pass' and pre['contract_sha256']==sha(cpath)
 semantic='clip((6d_force-ref_force)[:3],-20,20) N; shear_x,shear_y,normal';assert c['force_target_semantics']==semantic
 cache={};audits=[];metrics=[];frames=[]
 for run in pre['runs']:
  fold,seed=run['fold'],run['seed'];mp=Path(run['force_prediction_manifest']);assert sha(mp)==run['force_prediction_manifest_sha256'];m=json.loads(mp.read_text())
  assert m['fold']==fold and m['seed']==seed and m['variant']=='adapt' and m['formal'] and m['status']=='complete'
  assert m['force_checkpoint_sha256']==run['force_sha256'] and sha(m['force_checkpoint'])==run['force_sha256']
  for e in m['entries']:
   eid=e['episode_id'];role=e['roles_by_fold'][fold]
   if role not in ('train','validation','calibration'):continue
   me=meta[eid];assert eid in spl['splits'][fold][role] and me['task']=='slip'
   source=Path(me['path']);assert str(source) in me['source_files']
   if eid not in cache:
    with np.load(source,allow_pickle=False) as z:
     force=np.asarray(z['6d_force'],dtype=np.float64);ref=np.asarray(z['ref_force'],dtype=np.float64)
    shape=header(source,'tactile_img');assert force.shape==(me['frames'],6) and ref.shape==(6,) and shape[0]==len(force)
    assert np.isfinite(force).all() and np.isfinite(ref).all()
    raw=force[:,:3]-ref[None,:3];target=np.clip(raw,-20,20)
    cache[eid]=(raw,target)
    audits.append(dict(episode_id=eid,path=str(source),prior_raw_sha256=me['source_files'][str(source)],force_array_sha256=arrsha(force),reference_sha256=arrsha(ref),image_header_shape=list(shape),force_shape=list(force.shape),reference_shape=list(ref.shape),frames=len(force),status='pass'))
   raw,target=cache[eid];pp=Path(e['prediction_path']);assert sha(pp)==e['prediction_sha256'];pred=np.load(pp,allow_pickle=False);assert pred.shape==target.shape and len(pred)==e['frames'] and np.isfinite(pred).all()
   entry=next(x for x in c['entries'] if x['episode_id']==eid);lp=Path(entry['label_path']);assert sha(lp)==entry['label_sha256'];stage=np.load(lp,allow_pickle=False);assert len(stage)==len(pred)
   t=np.arange(13,len(pred));err=pred-target;dp=pred[t]-pred[t-5];dt=target[t]-target[t-5]
   for pop,mask in [('primary',np.isin(stage[t],[0,2])),('static',stage[t]==0),('gross',stage[t]==2),('incipient',stage[t]==1)]:
    if not mask.any():continue
    ix=t[mask]
    for j,axis in enumerate(['shear_x','shear_y','normal']):
     y=target[ix,j];pr=pred[ix,j];er=err[ix,j];de=dp[mask,j]-dt[mask,j]
     metrics.append(dict(fold=fold,seed=seed,role=role,episode_id=eid,leakage_group=me['leakage_group'],probe=me['group'],population=pop,axis=axis,n=len(ix),mae=float(np.abs(er).mean()),rmse=float(np.sqrt(np.square(er).mean())),bias=float(er.mean()),target_std=float(y.std()),prediction_std=float(pr.std()),target_mean_abs=float(np.abs(y).mean()),prediction_mean_abs=float(np.abs(pr).mean()),delta5_mae=float(np.abs(de).mean()),delta5_bias=float(de.mean()),target_delta5_std=float(dt[mask,j].std()),prediction_delta5_std=float(dp[mask,j].std()),target_clip_fraction=float((np.abs(raw[ix,j])>20).mean()),prediction_outside_clip_fraction=float((np.abs(pr)>20).mean())))
   for idx,ti in enumerate(t):
    frames.append(dict(fold=fold,seed=seed,role=role,episode_id=eid,leakage_group=me['leakage_group'],t=int(ti),stage=int(stage[ti]),force_mae=float(np.abs(err[ti]).mean()),delta5_mae=float(np.abs(dp[idx]-dt[idx]).mean()),target_norm=float(np.linalg.norm(target[ti])),prediction_norm=float(np.linalg.norm(pred[ti])),clipped=bool((np.abs(raw[ti])>20).any())))
 writecsv(a.output/'per_trial_axis.csv',metrics);writecsv(a.output/'frame_errors.csv',frames)
 result=dict(status='pass',protocol_sha256=sha(HERE/'PROTOCOL.json'),source_sha256=sha(Path(__file__)),precheck_sha256=sha(a.precheck),contract_sha256=sha(cpath),split_sha256=sha(sfile),same_acquisition=True,read_test_for_evaluated_fold=False,training_changed=False,normalization='predictions already de-standardized by accepted R5 exporter; target reference-relative clipped N',episodes=audits,unique_episodes=len(audits),runs=len(pre['runs']),output_hashes={n:sha(a.output/n) for n in ['per_trial_axis.csv','frame_errors.csv']},limits=json.loads((HERE/'PROTOCOL.json').read_text())['limits'])
 (a.output/'SUPPORT_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status='pass',unique_episodes=len(audits),trial_axis_rows=len(metrics),frame_rows=len(frames))))
if __name__=='__main__':main()
