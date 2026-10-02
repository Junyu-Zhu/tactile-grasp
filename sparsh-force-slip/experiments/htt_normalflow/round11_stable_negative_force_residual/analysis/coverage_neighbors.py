#!/usr/bin/env python3
"""Posthoc fixed-case visual-neighborhood proxy; CPU-only no model fitting."""
import argparse,json,hashlib,csv
from pathlib import Path
import torch,numpy as np
HERE=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def write(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 p=argparse.ArgumentParser();p.add_argument('--r10-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);protocol=json.loads((HERE/'COVERAGE_PROTOCOL.json').read_text());data_path=a.r10_root/'prepare/p4_s20260914/prepared.pt';dp_audit=data_path.with_name('audit.json');data=torch.load(data_path,map_location='cpu',weights_only=False);da=json.loads(dp_audit.read_text());assert da['output_hashes'][str(data_path)]==sha(data_path) and da['status']=='pass';assert data['fold']==protocol['fold'] and data['seed']==protocol['seed']
 run=a.r10_root/'formal/fusion/p4_s20260914';su=json.loads((run/'summary.json').read_text());assert su['status']=='complete' and not su['smoke'] and su['fold']==protocol['fold'] and su['seed']==protocol['seed'];assert su['output_hashes']['best.pth']==sha(run/'best.pth');ck=torch.load(run/'best.pth',map_location='cpu',weights_only=False);cfg=ck['config'];assert cfg['prepared']['path']==str(data_path) and cfg['prepared']['sha256']==sha(data_path);assert cfg['prepared_audit']['sha256']==sha(dp_audit)
 train=data['roles']['train'];val=data['roles']['validation'];assert set(train['leakage_group']).isdisjoint(val['leakage_group']);assert int(val['t'].min())>=13;mask=(train['stage']==0)|(train['stage']==2);v=train['x'][mask,:,:192].double().reshape(-1,192);mean=v.mean(0).float();std=v.std(0,unbiased=False).clamp_min(1e-6).float();norm=ck['normalizer'];assert torch.equal(mean,norm['mean'][:192]) and torch.equal(std,norm['std'][:192]);ztrain=(train['x'][:,:,:192]-mean)/std;zval=(val['x'][:,:,:192]-mean)/std;assert torch.isfinite(ztrain).all() and torch.isfinite(zval).all()
 predpath=run/'predictions_validation.csv';assert su['output_hashes'][predpath.name]==sha(predpath);preds=list(csv.DictReader(predpath.open()));assert [(r['episode_id'],int(r['t']),int(r['stage'])) for r in preds]==list(zip(val['episode_id'],val['t'].tolist(),val['stage'].tolist()));query=torch.nonzero(val['stage']==0).flatten();assert any(val['episode_id'][i]==protocol['episode'] for i in query.tolist());rows=[]
 for mode in ('last_base','history'):
  tx=ztrain[:,-1] if mode=='last_base' else ztrain.reshape(len(ztrain),-1);qx=zval[:,-1] if mode=='last_base' else zval.reshape(len(zval),-1);tx=tx.double();qx=qx.double();dimensions=tx.shape[1]
  for stage in (0,2):
   candidates=torch.nonzero(train['stage']==stage).flatten();features=tx[candidates];tn=(features*features).sum(1)
   for start in range(0,len(query),128):
    ids=query[start:start+128];q=qx[ids];dist=((q*q).sum(1)[:,None]+tn[None,:]-2*q@features.T).clamp_min(0)/dimensions
    for j,i in enumerate(ids.tolist()):
     excluded=torch.tensor([train['leakage_group'][k]==val['leakage_group'][i] for k in candidates.tolist()]);dist[j,excluded]=float('inf');minimum,index=dist[j].min(0);assert torch.isfinite(minimum);k=int(candidates[int(index)]);case=val['episode_id'][i]==protocol['episode'];prob=float(preds[i]['p_slip'])
     rows.append(dict(mode=mode,fold=protocol['fold'],seed=protocol['seed'],query_episode=val['episode_id'][i],query_t=int(val['t'][i]),query_group=val['leakage_group'][i],query_partition='historical_case' if case else 'other_validation_static',query_r10_probability=prob,case_high_confidence=case and prob>=.95,candidate_stage=stage,nearest_train_episode=train['episode_id'][k],nearest_train_t=int(train['t'][k]),nearest_train_group=train['leakage_group'][k],distance_rms=float(minimum.sqrt()),dimensions=dimensions,eligible_train_endpoints=int((~excluded).sum())))
 write(a.output/'nearest_endpoints.csv',rows);summaries=[];trial=[]
 for mode in ('last_base','history'):
  for stage in (0,2):
   for episode in sorted({r['query_episode'] for r in rows}):
    rr=[r for r in rows if r['mode']==mode and r['candidate_stage']==stage and r['query_episode']==episode];d=[r['distance_rms'] for r in rr];trial.append(dict(mode=mode,candidate_stage=stage,query_episode=episode,query_partition=rr[0]['query_partition'],n_endpoints=len(rr),median_distance=float(np.median(d)),q90_distance=float(np.quantile(d,.9))))
   for part in ('historical_case','other_validation_static','case_high_confidence'):
    rr=[r for r in rows if r['mode']==mode and r['candidate_stage']==stage and (r['case_high_confidence'] if part=='case_high_confidence' else r['query_partition']==part)];d=[r['distance_rms'] for r in rr]
    summaries.append(dict(mode=mode,candidate_stage=stage,query_partition=part,n_endpoints=len(rr),n_trials=len({r['query_episode'] for r in rr}),**{f'q{int(q*100):02d}':float(np.quantile(d,q)) if d else None for q in (0,.25,.5,.75,.9,1)}))
 write(a.output/'distance_distribution.csv',summaries);write(a.output/'per_trial_distribution.csv',trial);sources=[data_path,dp_audit,run/'summary.json',run/'best.pth',predpath,HERE/'COVERAGE_PROTOCOL.json',Path(__file__)];out=dict(status='pass',post_training=True,case_selected_from_R10=True,source_hashes={str(p):sha(p) for p in sources},train_only_candidates=True,same_group_excluded=True,normalizer_recomputed_from_train=True,test_consumed=False,calibration_fit=False,trained_model=False,endpoint_records=len(rows),output_hashes={p.name:sha(p) for p in a.output.glob('*.csv')});(a.output/'AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':'pass','queries':len(query),'records':len(rows)}))
if __name__=='__main__':main()
