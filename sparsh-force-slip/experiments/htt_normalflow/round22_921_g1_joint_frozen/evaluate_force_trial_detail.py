#!/usr/bin/env python3
"""Formal validation trial x horizon x axis force supplement; descriptive, locked models."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np,torch
import train_f1 as T
import evaluate_f1_f2 as E

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--inventory',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();runs=[r for r in json.loads(a.inventory.read_text())['runs'] if r['package']=='F1'];rows=[];index=[]
 for rr in runs:
  out=Path(rr['output']);s=json.loads((out/'summary.json').read_text());cm=json.loads((out/'COMMIT.json').read_text());bp=out/cm['best']['path'];ck=torch.load(bp,map_location='cpu',weights_only=False);ident=s['identity'];expected={'group':rr['group'],'fold':rr['fold'],'seed':rr['seed'],'data_sha256':rr['data_sha256'],'source_sha256':rr['trainer_sha256'],'support_inventory_sha256':rr['dependencies']['support_inventory_sha256'],'protocol_sha256':rr['protocol_sha256'],'formal':True};assert s['status']=='complete' and ck['identity']==ident and sha(bp)==cm['best']['sha256'] and all(ident.get(k)==v for k,v in expected.items())
  d=torch.load(rr['data'],map_location='cpu',weights_only=False);T.validate(d,rr['data'],rr['fold'],rr['seed'],Path(T.__file__).with_name('G1_PREPARE.json'));m=T.init_model(rr['group'],rr['seed']);m.load_state_dict(ck['model']);n=ck['normalizer'];r=d['roles']['validation'];raw=T.predict(m,T.normalize_x(r['x'],n,rr['group']),n,'cpu');ridge=E.ridge_fit(d['roles']['fit'],rr['group']);variants={rr['group']:raw,'hold':torch.zeros_like(raw),f"ridge-{rr['group']}":E.ridge_predict(r,rr['group'],ridge)}
  if rr['group']=='K-VF':variants['F2-half']=.5*raw
  truth=T.delta(r).numpy();future=r['y'].numpy();anchor=r['x'][:,-1,192:195].numpy();eps=np.asarray(r['episode_id'],object);leak=np.asarray(r['leakage_group'],object);q=np.max(np.abs(truth[:,2]),axis=1);stratum=np.where(q<=.25,'stable',np.where(q>=1,'changing','transition'))
  for name,pred in variants.items():
   pd=pred.numpy();pf=anchor[:,None,:]+pd
   for ep in sorted(set(eps)):
    mask=eps==ep
    for hi,h in enumerate((1,5,10)):
     for ai,axis in enumerate(('x','y','z','all')):
      de=pd[mask,hi]-truth[mask,hi];ae=pf[mask,hi]-future[mask,hi]
      if axis!='all':de=de[:,ai];ae=ae[:,ai]
      rows.append({'group':rr['group'],'fold':rr['fold'],'seed':rr['seed'],'variant':name,'role':'validation','episode':ep,'leakage_group':leak[mask][0],'horizon':h,'axis':axis,'endpoints':int(mask.sum()),'stable_endpoints':int((stratum[mask]=='stable').sum()),'transition_endpoints':int((stratum[mask]=='transition').sum()),'changing_endpoints':int((stratum[mask]=='changing').sum()),'gt_delta_abs_mean_n':float(np.abs(truth[mask,hi] if axis=='all' else truth[mask,hi,ai]).mean()),'delta_mae_n':float(np.abs(de).mean()),'delta_rmse_n':float(np.sqrt(np.square(de).mean())),'absolute_future_mae_n':float(np.abs(ae).mean()),'absolute_future_rmse_n':float(np.sqrt(np.square(ae).mean()))})
  index.append({'run':rr['run'],'best_sha256':sha(bp),'summary_sha256':sha(out/'summary.json'),'data_sha256':rr['data_sha256']})
 a.output.mkdir(parents=True,exist_ok=False);cols=list(rows[0]);
 with (a.output/'TRIAL_HORIZON_AXIS_METRICS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
 
 from scipy.stats import spearmanr
 corr=[]
 for variant in sorted({r['variant'] for r in rows}):
  for h in (1,5,10):
   for axis in ('x','y','z','all'):
    z=[r for r in rows if r['variant']==variant and r['horizon']==h and r['axis']==axis];x=np.asarray([r['gt_delta_abs_mean_n'] for r in z]);y=np.asarray([r['delta_mae_n'] for r in z]);corr.append({'variant':variant,'horizon':h,'axis':axis,'trials':len(z),'pearson_r':float(np.corrcoef(x,y)[0,1]),'spearman_r':float(spearmanr(x,y).statistic),'interpretation':'descriptive_not_causal'})
 with (a.output/'GT_ERROR_CORRELATION.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(corr[0]));w.writeheader();w.writerows(corr)
 (a.output/'INPUT_INDEX.json').write_text(json.dumps(index,indent=2)+'\n');summary={'schema':'round22_force_trial_horizon_axis_v1','status':'complete','runs':len(runs),'rows':len(rows),'role':'validation','horizons':[1,5,10],'axes':['x','y','z','all'],'test_consumed':False,'descriptive_only':True};(a.output/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()
