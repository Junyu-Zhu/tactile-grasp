#!/usr/bin/env python3
"""Descriptive validation diagnostics by slip stage and force horizon/axis."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np,torch
import train_frozen as D
import train_f1 as F

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def base(x):
 x=np.asarray(x,float).reshape(-1);q=np.quantile(x,[.01,.05,.5,.95,.99]);return {'n':len(x),'mean':float(x.mean()),'std':float(x.std()),'min':float(x.min()),'q01':q[0],'q05':q[1],'q50':q[2],'q95':q[3],'q99':q[4],'max':float(x.max()),'finite':bool(np.isfinite(x).all())}
def write(path,rows):
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--inventory',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();slip=[];force=[]
 for r in json.loads(a.inventory.read_text())['runs']:
  out=Path(r['output']);cm=json.loads((out/'COMMIT.json').read_text());ck=torch.load(out/cm['best']['path'],map_location='cpu',weights_only=False);data=torch.load(r['data'],map_location='cpu',weights_only=False);role=data['roles']['validation']
  meta={'package':r['package'],'group':r['group'],'fold':r['fold'],'seed':r['seed']}
  if r['package'] in ('E1','E2'):
   n=ck['normalizer'];x=(role['x']-n['mean'])/n['std'];m=D.init_model(r['group'],r['seed']);m.load_state_dict(ck['model']);pred=D.infer(m,x,'cpu').numpy();stage=role['stage'].numpy()
   for sid,name in ((-1,'all'),(0,'static'),(1,'incipient'),(2,'gross')):
    z=pred if sid<0 else pred[stage==sid];rec={**meta,'stage':name,**base(z),'near_zero_probability_fraction':float((z<1e-6).mean()),'near_one_probability_fraction':float((z>1-1e-6).mean())};slip.append(rec)
  else:
   n=ck['normalizer'];x=F.normalize_x(role['x'],n,r['group']);m=F.init_model(r['group'],r['seed']);m.load_state_dict(ck['model']);pred=F.predict(m,x,n,'cpu').numpy();truth=F.delta(role).numpy();anchor=role['x'][:,-1,192:195].numpy();future=anchor[:,None,:]+pred
   for hi,h in enumerate((1,5,10)):
    for ai,axis in enumerate(('x','y','z')):
     z=pred[:,hi,ai];rec={**meta,'horizon':h,'axis':axis,**base(z),'gt_std':float(truth[:,hi,ai].std()),'near_zero_delta_fraction':float((np.abs(z)<1e-6).mean()),'predicted_absolute_force_outside_20N_fraction':float((np.abs(future[:,hi,ai])>20).mean())};force.append(rec)
 a.output.mkdir(parents=True,exist_ok=False);write(a.output/'SLIP_PROBABILITY_BY_STAGE.csv',slip);write(a.output/'FORCE_BY_HORIZON_AXIS.csv',force);summary={'schema':'round22_prediction_diagnostics_v2','status':'pass' if all(r['finite'] for r in slip+force) else 'review','slip_runs':len({(r['group'],r['fold'],r['seed']) for r in slip}),'slip_rows':len(slip),'incipient_rows':sum(r['stage']=='incipient' for r in slip),'force_runs':len({(r['group'],r['fold'],r['seed']) for r in force}),'force_rows':len(force),'finite_rows':sum(r['finite'] for r in slip+force),'near_copy_diagnostic':'near_zero_delta_fraction per run/horizon/axis; descriptive only','collapse_scope':'per run/horizon/axis prediction std and GT std; no universal no-collapse claim','force_bounds':'unclipped predicted absolute force outside +/-20N fraction','model_or_threshold_changed':False,'test_consumed':False,'inventory_sha256':sha(a.inventory)};(a.output/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()
