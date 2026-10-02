#!/usr/bin/env python3
"""Descriptive validation prediction distributions; no model or threshold selection."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np,torch
import train_frozen as D
import train_f1 as F

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(x):
 x=np.asarray(x,float).reshape(-1);q=np.quantile(x,[.01,.05,.5,.95,.99]);return {'n':len(x),'mean':float(x.mean()),'std':float(x.std()),'min':float(x.min()),'q01':q[0],'q05':q[1],'q50':q[2],'q95':q[3],'q99':q[4],'max':float(x.max()),'finite':bool(np.isfinite(x).all()),'near_zero_fraction':float((np.abs(x)<1e-6).mean()),'near_one_fraction':float((x>1-1e-6).mean())}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--inventory',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();rows=[]
 for r in json.loads(a.inventory.read_text())['runs']:
  out=Path(r['output']);cm=json.loads((out/'COMMIT.json').read_text());ck=torch.load(out/cm['best']['path'],map_location='cpu',weights_only=False);data=torch.load(r['data'],map_location='cpu',weights_only=False)
  if r['package'] in ('E1','E2'):
   norm=ck['normalizer'];x=(data['roles']['validation']['x']-norm['mean'])/norm['std'];m=D.init_model(r['group'],r['seed']);m.load_state_dict(ck['model']);pred=D.infer(m,x,'cpu').numpy();rec=stats(pred);rec.update({'package':r['package'],'group':r['group'],'fold':r['fold'],'seed':r['seed'],'quantity':'slip_probability','collapse':bool(rec['std']<1e-8)})
  else:
   n=ck['normalizer'];x=F.normalize_x(data['roles']['validation']['x'],n,r['group']);m=F.init_model(r['group'],r['seed']);m.load_state_dict(ck['model']);pred=F.predict(m,x,n,'cpu').numpy();mag=np.linalg.norm(pred,axis=2);rec=stats(mag);rec.update({'package':r['package'],'group':r['group'],'fold':r['fold'],'seed':r['seed'],'quantity':'predicted_delta_l2_n','collapse':bool(rec['std']<1e-8)})
  rows.append(rec)
 a.output.mkdir(parents=True,exist_ok=False);cols=list(rows[0]);
 with (a.output/'DISTRIBUTIONS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
 summary={'schema':'round22_prediction_diagnostics_v1','status':'pass' if all(r['finite'] and not r['collapse'] for r in rows) else 'review','runs':len(rows),'finite_runs':sum(r['finite'] for r in rows),'collapsed_runs':sum(r['collapse'] for r in rows),'descriptive_only':True,'model_or_threshold_changed':False,'inventory_sha256':sha(a.inventory)};(a.output/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()
