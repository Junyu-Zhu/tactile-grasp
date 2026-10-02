#!/usr/bin/env python3
"""Fixed group-bootstrap threshold uncertainty; never selects deployment thresholds."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'evaluation'))
# Unique module name avoids collision with this file's evaluate.py.
import importlib.util
p=Path(__file__).resolve().parents[1]/'evaluation/evaluate.py';spec=importlib.util.spec_from_file_location('current_evaluator',p);e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
FPRS=(.01,.05,.1)
def group_thresholds(rows,weights):
 primary=[r for r in rows if r['stage']!=1];scores=np.array([r['score'] for r in primary]);labels=np.array([r['stage'] for r in primary]);w=np.array([weights.get(r['group'],0) for r in primary],dtype=float);valid=w>0;scores=scores[valid];labels=labels[valid];w=w[valid];neg=w[labels==0].sum();pos=w[labels==2].sum()
 if not neg:return 'invalid_no_static',{}
 if not pos:return 'invalid_no_gross',{}
 order=np.argsort(-scores,kind='stable');scores=scores[order];labels=labels[order];w=w[order];ends=np.r_[np.flatnonzero(np.diff(scores)),len(scores)-1];fp=np.r_[0,np.cumsum(w*(labels==0))[ends]]/neg;tp=np.r_[0,np.cumsum(w*(labels==2))[ends]]/pos;th=np.r_[np.nextafter(1.,np.inf),scores[ends]];result={}
 for fpr in FPRS:
  indices=np.flatnonzero(fp<=fpr+1e-15);i=max(indices,key=lambda i:(tp[i],-fp[i],th[i]));result[fpr]=dict(threshold=float(th[i]),bootstrap_calibration_fpr=float(fp[i]),bootstrap_calibration_recall=float(tp[i]),never_alarm=bool(th[i]>1))
 return 'valid',result

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-synthetic',action='store_true');a=p.parse_args();manifest=json.loads(a.manifest.read_text());synthetic,runs,folds=e.validate_manifest(manifest,a.allow_synthetic);protocol=json.loads(Path(__file__).with_name('PROTOCOL.json').read_text());a.output.mkdir(parents=True,exist_ok=True);data={};groups={};inputs={};draws={};canonical={}
 for run in runs:
  rr=e.load_rows(run['predictions']['calibration'],run['endpoints']['calibration'],'calibration',fold=run['fold']);data[(run['group'],run['fold'],run['seed'])]=rr;names=sorted({r['group'] for r in rr})
  signature=[(x['episode'],x['t'],x['stage'],x['group']) for x in rr]
  if run['fold'] in canonical:assert canonical[run['fold']]==signature
  else:canonical[run['fold']]=signature
  if run['fold'] in groups:assert groups[run['fold']]==names
  else:groups[run['fold']]=names
  for rec in (run['predictions']['calibration'],run['endpoints']['calibration']):inputs[rec['path']]=rec['sha256']
 for fi,fold in enumerate(folds):
  names=groups[fold];rng=np.random.default_rng(protocol['bootstrap_seed']+fi);draws[fold]=[dict(zip(names,np.bincount(rng.integers(0,len(names),len(names)),minlength=len(names)).tolist())) for _ in range(protocol['draws'])]
 rows=[]
 for run in runs:
  ident=(run['group'],run['fold'],run['seed']);rr=data[ident]
  for i,w in enumerate(draws[run['fold']]):
   status,points=group_thresholds(rr,w)
   for fpr in FPRS:rows.append(dict(group=ident[0],fold=ident[1],seed=ident[2],draw=i,fpr_constraint=fpr,status=status,**points.get(fpr,dict(threshold=None,bootstrap_calibration_fpr=None,bootstrap_calibration_recall=None,never_alarm=None))))
 summary=[]
 for run in runs:
  for fpr in FPRS:
   subset=[r for r in rows if (r['group'],r['fold'],r['seed'],r['fpr_constraint'])==(run['group'],run['fold'],run['seed'],fpr)];valid=[r for r in subset if r['status']=='valid'];t=np.array([r['threshold'] for r in valid]);summary.append(dict(group=run['group'],fold=run['fold'],seed=run['seed'],fpr_constraint=fpr,requested=len(subset),valid=len(valid),valid_fraction=len(valid)/len(subset),invalid_no_static=sum(r['status']=='invalid_no_static' for r in subset),invalid_no_gross=sum(r['status']=='invalid_no_gross' for r in subset),never_alarm_fraction=sum(r['never_alarm'] for r in valid)/len(valid) if valid else None,**{f'threshold_q{int(q*1000):03d}':float(np.quantile(t,q)) if len(t) else None for q in (.025,.5,.975)}))
 e.csvout(a.output/'bootstrap_thresholds.csv',rows);e.csvout(a.output/'threshold_uncertainty.csv',summary);e.js(a.output/'GROUP_DRAWS.json',draws);e.js(a.output/'AUDIT.json',dict(status='pass',synthetic=synthetic,runs=len(runs),draws_per_fold=protocol['draws'],shared_group_draws_across_models_seeds=True,validation_consumed=False,deployment_threshold_changed=False,manifest_sha256=e.sha(a.manifest),source_sha256=e.sha(__file__),protocol_sha256=e.sha(Path(__file__).with_name('PROTOCOL.json')),evaluator_sha256=e.sha(e.__file__),input_hashes=inputs,outputs={p.name:e.sha(p) for p in a.output.iterdir() if p.is_file() and p.name!='AUDIT.json'}))
if __name__=='__main__':main()
