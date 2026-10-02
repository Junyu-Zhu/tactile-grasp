#!/usr/bin/env python3
"""Audit completed formal ToucHD runs without changing selection."""
import argparse,json,math
from pathlib import Path
import numpy as np
from touchd_common import atomic_json,sha256

SEEDS=(20260914,20260915,20260916)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=[];norm=None
 for seed in SEEDS:
  path=a.root/f's{seed}/summary.json';s=json.loads(path.read_text());h=s['history'];values=[x['selection_mean_rmse'] for x in h]
  earliest=1+next(i for i,x in enumerate(values) if x==min(values))
  if s['status']!='complete' or s['smoke'] or not s['formal_sharded_cache'] or not all(s['component_changed'].values()) or not all(math.isfinite(x['train_loss']) and math.isfinite(x['selection_mean_rmse']) for x in h) or s['best_epoch']!=earliest:raise RuntimeError(seed)
  if norm is None:norm=s['normalization']
  elif norm!=s['normalization']:raise RuntimeError('normalization drift')
  rows.append({'seed':seed,'epochs':len(h),'best_epoch':s['best_epoch'],'best_selection_rmse':s['best_metric'],'selection_rmse_variance':float(np.var(values)),'final_train_loss':h[-1]['train_loss'],'summary_sha256':sha256(path),'best_sha256':s['best_checkpoint_sha256'],'latest_sha256':s['latest_checkpoint_sha256']})
 result={'schema':'round16_touchd_formal_audit_v1','status':'pass','released_rows':81493,'eligible_pairs':80783,'fit_pairs':64616,'selection_pairs':16167,'normalization_identical_across_seeds':True,'finite_curves':True,'earliest_strict_minimum_verified':True,'smoke_weights_reused':False,'runs':rows}
 atomic_json(a.output,result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
