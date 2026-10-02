#!/usr/bin/env python3
"""Audit the complete symmetric H/T_H formal force grid."""
import argparse,json,math
from pathlib import Path
from touchd_common import atomic_json,sha256
SEEDS=(20260914,20260915,20260916)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--touchd',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();runs=[]
 for fold in range(1,5):
  for seed in SEEDS:
   pair={r:json.loads((a.root/f'{r}_p{fold}_s{seed}/training_summary.json').read_text()) for r in ('H','T_H')}
   h,t=pair['H'],pair['T_H'];expected=sha256(a.touchd/f's{seed}/transferable_private_state.pth')
   if any(x['status']!='complete' or x['smoke'] or not x['formal_does_not_use_smoke'] or not all(x['component_changed'].values()) or not all(math.isfinite(z['train_loss']) and math.isfinite(z['selection_mean_rmse']) for z in x['history']) for x in pair.values()):raise RuntimeError((fold,seed,'completion'))
   if h['reset_head_sha256']!=t['reset_head_sha256'] or h['normalization']!=t['normalization'] or h['fit_samples']!=t['fit_samples'] or h['selection_samples']!=t['selection_samples']:raise RuntimeError((fold,seed,'symmetry'))
   if h['t_private'] is not None or t['t_private']['sha256']!=expected:raise RuntimeError((fold,seed,'route'))
   for x in pair.values():
    values=[z['selection_mean_rmse'] for z in x['history']];earliest=1+next(i for i,z in enumerate(values) if z==min(values))
    if x['best_epoch']!=earliest:raise RuntimeError((fold,seed,'selection'))
   runs.append({'fold':fold,'seed':seed,'reset_head_sha256':h['reset_head_sha256'],'normalization_equal':True,'H_best_epoch':h['best_epoch'],'T_H_best_epoch':t['best_epoch'],'H_best_rmse':h['best_metric'],'T_H_best_rmse':t['best_metric'],'H_summary_sha256':sha256(a.root/f'H_p{fold}_s{seed}/training_summary.json'),'T_H_summary_sha256':sha256(a.root/f'T_H_p{fold}_s{seed}/training_summary.json')})
 result={'schema':'round16_htt_force_formal_audit_v1','status':'pass','runs':24,'pairs':12,'same_seed_reset_head_bitwise_equal':True,'route_normalization_equal':True,'H_consumed_T':False,'T_H_consumed_matching_seed_T_private':True,'all_trainable_components_updated':True,'finite_curves':True,'earliest_strict_minimum_verified':True,'records':runs}
 atomic_json(a.output,result);print(json.dumps({'status':'pass','runs':24,'pairs':12}))
if __name__=='__main__':main()
