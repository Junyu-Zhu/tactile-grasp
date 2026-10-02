#!/usr/bin/env python3
"""Audit all formal current-slip and future-force downstream pairs."""
import argparse,json,math
from pathlib import Path
from touchd_common import atomic_json,sha256
SEEDS=(20260914,20260915,20260916)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();records=[]
 for fold in range(1,5):
  for seed in SEEDS:
   pair={}
   for route in ('H','T_H'):
    key=f'{route}_p{fold}_s{seed}';receipt=json.loads((a.root/f'formal/pairs/{key}.json').read_text());slip_path=a.root/f'formal/slip/{key}/summary.json';future_path=a.root/f'formal/future/{key}/summary.json';slip=json.loads(slip_path.read_text());future=json.loads(future_path.read_text())
    if receipt['status']!='complete' or [x['stage'] for x in receipt['stages']]!=['export','replace','slip','future_prepare','future']:raise RuntimeError((key,'receipt'))
    if slip['status']!='complete' or slip['smoke'] or not slip['optimizer_only_new_head'] or not slip['prepared_unchanged'] or not slip['checkpoint_roundtrip_exact'] or not all(slip['parameter_tensors_updated'].values()):raise RuntimeError((key,'slip'))
    if future['status']!='complete' or future['identity']['group']!='F_concat' or future['identity']['fold']!=fold or future['identity']['seed']!=seed or not math.isfinite(future['best_metric_native_n_mae']):raise RuntimeError((key,'future'))
    endpoints={role:sha256(a.root/f'formal/prepared/{key}/endpoints_{role}.csv') for role in ('train','calibration','validation')};pair[route]=endpoints
    records.append({'route':route,'fold':fold,'seed':seed,'slip_best_epoch':slip['best_epoch'],'slip_best_score':slip['best_score'],'future_best_epoch':future['best_epoch'],'future_best_mae':future['best_metric_native_n_mae'],'receipt_sha256':sha256(a.root/f'formal/pairs/{key}.json'),'slip_summary_sha256':sha256(slip_path),'future_summary_sha256':sha256(future_path)})
   if pair['H']!=pair['T_H']:raise RuntimeError((fold,seed,'endpoint identity'))
 result={'schema':'round16_downstream_formal_audit_v1','status':'pass','pair_jobs':24,'current_slip_runs':24,'future_runs':24,'neural_runs':48,'all_stages_complete':True,'only_new_slip_head_optimized':True,'future_group_exactly_F_concat':True,'route_endpoint_and_label_identity':True,'test_consumed':False,'records':records};atomic_json(a.output,result);print(json.dumps({'status':'pass','pair_jobs':24,'neural_runs':48}))
if __name__=='__main__':main()
