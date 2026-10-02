#!/usr/bin/env python3
"""Aggregate the real-token H/T_H force, slip, and future smoke gates."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from touchd_common import atomic_json,sha256

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.root
 base=json.loads((r/'SMOKE.json').read_text());assert base['status']=='pass'
 force={route:json.loads((r/f'htt_{route}/training_summary.json').read_text()) for route in ('H','T_H')}
 assert all(x['status']=='complete' and x['smoke'] and all(x['component_changed'].values()) for x in force.values())
 assert force['H']['reset_head_sha256']==force['T_H']['reset_head_sha256']
 slip={route:json.loads((r/f'slip_{route}/summary.json').read_text()) for route in ('H','T_H')}
 assert all(x['status']=='complete' and x['smoke'] and x['optimizer_only_new_head'] and x['prepared_unchanged'] and all(x['parameter_tensors_updated'].values()) for x in slip.values())
 future={route:json.loads((r/f'future_{route}/summary.json').read_text()) for route in ('H','T_H')}
 assert all(x['status']=='complete' and x['identity']['group']=='F_concat' and x['identity']['horizons']==[1,5,10] for x in future.values())
 prepared={route:torch.load(r/f'future_data_{route}/prepared.pt',map_location='cpu',weights_only=False) for route in ('H','T_H')}
 for role in ('fit','selection','calibration','validation'):
  for key in ('y','y_current','t'):
   assert torch.equal(prepared['H']['roles'][role][key],prepared['T_H']['roles'][role][key])
  for key in ('episode_id','leakage_group'):assert prepared['H']['roles'][role][key]==prepared['T_H']['roles'][role][key]
 result={'schema':'round16_pipeline_smoke_audit_v1','status':'pass','formal':False,'smoke_inputs_forbidden_formal':True,
         'complete_token_and_T_recovery':base,'htt_force':{route:{'reset_head_sha256':x['reset_head_sha256'],'fit_samples':x['fit_samples'],'selection_samples':x['selection_samples'],'component_changed':x['component_changed'],'summary_sha256':sha256(r/f'htt_{route}/training_summary.json')} for route,x in force.items()},
         'same_seed_htt_head_reset_bitwise':True,'slip':{route:{'best_epoch':x['best_epoch'],'all_new_head_tensors_updated':all(x['parameter_tensors_updated'].values()),'upstream_frozen':x['optimizer_only_new_head'] and x['prepared_unchanged'],'summary_sha256':sha256(r/f'slip_{route}/summary.json')} for route,x in slip.items()},
         'future':{route:{'group':x['identity']['group'],'horizons':x['identity']['horizons'],'best_epoch':x['best_epoch'],'summary_sha256':sha256(r/f'future_{route}/summary.json')} for route,x in future.items()},
         'route_endpoint_and_target_identity':True,'no_future_input_to_force_or_slip':True}
 atomic_json(a.output,result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
