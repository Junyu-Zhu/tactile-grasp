#!/usr/bin/env python3
import argparse,hashlib,importlib.util,json,math
from pathlib import Path
import torch

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def finite(v):
 if torch.is_tensor(v):return bool(torch.isfinite(v).all())
 if isinstance(v,dict):return all(finite(x) for x in v.values())
 if isinstance(v,(list,tuple)):return all(finite(x) for x in v)
 if isinstance(v,float):return math.isfinite(v)
 return True

def main():
 p=argparse.ArgumentParser();p.add_argument('--inventory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();inv=json.loads(a.inventory.read_text());sp=importlib.util.spec_from_file_location('r6_training_audit',inv['trainer']);tr=importlib.util.module_from_spec(sp);sp.loader.exec_module(tr);fail=[];review=[];rows=[];checks={}
 expected={(g,s) for g in ['A_visual','B_force','C_force_delta'] for s in [20260914,20260915,20260916]};checks['exact9']=len(inv['runs'])==9 and {(x['group'],x['seed']) for x in inv['runs']}==expected
 for raw,digest in inv['frozen_inputs'].items():checks['frozen:'+raw]=sha(Path(raw))==digest
 state=json.loads((Path(inv['output_root'])/'QUEUE_STATE.json').read_text());checks['queue_complete']=state['status']=='complete' and set(state['jobs'])=={x['id'] for x in inv['runs']} and all(v=='complete' for v in state['jobs'].values())
 for run in inv['runs']:
  d=json.loads((Path(run['output'])/'summary.json').read_text());best=torch.load(Path(run['output'])/'best.pth',map_location='cpu',weights_only=False);latest=torch.load(Path(run['output'])/'latest.pth',map_location='cpu',weights_only=False);cfg=d['run_config'];c={}
  c['formal_identity']=d['formal'] and not d['smoke'] and cfg['mode']=='formal' and cfg['group']==run['group'] and cfg['seed']==run['seed'] and cfg['horizon']==1
  c['config_hash']=tr.identity_sha(cfg)==d['run_identity_sha256']==best['run_identity_sha256']==latest['run_identity_sha256'];c['source_bundle']=cfg['code_bundle_sha256']==tr.code_bundle_sha()
  c['finite_checkpoints']=finite(best['model_state']) and finite(latest['model_state']) and finite(latest['optimizer_state']) and finite(d['history']) and finite(best['normalization'])
  c['normalizers_positive']=bool((best['normalization']['std']>0).all());c['all_parameters_updated']=d['checks']['all_parameter_tensors_updated'] and all(d['checks']['parameter_updates_by_tensor'].values())
  c['finite_gradients']=d['checks']['finite_loss_and_all_trainable_gradients'];c['roundtrip']=d['checks']['checkpoint_roundtrip_exact'];c['selection_separation']=d['checks']['calibration_and_outer_not_used_for_training_or_selection']
  model=tr.GRURisk(cfg['input_dim'],128);model.load_state_dict(best['model_state'],strict=True);c['strict_load']=True;c['only_future_params']=set(best['model_state'])=={'gru.weight_ih_l0','gru.weight_hh_l0','gru.bias_ih_l0','gru.bias_hh_l0','risk.weight','risk.bias'}
  for key in ['best','latest']:c['hash:'+key]=sha(Path(d['artifacts'][key]['path']))==d['artifacts'][key]['sha256']
  for role in ['calibration','outer']:
   e=d['evaluations'][role]
   if e['probability_std']<1e-8 or e['probability_unique']<=1:review.append({'id':run['id'],'role':role,'issue':'near-constant probabilities require diagnosis'})
  if not all(c.values()):fail.append({'id':run['id'],'failed':[k for k,v in c.items() if not v]})
  rows.append({'id':run['id'],'checks':c,'epochs':len(d['history']),'best_epoch':d['best_epoch'],'parameters':sum(p.numel() for p in model.parameters()),'outer_probability_std':d['evaluations']['outer']['probability_std'],'outer_unique_probabilities':d['evaluations']['outer']['probability_unique'],'best_sha256':d['artifacts']['best']['sha256'],'latest_sha256':d['artifacts']['latest']['sha256']})
 status='pass' if all(checks.values()) and not fail and not review else 'fail';result={'status':status,'verified_runs':len(rows),'checks':checks,'runs':rows,'failures':fail,'review_required':review,'scope':'engineering integrity and collapse screening, not requirement for positive effect'};a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':status,'runs':len(rows),'failures':fail,'review_required':review}));return 0 if status=='pass' else 1
if __name__=='__main__':raise SystemExit(main())
