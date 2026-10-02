#!/usr/bin/env python3
import argparse,hashlib,importlib.util,json,math
from pathlib import Path
import torch

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def finite(v):
 if torch.is_tensor(v):return bool(torch.isfinite(v).all())
 if isinstance(v,dict):return all(finite(x) for x in v.values())
 if isinstance(v,(list,tuple)):return all(finite(x) for x in v)
 if isinstance(v,float):return math.isfinite(v)
 return True

def main():
 p=argparse.ArgumentParser();p.add_argument('--inventory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();inv=json.loads(a.inventory.read_text());sp=importlib.util.spec_from_file_location('r7trainer',inv['trainer']);tr=importlib.util.module_from_spec(sp);sp.loader.exec_module(tr);checks={};rows=[];fail=[];review=[];initial={}
 expected={(g,s) for g in inv['groups'] for s in [20260914,20260915,20260916]};checks['exact_planned_runs']=len(inv['runs'])==len(expected) and {(x['group'],x['seed']) for x in inv['runs']}==expected and len(expected)<=12
 for raw,h in inv['frozen_inputs'].items():checks['frozen:'+raw]=sha(Path(raw))==h
 state=json.loads((Path(inv['output_root'])/'QUEUE_STATE.json').read_text());checks['queue_complete']=state['status']=='complete' and set(state['jobs'])=={x['id'] for x in inv['runs']} and all(x=='complete' for x in state['jobs'].values())
 for run in inv['runs']:
  out=Path(run['output']);d=json.loads((out/'summary.json').read_text());cfg=d['run_config'];best=torch.load(out/'best.pth',map_location='cpu',weights_only=False);latest=torch.load(out/'latest.pth',map_location='cpu',weights_only=False);c={}
  c['formal_identity']=d['status']=='complete' and d['formal'] and not d['smoke'] and cfg['mode']=='formal' and cfg['group']==run['group'] and cfg['seed']==run['seed'] and cfg['horizons']==inv['horizons'] and cfg['data_sha256']==inv['prepared_sha256']
  c['configuration']=tr.identity_sha(cfg)==d['run_identity_sha256']==best['run_identity_sha256']==latest['run_identity_sha256'];c['source_bundle']=cfg['code_bundle_sha256']==tr.code_bundle_sha()[0]
  c['best_embedded_latest']=tr.nested_equal(best['model_state'],latest['best_model_state']) and best['best_epoch']==latest['best_epoch']==d['best_epoch'] and best['best_selection_loss']==latest['best_selection_loss']==d['best_selection_loss']
  receipts=[json.loads(p.read_text()) for p in Path(inv['output_root']).glob(run['id']+'.receipt.attempt*.json')];success=[p for p in receipts if p['exit_code']==0 and p['status']=='complete'];c['successful_receipt']=len(success)==1
  if success:
   receipt=success[0];cmd=receipt['argv'];c['receipt_identity']=receipt['inventory_sha256']==sha(a.inventory) and '--execute-formal' in cmd and cmd[cmd.index('--group')+1]==run['group'] and int(cmd[cmd.index('--seed')+1])==run['seed'] and cmd[cmd.index('--output')+1]==run['output']
  c['finite']=finite(best['model_state']) and finite(latest['model_state']) and finite(latest['optimizer_state']) and finite(d['history'])
  c['earliest_selection_min']=d['best_epoch']==min(d['history'],key=lambda x:x['selection_unweighted_equal_horizon_bce'])['epoch']
  for key in inv['required_run_checks']:c[key]=d['audit'][key] is True
  model=tr.MultiWindowGRU(776,128,len(inv['horizons']));model.load_state_dict(best['model_state'],strict=True);c['only_future_params']=set(best['model_state'])=={'gru.weight_ih_l0','gru.weight_hh_l0','gru.bias_ih_l0','gru.bias_hh_l0','risk.weight','risk.bias'}
  seed=cfg['seed'];init=cfg['initialization_tensor_sha256'];c['shared_initialization']=seed not in initial or initial[seed]==init;initial[seed]=init
  for key in ['best','latest','config']:
   x=d['artifacts'][key];c['hash:'+key]=sha(Path(x['path']))==x['sha256']
  c['prediction_roles']=set(d['artifacts']['predictions'])=={'selection','calibration','outer'}
  for role,pops in d['artifacts']['predictions'].items():
   c['populations:'+role]=set(pops)=={'eligible','timeline'}
   for pop,x in pops.items():c['hash:'+role+':'+pop]=sha(Path(x['path']))==x['sha256']
  for population,hs in d['probability_diagnostics'].items():
   for h,x in hs.items():
    if x['std']<1e-8 or x['unique']<=1:review.append({'id':run['id'],'population':population,'horizon':h,'issue':'constant or nearconstant prediction'})
  if not all(c.values()):fail.append({'id':run['id'],'checks':[k for k,v in c.items() if not v]})
  rows.append({'id':run['id'],'checks':c,'epochs':len(d['history']),'best_epoch':d['best_epoch'],'parameters':sum(p.numel() for p in model.parameters()),'probability_diagnostics':d['probability_diagnostics'],'best_sha256':d['artifacts']['best']['sha256'],'latest_sha256':d['artifacts']['latest']['sha256']})
 result={'status':('fail' if not all(checks.values()) or fail else 'review_required' if review else 'pass'),'verified_runs':len(rows),'checks':checks,'runs':rows,'failures':fail,'review_required':review};a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'runs':len(rows),'failures':fail,'review_required':review}));return result['status']!='pass'
if __name__=='__main__':raise SystemExit(main())
