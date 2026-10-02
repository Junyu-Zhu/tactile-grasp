#!/usr/bin/env python3
"""Final training artifact audit, independent of reported scientific success."""
import argparse,csv,importlib.util,json,math,sys
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from run_formal import accepted
sys.path.insert(0,str(HERE/'current'))
from models import ForceAdapter,ForceConditionedSlip
from sources import load_decoupled_decoder,load_r3_slip_branch
spec=importlib.util.spec_from_file_location('frozen_future_audit',HERE/'future/future_pipeline.py')
future=importlib.util.module_from_spec(spec);spec.loader.exec_module(future)

def all_finite(value):
    if isinstance(value,torch.Tensor):return bool(torch.isfinite(value).all())
    if isinstance(value,np.ndarray):return bool(np.isfinite(value).all())
    if isinstance(value,dict):return all(all_finite(v) for v in value.values())
    if isinstance(value,(tuple,list)):return all(all_finite(v) for v in value)
    if isinstance(value,(float,int)):return math.isfinite(value)
    return True

def strict_model(job,summary,payload,shared):
    argv=job['argv'];get=lambda flag:argv[argv.index(flag)+1]
    if 'train-source' in argv:
        cache=shared.setdefault('source_cache',torch.load(get('--train-cache'),map_location='cpu',weights_only=False)) if 'source_cache' not in shared else shared['source_cache']
        dim=cache['z'].shape[1]+1+(0 if summary['variant']=='z_p_slip' else 3)+(3 if summary['variant']=='z_p_slip_force_pred_delta' else 0)
        cfg=summary['run_config'];model=future.LegacyFutureMLP(dim,cfg['hidden_dim'],len(cache['horizons']),cfg['dropout'])
    elif 'train-htt' in argv:
        cfg=summary['run_config'];model=future.GRURisk(769 if summary['variant']=='risk' else 774,cfg['state_output_dim'],summary['variant']=='full_state',cfg['hidden_dim'])
    else:
        if 'decoder' not in shared:shared['decoder']=load_decoupled_decoder(Path(get('--source-checkpoint')))[0]
        if 'train_force.py' in argv[1]:model=ForceAdapter(shared['decoder'])
        else:model=ForceConditionedSlip(load_r3_slip_branch(shared['decoder'],Path(get('--base-slip-checkpoint'))),summary['variant'],hidden_dim=64)
    model.load_state_dict(payload['model_state'],strict=True)
    return True

def main():
    p=argparse.ArgumentParser();p.add_argument('--inventory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    inv=json.loads(a.inventory.read_text());rows=[];failures=[];review=[];shared={}
    for j in inv['jobs']:
        if j['kind']!='neural':continue
        if not accepted(j):failures.append(j['id']+': incomplete/identity');continue
        s=json.loads(Path(j['acceptance_path']).read_text());checks={}
        checks['not_smoke']=s.get('smoke') is False
        checks['finite_history']=all(math.isfinite(v) for row in s.get('history',[]) for v in row.values() if isinstance(v,(float,int)))
        for k in ('finite_gradients_all_epochs','frozen_base_unchanged','checkpoint_restore_exact','force_parameters_changed'):
            if k in s:checks[k]=s[k] is True
        for k,v in s.get('checks',{}).items():checks[k]=v is True
        if s.get('all_trainable_tensors_changed') is False:review.append({'id':j['id'],'issue':'unchanged_trainable_tensor','details':s.get('trainable_changed')})
        best=Path(s.get('best_checkpoint',s.get('artifacts',{}).get('best',{}).get('path','')))
        payload=torch.load(best,map_location='cpu',weights_only=False)
        model=payload['model_state'];checks['checkpoint_all_tensors_finite']=all(bool(torch.isfinite(v).all()) for v in model.values())
        checks['strict_model_structure']=strict_model(j,s,payload,shared)
        if 'normalization' in payload:
            checks['normalization_finite']=all_finite(payload['normalization'])
        if s.get('predictions'):
            pred=Path(s['predictions']['validation']['path']);values=[]
            with pred.open() as f:
                for row in csv.DictReader(f):values.append(float(row['probability']))
            sd=float(np.std(values));checks['predictions_finite']=bool(np.isfinite(values).all())
            checks['predictions_in_unit_interval']=bool(((np.asarray(values)>=0)&(np.asarray(values)<=1)).all())
            if sd<1e-8:review.append({'id':j['id'],'issue':'constant_validation_probability','std':sd})
        if not all(checks.values()):failures.append({'id':j['id'],'checks':checks})
        rows.append({'id':j['id'],'status':'pass' if all(checks.values()) else 'fail','checks':checks,'best_checkpoint':str(best)})
    result={'status':'fail' if failures else ('needs_review' if review else 'pass'),'expected_neural_runs':66,'verified_runs':len(rows),'failures':failures,'review_required':review,'runs':rows,'scope':'engineering integrity; no requirement for positive scientific effect'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='runs'}))
    if failures:raise SystemExit(1)
if __name__=='__main__':main()
