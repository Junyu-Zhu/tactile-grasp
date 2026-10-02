#!/usr/bin/env python3
"""Validate every planned run and produce checkpoint/delivery indices."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import torch
import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('r8_training_auditor',HERE/'training/train.py')
train=importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    inv=json.loads(a.inventory.read_text())
    a.output.mkdir(parents=True,exist_ok=True)
    expected={(g,seed) for g in inv['groups'] for seed in (20260914,20260915,20260916)}
    if inv.get('schema')!='round8_formal_inventory_v1' or len(inv['runs'])!=len(expected) or {(r['group'],r['seed']) for r in inv['runs']}!=expected or len({str(Path(r['output']).resolve()) for r in inv['runs']})!=len(expected) or any(r['id']!=f"future_{r['group']}_{r['seed']}" for r in inv['runs']):
        raise ValueError('Incomplete or duplicate formal run grid')
    # A failed re-audit must not leave an old consumable manifest behind.
    manifest_path=a.output/'PREDICTION_MANIFEST.json'
    if manifest_path.exists():
        train.atomic_json(manifest_path,{'schema':'round8_prediction_manifest_v1','status':'audit_pending'})
    frozen={path:train.sha256(Path(path))==value for path,value in inv['frozen_inputs'].items()}
    if not all(frozen.values()):
        raise ValueError('Locked input drift')
    records=[]
    index=[]
    inits={}
    manifests=[]
    for run in inv['runs']:
        root=Path(run['output'])
        summary=json.loads((root/'summary.json').read_text())
        config=summary['run_config']
        best=torch.load(root/'best.pth',map_location='cpu',weights_only=False)
        latest=torch.load(root/'latest.pth',map_location='cpu',weights_only=False)
        losses=[x['selection_unweighted_cumulative_horizon_bce'] for x in summary['history']]
        expected_epoch=summary['history'][int(np.argmin(losses))]['epoch']
        checks={'schema':summary.get('schema')=='round8_future_timeline_v1','head_type':summary.get('head_type')==train.head_type(run['group']),'complete':summary.get('status')=='complete','formal':summary['formal'] and not summary['smoke'] and config['mode']=='formal',
                'group_seed':summary['group']==run['group'] and summary['seed']==run['seed'],
                'prepared':config['prepared_sha']==inv['prepared_sha256'],
                'configuration':config==json.loads((root/'config.json').read_text())==best['run_config']==latest['run_config'],
                'earliest_selection_min':summary['best_epoch']==best['best_epoch']==expected_epoch,
                'finite_history':bool(np.isfinite(losses).all()),
                'best_embedded_latest':train.nested_equal(best['model_state'],latest['best_model_state']),
                'finite_weights':all(bool(torch.isfinite(v).all()) for v in best['model_state'].values()),
                'identity':summary['run_identity_sha256']==train.identity_sha(config),
                'source_hashes':all(train.sha256(Path(path))==h for path,h in config['source_hashes'].items())}
        checks.update({k:summary['audit'][k] is True for k in inv['required_run_checks']})
        for name in ('best','latest','config'):
            artifact=summary['artifacts'][name]
            checks['canonical_'+name]=Path(artifact['path']).resolve()==(root/({'best':'best.pth','latest':'latest.pth','config':'config.json'}[name])).resolve()
            checks['hash_'+name]=train.sha256(Path(artifact['path']))==artifact['sha256']
            index.append({'id':run['id'],'group':run['group'],'seed':run['seed'],'kind':name,**artifact,
                          'epochs':len(losses),'best_epoch':expected_epoch,'parameters':summary['model_parameters']})
        predictions=summary['artifacts']['predictions']
        checks['prediction_grid']=set(predictions)=={'selection','calibration','outer'} and all(set(v)=={'eligible','timeline'} for v in predictions.values())
        for role,populations in predictions.items():
            for population,artifact in populations.items():
                checks[f'canonical_prediction_{role}_{population}']=Path(artifact['path']).resolve()==(root/f'predictions_{population}_{role}.csv').resolve()
                checks[f'prediction_{role}_{population}']=train.sha256(Path(artifact['path']))==artifact['sha256']
                if population=='timeline':
                    manifests.append({'group':run['group'],'seed':run['seed'],'role':role,'head_type':summary['head_type'],**artifact})
        variance_flags={name:[h for h,d in diagnostics.items() if d['std']<1e-6 or d['unique']<5] for name,diagnostics in summary['probability_diagnostics'].items()}
        checks['no_constant_collapse']=not any(variance_flags.values())
        records.append({'id':run['id'],'checks':checks,'probability_diagnostics':summary['probability_diagnostics'],'epochs':len(losses),'best_epoch':expected_epoch,'parameters':summary['model_parameters']})
        inits[(run['group'],run['seed'])]=config['initialization_tensor_sha256']
    initialization={}
    for seed in (20260914,20260915,20260916):
        initialization[f'P1_full_{seed}']=inits[('B_xyz',seed)]==inits[('C_xyz_delta',seed)]
        if ('P4_state',seed) in inits:
            initialization[f'P4_full_{seed}']=inits[('P4_direct',seed)]==inits[('P4_state',seed)]
        initialization[f'P2_gru_{seed}']=all(v==inits[('C_hazard',seed)][k] for k,v in inits[('C_xyz_delta',seed)].items() if k.startswith('gru.'))
    status='pass' if all(all(r['checks'].values()) for r in records) and all(initialization.values()) else 'fail'
    train.atomic_json(a.output/'TRAINING_AUDIT.json',{'schema':'round8_training_audit_v1','source_sha256':train.sha256(Path(__file__)),'inventory_sha256':train.sha256(a.inventory),'status':status,'runs':records,'run_count':len(records),'frozen_inputs':frozen,'initialization':initialization})
    index_temp=a.output/'CHECKPOINT_INDEX.csv.tmp'
    with index_temp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(index[0]));writer.writeheader();writer.writerows(index)
    index_temp.replace(a.output/'CHECKPOINT_INDEX.csv')
    train.atomic_json(a.output/'RUN_INVENTORY.json',{'status':'complete' if status=='pass' else 'audit_failed','runs':inv['runs'],'formal_runs':len(records),'groups':inv['groups']})
    pairs=[['C_xyz_delta','B_xyz'],['C_hazard','C_xyz_delta']]
    if 'P3_fusion_independent' in inv['groups']:
        pairs += [['P3_fusion_independent','P3_concat_independent'],['P3_fusion_independent','C_xyz_delta'],['P3_fusion_hazard','P3_fusion_independent'],['P3_fusion_hazard','C_hazard']]
    if 'P4_state' in inv['groups']:
        pairs += [['P4_state','P4_direct']]
    if status!='pass':
        train.atomic_json(manifest_path,{'schema':'round8_prediction_manifest_v1','status':'audit_failed'})
        raise SystemExit(1)
    train.atomic_json(manifest_path,{'status':'complete','checkpoint_index':{'path':str((a.output/'CHECKPOINT_INDEX.csv').resolve()),'sha256':train.sha256(a.output/'CHECKPOINT_INDEX.csv')},'training_audit':{'path':str((a.output/'TRAINING_AUDIT.json').resolve()),'sha256':train.sha256(a.output/'TRAINING_AUDIT.json')},'schema':'round8_prediction_manifest_v1','groups':inv['groups'],'seeds':[20260914,20260915,20260916],'horizons':[1,3,5], 'artifacts':manifests,'comparisons':[{'candidate':a,'comparator':b} for a,b in pairs],'prepared':{'path':inv['prepared_data'],'sha256':inv['prepared_sha256']}})
    print(json.dumps({'status':status,'runs':len(records)}))
    if status!='pass':
        raise SystemExit(1)


if __name__=='__main__':
    main()
