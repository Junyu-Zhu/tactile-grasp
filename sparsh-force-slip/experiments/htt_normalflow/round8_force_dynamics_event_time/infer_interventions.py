#!/usr/bin/env python3
"""Post-training causal-input diagnostics; never recalibrate on interventions."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import torch
import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('r8_intervention_train',HERE/'training/train.py')
train=importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--accepted-manifest',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    data=torch.load(args.data,map_location='cpu',weights_only=False)
    inv=json.loads(args.inventory.read_text())
    accepted=json.loads(args.accepted_manifest.read_text())
    proof=accepted.get('training_audit',{})
    if accepted.get('status')!='complete' or train.sha256(Path(proof['path']))!=proof['sha256'] or json.loads(Path(proof['path']).read_text()).get('status')!='pass':
        raise ValueError('Accepted training audit required')
    if Path(accepted['prepared']['path']).resolve()!=args.data.resolve() or accepted['prepared']['sha256']!=train.sha256(args.data):
        raise ValueError('Accepted prepared identity mismatch')
    if json.loads(Path(proof['path']).read_text())['inventory_sha256']!=train.sha256(args.inventory):
        raise ValueError('Accepted inventory identity mismatch')
    import csv
    index_proof=accepted['checkpoint_index']
    if train.sha256(Path(index_proof['path']))!=index_proof['sha256']:
        raise ValueError('Accepted checkpoint index drift')
    index=list(csv.DictReader(Path(index_proof['path']).open()))
    expected={(g,s) for g in accepted['groups'] for s in accepted['seeds']}
    if len(inv['runs'])!=len(expected) or {(r['group'],r['seed']) for r in inv['runs']}!=expected:
        raise ValueError('Incomplete or duplicate run grid')
    rows=data['timelines']['outer']
    artifacts=[]
    parity=[]
    for run in inv['runs']:
        group,seed=run['group'],run['seed']
        summary=json.loads((Path(run['output'])/'summary.json').read_text())
        if summary.get('status')!='complete' or summary['group']!=group or summary['seed']!=seed:
            raise ValueError('Unaccepted or mismatched run')
        parent=Path(summary['artifacts']['best']['path'])
        if train.sha256(parent)!=summary['artifacts']['best']['sha256']:
            raise ValueError('Checkpoint drift')
        matches=[r for r in index if r['group']==group and int(r['seed'])==seed and r['kind']=='best']
        if len(matches)!=1 or Path(matches[0]['path']).resolve()!=parent.resolve() or matches[0]['sha256']!=train.sha256(parent):
            raise ValueError('Checkpoint not in accepted best index')
        state=torch.load(parent,map_location='cpu',weights_only=False)
        if state['run_config']['mode']!='formal' or state['run_config']['prepared_sha']!=train.sha256(args.data):
            raise ValueError('Incompatible formal checkpoint')
        model=train.make_model(group).eval().requires_grad_(False).to(args.device)
        model.load_state_dict(state['model_state'],strict=True)
        before={k:train.tensor_sha(v) for k,v in model.state_dict().items()}
        x=train.assemble_inputs(data,rows,group)
        base=train.infer(model,x,group,args.device,128)
        reference=Path(summary['artifacts']['predictions']['outer']['timeline']['path'])
        accepted_refs=[r for r in accepted['artifacts'] if r['group']==group and r['seed']==seed and r['role']=='outer']
        if len(accepted_refs)!=1 or Path(accepted_refs[0]['path']).resolve()!=reference.resolve() or accepted_refs[0]['sha256']!=train.sha256(reference) or accepted_refs[0]['head_type']!=train.head_type(group):
            raise ValueError('Reference timeline not in accepted manifest')
        reference_proof=summary['artifacts']['predictions']['outer']['timeline']
        if train.sha256(reference)!=reference_proof['sha256']:
            raise ValueError('Reference timeline drift')
        import csv
        records=list(csv.DictReader(reference.open()))
        if len(records)!=len(rows['t']) or any(r['episode_id']!=rows['episode_id'][i] or int(r['t'])!=int(rows['t'][i]) for i,r in enumerate(records)):
            raise ValueError('Timeline row identity mismatch')
        expected=np.asarray([[float(r[f'p_future_H{h}_raw']) for h in (1,3,5)] for r in records])
        error=float(np.max(np.abs(expected-base['probabilities'])))
        if error>1e-6:
            raise ValueError(f'Unperturbed inference parity failed {group}/{seed}: {error}')
        parity.append({'group':group,'seed':seed,'max_abs':error})
        kinds=['force_input_fit_mean_zero','force_causal_lag1']
        if group=='P4_state':
            kinds.append('state_input_fit_mean_zero')
        for kind in kinds:
            altered=x.clone()
            hook=None
            if kind=='force_input_fit_mean_zero':
                altered[:,:,769:775]=0
            elif kind=='force_causal_lag1':
                # Lag only within available history; never import an earlier frame.
                altered[:,1:,769:772]=x[:,:-1,769:772]
                altered[:,5,772:775]=0
                altered[:,6:,772:775]=x[:,5:-1,772:775]
            else:
                def zero_state(module,inputs):
                    value=inputs[0].clone()
                    value[:,-15:]=0
                    return (value,)
                hook=model.risk.register_forward_pre_hook(zero_state)
            pred=train.infer(model,altered,group,args.device,128)
            if hook is not None:
                hook.remove()
            path=args.output/f'{group}_{seed}_{kind}_outer.csv'
            saved=train.atomic_predictions(path,data,rows,pred,group,timeline=True)
            artifacts.append({'intervention':kind,'group':group,'seed':seed,'role':'outer',
                              'path':saved['path'],'sha256':saved['sha256'],'head_type':train.head_type(group),
                              'checkpoint_sha256':train.sha256(parent)})
        if any(train.tensor_sha(v)!=before[k] for k,v in model.state_dict().items()):
            raise ValueError('Diagnostic model changed')
        del model
    result={'schema':'round8_intervention_manifest_v1','status':'complete','artifacts':artifacts,'unperturbed_parity':parity,
            'checkpoint_index_sha256':index_proof['sha256'],'accepted_manifest_sha256':train.sha256(args.accepted_manifest),'inventory_sha256':train.sha256(args.inventory),'trainer_sha256':train.sha256(HERE/'training/train.py'),'head_parameters_unchanged':True,'prepared_sha256':train.sha256(args.data),'source_sha256':train.sha256(Path(__file__)),
            'threshold_rule':'use only unperturbed selection/calibration choices; no refit',
            'limitations':'force/lag interventions may be out-of-distribution; not causal proof; first history force remains unchanged under lag; common validity stays fixed'}
    train.atomic_json(args.output/'INTERVENTION_MANIFEST.json',result)
    print(json.dumps({'status':'complete','artifacts':len(artifacts)}))


if __name__=='__main__':
    main()
