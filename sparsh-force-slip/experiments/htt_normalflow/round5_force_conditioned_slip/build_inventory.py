#!/usr/bin/env python3
"""Materialize fixed formal commands; never launches a process."""
import json,hashlib
from pathlib import Path
ROOT = Path('/home/zjy/document/tactile-grasp/sparsh-force-slip')
CODE = ROOT/'experiments/htt_normalflow/round5_force_conditioned_slip'
BASE = Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow')
OUT = BASE/'round5_force_conditioned_slip'
PY = '/home/zjy/miniconda3/envs/sparsh/bin/python'
OLD = '/vla1/zjy/sparsh_runs/force_slip_phase2/phase_lambda_decoupled_mae_lam010_20260528_000000/mae_decoupled_multitask/checkpoints/epoch-0030.pth'
SEEDS = (20260914,20260915,20260916)

def main():
    jobs=[]
    common=['--cache',str(OUT/'data/contract/contract.json'),'--cache-audit',str(OUT/'data/CACHE_HASH_AUDIT.json'),'--source-checkpoint',OLD]
    def add(ident,script,args,deps,accept,kind='neural'):
        jobs.append(dict(id=ident,kind=kind,argv=[PY,str(CODE/script),*map(str,args)],depends_on=deps,acceptance_path=str(accept),log_path=str(OUT/'logs/formal'/f'{ident}.log'),status='planned_not_started'))
    for fold in range(1,5):
        for seed in SEEDS:
            key=f'p{fold}_s{seed}'; role=['--fold',f'htt_leave_p{fold}','--seed',str(seed)]
            force=OUT/'formal/force'/key
            add('force_'+key,'current/train_force.py',common+role+['--output',force],[],force/'training_summary.json')
            predictions={}
            for variant in ('old','adapt'):
                pred=OUT/'formal/force_predictions'/variant/key; predictions[variant]=pred/'prediction_manifest.json'
                extra=['--force-checkpoint',force/'best.pth'] if variant=='adapt' else []
                add('predict_'+variant+'_'+key,'current/predict_force.py',common+role+['--variant',variant,'--output',pred]+extra,['force_'+key] if variant=='adapt' else [],pred/'prediction_manifest.json','cache')
            base=BASE/f'round3_mae_slip_adaptation/runs/B/fold_p{fold}/seed_{seed}/best.pth'
            add('evaluate_mae_'+key,'evaluate_slip.py',['--calibration',base.parent/'predictions/calibration.csv','--validation',base.parent/'predictions/validation.csv','--manifest',BASE/'round1/splits.json','--contract',OUT/'data/contract/contract.json','--cache-audit',OUT/'data/CACHE_HASH_AUDIT.json','--training-summary',base.parent/'training_summary.json','--model-id','mae-r3-b','--historical-baseline',*role,'--output',OUT/'formal/baseline_evaluation'/key],[],OUT/'formal/baseline_evaluation'/key/'metrics.json','evaluation')
            for variant in ('V','F-old','F-adapt'):
                run=OUT/'formal/slip'/variant/key; deps=[]; extra=[]
                if variant!='V':
                    source=variant[2:]; deps=['predict_'+source+'_'+key];extra=['--force-predictions',predictions[source]]
                ident='slip_'+variant+'_'+key
                add(ident,'current/train_slip.py',common+role+['--base-slip-checkpoint',base,'--variant',variant,'--output',run]+extra,deps,run/'training_summary.json')
                add('evaluate_'+variant+'_'+key,'evaluate_slip.py',['--calibration',run/'predictions_calibration.csv','--validation',run/'predictions_validation.csv','--manifest',BASE/'round1/splits.json','--contract',OUT/'data/contract/contract.json','--cache-audit',OUT/'data/CACHE_HASH_AUDIT.json','--training-summary',run/'training_summary.json','--model-id',variant,*role,'--output',run/'evaluation'],[ident],run/'evaluation/metrics.json','evaluation')
    protocol_file=Path(__file__).parent/'future/protocol.json'
    protocol_sha=hashlib.sha256(protocol_file.read_bytes()).hexdigest()
    future_common=['--protocol-sha256',protocol_sha]
    for seed in SEEDS:
        for variant in ('z_p_slip','z_p_slip_force','z_p_slip_force_pred_delta'):
            run=OUT/'formal/source_future'/variant/f'seed_{seed}'
            add(f'source_{variant}_{seed}','future/future_pipeline.py',['train-source','--train-cache',OUT/'future/cache/source_train_pred_delta_v2.pt','--validation-cache',OUT/'future/cache/source_val_pred_delta_v2.pt','--variant',variant,'--seed',seed,'--output',run,*future_common],[],run/'summary.json')
    key='p1_s20260914'
    contact=OUT/'formal/contact_upstream'
    add('export_contact','current/export_upstream.py',common+['--base-slip-checkpoint',BASE/'round3_mae_slip_adaptation/runs/B/fold_p1/seed_20260914/best.pth','--force-predictions',OUT/'formal/force_predictions/adapt'/key/'prediction_manifest.json','--fused-slip-checkpoint',OUT/'formal/slip/F-adapt'/key/'best.pth','--output',contact],['slip_F-adapt_'+key],contact/'prediction_manifest.json','cache')
    upstream=OUT/'future/cache/htt_formal_upstream.pt'
    add('assemble_contact','future/future_pipeline.py',['assemble-htt-upstream','--contract-json',OUT/'data/contract/contract.json','--prediction-manifest',contact/'prediction_manifest.json','--output',upstream],['export_contact'],upstream.with_suffix('.manifest.json'),'cache')
    for seed in SEEDS:
        for variant in ('risk','base','full_state'):
            run=OUT/'formal/htt_future'/variant/f'seed_{seed}'
            add(f'htt_future_{variant}_{seed}','future/future_pipeline.py',['train-htt','--upstream',upstream,'--support-audit',OUT/'future/htt/support_audit.json','--variant',variant,'--seed',seed,'--output',run,*future_common],['assemble_contact'],run/'summary.json')
    assert sum(j['kind']=='neural' for j in jobs)==66
    for job in jobs:
        if job['kind']=='neural':job['checkpoint_paths']=[str(Path(job['acceptance_path']).parent/name) for name in ('best.pth','latest.pth')]
    inv=dict(schema='round5_formal_dag_v1',preparation_status='pending_review',output_root=str(OUT),code_root=str(ROOT),deadline='2026-09-25T01:55:53+08:00',local_sync_proof=str(OUT/'LOCAL_SYNC_PROOF.json'),jobs=jobs)
    path=Path(__file__).parent/'RUN_INVENTORY.json';path.write_text(json.dumps(inv,indent=2)+'\n')
    print(path)
if __name__=='__main__':main()
