#!/usr/bin/env python3
"""Independent artifact-level consistency checks for R20 delivery."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from train import GROUPS, SEEDS, atomic_json, sha, state_hash


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--code',type=Path,required=True);a=p.parse_args();root=a.root
    prep=json.loads((root/'PREPARE.json').read_text());queue=json.loads((root/'QUEUE_RESULT.json').read_text())
    if queue['status']!='complete' or queue['verified_complete']!=12:raise ValueError('queue incomplete')
    runs=[]
    for group in GROUPS:
        for seed in SEEDS:
            path=root/'formal'/f'{group}_p1_s{seed}'
            summary=json.loads((path/'summary.json').read_text())
            latest=torch.load(path/'latest.pth',map_location='cpu',weights_only=False)
            best=torch.load(path/'best.pth',map_location='cpu',weights_only=False)
            cache=next(c for c in prep['caches'] if c['seed']==seed)
            identity=summary['identity']
            expected={'group':group,'seed':seed,'fold':1,'cache_sha256':cache['sha256'],
                      'source_sha256':sha(a.code/'train.py'),'protocol_sha256':sha(a.code/'PROTOCOL.md')}
            if any(identity.get(k)!=v for k,v in expected.items()):raise ValueError('run identity')
            if latest['identity']!=identity or best['identity']!=identity:raise ValueError('checkpoint identity')
            if state_hash(best['model'])!=state_hash(latest['best_state']):raise ValueError('best alias mismatch')
            metrics=[x['selection_true_delta_mae_n'] for x in latest['history']]
            best_epoch=int(np.argmin(metrics))
            if best_epoch!=summary['best_epoch'] or abs(min(metrics)-summary['best_metric_true_delta_n_mae'])>1e-9:
                raise ValueError('selection min mismatch')
            runs.append({'group':group,'seed':seed,'epochs':summary['epochs'],'best_epoch':best_epoch,
                         'best_metric_n':summary['best_metric_true_delta_n_mae'],
                         'best_sha256':sha(path/'best.pth'),'latest_sha256':sha(path/'latest.pth')})
    predictions=0;max_identity_error=0.;nonfinite=0
    for group in ('hold','linear','ideal_GT_hold')+GROUPS:
        for seed in SEEDS:
            for role in ('calibration','validation'):
                z=np.load(root/'evaluation'/'predictions'/f'{group}_p1_s{seed}_{role}.npz')
                n=len(z['t'])
                if n!=2712:raise ValueError('endpoint count')
                keys=list(zip(z['episode_id'].tolist(),z['t'].tolist()))
                if len(set(keys))!=n:raise ValueError('duplicate endpoint')
                for name in ('anchor','gt','true_delta','pred_delta','pred_future','current_gt'):
                    nonfinite+=int((~np.isfinite(z[name])).sum())
                err=np.max(np.abs(z['pred_future']-z['gt']-
                                  ((z['anchor']-z['current_gt'])[:,None,:]+(z['pred_delta']-z['true_delta']))))
                max_identity_error=max(max_identity_error,float(err))
                if not np.allclose(z['true_delta'],z['gt']-z['current_gt'][:,None,:],atol=2e-6):
                    raise ValueError('true delta mismatch')
                if group=='hold' and (np.count_nonzero(z['pred_delta']) or not np.array_equal(z['pred_future'],z['anchor'][:,None,:]+z['pred_delta'])):
                    raise ValueError('hold mismatch')
                if group=='ideal_GT_hold' and (np.count_nonzero(z['pred_delta']) or not np.array_equal(z['anchor'],z['current_gt'])):
                    raise ValueError('ideal hold mismatch')
                predictions+=1
    if nonfinite or max_identity_error>2e-5:raise ValueError('prediction numeric failure')
    ci=list(csv.DictReader(open(root/'report'/'PAIRED_CI.csv')))
    if len(ci)!=384 or any(int(r['valid_bootstrap'])!=2000 or int(r['invalid_bootstrap'])!=0 for r in ci):
        raise ValueError('CI incomplete')
    strata_ci=list(csv.DictReader(open(root/'report'/'STRATA_PAIRED_CI.csv')))
    if len(strata_ci)!=168 or any(int(r['valid_bootstrap'])!=2000 or int(r['invalid_bootstrap'])!=0 for r in strata_ci):
        raise ValueError('strata CI incomplete')
    numeric=list(csv.DictReader(open(root/'evaluation'/'metrics'/'NUMERIC_AXIS_WINDOW.csv')))
    if len(numeric)!=216:raise ValueError('axis/window numeric checks incomplete')
    cases=json.loads((root/'cases'/'CASES.json').read_text())
    candidates=list(csv.DictReader(open(root/'cases'/'CANDIDATES.csv')))
    if len(cases['cases'])!=8 or len(candidates)!=48 or any(len(x['seed_artifacts'])!=3 for x in cases['cases']):
        raise ValueError('cases incomplete')
    result={'schema':'round20_delivery_audit_v1','status':'pass','runs':runs,'prediction_arrays':predictions,
            'max_error_decomposition_abs_n':max_identity_error,'nonfinite_values':nonfinite,
            'paired_ci_rows':len(ci),'strata_paired_ci_rows':len(strata_ci),
            'numeric_axis_window_rows':len(numeric),'valid_draws_per_row':2000,
            'case_candidates':len(candidates),'cases':len(cases['cases']),
            'test_consumed':False}
    atomic_json(result,root/'AUDIT.json')
    print(json.dumps({'status':'pass','runs':len(runs),'prediction_arrays':predictions,
                      'max_identity_error_n':max_identity_error,'ci_rows':len(ci)}))

if __name__=='__main__':main()
