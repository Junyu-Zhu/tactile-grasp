#!/usr/bin/env python3
"""Prespecified paired trial bootstrap and descriptive R20 comparison."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from train import atomic_json

COMPARISONS=(('D-history','K-current'),('T-force','T-visual'),('T-force','D-history'),
             ('K-current','hold'),('D-history','hold'),('T-force','hold'),
             ('D-history','linear'),('T-force','linear'))
METRICS=('future_mae','change_mae','future_rmse','change_rmse')


def read_csv(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    trial=read_csv(a.evaluation/'metrics'/'TRIAL_METRICS.csv')
    run=read_csv(a.evaluation/'metrics'/'RUN_METRICS.csv')
    val=[r for r in trial if r['role']=='validation']
    groups=sorted({r['leakage_group'] for r in val})
    if len(groups)<3:raise ValueError('insufficient independent groups')
    rng=np.random.default_rng(20260920)
    draws=rng.integers(0,len(groups),size=(2000,len(groups)))
    indexed={}
    for r in val:
        key=(int(r['seed']),r['group'],int(r['horizon']),int(r['axis']),r['leakage_group'])
        if key in indexed:raise ValueError('multiple trials per leakage group require explicit aggregation')
        indexed[key]=r
    def point(seed,group,h,axis,leakage,metric):
        axes=(0,1,2) if axis=='all' else (axis,)
        records=[indexed.get((seed,group,h,a,leakage)) for a in axes]
        if any(r is None for r in records):return None
        values=np.array([float(r[metric]) for r in records])
        value=np.sqrt(np.mean(values*values)) if metric.endswith('rmse') else np.mean(values)
        return float(value),int(records[0]['n'])
    seeds=sorted({int(r['seed']) for r in val})
    def paired_estimate(lhs,rhs,weights,metric):
        if metric.endswith('rmse'):
            return float(np.sqrt(np.average(np.square(lhs),weights=weights))-
                         np.sqrt(np.average(np.square(rhs),weights=weights)))
        return float(np.average(lhs-rhs,weights=weights))
    rows=[]
    for numerator,denominator in COMPARISONS:
        for h in (1,5,10):
            for axis in (0,1,2,'all'):
                for metric in METRICS:
                    per_seed=[];draw_values=[];invalid=0;paired={}
                    for seed in seeds:
                        lhs_values=[];rhs_values=[];weights=[]
                        for group in groups:
                            lhs=point(seed,numerator,h,axis,group,metric);rhs=point(seed,denominator,h,axis,group,metric)
                            if lhs is None or rhs is None:raise ValueError('unpaired leakage groups')
                            lhs_values.append(lhs[0]);rhs_values.append(rhs[0]);weights.append(lhs[1])
                        paired[seed]=(np.array(lhs_values),np.array(rhs_values),np.array(weights))
                        per_seed.append(paired_estimate(*paired[seed],metric))
                    for ids in draws:
                        values=[]
                        for seed in seeds:
                            lhs,rhs,weight=paired[seed]
                            if not len(ids):break
                            values.append(paired_estimate(lhs[ids],rhs[ids],weight[ids],metric))
                        if len(values)==len(seeds):draw_values.append(float(np.mean(values)))
                        else:invalid+=1
                    rows.append({'comparison':f'{numerator}−{denominator}','horizon':h,'axis':axis,
                                 'metric':metric,'seed_estimates':json.dumps(per_seed),
                                 'seed_mean':float(np.mean(per_seed)),
                                 'ci_low':float(np.quantile(draw_values,.025)) if draw_values else '',
                                 'ci_high':float(np.quantile(draw_values,.975)) if draw_values else '',
                                 'valid_bootstrap':len(draw_values),'invalid_bootstrap':invalid,
                                 'independent_validation_groups':len(groups)})
    write_csv(a.output/'PAIRED_CI.csv',rows)
    summary={}
    for group in ('hold','linear','ideal_GT_hold','K-current','D-history','T-visual','T-force'):
        subset=[r for r in run if r['role']=='validation' and r['group']==group and r['kind'] in ('future','true_change')
                and r['stratum']=='all']
        summary[group]={kind:{str(h):float(np.mean([float(r['mae']) for r in subset if r['kind']==kind and int(r['horizon'])==h]))
                              for h in (1,5,10)} for kind in ('future','true_change')}
    atomic_json({'schema':'round20_report_v1','summary_validation_mae_n':summary,
                 'bootstrap':{'unit':'complete validation leakage group','draws':2000,
                              'shared_draws':True,'seed_handling':'mean within draw, descriptive only',
                              'groups':len(groups),'valid':min(r['valid_bootstrap'] for r in rows),
                              'invalid':max(r['invalid_bootstrap'] for r in rows)},
                 'comparisons':[f'{a}−{b}' for a,b in COMPARISONS],
                 'single_fold_development_only':True},a.output/'SUMMARY.json')
    print(json.dumps({'status':'complete','ci_rows':len(rows),'groups':len(groups),'summary':summary}))

if __name__=='__main__':main()
