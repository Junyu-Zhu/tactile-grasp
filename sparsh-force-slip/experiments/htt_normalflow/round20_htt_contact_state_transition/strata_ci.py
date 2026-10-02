#!/usr/bin/env python3
"""Shared complete-trial paired bootstrap for stable/transition/changing costs."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from train import GROUPS

PAIRS=(('D-history','K-current'),('T-force','T-visual'),('T-force','D-history'),
       ('K-current','hold'),('D-history','hold'),('T-visual','hold'),('T-force','hold'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    data={}
    for seed in (20260914,20260915,20260916):
        for group in ('hold',)+GROUPS:
            z=np.load(a.evaluation/'predictions'/f'{group}_p1_s{seed}_validation.npz')
            data[seed,group]={k:z[k] for k in ('episode_id','t','leakage_group','stratum','gt','true_delta','pred_future','pred_delta')}
    groups=sorted(set(data[20260914,'hold']['leakage_group']))
    rng=np.random.default_rng(20260920);draws=rng.integers(0,len(groups),size=(2000,len(groups)))
    rows=[]
    for lhs,rhs in PAIRS:
        for horizon_idx,h in enumerate((1,5,10)):
            for stratum in ('all','stable','transition','changing'):
                for kind in ('future','true_change'):
                    stats={};point=[]
                    for seed in (20260914,20260915,20260916):
                        l=data[seed,lhs];r=data[seed,rhs]
                        if not (np.array_equal(l['episode_id'],r['episode_id']) and np.array_equal(l['t'],r['t'])):
                            raise ValueError('unpaired endpoints')
                        target=l['gt'] if kind=='future' else l['true_delta']
                        lpred=l['pred_future'] if kind=='future' else l['pred_delta']
                        rpred=r['pred_future'] if kind=='future' else r['pred_delta']
                        err=np.mean(np.abs(lpred[:,horizon_idx]-target[:,horizon_idx])-
                                    np.abs(rpred[:,horizon_idx]-target[:,horizon_idx]),axis=1)
                        mask=np.ones(len(err),dtype=bool) if stratum=='all' else l['stratum']==stratum
                        count=np.array([np.sum(mask & (l['leakage_group']==g)) for g in groups])
                        sums=np.array([np.sum(err[mask & (l['leakage_group']==g)]) for g in groups])
                        stats[seed]=(sums,count)
                        point.append(float(sums.sum()/count.sum()) if count.sum() else np.nan)
                    values=[];invalid=0
                    for ids in draws:
                        seed_values=[]
                        for seed in (20260914,20260915,20260916):
                            sums,count=stats[seed]
                            if count[ids].sum()==0:break
                            seed_values.append(float(sums[ids].sum()/count[ids].sum()))
                        if len(seed_values)==3:values.append(float(np.mean(seed_values)))
                        else:invalid+=1
                    rows.append({'comparison':f'{lhs}−{rhs}','horizon':h,'stratum':stratum,'metric':kind+'_mae',
                                 'seed_estimates':json.dumps(point),'seed_mean':float(np.mean(point)),
                                 'ci_low':float(np.quantile(values,.025)) if values else '',
                                 'ci_high':float(np.quantile(values,.975)) if values else '',
                                 'valid_bootstrap':len(values),'invalid_bootstrap':invalid,
                                 'independent_validation_groups':len(groups)})
    with a.output.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({'status':'complete','rows':len(rows),'draws':2000,
                      'maximum_invalid_draws':max(r['invalid_bootstrap'] for r in rows)}))

if __name__=='__main__':main()
