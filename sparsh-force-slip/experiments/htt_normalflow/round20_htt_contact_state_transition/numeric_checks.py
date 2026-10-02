#!/usr/bin/env python3
"""Per-axis/window output diversity, copying, clipping and range flags."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from train import GROUPS


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed in (20260914,20260915,20260916):
        for role in ('calibration','validation'):
            for group in GROUPS:
                z=np.load(a.evaluation/'predictions'/f'{group}_p1_s{seed}_{role}.npz')
                for j,h in enumerate((1,5,10)):
                    for axis,name in enumerate('xyz'):
                        pred=z['pred_delta'][:,j,axis];true=z['true_delta'][:,j,axis]
                        future=z['pred_future'][:,j,axis];gt=z['gt'][:,j,axis]
                        pvar=float(pred.var());tvar=float(true.var())
                        rows.append({'seed':seed,'role':role,'group':group,'horizon':h,'axis':name,
                                     'n':len(pred),'pred_delta_variance_n2':pvar,
                                     'true_delta_variance_n2':tvar,
                                     'variance_ratio':pvar/tvar if tvar>0 else '',
                                     'std_collapse_flag_lt_0p1_gt':int(pvar<.01*tvar),
                                     'copy_fraction_abs_delta_le_0p01':float((np.abs(pred)<=.01).mean()),
                                     'pred_future_outside_20_fraction':float((np.abs(future)>20).mean()),
                                     'gt_future_at_clip_fraction':float((np.abs(gt)>=20).mean()),
                                     'pred_future_abs_max_n':float(np.abs(future).max()),
                                     'true_delta_abs_max_n':float(np.abs(true).max())})
    with a.output.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({'status':'complete','rows':len(rows),'std_flags':sum(x['std_collapse_flag_lt_0p1_gt'] for x in rows)}))

if __name__=='__main__':main()
