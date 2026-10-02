#!/usr/bin/env python3
"""Prespecified best/worst h10 true-change trial cases across all seeds."""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from train import GROUPS, atomic_json, sha

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import adapters


def csv_rows(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--prepare',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    trial=csv_rows(a.evaluation/'metrics'/'TRIAL_METRICS.csv')
    prepare=json.loads(a.prepare.read_text());support=json.loads(Path(prepare['caches'][0]['support']).read_text())
    entries={e['episode_id']:e for e in support['entries']}
    grouped={};candidate_rows=[]
    for group in GROUPS:
        scores={}
        for seed in (20260914,20260915,20260916):
            hold={(r['episode_id'],int(r['axis'])):r for r in trial if r['role']=='validation' and r['group']=='hold' and int(r['seed'])==seed and int(r['horizon'])==10}
            model={(r['episode_id'],int(r['axis'])):r for r in trial if r['role']=='validation' and r['group']==group and int(r['seed'])==seed and int(r['horizon'])==10}
            for key,r in model.items():
                scores.setdefault(key[0],[]).append(float(hold[key]['change_mae'])-float(r['change_mae']))
        if any(len(v)!=9 for v in scores.values()):raise ValueError('case trial missing seed/axis')
        grouped[group]={eid:float(np.mean(v)) for eid,v in scores.items()}
        for eid,values in sorted(scores.items()):
            candidate_rows.append({'group':group,'episode_id':eid,
                                   'all_seed_axis_mean_hold_minus_model_change_mae_n':float(np.mean(values)),
                                   'seed_axis_differences_n':json.dumps(values)})
    with (a.output/'CANDIDATES.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(candidate_rows[0]));w.writeheader();w.writerows(candidate_rows)
    selections=[]
    for group in GROUPS:
        ordered=sorted(grouped[group].items(),key=lambda x:(x[1],x[0]))
        selections.append({'group':group,'kind':'failure','episode_id':ordered[0][0],
                           'all_seed_axis_mean_hold_minus_model_change_mae_n':ordered[0][1]})
        selections.append({'group':group,'kind':'success','episode_id':ordered[-1][0],
                           'all_seed_axis_mean_hold_minus_model_change_mae_n':ordered[-1][1]})
    artifacts=[]
    for idx,case in enumerate(selections,1):
        eid=case['episode_id'];group=case['group'];paths=[]
        for seed in (20260914,20260915,20260916):
            npz=np.load(a.evaluation/'predictions'/f'{group}_p1_s{seed}_validation.npz')
            ix=np.flatnonzero(npz['episode_id']==eid)
            if not len(ix):raise ValueError('missing selected trial')
            t=npz['t'][ix];gt=npz['gt'][ix,2];anchor=npz['anchor'][ix]
            pred=npz['pred_future'][ix,2];truth_change=npz['true_delta'][ix,2]
            pred_change=npz['pred_delta'][ix,2]
            rows=[]
            for j,frame in enumerate(t):
                row={'t':int(frame)}
                for axis,name in enumerate('xyz'):
                    row.update({f'gt_future_{name}':float(gt[j,axis]),f'anchor_{name}':float(anchor[j,axis]),
                                f'pred_future_{name}':float(pred[j,axis]),
                                f'true_change_{name}':float(truth_change[j,axis]),
                                f'pred_change_{name}':float(pred_change[j,axis])})
                rows.append(row)
            csv_path=a.output/f'case_{idx}_s{seed}.csv'
            with csv_path.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            fig,axes=plt.subplots(3,2,figsize=(12,9),sharex=True)
            for axis,name in enumerate('xyz'):
                axes[axis,0].plot(t,gt[:,axis],label='GT future');axes[axis,0].plot(t,anchor[:,axis],label='current anchor')
                axes[axis,0].plot(t,pred[:,axis],label='pred future');axes[axis,0].set_ylabel(f'{name} N')
                axes[axis,1].plot(t,truth_change[:,axis],label='true change')
                axes[axis,1].plot(t,pred_change[:,axis],label='pred change')
            axes[0,0].legend();axes[0,1].legend();axes[2,0].set_xlabel('native current frame t');axes[2,1].set_xlabel('native current frame t')
            fig.suptitle(f"{case['kind']} {group} p1 s{seed} {eid} h10")
            fig.tight_layout();plot=a.output/f'case_{idx}_s{seed}.svg';fig.savefig(plot);plt.close(fig)
            paths.append({'seed':seed,'curves_csv':str(csv_path),'plot_svg':str(plot)})
        obj=adapters.load_htt(entries[eid]['source_path']);center=int(np.median(t))
        fig,axes=plt.subplots(1,2,figsize=(8,4))
        axes[0].imshow(obj.images[center]);axes[0].set_title(f'raw tactile t={center}')
        axes[1].imshow(obj.reference.astype(np.uint8));axes[1].set_title('raw reference')
        for ax in axes:ax.axis('off')
        fig.tight_layout();raw=a.output/f'case_{idx}_source.png';fig.savefig(raw,dpi=160);plt.close(fig)
        artifacts.append({**case,'source_path':entries[eid]['source_path'],'raw_image':str(raw),'seed_artifacts':paths})
    atomic_json({'schema':'round20_cases_v1','selection_rule':'per group best/worst complete validation trial by all-three-seed and axis mean h10 true-change MAE improvement over deployable hold; episode ID tie break',
                 'cases':artifacts,'test_consumed':False},a.output/'CASES.json')
    print(json.dumps({'status':'complete','cases':len(artifacts),'seed_curves':sum(len(x['seed_artifacts']) for x in artifacts)}))

if __name__=='__main__':main()
