#!/usr/bin/env python3
"""Hash all selected-seed causal caches and freeze formal inventory."""
import argparse
import json
from pathlib import Path

import torch

from train import GROUPS, SEEDS, atomic_json, sha, validate_data

ROOT=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    caches=[];reference=None
    for seed in SEEDS:
        path=ROOT/f'round14_htt_future_force_dual/prepared/p1_s{seed}/prepared.pt'
        d=torch.load(path,map_location='cpu',weights_only=False)
        validate_data(d)
        keys={role:[(eid,int(t),g) for eid,t,g in zip(r['episode_id'],r['t'],r['leakage_group'])]
              for role,r in d['roles'].items()}
        if reference is None:reference=keys
        elif keys!=reference:raise ValueError('seed endpoint identity mismatch')
        caches.append({'seed':seed,'path':str(path),'sha256':sha(path),
                       'roles':{role:{'endpoints':len(rows),'trials':len({x[0] for x in rows}),
                                      'leakage_groups':len({x[2] for x in rows})} for role,rows in keys.items()},
                       'upstream':d['provenance']['immutable_upstream'],
                       'support':d['provenance']['support'],
                       'support_sha256':d['provenance']['support_sha256']})
    runs=[{'group':group,'fold':1,'seed':seed,'cache_sha256':next(c['sha256'] for c in caches if c['seed']==seed),
           'run_dir':str(a.output/'formal'/f'{group}_p1_s{seed}')}
          for group in GROUPS for seed in SEEDS]
    result={'schema':'round20_prepare_v1','selected_fold':1,'support_audit':'Q2_SUPPORT_AUDIT.json',
            'caches':caches,'runs':runs,'upstream_scope':'read-only, gradients absent; no R19 score selection',
            'target':'clipped_reference_relative_future_minus_clipped_reference_relative_current',
            'test_consumed':False}
    atomic_json(result,a.output/'PREPARE.json')
    print(json.dumps({'status':'pass','cache_count':len(caches),'runs':len(runs),
                      'roles':caches[0]['roles']}))

if __name__=='__main__':main()
