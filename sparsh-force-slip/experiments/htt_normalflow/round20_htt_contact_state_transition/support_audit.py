#!/usr/bin/env python3
"""Read-only audit of R10 labels on complete-window R14 endpoints."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow')
ROLES = ('fit', 'selection', 'calibration', 'validation')
LABELS = {0: 'static', 1: 'incipient', 2: 'gross'}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def audit_fold(fold):
    cache = ROOT / f'round14_htt_future_force_dual/prepared/p{fold}_s20260914/prepared.pt'
    support = ROOT / f'round10_htt_force_supervision_adaptation/force_support/fold_p{fold}.json'
    data = torch.load(cache, map_location='cpu', weights_only=False)
    sup = json.loads(support.read_text())
    if data['provenance']['support_sha256'] != sha(support):
        raise ValueError('support identity mismatch')
    entries = {e['episode_id']: e for e in sup['entries']}
    labels = {eid: np.load(e['label_path']) for eid, e in entries.items()}
    roles = {}
    seen = set()
    for role in ROLES:
        r = data['roles'][role]
        by_class = {i: {'frames': 0, 'trials': set(), 'groups': set(), 'probes': set()} for i in LABELS}
        unknown = 0
        endpoint_ts = defaultdict(set)
        for eid, t, group in zip(r['episode_id'], r['t'].tolist(), r['leakage_group']):
            t = int(t)
            e = entries[eid]
            if e['role'] != role or e['leakage_group'] != group:
                raise ValueError('role or group mismatch')
            if t < 13 or t + 10 >= e['frames']:
                raise ValueError('incomplete history/future')
            endpoint_ts[eid].add(t)
            label = int(labels[eid][t])
            if label not in LABELS:
                unknown += 1
                continue
            v = by_class[label]
            v['frames'] += 1
            v['trials'].add(eid)
            v['groups'].add(group)
            v['probes'].add(e['probe'])
        role_groups = {entries[eid]['leakage_group'] for eid in endpoint_ts}
        if seen & role_groups:
            raise ValueError('role leakage overlap')
        seen |= role_groups
        segments = []
        for eid, ts in endpoint_ts.items():
            ys = labels[eid]
            starts = np.flatnonzero((ys == 1) & np.r_[True, ys[:-1] != 1])
            for start in starts:
                end = int(start)
                while end + 1 < len(ys) and ys[end + 1] == 1:
                    end += 1
                eligible = [t for t in ts if start <= t <= end]
                segments.append({'episode_id': eid, 'start': int(start), 'end': end,
                                 'length_frames': end-int(start)+1, 'complete_window_frames': len(eligible),
                                 'fractional_start': float(start/max(1, len(ys)-1))})
        roles[role] = {'endpoints': sum(len(ts) for ts in endpoint_ts.values()),
                       'trials': len(endpoint_ts), 'leakage_groups': len(role_groups),
                       'unknown_label_frames': unknown,
                       'class_support': {LABELS[i]: {'frames': v['frames'], 'independent_trials': len(v['trials']),
                                                    'leakage_groups': len(v['groups']), 'probes': sorted(v['probes'])}
                                         for i, v in by_class.items()},
                       'incipient_segments': segments}
    return {'fold': fold, 'cache': str(cache), 'cache_sha256': sha(cache),
            'support': str(support), 'support_sha256': sha(support), 'roles': roles}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    folds = [audit_fold(i) for i in range(1, 5)]
    def eligible(f):
        for role, r in f['roles'].items():
            minimum = 10 if role == 'fit' else 3
            if r['leakage_groups'] < 3 or any(v['independent_trials'] < minimum for v in r['class_support'].values()):
                return False
        return True
    for f in folds:
        f['q3_fold_support_pass'] = eligible(f)
        f['q2_support_pass'] = eligible(f)
    chosen = next((f['fold'] for f in folds if f['q3_fold_support_pass']), None)
    out = {'schema': 'round20_support_audit_v1', 'criteria_prelocked_in': 'PROTOCOL.md',
           'q2_all_four_folds_pass': all(f['q2_support_pass'] for f in folds),
           'q2_training_24_configs_authorized': False, 'q3_selected_fold': chosen,
           'folds': folds,
           'limitations': ['incipient is a rule-derived transition label, not independent physical early-warning truth',
                           'independent trials/leakage groups, not repeated windows, determine thresholds',
                           'gross relabeling as incipient would not resolve false alarms']}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'q2_all_four_folds_pass': out['q2_all_four_folds_pass'],
                      'q3_selected_fold': chosen,
                      'folds': [{'fold': f['fold'], 'pass': f['q3_fold_support_pass'],
                                 'incipient_trials': {role: f['roles'][role]['class_support']['incipient']['independent_trials'] for role in ROLES}}
                                for f in folds]}))

if __name__ == '__main__': main()
