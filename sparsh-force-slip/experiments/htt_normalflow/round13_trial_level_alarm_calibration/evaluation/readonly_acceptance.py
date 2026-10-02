#!/usr/bin/env python3
"""Read-only completeness and hash acceptance for a synchronized R13 delivery."""
import argparse, csv, hashlib, json
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def rows(p):
    with p.open() as f: return sum(1 for _ in csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--delivery',type=Path,required=True); ap.add_argument('--manifest',type=Path); a=ap.parse_args()
    d=a.delivery
    summary=json.loads((d/'results/SUMMARY.json').read_text())
    expected={
        'results/historical/metrics_48.csv':1440,
        'results/new_policies/thresholds.csv':864,
        'results/new_policies/metrics.csv':2592,
        'results/new_policies/calibration_bootstrap.csv':172800,
        'results/new_policies/confirm4_metrics.csv':432,
    }
    counts={k:rows(d/k) for k in expected}
    failures=[f'{k}: {counts[k]} != {v}' for k,v in expected.items() if counts[k]!=v]
    if summary.get('new_neural_trainings')!=0 or summary.get('test_role_consumed') is not False or summary.get('validation_used_for_threshold_fitting') is not False:
        failures.append('scope guard failed')
    if a.manifest:
        manifest=json.loads(a.manifest.read_text())
        for rec in manifest['files']:
            p=d/rec['relative_path']
            if not p.is_file() or sha(p)!=rec['sha256']: failures.append(f'hash mismatch {p}')
    result={'status':'pass' if not failures else 'fail','counts':counts,'failures':failures}
    print(json.dumps(result,indent=2))
    if failures: raise SystemExit(1)
if __name__=='__main__': main()
