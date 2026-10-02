#!/usr/bin/env python3
"""Bind 24 accepted R9 models and 12 accepted R10 fusion models to identical endpoints."""
import argparse,json
from pathlib import Path
import evaluate as e

def record(p):return {'path':str(p),'sha256':e.sha(p)}
def main():
 p=argparse.ArgumentParser()
 for key in ('r9-audit','r10-audit','r9-prepare-root','r10-prepare-root','output'):p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args();audits={'round9':json.loads(a.r9_audit.read_text()),'round10':json.loads(a.r10_audit.read_text())};runs=[]
 for group in e.GROUPS:
  source='round10' if group=='F_history_new' else 'round9';source_group='V_temporal' if group=='V_temporal' else 'F_history';audit_group=group if source=='round10' else source_group
  for f in range(1,5):
   for seed in e.SEEDS:
    fold=f'htt_leave_p{f}';found=[r for r in audits[source]['accepted_runs'] if (r['group'],r['fold'],r['seed'])==(audit_group,fold,seed)]
    if len(found)!=1:raise ValueError('Missing/duplicate accepted run')
    r=found[0];endpoints={}
    for role in ('calibration','validation'):
     old=a.r9_prepare_root/f'p{f}_s{seed}'/f'endpoints_{role}.csv';new=a.r10_prepare_root/f'p{f}_s{seed}'/f'endpoints_{role}.csv'
     if e.sha(old)!=e.sha(new):raise ValueError('Changed common endpoints')
     endpoints[role]=record(old)
    runs.append(dict(group=group,fold=fold,seed=seed,source_round=source,source_group=source_group,training_summary=r['training_summary'],checkpoint=r['checkpoint'],predictions={role:r['predictions'][role] for role in endpoints},endpoints=endpoints))
 e.js(a.output,dict(schema='round10_evaluation_manifest_v1',status='complete',synthetic=False,training_audits={'round9':record(a.r9_audit),'round10':record(a.r10_audit)},evaluation_source_sha256=e.sha(Path(__file__).with_name('evaluate.py')),evaluation_protocol_sha256=e.sha(Path(__file__).with_name('PROTOCOL.md')),builder_source_sha256=e.sha(__file__),runs=runs))
if __name__=='__main__':main()
