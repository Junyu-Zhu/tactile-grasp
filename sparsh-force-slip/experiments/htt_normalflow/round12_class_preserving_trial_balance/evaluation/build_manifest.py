#!/usr/bin/env python3
"""Bind 24 accepted new models and 48 historical models to all three common roles."""
import argparse,json
from pathlib import Path
import evaluate as e

def record(p):return {'path':str(p),'sha256':e.sha(p)}
def main():
 p=argparse.ArgumentParser()
 for key in ('r9-audit','r10-audit','r11-audit','r12-audit','r9-prepare-root','r10-prepare-root','output'):p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args();paths={'round9':a.r9_audit,'round10':a.r10_audit,'round11':a.r11_audit,'round12':a.r12_audit};audits={k:json.loads(v.read_text()) for k,v in paths.items()};runs=[]
 for group in e.GROUPS:
  source,source_group,audit_group=e.source_identity(group)
  for f in range(1,5):
   for seed in e.SEEDS:
    fold=f'htt_leave_p{f}';found=[r for r in audits[source]['accepted_runs'] if (r['group'],r['fold'],r['seed'])==(audit_group,fold,seed)]
    if len(found)!=1:raise ValueError('Missing/duplicate accepted run')
    r=found[0];endpoints={}
    for role in e.ROLES:
     old=a.r9_prepare_root/f'p{f}_s{seed}'/f'endpoints_{role}.csv';new=a.r10_prepare_root/f'p{f}_s{seed}'/f'endpoints_{role}.csv'
     if e.sha(old)!=e.sha(new):raise ValueError('Changed common endpoints')
     endpoints[role]=record(new)
    runs.append(dict(group=group,fold=fold,seed=seed,source_round=source,source_group=source_group,training_summary=r['training_summary'],checkpoint=r['checkpoint'],predictions={role:r['predictions'][role] for role in endpoints},endpoints=endpoints))
 manifest=dict(schema='round12_evaluation_manifest_v1',status='complete',synthetic=False,training_audits={k:record(v) for k,v in paths.items()},evaluation_source_sha256=e.sha(Path(__file__).with_name('evaluate.py')),evaluation_protocol_sha256=e.sha(Path(__file__).with_name('PROTOCOL.md')),builder_source_sha256=e.sha(__file__),runs=runs)
 e.validate_manifest(manifest);e.js(a.output,manifest)
if __name__=='__main__':main()
