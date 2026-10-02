#!/usr/bin/env python3
import argparse, hashlib, json, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
ap=argparse.ArgumentParser(); ap.add_argument('--require-sync',action='store_true'); args=ap.parse_args()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

bad=[]; unavailable=[]
src=json.loads((HERE/'SOURCE_INDEX.json').read_text())
for k,x in src['inputs'].items():
    p=REPO/x['repo_relative_path']
    if not p.exists(): unavailable.append(k); bad.append({'kind':'missing_source','key':k})
    elif sha(p)!=x['sha256']: bad.append({'kind':'source','key':k})
gen=json.loads((HERE/'SYNTHESIS_OUTPUTS.json').read_text())
for n,x in gen['outputs'].items():
    p=HERE/n
    if not p.exists() or sha(p)!=x['sha256']: bad.append({'kind':'generated','file':n})
idx=json.loads((HERE/'ARTIFACT_INDEX.json').read_text())
for x in idx['files']:
    p=HERE/x['relative_path']
    if not p.exists() or sha(p)!=x['sha256'] or p.stat().st_size!=x['bytes']: bad.append({'kind':'artifact','file':x['relative_path']})
snap=json.loads((HERE/'SOURCE_SNAPSHOT_MANIFEST.json').read_text())
for x in snap['snapshots']:
    p=HERE/x['snapshot']
    if not p.exists() or sha(p)!=x['sha256']: bad.append({'kind':'source_snapshot','file':x['snapshot']})
code=(HERE/'synthesize.py').read_text()
for forbidden in ('import torch','torch.load(','subprocess','ssh ','rsync '):
    if forbidden in code: bad.append({'kind':'scope_guard','token':forbidden})
required={'PROTOCOL.md','HANDOFF.md','COMPONENT_IDENTITY_MATRIX.csv','CROSS_ROUND_COMPARABILITY.md','CLAIMS_EVIDENCE.md','FORCE_EVIDENCE.csv','LEGACY_FORCE_TASK_REGRESSION.csv','SLIP_EVIDENCE.csv','SLIP_FOLD_SEED_FIXED_WORKPOINTS.csv','FUTURE_FORCE_EVIDENCE.csv','FAILURE_ANALYSIS_AND_GAPS.md','Q2_Q4_DISPOSITION.md','SUMMARY_ZH.md','REPRODUCE.md','EXECUTION_METADATA.json'}
for n in sorted(required):
    if not (HERE/n).exists(): bad.append({'kind':'required','file':n})
if args.require_sync:
    p=HERE/'SYNC_PROOF.json'
    if not p.exists(): bad.append({'kind':'sync','reason':'missing SYNC_PROOF.json'})
    else:
        z=json.loads(p.read_text())
        if z.get('status')!='pass' or z.get('files')!=z.get('matches'): bad.append({'kind':'sync','reason':'proof status/count'})
        for x in z.get('artifacts',[]):
            q=HERE/x['relative_path']
            if not q.exists() or sha(q)!=x['local_sha256'] or x['local_sha256']!=x['server_sha256'] or not x['match']: bad.append({'kind':'sync','file':x['relative_path']})
print(json.dumps({'status':'pass' if not bad else 'fail','sources':len(src['inputs']),'source_unavailable_on_this_root':unavailable,'generated':len(gen['outputs']),'indexed':len(idx['files']),'bad':bad},ensure_ascii=False))
sys.exit(bool(bad))
