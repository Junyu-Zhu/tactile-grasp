#!/usr/bin/env python3
"""Select bounded artifacts and verify their local byte identity after rsync."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['create','verify'])
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    a=p.parse_args()
    roots={'source':a.source.resolve(),'results':a.results.resolve()}
    if a.mode=='create':
        checkpoints=[]
        for f in sorted(a.results.rglob('*')):
            if f.is_file() and f.suffix in ['.pth','.pt']:
                rel=f.relative_to(a.results)
                formal=len(rel.parts)>2 and rel.parts[1]=='runs'
                checkpoints.append({'kind':'formal' if formal else 'smoke_proof_or_archive',
                                    'path':str(f.resolve()),'bytes':f.stat().st_size})
        r3=a.results.parent/'round3_mae_slip_adaptation/runs/B'
        for f in sorted(r3.glob('fold_p*/seed_*/*.pth')):
            checkpoints.append({'kind':'reused_mae_B','path':str(f.resolve()),'bytes':f.stat().st_size})
        with (a.results/'CHECKPOINTS.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=['kind','path','bytes'])
            writer.writeheader();writer.writerows(checkpoints)
        entries=[]; omitted=[]
        for scope,root in roots.items():
            for f in sorted(root.rglob('*')):
                if not f.is_file():continue
                r=f.relative_to(root)
                if any(x in r.parts for x in ['__pycache__','.pytest_cache','.git','delivery']):continue
                if scope=='source' and 'results' in r.parts:continue
                if f.name in ['DELIVERY_MANIFEST.json','LOCAL_SYNC_VERIFICATION.json','source_files.txt','results_files.txt']:continue
                reason=None
                if f.suffix in ['.pth','.pt']:reason='checkpoint remains server-side'
                elif 'cache' in r.parts and f.suffix not in ['.json','.md','.csv','.log','.txt']:reason='large cache remains server-side'
                elif f.stat().st_size>20*1024*1024:reason='artifact exceeds bounded mirror size'
                elif '.tmp' in f.name:reason='temporary file'
                if reason:
                    omitted.append({'scope':scope,'path':str(r),'bytes':f.stat().st_size,'reason':reason});continue
                entries.append({'scope':scope,'path':str(r),'bytes':f.stat().st_size,'sha256':sha(f)})
        a.manifest.write_text(json.dumps({'roots':{k:str(v) for k,v in roots.items()},'files':entries,'omitted':omitted},indent=2))
        for scope in roots:
            (a.manifest.parent/f'{scope}_files.txt').write_text(''.join(x['path']+'\n' for x in entries if x['scope']==scope))
        print(json.dumps({'selected':len(entries),'omitted':len(omitted)}))
    else:
        manifest=json.loads(a.manifest.read_text());errors=[]
        for row in manifest['files']:
            f=roots[row['scope']]/row['path']
            if not f.is_file() or f.stat().st_size!=row['bytes'] or sha(f)!=row['sha256']:
                errors.append({'scope':row['scope'],'path':row['path']})
        report={'status':'pass' if not errors else 'fail','files_verified':len(manifest['files']),
                'manifest_sha256':sha(a.manifest),'errors':errors}
        (a.results/'LOCAL_SYNC_VERIFICATION.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report))
        if errors:raise SystemExit(1)


if __name__=='__main__':main()
