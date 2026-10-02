#!/usr/bin/env python3
"""Index local R20 delivery files; large server-only arrays/weights stay indexed separately."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    files=[]
    for p in sorted(HERE.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(HERE)
        if '__pycache__' in rel.parts or rel.name.endswith(('.pyc','.lock')) or rel.name.startswith('ROOT_'):
            continue
        if rel.name in ('ARTIFACT_INDEX.json','SYNC_PROOF.json'):continue
        files.append({'relative_path':str(rel),'sha256':sha(p),'bytes':p.stat().st_size})
    out={'schema':'round20_artifact_index_v1','local_root':str(HERE),'files':files,
         'file_count':len(files),'large_server_only':['formal/*/latest.pth','formal/*/best.pth',
                                                       'evaluation/predictions/*.npz'],
         'checkpoint_index':'CHECKPOINT_INDEX.json','test_consumed':False}
    (HERE/'ARTIFACT_INDEX.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'files':len(files),'total_bytes':sum(x['bytes'] for x in files)}))

if __name__=='__main__':main()
