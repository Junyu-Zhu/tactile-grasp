#!/usr/bin/env python3
import hashlib, json
from pathlib import Path

HERE=Path(__file__).resolve().parent
EXCLUDE={'ARTIFACT_INDEX.json','SYNC_PROOF.json','FINAL_AUTHOR_STATUS.json','CURRENT_PLAN_SNAPSHOT.md','INDEPENDENT_FINAL_REVIEW.json','INDEPENDENT_FINAL_REVIEW.md'}

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

files=[]
for p in sorted(HERE.rglob('*')):
    rel=p.relative_to(HERE).as_posix()
    if not p.is_file() or p.name in EXCLUDE or (len(p.relative_to(HERE).parts)==1 and p.name.startswith('ROOT_')): continue
    files.append({'relative_path':rel,'sha256':sha(p),'bytes':p.stat().st_size})
obj={'schema':'round21_author_artifact_index_v1','scope':'author-owned bounded synthesis; ROOT/review files excluded','repo_relative_root':'experiments/htt_normalflow/round21_final_evidence_synthesis','local_root':'/home/zjy/Documents/grasp/tactile_grasp/sparsh-force-slip/experiments/htt_normalflow/round21_final_evidence_synthesis','server_root':'/home/zjy/document/tactile-grasp/sparsh-force-slip/experiments/htt_normalflow/round21_final_evidence_synthesis','files':files}
(HERE/'ARTIFACT_INDEX.json').write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'status':'pass','files':len(files)},ensure_ascii=False))
