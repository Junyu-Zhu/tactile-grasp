#!/usr/bin/env python3
"""Read-only per-file identity check for the Round 21 author package."""
import hashlib, json, subprocess, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
REMOTE='/home/zjy/document/tactile-grasp/sparsh-force-slip/experiments/htt_normalflow/round21_final_evidence_synthesis'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
idx=json.loads((HERE/'ARTIFACT_INDEX.json').read_text())
names=[x['relative_path'] for x in idx['files']]+['ARTIFACT_INDEX.json']
local={n:sha(HERE/n) for n in names}
cmd="cd '%s' && sha256sum %s"%(REMOTE,' '.join("'%s'"%n for n in names))
raw=subprocess.check_output(['ssh','zjy-4090',cmd],text=True)
remote={line.split(None,1)[1]:line.split()[0] for line in raw.splitlines()}
files=[{'relative_path':n,'local_sha256':local[n],'server_sha256':remote.get(n),'match':local[n]==remote.get(n)} for n in names]
obj={'schema':'round21_sync_proof_v1','status':'pass' if all(x['match'] for x in files) else 'fail','files':len(files),'matches':sum(x['match'] for x in files),'local_root':str(HERE),'server_root':REMOTE,'artifacts':files}
(HERE/'SYNC_PROOF.json').write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'status':obj['status'],'files':obj['files'],'matches':obj['matches']}))
sys.exit(obj['status']!='pass')
