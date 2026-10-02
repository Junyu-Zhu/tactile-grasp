#!/usr/bin/env python3
"""Verify delivered local files against remote bytes without loading models."""
import pathlib,hashlib,json,subprocess,datetime
P=pathlib.Path(__file__).resolve().parent
REMOTE='/home/zjy/document/tactile-grasp/sparsh-force-slip/experiments/htt_normalflow/'+P.name
EXCLUDED={'ROOT_G2_SYNC_PROOF.json','ROOT_G2_ACCEPTANCE.json'}
files=[p for p in sorted(P.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name not in EXCLUDED and not p.name.endswith(('.pyc','.tmp'))]
entries=[]
for p in files:
 rel=str(p.relative_to(P))
 remote=('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/'+P.name+'/'+rel.removeprefix('formal_evaluation/')) if rel.startswith('formal_evaluation/') else REMOTE+'/'+rel
 entries.append({'path':rel,'server_path':remote,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
code="""import pathlib,hashlib,json,sys
x=json.loads(sys.stdin.read()); root=pathlib.Path(x['root']); out=[]
for r in x['entries']:
 p=pathlib.Path(r['server_path']); out.append({'path':r['path'],'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None})
print(json.dumps(out))
"""
import shlex
r=subprocess.run(['ssh','zjy-4090','/home/zjy/miniconda3/envs/sparsh/bin/python -c '+shlex.quote(code)],input=json.dumps({'root':REMOTE,'entries':entries}),text=True,capture_output=True,check=True)
remote={e['path']:e['sha256'] for e in json.loads(r.stdout)}; bad=[e['path'] for e in entries if remote.get(e['path'])!=e['sha256']]
proof={'checked_at':datetime.datetime.now().astimezone().isoformat(),'passed':not bad,'file_count':len(entries),'local_root':str(P),'server_root':REMOTE,'entries':entries,'mismatches':bad,'excluded_self_referential_receipts':sorted(EXCLUDED),'large_checkpoints':'Remain on server; independently audited by ROOT_TRAINING_AUDIT.json'}
(P/'ROOT_G2_SYNC_PROOF.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps({'passed':not bad,'files':len(entries),'mismatches':bad}))
if bad:raise SystemExit(1)
