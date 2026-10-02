#!/usr/bin/env python3
"""Build a split-root delivery manifest and verify local/server SHA identity."""
import argparse,hashlib,json,os,subprocess,shlex
from datetime import datetime,timezone
from pathlib import Path

CODE_ROOT='/home/zjy/document/tactile-grasp/sparsh-force-slip/experiments/htt_normalflow/round13_trial_level_alarm_calibration'
OUT_ROOT='/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round13_trial_level_alarm_calibration'
ROOT_OUTPUT={'MACHINE_CONFIG.json','PRECHECK.json','PROTOCOL_LOCK.json','SMOKE.json','SUPPLEMENTARY_ACCEPTANCE.json','REVISION_RECORD.json','FINAL_STATUS.json','formal.log'}

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def atomic(p,x):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');os.replace(q,p)
def included(root,p):
    rel=p.relative_to(root)
    if any(x in ('server_snapshot','__pycache__') for x in rel.parts):return False
    if p.name in ('DELIVERY_MANIFEST.json','LOCAL_SYNC_PROOF.json'):return False
    if p.suffix in ('.pyc',):return False
    return rel.parts[0] in ('results','evaluation','reporting','reviews') or str(rel) in {
        'ARTIFACT_INDEX.json','HANDOFF.md','PROTOCOL.md','START_GOAL.md','PREPARATION_STATUS.json','PREPARATION_SYNC.json',
        'REPRODUCE.md','EXECUTION_PROVENANCE.json',*ROOT_OUTPUT}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--verify-remote',action='store_true');a=ap.parse_args();root=a.root.resolve()
    records=[]
    for p in sorted(x for x in root.rglob('*') if x.is_file() and included(root,x)):
        rel=str(p.relative_to(root))
        if rel.startswith('results/') or rel in ROOT_OUTPUT:remote=f'{OUT_ROOT}/{rel}'
        else:remote=f'{CODE_ROOT}/{rel}'
        records.append({'relative_path':rel,'sha256':sha(p),'bytes':p.stat().st_size,'remote':remote})
    manifest={'schema':'round13_delivery_manifest_v1','created_at':datetime.now(timezone.utc).astimezone().isoformat(),
              'status':'complete','files':records,'file_count':len(records),'total_bytes':sum(r['bytes'] for r in records),
              'new_neural_trainings':0,'test_role_consumed':False}
    atomic(root/'DELIVERY_MANIFEST.json',manifest)
    local_fail=[r['relative_path'] for r in records if sha(root/r['relative_path'])!=r['sha256']]
    proof={'schema':'round13_local_sync_proof_v1','created_at':datetime.now(timezone.utc).astimezone().isoformat(),
           'manifest_sha256':sha(root/'DELIVERY_MANIFEST.json'),'file_count':len(records),'local_failures':local_fail}
    if a.verify_remote:
        payload=json.dumps([{'path':r['remote'],'sha256':r['sha256']} for r in records])
        code="""import hashlib,json,sys,os\nrows=json.load(sys.stdin);bad=[]\nfor r in rows:\n p=r['path']\n if not os.path.isfile(p):bad.append({'path':p,'error':'missing'});continue\n h=hashlib.sha256()\n with open(p,'rb') as f:\n  for b in iter(lambda:f.read(1<<20),b''):h.update(b)\n if h.hexdigest()!=r['sha256']:bad.append({'path':p,'error':'sha','actual':h.hexdigest()})\nprint(json.dumps({'checked':len(rows),'failures':bad}))\n"""
        remote_command=f"/home/zjy/miniconda3/envs/sparsh/bin/python -c {shlex.quote(code)}"
        proc=subprocess.run(['ssh','zjy-4090',remote_command],input=payload,text=True,capture_output=True,check=True)
        remote=json.loads(proc.stdout);proof['remote_checked']=remote['checked'];proof['remote_failures']=remote['failures']
    proof['status']='pass' if not local_fail and not proof.get('remote_failures') else 'fail'
    atomic(root/'LOCAL_SYNC_PROOF.json',proof)
    print(json.dumps(proof,indent=2,ensure_ascii=False))
    if proof['status']!='pass':raise SystemExit(1)
if __name__=='__main__':main()
