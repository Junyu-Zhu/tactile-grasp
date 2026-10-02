#!/usr/bin/env python3
"""Generate machine receipt, run smoke, verify identities, and lock formal inputs."""
import argparse, hashlib, json, os, platform, socket, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def atomic(p,x):
    p.parent.mkdir(parents=True,exist_ok=True); q=p.with_suffix(p.suffix+'.tmp')
    q.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n'); os.replace(q,p)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--round-dir',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    index_path=a.round_dir/'ARTIFACT_INDEX.json'; protocol=a.round_dir/'PROTOCOL.md'; source=a.round_dir/'evaluation/r13_evaluate.py'
    index=json.loads(index_path.read_text()); runs=index['selected_runs']
    expected={(g,f,s) for g in ('V_original','F_history_original','V_class_trial_balanced','F_class_trial_balanced') for f in [f'htt_leave_p{i}' for i in range(1,5)] for s in (20260914,20260915,20260916)}
    actual={(r['group'],r['fold'],int(r['seed'])) for r in runs}
    if len(runs)!=48 or actual!=expected: raise SystemExit('48-grid identity failed')
    manifest_record=next(x for x in index['indexed_artifacts'] if x['relative_to_r12']=='results/formal_delivery/EVALUATION_MANIFEST.json')
    manifest_path=Path(manifest_record['server'])
    if sha(manifest_path)!=manifest_record['sha256']: raise SystemExit('parent manifest hash mismatch')
    parent=json.loads(manifest_path.read_text())
    if parent.get('status')!='complete' or parent.get('schema')!='round12_evaluation_manifest_v1':
        raise SystemExit('parent manifest not accepted')
    parent_runs=parent['runs']
    if not all(r in parent_runs for r in runs): raise SystemExit('selected run is not an exact parent-manifest record')
    verified=[]
    for r in runs:
        for kind in ('training_summary','checkpoint'):
            rec=r[kind]; p=Path(rec['path']); actual_sha=sha(p)
            if actual_sha!=rec['sha256']: raise SystemExit(f'hash mismatch {p}')
            verified.append({'path':str(p),'sha256':actual_sha,'kind':kind})
        for kind in ('predictions','endpoints'):
            for role in ('train','calibration','validation'):
                rec=r[kind][role]; p=Path(rec['path']); actual_sha=sha(p)
                if actual_sha!=rec['sha256']: raise SystemExit(f'hash mismatch {p}')
                verified.append({'path':str(p),'sha256':actual_sha,'kind':kind,'role':role})
    machine={'created_at':datetime.now(timezone.utc).astimezone().isoformat(),'hostname':socket.gethostname(),
             'platform':platform.platform(),'python':sys.version,'cpu_count':os.cpu_count(),
             'XFORMERS_DISABLED':os.environ.get('XFORMERS_DISABLED'),'new_neural_trainings':0,
             'device_policy':'CPU-only frozen prediction evaluation'}
    atomic(a.output/'MACHINE_CONFIG.json',machine)
    smoke=subprocess.run([sys.executable,str(source),'--smoke'],check=True,text=True,capture_output=True)
    smoke_result=json.loads(smoke.stdout); atomic(a.output/'SMOKE.json',smoke_result)
    lock={'schema':'round13_protocol_lock_v1','created_at':datetime.now(timezone.utc).astimezone().isoformat(),
          'status':'locked','model_count':48,'new_neural_trainings':0,'test_role_consumed':False,
          'selected_runs_exact_parent_manifest_subset':True,'parent_manifest_sha256':sha(manifest_path),
          'protocol_sha256':sha(protocol),'artifact_index_sha256':sha(index_path),'evaluation_source_sha256':sha(source),
          'python_executable':sys.executable,'machine_config_sha256':sha(a.output/'MACHINE_CONFIG.json'),
          'smoke_sha256':sha(a.output/'SMOKE.json'),'verified_records':len(verified),'verified_inputs':verified}
    atomic(a.output/'PROTOCOL_LOCK.json',lock)
    atomic(a.output/'PRECHECK.json',{'status':'pass','model_count':48,'verified_records':len(verified),
                                     'grid_complete':True,'smoke_status':smoke_result['status'],
                                     'no_test_paths':all('test' not in x['path'].lower() for x in verified)})
    print(json.dumps({'status':'pass','verified_records':len(verified),'lock':str(a.output/'PROTOCOL_LOCK.json')},indent=2))
if __name__=='__main__': main()
