#!/usr/bin/env python3
"""Explicitly gated, dependency-aware shared-GPU scheduler; defaults to dry-run."""
from __future__ import annotations
import argparse,datetime,fcntl,hashlib,json,os,subprocess,time
from pathlib import Path


def atomic(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(data,indent=2,ensure_ascii=False));temp.replace(path)


def verify_artifacts(value):
    """Validate declared file/hash pairs, never confuse tensor-state hashes with files."""
    if isinstance(value,list):
        return all(verify_artifacts(item) for item in value)
    if not isinstance(value,dict):return True
    for key,item in value.items():
        if isinstance(key,str) and key.startswith('/') and isinstance(item,str) and len(item)==64:
            if not Path(key).is_file() or digest_file(Path(key))!=item:return False
        if isinstance(item,str) and item.startswith('/'):
            expected=value.get(key+'_sha256')
            if key=='path':expected=value.get('sha256',expected)
            if expected is not None:
                path=Path(item)
                if not path.is_file():return False
                digest=hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
                if digest.hexdigest()!=expected:return False
        if not verify_artifacts(item):return False
    return True


def digest_file(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def command_identity(job):
    argv=job['argv'];inputs={}
    for index,value in enumerate(argv):
        if index and argv[index-1]=='--output':continue
        path=Path(value)
        if path.is_absolute() and path.is_file() and index!=0:inputs[value]=digest_file(path)
    return {'argv':argv,'input_sha256':inputs}


def launch_argv(job):
    argv=list(job['argv'])
    if any(x in argv for x in ('train-source','train-htt')) and '--output' in argv:
        run_dir=Path(argv[argv.index('--output')+1])
        if (run_dir/'latest.pth').is_file() and '--resume' not in argv:argv.append('--resume')
    return argv


def accepted(job, require_receipt=True):
    p=Path(job['acceptance_path'])
    if not p.is_file():return False
    data=json.loads(p.read_text())
    if data.get('status') not in job.get('accepted_statuses',['complete','pass']):return False
    if data.get('smoke') is True or data.get('formal') is False:raise ValueError('smoke output at formal acceptance path: '+str(p))
    config=p.parent/'config.json'
    if config.exists() and json.loads(config.read_text()).get('smoke') is True:raise ValueError('smoke config at formal run')
    if not verify_artifacts(data):return False
    if require_receipt and 'argv' in job:
        receipt=p.with_name(p.name+'.scheduler_receipt.json')
        if not receipt.is_file():return False
        if json.loads(receipt.read_text())!=command_identity(job):raise ValueError('formal command or input drift: '+job['id'])
    return True


def main():
    p=argparse.ArgumentParser();p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--execute-formal',action='store_true');p.add_argument('--gpus',default='0,1,2')
    p.add_argument('--max-parallel',type=int,default=3);a=p.parse_args()
    inv=json.loads(a.inventory.read_text());jobs=inv['jobs'];byid={j['id']:j for j in jobs}
    if len(byid)!=len(jobs):raise ValueError('duplicate job id')
    for j in jobs:
        if set(j['depends_on'])-byid.keys():raise ValueError('unknown dependencies')
        if not isinstance(j['argv'],list) or not all(isinstance(x,str) for x in j['argv']):raise ValueError('invalid argv')
        forbidden={'--smoke','--smoke-epochs','--allow-smoke','--allow-smoke-parent','--allow-unverified-cache','--interrupt-after-epoch'}
        if any(x.split('=')[0] in forbidden for x in j['argv']):raise ValueError('preparation option in formal command: '+j['id'])
    if not a.execute_formal:
        print(json.dumps({'status':'dry_run_no_process_started','jobs':len(jobs),'neural_jobs':sum(j['kind']=='neural' for j in jobs),
                          'not_ready':[j['id'] for j in jobs if any(not accepted(byid[d]) for d in j['depends_on'])]},indent=2));return
    if inv.get('preparation_status')!='accepted':raise RuntimeError('preparation not independently accepted')
    proof=Path(inv['local_sync_proof'])
    if not proof.is_file() or json.loads(proof.read_text()).get('status')!='pass':raise RuntimeError('missing accepted delivery parity')
    if json.loads(proof.read_text()).get('inventory_sha256')!=digest_file(a.inventory):raise RuntimeError('delivery proof does not bind this inventory')
    deadline=datetime.datetime.fromisoformat(inv['deadline']);cutoff=deadline-datetime.timedelta(days=2)
    gpus=a.gpus.split(',');limit=min(a.max_parallel,len(gpus));assert limit>0
    root=Path(inv['output_root']);root.mkdir(parents=True,exist_ok=True);state_path=root/'formal_scheduler_state.json'
    lock=(root/'formal_scheduler.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if state_path.exists():
        previous=json.loads(state_path.read_text())
        for pid in previous.get('active_pids',[]):
            try:os.kill(pid,0)
            except ProcessLookupError:continue
            raise RuntimeError(f'Prior owned child PID {pid} is still alive; inspect/wait before resuming, do not duplicate')
    state={j['id']:('complete' if accepted(j) else 'pending') for j in jobs};active={};failed=[];retry_oom=set()
    try:
        while any(s in ('pending','running') for s in state.values()):
            now=datetime.datetime.now(datetime.timezone.utc)
            for gpu,(proc,job,stream) in list(active.items()):
                code=proc.poll()
                if code is None:continue
                stream.close();del active[gpu]
                if code==0 and accepted(job,require_receipt=False):
                    acceptance=Path(job['acceptance_path'])
                    atomic(acceptance.with_name(acceptance.name+'.scheduler_receipt.json'),command_identity(job))
                    state[job['id']]='complete'
                else:
                    log=Path(job['log_path']).read_text(errors='replace')[-12000:]
                    if ('out of memory' in log.lower()) and job['id'] not in retry_oom:
                        retry_oom.add(job['id']);limit=1;state[job['id']]='pending'
                    else:state[job['id']]='failed';failed.append(job['id'])
            for j in jobs:
                if state[j['id']]=='pending' and any(state[d] in ('failed','blocked') for d in j['depends_on']):state[j['id']]='blocked'
            if now>=cutoff:
                for j in jobs:
                    if state[j['id']]=='pending':state[j['id']]='blocked_time_budget'
            for gpu in gpus:
                if gpu in active or len(active)>=limit:continue
                ready=next((j for j in jobs if state[j['id']]=='pending' and all(state[d]=='complete' for d in j['depends_on'])),None)
                if ready is None:break
                Path(ready['log_path']).parent.mkdir(parents=True,exist_ok=True)
                stream=Path(ready['log_path']).open('a')
                env=os.environ.copy();env.update({'CUDA_VISIBLE_DEVICES':gpu,'XFORMERS_DISABLED':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'})
                argv=launch_argv(ready)
                proc=subprocess.Popen(argv,cwd=inv['code_root'],env=env,stdout=stream,stderr=subprocess.STDOUT)
                active[gpu]=(proc,ready,stream);state[ready['id']]='running'
            atomic(state_path,{'status':'running' if active else 'pending','jobs':state,'oom_serial_retries':sorted(retry_oom),'max_parallel':limit,'active_pids':[item[0].pid for item in active.values()]})
            if not active and not any(s=='pending' for s in state.values()):break
            if not active and any(s=='pending' for s in state.values()):raise RuntimeError('dependency cycle or unresolved readiness')
            time.sleep(5)
    finally:
        # Do not terminate external user jobs. Own children survive interruption for recoverable inspection.
        for _,_,stream in active.values():stream.close()
    status='complete' if all(s=='complete' for s in state.values()) else 'needs_attention'
    atomic(state_path,{'status':status,'jobs':state,'oom_serial_retries':sorted(retry_oom),'max_parallel':limit})
    print(json.dumps({'status':status,'failed':failed}))
    if status!='complete':raise SystemExit(1)
if __name__=='__main__':main()
