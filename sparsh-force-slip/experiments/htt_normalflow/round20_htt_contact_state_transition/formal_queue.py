#!/usr/bin/env python3
"""Fault-isolating formal R20 GPU queue. Existing complete runs are preserved."""
import argparse
import fcntl
import json
import os
import queue
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

import torch

from train import atomic_json, sha, state_hash

GPU_UUIDS=(
 'GPU-da1f9813-5a0a-ad64-2e46-b0d3038a4f86',
 'GPU-3385e069-0c03-8de3-d739-d062819ebfc8',
 'GPU-d94b8dd4-1314-8cc3-1aeb-10667f5e3757',
)
CUTOFF=datetime.fromisoformat('2026-09-23T01:55:53+08:00')


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=3)
    p.add_argument('--probe-only',action='store_true')
    a=p.parse_args()
    if a.workers not in (1,2,3):raise ValueError('workers must be 1..3')
    a.output.mkdir(parents=True,exist_ok=True)
    lock_file=(a.output/'FORMAL_QUEUE.lock').open('a+')
    try:fcntl.flock(lock_file,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('another formal queue owns the lock')
    health={}
    for gpu in GPU_UUIDS[:a.workers]:
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu;env['XFORMERS_DISABLED']='1'
        cmd=[sys.executable,'-c','import torch; assert torch.cuda.device_count()==1; x=torch.ones((512,512),device="cuda:0"); y=x@x; torch.cuda.synchronize(); assert float(y.sum())==134217728.0; print(torch.cuda.get_device_name(0))']
        probe=subprocess.run(cmd,env=env,capture_output=True,text=True)
        health[gpu]={'ok':probe.returncode==0,'device':probe.stdout.strip(),'error':probe.stderr[-1000:]}
    atomic_json({'schema':'round20_queue_cuda_probe_v1','health':health},a.output/'QUEUE_CUDA_PROBE.json')
    if not all(v['ok'] for v in health.values()):raise RuntimeError('UUID CUDA probe failed')
    if a.probe_only:
        print(json.dumps({'status':'probe_pass','health':health}));return
    code=Path(__file__).resolve().parent
    control=json.loads((code/'ROOT_CONTROL.json').read_text())
    expected={'train_sha256':sha(code/'train.py'),'protocol_sha256':sha(code/'PROTOCOL.md'),
              'prepare_sha256':sha(a.prepare),'smoke_sha256':sha(code/'SMOKE.json'),
              'budget_sha256':sha(code/'BUDGET.json'),'queue_sha256':sha(__file__)}
    if control.get('formal_release') is not True or control.get('authorized_runs')!=12:
        raise RuntimeError('root release absent')
    if any(control.get(k)!=v for k,v in expected.items()):
        raise RuntimeError('root release hash mismatch')
    prep=json.loads(a.prepare.read_text())
    if len(prep['runs'])!=12 or len({(r['group'],r['seed']) for r in prep['runs']})!=12:
        raise ValueError('formal inventory must contain exactly twelve distinct runs')
    def completed(run):
        target=a.output/'formal'/f"{run['group']}_p1_s{run['seed']}"
        paths=[target/x for x in ('summary.json','latest.pth','best.pth')]
        if not all(path.exists() for path in paths):return False
        summary=json.loads(paths[0].read_text())
        ident=summary.get('identity',{})
        if summary.get('status')!='complete' or any(ident.get(k)!=v for k,v in {
            'group':run['group'],'seed':run['seed'],'fold':1,'cache_sha256':run['cache_sha256'],
            'source_sha256':expected['train_sha256'],'protocol_sha256':expected['protocol_sha256']}.items()):return False
        latest=torch.load(paths[1],map_location='cpu',weights_only=False)
        best=torch.load(paths[2],map_location='cpu',weights_only=False)
        return (latest['identity']==ident and best['identity']==ident and
                state_hash(best['model'])==state_hash(latest['best_state']))
    tasks=queue.Queue()
    for run in prep['runs']:
        if completed(run):continue
        tasks.put(run)
    lock=threading.Lock();statuses=[];quarantined=[];stop=threading.Event()
    def record(row):
        with lock:
            statuses.append(row)
            atomic_json({'schema':'round20_queue_status_v1','statuses':statuses,
                         'quarantined_gpu_uuids':quarantined,'remaining':tasks.qsize()},a.output/'QUEUE_STATUS.json')
    def worker(gpu):
        while not stop.is_set():
            try:run=tasks.get_nowait()
            except queue.Empty:return
            now=datetime.now().astimezone()
            if now>=CUTOFF:
                record({'run':run,'status':'not_dispatched_cutoff','time':now.isoformat()});tasks.task_done();stop.set();return
            cache=next(x for x in prep['caches'] if x['seed']==run['seed'])
            if sha(cache['path'])!=run['cache_sha256']:
                record({'run':run,'status':'fatal_cache_identity'});tasks.task_done();stop.set();return
            output=a.output/'formal'/f"{run['group']}_p1_s{run['seed']}"
            output.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,str(Path(__file__).with_name('train.py')),'--data',cache['path'],
                     '--output',str(output),'--group',run['group'],'--fold','1','--seed',str(run['seed']),
                     '--device','cuda:0']
            env=os.environ.copy();env['XFORMERS_DISABLED']='1';env['CUDA_VISIBLE_DEVICES']=gpu
            env.pop('R20_SMOKE_KILL_AFTER_LATEST_EPOCH',None)
            started=datetime.now().astimezone()
            with (output/'train.log').open('a') as log:
                log.write(json.dumps({'started':started.isoformat(),'gpu_uuid':gpu,'command':command})+'\n');log.flush()
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env)
            status='complete' if result.returncode==0 and completed(run) else 'failed'
            row={'group':run['group'],'seed':run['seed'],'gpu_uuid':gpu,'status':status,
                 'exit_code':result.returncode,'started':started.isoformat(),
                 'ended':datetime.now().astimezone().isoformat(),'log':str(output/'train.log')}
            record(row);tasks.task_done()
            if status!='complete':
                log_tail=(output/'train.log').read_text()[-4000:]
                hardware=any(term in log_tail for term in ('CUDA error','Xid','uncorrectable ECC','GPU has fallen off'))
                if hardware:
                    with lock:quarantined.append(gpu)
                else:stop.set()
                return
    threads=[threading.Thread(target=worker,args=(gpu,),daemon=False) for gpu in GPU_UUIDS[:a.workers]]
    for thread in threads:thread.start()
    for thread in threads:thread.join()
    verified=sum(completed(run) for run in prep['runs'])
    result={'status':'complete' if verified==12 and not quarantined and not stop.is_set() else 'incomplete',
            'verified_complete':verified,'newly_complete':sum(row['status']=='complete' for row in statuses),
            'remaining':12-verified,'quarantined_gpu_uuids':quarantined,
            'cutoff_reached':any(row['status']=='not_dispatched_cutoff' for row in statuses)}
    atomic_json(result,a.output/'QUEUE_RESULT.json')
    print(json.dumps(result))
    if result['status']!='complete':sys.exit(2)

if __name__=='__main__':main()
