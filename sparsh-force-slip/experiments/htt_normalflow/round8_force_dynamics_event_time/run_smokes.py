#!/usr/bin/env python3
"""Run real-cache two-epoch smoke and interrupted/resumed equivalence checks."""
import argparse
import concurrent.futures
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('r8_smoke_train', HERE / 'training/train.py')
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--groups', nargs='+', required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    data_sha = train.sha256(a.data)
    records = []

    def run(group, gpu, interrupted=False):
        out = a.output / (group + ('_resume' if interrupted else '_continuous'))
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), XFORMERS_DISABLED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', CUBLAS_WORKSPACE_CONFIG=':4096:8')
        command = [sys.executable, str(HERE/'training/train.py'), '--data', str(a.data), '--group', group, '--seed', '20260914', '--output', str(out), '--device', 'cuda:0', '--smoke']
        summary_path = out/'summary.json'
        if not summary_path.exists():
            with (a.output/(out.name+'.log')).open('w') as log:
                result = subprocess.run(command + (['--interrupt-after-epoch', '1'] if interrupted else []), env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'Smoke failed {out}; inspect log')
        if interrupted:
            if summary_path.exists() and json.loads(summary_path.read_text())['status']=='complete' and not (out/'INTERRUPTED_SUMMARY.json').exists():
                raise ValueError('Missing actual interruption evidence in reused smoke')
            d = json.loads(summary_path.read_text())
            if d['status'] != 'complete':
                train.atomic_json(out/'INTERRUPTED_SUMMARY.json', d)
                with (a.output/(out.name+'.resume.log')).open('w') as log:
                    result = subprocess.run(command+['--resume'], env=env, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f'Resume failed {out}')
        d = json.loads(summary_path.read_text())
        if d['status'] != 'complete' or d['formal'] or not d['smoke']:
            raise ValueError('Smoke identity invalid')
        checks = ['finite_all_parameter_gradients','all_parameter_tensors_updated','active_input_columns_all_received_nonzero_data_gradient','inactive_input_columns_received_no_data_gradient','checkpoint_roundtrip_exact','selection_only_checkpoint_choice']
        if not all(d['audit'][k] for k in checks):
            raise ValueError('Smoke numeric audit failed')
        return {'group': group, 'output': str(out), 'summary_sha256': train.sha256(summary_path), 'checks': {k:d['audit'][k] for k in checks}}

    for offset in range(0,len(a.groups),3):
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(run,g,i) for i,g in enumerate(a.groups[offset:offset+3])]
            records.extend(f.result() for f in futures)
    comparisons = {}
    representative = [g for g in ('C_xyz_delta','C_hazard','P4_state') if g in a.groups]
    for i,g in enumerate(representative):
        run(g,i%3,True)
        continuous = torch.load(a.output/(g+'_continuous')/'latest.pth', map_location='cpu', weights_only=False)
        resumed = torch.load(a.output/(g+'_resume')/'latest.pth', map_location='cpu', weights_only=False)
        # Full checkpoint/config identity is equal: output directory is not in config.
        fields = list(continuous)
        checks = {k:train.nested_equal(continuous[k], resumed[k]) for k in fields}
        if not all(checks.values()):
            raise ValueError(f'Interrupted equivalence failed {g}: {checks}')
        evidence_path=a.output/(g+'_resume')/'INTERRUPTED_SUMMARY.json'
        evidence=json.loads(evidence_path.read_text())
        if evidence.get('status')!='interrupted' or evidence.get('group')!=g or evidence.get('seed')!=20260914 or evidence.get('run_config')!=resumed['run_config'] or evidence.get('run_identity_sha256')!=resumed['run_identity_sha256'] or len(evidence.get('history',[]))!=1:
            raise ValueError('Invalid actual interruption evidence')
        comparisons[g] = {'full_checkpoint_comparison':checks,'interruption_summary_sha256':train.sha256(evidence_path),'interruption_identity_verified':True}
    unchanged = train.sha256(a.data)==data_sha
    if not unchanged:
        raise ValueError('Frozen prepared cache changed')
    result = {'status':'pass','groups':a.groups,'real_prepared_sha256':data_sha,'cache_unchanged':unchanged,
              'upstream_freeze':'Only future module instantiated in optimizer. Frozen MAE/force/current-slip immutable detached cache inputs reused; no upstream optimizer or gradient.',
              'smoke_is_not_formal_parent':True,'runs':records,'actual_interruption_comparisons':comparisons,
              'elapsed_seconds':time.time()-started,'trainer_sha256':train.sha256(HERE/'training/train.py'),
              'training_protocol_sha256':train.sha256(HERE/'training/protocol.json')}
    train.atomic_json(a.output.parent/'SMOKE_AUDIT.json',result)
    print(json.dumps({'status':'pass','groups':len(records),'seconds':result['elapsed_seconds']}))


if __name__ == '__main__':
    main()
