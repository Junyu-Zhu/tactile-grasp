"""Three shared-GPU queues; restart-safe dispatch of the fixed 24 experiments."""
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading

HERE = Path(__file__).resolve().parent
OUT = Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round3_mae_slip_adaptation')
PYTHON = '/home/zjy/miniconda3/envs/sparsh/bin/python'
CHECKPOINT = '/vla1/zjy/sparsh_runs/force_slip_phase2/phase_lambda_decoupled_mae_lam010_20260528_000000/mae_decoupled_multitask/checkpoints/epoch-0030.pth'
SEEDS = [20260914, 20260915, 20260916]
LOCK = threading.Lock()
STATE = {'status': 'running', 'runs': []}


def code_hashes():
    return {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
            for name in ('training.py', 'PROTOCOL.md', 'run_all.py')}


def save_state():
    STATE['updated_at'] = datetime.datetime.now().astimezone().isoformat()
    tmp = OUT / 'EXECUTION_STATE.tmp'
    tmp.write_text(json.dumps(STATE, indent=2) + '\n')
    tmp.replace(OUT / 'EXECUTION_STATE.json')


def queue(gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), XFORMERS_DISABLED='1',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               WANDB_MODE='disabled')
    jobs = [r for r in STATE['runs'] if r['gpu'] == gpu]
    for job in jobs:
        if code_hashes() != STATE['code_hashes']:
            raise RuntimeError('Training code/protocol changed during queue execution')
        run = Path(job['output'])
        run.mkdir(parents=True, exist_ok=True)
        command = [PYTHON, str(HERE/'training.py'), '--cache-dir', str(OUT/'cache'),
                   '--checkpoint', CHECKPOINT, '--fold', job['fold'], '--init', job['init'],
                   '--seed', str(job['seed']), '--output-dir', str(run), '--device', 'cuda:0',
                   '--workers', '2']
        with LOCK:
            job.update(status='running', command=command)
            save_state()
        with (run/'process.log').open('a') as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        with LOCK:
            job['exit_code'] = result.returncode
            summary = run/'training_summary.json'
            job['status'] = 'complete' if result.returncode == 0 and summary.exists() and json.loads(summary.read_text()).get('status') == 'complete' else 'failed'
            save_state()
        if job['status'] == 'failed':
            print(f"Failed {job['variant']} {job['fold']} {job['seed']}; inspect {run/'process.log'}", flush=True)
            # Do not burn through the same failure for every run on this GPU.
            return
        print(f"Completed {job['variant']} {job['fold']} {job['seed']}", flush=True)


def main():
    for gpu, seed in enumerate(SEEDS):
        for probe in range(1, 5):
            for variant, init in [('B','fresh'), ('C','old')]:
                STATE['runs'].append({'gpu':gpu, 'seed':seed, 'fold':f'htt_leave_p{probe}',
                                     'variant':variant, 'init':init, 'status':'pending',
                                     'output':str(OUT/'runs'/variant/f'fold_p{probe}'/f'seed_{seed}')})
    cache = json.loads((OUT/'cache/cache_manifest.json').read_text())
    if cache['status'] != 'complete':
        raise RuntimeError('Cache incomplete')
    audit = json.loads((OUT/'audit/audit.json').read_text())
    if audit['status'] != 'pass':
        raise RuntimeError('Provenance audit failed')
    expected_training_hash = code_hashes()['training.py']
    expected_protocol_hash = code_hashes()['PROTOCOL.md']
    expected_cache_hash = hashlib.sha256((OUT/'cache/cache_manifest.json').read_bytes()).hexdigest()
    for init in ('fresh', 'old'):
        smoke = json.loads((OUT/'smoke'/init/'training_summary.json').read_text())
        if smoke['status'] != 'smoke_complete' or not smoke['same_next_step_proof']['pass'] or not smoke['frozen_encoder_force_proof']['pass']:
            raise RuntimeError('Smoke not complete')
        if smoke['config']['training_source_sha256'] != expected_training_hash or smoke['config']['protocol_sha256'] != expected_protocol_hash or smoke['config']['cache_manifest_sha256'] != expected_cache_hash:
            raise RuntimeError('Smoke provenance differs from current run')
    STATE['protocol_sha256'] = hashlib.sha256((OUT/'PROTOCOL.md').read_bytes()).hexdigest()
    STATE['code_hashes'] = code_hashes()
    if STATE['code_hashes']['PROTOCOL.md'] != STATE['protocol_sha256']:
        raise RuntimeError('Output and source protocols differ')
    save_state()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(queue, range(3)))
    STATE['status'] = 'training_complete' if all(r['status']=='complete' for r in STATE['runs']) else 'needs_recovery'
    save_state()
    if STATE['status'] != 'training_complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
