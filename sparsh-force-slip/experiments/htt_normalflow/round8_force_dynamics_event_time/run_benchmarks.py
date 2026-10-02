#!/usr/bin/env python3
"""Run each preregistered representative benchmark, with bounded shared GPUs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--gpus', default='0,1,2')
    args = parser.parse_args()
    root = args.output_root
    inventory = json.loads((root/'FORMAL_INVENTORY.json').read_text())
    pending = [r for r in inventory['runs'] if r['seed'] == 20260914]
    output = root/'benchmark'
    output.mkdir(exist_ok=True)
    running = {}
    failed = []
    while pending or running:
        for gpu in args.gpus.split(','):
            if gpu in running or not pending or failed:
                continue
            run = pending.pop(0)
            destination = output/f"{run['group']}_20260914.json"
            # Do not reuse a benchmark solely because its filename exists.
            if destination.exists():
                raise ValueError(f'Existing benchmark must be identity-verified before reuse: {destination}')
            command = [sys.executable, str(HERE/'benchmark_e2e.py'),
                       '--data', inventory['prepared_data'],
                       '--checkpoint', str(Path(run['output'])/'best.pth'),
                       '--accepted-manifest', str(root/'formal_delivery/PREDICTION_MANIFEST.json'),
                       '--checkpoint-index', str(root/'formal_delivery/CHECKPOINT_INDEX.csv'),
                       '--source-checkpoint', '/vla1/zjy/sparsh_runs/force_slip_phase2/phase_lambda_decoupled_mae_lam010_20260528_000000/mae_decoupled_multitask/checkpoints/epoch-0030.pth',
                       '--device', 'cuda:0', '--out', str(destination)]
            log = (output/f"{run['group']}_20260914.log").open('w')
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=gpu, XFORMERS_DISABLED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            running[gpu] = (process, log, run, destination)
        time.sleep(1)
        for gpu, (process, log, run, destination) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            del running[gpu]
            if code != 0 or json.loads(destination.read_text()).get('status') != 'complete':
                failed.append(run['id'])
        if failed and not running:
            raise RuntimeError(f'Benchmark repair required: {failed}')
    print(json.dumps({'status': 'complete', 'groups': len(inventory['groups'])}))


if __name__ == '__main__':
    main()
