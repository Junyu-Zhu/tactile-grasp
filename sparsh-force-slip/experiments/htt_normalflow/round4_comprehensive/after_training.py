#!/usr/bin/env python3
"""Root final computation gate. Completion still requires review and delivery."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
BASE=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow')
OUT=BASE/'round4_comprehensive'
deadline=time.monotonic()+6*3600
encoders=['dino','ijepa','mae_letterbox']
while True:
    missing=[]
    for enc in encoders:
        state=OUT/f'encoders/state_{enc}.json'
        if state.exists() and any(r['status']=='failed' for r in json.loads(state.read_text())):
            raise RuntimeError(f'{enc} queue failed: root must repair before final computation')
        for fold in range(1,5):
            for seed in [20260914,20260915,20260916]:
                p=OUT/f'encoders/runs/{enc}/htt_leave_p{fold}/{seed}/training_summary.json'
                if not p.exists() or json.loads(p.read_text()).get('status')!='complete': missing.append(str(p))
    if not missing:break
    if time.monotonic()>deadline:raise TimeoutError('Formal run gate timeout; no completion claim')
    time.sleep(20)


def run(args):
    print('COMMAND',json.dumps([str(x) for x in args]),flush=True)
    subprocess.run([sys.executable,*map(str,args)],check=True)


run([HERE/'encoders/final_audit.py','--output-root',OUT/'encoders','--encoders',*encoders,
     '--output',OUT/'encoders/final_audit.json'])
run([HERE/'alarms/evaluate_alarms.py','--r3-root',BASE/'round3_mae_slip_adaptation',
     '--encoder-root',OUT/'encoders','--output',OUT/'alarms','--require-complete'])
for enc in ['mae',*encoders]:
    checkpoint=(BASE/'round3_mae_slip_adaptation/runs/B/fold_p1/seed_20260914/best.pth' if enc=='mae'
        else OUT/f'encoders/runs/{enc}/htt_leave_p1/20260914/best.pth')
    run([HERE/'benchmark/run.py','--encoder',enc,'--checkpoint',checkpoint,
        '--device','cuda:0','--output',OUT/f'benchmark/{enc}.json'])
run([HERE/'inventory.py'])
target=OUT/'COMPUTATION_GATE.json'
temp=target.with_suffix('.json.tmp')
temp.write_text(json.dumps({'status':'computations_complete_pending_independent_review_and_delivery',
    'encoder_runs':36,'detector_evaluations':48,'benchmarks':4},indent=2))
os.replace(temp,target)
