"""Continue this run from training completion to evaluation and acceptance."""
import json
from pathlib import Path
import subprocess
import time
import os

HERE=Path(__file__).resolve().parent
OUT=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round3_mae_slip_adaptation')
PY='/home/zjy/miniconda3/envs/sparsh/bin/python'
CKPT='/vla1/zjy/sparsh_runs/force_slip_phase2/phase_lambda_decoupled_mae_lam010_20260528_000000/mae_decoupled_multitask/checkpoints/epoch-0030.pth'


def main():
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',XFORMERS_DISABLED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    while True:
        state=json.loads((OUT/'EXECUTION_STATE.json').read_text())
        if state['status']=='needs_recovery':
            raise RuntimeError('Training queue needs recovery')
        if state['status']=='training_complete':break
        time.sleep(15)
    commands=[
        [PY,str(HERE/'evaluate.py'),'--splits','/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round1/splits.json',
         '--run-root',str(OUT),'--baseline-cache-index','/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round2/cache/index.json',
         '--output',str(OUT/'evaluation')],
        [PY,str(HERE/'final_audit.py'),'--root',str(OUT),'--checkpoint',CKPT,
         '--audit',str(OUT/'audit/audit.json'),'--output',str(OUT/'final_training_audit.json')]
    ]
    for command in commands:
        print('RUN',command,flush=True)
        subprocess.run(command,env=env,check=True)
    (OUT/'postprocess_complete.json').write_text(json.dumps({'status':'pass','scope':'evaluation and training artifact checks; final human-readable delivery pending'},indent=2)+'\n')


if __name__=='__main__':main()
