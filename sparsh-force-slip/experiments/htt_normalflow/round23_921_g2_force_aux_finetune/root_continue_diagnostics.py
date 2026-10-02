#!/usr/bin/env python3
"""Advance the authorized GPU phase once the prediction queue exits successfully."""
import pathlib,json,os,time,subprocess,datetime,hashlib
P=pathlib.Path(__file__).resolve().parent
pid=json.loads((P/'ROOT_PREDICTION_LAUNCH_DEDUP.json').read_text())['pid']
while True:
 try:live=b'run_prediction_queue.py' in pathlib.Path(f'/proc/{pid}/cmdline').read_bytes()
 except OSError:live=False
 if not live:break
 time.sleep(10)
s=json.loads((P/'PREDICTION_STATUS.json').read_text());assert s['status']=='complete' and s['complete']==48,s
cmd=['/home/zjy/miniconda3/envs/sparsh/bin/python',str(P/'run_gradient_queue.py'),'--inventory',str(P/'G2_RUN_INVENTORY.json'),'--ids',str(P/'G2_GRADIENT_DIAGNOSTIC_IDS.json'),'--output-root','/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round23_921_g2_force_aux_finetune','--gpu','0','1','2']
x={'at':datetime.datetime.now().astimezone().isoformat(),'command':cmd,'gradient_source_sha256':hashlib.sha256((P/'gradient_diagnostics.py').read_bytes()).hexdigest(),'scope':'Offline diagnostic only; no optimizer update'}
with (P/'logs/gradient_queue.log').open('ab') as log:
 q=subprocess.Popen(cmd,cwd=P,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'XFORMERS_DISABLED':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','CUBLAS_WORKSPACE_CONFIG':':4096:8'})
 x['pid']=q.pid;(P/'ROOT_GRADIENT_LAUNCH.json').write_text(json.dumps(x,indent=2));rc=q.wait()
print(json.dumps({'gradient_returncode':rc}),flush=True)
raise SystemExit(rc)
