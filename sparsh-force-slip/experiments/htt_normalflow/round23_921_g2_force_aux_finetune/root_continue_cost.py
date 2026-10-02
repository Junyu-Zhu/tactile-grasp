#!/usr/bin/env python3
"""Run four deployment benchmarks on GPU0 after its replay repair exits."""
import pathlib,json,os,time,subprocess,hashlib,datetime
P=pathlib.Path(__file__).resolve().parent
while True:
 try:live=b'repair_prediction_direct.py' in pathlib.Path('/proc/1482974/cmdline').read_bytes()
 except OSError:live=False
 if not live:break
 time.sleep(10)
large=pathlib.Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round23_921_g2_force_aux_finetune');s=json.loads((large/'predictions/D/p4_s20260915/SUMMARY.json').read_text());assert s['status']=='complete' and s['prediction_method']=='direct_repeated_window_exception'
(large/'costs').mkdir(exist_ok=True);records=[]
for g in 'ABCD':
 cmd=['/home/zjy/miniconda3/envs/sparsh/bin/python',str(P/'benchmark_deployment.py'),'--inventory',str(P/'G2_RUN_INVENTORY.json'),'--run',g+'/p1_s20260915','--output',str(large/'costs'/f'{g}.json'),'--device','cuda:0']
 with (P/'logs'/f'cost_{g}.log').open('ab') as log:
  q=subprocess.run(cmd,cwd=P,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','XFORMERS_DISABLED':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','CUBLAS_WORKSPACE_CONFIG':':4096:8'})
 records.append({'group':g,'returncode':q.returncode,'command':cmd});(P/'ROOT_COST_STATE.json').write_text(json.dumps({'at':datetime.datetime.now().astimezone().isoformat(),'records':records,'source_sha256':hashlib.sha256((P/'benchmark_deployment.py').read_bytes()).hexdigest()},indent=2))
 if q.returncode:raise SystemExit(q.returncode)
print('Four cost measurements complete',flush=True)
