#!/usr/bin/env python3
"""Run dependent export/prepare/current-slip/future-force for one accepted force run."""
from __future__ import annotations
import argparse,json,os,subprocess,time
from pathlib import Path
from touchd_common import atomic_json,sha256

PY='/home/zjy/miniconda3/envs/sparsh/bin/python'
REPO=Path('/home/zjy/document/tactile-grasp/sparsh-force-slip')
C=Path(__file__).resolve().parent
R=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow')

def run(name,command,log):
 with log.open('a') as stream:rc=subprocess.run([PY,*map(str,command)],stdout=stream,stderr=subprocess.STDOUT).returncode
 if rc:raise RuntimeError(f'{name} failed: {log}')

def main():
 p=argparse.ArgumentParser();p.add_argument('--route',choices=('H','T_H'),required=True);p.add_argument('--fold',type=int,choices=range(1,5),required=True);p.add_argument('--seed',type=int,choices=(20260914,20260915,20260916),required=True);a=p.parse_args();began=time.time()
 key=f'{a.route}_p{a.fold}_s{a.seed}';o=R/'round16_touchd_force_transfer';logs=o/'logs/downstream';logs.mkdir(parents=True,exist_ok=True)
 force=o/f'formal/htt_force/{key}';fs=json.loads((force/'training_summary.json').read_text());assert fs['status']=='complete' and not fs['smoke'] and sha256(force/'best.pth')==fs['best_checkpoint_sha256']
 support=R/f'round10_htt_force_supervision_adaptation/force_support/fold_p{a.fold}.json';r9=R/f'round9_htt_temporal_force_fusion/prepare/p{a.fold}_s{a.seed}/prepared.pt'
 pred=o/f'formal/predictions/{key}';prepared=o/f'formal/prepared/{key}';slip=o/f'formal/slip/{key}';future_data=o/f'formal/future_prepared/{key}/prepared.pt';future=o/f'formal/future/{key}'
 stages=[('export',[C/'export_htt_force.py','--run',force,'--manifest',support,'--output',pred,'--device','cuda:0']),
 ('replace',[REPO/'experiments/htt_normalflow/round10_htt_force_supervision_adaptation/fusion/replace_force.py','--r9-data',r9,'--predictions',pred/'prediction_manifest.json','--output',prepared]),
 ('slip',[REPO/'experiments/htt_normalflow/round12_class_preserving_trial_balance/training/train.py','--data',prepared/'prepared.pt','--group','F_class_trial_balanced','--fold',f'htt_leave_p{a.fold}','--seed',a.seed,'--output',slip,'--execute-formal']),
 ('future_prepare',[REPO/'experiments/htt_normalflow/round14_htt_future_force_dual/future_train.py','prepare','--prepared',prepared/'prepared.pt','--support',support,'--output',future_data]),
 ('future',[REPO/'experiments/htt_normalflow/round14_htt_future_force_dual/future_train.py','train','--data',future_data,'--output',future,'--group','F_concat','--fold',a.fold,'--seed',a.seed,'--device','cuda:0'])]
 completed=[]
 for name,command in stages:
  started=time.time();run(name,command,logs/f'{key}_{name}.log');completed.append({'stage':name,'seconds':time.time()-started})
 ss=json.loads((slip/'summary.json').read_text());ff=json.loads((future/'summary.json').read_text())
 if ss['status']!='complete' or ss['smoke'] or ff['status']!='complete':raise RuntimeError('downstream receipt')
 receipt={'status':'complete','route':a.route,'fold':a.fold,'seed':a.seed,'seconds':time.time()-began,'stages':completed,
          'force_summary_sha256':sha256(force/'training_summary.json'),'prediction_manifest_sha256':sha256(pred/'prediction_manifest.json'),
          'prepared_sha256':sha256(prepared/'prepared.pt'),'slip_summary_sha256':sha256(slip/'summary.json'),'future_data_sha256':sha256(future_data),'future_summary_sha256':sha256(future/'summary.json')}
 rp=o/f'formal/pairs/{key}.json';atomic_json(rp,receipt);print(json.dumps(receipt))
if __name__=='__main__':main()
