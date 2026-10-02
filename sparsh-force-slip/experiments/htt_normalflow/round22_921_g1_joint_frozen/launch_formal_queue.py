#!/usr/bin/env python3
"""Detach one root-authorized round22 queue with absolute log and PID paths."""
import argparse,json,os,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
PYTHON='/home/zjy/miniconda3/envs/sparsh/bin/python'
RUN_ROOT='/vla1/zjy/sparsh_runs/force_slip_htt_normalflow'
SOURCE_ROOT='/vla1/zjy/sparsh_runs/force_slip_phase2'
REPO=HERE.parents[3]

def command(lane):
 if lane=='frozen':
  return [PYTHON,str(HERE/'formal_frozen_queue.py'),'--authorization',str(HERE/'G1_FORMAL_AUTHORIZATION.json'),'--inventory',str(HERE/'FROZEN_RUN_INVENTORY.json'),'--state',str(HERE/'FROZEN_QUEUE_STATE_V8.json'),'--lock',f'{RUN_ROOT}/round22_921_g1_joint_frozen/frozen_queue_v8.lock','--python',PYTHON,'--gpu','0','1','2','--execute']
 return [PYTHON,str(HERE/'formal_queue.py'),'--authorization',str(HERE/'G2_FORMAL_AUTHORIZATION.json'),'--run-inventory',str(HERE/'G2_RUN_INVENTORY.json'),'--state',str(HERE/'G2_QUEUE_STATE_RELEASE.json'),'--lock',f'{RUN_ROOT}/round22_921_g1_joint_frozen/g2_queue_release.lock','--python',PYTHON,'--repo',str(REPO),'--run-root',RUN_ROOT,'--source-root',SOURCE_ROOT,'--gpu','0','1','2','--execute']

def main():
 ap=argparse.ArgumentParser();ap.add_argument('lane',choices=('frozen','e3'));a=ap.parse_args(); logs=HERE/'logs';logs.mkdir(exist_ok=True);pidpath=logs/f'formal_{a.lane}_queue.pid';logpath=logs/f'formal_{a.lane}_queue.log'
 if pidpath.exists():
  try: pid=int(pidpath.read_text().strip()); os.kill(pid,0)
  except (ValueError,ProcessLookupError,PermissionError): pass
  else: raise SystemExit(f'{a.lane} queue already alive pid={pid}')
 env=os.environ.copy();env['XFORMERS_DISABLED']='1';env['OMP_NUM_THREADS']='4';env['MKL_NUM_THREADS']='4';env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
 with logpath.open('ab',buffering=0) as log:
  proc=subprocess.Popen(command(a.lane),cwd=HERE,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 pidpath.write_text(f'{proc.pid}\n');print(json.dumps({'lane':a.lane,'pid':proc.pid,'pid_file':str(pidpath),'log':str(logpath),'command':command(a.lane)}))
if __name__=='__main__':main()
