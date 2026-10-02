#!/usr/bin/env python3
"""Root-authorized sequential launcher for the immutable R16 stage inventories."""
import argparse,json,subprocess,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUTPUT=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round16_touchd_force_transfer')

def main():
 p=argparse.ArgumentParser();p.add_argument('--root-authorized',action='store_true');p.add_argument('--gpus',default='0,1,2');a=p.parse_args()
 if not a.root_authorized:raise SystemExit('formal pipeline requires explicit root authorization flag')
 ready=json.loads((OUTPUT/'DISPATCH_READY.json').read_text())
 if ready['status']!='ready_for_root_formal_decision' or ready['formal_runs']!=75:raise RuntimeError('dispatch gate is not ready')
 stages=(('TOUCHD.json','touchd'),('HTT_FORCE.json','htt_force'),('DOWNSTREAM.json','downstream'))
 for inventory,queue in stages:
  command=[sys.executable,str(HERE/'run_stage.py'),'--inventory',str(OUTPUT/'inventories'/inventory),'--output',str(OUTPUT/'queues'/queue),'--gpus',a.gpus]
  if subprocess.run(command).returncode:raise SystemExit(f'failed stage {queue}')
 print(json.dumps({'status':'complete','formal_neural_runs':75,'stages':[q for _,q in stages]}))
if __name__=='__main__':main()
