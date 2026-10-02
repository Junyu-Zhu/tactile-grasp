#!/usr/bin/env python3
import argparse,subprocess
from pathlib import Path
SEEDS=(20260914,20260915,20260916)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--device',default='cuda:1');a=p.parse_args();code=Path(__file__).resolve().parent;r10=a.root.parent/'round10_htt_force_supervision_adaptation';py='/home/zjy/miniconda3/envs/sparsh/bin/python'
 for route in ('H','T_H'):
  for fold in range(1,5):
   for seed in SEEDS:
    key=f'{route}_p{fold}_s{seed}';out=a.root/'evaluation/old_force_regression/exports'/key;out.mkdir(parents=True,exist_ok=True)
    with (out/'export.log').open('a') as log:subprocess.run([py,str(code/'export_old_force_regression.py'),'--run',str(a.root/'formal/htt_force'/key),'--support',str(r10/f'force_support/fold_p{fold}.json'),'--output',str(out),'--device',a.device,'--batch-size','128'],stdout=log,stderr=subprocess.STDOUT,check=True)
 subprocess.run([py,str(code/'evaluate_old_force_regression.py'),'--root',str(a.root/'evaluation/old_force_regression/exports'),'--r10',str(r10),'--output',str(a.root/'evaluation/old_force_regression')],check=True)
if __name__=='__main__':main()
