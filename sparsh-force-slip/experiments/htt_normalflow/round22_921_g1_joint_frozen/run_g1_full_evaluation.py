#!/usr/bin/env python3
"""Reproduce the complete accepted G1 evaluation from the frozen formal inventory."""
import argparse,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
PYTHON=sys.executable
R13=HERE.parent/'round13_trial_level_alarm_calibration/evaluation/r13_evaluate.py'
SUPPORT=HERE.parent/'round10_htt_force_supervision_adaptation/results/force_support/PREPARE_AUDIT.json'
def run(args):subprocess.run([PYTHON,*map(str,args)],cwd=HERE,env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'},check=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--inventory',type=Path,default=HERE/'FROZEN_RUN_INVENTORY.json');ap.add_argument('--output-root',type=Path,required=True);ap.add_argument('--device',default='cpu');a=ap.parse_args();assert not a.output_root.exists();a.output_root.mkdir(parents=True)
 run([HERE/'evaluate_frozen_detection.py','--inventory',a.inventory,'--output',a.output_root/'detection_core','--r13',R13,'--device',a.device,'--bootstrap-draws','2000'])
 run([HERE/'evaluate_force_full.py','--inventory',a.inventory,'--output',a.output_root/'force_core','--device',a.device,'--bootstrap-draws','2000'])
 run([HERE/'evaluate_force_trial_detail.py','--inventory',a.inventory,'--output',a.output_root/'force_trial_detail_v2'])
 run([HERE/'formal_prediction_diagnostics_v2.py','--inventory',a.inventory,'--output',a.output_root/'prediction_diagnostics_v2'])
 run([HERE/'generate_detection_supplement.py','--inventory',a.inventory,'--core',a.output_root/'detection_core','--support-audit',SUPPORT,'--output',a.output_root/'detection_supplement'])
 run([HERE/'detection_case_numerics.py','--inventory',a.inventory,'--core',a.output_root/'detection_core','--output',a.output_root/'detection_supplement'])
 run([HERE/'generate_force_failure_figures.py','--inventory',a.inventory,'--core',a.output_root/'force_core','--support-audit',SUPPORT,'--output',a.output_root/'force_failure_figures'])
 run([HERE/'finalize_g1_reporting.py','--root',a.output_root,'--code',HERE,'--support-audit',SUPPORT,'--report-name','reporting'])
 print(a.output_root)
if __name__=='__main__':main()
