#!/usr/bin/env python3
"""Run all per-model R16 evaluations after the 75-run queue completes."""
import json,subprocess,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent;R=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow');O=R/'round16_touchd_force_transfer';PY=sys.executable

def run(command,log):
 log.parent.mkdir(parents=True,exist_ok=True)
 with log.open('a') as f:
  if subprocess.run([PY,*map(str,command)],stdout=f,stderr=subprocess.STDOUT).returncode:raise RuntimeError(str(log))

def main():
 downstream=json.loads((O/'queues/downstream/status.json').read_text())
 if downstream['status']!='complete' or len(downstream['jobs'])!=24:raise RuntimeError('downstream incomplete')
 root=O/'evaluation';logs=root/'logs';manifest=root/'EVALUATION_MANIFEST.json'
 run([HERE/'build_evaluation_manifest.py','--output',manifest],logs/'manifest.log')
 run([HERE/'evaluate_slip_registered.py','--manifest',manifest,'--output',root/'slip'],logs/'slip.log')
 complete=[]
 for route in ('H','T_H'):
  for fold in range(1,5):
   for seed in (20260914,20260915,20260916):
    key=f'{route}_p{fold}_s{seed}';force=root/'force'/key;future=root/'future'/key;diag=root/'future_diagnostics'/key
    run([HERE/'evaluate_force.py','--support',R/f'round10_htt_force_supervision_adaptation/force_support/fold_p{fold}.json','--predictions',O/f'formal/predictions/{key}/prediction_manifest.json','--prepared',O/f'formal/prepared/{key}/prepared.pt','--output',force],logs/f'{key}_force.log')
    run([HERE.parent/'round14_htt_future_force_dual/evaluate.py','--data',O/f'formal/future_prepared/{key}/prepared.pt','--checkpoint',O/f'formal/future/{key}/best.pth','--group','F_concat','--output',future,'--device','cpu'],logs/f'{key}_future.log')
    run([HERE/'evaluate_future_diagnostics.py','--evaluation',future,'--output',diag],logs/f'{key}_future_diag.log')
    run([HERE/'audit_future_ranges.py','--evaluation',future,'--output',root/'future_ranges'/key],logs/f'{key}_future_ranges.log');complete.append(key)
 run([HERE/'bootstrap_registered.py','--evaluation',root,'--output',root/'bootstrap'],logs/'bootstrap.log')
 run([HERE/'select_failure_cases.py','--evaluation',root,'--output',root/'failure_cases'],logs/'failure_cases.log')
 run([HERE/'plot_fixed_case.py','--manifest',manifest,'--evaluation',root,'--output',root/'fixed_case_figure'],logs/'fixed_case_figure.log')
 run([HERE/'build_report.py','--root',root,'--output',root/'reporting'],logs/'report.log')
 receipt={'schema':'round16_evaluation_execution_v1','status':'complete','models':24,'force_evaluations':24,'future_evaluations':24,'slip_runs':36,'paired_bootstrap_draws_per_fold':2000,'test_consumed':False,'keys':complete}
 root.mkdir(parents=True,exist_ok=True);(root/'EXECUTION.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
