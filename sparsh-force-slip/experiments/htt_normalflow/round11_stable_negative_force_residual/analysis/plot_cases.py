#!/usr/bin/env python3
"""Fixed selection correction/harm cases with every seed, never retune thresholds."""
import argparse,json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'evaluation'))
import evaluate as e

def main():
 p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);audit=json.loads((a.analysis/'AUDIT.json').read_text());assert audit['status']=='pass' and audit['runs']==36
 for name,h in audit['outputs'].items():assert e.sha(a.analysis/name)==h
 artifacts={Path(r['path']).name:r for r in audit['prediction_artifacts']};cases=[r for r in e.readcsv(a.analysis/'representative_cases.csv') if r['selected']=='True'];metrics=e.readcsv(a.analysis/'metrics.csv')
 for case in cases:
  fig,axes=plt.subplots(2,3,figsize=(15,7),squeeze=False)
  for j,seed in enumerate((20260914,20260915,20260916)):
   name=f"F_residual_balanced_{case['fold']}_{seed}_components.csv";rr=[r for r in e.readcsv(e.verify(artifacts[name])) if r['episode_id']==case['episode']];assert rr
   ms=[r for r in metrics if r['group']=='F_residual_balanced' and r['fold']==case['fold'] and int(r['seed'])==seed and r['point']=='FPR0.05' and r['rule']=='raw'];full=float(next(r for r in ms if r['intervention']=='unperturbed')['threshold']);base=float(next(r for r in ms if r['intervention']=='base_only_independent_calibration')['threshold']);t=[int(r['t']) for r in rr]
   axes[0,j].plot(t,[float(r['base_probability']) for r in rr],label='C base');axes[0,j].plot(t,[float(r['full_probability']) for r in rr],label='C full');axes[0,j].axhline(full,color='black',ls=':',label='full cal threshold');axes[0,j].axhline(base,color='blue',ls='--',alpha=.4,label='base-only cal threshold');axes[0,j].step(t,[int(r['stage'])/2 for r in rr],color='grey',alpha=.3,label='stage/2');axes[0,j].set_ylim(-.05,1.05);axes[0,j].set_title(str(seed))
   axes[1,j].plot(t,[float(r['residual_logit']) for r in rr],label='bounded residual logit');axes[1,j].axhline(0,color='grey');axes[1,j].axhline(1.9,color='red',ls=':');axes[1,j].axhline(-1.9,color='red',ls=':');axes[1,j].set_ylim(-2.05,2.05);axes[1,j].set_xlabel('Frame')
  for row in axes:
   for ax in row:ax.grid(alpha=.2);ax.legend(fontsize=7)
  fig.suptitle(case['kind']+' / '+case['fold']+' / '+case['episode']);fig.tight_layout();fig.savefig(a.output/(case['kind']+'.png'),dpi=140);plt.close(fig)
 e.js(a.output/'PLOT_AUDIT.json',dict(status='pass',case_count=len(cases),analysis_audit_sha256=e.sha(a.analysis/'AUDIT.json'),source_sha256=e.sha(__file__),outputs={f.name:e.sha(f) for f in a.output.glob('*.png')}))
if __name__=='__main__':main()
