#!/usr/bin/env python3
"""Descriptive plots from accepted evaluations; no model/threshold selection."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--data',type=Path,required=True);a=p.parse_args();r=a.results;fig=r/'figures';fig.mkdir(exist_ok=True);rows=list(csv.DictReader((r/'metrics.csv').open()));groups=['A_visual','B_force','C_force_delta'];seeds=[20260914,20260915,20260916]
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 f,axs=plt.subplots(1,3,figsize=(11,3.6));fields=[('AP','fixed_0.5','Average precision'),('FPR','calibration_FPR_0.01','Actual FPR at cal 1%'),('recall','calibration_FPR_0.01','Recall at cal 1%')]
 for ax,(field,point,title) in zip(axs,fields):
  vals=[[float(next(x[field] for x in rows if x['method']==f'future_{g}_{s}' and x['point']==point)) for s in seeds] for g in groups];ax.bar(range(3),[np.mean(v) for v in vals],color=['#8795a5','#dcaa5a','#4d9b8b'],alpha=.7)
  for i,v in enumerate(vals):ax.scatter(np.full(3,i),v,c='black',s=15)
  ax.set_xticks(range(3),['Visual','+ Force','+ Force delta']);ax.set_title(title);ax.set_ylim(bottom=0)
 f.suptitle('Source H1: three seeds on the same development population');f.tight_layout();f.savefig(fig/'source_h1_comparison.png',dpi=170);plt.close(f)
 data=torch.load(a.data,map_location='cpu',weights_only=False);role=data['roles']['outer'];ep=np.asarray(role['episode_id']);t=np.asarray(role['t']);y=np.asarray(role['y']);pred=np.load(r/'predictions.npz');summary=json.loads((r/'summary.json').read_text());key='future_C_force_delta_20260914';threshold=summary['thresholds'][key]['calibration_FPR_0.01'];candidates=[]
 for e in sorted(set(ep)):
  ix=np.where(ep==e)[0];negative=ix[y[ix]==0];positive=ix[y[ix]==1]
  candidates.append((e,float(np.mean(pred[key][negative]>=threshold)) if len(negative) else -1,float(np.min(pred[key][positive])) if len(positive) else 2))
 selected={'highest_static_false_alarm_fraction':sorted(candidates,key=lambda x:(-x[1],x[0]))[0][0],'lowest_pre_onset_positive_score':sorted((x for x in candidates if x[2]<=1),key=lambda x:(x[2],x[0]))[0][0]};selection=[]
 for kind,e in selected.items():
  ix=np.where(ep==e)[0];ix=ix[np.argsort(t[ix])];f,ax=plt.subplots(figsize=(9,3.5))
  for g,color in zip(groups,['#687d92','#b77e29','#238670']):ax.plot(t[ix],pred[f'future_{g}_20260914'][ix],label=g,color=color)
  ax.plot(t[ix],pred['current_slip'][ix],color='gray',ls=':',label='current slip');ax.axhline(threshold,color='#238670',ls='--',label='C calibration 1% threshold');onset=role['first_current_slip_t'][ix[0]]
  if onset is not None:ax.axvline(onset,color='black',ls='--',label='dataset-label onset')
  pos=ix[y[ix]==1];ax.scatter(t[pos],np.ones(len(pos)),marker='v',color='red',s=25,label='H1 positive endpoint');ax.set_ylim(-.02,1.06);ax.set_xlabel('Original sample index (not milliseconds)');ax.set_ylabel('Score');ax.set_title(kind+'\n'+e);ax.legend(fontsize=8,ncol=3);f.tight_layout();f.savefig(fig/(kind+'.png'),dpi=170);plt.close(f);selection.append({'selection':kind,'episode_id':e,'seed':20260914,'point':'calibration_FPR_0.01','purpose':'illustration only; selected after scoring and not used for any training/calibration choice'})
 (r/'PLOT_SELECTION.json').write_text(json.dumps(selection,indent=2)+'\n');h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(r/'PLOT_PROVENANCE.json').write_text(json.dumps({'status':'complete','source_sha256':{str(Path(__file__).resolve()):h(Path(__file__).resolve()),str(r/'metrics.csv'):h(r/'metrics.csv'),str(r/'predictions.npz'):h(r/'predictions.npz')},'output_sha256':{str(p):h(p) for p in fig.glob('*.png')}},indent=2)+'\n')
 print(json.dumps({'figures':len(list(fig.glob('*.png')))}))
if __name__=='__main__':main()
