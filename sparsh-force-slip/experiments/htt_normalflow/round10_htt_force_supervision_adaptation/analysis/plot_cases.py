#!/usr/bin/env python3
"""Selected diagnostic overlays, all three seeds; no new case selection."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'evaluation'))
import evaluate as e

def main():
 p=argparse.ArgumentParser();p.add_argument('--association',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--force-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 audit=json.loads((a.association/'AUDIT.json').read_text());assert audit['status']=='pass'
 for n,h in audit['outputs'].items():assert e.sha(a.association/n)==h
 manifest=json.loads(a.manifest.read_text());e.validate_manifest(manifest);cases=[r for r in e.readcsv(a.association/'representative_cases.csv') if r['selected']=='True'];frames={};sources={}
 selected={(r['fold'],r['episode']) for r in cases}
 for f in sorted(a.force_root.rglob('frame_errors.csv')):
  fa=json.loads(f.with_name('AUDIT.json').read_text());assert fa['status']=='pass' and fa['output_hashes']['frame_errors.csv']==e.sha(f)
  if fa.get('regression'):continue
  sources[str(f)]=e.sha(f)
  for r in e.readcsv(f):
   if r['role']=='validation' and (r['fold'],r['episode_id']) in selected:
    k=(r['fold'],int(r['seed']),r['episode_id'],r['variant']);frames.setdefault(k,[]).append(r)
 for case in cases:
  fold,episode=case['fold'],case['episode'];fig,ax=plt.subplots(4,3,figsize=(16,11),squeeze=False)
  for column,seed in enumerate(e.SEEDS):
   for group in e.GROUPS:
    r=next(r for r in manifest['runs'] if (r['group'],r['fold'],r['seed'])==(group,fold,seed));rr=e.load_rows(r['predictions']['validation'],r['endpoints']['validation'],'validation');rr=[x for x in rr if x['episode']==episode];cal=e.load_rows(r['predictions']['calibration'],r['endpoints']['calibration'],'calibration');threshold=e.choose(cal)['FPR0.05'];line=ax[0,column].plot([x['t'] for x in rr],[x['score'] for x in rr],label=group)[0];ax[0,column].axhline(threshold,color=line.get_color(),linestyle=':',alpha=.55)
   ax[0,column].step([x['t'] for x in rr],[x['stage']/2 for x in rr],color='grey',alpha=.4,label='stage/2');ax[0,column].set_title(f'{fold} / {seed}');ax[0,column].set_ylim(-.05,1.05)
   for variant in ('old','new'):
    ff=sorted(frames[(fold,seed,episode,variant)],key=lambda x:int(x['t']));ts=[int(x['t']) for x in ff];pred=np.array([json.loads(x['prediction_xyz']) for x in ff]);gt=np.array([json.loads(x['target_xyz']) for x in ff])
    for j,name in enumerate(('shear_x','shear_y','normal')):
     ax[j+1,column].plot(ts,pred[:,j],label=variant)
     if variant=='old':ax[j+1,column].plot(ts,gt[:,j],color='black',linestyle='--',label='same NPZ target')
     ax[j+1,column].set_ylabel(name+' (N)');ax[j+1,column].set_xlabel('Frame')
  for row in ax:
   for item in row:item.grid(alpha=.2);item.legend(fontsize=6)
  fig.suptitle(case['kind']+' : '+episode);fig.tight_layout();fig.savefig(a.output/(case['kind']+'.png'),dpi=140);plt.close(fig)
 e.js(a.output/'PLOT_AUDIT.json',dict(status='pass',cases=len(cases),sources={**sources,str(a.manifest):e.sha(a.manifest),str(a.association/'AUDIT.json'):e.sha(a.association/'AUDIT.json'),str(Path(__file__)):e.sha(__file__)},outputs={f.name:e.sha(f) for f in a.output.glob('*.png')}))
if __name__=='__main__':main()
