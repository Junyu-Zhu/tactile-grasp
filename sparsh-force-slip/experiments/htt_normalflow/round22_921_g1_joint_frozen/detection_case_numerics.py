#!/usr/bin/env python3
import argparse,csv,json
from pathlib import Path
import numpy as np,torch
import train_frozen as T

def read(p):return list(csv.DictReader(open(p,newline='')))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--inventory',type=Path,required=True);ap.add_argument('--core',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();inv=json.loads(a.inventory.read_text())['runs'];thr={(int(r['fold']),int(r['seed']),r['group']):float(r['threshold']) for r in read(a.core/'CALIBRATION_THRESHOLDS.csv') if r['policy']=='FPR5'};rows=[]
 for item in json.loads((a.core/'SELECTED_CASES.json').read_text())['selected']:
  for group in item['comparison'].split('-'):
   rr=next(r for r in inv if r['group']==group and r['fold']==int(item['fold']) and r['seed']==int(item['seed']));out=Path(rr['output']);cm=json.loads((out/'COMMIT.json').read_text());ck=torch.load(out/cm['best']['path'],map_location='cpu',weights_only=False);d=torch.load(rr['data'],map_location='cpu',weights_only=False);r=d['roles']['validation'];mask=np.asarray(r['episode_id'],object)==item['episode'];n=ck['normalizer'];m=T.init_model(group,rr['seed']);m.load_state_dict(ck['model']);prob=T.infer(m,(r['x']-n['mean'])/n['std'],'cpu').numpy()[mask];stage=r['stage'].numpy()[mask];t=thr[int(item['fold']),int(item['seed']),group];rows.append({'rule':item['rule'],'fold':item['fold'],'seed':item['seed'],'episode':item['episode'],'comparison':item['comparison'],'group':group,'cal_FPR5_threshold_full_precision':format(t,'.17g'),'probability_min':format(float(prob.min()),'.17g'),'probability_max':format(float(prob.max()),'.17g'),'near_one_fraction_gt_0.999999':float((prob>.999999).mean()),'static_false_alarm_frames':int(((stage==0)&(prob>=t)).sum()),'gross_miss_frames':int(((stage==2)&(prob<t)).sum()),'note':'threshold and probability are full precision; near-one saturation can visually overlap p=1 while probability remains below threshold'})
 with (a.output/'FAILURE_NUMERICS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (a.output/'FAILURE_FIGURE_NOTES.md').write_text('''# Detection failure figure notes\n\nFigures use the fixed preregistered fold, seed, episode, comparison and calibration-FPR5 threshold. `FAILURE_NUMERICS.csv` preserves full-precision thresholds and probability ranges. Several traces and thresholds are both very near 1, so rasterized curves can overlap visually at 1 even when a gross-frame probability remains numerically below its threshold and is correctly marked as a miss.\n''');print(json.dumps({'rows':len(rows),'status':'complete'}))
if __name__=='__main__':main()
