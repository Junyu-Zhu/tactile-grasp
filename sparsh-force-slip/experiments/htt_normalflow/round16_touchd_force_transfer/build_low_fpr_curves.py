#!/usr/bin/env python3
"""Materialize validation raw-score ROC points in the registered low-FPR region."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent/'round13_trial_level_alarm_calibration/evaluation'));import r13_evaluate as r13
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);m=json.loads(a.manifest.read_text());rows=[];inputs={}
 for run in m['runs']:
  episodes=r13.load_episodes(run,'validation');inputs[run['predictions']['validation']['path']]=run['predictions']['validation']['sha256'];inputs[run['endpoints']['validation']['path']]=run['endpoints']['validation']['sha256']
  scores=np.concatenate([e.score[e.stage!=1] for e in episodes]);labels=np.concatenate([e.stage[e.stage!=1]==2 for e in episodes]);assert labels.any() and not labels.all()
  order=np.argsort(-scores,kind='stable');scores=scores[order];labels=labels[order];ends=np.r_[np.flatnonzero(np.diff(scores)),len(scores)-1];tp=np.cumsum(labels)[ends];fp=np.cumsum(~labels)[ends];fpr=fp/(~labels).sum();tpr=tp/labels.sum()
  meta={k:run[k] for k in ('group','fold','seed')};rows.append({**meta,'role':'validation','rule':'raw','threshold':'inf','fpr':0.0,'tpr':0.0,'tp':0,'fp':0,'positive_frames':int(labels.sum()),'negative_frames':int((~labels).sum())})
  stop=int(np.searchsorted(fpr,.1,side='right'))
  for i in range(stop):rows.append({**meta,'role':'validation','rule':'raw','threshold':float(scores[ends[i]]),'fpr':float(fpr[i]),'tpr':float(tpr[i]),'tp':int(tp[i]),'fp':int(fp[i]),'positive_frames':int(labels.sum()),'negative_frames':int((~labels).sum())})
 with (a.output/'LOW_FPR_CURVES.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 colors={'V':'#555555','H':'#2166ac','T_H':'#b2182b'};W,H=900,620;x0,y0,pw,ph=80,40,760,500;parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"><rect width="100%" height="100%" fill="white"/><rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="none" stroke="black"/>']
 for run in m['runs']:
  rr=[x for x in rows if x['group']==run['group'] and x['fold']==run['fold'] and int(x['seed'])==int(run['seed'])];points=' '.join(f'{x0+min(.1,float(x["fpr"]))*pw/.1:.2f},{y0+(1-float(x["tpr"]))*ph:.2f}' for x in rr);parts.append(f'<polyline points="{points}" fill="none" stroke="{colors[run["group"]]}" opacity=".35" stroke-width="1"/>')
 for q in (0,.05,.1):
  x=x0+q*pw/.1;parts += [f'<line x1="{x}" x2="{x}" y1="{y0+ph}" y2="{y0+ph+5}" stroke="black"/>',f'<text x="{x}" y="{y0+ph+18}" text-anchor="middle" font-size="11">{q:.2f}</text>']
 for q in (0,.5,1):
  y=y0+(1-q)*ph;parts += [f'<line x1="{x0-5}" x2="{x0}" y1="{y}" y2="{y}" stroke="black"/>',f'<text x="{x0-9}" y="{y+4}" text-anchor="end" font-size="11">{q:g}</text>']
 for i,(g,c) in enumerate(colors.items()):parts.append(f'<line x1="{x0+520+i*80}" x2="{x0+545+i*80}" y1="575" y2="575" stroke="{c}" stroke-width="3"/><text x="{x0+550+i*80}" y="580" font-size="12">{g}</text>')
 parts.append(f'<text x="{x0+pw/2}" y="600" text-anchor="middle">validation frame FPR (0–0.1)</text><text transform="translate(20 {y0+ph/2}) rotate(-90)" text-anchor="middle">gross-frame TPR</text></svg>');(a.output/'LOW_FPR_CURVES.svg').write_text(''.join(parts))
 result={'schema':'round16_low_fpr_curve_artifact_v1','status':'complete','runs':len(m['runs']),'rows':len(rows),'role':'validation','rule':'raw score ranking; incipient excluded exactly as registered rank metrics','fpr_range':[0,.1],'validation_selects_workpoint':False,'test_consumed':False,'input_hashes':inputs,'csv_sha256':r13.sha256(a.output/'LOW_FPR_CURVES.csv'),'svg_sha256':r13.sha256(a.output/'LOW_FPR_CURVES.svg')};r13.write_json(a.output/'SUMMARY.json',result);print(json.dumps({'status':'complete','rows':len(rows)}))
if __name__=='__main__':main()
