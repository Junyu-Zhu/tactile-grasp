"""Synthetic 36-identity end-to-end evaluator fixture, never formal evidence."""
import csv,json,sys
from pathlib import Path
import evaluate as e
p=Path(sys.argv[1]);p.mkdir(parents=True,exist_ok=True)
def rec(x):return {'path':str(x.resolve()),'sha256':e.sha(x)}
a=p/'audit.json';a.write_text(json.dumps({'status':'pass','synthetic':True}));s=p/'summary.json';s.write_text('{}');ck=p/'checkpoint.txt';ck.write_text('synthetic');runs=[]
for fold in ('htt_leave_p1','htt_leave_p2','htt_leave_p3','htt_leave_p4'):
 endpoints={};predictions={}
 for role in ('calibration','validation'):
  ep=p/f'{fold}_{role}_end.csv';pr=p/f'{fold}_{role}_pred.csv';ee=[];pp=[]
  for j in range(3):
   for t,y in enumerate([0,0,1,2,2,0],13):
    episode=f'{role}_{j}';ee.append(dict(episode_id=episode,t=t,stage=y,leakage_group=episode,role=role));pp.append(dict(episode_id=episode,t=t,stage=y,p_slip=.2+.6*(y==2)+.01*j,role=role))
  e.csvout(ep,ee);e.csvout(pr,pp);endpoints[role]=rec(ep);predictions[role]=rec(pr)
 for group in e.GROUPS:
  for seed in e.SEEDS:runs.append(dict(group=group,fold=fold,seed=seed,predictions=predictions,endpoints=endpoints,training_summary=rec(s),checkpoint=rec(ck)))
e.js(p/'manifest.json',dict(status='complete',synthetic=True,runs=runs))
