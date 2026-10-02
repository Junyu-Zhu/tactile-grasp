#!/usr/bin/env python3
"""Append accepted R3/R5 predictions to a new R9 manifest without retraining."""
import argparse,json
from pathlib import Path
import evaluate as e
p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--r5-inventory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();m=json.loads(a.manifest.read_text());inv=json.loads(a.r5_inventory.read_text());new=[r for r in m['runs'] if not r.get('historical',False)]
if len(new)!=36 or len(m['runs'])!=36:raise ValueError('Require original R9-only 36-run manifest')
lookup={(r['fold'],r['seed']):r['endpoints'] for r in new};added=[]
for job in inv['jobs']:
 if not any(str(x).endswith('/evaluate_slip.py') for x in job.get('argv',[])):continue
 path=Path(job['acceptance_path']);record=json.loads(path.read_text());argv=job['argv'];arg=lambda key:argv[argv.index(key)+1]
 if record.get('status')!='complete' or record.get('formal') is not True:raise ValueError('Unaccepted history')
 f,s=arg('--fold'),int(arg('--seed'));model=arg('--model-id');prov=record['provenance']
 if record['fold']!=f or record['seed']!=s or record['model_id']!=model:raise ValueError('History identity mismatch')
 predictions={role:{'path':arg('--'+role),'sha256':prov['files'][arg('--'+role)]} for role in ('calibration','validation')}
 rec=dict(group='historical_'+model,fold=f,seed=s,historical=True,historical_evaluation={'path':str(path),'sha256':e.sha(path)},training_summary={'path':prov['training_summary_path'],'sha256':prov['training_summary_sha256']},checkpoint={'path':prov['best_checkpoint'],'sha256':prov['best_checkpoint_sha256']},predictions=predictions,endpoints=lookup[f,s])
 for proof in [rec['training_summary'],rec['checkpoint'],*predictions.values()]:e.verify(proof)
 added.append(rec)
expected={(g,f,s) for g in ('historical_mae-r3-b','historical_V','historical_F-old','historical_F-adapt') for f in ('htt_leave_p1','htt_leave_p2','htt_leave_p3','htt_leave_p4') for s in e.SEEDS}
if len(added)!=48 or {(r['group'],r['fold'],r['seed']) for r in added}!=expected:raise ValueError('Incomplete history grid')
m['runs']+=added;m['history_inventory']={'path':str(a.r5_inventory),'sha256':e.sha(a.r5_inventory)};e.js(a.output,m)
