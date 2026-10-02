#!/usr/bin/env python3
"""Derive inventory, checkpoint index and measured budget from actual receipts."""
import argparse,csv,json
from pathlib import Path

def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;inv=json.loads((r/'FORMAL_INVENTORY.json').read_text());audit=json.loads((r/'TRAINING_AUDIT.json').read_text());assert audit['status']=='pass';runs=[];idx=[];receipts=[]
 for run in inv['runs']:
  out=Path(run['output']);d=json.loads((out/'summary.json').read_text());assert d['status']=='complete' and d['formal']
  rr=[json.loads(p.read_text()) for p in (r/'formal_queue').glob(run['id']+'.receipt.attempt*.json')];good=[x for x in rr if x['status']=='complete'];assert len(good)==1;receipts+=rr
  runs.append(dict(run,status='complete',horizons=inv['horizons'],summary=str(out/'summary.json'),log=good[0]['log'],best=d['artifacts']['best'],latest=d['artifacts']['latest']))
  for k in ['best','latest']:idx.append({'id':run['id'],'group':run['group'],'seed':run['seed'],'kind':k,'path':d['artifacts'][k]['path'],'sha256':d['artifacts'][k]['sha256'],'epochs':len(d['history']),'best_epoch':d['best_epoch'],'parameters':d['model_parameters'],'log':good[0]['log']})
 with (r/'CHECKPOINT_INDEX.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(idx[0]));w.writeheader();w.writerows(idx)
 save(r/'RUN_INVENTORY.json',{'status':'all_formal_training_complete','training_runs_executed':len(runs),'max_runs':12,'runs':runs,'analysis_review_delivery':'Tracked by FINAL_STATUS.json; this inventory alone is not final completion'})
 state=json.loads((r/'formal_queue/QUEUE_STATE.json').read_text());save(r/'RUNTIME_BUDGET_FINAL.json',{'goal_started_unix':1789470628,'formal_started_unix':min(x['started_unix'] for x in receipts),'formal_finished_unix':max(x['finished_unix'] for x in receipts),'formal_queue_wall_seconds':max(x['finished_unix'] for x in receipts)-min(x['started_unix'] for x in receipts),'summed_job_seconds':sum(x['finished_unix']-x['started_unix'] for x in receipts),'attempts':len(receipts),'adjustments':state['adjustments'],'deadline':'2026-09-25T01:55:53+08:00','no_new_training_after':'2026-09-23T01:55:53+08:00','remaining_training':0,'scope':'Frozen-feature training throughput only; not end-to-end inference latency'})
 print(json.dumps({'completed_runs':len(runs),'checkpoint_entries':len(idx)}))
if __name__=='__main__':main()
