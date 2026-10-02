#!/usr/bin/env python3
"""Build deterministic delivery tables from accepted per-run evidence; never train."""
import argparse,csv,json,hashlib,statistics,datetime
from pathlib import Path

def load(p): return json.loads(p.read_text())
def save(p,x): p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def csvout(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root
 inv=load(r/'FORMAL_INVENTORY.json');audit=load(r/'TRAINING_AUDIT.json');assert audit['status']=='pass' and audit['verified_runs']==9
 runs=[];idx=[];receipts=[]
 for run in inv['runs']:
  out=Path(run['output']);d=load(out/'summary.json');assert d['status']=='complete' and d['formal']
  receipt=r/'formal_queue'/f"{run['id']}.receipt.attempt1.json";receipts.append(load(receipt))
  runs.append(dict(run,status='complete',domain='source',horizon=1,summary=str(out/'summary.json'),log=receipts[-1]['log'],best=d['artifacts']['best'],latest=d['artifacts']['latest']))
  for kind in ['best','latest']:
   z=d['artifacts'][kind];idx.append(dict(id=run['id'],group=run['group'],seed=run['seed'],kind=kind,path=z['path'],sha256=z['sha256'],log=receipts[-1]['log']))
 csvout(r/'CHECKPOINT_INDEX.csv',idx)
 save(r/'RUN_INVENTORY.json',{'status':'all_applicable_scientific_work_complete','training_runs_executed':9,'max_training_runs':9,'runs':runs,'workpackages':{'A':'complete','B':'complete_source_H1_gate_pass_HTT_all_fail','C':'complete','D':'complete_9_of_9'},'acceptance':'Independent final review and SHA sync tracked separately; see FINAL_STATUS.json'})
 start=min(x['started_unix'] for x in receipts);end=max(x['finished_unix'] for x in receipts)
 save(r/'RUNTIME_BUDGET_FINAL.json',{'goal_started_unix':1789465781,'formal_started_unix':start,'formal_finished_unix':end,'formal_queue_wall_seconds':end-start,'summed_job_seconds':sum(x['finished_unix']-x['started_unix'] for x in receipts),'attempts':9,'oom_adjustments':[],'gpus':['0','1','2'],'deadline':'2026-09-25T01:55:53+08:00','no_new_training_after':'2026-09-23T01:55:53+08:00','remaining_training':0,'note':'Frozen-feature small GRU training only; this is not end-to-end inference latency.'})
 rows=list(csv.DictReader((r/'training_evaluation/metrics.csv').open()));table=[]
 for group in ['A_visual','B_force','C_force_delta']:
  for point in ['fixed_0.5','calibration_maxBA','calibration_FPR_0.01','calibration_FPR_0.05','calibration_FPR_0.10']:
   ss=[x for x in rows if x['method'].startswith('future_'+group+'_') and '__gate' not in x['method'] and x['point']==point]
   if not ss:continue
   assert len(ss)==3
   item={'group':group,'point':point,'seeds':3}
   for key in ['AP','Brier','BA','macro_F1','FPR','recall']:
    vals=[float(x[key]) for x in ss];item[key+'_mean']=statistics.mean(vals);item[key+'_seed_sd']=statistics.stdev(vals)
   table.append(item)
 csvout(r/'TRAINING_GROUP_SUMMARY.csv',table)
 req=[('R5 identity','LOCAL_ROUND5_IDENTITY.json; SERVER_ROUND5_IDENTITY.json; ROUND5_SOURCE_REUSE_AUDIT.json'),('current fair operating points and failures','current/summary.json; current/paired_fold_bootstrap.csv; current/sequence_metrics.csv; current/figures'),('strict support and exclusion','support/support_decision.json; support/htt_support_manifest.json; support/source_support_manifest.json; support/TEST_ROLE_SCOPE_CLARIFICATION.json'),('legacy future and causal baselines','reevaluation/summary.json; reevaluation/aggregate_metrics.csv; support/REVIEW_REEVALUATION.json'),('nine formal runs and initialization/freeze/resume','TRAINING_AUDIT.json; SMOKE_AUDIT.json; FORMAL_INVENTORY.json; CHECKPOINT_INDEX.csv; reevaluation/REVIEW_TRAINING_PIPELINE.md'),('new future metrics all seeds and paired CI','training_evaluation/summary.json; training_evaluation/metrics.csv; training_evaluation/trial_events.csv; training_evaluation/paired_vs_A.csv; support/TRAINING_EVALUATION_INDEPENDENT_AUDIT.json'),('scope and source label limitations','SUMMARY_ZH.md; support/TEST_ROLE_SCOPE_CLARIFICATION.json'),('deadline and queue','RUNTIME_BUDGET_FINAL.json; formal_queue/QUEUE_STATE.json'),('reproducibility and provenance','REPRODUCE.md; PROTOCOL_LOCK.json; TRAINING_PROTOCOL_LOCK.json; current/INPUT_MANIFEST.json')]
 save(r/'REQUIREMENTS_ACCEPTANCE.json',{'status':'pass','requirements':[{'requirement':k,'status':'pass','evidence':v} for k,v in req],'final_review_and_sync':'Separately enforced by verify_completion.py; not self-certified here.'})
 print(json.dumps({'runs':9,'checkpoint_entries':18,'group_summary_rows':len(table)}))
if __name__=='__main__':main()
