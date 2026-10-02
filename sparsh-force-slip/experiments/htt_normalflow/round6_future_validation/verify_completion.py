#!/usr/bin/env python3
"""Fail closed on incomplete scientific, conditional and delivery acceptance."""
import argparse,hashlib,json
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--pre-sync',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args();r=a.root;checks={}
 def read(n):
  q=r/n
  if not q.is_file():raise ValueError('Missing required artifact '+str(q))
  return json.loads(q.read_text())
 try:
  for n in ['LOCAL_ROUND5_IDENTITY.json','SERVER_ROUND5_IDENTITY.json']:
   d=read(n);checks[n]=d['status']=='pass' and all(d['checks'].values())
  for n in ['current/summary.json','reevaluation/summary.json']:
   checks[n]=read(n)['status']=='complete'
  decision=read('support/support_decision.json');checks['support_audit']=decision['status']=='complete'
  inventory=read('RUN_INVENTORY.json');runs=inventory['runs'];checks['max9']=len(runs)<=9
  if decision['training_triggered']:
   checks['training_complete']=len(runs)==9 and all(x['status']=='complete' for x in runs)
   checks['training_verified']=read('TRAINING_AUDIT.json')['status']=='pass' and read('TRAINING_AUDIT.json')['verified_runs']==9
   ev=read('training_evaluation/summary.json');checks['nine_runs_evaluated']=ev['status']=='complete' and ev['neural_runs']==9
   checks['evaluation_outputs_unchanged']=all(sha(r/'training_evaluation'/n)==v for n,v in ev['output_hashes'].items())
  else:
   checks['no_training_without_gate']=all(x['status']=='not_triggered' and x.get('reason') for x in runs) and inventory['training_runs_executed']==0
  review=read('INDEPENDENT_FINAL_REVIEW.json');checks['independent_review']=review['status']=='pass' and not review['unresolved_findings']
  required=['SUMMARY_ZH.md','PROTOCOL.md','support/support_decision.json','current/summary.json','reevaluation/summary.json','RUN_INVENTORY.json']
  bindings=review['reviewed_artifact_sha256'];checks['review_binding_coverage']=all(n in bindings for n in required)
  checks['reviewed_files_unchanged']=all(sha(r/n)==v for n,v in bindings.items())
  for n in ['SUMMARY_ZH.md','REPRODUCE.md','CHECKPOINT_INDEX.csv','REQUIREMENTS_ACCEPTANCE.json']:
   checks['exists:'+n]=(r/n).is_file() and (r/n).stat().st_size>0
  req=read('REQUIREMENTS_ACCEPTANCE.json');checks['all_applicable_requirements']=all(x['status'] in ['pass','not_applicable_with_evidence'] and x.get('evidence') for x in req['requirements'])
  if not a.pre_sync:
   sync=read('LOCAL_SYNC_PROOF.json');checks['sync_pass']=sync['status']=='pass' and sync['mismatches']==[]
  result={'status':'pass' if all(checks.values()) else 'fail','checks':checks,'pre_sync':a.pre_sync}
 except (KeyError,ValueError,OSError) as e:result={'status':'fail','checks':checks,'error':str(e),'pre_sync':a.pre_sync}
 if a.output:a.output.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result));return 0 if result['status']=='pass' else 1
if __name__=='__main__':raise SystemExit(main())
