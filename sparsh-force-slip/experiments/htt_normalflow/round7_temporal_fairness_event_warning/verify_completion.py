#!/usr/bin/env python3
"""Fail closed on applicable R7 scientific, review and delivery requirements."""
import argparse,hashlib,json
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);ap.add_argument('--pre-sync',action='store_true');ap.add_argument('--output',type=Path);a=ap.parse_args();r=a.root;checks={}
 def read(n):return json.loads((r/n).read_text())
 try:
  for n in ['LOCAL_ROUND6_IDENTITY.json','SERVER_ROUND6_IDENTITY.json']:
   d=read(n);checks[n]=d['status']=='pass' and all(d['checks'].values())
  d=read('audit/support_decision.json');checks['support_complete']=d['status']=='complete'
  inv=read('RUN_INVENTORY.json');checks['max12']=len(inv['runs'])<=12
  if d['training_triggered']:
   count=12 if d['D_triggered'] else 9
   checks['all_formal_complete']=len(inv['runs'])==count and all(x['status']=='complete' for x in inv['runs'])
   tr=read('TRAINING_AUDIT.json');checks['training_verified']=tr['status']=='pass' and tr['verified_runs']==count
  else:checks['no_unsupported_training']=inv['training_runs_executed']==0
  for n in ['evaluation/formal_analysis/summary.json','current_reference/REUSE_AUDIT.json']:checks[n]=read(n)['status']=='complete'
  ev=read('evaluation/formal_analysis/summary.json')
  if d['training_triggered']:
   checks['evaluation_run_coverage']=ev['evaluated_runs']==count and ev['expected_runs']==count
   checks['evaluation_output_hashes']=bool(ev['output_hashes']) and all(Path(n).is_file() and sha(Path(n))==h for n,h in ev['output_hashes'].items())
  ea=read('EVALUATION_AUDIT.json');checks['evaluation_audited']=ea['status']=='pass' and not ea['failures']
  ref=read('r6_reference/REFERENCE_MANIFEST.json');checks['r6_reference_complete']=ref['status']=='complete' and ref['reference_only'] and not ref['formal_round7_run'] and len(ref['artifacts'])==27
  req=read('REQUIREMENTS_ACCEPTANCE.json');checks['requirements_coverage']={x['id'] for x in req['requirements']}=={'history','support','training','visual_delta','events','calibration','attribution','failure_cases','current_reference','freeze_resume','reproduce'};checks['all_requirements']=all(x['status'] in ['pass','not_applicable_with_evidence'] and x.get('evidence') and all((r/n).exists() for n in x['evidence']) for x in req['requirements'])
  review=read('INDEPENDENT_FINAL_REVIEW.json');checks['review_pass']=review['status']=='pass' and not review['unresolved_findings'];bind=review['reviewed_artifact_sha256']
  required=['SUMMARY_ZH.md','PROTOCOL.md','NUMERIC_PROTOCOL.json','audit/support_decision.json','audit/history_dependency_audit.json','evaluation/formal_analysis/summary.json','RUN_INVENTORY.json','REQUIREMENTS_ACCEPTANCE.json','REPRODUCE.md']
  if d['training_triggered']:required+=['TRAINING_AUDIT.json','SMOKE_AUDIT.json','FORMAL_INVENTORY.json','CHECKPOINT_INDEX.csv','EVALUATION_AUDIT.json','analysis/AP_BOOTSTRAP.json','SEED_SUMMARIES.json']
  checks['review_coverage']=all(n in bind for n in required);checks['reviewed_unchanged']=all(sha(r/n)==h for n,h in bind.items())
  if not a.pre_sync:
   sync=read('LOCAL_SYNC_PROOF.json');checks['sync']=sync['status']=='pass' and sync['mismatches']==[]
  result={'status':'pass' if all(checks.values()) else 'fail','checks':checks,'pre_sync':a.pre_sync}
 except (OSError,KeyError,ValueError) as e:result={'status':'fail','checks':checks,'pre_sync':a.pre_sync,'error':str(e)}
 if a.output:a.output.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result));return result['status']!='pass'
if __name__=='__main__':raise SystemExit(main())
