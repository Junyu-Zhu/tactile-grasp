#!/usr/bin/env python3
"""Independent output identity, raw confusion and reporting arithmetic audit."""
import csv,json,hashlib,argparse
from pathlib import Path
import numpy as np

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def rows(p):
 with open(p) as f:return list(csv.DictReader(f))
def run(root,out):
 e=root/'current_evaluation';rep=root/'current_reporting';s=json.loads((e/'summary.json').read_text());assert s['status']=='complete' and s['synthetic'] is False and s['new_runs']==36 and s['runs']==84
 for name,h in s['output_hashes'].items():assert sha(e/name)==h
 for p,h in s['input_hashes'].items():assert sha(p)==h
 ra=json.loads((rep/'REPORT_AUDIT.json').read_text());assert ra['status']=='pass' and ra['synthetic'] is False and ra['evaluation_summary_sha256']==sha(e/'summary.json')
 for name,h in ra['outputs'].items():assert sha(rep/name)==h
 m=rows(e/'metrics.csv');pairs=rows(e/'paired_ci.csv');summary=rows(e/'summary_metrics.csv');checks=[]
 for group in ('V_temporal','F_history','F_delta'):
  for fold in range(1,5):
   for seed in (20260914,20260915,20260916):
    p=root/f'formal/{group}_p{fold}_s{seed}/predictions_validation.csv';rr=rows(p);stage=np.array([int(r['stage']) for r in rr]);prob=np.array([float(r['p_slip']) for r in rr]);z=stage!=1;y=stage[z]==2;a=prob[z]>=.5
    counts=[int((~y&~a).sum()),int((~y&a).sum()),int((y&~a).sum()),int((y&a).sum())]
    item=next(r for r in m if r['group']==group and r['fold']==f'htt_leave_p{fold}' and int(r['seed'])==seed and r['point']=='fixed_0.5' and r['rule']=='raw' and r['role']=='validation')
    assert counts==[int(item[k]) for k in ('tn','fp','fn','tp')]
    assert abs(float(item['positive_prevalence'])-float(y.mean()))<1e-12
    checks.append({'group':group,'fold':fold,'seed':seed,'confusion':counts})
 for r in summary:
  selected=[float(x[r['metric']]) for x in m if x['group']==r['group'] and x['point']==r['point'] and x['rule']==r['rule'] and x['role']=='validation' and x[r['metric']]!=''];selected=[x for x in selected if np.isfinite(x)]
  assert len(selected)==int(r['available_runs'])
  if selected:assert abs(float(r['mean'])-np.mean(selected))<1e-12
  if len(selected)>1:assert abs(float(r['sd_descriptive'])-np.std(selected,ddof=1))<1e-12
 for r in pairs:
  assert r['unit']=='complete_leakage_group_shared_across_seeds' and r['fold'] in [f'htt_leave_p{i}' for i in range(1,5)]
  assert int(r['valid_bootstraps'])<=200
  if r['status']=='available':assert float(r['ci_lower'])<=float(r['ci_upper'])
 result={'status':'pass','scope':'current results identity and independent primary arithmetic; no physical claims','input_artifacts_verified':len(s['input_hashes']),'output_artifacts_verified':len(s['output_hashes']),'independent_fixed05_confusions':checks,'summary_rows_checked':len(summary),'paired_ci_rows_checked':len(pairs),'evaluation_summary_sha256':sha(e/'summary.json'),'report_audit_sha256':sha(rep/'REPORT_AUDIT.json'),'review_source_sha256':sha(__file__),'limitations':['Repeated validation checkpoint selection','Overlapping folds not inferentially pooled','HTT force-rule labels','Observed validation FPR far above nominal calibration constraints','Current event coverage does not prove future lead']}
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='independent_fixed05_confusions'}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.root,a.output)
