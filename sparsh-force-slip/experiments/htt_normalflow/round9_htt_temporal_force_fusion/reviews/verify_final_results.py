#!/usr/bin/env python3
"""Independent R9 consolidated numeric and sensitivity-output review."""
import argparse,json,csv,hashlib,statistics
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def rows(p):
 with open(p) as f:return list(csv.DictReader(f))
def verify_mapping(mapping):
 for p,h in mapping.items():assert sha(p)==h,p

def run(root,out):
 report=root/'reporting';a=json.loads((report/'REPORT_AUDIT.json').read_text());assert a['status']=='pass' and a['formal_runs']==36 and a['historical_comparisons']==48 and a['test_role_consumed'] is False
 verify_mapping(a['sources']);verify_mapping({str(report/p):h for p,h in a['output_hashes'].items()})
 sens=root/'current_sensitivity';s=json.loads((sens/'AUDIT.json').read_text());assert s['status']=='pass' and s['runs']==36 and s['metrics']==1728 and s['perturbed_predictions']==72 and s['V_force_invariance'] and s['all_model_states_unchanged'] and s['no_recalibration']
 assert len(s['original_prediction_parity'])==36 and all(r['max_abs']<=1e-6 for r in s['original_prediction_parity'])
 verify_mapping(s['sources']);verify_mapping({str(sens/p):h for p,h in s['output_hashes'].items()});verify_mapping({r['path']:r['sha256'] for r in s['prediction_artifacts']});assert len(s['prediction_artifacts'])==108
 for fold in range(1,5):
  for seed in (20260914,20260915,20260916):
   stem=f'V_temporal_htt_leave_p{fold}_{seed}_';base=rows(sens/'predictions'/f'{stem}unperturbed.csv')
   for kind in ('force_fit_mean_zero','force_causal_lag1'):assert rows(sens/'predictions'/f'{stem}{kind}.csv')==base
 m=rows(root/'current_evaluation/metrics.csv');sm=rows(sens/'metrics.csv');assert len(sm)==1728
 keys=('group','fold','seed','point','rule');original={tuple(r[k] for k in keys):r for r in m if r['role']=='validation' and not r['group'].startswith('historical_')}
 for r in sm:
  z=original[tuple(r[k] for k in keys)];assert float(r['threshold'])==float(z['threshold'])
  if r['intervention']=='unperturbed':
   for k in ('tn','fp','fn','tp','events','hits','false_starts','static_alarming_frames','delay_sum','delay_n'):assert float(r[k])==float(z[k]),k
 for r in rows(report/'KEY_CURRENT_WORKPOINTS.csv'):
  xx=[z for z in m if z['group']==r['group'] and z['point']==r['point'] and z['rule']=='raw' and z['role']=='validation'];assert len(xx)==12
  for k,v in r.items():
   if k not in ('group','point'):assert abs(float(v)-statistics.mean(float(z[k]) for z in xx))<1e-12
 for r in rows(report/'CURRENT_FORCE_SENSITIVITY.csv'):
  xx=[z for z in sm if z['group']==r['group'] and z['intervention']==r['intervention'] and z['point']=='FPR0.05' and z['rule']=='raw'];assert len(xx)==12
  for k,v in r.items():
   if k not in ('group','intervention'):assert abs(float(v)-statistics.mean(float(z[k]) for z in xx))<1e-12
 for r in rows(report/'DEPLOYMENT_COST.csv'):
  b=json.loads((root/'benchmark'/f"{r['group']}.json").read_text());assert b['status']=='complete' and b['parity']['pass'] and all(b['frozen'].values());vv=b['measurements'][r['mode']];assert abs(float(r['median_ms'])-statistics.median(t for x in vv for t in x['samples_ms']))<1e-10
  assert int(r['total_parameters'])==sum(b['parameter_counts'].values());assert b['force_executed']==(r['group']!='V_temporal')
  assert all(z['pass'] for z in b['parity']['blocks'].values())
 f=json.loads((root/'future_robustness/analysis/AUDIT.json').read_text());assert f['status']=='pass'
 force=json.loads((root/'force_diagnostics/REPORT_AUDIT.json').read_text());assert force['status']=='pass'
 association=json.loads((root/'force_diagnostics/ASSOCIATION_AUDIT.json').read_text());assert association['status']=='pass'
 checks={'consolidated_sources_and_outputs':True,'all_1728_sensitivity_workpoints':True,'all108_prediction_artifact_hashes':True,'V_24_perturbed_CSVs_identical':True,'all36_original_parity':True,'original_thresholds_all1728_unchanged':True,'unperturbed_event_and_confusion_match_current':True,'all_current_summary_means':True,'all_sensitivity_summary_means':True,'cost_numbers_intermediate_parity_and_freezing':True,'future_force_association_audits_accepted':True}
 result={'status':'pass','reviewer':'r9_prepare independent of root consolidated report and sensitivity implementation','scope':'scientific deliverables and numeric evidence; final local sync separately required','checks':checks,'report_audit_sha256':sha(report/'REPORT_AUDIT.json'),'summary_sha256':sha(report/'SUMMARY_ZH.md'),'current_sensitivity_audit_sha256':sha(sens/'AUDIT.json'),'source_sha256':sha(__file__),'conclusions':['HTT force history improvement small and heterogeneous','HTT explicit delta no consistent incremental benefit','Validation FPR far above calibration target; not low-FPR deployment success','Force error association includes visual-control confounding, not causality','Original-domain future evidence remains development-only and missing history raises false alarms','No new SSL, force, future, NormalFlow neural training or physical/test use','Lightweight pertains to added143k head, not94-101M full system'],'remaining_acceptance':['Read-only final local/server SHA sync proof verification']}
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.root,a.output)
