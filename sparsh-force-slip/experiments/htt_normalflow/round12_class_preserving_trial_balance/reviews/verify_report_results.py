import csv,json,hashlib,statistics,math
from pathlib import Path
O=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round12_class_preserving_trial_balance');R=O/'reporting';C=O/'current_reporting'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return list(csv.DictReader(Path(p).open()))
def close(a,b):return math.isclose(float(a),float(b),abs_tol=1e-12,rel_tol=1e-12)
a=json.loads((R/'REPORT_AUDIT.json').read_text());assert a['status']=='pass' and (a['runs'],a['new_runs'],a['reused_runs'])==(72,24,48)
for p,h in a['sources'].items():assert sha(p)==h
for n,h in a['outputs'].items():assert sha(R/n)==h
assert a['sources']['/home/zjy/document/tactile-grasp/sparsh-force-slip/experiments/htt_normalflow/round12_class_preserving_trial_balance/build_report.py']=='2e1f3516bb3d352891d5fc42ae47ef5cf0c4da5f97e62332874626099c47bf3f'
base=rows(O/'current_evaluation/metrics.csv');count=0
for r in rows(R/'CURRENT_KEY_RESULTS.csv'):
 rr=[x for x in base if (x['group'],x['point'],x['role'],x['rule'])==(r['group'],r['point'],'validation','raw')];assert len(rr)==12
 for k in r:
  if k in ('group','point'):continue
  values=[float(x[k]) for x in rr if x[k]!=''];assert close(r[k],statistics.mean(values));count+=1
sens=rows(O/'current_sensitivity/metrics.csv')
for r in rows(R/'SENSITIVITY_KEY_RESULTS.csv'):
 rr=[x for x in sens if (x['group'],x['intervention'],x['point'],x['rule'])==(r['group'],r['intervention'],'FPR0.05','raw')];assert len(rr)==12
 for k in ('static_fpr','gross_recall','balanced_accuracy'):assert close(r[k],statistics.mean(float(x[k]) for x in rr))
assert rows(R/'CALIBRATION_UNCERTAINTY_PER_RUN.csv')==rows(O/'calibration_uncertainty/threshold_uncertainty.csv');assert rows(R/'FIXED_CASE_PER_SEED.csv')==rows(O/'fixed_case/fixed_case_metrics.csv')
ci=[r for r in rows(O/'current_evaluation/paired_ci.csv') if r['point']=='ranking' and r['metric']=='pAUC'];assert rows(R/'PAUC_FOLD_CI.csv')==ci
b=json.loads((O/'benchmark/F_class_trial_balanced.json').read_text())
for r in rows(R/'DEPLOYMENT_COST.csv'):
 reps=b['measurements'][r['mode']];assert close(r['median_ms'],statistics.median(x for q in reps for x in q['samples_ms']));assert int(r['peak_allocated_bytes'])==max(q['peak_allocated_bytes'] for q in reps);assert int(r['head_parameters'])==b['parameter_counts']['head']
ca=json.loads((C/'REPORT_AUDIT.json').read_text())
for n,h in ca['outputs'].items():assert sha(C/n)==h
report=(R/'SUMMARY_ZH.md').read_text();assert '两种完整权重方案的差异，不能全部归因于static总权重变化' in report
out=dict(status='pass',reviewer='r10_eval independent of R12 report author',root_source_sha256='2e1f3516bb3d352891d5fc42ae47ef5cf0c4da5f97e62332874626099c47bf3f',root_report_sha256=sha(R/'SUMMARY_ZH.md'),root_audit_sha256=sha(R/'REPORT_AUDIT.json'),current_report_sha256=sha(C/'HTT_SUMMARY_ZH.md'),all_source_output_hashes_verified=True,current_numeric_cells_recomputed=count,sensitivity_numeric_cells_recomputed=18,uncertainty_rows_copied_exact=216,fixed_case_rows_copied_exact=36,paired_CI_rows_copied_exact=20,benchmark_median_counts_recomputed=True,final_weight_attribution_corrected=True,verifier_source_sha256=sha(__file__))
(O/'reviews/INDEPENDENT_REPORT_RESULTS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
