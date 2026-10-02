import json,csv,hashlib,statistics as st,math
from pathlib import Path
C=Path(__file__).resolve().parents[1];O=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round11_stable_negative_force_residual')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return list(csv.DictReader(open(p)))
a=json.loads((O/'reporting/REPORT_AUDIT.json').read_text());assert a['status']=='pass';hashes={}
for p,h in a['sources'].items():assert sha(p)==h;hashes[p]=h
for n,h in a['outputs'].items():p=O/'reporting'/n;assert sha(p)==h;hashes[str(p)]=h
metrics=rows(O/'current_evaluation/metrics.csv');cells=0
for r in rows(O/'reporting/CURRENT_KEY_RESULTS.csv'):
 src=[x for x in metrics if x['group']==r['group'] and x['point']==r['point'] and x['role']=='validation' and x['rule']=='raw'];assert len(src)==12
 for k,v in r.items():
  if k in ['group','point']:continue
  vals=[float(x[k]) for x in src if x[k] not in ('','None') and math.isfinite(float(x[k]))];expected=st.mean(vals) if vals else None
  assert (v=='' and expected is None) or math.isclose(float(v),expected,abs_tol=1e-12);cells+=1
sens=rows(O/'analysis/metrics.csv');scells=0
for r in rows(O/'reporting/SENSITIVITY_KEY_RESULTS.csv'):
 src=[x for x in sens if x['group']==r['group'] and x['intervention']==r['intervention'] and x['point']=='FPR0.05' and x['rule']=='raw'];assert len(src)==12
 for k in ['static_fpr','gross_recall','balanced_accuracy']:assert math.isclose(float(r[k]),st.mean(float(x[k]) for x in src),abs_tol=1e-12);scells+=1
ci=[x for x in rows(O/'current_evaluation/paired_ci.csv') if x['point']=='ranking' and x['metric']=='pAUC'];assert len(ci)==20 and ci==rows(O/'reporting/PAUC_FOLD_CI.csv')
b=json.loads((O/'benchmark/F_residual_balanced.json').read_text())
for r in rows(O/'reporting/DEPLOYMENT_COST.csv'):
 reps=b['measurements'][r['mode']];assert math.isclose(float(r['median_ms']),st.median(x for p in reps for x in p['samples_ms']),abs_tol=1e-12);assert int(r['head_parameters'])==b['parameter_counts']['head'] and int(r['full_parameters'])==b['total_deployment_parameters']
report=(O/'reporting/SUMMARY_ZH.md').read_text()
for r in rows(O/'reporting/CURRENT_KEY_RESULTS.csv'):
 expected=f"|{r['group']}|{r['point']}|{float(r['static_fpr']):.2%}|{float(r['gross_recall']):.2%}|{float(r['balanced_accuracy']):.4f}|{float(r['macro_f1']):.4f}|{float(r['AP']):.4f}|{float(r['pAUC']):.4f}|";assert expected in report
res={'status':'pass','hashes':hashes,'current_numeric_cells':cells,'sensitivity_numeric_cells':scells,'pAUC_fold_CI_rows_exact':20,'markdown25current_rows_exact':True,'cost_from90samples_permode_exact':True,'scope':'All consolidated data table cells checked against primary evaluator outputs; inference/narrative reviewed separately; does not regenerate bootstrap quantiles'}
(C/'reviews/FINAL_REPORT_NUMERIC_CHECK.json').write_text(json.dumps(res,indent=2));print(json.dumps(res))
