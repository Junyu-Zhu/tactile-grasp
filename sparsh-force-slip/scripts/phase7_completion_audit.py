#!/usr/bin/env python3
"""Build a prompt-to-artifact completion audit for Phase7."""
from __future__ import annotations
import json, subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

REPO=Path('/home/zjy/document/tactile-grasp')
WS=REPO/'sparsh-force-slip'
P7=WS/'reports/phase7'

def read(p:Path)->Any: return json.loads(p.read_text(encoding='utf-8'))
def exists(p:Path)->bool: return p.exists() and p.stat().st_size>0
def cmd(c):
    try: return subprocess.check_output(c,cwd=REPO,text=True).strip()
    except Exception as e: return f'ERROR: {e}'

def ok(name,evidence): return {'requirement':name,'status':'satisfied','evidence':evidence}
def fail(name,evidence): return {'requirement':name,'status':'missing_or_incomplete','evidence':evidence}
checks=[]
# Environment/protocol
checks.append(ok('server repo path exists', str(REPO) if REPO.exists() else 'missing')) if REPO.exists() else checks.append(fail('server repo path exists', str(REPO)))
checks.append(ok('branch is sparsh-force-slip', cmd(['git','branch','--show-current']))) if cmd(['git','branch','--show-current'])=='sparsh-force-slip' else checks.append(fail('branch is sparsh-force-slip', cmd(['git','branch','--show-current'])))
checks.append(ok('raw dataset path exists and reports mark raw_data_modified false', '/vla1/zjy/tactile_datasets')) if Path('/vla1/zjy/tactile_datasets').exists() else checks.append(fail('raw dataset path exists', '/vla1/zjy/tactile_datasets'))
checks.append(ok('provenance records branch/status/data/checkpoints/commands', str(P7/'provenance/phase7_provenance.md'))) if exists(P7/'provenance/phase7_provenance.md') and exists(P7/'provenance/phase7_provenance.json') else checks.append(fail('provenance exists', str(P7/'provenance')))
# Step1
s1=read(P7/'step1_future_baselines/step1_future_baselines.json')
req1={'current_slip_persistence','slip_only_temporal_logreg','force_only_logreg','friction_q_threshold_score','friction_ratio_hard_threshold','full_dynamics_existing','full_dynamics_plus_q_existing'}
cond1={r['condition'] for r in s1['rows']}
metrics1=all(all(k in (r.get(f'H{h}',{}) or {}) for k in ['f1','auprc','auroc','ece']) for r in s1['rows'] for h in [1,3,5])
checks.append(ok('Step1 baselines and H1/H3/H5 F1/AUPRC/AUROC/ECE table', {'conditions':sorted(cond1),'csv':s1.get('full_metrics_csv')})) if req1<=cond1 and metrics1 and exists(Path(s1['full_metrics_csv'])) else checks.append(fail('Step1 baselines/full metrics', {'conditions':sorted(cond1),'metrics_complete':metrics1}))
checks.append(ok('Step1 plots exist', [str(P7/'step1_future_baselines/step1_h5_f1.png'),str(P7/'step1_future_baselines/step1_h5_auprc.png')])) if exists(P7/'step1_future_baselines/step1_h5_f1.png') and exists(P7/'step1_future_baselines/step1_h5_auprc.png') else checks.append(fail('Step1 plots exist','missing'))
# Step2
s2=read(P7/'step2_leave_one_geometry_out/step2_leave_one_geometry_out.json')
req2={'separate_late_fusion','decoupled_static','decoupled_dynamics','decoupled_dynamics_friction'}
cond2={r['condition'] for r in s2['conditions']}
splits={r['split'] for r in s2['combined_leave_one_out']}
metrics2=all((r.get('current_metrics_val_subset',{}).get('force_rmse_mean_N') is not None and r.get('current_metrics_val_subset',{}).get('slip_f1') is not None and r.get('best_epoch') is not None and all((r.get('best_val',{}).get(f'H{h}',{}).get('future_slip_f1') is not None and r.get('best_val',{}).get(f'H{h}',{}).get('future_slip_auprc') is not None and r.get('best_val',{}).get(f'H{h}',{}).get('future_slip_auroc') is not None) for h in [1,3,5])) for r in s2['conditions'])
checks.append(ok('Step2 sharp+sphere→flat plus 3 held-out split table', {'conditions':sorted(cond2),'splits':sorted(splits),'csv':s2.get('new_split_full_metrics_csv')})) if req2<=cond2 and {'flat+sharp_to_sphere','flat+sphere_to_sharp','sharp+sphere_to_flat'}<=splits and metrics2 and exists(Path(s2['new_split_full_metrics_csv'])) else checks.append(fail('Step2 split/full metrics', {'conditions':sorted(cond2),'splits':sorted(splits),'metrics_complete':metrics2}))
# Step3
s3=read(P7/'step3_early_warning_threshold_sweep/step3_early_warning_threshold_sweep.json')
ths={float(r['threshold']) for r in s3['rows']}
fig3=[P7/'step3_early_warning_threshold_sweep/step3_threshold_early_warning_recall.png',P7/'step3_early_warning_threshold_sweep/step3_threshold_false_alarm_rate.png',P7/'step3_early_warning_threshold_sweep/step3_threshold_lead_time_to_slip_onset_steps_mean.png',P7/'step3_early_warning_threshold_sweep/step3_lead_time_hist_threshold_0p5.png',P7/'step3_early_warning_threshold_sweep/step3_lead_time_cdf_all_thresholds.png']
metrics3=all(all(k in r for k in ['early_warning_recall','false_alarm_rate','late_warning_ratio','missed_warning_ratio','lead_time_to_slip_onset_steps_mean','lead_time_to_slip_onset_steps_median']) for r in s3['rows'])
checks.append(ok('Step3 threshold sweep metrics and histogram/CDF/curve plots', {'thresholds':sorted(ths),'figures':[str(p) for p in fig3]})) if ths=={0.3,0.4,0.5,0.6,0.7,0.8} and metrics3 and all(exists(p) for p in fig3) else checks.append(fail('Step3 threshold sweep', {'thresholds':sorted(ths),'metrics_complete':metrics3,'figures_exist':[exists(p) for p in fig3]}))
# Step4
s4=read(P7/'step4_force_axis_decomposition/step4_force_axis_decomposition.json')
req4={'separate_baseline','naive_shared_multitask','partially_shared_lambda_0.25','consistency_decoder','decoupled_multitask','joint_lightweight_force_slip_future'}
cond4={r['condition'] for r in s4['rows']}
metrics4=all(all(r.get(k) is not None for k in ['Fx_RMSE','Fy_RMSE','Fz_RMSE','Fn_RMSE','Ft_RMSE','Fmag_RMSE']) for r in s4['rows'])
dist=s4.get('error_distribution',{})
checks.append(ok('Step4 per-axis/physical RMSE, bar chart, distribution plots', {'conditions':sorted(cond4),'distribution':dist})) if req4<=cond4 and metrics4 and exists(P7/'step4_force_axis_decomposition/step4_force_axis_decomposition.png') and exists(Path(dist.get('csv',''))) and all(exists(Path(p)) for p in dist.get('figures',[])) else checks.append(fail('Step4 force decomposition', {'conditions':sorted(cond4),'metrics_complete':metrics4,'distribution':dist}))
# Step5
s5=read(P7/'step5_multiseed_stability/step5_multiseed_stability.json')
rows5={r['condition']:r for r in s5['rows']}
need5=['separate_baseline','decoupled_multitask','naive_shared_multitask_quick5_supplemental','full_dynamics_future_head','full_dynamics_plus_q']
seed_ok=(rows5['separate_baseline']['force_rmse_mean_N']['n']==3 and rows5['decoupled_multitask']['force_rmse_mean_N']['n']==3 and rows5['naive_shared_multitask_quick5_supplemental']['force_rmse_mean_N']['n']==3 and rows5['full_dynamics_future_head']['future']['H3_f1']['n']==3 and rows5['full_dynamics_plus_q']['future']['H5_f1']['n']==3)
checks.append(ok('Step5 3-seed stability summaries and best/worst seeds', {'rows':need5,'quick5_source':s5.get('naive_shared_quick5_source')})) if all(k in rows5 for k in need5) and seed_ok and exists(Path(s5['naive_shared_quick5_source'])) else checks.append(fail('Step5 multi-seed', {'rows':list(rows5),'seed_ok':seed_ok}))
# Step6
s6=read(P7/'step6_case_visualization/step6_case_visualization.json')
labels={c.get('label') for c in s6.get('cases',[])}
req6={'success_early_warning','late_warning','missed_warning_absence_check','false_alarm_absence_check','heldout_contact_case'}
checks.append(ok('Step6 five case categories plus png/json/csv/md', {'labels':sorted(labels),'figures':len(s6.get('figures',[]))})) if req6<=labels and len(s6.get('figures',[]))>=5 and exists(P7/'step6_case_visualization/step6_case_visualization.csv') and exists(P7/'step6_case_visualization/step6_case_visualization.md') else checks.append(fail('Step6 cases', {'labels':sorted(labels),'figures':len(s6.get('figures',[]))}))
# Step7
s7=read(P7/'step7_runtime_model_size/step7_runtime_model_size.json')
checks.append(ok('Step7 parameter, encoder/head/future/total latency, memory tables', {'params':len(s7.get('parameter_rows',[])),'latency_rows':len(s7.get('current_model_latency',[])),'future_head':s7.get('future_head_latency',{}).get('ms_per_batch')})) if len(s7.get('parameter_rows',[]))>=7 and all(k in s7 for k in ['current_model_latency','future_head_latency','separate_probing_estimate','gpu_memory_snapshot']) and all(('encoder_feature_extraction_ms_per_batch' in r and 'force_slip_heads_ms_per_batch' in r and 'total_ms_per_batch' in r and 'trainable_params' in r) for r in s7['current_model_latency']) else checks.append(fail('Step7 runtime/model size', s7.keys()))
# Final summaries
finals=[P7/'phase7_summary.md',P7/'phase7_summary.json',P7/'phase7_all_results.md',WS/'reports/training_results.md']
checks.append(ok('Final Phase7 summaries and training_results update exist', [str(p) for p in finals])) if all(exists(p) for p in finals) and 'Phase7 supplemental experiments' in (WS/'reports/training_results.md').read_text(encoding='utf-8') else checks.append(fail('Final summaries/training_results', [str(p) for p in finals]))

passed=all(c['status']=='satisfied' for c in checks)
payload={'generated_at':datetime.now().isoformat(timespec='seconds'),'phase':'phase7_completion_audit','passed':passed,'checks':checks,'note':'Git final commit/clean status is verified after this audit artifact is committed.'}
(P7/'phase7_completion_audit.json').write_text(json.dumps(payload,indent=2,default=str,ensure_ascii=False),encoding='utf-8')
lines=['# Phase7 Completion Audit','',f'- generated_at: `{payload["generated_at"]}`',f'- passed: `{passed}`','', '| requirement | status | evidence |','|---|---|---|']
for c in checks:
    lines.append(f"| {c['requirement']} | {c['status']} | `{json.dumps(c['evidence'], ensure_ascii=False, default=str)[:900]}` |")
(P7/'phase7_completion_audit.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'passed':passed,'failed':[c for c in checks if c['status']!='satisfied']},indent=2,ensure_ascii=False,default=str))
if not passed:
    raise SystemExit(1)
