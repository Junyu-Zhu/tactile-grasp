#!/usr/bin/env python3
"""Independent mechanical checks for the bounded F1-F5 review repairs."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from PIL import Image
from touchd_common import atomic_json,sha256
def read(path):
 with path.open(newline='') as f:return list(csv.DictReader(f))
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--code',type=Path,required=True);a=p.parse_args();e=a.root/'evaluation';checks={}
 old=json.loads((e/'old_force_regression/AUDIT.json').read_text());assert old['status']=='pass' and old['runs']==24 and not old['test_consumed'] and old['historical_not_in_r16_transfer_ci'] and old['validation_rows_per_route_axis']==141 and 'not a count of independent trials' in old['summary_trials_field_semantics'] and old['axis_mapping']=={'model_output_0':'shear_x','model_output_1':'shear_y','model_output_2':'normal','source':'accepted R10/R5 force target contract'}
 assert old['historical_aggregation'].startswith('trials-weighted mean') and all(abs(old['historical_validation_episode_weighted_mae_n'][v][axis]-x)<1e-12 for v,axis,x in (('old','shear_x',0.48710391464385583),('old','shear_y',0.6958150218987296),('old','normal',0.944644181440908),('new','shear_x',0.8315010602381212),('new','shear_y',0.819758489199564),('new','normal',1.2490179156157988)));checks['F1_old_force_regression']='pass'
 manifests=list((e/'old_force_regression/exports').glob('*/prediction_manifest.json'));assert len(manifests)==24
 for path in manifests:
  m=json.loads(path.read_text());assert not m['test_consumed'] and not m['training_or_tuning'] and set(m['roles'])=={'train','validation','calibration'} and all(x['role']!='test' for x in m['entries']) and m['frozen_before_after_bitwise']
 case=json.loads((e/'case_evidence/AUDIT.json').read_text());assert case['status']=='complete' and case['curve_plots']==48 and case['raw_tactile_frames']>=4 and case['curve_source_rows']>0 and not case['test_consumed'];plots=read(e/'case_evidence/CASE_PLOT_INDEX.csv');assert len(plots)==48 and all(sha256(Path(x['path']))==x['sha256'] for x in plots);case_svg=Path(plots[0]['path']).read_text();assert all(x in case_svg for x in ('>GT<','>prediction<','>frame t<','>force (N)<'))
 raw=read(e/'case_evidence/TACTILE_RAW_INDEX.csv');source=np.load(raw[0]['source_npz'])['tactile_img']
 for row in raw:assert np.array_equal(np.asarray(Image.open(row['path'])),source[int(row['source_index'])]) and sha256(Path(row['path']))==row['sha256']
 checks['F2_cases_and_raw_images']='pass'
 windows=json.loads((e/'window_diagnostics/AUDIT.json').read_text());assert windows['status']=='complete' and windows['window_definition']['size']==32 and windows['window_definition']['final_window_minimum']==8 and not windows['selection_or_tuning_use'] and not windows['test_consumed']
 for name,key in [('FORCE_TRIAL_AXIS_DIAGNOSTICS.csv','force_trial_axis'),('FORCE_WINDOW_AXIS_DIAGNOSTICS.csv','force_window_axis'),('FUTURE_WINDOW_AXIS_DIAGNOSTICS.csv','future_window_axis')]:
  path=e/'window_diagnostics'/name;count=0
  with path.open(newline='') as f:
   for row in csv.DictReader(f):
    count+=1;assert row['role']!='test' and row['copy_reference'] and 0<=float(row['prediction_exact_boundary_fraction'])<=1 and 0<=float(row['prediction_outside_20_fraction'])<=1
    if 'WINDOW' in name:assert 8<=int(row['n'])<=32
  assert count==windows['rows'][key] and sha256(path)==windows['output_hashes'][name]
 checks['F3_trial_and_window_diagnostics']='pass'
 low=json.loads((e/'low_fpr_curves/SUMMARY.json').read_text());curve=read(e/'low_fpr_curves/LOW_FPR_CURVES.csv');assert low['status']=='complete' and low['runs']==36 and len(curve)==low['rows'] and not low['validation_selects_workpoint'] and all(0<=float(x['fpr'])<=.1+1e-12 for x in curve);low_svg=(e/'low_fpr_curves/LOW_FPR_CURVES.svg').read_text();assert all(x in low_svg for x in ('>0.00<','>0.05<','>0.10<','>0.5<','>1<'));checks['F4_low_fpr_curves']='pass'
 note=json.loads((a.root/'audit/FUTURE_SUMMARY_NOTE_REPAIR.json').read_text());assert note['status']=='pass' and note['files']==24 and not note['metrics_or_predictions_changed']
 expected="Change error and persistence use this route's own predicted-current anchor; ideal GT-current persistence is non-deployable."
 for path in (e/'future').glob('*/SUMMARY.json'):assert json.loads(path.read_text())['note']==expected
 report=(a.root/'audit/DELIVERY_REPORT.md').read_text();assert 'ToucHD 域内 `force_head`' in report and '不自动替换已验收的 R10/R12 管线' in report and '这不是所有 P4 研究的硬门槛' in report;checks['F5_wording_and_scope']='pass'
 sources={name:sha256(a.code/name) for name in ('export_old_force_regression.py','evaluate_old_force_regression.py','run_old_force_regression.py','build_case_evidence.py','build_window_diagnostics.py','build_low_fpr_curves.py','repair_future_summary_notes.py')}
 result={'schema':'round16_bounded_review_repairs_audit_v1','status':'pass','checks':checks,'training_rerun':False,'existing_ci_rerun':False,'test_consumed':False,'source_hashes':sources};atomic_json(a.root/'audit/REVIEW_REPAIRS_AUDIT.json',result);print(json.dumps(result))
if __name__=='__main__':main()
