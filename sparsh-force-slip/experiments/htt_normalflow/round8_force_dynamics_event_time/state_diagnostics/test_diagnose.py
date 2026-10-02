import csv,hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;spec=importlib.util.spec_from_file_location('diag',HERE/'diagnose.py');M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
class Tests(unittest.TestCase):
 def test_manifest_requires_pass_audit_and_unique_outer_grid(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);index=root/'index.csv';index.write_text('x\n')
   audit=root/'audit.json';audit.write_text(json.dumps({'schema':'round8_training_audit_v1','status':'pass','run_count':24}))
   digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
   artifacts=[{'group':'P4_state','role':'outer','seed':s,'schema':'round8_future_timeline_v1'} for s in M.SEEDS]
   manifest=root/'manifest.json';manifest.write_text(json.dumps({'schema':'round8_prediction_manifest_v1','status':'complete','training_audit':{'path':str(audit),'sha256':digest(audit)},'checkpoint_index':{'path':str(index),'sha256':digest(index)},'artifacts':artifacts}))
   M.accepted_manifest(manifest)
   payload=json.loads(manifest.read_text());payload['artifacts'].append(dict(artifacts[0]));manifest.write_text(json.dumps(payload))
   with self.assertRaisesRegex(ValueError,'duplicated'):M.accepted_manifest(manifest)
 def test_linear_model_must_be_bound_by_complete_evaluation(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);model=root/'state_linear_baseline.json';model.write_text('{}')
   digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
   summary=root/'summary.json';summary.write_text(json.dumps({'format':'round8_formal_evaluation_v1','status':'complete','test_role_consumed':False,'output_hashes':{'state_linear_baseline.json':digest(model)}}))
   M.validated_linear_source(summary,model)
   model.write_text('{"changed":true}')
   with self.assertRaisesRegex(ValueError,'not accepted'):M.validated_linear_source(summary,model)
 def test_per_trial_variance_and_copy_metrics(self):
  rows=[]
  for t in (1,2,3):
   row={'episode_id':'e','leakage_group':'g','t':str(t),'common_population':str(t != 3),
        'predicted_force_x':'1','predicted_force_y':'1','predicted_force_z':'1'}
   for step in M.STEPS:
    row[f'state_target_mask_tplus{step}']='True'
    for axis in M.AXES:
     row[f'predicted_state_force_tplus{step}_{axis}']='2';row[f'target_state_force_tplus{step}_{axis}']=str(2+t)
   rows.append(row)
  linear={(row['episode_id'],int(row['t']),step):(3.,3.,3.) for row in rows for step in M.STEPS}
  result=M.analyze_rows(rows,20260914,linear)
  self.assertEqual(len(result),10)
  primary=[r for r in result if r['population']=='primary_common' and r['step']==1][0]
  full=[r for r in result if r['population']=='full_timeline' and r['step']==1][0]
  self.assertEqual(primary['n_observed_frames'],2);self.assertEqual(full['n_observed_frames'],3)
  self.assertEqual(full['prediction_temporal_variance_xyz'],0)
  self.assertEqual(full['prediction_vs_current_mse_xyz'],1)
  self.assertEqual(full['target_temporal_variance_xyz'],2/3)
  self.assertEqual(full['linear_target_mse_xyz'],5/3)
  rows[-1]['leakage_group']='other'
  with self.assertRaisesRegex(ValueError,'leakage'):M.analyze_rows(rows,20260914,linear)
 def test_summary_requires_all_seed_steps(self):
  metrics=('prediction_temporal_variance_xyz','target_temporal_variance_xyz','persistence_temporal_variance_xyz','prediction_target_mse_xyz','prediction_target_mae_xyz','prediction_vs_current_mse_xyz','prediction_vs_current_mae_xyz','target_vs_current_mse_xyz','target_vs_current_mae_xyz','linear_temporal_variance_xyz','linear_target_mse_xyz','linear_target_mae_xyz','linear_vs_current_mse_xyz','linear_vs_current_mae_xyz')
  rows=[{'population':p,'seed':s,'step':k,'n_observed_frames':1,**{m:1. for m in metrics}} for p in ('primary_common','full_timeline') for s in M.SEEDS for k in M.STEPS]
  self.assertEqual(len(M.summarize(rows)),30)
  with self.assertRaisesRegex(ValueError,'incomplete'):M.summarize(rows[:-1])
if __name__=='__main__':unittest.main()
