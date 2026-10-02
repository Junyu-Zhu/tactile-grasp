"""Synthetic, isolated end-to-end aggregate contract test; never formal evidence."""
import tempfile,pathlib,json,csv,hashlib,subprocess,sys,math,importlib.util
C=pathlib.Path('/home/zjy/document/tactile-grasp/sparsh-force-slip/experiments/htt_normalflow/round10_htt_force_supervision_adaptation')
p=C/'force/aggregate.py';sp=importlib.util.spec_from_file_location('aggregate_test',p);a=importlib.util.module_from_spec(sp);sp.loader.exec_module(a)
assert math.isclose(a.aggregate([{'n':'1','rmse':'3'},{'n':'3','rmse':'4'}],'rmse',True),math.sqrt(57/4))
assert a.aggregate([{'n':'1','rmse':'3'},{'n':'3','rmse':'4'}],'rmse',False)==3.5
assert a.aggregate([{'n':'1','prediction_std':'3'},{'n':'3','prediction_std':'4'}],'prediction_std',True)==3.75
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
with tempfile.TemporaryDirectory(prefix='r10_synthetic_aggregate_') as t:
 t=pathlib.Path(t);root=t/'inputs';support=t/'support';support.mkdir();contract=[]
 for task in ['slip','force']:
  for k,role in enumerate(['train','train','validation','calibration']):contract.append({'episode_id':f'{task}_{k}','task':task,'frames':50,'roles_by_fold':{f:role for f in a.FOLDS}})
 contract.append({'episode_id':'force_short','task':'force','frames':5,'roles_by_fold':{f:'calibration' for f in a.FOLDS}})
 cp=t/'contract.json';cp.write_text(json.dumps({'entries':contract}))
 for fold in a.FOLDS:
  entries=[{'episode_id':f'slip_{k}','role':role,'outer_role':('train' if role in ('fit','selection') else role)} for k,role in enumerate(['fit','selection','validation','calibration'])]
  (support/f'fold_p{fold[-1]}.json').write_text(json.dumps({'status':'pass','fold':fold,'entries':entries,'provenance':{'contract':str(cp),'contract_sha256':sha(cp)}}))
  for seed in a.SEEDS:
   for regression in [False,True]:
    d=root/f'{fold}_{seed}_{regression}';d.mkdir(parents=True);rows=[]
    for k,role in enumerate(['train','train','validation','calibration']):
     for variant in ['old','new']:
      for axis in ['shear_x','shear_y','normal']:
       row={'fold':fold,'seed':seed,'role':role,'axis':axis,'variant':variant,'task':'old_force_regression' if regression else 'slip_force','episode_id':f'{"force" if regression else "slip"}_{k}','population':'all','leakage_group':f'group{k}','n':10}
       row.update({m:1.0 for m in a.METRICS});row['mae']=row['rmse']=2.0 if variant=='new' else 1.0
       if regression:row['target_clip_fraction']=''
       rows.append(row)
    fp=d/'per_trial_axis.csv'
    with fp.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (d/'AUDIT.json').write_text(json.dumps({'status':'pass','test_consumed':False,'fold':fold,'seed':seed,'regression':regression,'rows':len(rows),'trials':4,'source_trials':5 if regression else 4,'evaluated_trials':4,'excluded_no_common_endpoint':[{'episode_id':'force_short','frames':5,'role':'calibration','reason':'no_common_endpoint_t_ge13'}] if regression else [],'output_hashes':{'per_trial_axis.csv':sha(fp)},'support_sha256':sha(support/f'fold_p{fold[-1]}.json')}))
 out=t/'out';cmd=[sys.executable,str(p),'--root',str(root),'--support',str(support),'--output',str(out)];subprocess.run(cmd,check=True,capture_output=True)
 with (out/'paired_group_ci.csv').open() as f:cis=list(csv.DictReader(f))
 for r in cis:
  if r['metric'] in ('mae','rmse'):assert float(r['difference_new_minus_old'])==float(r['ci_lower'])==float(r['ci_upper'])==1 and r['valid_bootstraps']=='200'
 assert {r['fold'] for r in cis}==set(a.FOLDS)
 fp=next(root.rglob('per_trial_axis.csv'));data=a.read(fp);data=[r for r in data if r['episode_id']!=data[0]['episode_id']]
 with fp.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 ap=fp.with_name('AUDIT.json');audit=json.loads(ap.read_text());audit['output_hashes']['per_trial_axis.csv']=sha(fp);ap.write_text(json.dumps(audit))
 result=subprocess.run(cmd,capture_output=True);assert result.returncode!=0,'whole-trial deletion accepted'
 print(json.dumps({'status':'pass','synthetic_only':True,'checks':['weighted_RMSE_exact','macro_RMSE_exact','within_trial_std_weighting','complete_24_grid_pipeline','three_seed_mean_CI_exact_constant_delta','200_bootstraps_per_finite_cell','folds_separate','whole_trial_deletion_rejected','explicit_short_trial_exclusion'],'source_sha256':sha(p)}))
