"""CPU synthetic arithmetic/role/paired-bootstrap checks, never formal data."""
import importlib.util,json,tempfile,subprocess,sys
from pathlib import Path
s=importlib.util.spec_from_file_location('agg',Path(__file__).with_name('aggregate.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
with tempfile.TemporaryDirectory(prefix='r10-force-aggregate-synthetic-') as td:
 root=Path(td);contract=root/'contract.json';contract.write_text(json.dumps({'entries':[dict(episode_id=ep,task='force',frames=20,roles_by_fold={f:'train' for f in m.FOLDS}) for ep in ('a','b')]}));support=root/'support';inp=root/'input';out=root/'output';support.mkdir();inp.mkdir()
 for fold in m.FOLDS:
  sp=support/f'fold_p{fold[-1]}.json';sp.write_text(json.dumps(dict(status='pass',fold=fold,entries=[dict(episode_id='a',role='fit',outer_role='train'),dict(episode_id='b',role='selection',outer_role='train')],provenance={'contract':str(contract),'contract_sha256':m.sha(contract)})));
  for seed in m.SEEDS:
   for reg in (False,True):
    d=inp/f'{reg}_{fold}_{seed}';d.mkdir();rows=[]
    for variant in ('old','new'):
     for episode,n in [('a',2),('b',6)]:
      for axis in ('shear_x','shear_y','normal'):
       value=(2 if episode=='a' else 4)-(variant=='new');r=dict(fold=fold,seed=seed,variant=variant,task='old_force_regression' if reg else 'slip_force',role='train',episode_id=episode,leakage_group=episode,population='all',axis=axis,n=n);r.update({k:value for k in m.METRICS});rows.append(r)
    f=d/'per_trial_axis.csv';m.csvout(f,rows);(d/'AUDIT.json').write_text(json.dumps(dict(status='pass',test_consumed=False,regression=reg,fold=fold,seed=seed,rows=len(rows),trials=2,source_trials=2,evaluated_trials=2,excluded_no_common_endpoint=[],support_sha256=m.sha(sp),output_hashes={'per_trial_axis.csv':m.sha(f)})))
 subprocess.run([sys.executable,str(Path(m.__file__)),'--root',str(inp),'--support',str(support),'--output',str(out)],check=True)
 ci=m.read(out/'paired_group_ci.csv');assert all(float(r['difference_new_minus_old'])==-1 and float(r['ci_lower'])==-1 and float(r['ci_upper'])==-1 for r in ci)
 summary=m.read(out/'per_run_summary.csv');assert {'fit','selection','legacy_force_train'}=={r['role'] for r in summary}
 row=next(r for r in summary if r['task']=='old_force_regression' and r['variant']=='old' and r['aggregation']=='frame_weighted');assert abs(float(row['rmse'])-(13**.5))<1e-12
 print(json.dumps({'status':'pass','synthetic_only':True,'checks':['24 grid','internal roles restored','shared seed bootstrap difference exact','frameweighted RMSE sqrtMSE']}))
