#!/usr/bin/env python3
"""C-versus-A/B errors at each model's own fixed calibration working point."""
import argparse,importlib.util,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('r11_compare_evaluation',HERE.parent/'evaluation/evaluate.py');e=importlib.util.module_from_spec(s);s.loader.exec_module(e)

def compare(stage,base_scores,candidate_scores,base_threshold,candidate_threshold):
 y=np.asarray(stage);b=np.asarray(base_scores)>=base_threshold;c=np.asarray(candidate_scores)>=candidate_threshold
 if y.shape!=b.shape or y.shape!=c.shape or not np.isin(y,[0,1,2]).all():raise ValueError('invalid aligned classes')
 return dict(static_frames=int((y==0).sum()),gross_frames=int((y==2).sum()),incipient_excluded=int((y==1).sum()),static_corrected=int(((y==0)&b&~c).sum()),static_harmed=int(((y==0)&~b&c).sum()),gross_corrected=int(((y==2)&~b&c).sum()),gross_harmed=int(((y==2)&b&~c).sum()),common_static_error=int(((y==0)&b&c).sum()),common_gross_error=int(((y==2)&~b&~c).sum()),base_static_errors=int(((y==0)&b).sum()),candidate_static_errors=int(((y==0)&c).sum()),base_gross_errors=int(((y==2)&~b).sum()),candidate_gross_errors=int(((y==2)&~c).sum()))

def main():
 p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);protocol=json.loads((HERE/'COMPARE_PROTOCOL.json').read_text());audit=json.loads(a.audit.read_text());groups=('V_balanced','F_history_balanced','F_residual_balanced');seeds=(20260914,20260915,20260916);folds=[f'htt_leave_p{i}' for i in range(1,5)]
 if audit.get('status')!='pass' or audit.get('schema')!='round11_fusion_training_audit_v1' or audit.get('expected_count')!=36 or audit.get('test_role_consumed') is not False:raise ValueError('formal acceptance required')
 runs=audit['accepted_runs'];assert len(runs)==36 and {(r['group'],r['fold'],r['seed']) for r in runs}=={(g,f,s) for g in groups for f in folds for s in seeds}
 inputs={str(a.audit):e.sha(a.audit)};data={};points={}
 for r in runs:
  summary=json.loads(e.verify(r['training_summary']).read_text());assert summary['schema']=='round11_htt_fusion_summary_v1' and summary['status']=='complete' and not summary['smoke'];assert all(summary[k]==r[k] for k in ('group','fold','seed'))
  for name in ('training_summary','checkpoint'):e.verify(r[name]);inputs[r[name]['path']]=r[name]['sha256']
  config_path=Path(r['training_summary']['path']).with_name('config.json');assert summary['output_hashes']['config.json']==e.sha(config_path);cfg=json.loads(config_path.read_text());assert all(cfg[k]==r[k] for k in ('group','fold','seed')) and not cfg['smoke']
  prep=Path(cfg['prepared']['path']);pa=e.verify(cfg['prepared_audit']);aa=json.loads(pa.read_text());assert aa['status']=='pass';inputs[str(config_path)]=e.sha(config_path);inputs[str(pa)]=e.sha(pa);roles={}
  for role in ('calibration','validation'):
   ep=prep.with_name(f'endpoints_{role}.csv');record={'path':str(ep),'sha256':aa['output_hashes'][str(ep)]};roles[role]=e.load_rows(r['predictions'][role],record,role,fold=r['fold']);inputs[str(ep)]=record['sha256'];inputs[r['predictions'][role]['path']]=r['predictions'][role]['sha256']
  assert {x['group'] for x in roles['calibration']}.isdisjoint({x['group'] for x in roles['validation']})
  key=(r['group'],r['fold'],r['seed']);data[key]=roles;points[key]=e.choose(roles['calibration'])['FPR0.05']
 results=[]
 for candidate,base in protocol['comparisons']:
  for fold in folds:
   for seed in seeds:
    cr=data[(candidate,fold,seed)]['validation'];br=data[(base,fold,seed)]['validation'];identity=lambda rr:[(r['episode'],r['t'],r['stage'],r['group']) for r in rr];assert identity(cr)==identity(br)
    assert identity(data[(candidate,fold,seed)]['calibration'])==identity(data[(base,fold,seed)]['calibration'])
    ct=points[(candidate,fold,seed)];bt=points[(base,fold,seed)]
    for episode in sorted({r['episode'] for r in cr}):
     ix=[i for i,r in enumerate(cr) if r['episode']==episode];result=compare([cr[i]['stage'] for i in ix],[br[i]['score'] for i in ix],[cr[i]['score'] for i in ix],bt,ct)
     assert result['candidate_static_errors']-result['base_static_errors']==result['static_harmed']-result['static_corrected'];assert result['candidate_gross_errors']-result['base_gross_errors']==result['gross_harmed']-result['gross_corrected']
     results.append(dict(candidate=candidate,base=base,fold=fold,seed=seed,episode_id=episode,leakage_group=cr[ix[0]]['group'],base_threshold=bt,candidate_threshold=ct,point='FPR0.05',rule='raw',base_never_alarm=bt>1,candidate_never_alarm=ct>1,**result))
 e.csvout(a.output/'per_trial.csv',results)
 sums=[]
 for candidate,base in protocol['comparisons']:
  for fold in folds:
   for seed in seeds:
    rr=[r for r in results if (r['candidate'],r['base'],r['fold'],r['seed'])==(candidate,base,fold,seed)];sums.append(dict(candidate=candidate,base=base,fold=fold,seed=seed,trials=len(rr),**{k:sum(r[k] for r in rr) for k in result}))
 e.csvout(a.output/'per_run.csv',sums);e.js(a.output/'AUDIT.json',dict(status='pass',runs=36,comparisons=2,trial_records=len(results),run_records=len(sums),protocol_sha256=e.sha(HERE/'COMPARE_PROTOCOL.json'),source_sha256=e.sha(__file__),evaluator_sha256=e.sha(HERE.parent/'evaluation/evaluate.py'),input_hashes=inputs,outputs={f.name:e.sha(f) for f in a.output.glob('*.csv')},same_calibration_thresholds_as_full_models=True,no_validation_recalibration=True,pure_structural_causality=False))
if __name__=='__main__':main()
