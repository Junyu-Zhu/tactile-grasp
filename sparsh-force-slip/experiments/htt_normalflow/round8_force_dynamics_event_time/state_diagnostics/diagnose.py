#!/usr/bin/env python3
"""Per-trial temporal diagnostics for the supervised P4_state output; read-only."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,statistics
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
import torch

SEEDS=(20260914,20260915,20260916);STEPS=range(1,6);AXES=('x','y','z')

def sha256(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def atomic_csv(path:Path,rows:list[dict[str,Any]])->None:
 if not rows:raise ValueError('empty diagnostic table')
 tmp=path.with_name(path.name+f'.tmp.{os.getpid()}');path.parent.mkdir(parents=True,exist_ok=True)
 with tmp.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 os.replace(tmp,path)

def atomic_json(path:Path,value:Any)->None:
 tmp=path.with_name(path.name+f'.tmp.{os.getpid()}');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));os.replace(tmp,path)

def accepted_manifest(path:Path)->tuple[dict[str,Any],dict[str,str]]:
 manifest=json.loads(path.read_text())
 if manifest.get('schema')!='round8_prediction_manifest_v1' or manifest.get('status')!='complete':raise ValueError('accepted manifest required')
 sources={str(path.resolve()):sha256(path)}
 for key in ('training_audit','checkpoint_index'):
  spec=manifest.get(key)
  if not spec:raise ValueError(f'manifest missing {key}')
  bound=Path(spec['path'])
  if sha256(bound)!=spec['sha256']:raise ValueError(f'{key} SHA mismatch')
  sources[str(bound.resolve())]=spec['sha256']
 audit=json.loads(Path(manifest['training_audit']['path']).read_text())
 if audit.get('schema')!='round8_training_audit_v1' or audit.get('status')!='pass' or int(audit.get('run_count',0))!=24:
  raise ValueError('passing 24-run training audit required')
 artifacts=[x for x in manifest.get('artifacts',[]) if x['group']=='P4_state' and x['role']=='outer']
 if len(artifacts)!=len(SEEDS) or [(int(x['seed']),x['schema']) for x in sorted(artifacts,key=lambda x:int(x['seed']))] != [(seed,'round8_future_timeline_v1') for seed in SEEDS]:
  raise ValueError('P4_state outer grid incomplete or duplicated')
 return manifest,sources

def variance_xyz(values:list[tuple[float,float,float]])->float:
 if not values:return math.nan
 return sum(statistics.pvariance(axis) for axis in zip(*values))/3

def mean_metric(left:list[tuple[float,float,float]],right:list[tuple[float,float,float]],absolute:bool=False)->float:
 values=[abs(a-b) if absolute else (a-b)**2 for x,y in zip(left,right) for a,b in zip(x,y)]
 return statistics.mean(values) if values else math.nan

def analyze_rows(rows:list[dict[str,str]],seed:int,linear:dict[tuple[str,int,int],tuple[float,float,float]]|None=None)->list[dict[str,Any]]:
 by_episode=defaultdict(list)
 for row in rows:by_episode[row['episode_id']].append(row)
 result=[]
 for episode,sequence in sorted(by_episode.items()):
  sequence.sort(key=lambda row:int(row['t']))
  if len({int(row['t']) for row in sequence})!=len(sequence):raise ValueError('duplicate timeline t')
  if len({row['leakage_group'] for row in sequence})!=1:raise ValueError('episode crosses leakage groups')
  for population in ('primary_common','full_timeline'):
   for step in STEPS:
    valid=[row for row in sequence if row[f'state_target_mask_tplus{step}'].lower() in {'true','1'}
           and (population=='full_timeline' or row['common_population'].lower() in {'true','1'})]
    if not valid:continue
    pred=[tuple(float(row[f'predicted_state_force_tplus{step}_{axis}']) for axis in AXES) for row in valid]
    target=[tuple(float(row[f'target_state_force_tplus{step}_{axis}']) for axis in AXES) for row in valid]
    current=[tuple(float(row[f'predicted_force_{axis}']) for axis in AXES) for row in valid]
    metrics={'prediction_temporal_variance_xyz':variance_xyz(pred),'target_temporal_variance_xyz':variance_xyz(target),
             'persistence_temporal_variance_xyz':variance_xyz(current),
             'prediction_target_mse_xyz':mean_metric(pred,target),'prediction_target_mae_xyz':mean_metric(pred,target,True),
             'prediction_vs_current_mse_xyz':mean_metric(pred,current),'prediction_vs_current_mae_xyz':mean_metric(pred,current,True),
             'target_vs_current_mse_xyz':mean_metric(target,current),'target_vs_current_mae_xyz':mean_metric(target,current,True)}
    if linear is not None:
     linear_pred=[linear[(row['episode_id'],int(row['t']),step)] for row in valid]
     metrics.update({'linear_temporal_variance_xyz':variance_xyz(linear_pred),
                     'linear_target_mse_xyz':mean_metric(linear_pred,target),
                     'linear_target_mae_xyz':mean_metric(linear_pred,target,True),
                     'linear_vs_current_mse_xyz':mean_metric(linear_pred,current),
                     'linear_vs_current_mae_xyz':mean_metric(linear_pred,current,True)})
    if not all(math.isfinite(v) for v in metrics.values()):raise ValueError('nonfinite diagnostic')
    result.append({'population':population,'seed':seed,'episode_id':episode,'leakage_group':valid[0]['leakage_group'],'step':step,
                   'n_observed_frames':len(valid),'first_t':int(valid[0]['t']),'last_t':int(valid[-1]['t']),**metrics})
 return result

def summarize(rows:list[dict[str,Any]])->list[dict[str,Any]]:
 grouped=defaultdict(list)
 for row in rows:grouped[(row['population'],row['seed'],row['step'])].append(row)
 expected={(population,seed,step) for population in ('primary_common','full_timeline') for seed in SEEDS for step in STEPS}
 if set(grouped)!=expected:raise ValueError('seed/step diagnostic grid incomplete')
 metrics=('prediction_temporal_variance_xyz','target_temporal_variance_xyz','prediction_target_mse_xyz',
          'persistence_temporal_variance_xyz','prediction_target_mae_xyz','prediction_vs_current_mse_xyz',
          'prediction_vs_current_mae_xyz','target_vs_current_mse_xyz','target_vs_current_mae_xyz',
          'linear_temporal_variance_xyz','linear_target_mse_xyz','linear_target_mae_xyz',
          'linear_vs_current_mse_xyz','linear_vs_current_mae_xyz')
 result=[]
 for (population,seed,step),subset in sorted(grouped.items()):
  item={'population':population,'seed':seed,'step':step,'trials':len(subset),'observed_frames':sum(r['n_observed_frames'] for r in subset),
        'aggregation':'equal-trial descriptive mean/median; temporal variance computed within each trial'}
  for metric in metrics:
   values=[r[metric] for r in subset];item[metric+'_trial_mean']=statistics.mean(values);item[metric+'_trial_median']=statistics.median(values)
  result.append(item)
 return result

def validated_linear_source(summary_path:Path,model_path:Path)->dict[str,str]:
 summary=json.loads(summary_path.read_text())
 if summary.get('format')!='round8_formal_evaluation_v1' or summary.get('status')!='complete' or summary.get('test_role_consumed') is not False:
  raise ValueError('complete formal evaluation summary required')
 expected=summary_path.parent/'state_linear_baseline.json'
 if model_path.resolve()!=expected.resolve() or summary.get('output_hashes',{}).get('state_linear_baseline.json')!=sha256(model_path):
  raise ValueError('state linear model is not accepted by formal evaluation summary')
 return {str(summary_path.resolve()):sha256(summary_path),str(model_path.resolve()):sha256(model_path)}

def load_linear_predictions(manifest:dict[str,Any],summary_path:Path,model_path:Path)->tuple[dict[tuple[str,int,int],tuple[float,float,float]],dict[str,str]]:
 prepared_spec=manifest.get('prepared')
 if not prepared_spec:raise ValueError('prediction manifest has no prepared binding')
 prepared_path=Path(prepared_spec['path'])
 if sha256(prepared_path)!=prepared_spec['sha256']:raise ValueError('prepared payload SHA mismatch')
 sources=validated_linear_source(summary_path,model_path)
 model=json.loads(model_path.read_text())
 if model.get('schema')!='round8_fit_only_state_linear_v1' or model.get('fit_role')!='fit_train':raise ValueError('invalid fit-only state linear model')
 outer=torch.load(prepared_path,map_location='cpu',weights_only=False)['timelines']['outer']
 history=torch.as_tensor(outer['signed_force_xyz']).detach().cpu().double()
 if tuple(history.shape[1:])!=(9,3):raise ValueError('invalid prepared signed-force history')
 x=history.reshape(len(history),-1).numpy();mean=np.asarray(model['input_mean'],dtype=float);std=np.asarray(model['input_std'],dtype=float)
 design=np.c_[np.ones(len(x)),(x-mean)/std]
 steps=sorted(model['steps'],key=lambda item:int(item['step']))
 if [int(item['step']) for item in steps]!=list(STEPS):raise ValueError('linear model step grid incomplete')
 predictions=np.stack([design@np.asarray(item['weights'],dtype=float) for item in steps],axis=1)
 if predictions.shape!=(len(history),5,3) or not np.isfinite(predictions).all():raise ValueError('invalid linear predictions')
 result={}
 for index,(episode,t) in enumerate(zip(outer['episode_id'],outer['t'].tolist())):
  for step in STEPS:result[(str(episode),int(t),step)]=tuple(float(v) for v in predictions[index,step-1])
 return result,{str(prepared_path.resolve()):prepared_spec['sha256'],**sources}

def main()->None:
 p=argparse.ArgumentParser();p.add_argument('--prediction-manifest',type=Path,required=True);p.add_argument('--evaluation-summary',type=Path,required=True);p.add_argument('--state-linear-model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 manifest,sources=accepted_manifest(a.prediction_manifest)
 artifacts=[x for x in manifest['artifacts'] if x['group']=='P4_state' and x['role']=='outer']
 linear,linear_sources=load_linear_predictions(manifest,a.evaluation_summary,a.state_linear_model)
 all_rows=[];sources.update(linear_sources)
 for artifact in sorted(artifacts,key=lambda x:int(x['seed'])):
  path=Path(artifact['path'])
  if sha256(path)!=artifact['sha256']:raise ValueError('timeline SHA mismatch')
  with path.open(newline='') as f:rows=list(csv.DictReader(f))
  if len(rows)!=int(artifact['rows']):raise ValueError('timeline row count mismatch')
  identities={(row['episode_id'],int(row['t']),step) for row in rows for step in STEPS}
  if identities!=set(linear):raise ValueError('prepared/timeline identity drift')
  all_rows.extend(analyze_rows(rows,int(artifact['seed']),linear));sources[str(path.resolve())]=artifact['sha256']
 summary=summarize(all_rows);a.output.mkdir(parents=True,exist_ok=True)
 atomic_csv(a.output/'P4_STATE_PER_TRIAL_TEMPORAL_DIAGNOSTICS.csv',all_rows)
 atomic_csv(a.output/'P4_STATE_SEED_STEP_SUMMARY.csv',summary)
 outputs=[a.output/'P4_STATE_PER_TRIAL_TEMPORAL_DIAGNOSTICS.csv',a.output/'P4_STATE_SEED_STEP_SUMMARY.csv']
 result={'schema':'round8_p4_state_temporal_diagnostic_v1','status':'complete','seeds':list(SEEDS),'steps':list(STEPS),
         'source_hashes':sources,'source_sha256':sha256(Path(__file__)),'reproduce_sha256':sha256(Path(__file__).with_name('REPRODUCE.md')),
         'output_hashes':{p.name:sha256(p) for p in outputs},'test_role_consumed':False,
         'populations':{'primary_common':'common stable/risk endpoint population','full_timeline':'all state-observed rows; may include already-slipping frames'},
         'interpretation':'continuous descriptive diagnostic; temporal variance is computed within each trial; fit-linear parameters are read from the fit_train-only formal model; no low-variance threshold, model selection, calibration, refit, or outer tuning',
         'boundary':'target and current are frozen predicted-force representations, not physical force ground truth; P4_direct is excluded because its 15D state has no state supervision'}
 atomic_json(a.output/'STATE_DIAGNOSTIC_AUDIT.json',result);print(json.dumps({'status':'complete','trial_step_rows':len(all_rows),'summary_rows':len(summary)}))
if __name__=='__main__':main()
