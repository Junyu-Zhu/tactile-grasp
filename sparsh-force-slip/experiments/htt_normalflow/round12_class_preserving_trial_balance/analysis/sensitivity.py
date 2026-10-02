#!/usr/bin/env python3
"""Frozen current heads, original thresholds, two preregistered force interventions."""
import argparse,importlib.util,json
from pathlib import Path
import torch,numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
tr=load('r9_sens_train',ROOT/'training/train.py');ev=load('r9_sens_eval',ROOT/'evaluation/evaluate.py')

def run(a):
 tr.configure(20260914);audit=json.loads(a.audit.read_text());runs=audit['accepted_runs'];groups=('V_class_trial_balanced','F_class_trial_balanced');grid={(g,f'htt_leave_p{f}',s) for g in groups for f in range(1,5) for s in (20260914,20260915,20260916)}
 if audit.get('schema')!='round12_fusion_training_audit_v1' or audit.get('status')!='pass' or audit.get('test_role_consumed') is not False or audit.get('expected_count')!=24 or len(runs)!=24 or {(r['group'],r['fold'],r['seed']) for r in runs}!=grid:raise ValueError('accepted 24 required')
 out=a.output;out.mkdir(parents=True,exist_ok=True);protocol=json.loads((HERE/'PROTOCOL.json').read_text())
 all_metrics=[];all_trials=[];artifacts=[];parity=[];sources={str(a.audit):tr.sha(a.audit),str(HERE/'PROTOCOL.json'):tr.sha(HERE/'PROTOCOL.json'),str(Path(__file__).resolve()):tr.sha(__file__),str(ROOT/'training/train.py'):tr.sha(ROOT/'training/train.py'),str(ROOT/'evaluation/evaluate.py'):tr.sha(ROOT/'evaluation/evaluate.py')}
 for rec in runs:
  cp=ev.verify(rec['checkpoint']);ck=torch.load(cp,map_location='cpu',weights_only=False);config=ck['config'];data_path=ev.verify(config['prepared']);data=torch.load(data_path,map_location='cpu',weights_only=False);role=data['roles']['validation'];raw=role['x'];norm=ck['normalizer'];x=tr.normalize(raw,norm)
  if config['smoke'] or any(config[k]!=rec[k] for k in ('group','fold','seed')):raise ValueError('checkpoint run identity')
  for path,h in config['sources'].items():
   if tr.sha(path)!=h:raise ValueError('checkpoint source changed')
   sources[path]=h
  prepared_audit_path=ev.verify(config['prepared_audit']);prepared_audit=json.loads(prepared_audit_path.read_text());tr.validate_data(data,prepared_audit,data_path,rec['fold'],rec['seed'])
  sources[str(prepared_audit_path)]=config['prepared_audit']['sha256']
  sources[str(cp)]=rec['checkpoint']['sha256'];sources[str(data_path)]=config['prepared']['sha256']
  model=tr.TemporalHead(rec['group']).to(a.device);model.load_state_dict(ck['model_state']);model.eval().requires_grad_(False);before=tr.state_hash(model.state_dict())
  accepted=ev.readcsv(ev.verify(rec['predictions']['validation']));calrec=rec['predictions']['calibration'];calpath=str(data_path.with_name('endpoints_calibration.csv'));calep={'path':calpath,'sha256':prepared_audit['output_hashes'][calpath]}
  identity=[(r['episode_id'],int(r['t']),int(r['stage']),r['role']) for r in accepted]
  if identity!=[(e,int(t),int(s),'validation') for e,t,s in zip(role['episode_id'],role['t'],role['stage'])]:raise ValueError('validation row identity')
  cal=ev.load_rows(calrec,calep,'calibration',fold=rec['fold']);points=ev.choose(cal)
  for r in rec['predictions'].values():sources[r['path']]=r['sha256']
  original=tr.infer(model,x,a.device);recorded=torch.tensor([float(r['p_slip']) for r in accepted]);err=float((original-recorded).abs().max())
  if err>protocol['inference_parity_atol']:raise ValueError('original prediction parity')
  parity.append({k:rec[k] for k in ('group','fold','seed')}|{'max_abs':err})
  zero=x.clone();zero[:,:,192:198]=0
  lag=raw.clone();lag[:,1:,192:195]=raw[:,:-1,192:195];lag[:,5:,195:198]=lag[:,5:,192:195]-lag[:,:4,192:195];lag=tr.normalize(lag,norm)
  for name,z in [('unperturbed',x),('force_fit_mean_zero',zero),('force_causal_lag1',lag)]:
   pred=original if name=='unperturbed' else tr.infer(model,z,a.device)
   if rec['group']=='V_class_trial_balanced' and not torch.equal(pred,original):raise ValueError('V force invariance')
   rows=[{'episode':e,'t':int(t),'stage':int(s),'score':float(p),'group':g} for e,t,s,p,g in zip(role['episode_id'],role['t'],role['stage'],pred,role['leakage_group'])]
   rank=ev.rank_metrics(rows);meta={k:rec[k] for k in ('group','fold','seed')};meta['intervention']=name
   for point,threshold in points.items():
    for rule,k in ev.RULES.items():
     trials=ev.trial_metrics(rows,threshold,k);all_metrics.append({**meta,'point':point,'rule':rule,'threshold':threshold,'never_alarm':threshold>1,**ev.aggregate(trials),**rank})
     all_trials.extend({**row,**meta,'leakage_group':row['group'],'point':point,'rule':rule} for row in trials)
   path=out/'predictions'/f"{rec['group']}_{rec['fold']}_{rec['seed']}_{name}.csv";ev.csvout(path,[{'episode_id':r['episode'],'t':r['t'],'stage':r['stage'],'p_slip':r['score'],'role':'validation'} for r in rows]);artifacts.append({'path':str(path),'sha256':tr.sha(path)})
  if before!=tr.state_hash(model.state_dict()):raise ValueError('model mutated')
  print(rec['group']+' '+rec['fold']+' '+str(rec['seed'])+' pass',flush=True)
 ev.csvout(out/'metrics.csv',all_metrics);ev.csvout(out/'per_trial.csv',all_trials)
 ev.js(out/'AUDIT.json',{'status':'pass','runs':24,'perturbed_predictions':48,'metrics':len(all_metrics),'all_model_states_unchanged':True,'V_force_invariance':True,'original_prediction_parity':parity,'sources':sources,'protocol_sha256':tr.sha(HERE/'PROTOCOL.json'),'output_hashes':{'metrics.csv':tr.sha(out/'metrics.csv'),'per_trial.csv':tr.sha(out/'per_trial.csv')},'prediction_artifacts':artifacts,'no_recalibration':True})
 print(json.dumps({'status':'pass','metrics':len(all_metrics)}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');run(p.parse_args())
