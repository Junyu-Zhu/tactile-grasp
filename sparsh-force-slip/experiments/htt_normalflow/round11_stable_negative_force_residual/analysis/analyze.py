#!/usr/bin/env python3
"""Fixed-threshold force interventions and additive residual diagnostics."""
import argparse,importlib.util,json
from pathlib import Path
import numpy as np
import torch
from diagnostics import correctness,residual_stats,sequence_bin
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
tr=load('r11_analysis_train',ROOT/'training/train.py');ev=load('r11_analysis_eval',ROOT/'evaluation/evaluate.py')

def predict(model,x,device,components=False):
 outs=[]
 with torch.inference_mode():
  for i in range(0,len(x),256):
   z=x[i:i+256].to(device)
   if components:outs.append({k:v.detach().cpu().reshape(-1) for k,v in model.components(z).items()})
   else:outs.append(torch.sigmoid(model(z)).cpu().reshape(-1))
 return {k:torch.cat([o[k] for o in outs]) for k in outs[0]} if components else torch.cat(outs)

def main():
 p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 protocol=json.loads((HERE/'PROTOCOL.json').read_text());audit=json.loads(a.audit.read_text());groups=protocol['groups'];grid={(g,f'htt_leave_p{f}',s) for g in groups for f in range(1,5) for s in protocol['seeds']}
 assert audit['status']=='pass' and audit['schema']=='round11_fusion_training_audit_v1' and audit['test_role_consumed'] is False and audit['expected_count']==36
 assert len(audit['accepted_runs'])==36 and {(r['group'],r['fold'],r['seed']) for r in audit['accepted_runs']}==grid
 tr.configure(protocol['seeds'][0]);metrics=[];trials=[];residuals=[];corrections=[];artifacts=[];parity=[];sources={str(a.audit):tr.sha(a.audit),str(HERE/'PROTOCOL.json'):tr.sha(HERE/'PROTOCOL.json'),str(Path(__file__)):tr.sha(__file__),str(HERE/'diagnostics.py'):tr.sha(HERE/'diagnostics.py')}
 for rec in audit['accepted_runs']:
  cp=ev.verify(rec['checkpoint']);ck=torch.load(cp,map_location='cpu',weights_only=False);cfg=ck['config'];assert not cfg['smoke'] and all(cfg[k]==rec[k] for k in ('group','fold','seed'))
  for path,h in cfg['sources'].items():assert tr.sha(path)==h;sources[path]=h
  dp=ev.verify(cfg['prepared']);data=torch.load(dp,map_location='cpu',weights_only=False);ap=ev.verify(cfg['prepared_audit']);da=json.loads(ap.read_text());tr.validate_data(data,da,dp,rec['fold'],rec['seed']);sources[str(cp)]=rec['checkpoint']['sha256'];sources[str(dp)]=cfg['prepared']['sha256'];sources[str(ap)]=cfg['prepared_audit']['sha256']
  model=tr.TemporalHead(rec['group']).to(a.device);model.load_state_dict(ck['model_state']);model.eval().requires_grad_(False);before=tr.state_hash(model.state_dict());norm=ck['normalizer'];role=data['roles']['validation'];raw=role['x'];x=tr.normalize(raw,norm);calrole=data['roles']['calibration'];calx=tr.normalize(calrole['x'],norm)
  def rows(prob,rr):return [dict(episode=e,t=int(t),stage=int(s),score=float(p),group=g) for e,t,s,p,g in zip(rr['episode_id'],rr['t'],rr['stage'],prob,rr['leakage_group'])]
  cr=rows(predict(model,calx,a.device),calrole);original=predict(model,x,a.device);vr=rows(original,role)
  for name,generated,rr in [('calibration',cr,calrole),('validation',vr,role)]:
   ep=dp.with_name(f'endpoints_{name}.csv');accepted=ev.load_rows(rec['predictions'][name],{'path':str(ep),'sha256':da['output_hashes'][str(ep)]},name,fold=rec['fold']);assert [(r['episode'],r['t'],r['stage'],r['group']) for r in accepted]==[(r['episode'],r['t'],r['stage'],r['group']) for r in generated];error=max(abs(r['score']-s['score']) for r,s in zip(accepted,generated));assert error<=protocol['parity_atol'];parity.append({k:rec[k] for k in ('group','fold','seed')}|dict(role=name,max_abs=error));sources[rec['predictions'][name]['path']]=rec['predictions'][name]['sha256']
  points={k:v for k,v in ev.choose(cr).items() if k in protocol['thresholds']};zero=x.clone();zero[:,:,192:198]=0;lag=raw.clone();lag[:,1:,192:195]=raw[:,:-1,192:195];lag[:,5:,195:198]=lag[:,5:,192:195]-lag[:,:4,192:195];lag=tr.normalize(lag,norm)
  variants=[('unperturbed',original,points),('force_fit_mean_zero',predict(model,zero,a.device),points),('force_causal_lag1',predict(model,lag,a.device),points)]
  if rec['group']=='F_residual_balanced':
   c=predict(model,x,a.device,True);cc=predict(model,calx,a.device,True);assert set(c)=={'base_logit','residual_logit','logit'} and torch.allclose(c['logit'],c['base_logit']+c['residual_logit'],atol=1e-6);assert torch.allclose(torch.sigmoid(c['logit']),original,atol=1e-6)
   base=torch.sigmoid(c['base_logit']);basepoints={k:v for k,v in ev.choose(rows(torch.sigmoid(cc['base_logit']),calrole)).items() if k in points};variants += [('residual_off_full_threshold',base,points),('base_only_independent_calibration',base,basepoints)]
   threshold=points['FPR0.05'];changes=correctness(role['stage'],base,original,threshold);epinfo={e['episode_id']:e for e in data['episodes']};bound=float(model.residual_bound);assert bound==protocol['residual_bound']
   for eid in sorted(set(role['episode_id'])):
    ix=np.array([i for i,e in enumerate(role['episode_id']) if e==eid]);ep=epinfo[eid];meta={k:rec[k] for k in ('group','fold','seed')}|dict(episode=eid,probe=str(ep.get('probe',role['probe'][int(ix[0])])),leakage_group=role['leakage_group'][int(ix[0])]);tt=ev.trial_metrics([vr[i] for i in ix],threshold,1)[0]
    corrections.append({**meta,**{k:int(v[ix].sum()) for k,v in changes.items()},'static_fpr':ev.divide(tt['fp'],tt['fp']+tt['tn']),'static_alarming_frames':tt['fp'],'threshold':threshold})
    q=np.array([sequence_bin(int(role['t'][i]),len(ep['stage'])) for i in ix]);stage=np.array(role['stage'][ix]);vals=np.array(c['residual_logit'][ix])
    for st in ('all',0,1,2):
     for quartile in ('all',0,1,2,3):
      mask=np.ones(len(ix),bool)
      if st!='all':mask&=stage==st
      if quartile!='all':mask&=q==quartile
      residuals.append({**meta,'stage':st,'sequence_quartile':quartile,**residual_stats(vals[mask],bound)})
   path=a.output/'predictions'/f"{rec['group']}_{rec['fold']}_{rec['seed']}_components.csv";ev.csvout(path,[dict(episode_id=e,t=int(t),stage=int(s),base_probability=float(b),full_probability=float(f),residual_logit=float(r),role='validation') for e,t,s,b,f,r in zip(role['episode_id'],role['t'],role['stage'],base,original,c['residual_logit'])]);artifacts.append({'path':str(path),'sha256':tr.sha(path)})
  for name,prob,working in variants:
   if rec['group']=='V_balanced':assert torch.equal(prob,original)
   rr=rows(prob,role);meta={k:rec[k] for k in ('group','fold','seed')}|dict(intervention=name);rank=ev.rank_metrics(rr)
   path=a.output/'predictions'/f"{rec['group']}_{rec['fold']}_{rec['seed']}_{name}.csv";ev.csvout(path,[dict(episode_id=r['episode'],t=r['t'],stage=r['stage'],p_slip=r['score'],role='validation') for r in rr]);artifacts.append({'path':str(path),'sha256':tr.sha(path)})
   for point,threshold in working.items():
    for rule,k in ev.RULES.items():
     tt=ev.trial_metrics(rr,threshold,k);metrics.append({**meta,'point':point,'rule':rule,'threshold':threshold,'calibration_source':'base_only' if name=='base_only_independent_calibration' else 'complete_unperturbed','never_alarm':threshold>1,**ev.aggregate(tt),**rank});trials.extend({**t,**meta,'leakage_group':t['group'],'point':point,'rule':rule} for t in tt)
  assert tr.state_hash(model.state_dict())==before;print(rec['group'],rec['fold'],rec['seed'],'pass',flush=True)
 for name,rr in [('metrics',metrics),('per_trial',trials),('residual_distribution',residuals),('correction_harm',corrections)]:ev.csvout(a.output/(name+'.csv'),rr)
 cases=[]
 for kind,key in [('correction','corrected'),('harm','harmed'),('persistent_static','static_fpr')]:
  rr=[r for r in corrections if np.isfinite(r[key])];rr=sorted(rr,key=lambda r:(-r[key],-r['static_alarming_frames'],r['fold'],r['seed'],r['episode']))
  cases.extend(dict(kind=kind,rank=i,selected=i==0,**r) for i,r in enumerate(rr))
 ev.csvout(a.output/'representative_cases.csv',cases);ev.js(a.output/'AUDIT.json',dict(status='pass',runs=36,all_parameters_frozen=True,V_force_invariance=True,parity=parity,sources=sources,prediction_artifacts=artifacts,outputs={f.name:tr.sha(f) for f in a.output.glob('*.csv')},no_validation_recalibration=True))
if __name__=='__main__':main()
