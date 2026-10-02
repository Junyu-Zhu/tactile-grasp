#!/usr/bin/env python3
"""Independent final scoring of nine new GRUs; no fitting outside fit/cal roles."""
import argparse,csv,hashlib,importlib.util,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

ROOT=Path(__file__).resolve().parent

def loadmodule(path):
 s=importlib.util.spec_from_file_location('r6_trainer_eval',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def save(p,d):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');q.replace(p)
def csvsave(p,rows):
 if not rows:return
 q=p.with_suffix('.tmp')
 with q.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 q.replace(p)

def metrics(y,p,t,w=None):
 y=np.asarray(y,int);p=np.asarray(p,float);w=np.ones(len(y)) if w is None else np.asarray(w,float);a=p>=t
 tp=float(w[(y==1)&a].sum());fn=float(w[(y==1)&~a].sum());fp=float(w[(y==0)&a].sum());tn=float(w[(y==0)&~a].sum())
 if tp+fn==0 or tn+fp==0:return None
 return {'AP':float(average_precision_score(y,p,sample_weight=w)),'Brier':float(np.average((p-y)**2,weights=w)),'BA':.5*(tp/(tp+fn)+tn/(tn+fp)),'macro_F1':.5*(2*tp/max(1e-30,2*tp+fp+fn)+2*tn/max(1e-30,2*tn+fp+fn)),'FPR':fp/(fp+tn),'recall':tp/(tp+fn),'tn':tn,'fp':fp,'fn':fn,'tp':tp,'prevalence':(tp+fn)/w.sum(),'never_alarm':bool(tp+fp==0)}

def thresholds(y,p):
 y=np.asarray(y,int);p=np.asarray(p,float)
 if set(y)!={0,1}:raise ValueError('Calibration must have both classes')
 order=np.argsort(-p,kind='stable');ys=y[order];ps=p[order];end=np.r_[np.where(ps[:-1]!=ps[1:])[0],len(ps)-1]
 tp=np.r_[0,np.cumsum(ys)[end]];fp=np.r_[0,(end+1)-np.cumsum(ys)[end]];ts=np.r_[np.nextafter(1.,np.inf),ps[end]];rec=tp/y.sum();fpr=fp/(len(y)-y.sum());ba=(rec+1-fpr)/2
 best=max(range(len(ts)),key=lambda i:(ba[i],-fpr[i],ts[i]));out={'fixed_0.5':.5,'calibration_maxBA':float(ts[best])}
 for limit in [.01,.05,.1]:
  ix=max((i for i in range(len(ts)) if fpr[i]<=limit+1e-15),key=lambda i:(rec[i],-fpr[i],ts[i]));out[f'calibration_FPR_{limit:.2f}']=float(ts[ix])
 return out

def starts(ts,alarm):
 ts=np.asarray(ts,int);alarm=np.asarray(alarm,bool)
 return int(np.sum(alarm & np.r_[True,(~alarm[:-1])|(np.diff(ts)!=1)]))

def trial_rows(role,p,t,method,point):
 out=[];ep=np.asarray(role['episode_id']);ts=np.asarray(role['t']);y=np.asarray(role['y']);ons=role['first_current_slip_t']
 for e in sorted(set(ep)):
  ix=np.where(ep==e)[0];ix=ix[np.argsort(ts[ix])];a=p[ix]>=t;neg=y[ix]==0;pos=y[ix]==1;early=ts[ix][a&pos]
  out.append({'method':method,'point':point,'episode_id':str(e),'frames':len(ix),'positive_frames':int(pos.sum()),'negative_frames':int(neg.sum()),'false_alarm_frames':int((a&neg).sum()),'false_alarm_starts':starts(ts[ix],a&neg),'any_false_alarm':bool((a&neg).any()),'eligible_positive_event':bool(pos.any()),'event_detected':bool(len(early)) if pos.any() else None,'first_in_window_alarm':int(early[0]) if len(early) else None,'lead_frames':int(ons[ix[0]]-early[0]) if len(early) else None,'missed_pre_onset':bool(not len(early)) if pos.any() else None,'late_status':'not_observable_from_pre_onset_only_population'})
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--inventory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 data=torch.load(a.data,map_location='cpu',weights_only=False);inv=json.loads(a.inventory.read_text());tr=loadmodule(Path(inv['trainer']));roles=data['roles'];mean=data['normalization_C']['mean'];std=data['normalization_C']['std'];pred={};source={str(a.data):sha(a.data),str(a.inventory):sha(a.inventory),str(Path(inv['trainer'])):sha(Path(inv['trainer']))};fit=roles['fit_train'];cal=roles['calibration'];outer=roles['outer'];y=np.asarray(outer['y'],int);yc=np.asarray(cal['y'],int)
 expected={(g,s) for g in ['A_visual','B_force','C_force_delta'] for s in [20260914,20260915,20260916]}
 if len(inv['runs'])!=9 or {(x['group'],x['seed']) for x in inv['runs']}!=expected or len({x['output'] for x in inv['runs']})!=9 or any(x['id']!=f"future_{x['group']}_{x['seed']}" for x in inv['runs']):raise ValueError('Nine exact identities required')
 for raw,digest in inv['frozen_inputs'].items():
  if sha(raw)!=digest:raise ValueError('Changed frozen input '+raw)
  source[raw]=digest
 decision=json.loads(Path(inv['support_decision']).read_text());prep=json.loads(Path(inv['prepare_audit']).read_text());queue=json.loads((Path(inv['output_root'])/'QUEUE_STATE.json').read_text())
 if decision['status']!='complete' or not decision['training_triggered'] or prep['status']!='complete' or prep['output']['sha256']!=source[str(a.data)] or queue['status']!='complete' or set(queue['jobs'])!={x['id'] for x in inv['runs']} or any(v!='complete' for v in queue['jobs'].values()):raise ValueError('Unaccepted parent or queue')
 source[str(Path(__file__).resolve())]=sha(Path(__file__).resolve())
 for run in inv['runs']:
  folder=Path(run['output']);summary=json.loads((folder/'summary.json').read_text());ckpath=folder/'best.pth'
  if summary['status']!='complete' or not summary['formal'] or summary['group']!=run['group'] or summary['seed']!=run['seed']:raise ValueError('Unaccepted run')
  for artifact in ['best','latest']:
   item=summary['artifacts'][artifact]
   if sha(item['path'])!=item['sha256']:raise ValueError('Checkpoint changed')
   source[item['path']]=item['sha256']
  receipt_path=Path(inv['output_root'])/f"{run['id']}.receipt.attempt{queue['attempts'][run['id']]}.json";receipt=json.loads(receipt_path.read_text())
  if receipt['status']!='complete' or receipt['inventory_sha256']!=source[str(a.inventory)] or receipt['id']!=run['id']:raise ValueError('Missing/mismatched queue receipt')
  source[str(receipt_path)]=sha(receipt_path)
  cfg=summary['run_config']
  if cfg['mode']!='formal' or cfg['horizon']!=1 or cfg['group']!=run['group'] or cfg['seed']!=run['seed']:raise ValueError('Wrong run config')
  ck=torch.load(ckpath,map_location='cpu',weights_only=False)
  if ck['run_config']['data_sha256']!=source[str(a.data)] or ck['run_config']['code_bundle_sha256']!=tr.code_bundle_sha():raise ValueError('Training data or executed source identity drift')
  if ck['run_identity_sha256']!=summary['run_identity_sha256'] or ck['run_config']!=summary['run_config']:raise ValueError('Checkpoint/config mismatch')
  dim=data['group_input_dims'][run['group']];model=tr.GRURisk(dim,128);model.load_state_dict(ck['model_state'],strict=True);model.eval();raw={}
  for role in ['calibration','outer']:
   x=(roles[role]['x_C'].float()[:,:,:dim]-mean[:dim])/std[:dim]
   with torch.no_grad():raw[role]=torch.cat([model(chunk).sigmoid() for chunk in x.split(512)]).numpy()
   if not np.isfinite(raw[role]).all():raise ValueError('Nonfinite predictions')
  key=run['id'];pred[key]=raw;pred[key+'__gate']={role:raw[role]*roles[role]['x_C'][:,-1,768].numpy() for role in raw};source[str(ckpath)]=sha(ckpath)
 # Same causal information, fit-only simple rules. No trajectory total length.
 max_t=float(fit['t'].max());feat=lambda r:np.stack([np.asarray(r['t'])/max_t,(np.asarray(r['t'])/max_t)**2],1)
 pos=LogisticRegression(C=1,class_weight='balanced',solver='lbfgs',random_state=20260915,max_iter=1000).fit(feat(fit),np.asarray(fit['y']))
 def delta(r):return np.linalg.norm(r['x_C'][:,-1,772:775].numpy(),axis=1)
 df=delta(fit);center=float(np.median(df));scale=max(float(np.quantile(df,.75)-np.quantile(df,.25)),1e-6)
 for name in ['current_slip','history_mean4','history_trend','force_delta_rule','sequence_position']:pred[name]={}
 for name,r in [('calibration',cal),('outer',outer)]:
  hist=r['x_C'][:,:,768].numpy();pred['current_slip'][name]=hist[:,-1];pred['history_mean4'][name]=hist.mean(1);pred['history_trend'][name]=np.clip(2*hist[:,-1]-hist[:,0],0,1);pred['force_delta_rule'][name]=1/(1+np.exp(-np.clip((delta(r)-center)/scale,-60,60)));pred['sequence_position'][name]=pos.predict_proba(feat(r))[:,1]
 rows=[];trials=[];configs={};cis=[];groups=np.asarray(outer['leakage_group']);names=sorted(set(groups));groupidx=np.array([names.index(g) for g in groups]);rng=np.random.default_rng(20260915);weights=[np.bincount(rng.integers(0,len(names),len(names)),minlength=len(names))[groupidx] for _ in range(200)]
 for method,pr in pred.items():
  configs[method]=thresholds(yc,pr['calibration']);ap0=None
  for point,t in configs[method].items():
   m=metrics(y,pr['outer'],t);rows.append({'method':method,'point':point,'threshold':t,**m});trials+=trial_rows(outer,pr['outer'],t,method,point)
   values=defaultdict(list)
   for w in weights:
    z=metrics(y,pr['outer'],t,w)
    if z:
     for metric in ['AP','Brier','BA','macro_F1','FPR','recall']:values[metric].append(z[metric])
   for metric,v in values.items():cis.append({'method':method,'point':point,'metric':metric,'lower':float(np.quantile(v,.025)),'upper':float(np.quantile(v,.975)),'valid_draws':len(v),'requested_draws':200})
 paired=[]
 for candidate in ['B_force','C_force_delta']:
  for point in configs[inv['runs'][0]['id']]:
   byseed=[]
   for seed in [20260914,20260915,20260916]:
    ka=next(x['id'] for x in inv['runs'] if x['group']=='A_visual' and x['seed']==seed);kc=next(x['id'] for x in inv['runs'] if x['group']==candidate and x['seed']==seed);byseed.append((ka,kc))
   for metric in ['AP','Brier','BA','FPR','recall']:
    vals=[]
    for w in weights:
     diffs=[]
     for ka,kc in byseed:
      ma=metrics(y,pred[ka]['outer'],configs[ka][point],w);mc=metrics(y,pred[kc]['outer'],configs[kc][point],w)
      if ma and mc:diffs.append(mc[metric]-ma[metric])
     if len(diffs)==3:vals.append(float(np.mean(diffs)))
    observed=float(np.mean([metrics(y,pred[kc]['outer'],configs[kc][point])[metric]-metrics(y,pred[ka]['outer'],configs[ka][point])[metric] for ka,kc in byseed]));paired.append({'candidate':candidate,'baseline':'A_visual','point':point,'metric':metric,'mean_difference':observed,'lower':float(np.quantile(vals,.025)),'upper':float(np.quantile(vals,.975)),'valid_draws':len(vals),'shared_draw_all_three_seeds':True})
 csvsave(a.output/'metrics.csv',rows);csvsave(a.output/'trial_events.csv',trials);csvsave(a.output/'confidence_intervals.csv',cis);csvsave(a.output/'paired_vs_A.csv',paired)
 with (a.output/'predictions.tmp').open('wb') as stream:np.savez_compressed(stream,**{k:v['outer'] for k,v in pred.items()})
 (a.output/'predictions.tmp').replace(a.output/'predictions.npz')
 support=json.loads(Path(data['provenance']['source_manifest']['path']).read_text());alltr=support['horizons']['1']['roles']['outer']['trials'];censored=[{'episode_id':x['episode_id'],'first_current_slip_t':x['first_current_slip_t'],'positive_windows':len(x['positive_endpoints']),'negative_windows':len(x['negative_endpoints']),'status':'eligible_event' if x['positive_endpoints'] else ('no_observable_pre_onset_positive' if x['first_current_slip_t'] is not None else 'no_observed_onset')} for x in alltr];csvsave(a.output/'event_support_and_censoring.csv',censored)
 summary={'all_outer_trials':len(alltr),'outer_trials_with_onset':sum(x['first_current_slip_t'] is not None for x in alltr),'outer_eligible_onset_trials':sum(bool(x['positive_endpoints']) for x in alltr),'status':'complete','neural_runs':9,'horizon':1,'evaluation_frames':len(y),'positive_frames':int(y.sum()),'evaluation_eligible_trials':len(names),'methods':list(pred),'source_hashes':source,'thresholds':configs,'simple_rule_fit':{'position_train_max_t':max_t,'delta_train_median':center,'delta_train_iqr':scale},'limitations':['Outer remains historical development validation; legacy upstream exposure and checkpoint selection remain limitations.','All scored inputs precede first dataset-label onset; physical onset measurement is not independently established.','H1 lead=1 is fixed by window definition, not independently learned lead duration.','Late alarms are not observable from the pre-onset-only population.','Manual-score Brier is descriptive error, not proof of probability calibration.'],'output_hashes':{q.name:sha(q) for q in a.output.iterdir() if q.is_file() and q.name!='summary.json'}}
 save(a.output/'summary.json',summary);print(json.dumps({'status':'complete','metric_rows':len(rows),'trial_rows':len(trials)}))
if __name__=='__main__':main()
