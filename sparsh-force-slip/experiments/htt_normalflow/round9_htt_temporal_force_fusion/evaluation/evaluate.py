#!/usr/bin/env python3
"""Fixed-protocol, group-paired HTT current detection evaluation."""
import argparse,csv,hashlib,json,math,os
from pathlib import Path
import numpy as np

GROUPS=('V_temporal','F_history','F_delta'); SEEDS=(20260914,20260915,20260916)
PAIRS=(('F_history','V_temporal'),('F_delta','F_history'))
RULES={'raw':1,'confirm2':2}

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def verify(rec):
 p=Path(rec['path'])
 if not p.is_file() or sha(p)!=rec['sha256']:raise ValueError(f'Hash mismatch {p}')
 return p
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.generic):x=x.item()
 if isinstance(x,float) and not math.isfinite(x):return None
 return x
def js(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(clean(x),indent=2,ensure_ascii=False));os.replace(q,p)
def csvout(p,rows):
 if not rows:return
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(clean(rows))
def divide(a,b):return a/b if b else float('nan')
def ratio_metrics(v):
 tn,fp,fn,tp=v[:4];n=tn+fp+fn+tp
 return dict(static_fpr=divide(fp,tn+fp),gross_recall=divide(tp,tp+fn),balanced_accuracy=(divide(tn,tn+fp)+divide(tp,tp+fn))/2,macro_f1=(divide(2*tn,2*tn+fp+fn) if 2*tn+fp+fn else 0)/2+(divide(2*tp,2*tp+fp+fn) if 2*tp+fp+fn else 0)/2,positive_prevalence=divide(tp+fn,n),tn=tn,fp=fp,fn=fn,tp=tp)

def readcsv(path):
 with Path(path).open() as stream:return list(csv.DictReader(stream))

def load_rows(pred,endpoint,role,historical=False,fold=None):
 expected={}
 for r in readcsv(verify(endpoint)):
  key=(r['episode_id'],int(r['t']))
  if key in expected or r.get('role')!=role or (fold is not None and 'fold' in r and r['fold']!=fold):raise ValueError('Duplicate endpoint or wrong role/fold')
  expected[key]=(int(r['stage']),r['leakage_group'])
 episode_groups={}
 for (episode,t),(stage,group) in expected.items():
  if episode in episode_groups and episode_groups[episode]!=group:raise ValueError('Episode has inconsistent leakage group')
  episode_groups[episode]=group
 rows=[];seen=set()
 for r in readcsv(verify(pred)):
  key=(r.get('episode_id',r.get('episode')),int(r.get('t',r.get('frame')) if historical else r['t']))
  if key in seen:raise ValueError('Duplicate prediction')
  seen.add(key)
  if key not in expected:
   if historical:continue
   raise ValueError('Unexpected prediction endpoint')
  stage,group=expected[key];score=float(r.get('p_slip',r.get('probability',r.get('score','nan'))))
  if int(r['stage'])!=stage or stage not in (0,1,2) or not np.isfinite(score) or not 0<=score<=1 or (r.get('role',role) if historical else r.get('role'))!=role:raise ValueError('Prediction label/score/role mismatch')
  rows.append(dict(episode=key[0],t=key[1],stage=stage,score=score,group=group))
 if { (r['episode'],r['t']) for r in rows}!=set(expected) or not rows:raise ValueError('Missing prediction endpoints')
 return sorted(rows,key=lambda r:(r['episode'],r['t']))

def thresholds(rows):
 s=np.array([r['score'] for r in rows]);y=np.array([r['stage'] for r in rows]);order=np.argsort(-s,kind='stable');s=s[order];y=y[order]
 ends=np.r_[np.flatnonzero(np.diff(s)),len(s)-1];tp=np.r_[0,np.cumsum(y==2)[ends]];fp=np.r_[0,np.cumsum(y==0)[ends]];pos=(y==2).sum();neg=(y==0).sum()
 if not pos or not neg:raise ValueError('Threshold calibration requires both primary classes')
 th=np.r_[np.nextafter(1.,np.inf),s[ends]];fpr=fp/neg;rec=tp/pos;ba=(1-fpr+rec)/2
 return [dict(threshold=float(t),static_fpr=float(f),gross_recall=float(r),balanced_accuracy=float(b),precision=float(pp)) for t,f,r,b,pp in zip(th,fpr,rec,ba,np.divide(tp,tp+fp,out=np.ones(len(tp),float),where=(tp+fp)>0))]
def choose(rows):
 c=thresholds(rows);out={'fixed_0.5':.5,'maxBA':max(c,key=lambda x:(x['balanced_accuracy'],-x['static_fpr'],x['threshold']))['threshold']}
 for f in (.01,.05,.1):out[f'FPR{f:.2f}']=max((x for x in c if x['static_fpr']<=f+1e-15),key=lambda x:(x['gross_recall'],-x['static_fpr'],x['threshold']))['threshold']
 for r in (.8,.9,.95):out[f'recall{r:.2f}']=max((x for x in c if x['gross_recall']>=r-1e-15),key=lambda x:(-x['static_fpr'],x['gross_recall'],x['threshold']))['threshold']
 return out

def rank_metrics(rows,group_weights=None):
 y=np.array([r['stage'] for r in rows]);s=np.array([r['score'] for r in rows]);w=np.array([1 if group_weights is None else group_weights.get(r['group'],0) for r in rows],float);mask=(y!=1)&(w>0);y=y[mask]==2;s=s[mask];w=w[mask]
 p=w[y].sum();n=w[~y].sum()
 if not p or not n:return dict(AP=float('nan'),pAUC=float('nan'),brier=float('nan'))
 order=np.argsort(-s,kind='stable');y=y[order];s=s[order];w=w[order];ends=np.r_[np.flatnonzero(np.diff(s)),len(s)-1];tp=np.cumsum(w*y)[ends];fp=np.cumsum(w*(~y))[ends];rec=tp/p;fpr=np.r_[0,fp/n];tpr=np.r_[0,rec]
 stop=np.searchsorted(fpr,.1,side='right');xx=np.r_[fpr[:stop],.1];yy=np.r_[tpr[:stop],np.interp(.1,fpr,tpr)]
 return dict(AP=float(np.sum(np.diff(np.r_[0,rec])*tp/(tp+fp))),pAUC=float(np.sum(np.diff(xx)*(yy[1:]+yy[:-1])/2)/.1),brier=float(np.sum(w*(s-y)**2)/w.sum()))

def trial_metrics(rows,threshold,k):
 result=[]
 for ep in sorted({r['episode'] for r in rows}):
  rr=[r for r in rows if r['episode']==ep];st=np.array([r['stage'] for r in rr]);alarm=np.zeros(len(rr),bool);starts=np.zeros(len(rr),bool);active=False;count=0;breaks=np.array([i==0 or r['t']!=rr[i-1]['t']+1 for i,r in enumerate(rr)])
  for i,r in enumerate(rr):
   if breaks[i]:active=False;count=0
   if active:
    if r['score']<threshold:active=False;count=0
   else:
    count=count+1 if r['score']>=threshold else 0
    if count>=k:active=True;starts[i]=True
   alarm[i]=active
  tp=int(((st==2)&alarm).sum());fp=int(((st==0)&alarm).sum());tn=int(((st==0)&~alarm).sum());fn=int(((st==2)&~alarm).sum());events=hits=left=right=coverage=0;delays=[]
  for i in range(len(rr)):
   if st[i]!=2 or (i>0 and not breaks[i] and st[i-1]==2):continue
   j=i+1
   while j<len(rr) and not breaks[j] and st[j]==2:j+=1
   hit=np.flatnonzero(alarm[i:j]);coverage+=bool(len(hit));left+=int(breaks[i]);right+=int(j==len(rr) or (j<len(rr) and breaks[j]))
   if not breaks[i]:
    events+=1
    if len(hit):hits+=1;delays.append(int(hit[0]))
  result.append(dict(episode=ep,group=rr[0]['group'],tn=tn,fp=fp,fn=fn,tp=tp,events=events,hits=hits,left_censored=left,right_boundary=right,gross_segments=events+left,segments_covered=coverage,false_starts=int((starts&(st==0)).sum()),static_alarming_frames=fp,delay_sum=sum(delays),delay_n=len(delays),trial_count=1,incipient_frames=int((st==1).sum()),observed_alarm=int(alarm.any())))
 return result

def aggregate(trials,weights=None):
 keys=('tn','fp','fn','tp','events','hits','left_censored','right_boundary','gross_segments','segments_covered','false_starts','static_alarming_frames','delay_sum','delay_n','trial_count','incipient_frames','observed_alarm');v={key:sum(r[key]*(1 if weights is None else weights.get(r['group'],0)) for r in trials) for key in keys};out=ratio_metrics([v[k] for k in ('tn','fp','fn','tp')]);out.update(v);out.update(event_recall=divide(v['hits'],v['events']),segment_coverage=divide(v['segments_covered'],v['gross_segments']),false_starts_per_trial=divide(v['false_starts'],v['trial_count']),mean_delay=divide(v['delay_sum'],v['delay_n']),static_alarming_fraction=divide(v['fp'],v['tn']+v['fp']),observed_no_alarm=v['observed_alarm']==0)
 return out

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--allow-synthetic',action='store_true');args=parser.parse_args();manifest=json.loads(args.manifest.read_text());out=args.output;out.mkdir(parents=True,exist_ok=True)
 if manifest.get('status')!='complete':raise ValueError('Incomplete manifest')
 audit=json.loads(verify(manifest['accepted_training_audit']).read_text())
 if audit.get('status')!='pass':raise ValueError('Training audit not accepted')
 synthetic=bool(audit.get('synthetic',False))
 if synthetic and not args.allow_synthetic:raise ValueError('Synthetic requires explicit flag')
 if not synthetic and (audit.get('schema')!='round9_training_audit_v1' or audit.get('expected_count')!=36 or audit.get('test_role_consumed') is not False):raise ValueError('Invalid training audit schema')
 runs=manifest['runs'];new=[r for r in runs if not r.get('historical',False)];folds=sorted({r['fold'] for r in new});grid={(g,f,s) for g in GROUPS for f in folds for s in SEEDS}
 if folds!=['htt_leave_p1','htt_leave_p2','htt_leave_p3','htt_leave_p4'] or len(new)!=36 or {(r['group'],r['fold'],r['seed']) for r in new}!=grid:raise ValueError('Expected full 36-run grid')
 if not synthetic:
  accepted={(r['group'],r['fold'],r['seed']):r for r in audit['accepted_runs']}
  if len(audit['accepted_runs'])!=36 or set(accepted)!=grid:raise ValueError('Incomplete accepted audit grid')
  for run in new:
   item=accepted[(run['group'],run['fold'],run['seed'])]
   for key in ('training_summary','checkpoint'):
    if run[key]!=item[key]:raise ValueError('Manifest model not accepted by audit')
   for role in ('calibration','validation'):
    if run['predictions'][role]!=item['predictions'][role]:raise ValueError('Manifest prediction not accepted by audit')
   record=json.loads(verify(run['training_summary']).read_text())
   if record.get('schema')!='round9_htt_training_summary_v1' or record.get('smoke') is True or record.get('status')!='complete':raise ValueError('Summary is not formal complete R9')
 for run in runs:
  if run.get('historical',False):
   historic=json.loads(verify(run['historical_evaluation']).read_text());prov=historic['provenance']
   if historic.get('status')!='complete' or historic.get('formal') is not True or historic['fold']!=run['fold'] or historic['seed']!=run['seed'] or 'historical_'+historic['model_id']!=run['group']:raise ValueError('Invalid historical evaluation identity')
   if run['training_summary']!={'path':prov['training_summary_path'],'sha256':prov['training_summary_sha256']} or run['checkpoint']!={'path':prov['best_checkpoint'],'sha256':prov['best_checkpoint_sha256']}:raise ValueError('Historical checkpoint chain mismatch')
   for rec in run['predictions'].values():
    if prov['files'].get(rec['path'])!=rec['sha256']:raise ValueError('Historical prediction chain mismatch')
 identities=[(r['group'],r['fold'],r['seed']) for r in runs]
 if len(set(identities))!=len(identities):raise ValueError('Duplicate run identity')
 datasets={};points={};trialcache={};metrics=[];trials=[];inc=[];curve=[];fail=[];canonical={};input_hashes={}
 for run in runs:
  ident=(run['group'],run['fold'],run['seed']);meta=dict(zip(('group','fold','seed'),ident));datasets[ident]={}
  for key in ('training_summary','checkpoint'):verify(run[key]);input_hashes[str(run[key]['path'])]=run[key]['sha256']
  for role in ('calibration','validation'):
   rr=load_rows(run['predictions'][role],run['endpoints'][role],role,run.get('historical',False),run['fold']);datasets[ident][role]=rr;signature=[(r['episode'],r['t'],r['stage'],r['group']) for r in rr];ck=(run['fold'],role)
   if ck in canonical and canonical[ck]!=signature:raise ValueError('Non-common endpoints across runs')
   canonical[ck]=signature
   for rec in (run['predictions'][role],run['endpoints'][role]):input_hashes[rec['path']]=rec['sha256']
  if {r['group'] for r in datasets[ident]['calibration']}&{r['group'] for r in datasets[ident]['validation']}:raise ValueError('Calibration validation leakage')
  points[ident]=choose(datasets[ident]['calibration'])
  for role,rr in datasets[ident].items():
   rank=rank_metrics(rr);ss=np.array([r['score'] for r in rr if r['stage']==1]);inc.append({**meta,'role':role,'count':len(ss),'mean':float(ss.mean()) if len(ss) else None,'q10':float(np.quantile(ss,.1)) if len(ss) else None,'q50':float(np.quantile(ss,.5)) if len(ss) else None,'q90':float(np.quantile(ss,.9)) if len(ss) else None})
   for point,threshold in points[ident].items():
    for rule,k in RULES.items():
     tt=trial_metrics(rr,threshold,k);trialcache[(ident,role,point,rule)]=tt;agg=aggregate(tt);metrics.append({**meta,'role':role,'point':point,'rule':rule,'threshold':threshold,'never_alarm':threshold>1,'selection':'fixed' if point=='fixed_0.5' else 'calibration_raw_only',**agg,**rank});trials.extend({**x,**meta,'leakage_group':x['group'],'role':role,'point':point,'rule':rule} for x in tt)
   for x in thresholds(rr):curve.append({**meta,'role':role,'scope':'descriptive_only',**x})
  tt=trialcache[(ident,'validation','FPR0.05','raw')]
  fps=[x for x in tt if x['fp']>0];fns=[x for x in tt if x['events']>0]
  if fps:
   x=sorted(fps,key=lambda x:(-divide(x['fp'],x['tn']+x['fp']),-x['false_starts'],x['episode']))[0];fail.append({**x,**meta,'leakage_group':x['group'],'kind':'false_positive'})
  if fns:
   x=sorted(fns,key=lambda x:(divide(x['hits'],x['events']),-(x['events']-x['hits']),x['episode']))[0];fail.append({**x,**meta,'leakage_group':x['group'],'kind':'missed_event'})
  print('evaluated',ident,flush=True)
 csvout(out/'metrics.csv',metrics);csvout(out/'trials.csv',trials);csvout(out/'incipient_distribution.csv',inc);csvout(out/'roc_points.csv',curve);csvout(out/'failure_cases.csv',fail)
 # Shared fold-level leakage-group draws, never independent seed/frame resampling.
 paired=[];seed_diffs=[];ranknames=('AP','pAUC','brier');temporal=('static_fpr','gross_recall','event_recall','false_starts_per_trial','static_alarming_fraction','mean_delay')
 for fi,fold in enumerate(folds):
  names=sorted({r['group'] for r in datasets[(GROUPS[0],fold,SEEDS[0])]['validation']});rng=np.random.default_rng(20260916+fi);draws=[dict(zip(names,np.bincount(rng.integers(0,len(names),len(names)),minlength=len(names)))) for _ in range(200)]
  for candidate,base in PAIRS:
   for point,rule in [('ranking','raw')]+[(p,r) for p in points[(base,fold,SEEDS[0])] for r in RULES]:
    keys=ranknames if point=='ranking' else temporal;rep={k:[] for k in keys};raw={k:[] for k in keys}
    for wi,weights in enumerate([None]+draws):
     dv={k:[] for k in keys}
     for seed in SEEDS:
      vals=[]
      for group in (candidate,base):
       ident=(group,fold,seed);vals.append(rank_metrics(datasets[ident]['validation'],weights) if point=='ranking' else aggregate(trialcache[(ident,'validation',point,rule)],weights))
      for key in keys:
       d=vals[0][key]-vals[1][key];dv[key].append(d)
       if wi==0:seed_diffs.append(dict(fold=fold,candidate=candidate,base=base,seed=seed,point=point,rule=rule,metric=key,difference=d))
     for key in keys:
      value=float(np.mean(dv[key]))
      if wi==0:raw[key]=value
      elif np.isfinite(value):rep[key].append(value)
    for key in keys:
     v=rep[key];paired.append(dict(fold=fold,candidate=candidate,base=base,point=point,rule=rule,metric=key,difference=raw[key],ci_lower=float(np.quantile(v,.025)) if v else None,ci_upper=float(np.quantile(v,.975)) if v else None,valid_bootstraps=len(v),requested_bootstraps=200,groups=len(names),status='available' if v else 'undefined',unit='complete_leakage_group_shared_across_seeds'))
   print('bootstrap complete',fold,candidate,flush=True)
 csvout(out/'paired_ci.csv',paired);csvout(out/'paired_seed_differences.csv',seed_diffs)
 summary=[]
 for group in sorted({r['group'] for r in runs}):
  for point in sorted({r['point'] for r in metrics}):
   for rule in RULES:
    subset=[r for r in metrics if r['group']==group and r['point']==point and r['rule']==rule and r['role']=='validation']
    for key in (*ranknames,*temporal,'balanced_accuracy','macro_f1'):
     values=np.array([r[key] for r in subset]);valid=values[np.isfinite(values)];summary.append(dict(group=group,point=point,rule=rule,metric=key,mean=float(valid.mean()) if len(valid) else None,sd_descriptive=float(valid.std(ddof=1)) if len(valid)>1 else None,available_runs=len(valid),total_runs=len(subset),scope='overlapping_fold_seed_descriptive_not_independent'))
 csvout(out/'summary_metrics.csv',summary)
 descriptive=[]
 for ident,roles in datasets.items():
  rr=roles['validation']; selected=choose(rr); candidates=thresholds(rr)
  for name,threshold in selected.items():
   if name.startswith(('FPR','recall')):
    x=next(x for x in candidates if x['threshold']==threshold);descriptive.append(dict(group=ident[0],fold=ident[1],seed=ident[2],point=name,scope='validation_envelope_not_deployable',**x))
 csvout(out/'descriptive_workpoints.csv',descriptive)
 js(out/'summary.json',dict(status='complete',synthetic=synthetic,accepted_training_audit_sha256=sha(verify(manifest['accepted_training_audit'])),runs=len(runs),new_runs=len(new),manifest_sha256=sha(args.manifest),source_sha256=sha(__file__),protocol_sha256=sha(Path(__file__).with_name('PROTOCOL.md')),input_hashes=input_hashes,output_hashes={p.name:sha(p) for p in out.glob('*.csv')},limitations=['validation checkpoint selection','overlapping development folds','force-derived HTT labels','image-derived predicted force','no physical validation']))

if __name__=='__main__':main()
