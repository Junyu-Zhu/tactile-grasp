#!/usr/bin/env python3
"""Round-6 work package A: frozen analysis of 48 accepted Round-5 current predictions."""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,math,os
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
R6=HERE.parent
HTT=R6.parent
R5=HTT/'round5_force_conditioned_slip'
MODELS=('mae-r3-b','V','F-old','F-adapt'); CANDIDATES=('mae-r3-b','F-old','F-adapt')
FOLDS=tuple(f'htt_leave_p{i}' for i in range(1,5)); SEEDS=(20260914,20260915,20260916)
RECALL_LEVELS=(0.80,0.90,0.95); FPR_LEVELS=(0.01,0.05,0.10); BOOTSTRAPS=200

def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
r5eval=module('round5_eval_for_r6',R5/'evaluate_slip.py'); scheduler=module('round5_scheduler_for_r6',R5/'run_formal.py')

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def atomic_json(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.tmp.{os.getpid()}');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n');os.replace(tmp,path)
def atomic_csv(path,rows,fields=None):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fields=fields or list(rows[0]);tmp=path.with_name(path.name+f'.tmp.{os.getpid()}')
 with tmp.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 os.replace(tmp,path)
def argv_value(argv,key):
 if argv.count(key)!=1:raise ValueError(f'exactly one {key} required')
 return argv[argv.index(key)+1]
def stable_seed(*parts):return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:8],'little')
def expected():return {(m,f,s) for m in MODELS for f in FOLDS for s in SEEDS}
def identity(job):
 a=job['argv'];return argv_value(a,'--model-id'),argv_value(a,'--fold'),int(argv_value(a,'--seed'))

def discover(inventory):
 jobs={}; errors=[]
 for j in inventory.get('jobs',[]):
  if j.get('kind')!='evaluation':continue
  try:k=identity(j)
  except Exception as e:errors.append(f"{j.get('id')}: {e}");continue
  if k in jobs:errors.append(f'duplicate {k}')
  jobs[k]=j
  if (k[0]=='mae-r3-b') != ('--historical-baseline' in j['argv']):errors.append(f'historical flag mismatch {k}')
 if set(jobs)!=expected():errors.append(f'identity mismatch missing={sorted(expected()-set(jobs))} extra={sorted(set(jobs)-expected())}')
 if len(jobs)!=48:errors.append(f'expected 48, got {len(jobs)}')
 if errors:raise ValueError('; '.join(errors))
 return jobs

def binary_arrays(rows):
 chosen=[r for r in rows if r['stage'] in (0,2)]
 y=np.asarray([r['stage']==2 for r in chosen],dtype=bool); score=np.asarray([r['score'] for r in chosen],float)
 if not len(y) or y.all() or (~y).all():raise ValueError('binary population lacks a class')
 return chosen,y,score

def curve(rows):
 _,y,s=binary_arrays(rows);order=np.argsort(-s,kind='mergesort');ss=s[order];yy=y[order].astype(np.int64)
 ends=np.r_[np.flatnonzero(ss[1:]!=ss[:-1]),len(ss)-1];tp=np.cumsum(yy)[ends];predicted=ends+1;fp=predicted-tp
 p=int(y.sum());n=int((~y).sum());out=[{'threshold':'inf','static_fpr':0.0,'gross_recall':0.0,'precision':1.0,'tp':0,'fp':0,'fn':p,'tn':n}]
 for i,end in enumerate(ends):
  tpi=int(tp[i]);fpi=int(fp[i]);rec=tpi/p;fpr=fpi/n;prec=tpi/(tpi+fpi)
  out.append({'threshold':float(ss[end]),'static_fpr':fpr,'gross_recall':rec,'precision':prec,'tp':tpi,'fp':fpi,'fn':p-tpi,'tn':n-fpi})
 return out

def choose_point(rows,kind,target):
 points=curve(rows)
 if kind=='recall':
  feasible=[p for p in points if p['gross_recall']+1e-15>=target]
  return min(feasible,key=lambda p:(p['static_fpr'],-p['gross_recall'],-(-math.inf if p['threshold']=='inf' else p['threshold'])))
 if kind=='fpr':
  feasible=[p for p in points if p['static_fpr']<=target+1e-15]
  return min(feasible,key=lambda p:(-p['gross_recall'],p['static_fpr'],-(-math.inf if p['threshold']=='inf' else p['threshold'])))
 raise KeyError(kind)

def evaluate_threshold(rows,threshold):
 _,y,s=binary_arrays(rows); pred=np.zeros(len(s),bool) if threshold=='inf' else s>=float(threshold)
 tp=int((pred&y).sum());fp=int((pred&~y).sum());fn=int((~pred&y).sum());tn=int((~pred&~y).sum())
 return {'threshold':threshold,'static_fpr':fp/(fp+tn),'gross_recall':tp/(tp+fn),'precision':tp/(tp+fp) if tp+fp else 1.0,'tp':tp,'fp':fp,'fn':fn,'tn':tn,'never_alarm':bool(not pred.any())}

def split_episodes(rows):
 d=defaultdict(list)
 for r in rows:d[r['episode']].append(r)
 return {k:sorted(v,key=lambda x:x['t']) for k,v in d.items()}
def episode_metadata(episode):
 parts=episode.split('/')
 return {'task':parts[0] if parts else '', 'probe':parts[1] if len(parts)>1 else ''}
def segments(stage):
 gross=np.asarray(stage)==2; starts=np.flatnonzero(gross & np.r_[True,~gross[:-1]]);ends=np.flatnonzero(gross & np.r_[~gross[1:],True])+1;return list(zip(starts,ends))
def sequence_metrics(rows,threshold):
 episodes=split_episodes(rows); static=alarm_frames=false_starts=allseg=hitseg=left=unc=unchit=0;delays=[]
 for seq in episodes.values():
  scores=np.asarray([r['score'] for r in seq]);stage=np.asarray([r['stage'] for r in seq]);alarm=np.zeros(len(seq),bool) if threshold=='inf' else scores>=float(threshold)
  starts=alarm & np.r_[True,~alarm[:-1]];sm=stage==0
  static+=int(sm.sum());alarm_frames+=int((alarm&sm).sum());false_starts+=int((starts&sm).sum())
  for a,b in segments(stage):
   allseg+=1
   if alarm[a:b].any():hitseg+=1
   if a==0:left+=1;continue
   unc+=1;hits=np.flatnonzero(alarm[a:b])
   if len(hits):unchit+=1;delays.append(int(hits[0]))
 return {'episodes':len(episodes),'false_alarm_starts':false_starts,'false_alarm_starts_per_trial':false_starts/len(episodes),'static_alarming_fraction':alarm_frames/static if static else None,'gross_segments_including_left_censored':allseg,'gross_segment_alarm_coverage':hitseg/allseg if allseg else None,'left_censored_gross_segments':left,'uncensored_gross_events':unc,'uncensored_gross_event_recall':unchit/unc if unc else None,'uncensored_mean_delay_frames':float(np.mean(delays)) if delays else None,'uncensored_median_delay_frames':float(np.median(delays)) if delays else None}

def group_map(manifest):return {r['id']:r['leakage_group'] for r in manifest['episodes']}
def grouped(rows,groups):
 d=defaultdict(list)
 for r in rows:d[groups[r['episode']]].append(r)
 return dict(d)
def sample_groups(d,names):return [r for name in names for r in d[name]]

def ranks(a):
 a=np.asarray(a,float);order=np.argsort(a,kind='mergesort');out=np.empty(len(a),float);i=0
 while i<len(a):
  j=i+1
  while j<len(a) and a[order[j]]==a[order[i]]:j+=1
  out[order[i:j]]=(i+j-1)/2+1;i=j
 return out
def weighted_static_mean(rows):
 selected=[r for r in rows if r['static_frames']>0]
 if not selected:raise ValueError('role has no static frames')
 return float(np.average([r['mean_score'] for r in selected],weights=[r['static_frames'] for r in selected]))
def corr(a,b):
 a=np.asarray(a,float);b=np.asarray(b,float)
 if len(a)<2 or np.std(a)==0 or np.std(b)==0:return None
 return float(np.corrcoef(a,b)[0,1])

def load_runs(jobs,manifest,contract,audit):
 labels={f:r5eval.load_authoritative_labels(contract,audit,manifest,Path(argv_value(next(iter(jobs.values()))['argv'],'--manifest')),f)[0] for f in FOLDS}
 runs={}; source={}
 for k,j in sorted(jobs.items()):
  if not scheduler.accepted(j):raise ValueError(f'R5 scheduler acceptance failed {k}')
  mp=Path(j['acceptance_path']);payload=json.loads(mp.read_text())
  if payload.get('status')!='complete' or payload.get('formal') is not True or (payload['model_id'],payload['fold'],int(payload['seed']))!=k:raise ValueError(f'metrics identity/status mismatch {k}')
  source[str(mp)]=sha(mp)
  role_rows={}
  for role,opt in [('calibration','--calibration'),('validation','--validation')]:
   pp=Path(argv_value(j['argv'],opt)); recorded=payload['provenance']['files'].get(str(pp))
   if not recorded or sha(pp)!=recorded:raise ValueError(f'prediction hash mismatch {k}/{role}')
   source[str(pp)]=recorded;role_rows[role]=r5eval.read_rows(pp,manifest,k[1],role,labels[k[1]])
  runs[k]={'payload':payload,'rows':role_rows,'metrics_path':mp}
 # exact paired population within each fold/role across all models/seeds
 for f in FOLDS:
  for role in ('calibration','validation'):
   ref=None
   for m in MODELS:
    for s in SEEDS:
     keys=[(r['episode'],r['t'],r['stage']) for r in runs[(m,f,s)]['rows'][role]]
     if ref is None:ref=keys
     elif keys!=ref:raise ValueError(f'population mismatch {f}/{role}/{m}/{s}')
 return runs,source

def make_points(runs):
 operational=[];descriptive=[];curves=[];sequences=[]
 for (m,f,s),run in sorted(runs.items()):
  for role in ('calibration','validation'):
   for p in curve(run['rows'][role]):curves.append({'model':m,'fold':f,'seed':s,'role':role,**p})
  for kind,levels in [('recall',RECALL_LEVELS),('fpr',FPR_LEVELS)]:
   for target in levels:
    selected=choose_point(run['rows']['calibration'],kind,target); val=evaluate_threshold(run['rows']['validation'],selected['threshold']);seq=sequence_metrics(run['rows']['validation'],selected['threshold'])
    migration_failed=val['gross_recall']+1e-15<target if kind=='recall' else val['static_fpr']>target+1e-15
    operational.append({'model':m,'fold':f,'seed':s,'selection_role':'calibration','application_role':'validation','target_kind':kind,'target':target,'threshold':selected['threshold'],'calibration_static_fpr':selected['static_fpr'],'calibration_gross_recall':selected['gross_recall'],'validation_static_fpr':val['static_fpr'],'validation_gross_recall':val['gross_recall'],'validation_precision':val['precision'],'validation_never_alarm':val['never_alarm'],'migration_failed':migration_failed,'migration_margin':val['gross_recall']-target if kind=='recall' else target-val['static_fpr']})
    sequences.append({'model':m,'fold':f,'seed':s,'target_kind':kind,'target':target,'threshold':selected['threshold'],**seq})
    oracle=choose_point(run['rows']['validation'],kind,target)
    descriptive.append({'model':m,'fold':f,'seed':s,'scope':'descriptive_validation_only','target_kind':kind,'target':target,**oracle})
 return operational,descriptive,curves,sequences

def trial_diagnostics(runs,operational,manifest):
 groups=group_map(manifest);op={(r['model'],r['fold'],int(r['seed']),r['target_kind'],float(r['target'])):r for r in operational}
 migration=[];distributions=[]
 for (model,fold,seed),run in sorted(runs.items()):
  for role in ('calibration','validation'):
   for episode,seq in split_episodes(run['rows'][role]).items():
    meta=episode_metadata(episode);static=[r for r in seq if r['stage']==0];incipient=[r for r in seq if r['stage']==1];gross=[r for r in seq if r['stage']==2]
    scores=np.asarray([r['score'] for r in static],float)
    distributions.append({'model':model,'fold':fold,'seed':seed,'role':role,'episode':episode,**meta,'leakage_group':groups[episode],'slice':'all_static','sequence_quarter':'all','static_frames':len(static),'incipient_frames_in_episode':len(incipient),'gross_frames_in_episode':len(gross),'mean_score':float(np.mean(scores)) if len(scores) else None,'std_score':float(np.std(scores)) if len(scores) else None,'q50_score':float(np.quantile(scores,.5)) if len(scores) else None,'q90_score':float(np.quantile(scores,.9)) if len(scores) else None,'q95_score':float(np.quantile(scores,.95)) if len(scores) else None,'max_score':float(np.max(scores)) if len(scores) else None})
    for quarter in range(4):
     selected=[]
     for i,r in enumerate(seq):
      q=min(3,int(4*i/max(1,len(seq))))
      if q==quarter and r['stage']==0:selected.append(r['score'])
     arr=np.asarray(selected,float)
     distributions.append({'model':model,'fold':fold,'seed':seed,'role':role,'episode':episode,**meta,'leakage_group':groups[episode],'slice':'static_sequence_quarter','sequence_quarter':quarter+1,'static_frames':len(arr),'incipient_frames_in_episode':len(incipient),'gross_frames_in_episode':len(gross),'mean_score':float(np.mean(arr)) if len(arr) else None,'std_score':float(np.std(arr)) if len(arr) else None,'q50_score':float(np.quantile(arr,.5)) if len(arr) else None,'q90_score':float(np.quantile(arr,.9)) if len(arr) else None,'q95_score':float(np.quantile(arr,.95)) if len(arr) else None,'max_score':float(np.max(arr)) if len(arr) else None})
    if role!='validation':continue
    for kind,levels in [('recall',RECALL_LEVELS),('fpr',FPR_LEVELS)]:
     for target in levels:
      threshold=op[(model,fold,seed,kind,target)]['threshold'];ev=None
      if static and gross:ev=evaluate_threshold(seq,threshold)
      elif static:
       alarm=np.zeros(len(static),bool) if threshold=='inf' else np.asarray([r['score'] for r in static])>=float(threshold);ev={'static_fpr':float(np.mean(alarm)),'gross_recall':None}
      elif gross:
       alarm=np.zeros(len(gross),bool) if threshold=='inf' else np.asarray([r['score'] for r in gross])>=float(threshold);ev={'static_fpr':None,'gross_recall':float(np.mean(alarm))}
      seqm=sequence_metrics(seq,threshold)
      assessable=(bool(gross) if kind=='recall' else bool(static));failed=(ev['gross_recall']<target if kind=='recall' else ev['static_fpr']>target) if assessable else None
      migration.append({'model':model,'fold':fold,'seed':seed,'episode':episode,**meta,'leakage_group':groups[episode],'target_kind':kind,'target':target,'threshold':threshold,'static_frames':len(static),'incipient_frames':len(incipient),'gross_frames':len(gross),'trial_static_fpr':ev['static_fpr'],'trial_gross_recall':ev['gross_recall'],'assessable_for_target':assessable,'migration_failed':failed,'not_assessable_reason':'' if assessable else ('no_gross_frames' if kind=='recall' else 'no_static_frames'),'false_alarm_starts':seqm['false_alarm_starts'],'static_alarming_fraction':seqm['static_alarming_fraction'],'left_censored_gross_segments':seqm['left_censored_gross_segments'],'uncensored_gross_events':seqm['uncensored_gross_events'],'uncensored_gross_event_recall':seqm['uncensored_gross_event_recall'],'uncensored_mean_delay_frames':seqm['uncensored_mean_delay_frames']})
 # role-level distribution shift is descriptive and model/identity complete
 shifts=[]
 for model in MODELS:
  for fold in FOLDS:
   for seed in SEEDS:
    rr=[r for r in distributions if r['model']==model and r['fold']==fold and int(r['seed'])==seed and r['slice']=='all_static']
    by={role:[r for r in rr if r['role']==role] for role in ('calibration','validation')}
    for role in by:
     if not by[role]:raise ValueError('missing static distribution role')
    cal=weighted_static_mean(by['calibration']);val=weighted_static_mean(by['validation'])
    shifts.append({'model':model,'fold':fold,'seed':seed,'calibration_static_frames':sum(r['static_frames'] for r in by['calibration']),'validation_static_frames':sum(r['static_frames'] for r in by['validation']),'calibration_frame_weighted_mean_static_score':float(cal),'validation_frame_weighted_mean_static_score':float(val),'validation_minus_calibration_mean_static_score':float(val-cal),'image_geometry_background_quantitative_status':'not_observable_from_frozen_probability_files; see bound Round-4 audit'})
 return migration,distributions,shifts

def representative_failures(runs,trial_rows):
 chosen=[]
 for model in MODELS:
  fp=[r for r in trial_rows if r['model']==model and r['target_kind']=='fpr' and float(r['target'])==.05 and r['assessable_for_target'] and r['trial_static_fpr'] is not None]
  best=sorted(fp,key=lambda r:(-float(r['trial_static_fpr']),r['fold'],int(r['seed']),r['episode']))[0];chosen.append({**best,'failure_type':'highest_static_fpr_at_calibration_fpr_0.05'})
  fn=[r for r in trial_rows if r['model']==model and r['target_kind']=='recall' and float(r['target'])==.90 and int(r['uncensored_gross_events'])>0 and r['trial_gross_recall'] is not None]
  best=sorted(fn,key=lambda r:(float(r['trial_gross_recall']),r['fold'],int(r['seed']),r['episode']))[0];chosen.append({**best,'failure_type':'lowest_gross_recall_at_calibration_recall_0.90'})
 return chosen

def paired_bootstrap(runs,operational,manifest):
 op={(r['model'],r['fold'],int(r['seed']),r['target_kind'],float(r['target'])):r for r in operational};groups=group_map(manifest); seedrows=[];foldrows=[]
 for scope in ('operational_transfer','descriptive_validation_only'):
  for cand in CANDIDATES:
   for fold in FOLDS:
    validation={s:{m:grouped(runs[(m,fold,s)]['rows']['validation'],groups) for m in (cand,'V')} for s in SEEDS};names=sorted(next(iter(validation.values()))[cand])
    for s in SEEDS:
     if sorted(validation[s]['V'])!=names:raise ValueError('paired groups differ')
    for kind,levels in [('recall',RECALL_LEVELS),('fpr',FPR_LEVELS)]:
     for target in levels:
      points=[]
      for s in SEEDS:
       if scope=='operational_transfer':
        ca=op[(cand,fold,s,kind,target)];ba=op[('V',fold,s,kind,target)]
        points.append((ba['validation_static_fpr']-ca['validation_static_fpr']) if kind=='recall' else (ca['validation_gross_recall']-ba['validation_gross_recall']))
       else:
        ca=choose_point(runs[(cand,fold,s)]['rows']['validation'],kind,target);ba=choose_point(runs[('V',fold,s)]['rows']['validation'],kind,target)
        points.append((ba['static_fpr']-ca['static_fpr']) if kind=='recall' else (ca['gross_recall']-ba['gross_recall']))
       seedrows.append({'scope':scope,'candidate':cand,'baseline':'V','fold':fold,'seed':s,'target_kind':kind,'target':target,'advantage_point':points[-1],'positive_is_candidate_better':True})
      rng=np.random.default_rng(stable_seed(scope,cand,fold,kind,target));draws=[]
      for _ in range(BOOTSTRAPS):
       sample=list(rng.choice(names,len(names),replace=True));diff=[]
       for s in SEEDS:
        cr=sample_groups(validation[s][cand],sample);br=sample_groups(validation[s]['V'],sample)
        try:
         if scope=='operational_transfer':
          ct=op[(cand,fold,s,kind,target)]['threshold'];bt=op[('V',fold,s,kind,target)]['threshold'];ce=evaluate_threshold(cr,ct);be=evaluate_threshold(br,bt)
         else:ce=choose_point(cr,kind,target);be=choose_point(br,kind,target)
         diff.append((be['static_fpr']-ce['static_fpr']) if kind=='recall' else (ce['gross_recall']-be['gross_recall']))
        except ValueError:diff=[];break
       if len(diff)==3 and all(math.isfinite(x) for x in diff):draws.append(float(np.mean(diff)))
      foldrows.append({'scope':scope,'candidate':cand,'baseline':'V','fold':fold,'target_kind':kind,'target':target,'three_seed_mean_advantage':float(np.mean(points)),'ci_lower':float(np.quantile(draws,.025)) if draws else None,'ci_upper':float(np.quantile(draws,.975)) if draws else None,'bootstrap_groups':len(names),'replicates_requested':BOOTSTRAPS,'replicates_valid_two_class':len(draws),'shared_group_draw_across_models_and_three_seeds':True,'positive_is_candidate_better':True})
 return seedrows,foldrows

def r5_snapshot(runs):
 out=[]
 for (m,f,s),run in sorted(runs.items()):
  for name,item in run['payload']['operating_points'].items():
   for mode in ('raw','sequential'):
    if mode not in item:continue
    for role in ('calibration','validation'):
     x=item[mode][role];out.append({'model':m,'fold':f,'seed':s,'operating_point':name,'mode':mode,'role':role,'threshold':x['threshold'],'confirmation_k':x['confirmation_k'],'release_ratio':x['release_ratio'],'static_fpr':x['static_fpr'],'gross_recall':x['gross_recall'],'false_alarm_starts_per_trial':x['false_alarm_starts_per_trial'],'mean_detection_delay_frames':x['mean_detection_delay_frames'],'never_alarm':x['never_alarm'],'source_metrics_path':str(run['metrics_path'])})
 return out

def force_association(operational,r5root):
 feats=[]
 for f in FOLDS:
  p=f.split('p')[-1]
  for s in SEEDS:
   path=r5root/f'analysis_force/htt/p{p}_s{s}_validation/metrics.json';x=json.loads(path.read_text());axis=x['supervised_regression']['axis'];inv=x['supervised_regression']['invariant']
   feats.append({'fold':f,'seed':s,'axis_abs_bias_mean_n':float(np.mean(np.abs(axis['bias_n']))),'shear_x_bias_n':axis['bias_n'][0],'shear_y_bias_n':axis['bias_n'][1],'normal_bias_n':axis['bias_n'][2],'Fn_bias_n':inv['Fn']['bias_n'],'Ft_bias_n':inv['Ft']['bias_n'],'Fmag_bias_n':inv['Fmag']['bias_n'],'Fmag_rmse_n':inv['Fmag']['rmse_n'],'target_any_saturation_fraction':x['target_saturation']['any_axis_fraction'],'source_path':str(path),'source_sha256':sha(path)})
 op={(r['model'],r['fold'],int(r['seed']),r['target_kind'],float(r['target'])):r for r in operational};obs=[]
 for model in ('F-old','F-adapt'):
  for kind,levels in [('recall',RECALL_LEVELS),('fpr',FPR_LEVELS)]:
   for target in levels:
    for ft in feats:
     c=op[(model,ft['fold'],ft['seed'],kind,target)];v=op[('V',ft['fold'],ft['seed'],kind,target)]
     advantage=(v['validation_static_fpr']-c['validation_static_fpr']) if kind=='recall' else (c['validation_gross_recall']-v['validation_gross_recall'])
     obs.append({**ft,'model':model,'target_kind':kind,'target':target,'migration_margin':c['migration_margin'],'relative_V_advantage':advantage,'pairing_unit':'fold_seed_aggregate_only'})
 summaries=[];features=['axis_abs_bias_mean_n','shear_x_bias_n','shear_y_bias_n','normal_bias_n','Fn_bias_n','Ft_bias_n','Fmag_bias_n','Fmag_rmse_n','target_any_saturation_fraction']
 for model in ('F-old','F-adapt'):
  for kind,levels in [('recall',RECALL_LEVELS),('fpr',FPR_LEVELS)]:
   for target in levels:
    rr=[r for r in obs if r['model']==model and r['target_kind']==kind and r['target']==target]
    for outcome in ('migration_margin','relative_V_advantage'):
     for feature in features:
      a=[r[feature] for r in rr];b=[r[outcome] for r in rr]
      summaries.append({'model':model,'target_kind':kind,'target':target,'force_feature':feature,'outcome':outcome,'n_fold_seed_aggregates':len(rr),'pearson_r':corr(a,b),'spearman_rho':corr(ranks(a),ranks(b)),'interpretation':'descriptive aggregate association; overlapping folds; no trial pairing or causal claim'})
 return feats,obs,summaries

def render(curves,operational,assoc,representatives,runs,output):
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 d=Path(output)/'figures';d.mkdir(parents=True,exist_ok=True);colors=dict(zip(MODELS,['black','tab:blue','tab:green','tab:orange']))
 for plot in ('roc','pr','low_fpr'):
  fig,axes=plt.subplots(3,4,figsize=(17,11),sharex=False,sharey=True)
  for i,seed in enumerate(SEEDS):
   for j,fold in enumerate(FOLDS):
    ax=axes[i,j]
    for model in MODELS:
     rr=[r for r in curves if r['role']=='validation' and r['model']==model and r['fold']==fold and int(r['seed'])==seed]
     if plot=='pr':x=[r['gross_recall'] for r in rr];y=[r['precision'] for r in rr];ax.set_xlim(0,1);ax.set_ylim(0,1)
     else:x=[r['static_fpr'] for r in rr];y=[r['gross_recall'] for r in rr];ax.set_xlim(0,.1 if plot=='low_fpr' else 1);ax.set_ylim(0,1)
     ax.plot(x,y,lw=1,color=colors[model],label=model if i==0 and j==0 else '_nolegend_')
    ax.set_title(f'{fold} seed {seed}');ax.grid(alpha=.2)
  fig.legend(loc='upper center',ncol=4);fig.supxlabel('recall' if plot=='pr' else 'static FPR');fig.supylabel('precision' if plot=='pr' else 'gross recall');fig.tight_layout(rect=(.02,.02,1,.95));fig.savefig(d/f'{plot}_all_folds_seeds.png',dpi=160);plt.close(fig)
 # migration failures by model/target
 labels=[f'R{int(x*100)}' for x in RECALL_LEVELS]+[f'FPR{int(x*100)}' for x in FPR_LEVELS];fig,ax=plt.subplots(figsize=(10,5));x=np.arange(6);w=.18
 for i,m in enumerate(MODELS):
  vals=[]
  for kind,levels in [('recall',RECALL_LEVELS),('fpr',FPR_LEVELS)]:
   for target in levels:
    rr=[r for r in operational if r['model']==m and r['target_kind']==kind and r['target']==target];vals.append(np.mean([str(r['migration_failed']).lower()=='true' if isinstance(r['migration_failed'],str) else r['migration_failed'] for r in rr]))
  ax.bar(x+(i-1.5)*w,vals,w,label=m)
 ax.set_xticks(x,labels);ax.set_ylabel('validation migration failure fraction');ax.set_ylim(0,1);ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(d/'threshold_migration_failures.png',dpi=160);plt.close(fig)
 # association heat map: strongest absolute rho per target/outcome, descriptive
 rows=assoc;mat=[];yl=[]
 for m in ('F-old','F-adapt'):
  for kind,levels in [('recall',RECALL_LEVELS),('fpr',FPR_LEVELS)]:
   for target in levels:
    rr=[r for r in rows if r['model']==m and r['target_kind']==kind and r['target']==target and r['outcome']=='relative_V_advantage'];best=max(rr,key=lambda r:abs(r['spearman_rho']) if r['spearman_rho'] is not None else -1);mat.append([best['spearman_rho'] or 0]);yl.append(f'{m} {kind} {target:g}\n{best["force_feature"]}')
 fig,ax=plt.subplots(figsize=(6,8));im=ax.imshow(mat,vmin=-1,vmax=1,cmap='coolwarm',aspect='auto');ax.set_yticks(range(len(yl)),yl,fontsize=7);ax.set_xticks([0],['strongest |rho|']);fig.colorbar(im,ax=ax,label='Spearman rho');fig.tight_layout();fig.savefig(d/'force_bias_association_descriptive.png',dpi=160);plt.close(fig)
 for row in representatives:
  seq=split_episodes(runs[(row['model'],row['fold'],int(row['seed']))]['rows']['validation'])[row['episode']];frames=np.asarray([r['t'] for r in seq]);scores=np.asarray([r['score'] for r in seq]);stage=np.asarray([r['stage'] for r in seq]);threshold=row['threshold']
  fig,ax=plt.subplots(figsize=(11,3.8));ax.plot(frames,scores,label='p_slip');
  if threshold!='inf':ax.axhline(float(threshold),ls='--',color='black',label='calibration-selected threshold')
  ax.fill_between(frames,0,1,where=stage==1,color='orange',alpha=.12,label='incipient');ax.fill_between(frames,0,1,where=stage==2,color='red',alpha=.12,label='gross');ax.set_ylim(0,1);ax.set_xlabel('original frame');ax.set_ylabel('score');ax.set_title(f"{row['model']} {row['failure_type']} {row['fold']} seed {row['seed']} {row['episode']}",fontsize=8,wrap=True);ax.legend(ncol=4,fontsize=8);fig.tight_layout();name=f"failure_{row['model'].replace('-','_')}_{'fp' if row['failure_type'].startswith('highest') else 'fn'}.png";fig.savefig(d/name,dpi=160,bbox_inches='tight');plt.close(fig)

def report(operational,foldpairs,assoc,trial_rows,output):
 def avg(model,kind,target,key):return float(np.mean([float(r[key]) for r in operational if r['model']==model and r['target_kind']==kind and r['target']==target]))
 lines=['# 第六轮工作包 A：当前滑移检测阈值迁移与低误报分析','', '本报告只复用第五轮已验收的48份概率；没有训练、重新推理、test-role或阈值搜索扩展。所有目标点在计算前固定。','', '## Calibration 阈值迁移到 validation','', '|模型|目标|validation实际FPR|validation gross召回|迁移失败运行|','|---|---:|---:|---:|---:|']
 for m in MODELS:
  for kind,levels in [('recall',RECALL_LEVELS),('fpr',FPR_LEVELS)]:
   for t in levels:
    rr=[r for r in operational if r['model']==m and r['target_kind']==kind and r['target']==t];lines.append(f"|{m}|{'recall' if kind=='recall' else 'FPR'}={t:.2f}|{np.mean([r['validation_static_fpr'] for r in rr]):.4f}|{np.mean([r['validation_gross_recall'] for r in rr]):.4f}|{sum(r['migration_failed'] for r in rr)}/12|")
 lines+=['','召回约束和低误报约束都只在 calibration 选阈值，再原样迁移。失败数反映分布迁移；validation 描述性包络另表保存，不能作为部署工作点。','', '## 相对 V 的共享组级配对结果','', '正优势定义：召回目标为 `V FPR - 候选FPR`，FPR目标为 `候选recall - V recall`。每折200次抽样共享给候选/V及三个种子；无双类重复不会进入区间，有效数逐行保留。','']
 for m in CANDIDATES:
  vals=[r for r in foldpairs if r['scope']=='operational_transfer' and r['candidate']==m];positive=sum(r['three_seed_mean_advantage']>0 for r in vals);whole=sum(r['ci_lower'] is not None and r['ci_lower']>0 for r in vals);lines.append(f'- {m}: 24个折×目标比较中，点估计正向 {positive}/24，95%区间整体高于0为 {whole}/24。')
 assess=[r for r in trial_rows if r['assessable_for_target']];failed=sum(r['migration_failed'] for r in assess)
 lines+=['','## 逐试次、序列与力偏差边界','', f'六个新工作点共得到 {len(trial_rows)} 条逐试次记录，其中 {len(assess)} 条对相应约束具有所需类别，{failed} 条发生试次级迁移失败；缺少static或gross类别的记录显式标记不可评价，没有计作成功。稳定负例覆盖、static分数分布及序列四等分位置见 `static_score_distribution.csv` 和 `distribution_shift_summary.csv`。', '', '逐运行误告警起点、static告警占比、含左删失gross段覆盖，以及非左删失事件召回/延迟见 `sequence_metrics.csv`；固定规则代表曲线覆盖每个模型的高误报例和低召回例。本轮raw工作点不替换第五轮已经冻结的sequential确认/滞回规则；原规则逐项保存在 `r5_operating_points.csv`。','', '图像几何和背景不能从冻结概率文件恢复，本轮没有重新编码图像；只绑定既有Round-4图像审计作为上下文证据，不把分数分布关联解释成图像因果。', '', '力偏差只按fold×seed汇总层与F-old/F-adapt迁移量关联。force与slip不是同批试次，没有按同名试次配对；n=12且四折重叠，相关系数只作后续采集假设，不支持物理因果。','', '## 结论边界','', '本工作包回答阈值可迁移性和排序包络，不证明独立盲测、真实GSmini部署、稳定提前预警或轻量世界模型。validation参与历史checkpoint选择，描述性曲线尤其不能用于再次选模型。','']
 p=Path(output)/'SUMMARY_ZH.md';tmp=p.with_name(p.name+f'.tmp.{os.getpid()}');tmp.write_text('\n'.join(lines));os.replace(tmp,p)

def run(a):
 protocol=HERE/'PROTOCOL.md'; invp=a.inventory.resolve();r5root=a.r5_root.resolve();out=a.output.resolve();testp=a.test_results.resolve();tests=json.loads(testp.read_text())
 if tests.get('status')!='pass':raise ValueError('tests evidence is not pass')
 inv=json.loads(invp.read_text());jobs=discover(inv)
 triple={(argv_value(j['argv'],'--manifest'),argv_value(j['argv'],'--contract'),argv_value(j['argv'],'--cache-audit')) for j in jobs.values()}
 if len(triple)!=1:raise ValueError('authoritative chains differ')
 mp,cp,ap=map(Path,next(iter(triple)));manifest=json.loads(mp.read_text());runs,sources=load_runs(jobs,manifest,cp,ap)
 amendment1=HERE/'AMENDMENT_01.md';amendment2=HERE/'AMENDMENT_02.md';sources.update({str(invp):sha(invp),str(protocol):sha(protocol),str(amendment1):sha(amendment1),str(amendment2):sha(amendment2),str(testp):sha(testp),str(mp):sha(mp),str(cp):sha(cp),str(ap):sha(ap)})
 for contextual in (Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round4_comprehensive/diagnostics/image_audit.json'),Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round4_comprehensive/diagnostics/REPORT_ZH.md')):
  if contextual.is_file():sources[str(contextual)]=sha(contextual)
 operational,descriptive,curves,sequences=make_points(runs);trialrows,score_distributions,distribution_shifts=trial_diagnostics(runs,operational,manifest);representatives=representative_failures(runs,trialrows);snapshot=r5_snapshot(runs);feats,assocobs,assocsum=force_association(operational,r5root);seedpairs,foldpairs=paired_bootstrap(runs,operational,manifest)
 for row in feats:sources[row['source_path']]=row['source_sha256']
 out.mkdir(parents=True,exist_ok=True)
 files={'operational_points.csv':operational,'validation_descriptive_envelope.csv':descriptive,'curve_points.csv':curves,'sequence_metrics.csv':sequences,'paired_seed_differences.csv':seedpairs,'paired_fold_bootstrap.csv':foldpairs,'r5_operating_points.csv':snapshot,'force_bias_features.csv':feats,'force_bias_association_observations.csv':assocobs,'force_bias_association_summary.csv':assocsum,'trial_threshold_migration.csv':trialrows,'static_score_distribution.csv':score_distributions,'distribution_shift_summary.csv':distribution_shifts,'representative_failures.csv':representatives}
 for name,rr in files.items():atomic_csv(out/name,rr)
 atomic_json(out/'INPUT_MANIFEST.json',{'status':'pass','expected_evaluations':48,'models':MODELS,'folds':FOLDS,'seeds':SEEDS,'source_hashes':sources})
 render(curves,operational,assocsum,representatives,runs,out);report(operational,foldpairs,assocsum,trialrows,out)
 outputs={}
 for p in sorted(out.rglob('*')):
  if p.is_file() and p.name not in ('summary.json',):outputs[str(p)]=sha(p)
 summary={'status':'complete','format':'round6_current_threshold_analysis_v1','expected_evaluations':48,'evaluations':len(runs),'all_expected_identities_present':set(runs)==expected(),'fixed_recall_levels':RECALL_LEVELS,'fixed_fpr_levels':FPR_LEVELS,'bootstrap_repetitions':BOOTSTRAPS,'source_hashes':sources,'output_hashes':outputs,'tests':tests,'limitations':['validation is a development role used in checkpoint selection','folds overlap','validation envelope is descriptive only','force association is fold-seed aggregate only, not trial paired or causal','no test-role or physical deployment claim']}
 atomic_json(out/'summary.json',summary);return summary

def parse():
 p=argparse.ArgumentParser();p.add_argument('--inventory',type=Path,required=True);p.add_argument('--r5-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--test-results',type=Path,required=True);return p.parse_args()
if __name__=='__main__':print(json.dumps({k:v for k,v in run(parse()).items() if k in ('status','evaluations')},indent=2))
