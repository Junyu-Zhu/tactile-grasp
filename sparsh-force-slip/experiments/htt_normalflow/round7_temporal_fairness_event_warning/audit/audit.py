#!/usr/bin/env python3
"""Audit effective input history, support, labels, roles and background provenance."""
from __future__ import annotations
import argparse,hashlib,json,os,pickle
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;PROTOCOL=HERE/'protocol.json';AMENDMENT=HERE/'PROTOCOL_AMENDMENT.json';HORIZONS=(1,3,5);ROLES=('fit_train','selection','calibration','outer')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def ident(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+f'.tmp.{os.getpid()}');q.write_text(json.dumps(d,indent=2,ensure_ascii=False));os.replace(q,p)
def group(meta):return f"source/{meta['dataset']}/{meta['trajectory']}"
def load_cache(path,split):
 d=torch.load(path,map_location='cpu',weights_only=False)
 if d.get('schema')!='round5_source_pred_delta_v1' or d.get('split')!=split or d.get('horizons')!=[1,3,5]:raise ValueError('bad R5 cache')
 if not all(torch.isfinite(d[k].float()).all() for k in ('z','p_slip','force_features','predicted_delta_force_xyz','future_slip')):raise ValueError('nonfinite cache')
 parent=Path(d['source_path'])
 if sha(parent)!=d['source_sha256']:raise ValueError('parent raw cache drift')
 return d,torch.load(parent,map_location='cpu',weights_only=False)
def trial_summary(trials):
 pos=sorted({x['leakage_group'] for x in trials if x['positive_endpoints']});neg=sorted({x['leakage_group'] for x in trials if x['negative_endpoints']})
 endpoints=sorted((x['episode_id'],t,1) for x in trials for t in x['positive_endpoints'])+sorted((x['episode_id'],t,0) for x in trials for t in x['negative_endpoints'])
 return {'positive_event_trials':len(pos),'negative_window_trials':len(neg),'positive_endpoints':sum(len(x['positive_endpoints']) for x in trials),'negative_endpoints':sum(len(x['negative_endpoints']) for x in trials),'positive_groups':pos,'negative_groups':neg,'endpoint_identity_sha256':ident(sorted(endpoints))}
def main():
 p=argparse.ArgumentParser();p.add_argument('--r6-source-manifest',type=Path,required=True);p.add_argument('--train-cache',type=Path,required=True);p.add_argument('--val-cache',type=Path,required=True);p.add_argument('--derived-root',type=Path,required=True);p.add_argument('--generation-code',type=Path,required=True);p.add_argument('--dataset-code',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output
 proto=json.loads(PROTOCOL.read_text());amend=json.loads(AMENDMENT.read_text())
 if amend['locked_original_protocol_sha256']!=sha(PROTOCOL):raise ValueError('protocol lock drift')
 r6=json.loads(a.r6_source_manifest.read_text());role_by_group={g:r for r in ROLES for g in r6['roles'][r]['leakage_groups']}
 train,parent_train=load_cache(a.train_cache,'train');val,parent_val=load_cache(a.val_cache,'val')
 caches=[(train,parent_train),(val,parent_val)];dataset_payload={};background=[]
 datasets=sorted({m['dataset'] for cache,_ in caches for m in cache['metadata']})
 for name in datasets:
  path=a.derived_root/name/'dataset_slip_forces.pkl';d=pickle.load(path.open('rb'));dataset_payload[name]=d
  bg=int(np.flatnonzero(np.asarray(d['in_contact'])==0)[0]);mins=[int(np.min(np.asarray(v['indexes']))) for v in d['trajectories'].values() if len(v['indexes'])]
  background.append({'dataset':name,'metadata_path':str(path),'metadata_sha256':sha(path),'first_global_noncontact_frame':bg,'minimum_trajectory_frame':min(mins),'reference_precedes_every_trajectory_frame':bg<=min(mins)})
 rows_by_group={};cache_lookup={};parent_lookup={}
 for cache,parent in caches:
  if len(cache['metadata'])!=len(cache['z']):raise ValueError('cache metadata length')
  if len(parent['metadata'])!=len(parent['current_slip']):raise ValueError('parent metadata length')
  for j,m in enumerate(parent['metadata']):
   key=(group(m),int(m['sample']))
   if key in parent_lookup:raise ValueError('duplicate parent cache row')
   parent_lookup[key]=(parent,j)
  for i,m in enumerate(cache['metadata']):
   key=(group(m),int(m['sample']))
   if key in cache_lookup:raise ValueError('duplicate cache row')
   cache_lookup[key]=(cache,i)
  grouped=defaultdict(list)
  for i,m in enumerate(cache['metadata']):grouped[group(m)].append((int(m['sample']),i,m['dataset'],str(m['trajectory'])))
  for g,items in grouped.items():
   if g not in role_by_group:raise ValueError(f'group absent from R6 roles {g}')
   rows_by_group[g]={'cache':cache,'items':sorted(items),'role':role_by_group[g]}
 role_sets={r:{g for g,v in rows_by_group.items() if v['role']==r} for r in ROLES}
 for i,r in enumerate(ROLES):
  for q in ROLES[:i]:
   if role_sets[r]&role_sets[q]:raise ValueError('role leakage')
 trials_by_h={h:{r:[] for r in ROLES} for h in HORIZONS};timeline={r:[] for r in ROLES};target_mismatch=0;current_mismatch=0;dep_examples=[]
 for g,row in sorted(rows_by_group.items()):
  pos={t:i for t,i,_,_ in row['items']};dataset=row['items'][0][2];traj_key=row['items'][0][3];raw=dataset_payload[dataset]['trajectories'];traj=raw[int(traj_key)] if int(traj_key) in raw else raw[traj_key];labels=np.asarray(traj['slip_label'],dtype=np.int8);onsets=np.flatnonzero(labels==1);onset=None if not len(onsets) else int(onsets[0]);per_h={h:{'positive':[],'negative':[],'right_censored':0} for h in HORIZONS}
  for t,i,_,_ in row['items']:
   parent,parent_i=parent_lookup[(g,t)]
   current_mismatch+=int(int(parent['current_slip'][parent_i])!=int(labels[t]))
   history=list(range(t-8,t+1));history_ok=t>=18 and all((g,s) in cache_lookup for s in history)
   if not history_ok:continue
   raw_union_base=sorted({u for s in history for u in (s-5,s)});raw_union_aux=sorted({u for s in range(t-3,t+1) for r in (s,s-5) for u in (r-5,r)})
   if raw_union_base!=list(range(t-13,t+1)) or min(raw_union_aux)<t-13 or max(raw_union_aux)>t:raise ValueError('effective history mismatch')
   masks=[];targets=[];right=[];future_any=[]
   for hi,h in enumerate(HORIZONS):
    valid=t+h<len(labels);masks.append(valid);right.append(not valid)
    raw_future=None if not valid else int(np.asarray(labels[t+1:t+h+1]).sum()>0);future_any.append(raw_future)
    prestatic=bool(labels[t]==0 and (onset is None or t<onset))
    target=None if not (valid and prestatic) else int(onset is not None and t<onset<=t+h);targets.append(target)
    if valid:
     stored=int(row['cache']['future_slip'][i,hi]>=.5);target_mismatch+=int(stored!=raw_future)
     if prestatic:
      if raw_future!=target:raise ValueError('future-any and first-onset disagree before first onset')
      (per_h[h]['positive'] if target else per_h[h]['negative']).append(t)
    else:per_h[h]['right_censored']+=1
   timeline[row['role']].append({'episode_id':g,'leakage_group':g,'t':t,'current_slip':int(labels[t]),'first_onset_t':onset,'history_times':history,'future_any_slip_labels':future_any,'targets':targets,'horizon_mask':[bool(m and prestatic) for m in masks],'future_window_complete':masks,'right_censored':right,'common_complete':bool(all(masks) and prestatic),'pre_first_onset_static':prestatic})
   if len(dep_examples)<20:dep_examples.append({'episode_id':g,'t':t,'base_output_times':history,'base_raw_frame_union':raw_union_base,'aux_output_times':list(range(t-3,t+1)),'aux_raw_frame_union':raw_union_aux})
  for h in HORIZONS:
   trials_by_h[h][row['role']].append({'episode_id':g,'leakage_group':g,'first_onset_t':onset,'positive_endpoints':per_h[h]['positive'],'negative_endpoints':per_h[h]['negative'],'right_censored_candidates':per_h[h]['right_censored']})
 if target_mismatch:raise ValueError(f'future-any target mismatches {target_mismatch}')
 if current_mismatch:raise ValueError(f'current-slip target mismatches {current_mismatch}')
 thresholds=proto['support_thresholds'];horizon_results={};supported=[]
 for h in HORIZONS:
  roles={}
  for role in ROLES:
   summary=trial_summary(trials_by_h[h][role]);missing=[f'{k}={summary[k]}<{v}' for k,v in thresholds[role].items() if summary[k]<v];roles[role]={'summary':summary,'threshold':thresholds[role],'pass':not missing,'missing':missing,'trials':trials_by_h[h][role]}
  ok=all(x['pass'] for x in roles.values());horizon_results[str(h)]={'pass':ok,'roles':roles};supported+=([h] if ok else [])
 common_by_role={r:[x for x in timeline[r] if x['common_complete'] and x['pre_first_onset_static']] for r in ROLES}
 common_counts={r:{'endpoints':len(v),'positive_by_horizon':{str(h):sum(x['targets'][i]==1 for x in v) for i,h in enumerate(HORIZONS)},'identity_sha256':ident([(x['episode_id'],x['t'],x['targets']) for x in v])} for r,v in common_by_role.items()}
 history={'format':'round7_history_dependency_audit_v1','status':'complete','protocol_sha256':sha(PROTOCOL),'amendment_sha256':sha(AMENDMENT),'source_code':{'generation':{'path':str(a.generation_code),'sha256':sha(a.generation_code)},'dataset':{'path':str(a.dataset_code),'sha256':sha(a.dataset_code)}},'dependencies':[{'feature':'z(s), pSlip(s), predForce(s)','upstream_output_times':['s'],'raw_image_times':['s-5','s'],'earliest_relative_to_endpoint_for_history9':-13,'latest':0},{'feature':'predDelta(s), latest4 only','upstream_output_times':['s-5','s'],'raw_image_times':['s-10','s-5','s'],'earliest_relative_to_endpoint':-13,'latest':0},{'feature':'visualDelta(s), latest4 only','upstream_output_times':['s-5','s'],'raw_image_times':['s-10','s-5','s'],'earliest_relative_to_endpoint':-13,'latest':0}],'mathematical_raw_frame_minimum_relative_to_endpoint':-13,'actual_cache_minimum_sample':min(int(m['sample']) for m in train['metadata']+val['metadata']),'effective_endpoint_minimum':18,'minimum_t_interpretation':'t=13 is only the mathematical raw-image dependency bound; the audited R5-v2 feature cache begins at s=10, so an exact nine-output history requires t>=18 and no t=13 endpoint is claimed available','examples':dep_examples,'background_reference':{'selection':'first global in_contact==0 frame independently within each dataset','uses_labels_or_future_force':False,'all_references_precede_trajectory_frames':all(x['reference_precedes_every_trajectory_frame'] for x in background),'datasets':background,'interpretation':'when the preceding check is true this is a fixed pre-contact reference available before each audited trajectory; deployment still must acquire an equivalent pre-contact reference before inference'},'future_any_target_mismatches':target_mismatch,'current_slip_target_mismatches':current_mismatch,'test_or_global_heldout_read':False}
 support={'format':'round7_multiwindow_support_manifest_v1','status':'complete','protocol_sha256':sha(PROTOCOL),'amendment_sha256':sha(AMENDMENT),'r6_source_manifest':{'path':str(a.r6_source_manifest),'sha256':sha(a.r6_source_manifest)},'inputs':{'train_cache':{'path':str(a.train_cache),'sha256':sha(a.train_cache)},'val_cache':{'path':str(a.val_cache),'sha256':sha(a.val_cache)}},'roles':{r:{'leakage_groups':sorted(role_sets[r]),'identity_sha256':ident(sorted(role_sets[r]))} for r in ROLES},'horizons':horizon_results,'supported_horizons':supported,'common_population':common_counts,'full_timeline':{r:{'rows':timeline[r],'identity_sha256':ident([(x['episode_id'],x['t']) for x in timeline[r]])} for r in ROLES},'target_mismatches':0,'test_or_global_heldout_read':False}
 out.mkdir(parents=True,exist_ok=True);save(out/'history_dependency_audit.json',history);save(out/'support_manifest.json',support)
 joint_supported=set(supported)==set(HORIZONS)
 decision={'format':'round7_support_decision_v1','status':'complete','training_triggered':joint_supported,'supported_horizons':supported,'D_triggered':joint_supported,'D_reason':'causal z(s)-z(s-5) is constructable at the predeclared latest-four slots on the common history' if joint_supported else 'joint H1/H3/H5 support gate failed','common_population_required':True,'effective_minimum_t':18,'reasons':[] if joint_supported else ['joint H1/H3/H5 support requires every role threshold at every horizon'],'artifacts':{'history_dependency_audit':{'path':str((out/'history_dependency_audit.json').resolve()),'sha256':sha(out/'history_dependency_audit.json')},'support_manifest':{'path':str((out/'support_manifest.json').resolve()),'sha256':sha(out/'support_manifest.json')}}}
 save(out/'support_decision.json',decision);print(json.dumps(decision,indent=2))
if __name__=='__main__':main()
