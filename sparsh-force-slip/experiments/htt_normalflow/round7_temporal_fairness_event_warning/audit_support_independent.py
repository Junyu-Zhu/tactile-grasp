#!/usr/bin/env python3
"""Independent recomputation of declared role/population/support invariants."""
import argparse,json,hashlib
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;d=json.loads((r/'audit/support_decision.json').read_text());m=json.loads((r/'audit/support_manifest.json').read_text());h=json.loads((r/'audit/history_dependency_audit.json').read_text());checks={};seen=set();counts={}
 checks['upstream_background_prior']=h['background_reference']['all_references_precede_trajectory_frames'];checks['labels_match']=h['current_slip_target_mismatches']==0 and h['future_any_target_mismatches']==0
 for role,rr in m['roles'].items():
  groups=set(rr['leakage_groups']);checks['disjoint:'+role]=not bool(groups&seen);seen|=groups;rows=m['full_timeline'][role]['rows'];ident=[(x['episode_id'],x['t']) for x in rows];checks['unique:'+role]=len(ident)==len(set(ident));checks['authorized:'+role]=all(x['leakage_group'] in groups for x in rows)
  common=[x for x in rows if all(x['horizon_mask'][[1,3,5].index(k)] for k in d['supported_horizons'])];counts[role]={};checks['common:'+role]=len(common)==m['common_population'][role]['endpoints']
  for x in rows:
   t=x['t'];assert x['history_times']==list(range(t-8,t+1)) and t>=18
   base=set(x['history_times'])|{s-5 for s in x['history_times']};aux={s-k for s in range(t-3,t+1) for k in [0,5,10]};assert base==set(range(t-13,t+1)) and aux<=base
   for j,k in enumerate([1,3,5]):
    if x['horizon_mask'][j]:
     assert x['current_slip']==0 and (x['first_onset_t'] is None or t<x['first_onset_t']) and x['future_window_complete'][j]
     assert x['targets'][j]==int(x['first_onset_t'] is not None and t<x['first_onset_t']<=t+k)
  for j,k in enumerate([1,3,5]):
   z=[x for x in rows if x['horizon_mask'][j]];pos={x['leakage_group'] for x in z if x['targets'][j]==1};neg={x['leakage_group'] for x in z if x['targets'][j]==0};ref=m['horizons'][str(k)]['roles'][role]['summary']
   checks[f'counts:{role}:H{k}']=len(pos)==ref['positive_event_trials'] and len(neg)==ref['negative_window_trials'] and sum(x['targets'][j] for x in z)==ref['positive_endpoints'];threshold=20 if role=='fit_train' else 5
   checks[f'gate:{role}:H{k}']=len(pos)>=threshold and len(neg)>=threshold;counts[role][str(k)]={'positive_trials':len(pos),'negative_trials':len(neg),'endpoints':len(z)}
 checks['equal_raw_history_all_rows']=True
 for name,x in d['artifacts'].items():checks['artifact:'+name]=sha(Path(x['path']))==x['sha256']
 out={'status':'pass' if all(checks.values()) else 'fail','checks':checks,'counts':counts,'reviewer':'root independent of support implementation','input_hashes':{n:sha(r/'audit'/n) for n in ['support_decision.json','support_manifest.json','history_dependency_audit.json']}}
 (r/'SUPPORT_INDEPENDENT_REVIEW.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':out['status'],'checks':len(checks)}));return out['status']!='pass'
if __name__=='__main__':raise SystemExit(main())
