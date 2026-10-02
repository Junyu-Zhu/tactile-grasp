#!/usr/bin/env python3
"""Complete-grid force diagnostics; within-fold group bootstrap shared over seeds."""
import argparse,csv,json,hashlib,math
from pathlib import Path
import numpy as np
SEEDS=(20260914,20260915,20260916)
FOLDS=tuple(f'htt_leave_p{i}' for i in range(1,5))
METRICS=('mae','rmse','bias','prediction_std','target_std','prediction_mean_abs','target_mean_abs','delta5_mae','delta5_bias','prediction_delta5_std','target_delta5_std','target_clip_fraction','prediction_outside_clip_fraction')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):
 with Path(p).open() as f:return list(csv.DictReader(f))
def csvout(p,rows):
 if not rows:raise ValueError(f'empty {p}')
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def number(v):return float(v) if v not in ('',None) else float('nan')
def avg(v,w=None):
 v=np.asarray(v,float);valid=np.isfinite(v)
 if not valid.any():return None
 return float(np.average(v[valid],weights=np.asarray(w)[valid] if w is not None else None))
def aggregate(rr,metric,weighted):
 vals=np.asarray([number(r[metric]) for r in rr]);weights=np.asarray([int(r['n']) for r in rr]) if weighted else None
 if metric=='rmse' and weighted:
  v=avg(vals**2,weights);return math.sqrt(v) if v is not None else None
 return avg(vals,weights)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--support',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 sources={str(Path(__file__)):sha(__file__)};rows=[];seen=set();support={};support_outer={};contracts={}
 for fold in FOLDS:
  sp=a.support/f'fold_p{fold[-1]}.json';sm=json.loads(sp.read_text());assert sm['status']=='pass' and sm['fold']==fold;sources[str(sp)]=sha(sp);support[fold]={e['episode_id']:e['role'] for e in sm['entries']};support_outer[fold]={e['episode_id']:e['outer_role'] for e in sm['entries']}
  cp=Path(sm['provenance']['contract']);assert sha(cp)==sm['provenance']['contract_sha256'];contracts[fold]=json.loads(cp.read_text());sources[str(cp)]=sha(cp)
 files=sorted(a.root.rglob('per_trial_axis.csv'))
 for fp in files:
  ap=fp.with_name('AUDIT.json');au=json.loads(ap.read_text());assert au['status']=='pass' and au['test_consumed'] is False and au['output_hashes']['per_trial_axis.csv']==sha(fp)
  assert au['fold'] in FOLDS and au['seed'] in SEEDS
  key=(bool(au['regression']),au['fold'],au['seed']);assert key not in seen;seen.add(key)
  sp=a.support/f'fold_p{au["fold"][-1]}.json';assert au['support_sha256']==sha(sp)
  sources[str(fp)]=sha(fp);sources[str(ap)]=sha(ap)
  data=read(fp);identities=set();assert len(data)==au['rows'] and len({r['episode_id'] for r in data})==au['trials']
  allowed={e['episode_id']:e['roles_by_fold'][au['fold']] for e in contracts[au['fold']]['entries'] if e['task']=='force' and e['roles_by_fold'][au['fold']] in ('train','validation','calibration')} if au['regression'] else support_outer[au['fold']]
  ce={e['episode_id']:e for e in contracts[au['fold']]['entries']}
  excluded={eid for eid in allowed if ce[eid]['frames']<=13};eligible={eid for eid in allowed if ce[eid]['frames']>13}
  assert au['source_trials']==len(allowed) and au['evaluated_trials']==len(eligible) and au['trials']==len(eligible)
  assert {x['episode_id'] for x in au['excluded_no_common_endpoint']}==excluded
  assert all(x['frames']==ce[x['episode_id']]['frames'] and x['role']==allowed[x['episode_id']] and x['reason']=='no_common_endpoint_t_ge13' for x in au['excluded_no_common_endpoint'])
  assert {r['episode_id'] for r in data}==eligible
  for r in data:
   assert r['fold']==au['fold'] and int(r['seed'])==au['seed'] and r['role'] in ('train','validation','calibration') and r['axis'] in ('shear_x','shear_y','normal') and r['variant'] in ('old','new')
   assert r['task']==('old_force_regression' if au['regression'] else 'slip_force')
   ident=(r['variant'],r['episode_id'],r['population'],r['axis']);assert ident not in identities;identities.add(ident)
   assert r['role']==allowed[r['episode_id']]
   r['outer_role']=r['role'];r['role']=('legacy_force_train' if r['role']=='train' else r['role']) if au['regression'] else support[r['fold']][r['episode_id']]
   assert int(r['n'])>0 and all(np.isfinite(number(r[k])) for k in METRICS if k!='target_clip_fraction');rows.append(r)
  for prefix in {z[:3] for z in identities}:assert {z[3] for z in identities if z[:3]==prefix}=={'shear_x','shear_y','normal'}
  for ident in identities:assert (('new' if ident[0]=='old' else 'old'),*ident[1:]) in identities
 expected={(reg,f,s) for reg in (False,True) for f in FOLDS for s in SEEDS};assert seen==expected,(seen^expected)
 # Per-run macro averages and frame-weighted summaries: the latter's std columns
 # are weighted means of within-trial std, never a pooled standard deviation.
 keys=('task','fold','seed','role','variant','population','axis');groups={}
 for r in rows:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
 summaries=[]
 for key,rr in sorted(groups.items()):
  for weighted,label in [(False,'complete_trial_macro'),(True,'frame_weighted')]:
   summaries.append(dict(zip(keys,key))|dict(aggregation=label,trials=len(rr),frames=sum(int(r['n']) for r in rr),**{m:aggregate(rr,m,weighted) for m in METRICS}))
 csvout(a.output/'per_run_summary.csv',summaries)
 # Compare only paired same-trial force observations, retain each seed before averaging.
 paired=[];lookup={(r['task'],r['fold'],int(r['seed']),r['role'],r['episode_id'],r['population'],r['axis'],r['variant']):r for r in rows}
 for key,r in sorted(lookup.items()):
  if key[-1]!='old':continue
  nr=lookup[(*key[:-1],'new')];assert nr['leakage_group']==r['leakage_group'] and nr['n']==r['n']
  paired.append(dict(zip(('task','fold','seed','role','episode_id','population','axis'),key[:-1]))|dict(leakage_group=r['leakage_group'],n=int(r['n']),**{m+'_new_minus_old':number(nr[m])-number(r[m]) for m in METRICS}))
 csvout(a.output/'per_seed_trial_differences.csv',paired)
 ci=[];summarized=[];ci_keys=('task','fold','role','population','axis');blocks={}
 for r in paired:blocks.setdefault(tuple(r[k] for k in ci_keys),[]).append(r)
 for key,rr in sorted(blocks.items()):
  episodes=sorted({r['episode_id'] for r in rr});leakmap={r['episode_id']:r['leakage_group'] for r in rr};leakgroups=sorted(set(leakmap.values()))
  byep={ep:[r for r in rr if r['episode_id']==ep] for ep in episodes};assert all({r['seed'] for r in er}==set(SEEDS) and len(er)==3 for er in byep.values())
  # Seed means occur within each complete trial before group resampling.
  values={m:np.array([np.mean([r[m+'_new_minus_old'] for r in byep[ep]]) for ep in episodes]) for m in METRICS}
  rng=np.random.default_rng(20260916+FOLDS.index(key[1]));draws=rng.integers(0,len(leakgroups),(200,len(leakgroups)));groupix=np.array([leakgroups.index(leakmap[ep]) for ep in episodes]);weights=[np.bincount(draw,minlength=len(leakgroups))[groupix] for draw in draws]
  for m,v in values.items():
   point=avg(v);bootstrap=[avg(v,w) for w in weights];valid=[x for x in bootstrap if x is not None and np.isfinite(x)]
   ci.append(dict(zip(ci_keys,key))|dict(metric=m,difference_new_minus_old=point,ci_lower=float(np.quantile(valid,.025)) if valid else None,ci_upper=float(np.quantile(valid,.975)) if valid else None,trials=len(episodes),leakage_groups=len(leakgroups),seeds=3,requested_bootstraps=200,valid_bootstraps=len(valid),unit='complete_trial_macro_seedmean_shared_leakage_group_resampling',status='available' if valid else 'undefined'))
  for ep in episodes:summarized.append(dict(zip(ci_keys,key))|dict(episode_id=ep,leakage_group=leakmap[ep],seeds=3,**{m+'_new_minus_old':float(values[m][episodes.index(ep)]) for m in METRICS}))
 csvout(a.output/'paired_group_ci.csv',ci);csvout(a.output/'seedmean_trial_differences.csv',summarized)
 audit={'status':'pass','formal_run_pairs':24,'slip_force_pairs':12,'old_force_regression_pairs':12,'source_hashes':sources,'input_rows':len(rows),'output_hashes':{p.name:sha(p) for p in a.output.glob('*.csv')},'bootstrap':{'count':200,'seed_rule':'20260916 + zero-based fold index','unit':'complete leakage group within task/fold/role/population/axis; all three seeds shared','overlapping_folds_pooled':False},'limits':['No new force fitting uses outer validation/calibration; inherited R5 checkpoint had historical validation selection','Frame-weighted std entries are means of within-trial std, not pooled time-series std','RMSE frame-weighted is sqrt(weighted per-trial MSE); trial macro RMSE is arithmetic mean','HTT labels partly force-rule-derived; no causal or real-world claim','All physical errors retain separate native axes; no cross-coordinate pooled error'],'missing_fit_assignment':False}
 (a.output/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps({'status':'pass','rows':len(rows),'ci':len(ci)}))
if __name__=='__main__':main()
