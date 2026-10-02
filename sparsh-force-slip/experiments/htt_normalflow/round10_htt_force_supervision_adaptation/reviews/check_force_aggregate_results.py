import pathlib,json,csv,hashlib,numpy as np,math,datetime,collections
R=pathlib.Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round10_htt_force_supervision_adaptation');D=R/'force_aggregate'
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(pathlib.Path(p).read_text())
def read(p):
 with pathlib.Path(p).open() as f:return list(csv.DictReader(f))
def num(v):return float(v) if v not in ('',None) else float('nan')
def eq(a,b):return (math.isnan(a) and math.isnan(b)) or np.isclose(a,b,rtol=1e-10,atol=1e-12)
a=load(D/'AUDIT.json');assert a['status']=='pass' and a['formal_run_pairs']==24
for p,h in a['source_hashes'].items():assert sha(p)==h
for p,h in a['output_hashes'].items():assert sha(D/p)==h
support={};raw=[];grid=set();groups=collections.defaultdict(list)
for f in range(1,5):m=load(R/f'force_support/fold_p{f}.json');support[m['fold']]={e['episode_id']:e for e in m['entries']}
for p in a['source_hashes']:
 if not p.endswith('per_trial_axis.csv'):continue
 au=load(pathlib.Path(p).with_name('AUDIT.json'));grid.add((au['regression'],au['fold'],au['seed']));rr=read(p);assert len(rr)==au['rows'] and len({r['episode_id'] for r in rr})==au['evaluated_trials']
 for r in rr:
  if r['task']=='slip_force':r['role']=support[r['fold']][r['episode_id']]['role']
  elif r['role']=='train':r['role']='legacy_force_train'
  raw.append(r);groups[tuple(r[k] for k in ('task','fold','seed','role','variant','population','axis'))].append(r)
assert len(raw)==a['input_rows']==27054 and grid=={(b,f'htt_leave_p{f}',s) for b in [False,True] for f in range(1,5) for s in [20260914,20260915,20260916]}
summ=read(D/'per_run_summary.csv');metrics=[k for k in summ[0] if k not in ('task','fold','seed','role','variant','population','axis','aggregation','trials','frames')]
for r in summ:
 rr=groups[tuple(r[k] for k in ('task','fold','seed','role','variant','population','axis'))];assert len(rr)==int(r['trials']) and sum(int(x['n']) for x in rr)==int(r['frames']);weighted=r['aggregation']=='frame_weighted'
 for metric in metrics:
  v=np.array([num(x[metric]) for x in rr]);w=np.array([int(x['n']) for x in rr],float) if weighted else np.ones(len(rr));mask=np.isfinite(v)
  val=(np.sqrt(np.average(v[mask]**2,weights=w[mask])) if metric=='rmse' and weighted else np.average(v[mask],weights=w[mask])) if mask.any() else float('nan');assert eq(num(r[metric]),float(val)),(r,metric,val)
lookup={tuple(r[k] for k in ('task','fold','seed','role','episode_id','population','axis','variant')):r for r in raw};diffs=read(D/'per_seed_trial_differences.csv');epgroups=collections.defaultdict(list)
for r in diffs:
 key=tuple(r[k] for k in ('task','fold','seed','role','episode_id','population','axis'));old=lookup[(*key,'old')];new=lookup[(*key,'new')];assert old['leakage_group']==new['leakage_group']==r['leakage_group']
 for metric in metrics:assert eq(num(r[metric+'_new_minus_old']),num(new[metric])-num(old[metric]))
 epgroups[tuple(r[k] for k in ('task','fold','role','episode_id','population','axis'))].append(r)
means=read(D/'seedmean_trial_differences.csv');blocks=collections.defaultdict(list)
for r in means:
 rr=epgroups[tuple(r[k] for k in ('task','fold','role','episode_id','population','axis'))];assert len(rr)==3 and {x['seed'] for x in rr}=={'20260914','20260915','20260916'}
 for metric in metrics:assert eq(num(r[metric+'_new_minus_old']),float(np.mean([num(x[metric+'_new_minus_old']) for x in rr])))
 blocks[tuple(r[k] for k in ('task','fold','role','population','axis'))].append(r)
cis=read(D/'paired_group_ci.csv')
for r in cis:
 rr=blocks[tuple(r[k] for k in ('task','fold','role','population','axis'))];assert int(r['trials'])==len(rr) and int(r['seeds'])==3 and int(r['requested_bootstraps'])==200;assert int(r['leakage_groups'])==len({x['leakage_group'] for x in rr})
 v=np.array([num(x[r['metric']+'_new_minus_old']) for x in rr]);v=v[np.isfinite(v)];assert eq(num(r['difference_new_minus_old']),float(v.mean()) if len(v) else float('nan'))
 assert 0<=int(r['valid_bootstraps'])<=200
 if int(r['valid_bootstraps']):assert np.isfinite(num(r['ci_lower'])) and num(r['ci_lower'])<=num(r['ci_upper']) and r['status']=='available'
 else:assert r['status']=='undefined'
print(json.dumps({'status':'pass','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_hashes_checked':len(a['source_hashes']),'output_hashes_checked':len(a['output_hashes']),'all24_grid':True,'input_rows':len(raw),'all_per_run_summary_fields_recomputed':len(summ),'all_seed_trial_differences_recomputed':len(diffs),'all_seedmean_trial_differences_recomputed':len(means),'CI_point_and_structure_checked':len(cis),'CI_interval_resampling_not_repeated':True,'source_audit_sha256':sha(D/'AUDIT.json'),'no_independent_fold_pooling_claim':True}))
