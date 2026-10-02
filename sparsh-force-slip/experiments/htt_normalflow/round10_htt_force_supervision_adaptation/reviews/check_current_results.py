import pathlib,json,csv,hashlib,numpy as np,math,datetime
R=pathlib.Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round10_htt_force_supervision_adaptation');C=pathlib.Path('/home/zjy/document/tactile-grasp/sparsh-force-slip/experiments/htt_normalflow/round10_htt_force_supervision_adaptation');D=R/'current_evaluation'
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(pathlib.Path(p).read_text())
def rows(p):
 with pathlib.Path(p).open() as f:return list(csv.DictReader(f))
def num(x):return float(x) if x not in ('',None) else float('nan')
def eq(a,b):return (math.isnan(a) and math.isnan(b)) or np.isclose(a,b,atol=1e-12,rtol=1e-10)
s=load(D/'summary.json');manifest=load(R/'formal_delivery/EVALUATION_MANIFEST.json');assert s['status']=='complete' and not s['synthetic'] and s['runs']==36 and s['new_runs']==12 and s['reused_runs']==24;assert sha(R/'formal_delivery/EVALUATION_MANIFEST.json')==s['manifest_sha256']
for p,h in s['input_hashes'].items():assert sha(p)==h
for p,h in s['output_hashes'].items():assert sha(D/p)==h
assert sha(C/'evaluation/evaluate.py')==s['source_sha256'] and sha(C/'evaluation/PROTOCOL.md')==s['protocol_sha256']
metrics=rows(D/'metrics.csv');lookup={(r['group'],r['fold'],int(r['seed']),r['role'],r['point'],r['rule']):r for r in metrics};assert len(lookup)==len(metrics)==1152
confusions=[];canonical={};valgroups={}
for run in manifest['runs']:
 key=(run['group'],run['fold'],run['seed']);assert len([r for r in metrics if (r['group'],r['fold'],int(r['seed']))==key])==32
 for role in ('calibration','validation'):
  pred=rows(run['predictions'][role]['path']);ep=rows(run['endpoints'][role]['path']);ids=[(r['episode_id'],int(r.get('t',r.get('frame'))),int(r['stage'])) for r in pred];other=[(r['episode_id'],int(r.get('t',r.get('frame'))),int(r['stage'])) for r in ep];assert ids==other and len(set(ids))==len(ids) and all(t>=13 for _,t,_ in ids)
  ck=(run['fold'],role)
  if ck in canonical:assert canonical[ck]==ids
  else:canonical[ck]=ids
  if role=='validation':valgroups[run['fold']]={r['leakage_group'] for r in pred}
  rr=[r for r in pred if int(r['stage']) in (0,2)];y=np.array([int(r['stage'])==2 for r in rr]);p=np.array([float(r['p_slip']) for r in rr]);predicted=p>=.5;r=lookup[(*key,role,'fixed_0.5','raw')]
  counts={'tn':int((~y&~predicted).sum()),'fp':int((~y&predicted).sum()),'fn':int((y&~predicted).sum()),'tp':int((y&predicted).sum())}
  assert all(int(r[k])==v for k,v in counts.items());assert eq(num(r['positive_prevalence']),float(y.mean()))
  if role=='validation':confusions.append({'group':run['group'],'fold':run['fold'],'seed':run['seed'],**counts})
summary=rows(D/'summary_metrics.csv');assert len(summary)==528
for r in summary:
 subset=[x for x in metrics if (x['group'],x['point'],x['rule'],x['role'])==(r['group'],r['point'],r['rule'],'validation')];assert len(subset)==int(r['total_runs'])==12
 v=np.array([num(x[r['metric']]) for x in subset]);v=v[np.isfinite(v)];assert len(v)==int(r['available_runs']);assert eq(num(r['mean']),float(v.mean()) if len(v) else float('nan'));assert eq(num(r['sd_descriptive']),float(v.std(ddof=1)) if len(v)>1 else float('nan'))
ci=rows(D/'paired_ci.csv');seedrows=rows(D/'paired_seed_differences.csv');assert len(ci)==1188
for r in ci:
 assert r['unit']=='complete_leakage_group_shared_across_seeds' and int(r['requested_bootstraps'])==200 and 0<=int(r['valid_bootstraps'])<=200 and int(r['groups'])==len(valgroups[r['fold']])
 if int(r['valid_bootstraps']):assert r['status']=='available' and np.isfinite(num(r['ci_lower'])) and num(r['ci_lower'])<=num(r['ci_upper'])
 else:assert r['status']=='undefined' and math.isnan(num(r['ci_lower'])) and math.isnan(num(r['ci_upper']))
 point='fixed_0.5' if r['point']=='ranking' else r['point'];ds=[]
 for seed in (20260914,20260915,20260916):
  cand=lookup[(r['candidate'],r['fold'],seed,'validation',point,r['rule'])];base=lookup[(r['base'],r['fold'],seed,'validation',point,r['rule'])];diff=num(cand[r['metric']])-num(base[r['metric']]);ds.append(diff)
  sr=[x for x in seedrows if all(x[k]==r[k] for k in ('fold','candidate','base','point','rule','metric')) and int(x['seed'])==seed];assert len(sr)==1 and eq(num(sr[0]['difference']),diff)
 assert eq(num(r['difference']),float(np.mean(ds)))
print(json.dumps({'status':'pass','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runs':36,'all_input_hashes_checked':len(s['input_hashes']),'all_output_hashes_checked':len(s['output_hashes']),'fixed_0p5_confusions_recomputed':36,'calibration_confusions_additionally_recomputed':36,'all72_prediction_endpoint_lists_exact_and_common':True,'summary_rows_mean_sd_recomputed':len(summary),'CI_rows_structure_and_point_differences_checked':len(ci),'seed_differences_recomputed':len(seedrows),'bootstrap_intervals_not_independently_resampled_here':True,'CI_statuses':{st:sum(r['status']==st for r in ci) for st in {r['status'] for r in ci}},'scope':'actual complete36 current models; source audit separate; no independent-test or pooled-fold claim','confusions':confusions,'summary_audit_sha256':sha(D/'summary.json'),'manifest_sha256':sha(R/'formal_delivery/EVALUATION_MANIFEST.json')}))
