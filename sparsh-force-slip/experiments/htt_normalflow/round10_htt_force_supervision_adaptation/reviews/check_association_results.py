import pathlib,json,csv,hashlib,numpy as np,math,datetime
from scipy.stats import rankdata
R=pathlib.Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round10_htt_force_supervision_adaptation');D=R/'association'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p):
 with pathlib.Path(p).open() as f:return list(csv.DictReader(f))
def load(p):return json.loads(pathlib.Path(p).read_text())
def num(v):return float(v) if v not in ('',None) else float('nan')
def eq(x,y):return (math.isnan(x) and math.isnan(y)) or np.isclose(x,y,rtol=1e-10,atol=1e-12)
def divide(x,y):return x/y if y else float('nan')
a=load(D/'AUDIT.json');assert a['status']=='pass' and not a['causal_claim']
for p,h in a['sources'].items():assert sha(p)==h
for p,h in a['outputs'].items():assert sha(D/p)==h
pa=load(D/'figures/PLOT_AUDIT.json');assert pa['status']=='pass' and pa['cases']==4
for p,h in pa['sources'].items():assert sha(p)==h
for p,h in pa['outputs'].items():assert sha(D/'figures'/p)==h
cur={}
for r in read(R/'current_evaluation/trials.csv'):
 if r['role']=='validation' and r['point']=='FPR0.05' and r['rule']=='raw':cur[(r['fold'],r['seed'],r['episode'],r['group'])]=r
cal={(r['fold'],r['seed']):num(r['static_fpr']) for r in read(R/'current_evaluation/metrics.csv') if r['role']=='calibration' and r['point']=='FPR0.05' and r['rule']=='raw' and r['group']=='F_history_new'}
force={}
for p in a['sources']:
 if not p.endswith('per_trial_axis.csv'):continue
 for r in read(p):
  if r['task']=='slip_force' and r['role']=='validation' and r['population']=='all':force[(r['fold'],r['seed'],r['episode_id'],r['variant'],r['axis'])]=num(r['mae'])
for key in {k[:4] for k in force}:force[(*key,'mean_axes')]=np.mean([force[(*key,j)] for j in ['shear_x','shear_y','normal']])
paired=read(D/'per_seed_trial.csv');assert len(paired)==a['paired_rows']==600;expectedex=set()
for r in paired:
 key=(r['fold'],r['seed'],r['episode']);v=cur[(*key,'V_temporal')];o=cur[(*key,'F_history_old')];n=cur[(*key,'F_history_new')]
 def rates(x):return divide(num(x['fp']),num(x['fp'])+num(x['tn'])),divide(num(x['fn']),num(x['fn'])+num(x['tp']))
 vf,vm=rates(v);of,om=rates(o);nf,nm=rates(n);old=force[(*key,'old',r['axis'])];new=force[(*key,'new',r['axis'])]
 ex={'old_MAE':old,'new_MAE':new,'delta_MAE':new-old,'V_static_fpr':vf,'old_static_fpr':of,'new_static_fpr':nf,'delta_static_fpr':nf-of,'old_minus_V_fpr':of-vf,'new_minus_V_fpr':nf-vf,'delta_gross_miss':nm-om,'old_gross_miss':om,'new_gross_miss':nm,'calibration_FPR':cal[key[:2]],'threshold_transfer_gap':abs(nf-cal[key[:2]]),'delta_false_starts':num(n['false_starts'])-num(o['false_starts'])}
 for k,value in ex.items():assert eq(num(r[k]),value),(key,k)
 for label,value in [('no_static',vf),('no_gross',vm)]:
  if not np.isfinite(value):expectedex.add((*key,label))
exrows=read(D/'exclusions.csv');assert {(r['fold'],r['seed'],r['episode'],r['reason']) for r in exrows}==expectedex and len(exrows)==len(expectedex)
means=read(D/'seedmean_trial.csv');assert len(means)==a['seedmean_rows']==200
for r in means:
 rr=[x for x in paired if all(x[k]==r[k] for k in ['fold','episode','axis'])];assert len(rr)==3 and {x['seed'] for x in rr}=={'20260914','20260915','20260916'}
 for k in r:
  if k not in ('fold','episode','axis'):assert eq(num(r[k]),float(np.mean([num(x[k]) for x in rr])))
cs=read(D/'correlations.csv');assert len(cs)==a['correlations']==80
for r in cs:
 xy=np.array([[num(x[r['x']]),num(x[r['y']])] for x in means if x['fold']==r['fold'] and x['axis']==r['axis']]);xy=xy[np.isfinite(xy).all(1)];assert len(xy)==int(r['n_trials']);valid=len(xy)>=3 and len(set(xy[:,0]))>1 and len(set(xy[:,1]))>1
 value=float(np.corrcoef(rankdata(xy[:,0]),rankdata(xy[:,1]))[0,1]) if valid else float('nan');assert eq(num(r['spearman_r']),value) and r['status']==('descriptive' if valid else 'undefined')
cases=read(D/'representative_cases.csv');selected=[r for r in cases if r['selected']=='True'];assert len(selected)==4
for r in selected:
 assert int(r['rank'])==0;sr=next(x for x in means if all(x[k]==r[k] for k in ['fold','episode','axis']));assert r['axis']=='mean_axes'
 for k in sr:assert r[k]==sr[k]
 if r['kind']=='force_improved_slip_not':assert num(r['delta_MAE'])<0 and num(r['delta_static_fpr'])>=0
 if r['kind']=='lower_fpr_recall_cost':assert num(r['delta_static_fpr'])<0 and num(r['delta_gross_miss'])>0
print(json.dumps({'status':'pass','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'paired_rows_all_fields_recomputed':600,'seedmean_rows_recomputed':200,'Spearman_rank_correlation_recomputed':80,'exclusions_exact':len(exrows),'source_hashes_checked':len(a['sources']),'plot_source_hashes_checked':len(pa['sources']),'four_selected_cases_metadata_exact':True,'force_improved_count':sum(num(r['delta_MAE'])<0 for r in means if r['axis']=='mean_axes'),'fold_trial_records':50,'limitations':['force_improved_slip_not is FPR-not-decreased predicate; selected example has FPR floor0 and improved gross recall, not absence of any slip benefit','50 fold-trial records overlap across folds, not50 independent physical events','three-axis mean MAE descriptive within native HTT coordinate only; no domain-coordinate merging','no significance or causal claim'], 'association_audit_sha256':sha(D/'AUDIT.json'),'plot_audit_sha256':sha(D/'figures/PLOT_AUDIT.json')}))
