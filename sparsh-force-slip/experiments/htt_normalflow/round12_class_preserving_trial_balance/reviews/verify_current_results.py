import csv,json,hashlib,math
from pathlib import Path
import numpy as np
O=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round12_class_preserving_trial_balance');E=O/'current_evaluation';M=O/'formal_delivery/EVALUATION_MANIFEST.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return list(csv.DictReader(Path(p).open()))
def near(a,b):
 if a in ('',None):return not np.isfinite(b)
 return math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-9)
s=json.loads((E/'summary.json').read_text());assert s['status']=='complete' and (s['runs'],s['new_runs'],s['reused_runs'])==(72,24,48) and not s['synthetic'];assert s['manifest_sha256']==sha(M)
for p,h in s['input_hashes'].items():assert sha(p)==h
for n,h in s['output_hashes'].items():assert sha(E/n)==h
m=json.loads(M.read_text());metrics=rows(E/'metrics.csv');tails=rows(E/'static_tail_distribution.csv');migration=rows(E/'threshold_migration.csv');confusions=thresholdchecks=tailchecks=0
by={(r['group'],r['fold'],int(r['seed']),r['role'],r['point'],r['rule']):r for r in metrics};assert len(by)==len(metrics)==3456
for run in m['runs']:
 key=(run['group'],run['fold'],run['seed']);roledata={}
 for role in ('train','calibration','validation'):
  rr=rows(run['predictions'][role]['path']);ep=rows(run['endpoints'][role]['path']);assert [(r['episode_id'],r['t'],r['stage'],r['role']) for r in rr]==[(r['episode_id'],r['t'],r['stage'],r['role']) for r in ep]
  y=np.array([int(r['stage']) for r in rr]);p=np.array([float(r['p_slip']) for r in rr]);roledata[role]=(y,p);assert np.isfinite(p).all()
  tp=int(((y==2)&(p>=.5)).sum());fp=int(((y==0)&(p>=.5)).sum());tn=int(((y==0)&(p<.5)).sum());fn=int(((y==2)&(p<.5)).sum());mr=by[(*key,role,'fixed_0.5','raw')];assert all(int(float(mr[n]))==v for n,v in [('tp',tp),('fp',fp),('tn',tn),('fn',fn)]);confusions+=1
  tt=next(r for r in tails if (r['group'],r['fold'],int(r['seed']),r['role'])==(*key,role));st=p[y==0]
  for q in (.5,.9,.95,.99,1.):assert near(tt[f'q{int(q*100):02d}'],np.quantile(st,q));tailchecks+=1
  for r in migration:
   if (r['group'],r['fold'],int(r['seed']),r['role'])==(*key,role):assert near(r['static_fpr'],np.mean(st>=float(r['threshold'])))
 y,p=roledata['calibration'];thresholds=np.r_[np.nextafter(1.,np.inf),np.unique(p)[::-1]];c=[]
 for t in thresholds:
  f=float(np.mean(p[y==0]>=t));rec=float(np.mean(p[y==2]>=t));c.append((t,f,rec,(1-f+rec)/2))
 selected={'fixed_0.5':.5,'maxBA':max(c,key=lambda x:(x[3],-x[1],x[0]))[0]}
 for f in (.01,.05,.1):selected[f'FPR{f:.2f}']=max((x for x in c if x[1]<=f+1e-15),key=lambda x:(x[2],-x[1],x[0]))[0]
 for r in (.8,.9,.95):selected[f'recall{r:.2f}']=max((x for x in c if x[2]>=r-1e-15),key=lambda x:(-x[1],x[2],x[0]))[0]
 for point,t in selected.items():
  for role in ('train','calibration','validation'):
   for rule in ('raw','confirm2'):assert near(by[(*key,role,point,rule)]['threshold'],t);thresholdchecks+=1
summary=rows(E/'summary_metrics.csv');meanchecks=0
for r in summary:
 rr=[x for x in metrics if (x['group'],x['point'],x['rule'],x['role'])==(r['group'],r['point'],r['rule'],'validation')];v=np.array([float(x[r['metric']]) if x[r['metric']] else np.nan for x in rr]);v=v[np.isfinite(v)];assert len(rr)==12 and int(r['available_runs'])==len(v);assert near(r['mean'],v.mean() if len(v) else np.nan);assert near(r['sd_descriptive'],v.std(ddof=1) if len(v)>1 else np.nan);meanchecks+=1
sd=rows(E/'paired_seed_differences.csv');ci=rows(E/'paired_ci.csv');dchecks=0
for r in sd:
 point='fixed_0.5' if r['point']=='ranking' else r['point'];rule=r['rule'];seed=int(r['seed']);a=by[(r['candidate'],r['fold'],seed,'validation',point,rule)];b=by[(r['base'],r['fold'],seed,'validation',point,rule)];av=float(a[r['metric']]) if a[r['metric']] else np.nan;bv=float(b[r['metric']]) if b[r['metric']] else np.nan;assert near(r['difference'],av-bv);dchecks+=1
for r in ci:
 rr=[x for x in sd if all(x[k]==r[k] for k in ('fold','candidate','base','point','rule','metric'))];assert len(rr)==3;v=np.array([float(x['difference']) if x['difference'] else np.nan for x in rr]);assert near(r['difference'],v.mean());assert 0<=int(r['valid_bootstraps'])<=int(r['requested_bootstraps'])==200
 if r['status']=='available':assert float(r['ci_lower'])<=float(r['ci_upper']) and int(r['valid_bootstraps'])>0
out=dict(status='pass',reviewer='r10_eval independent of R12 evaluator author',input_hash_count=len(s['input_hashes']),output_hash_count=len(s['output_hashes']),model_count=72,confusions_recomputed=confusions,calibration_threshold_applications_recomputed=thresholdchecks,tail_quantiles_recomputed=tailchecks,summary_mean_sd_recomputed=meanchecks,seed_differences_recomputed=dchecks,ci_structure_and_point_checks=len(ci),bootstrap_distributions_rerun=False,scope='all source/output hashes, all fixed0.5 raw confusion matrices, exact independent calibration thresholds, mean/SD, seed differences and CI structure/point values; CI bootstrap distribution not repeated',summary_sha256=sha(E/'summary.json'),verifier_source_sha256=sha(__file__))
(O/'reviews').mkdir(exist_ok=True);(O/'reviews/INDEPENDENT_CURRENT_RESULTS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
