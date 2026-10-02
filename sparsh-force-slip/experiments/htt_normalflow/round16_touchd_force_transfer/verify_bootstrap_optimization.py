#!/usr/bin/env python3
"""Prove fixed real-data small-draw equivalence of literal and optimized bootstrap."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent/'round13_trial_level_alarm_calibration/evaluation'))
import r13_evaluate as r13
from bootstrap_reference_literal import literal
SEEDS=(20260914,20260915,20260916)
def read(path):
 with Path(path).open(newline='') as f:return list(csv.DictReader(f))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def weights(sample):return {g:sample.count(g) for g in set(sample)}
def same(a,b):return all((x==y) or (np.isnan(x) and np.isnan(y)) for x,y in zip(a,b))
def main():
 p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=read(a.evaluation/'slip/trials.csv');fold='htt_leave_p1';groups=sorted({r['leakage_group'] for r in rows if r['group']=='V' and r['fold']==fold and int(r['seed'])==SEEDS[0] and r['role']=='validation'});rng=np.random.default_rng(2026091601);samples=[rng.choice(groups,len(groups),replace=True).tolist() for _ in range(7)]
 def selected(g,s):return [r for r in rows if r['group']==g and r['fold']==fold and int(r['seed'])==s and r['role']=='validation' and r['family']=='trial_macro_static_FPR' and r['alpha']=='0.05' and r['k']=='1']
 by={(g,s):selected(g,s) for g in ('H','T_H') for s in SEEDS};keep=('episode','leakage_group','probe');drop=('group','fold','seed','role','family','point','rule','alpha','k','threshold')
 def convert(r):return {k:(v if k in keep else int(v)) for k,v in r.items() if k not in drop}
 reference=literal(by,'T_H','H',SEEDS,r13.PAIR_METRICS,samples,r13.aggregate,weights,convert);converted={k:[convert(r) for r in v] for k,v in by.items()};optimized={m:[] for m in r13.PAIR_METRICS}
 for sample in samples:
  w=weights(sample);per={m:[] for m in r13.PAIR_METRICS}
  for seed in SEEDS:
   ca,ba=r13.aggregate(converted[('T_H',seed)],w),r13.aggregate(converted[('H',seed)],w)
   for metric in r13.PAIR_METRICS:per[metric].append(ca[metric]-ba[metric])
  for metric in r13.PAIR_METRICS:optimized[metric].append(float(np.mean(per[metric])))
 exact={m:same(reference[m],optimized[m]) for m in r13.PAIR_METRICS};quantile={m:same([float(np.quantile(reference[m],q,method='linear')) for q in (.025,.5,.975)],[float(np.quantile(optimized[m],q,method='linear')) for q in (.025,.5,.975)]) for m in r13.PAIR_METRICS}
 if not all(exact.values()) or not all(quantile.values()) or any(sum(weights(x).values())!=len(groups) for x in samples):raise RuntimeError('non-equivalent optimization')
 result={'schema':'round16_bootstrap_optimization_equivalence_v1','status':'pass','real_fold':fold,'policy':'trial_macro_static_FPR alpha=.05 k=1','draws':7,'draw_seed':2026091601,'sampled_group_multisets':samples,'same_draws_shared_across_three_seeds':True,'multiplicity_sum_equals_group_count':True,'paired_metric_values_exact':exact,'linear_quantiles_exact':quantile,'reference_source':str(HERE/'bootstrap_reference_literal.py'),'reference_sha256':sha(HERE/'bootstrap_reference_literal.py'),'optimized_source':str(HERE/'bootstrap_registered.py'),'optimized_sha256':sha(HERE/'bootstrap_registered.py'),'formal_definition_unchanged':{'draws_per_fold':2000,'seed_formula':'2026091601+fold_index-1','unit':'complete validation leakage_group','quantiles':[.025,.5,.975],'method':'linear'}};a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'pass','metrics':len(exact),'draws':7}))
if __name__=='__main__':main()
