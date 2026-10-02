#!/usr/bin/env python3
"""Registered 2,000-draw paired leakage-group bootstrap for R16 evaluations."""
import argparse,csv,json,math,sys
from collections import defaultdict
from pathlib import Path
import numpy as np,torch

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent/'round13_trial_level_alarm_calibration/evaluation'))
import r13_evaluate as r13
SEEDS=(20260914,20260915,20260916);DRAWS=2000

def read(path):
 with Path(path).open(newline='') as f:return list(csv.DictReader(f))
def write(path,rows):
 with Path(path).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def num(row,key):return float(row[key])
def ci(values):
 a=np.asarray([x for x in values if math.isfinite(x)]);return (float(np.quantile(a,.025,method='linear')),float(np.quantile(a,.5,method='linear')),float(np.quantile(a,.975,method='linear')),len(a))
def weights(sample):return {g:sample.count(g) for g in set(sample)}
def weighted(rows,metric,w):
 denominator=sum(w.get(r['leakage_group'],0)*int(r['n']) for r in rows)
 if not denominator:return float('nan')
 if metric=='rmse':return math.sqrt(sum(w.get(r['leakage_group'],0)*int(r['n'])*num(r,metric)**2 for r in rows)/denominator)
 return sum(w.get(r['leakage_group'],0)*int(r['n'])*num(r,metric) for r in rows)/denominator

def main():
 p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 slip=read(a.evaluation/'slip/trials.csv');canonical={}
 for fold in range(1,5):
  groups=sorted({r['leakage_group'] for r in slip if r['group']=='V' and r['fold']==f'htt_leave_p{fold}' and int(r['seed'])==SEEDS[0] and r['role']=='validation'})
  rng=np.random.default_rng(2026091601+fold-1);canonical[fold]=[rng.choice(groups,len(groups),replace=True).tolist() for _ in range(DRAWS)]
 (a.output/'GROUP_DRAWS.json').write_text(json.dumps(canonical,indent=2)+'\n')
 results=[]
 # Slip: every registered policy, paired routes/groups, metrics inherited from R13.
 policy=lambda r:(r['family'],r.get('point',''),r.get('rule',''),r.get('alpha',''),r['k'])
 policies=sorted({policy(r) for r in slip if r['role']=='validation'})
 for fold in range(1,5):
  for pol in policies:
   for candidate,base in (('T_H','H'),('H','V'),('T_H','V')):
    by={(g,s):[r for r in slip if r['role']=='validation' and r['fold']==f'htt_leave_p{fold}' and r['group']==g and int(r['seed'])==s and policy(r)==pol] for g in (candidate,base) for s in SEEDS}
    if any(not x for x in by.values()):continue
    keep=('episode','leakage_group','probe');drop=('group','fold','seed','role','family','point','rule','alpha','k','threshold')
    converted={(g,s):[{k:(v if k in keep else int(v)) for k,v in r.items() if k not in drop} for r in by[(g,s)]] for g in (candidate,base) for s in SEEDS}
    reps={metric:[] for metric in r13.PAIR_METRICS}
    for sample in canonical[fold]:
     w=weights(sample);seed_diffs={metric:[] for metric in r13.PAIR_METRICS}
     for seed in SEEDS:
      ca=r13.aggregate(converted[(candidate,seed)],w);ba=r13.aggregate(converted[(base,seed)],w)
      for metric in r13.PAIR_METRICS:seed_diffs[metric].append(ca[metric]-ba[metric])
     for metric in r13.PAIR_METRICS:reps[metric].append(float(np.mean(seed_diffs[metric])))
    for metric in r13.PAIR_METRICS:
     lo,med,hi,n=ci(reps[metric]);results.append({'task':'slip','fold':fold,'candidate':candidate,'base':base,'policy':'|'.join(pol),'metric':metric,'q025':lo,'q50':med,'q975':hi,'valid':n})
 # Force: validation episode/axis statistics, T_H-H.
 force={}
 for route in ('H','T_H'):
  for fold in range(1,5):
   for seed in SEEDS:force[(route,fold,seed)]=[r for r in read(a.evaluation/f'force/{route}_p{fold}_s{seed}/trial_axis_metrics.csv') if r['role']=='validation']
 for fold in range(1,5):
  for axis in ('fx','fy','fz'):
   for metric in ('mae','rmse','bias','amplitude_mae'):
    reps=[]
    for sample in canonical[fold]:
     w=weights(sample);reps.append(float(np.mean([weighted([r for r in force[('T_H',fold,s)] if r['axis']==axis],metric,w)-weighted([r for r in force[('H',fold,s)] if r['axis']==axis],metric,w) for s in SEEDS])))
    lo,med,hi,n=ci(reps);results.append({'task':'force','fold':fold,'candidate':'T_H','base':'H','policy':axis,'metric':metric,'q025':lo,'q50':med,'q975':hi,'valid':n})
 # Future: route effect, route-specific gain over predicted-current persistence, and gain difference.
 future={}
 for route in ('H','T_H'):
  for fold in range(1,5):
   for seed in SEEDS:
    key=f'{route}_p{fold}_s{seed}';rows=read(a.evaluation/f'future_diagnostics/{key}/trial_diagnostics.csv')
    pred=torch.load(a.evaluation/f'future/{key}/predictions_validation.pt',map_location='cpu',weights_only=False)
    mapping={}
    for episode,group in zip(pred['episode_id'],pred['leakage_group']):
     episode,group=str(episode),str(group)
     if episode in mapping and mapping[episode]!=group:raise RuntimeError('future episode group drift')
     mapping[episode]=group
    future[(route,fold,seed)]=[{**r,'leakage_group':mapping[r['episode_id']]} for r in rows if r['role']=='validation']
 future_index={(route,fold,seed,method,horizon,axis,stratum):[r for r in rows if r['method']==method and r['horizon']==horizon and r['axis']==axis and r['stratum']==stratum] for (route,fold,seed),rows in future.items() for method in ('neural','predicted_current_persistence','fit_ridge_linear','gt_current_persistence_ideal_only') for horizon in ('1','5','10') for axis in ('fx','fy','fz') for stratum in ('all','stable','transitional','changing')}
 for fold in range(1,5):
  for horizon in ('1','5','10'):
   for axis in ('fx','fy','fz'):
    for stratum in ('all','stable','transitional','changing'):
     for metric in ('future_mae','future_mse','change_mae','change_mse'):
      specs=(('T_H:neural','H:neural'),('H:neural','H:predicted_current_persistence'),('T_H:neural','T_H:predicted_current_persistence'),('H:neural','H:fit_ridge_linear'),('T_H:neural','T_H:fit_ridge_linear'))
      cached={}
      for candidate,base in specs:
       reps=[]
       cr,cm=candidate.split(':');br,bm=base.split(':')
       for sample in canonical[fold]:
        w=weights(sample);seed_diffs=[]
        for seed in SEEDS:
         left,right=future_index[(cr,fold,seed,cm,horizon,axis,stratum)],future_index[(br,fold,seed,bm,horizon,axis,stratum)]
         if not left or not right:continue
         seed_diffs.append(weighted(left,metric,w)-weighted(right,metric,w))
        reps.append(float(np.mean(seed_diffs)) if seed_diffs else float('nan'))
       cached[(candidate,base)]=reps;lo,med,hi,n=ci(reps);results.append({'task':'future','fold':fold,'candidate':candidate,'base':base,'policy':f'h{horizon}|{axis}|{stratum}','metric':metric,'q025':lo,'q50':med,'q975':hi,'valid':n})
      gain=[a-b for a,b in zip(cached[('T_H:neural','T_H:predicted_current_persistence')],cached[('H:neural','H:predicted_current_persistence')])]
      lo,med,hi,n=ci(gain);results.append({'task':'future','fold':fold,'candidate':'T_H_gain_over_persistence','base':'H_gain_over_persistence','policy':f'h{horizon}|{axis}|{stratum}','metric':metric,'q025':lo,'q50':med,'q975':hi,'valid':n})
      linear_gain=[a-b for a,b in zip(cached[('T_H:neural','T_H:fit_ridge_linear')],cached[('H:neural','H:fit_ridge_linear')])]
      lo,med,hi,n=ci(linear_gain);results.append({'task':'future','fold':fold,'candidate':'T_H_gain_over_fit_ridge','base':'H_gain_over_fit_ridge','policy':f'h{horizon}|{axis}|{stratum}','metric':metric,'q025':lo,'q50':med,'q975':hi,'valid':n})
 write(a.output/'PAIRED_CI.csv',results)
 summary={'schema':'round16_paired_bootstrap_v1','status':'complete','draws_per_fold':DRAWS,'seed_formula':'2026091601+fold_index-1','unit':'complete validation leakage_group','shared_across_routes_methods_rules_horizons_seeds':True,'folds_pooled':False,'rows':len(results),'tasks':['slip','force','future'],'test_consumed':False}
 (a.output/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()
