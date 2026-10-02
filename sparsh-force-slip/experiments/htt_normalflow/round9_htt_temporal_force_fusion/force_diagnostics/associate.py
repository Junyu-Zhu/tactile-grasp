import argparse,csv,json,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import rankdata
from analyze import sha,writecsv
H=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--force-dir',type=Path,required=True);p.add_argument('--current-dir',type=Path,required=True);a=p.parse_args();pr=json.loads((H/'ASSOCIATION_PROTOCOL.json').read_text())
 au=json.loads((a.force_dir/'SUPPORT_AUDIT.json').read_text());cu=json.loads((a.current_dir/'summary.json').read_text());assert au['status']=='pass' and cu['status']=='complete' and cu['synthetic'] is False
 for n in ['frame_errors.csv','per_trial_axis.csv']:assert sha(a.force_dir/n)==au['output_hashes'][n]
 ct=a.current_dir/'trials.csv';hs=[v for k,v in cu['output_hashes'].items() if Path(k).name=='trials.csv'];assert len(hs)==1 and hs[0]==sha(ct)
 read=lambda p:list(csv.DictReader(p.open()))
 frows=read(a.force_dir/'frame_errors.csv');arows=read(a.force_dir/'per_trial_axis.csv');trows=read(ct)
 err=defaultdict(list);ax=defaultdict(list)
 for r in frows:
  if r['role']=='validation' and r['stage'] in ['0','2']:err[(r['fold'],r['seed'],r['episode_id'],r['stage'])].append(r)
 for r in arows:
  if r['role']=='validation' and r['population'] in ['static','gross']:ax[(r['fold'],r['seed'],r['episode_id'],'0' if r['population']=='static' else '2')].append(r)
 joined=[];missing=[]
 for r in trows:
  if r['role']!='validation' or r['group'] not in pr['groups'] or r['point']!=pr['operating_point'] or r['rule']!=pr['rule']:continue
  for stage,outcome,positive,total in [('0','static_fpr','fp',int(r['fp'])+int(r['tn'])),('2','gross_miss_rate','fn',int(r['fn'])+int(r['tp']))]:
   key=r['fold'],r['seed'],r['episode'],stage;base=dict(fold=r['fold'],seed=int(r['seed']),group=r['group'],episode_id=r['episode'],leakage_group=r['leakage_group'],outcome=outcome)
   if total==0:missing.append(dict(base,reason='no_corresponding_stage_endpoints'));continue
   rr=err.get(key,[]);aa=ax.get(key,[])
   if not rr or len(aa)!=3:missing.append(dict(base,reason='missing_GT_or_axis'));continue
   assert len(rr)==total and len({int(v['t']) for v in rr})==total
   joined.append(dict(base,n=total,error_rate=int(r[positive])/total,force_mae_N=float(np.mean([float(v['force_mae']) for v in rr])),delta5_mae_N=float(np.mean([float(v['delta5_mae']) for v in rr])),bias_l2_N=float(np.linalg.norm([float(v['bias']) for v in aa])),clip_frame_fraction=float(np.mean([v['clipped']=='True' for v in rr]))))
 averaged=[];by=defaultdict(list)
 for r in joined:by[(r['fold'],r['group'],r['episode_id'],r['outcome'])].append(r)
 for (fold,g,ep,outcome),rr in by.items():
  assert len(rr)==3 and {r['seed'] for r in rr}=={20260914,20260915,20260916}
  averaged.append(dict(fold=fold,group=g,episode_id=ep,leakage_group=rr[0]['leakage_group'],outcome=outcome,seeds=3,**{k:float(np.mean([r[k] for r in rr])) for k in ['error_rate']+pr['predictors']}))
 corrs=[]
 for fold in sorted({r['fold'] for r in averaged}):
  for g in pr['groups']:
   for outcome in pr['outcomes']:
    rr=[r for r in averaged if r['fold']==fold and r['group']==g and r['outcome']==outcome]
    for key in pr['predictors']:
     x=np.asarray([r[key] for r in rr]);y=np.asarray([r['error_rate'] for r in rr]);valid=len(rr)>=3 and np.ptp(x)>0 and np.ptp(y)>0
     rho=float(np.corrcoef(rankdata(x),rankdata(y))[0,1]) if valid else None
     corrs.append(dict(fold=fold,group=g,outcome=outcome,predictor=key,trial_count=len(rr),leakage_groups=len({r['leakage_group'] for r in rr}),spearman=rho,status='descriptive_available' if valid else 'undefined_constant_or_less_than3',mean_error_rate=float(y.mean()) if len(y) else None))
 for name,rows in [('association_per_seed_trial.csv',joined),('association_seedmean_trial.csv',averaged),('association_correlations.csv',corrs),('association_exclusions.csv',missing)]:writecsv(a.force_dir/name,rows)
 lines=['# 力误差与当前slip失败的描述性关联','','固定validation/raw/FPR0.05工作点，先在每折每试次平均3个种子，再计算试次间Spearman。不重新选阈值，不根据结果选择关联指标。','', '|折|组|结果|力误差指标|有效试次|Spearman|状态|','|---|---|---|---|---|---|---|']
 for r in corrs:lines.append('|'+ '|'.join(str(r[k]) for k in ['fold','group','outcome','predictor','trial_count','spearman','status'])+'|')
 lines+=['','观察：静态误报关联并非各折一致；p2/p3/p4的力MAE正相关也出现在纯视觉V对照，因此不能将该关联归因于力输入通路。gross漏报关联跨折方向变化，缺少一致关系。有效试次数很少，所有相关仅为失败诊断线索。','','缺对应static/gross的试次只从相应结果剔除，记录association_exclusions.csv；无GT亦单独记录，不能补零。常数指标无法计算相关。所有折均保留，不将重叠折/种子当独立样本，不给p值或因果结论。HTT标签有力规则成分且validation用于模型选择，相关不是独立物理机制证明。']
 (a.force_dir/'ASSOCIATION_ZH.md').write_text('\n'.join(lines)+'\n')
 files=['association_per_seed_trial.csv','association_seedmean_trial.csv','association_correlations.csv','association_exclusions.csv','ASSOCIATION_ZH.md']
 (a.force_dir/'ASSOCIATION_AUDIT.json').write_text(json.dumps(dict(status='pass',protocol_sha256=sha(H/'ASSOCIATION_PROTOCOL.json'),source_sha256=sha(Path(__file__)),force_audit_sha256=sha(a.force_dir/'SUPPORT_AUDIT.json'),current_summary_sha256=sha(a.current_dir/'summary.json'),current_trials_sha256=sha(ct),joined_seed_trials=len(joined),averaged_trials=len(averaged),exclusions=len(missing),missing_GT=sum(r['reason']=='missing_GT_or_axis' for r in missing),correlations=len(corrs),output_hashes={n:sha(a.force_dir/n) for n in files}),indent=2)+'\n');print('association pass',len(corrs))
if __name__=='__main__':main()
