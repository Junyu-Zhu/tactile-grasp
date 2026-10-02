from pathlib import Path
import csv,json,hashlib,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1]; R=P.parent
sources=[]
def read(rel):
 f=R/rel;sources.append({'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
 return list(csv.DictReader(f.open()))
def num(row,k):return float(row[k])
def save(fig,name):
 fig.savefig(P/'figures'/f'{name}.pdf',bbox_inches='tight')
 fig.savefig(P/'figures'/f'{name}.svg',bbox_inches='tight')
 fig.savefig(P/'figures'/f'{name}.png',dpi=220,bbox_inches='tight');plt.close(fig)
def table(name,rows):
 with (P/'tables'/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','axes.grid':True,'grid.alpha':.2,'axes.axisbelow':True})
colors=['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00','#666666','#56B4E9','#332288']
rq1=read('round24_921_g3_final_review/tables/RQ1_RUN_RESULTS.csv')
c1=read('round24_921_g3_final_review/tables/RQ1_PAIRED_CI.csv')
rq2=read('round24_921_g3_final_review/tables/RQ2_RUN_RESULTS.csv')
c2=read('round24_921_g3_final_review/tables/RQ2_PAIRED_CI.csv')
folds=read('round24_921_g3_final_review/tables/RQ1_FOLD_CONTRASTS.csv')
for n,rows in [('RQ1_all_runs.csv',rq1),('RQ1_paired_CI.csv',c1),('RQ2_all_runs.csv',rq2),('RQ2_paired_CI.csv',c2),('RQ1_fold_contrasts.csv',folds)]:table(n,rows)
# all runs on tradeoff scatter; non-pooled group centroids
fig,axs=plt.subplots(1,2,figsize=(10.5,5.2))
for ax,stage,groups,title in zip(axs,['G1_frozen','G2_finetuned'],[['V','C','M','MB'],['A','B','C','D']],['(a) Frozen MAE: G1','(b) Last-two-block fine-tuning: G2']):
 for i,g in enumerate(groups):
  rows=[r for r in rq1 if r['stage']==stage and r['group']==g];assert len(rows)==12,(stage,g)
  x=np.array([num(r,'frame_static_FPR')*100 for r in rows]);y=np.array([num(r,'gross_recall')*100 for r in rows])
  ax.scatter(x,y,s=18,alpha=.35,color=colors[i]);ax.scatter(x.mean(),y.mean(),s=85,marker='D',color=colors[i],edgecolor='black',linewidth=.5,label=f'{g}: {x.mean():.2f}%, {y.mean():.2f}%')
 ax.axvline(5,color='gray',ls='--',lw=1,label='Calibration target: 5%');ax.set(title=title,xlabel='Actual validation static FPR (%)',ylabel='Gross recall (%)',xlim=(-1,65),ylim=(0,102));ax.legend(fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2)
fig.suptitle('Calibration-only FPR5 thresholds; dots: all 12 runs; diamonds: descriptive means',fontsize=10)
fig.text(.5,.015,'Separate pipelines: no cross-panel ranking or attribution solely to fine-tuning.',ha='center',fontsize=8)
fig.subplots_adjust(bottom=.32,top=.83,wspace=.25);save(fig,'Fig01_slip_operating_points')
# pAUC seed points by group, no SEM
fig,axs=plt.subplots(1,2,figsize=(10,3.8))
for ax,stage,groups,title in zip(axs,['G1_frozen','G2_finetuned'],[['V','C','M','MB'],['A','B','C','D']],['(a) G1: frozen','(b) G2: fine-tuned']):
 for i,g in enumerate(groups):
  rows=sorted([r for r in rq1 if r['stage']==stage and r['group']==g],key=lambda r:(r['fold'],r['seed']))
  ys=[num(r,'pAUC') for r in rows];ax.bar(i,np.mean(ys),color=colors[i],alpha=.55,width=.65)
  for j,r in enumerate(rows):ax.plot(i+(int(r['fold'])-2.5)*.13+(int(r['seed'])-20260915)*.025,num(r,'pAUC'),marker=['o','s','^','D'][int(r['fold'])-1],color=colors[i],ms=4,markeredgecolor='black',markeredgewidth=.3)
  ax.text(i,np.mean(ys)+.025,f'{np.mean(ys):.4f}',ha='center',fontsize=8)
 ax.set(xticks=range(4),xticklabels=groups,ylim=(0,1.02),ylabel='Normalized pAUC [0, 0.1]',title=title)
fig.suptitle('All folds and seeds retained; descriptive means, not independent replicates',fontsize=10)
fig.text(.5,-.01,'Markers: folds 1/2/3/4 = circle/square/triangle/diamond. Do not rank across panels.',ha='center',fontsize=8)
fig.tight_layout();save(fig,'Fig02_slip_pauc_all_runs')
# forest single figure G1 M-V + all G2 contrasts
fig,axs=plt.subplots(1,3,figsize=(11.5,7.7),sharey=True)
keys=[('G1_frozen','M-V')]+[('G2_finetuned',c) for c in ['B-A','D-C','C-A','D-B']]
labels=[]
for k,(stage,contrast) in enumerate(keys):
 for fold in range(1,5):labels.append(f'{"G1" if k==0 else "G2"} {contrast} / fold {fold}')
for ax,metric,factor,title in zip(axs,['pAUC','frame_static_FPR','gross_recall'],[1,100,100],['Change in pAUC (higher better)','Change in FPR (pp; lower better)','Change in recall (pp; higher better)']):
 for k,(stage,contrast) in enumerate(keys):
  for f in range(1,5):
   z=next(r for r in c1 if r['stage']==stage and r['contrast']==contrast and int(r['fold'])==f and r['metric']==metric)
   y=k*5+f-1;lo=num(z,'q025')*factor;hi=num(z,'q975')*factor;center=num(z,'mean')*factor
   ax.plot([lo,hi],[y,y],color=colors[k],lw=1.5);ax.plot(center,y,'o',color=colors[k],ms=4)
 ax.axvline(0,color='gray',ls='--');ax.set_title(title,fontsize=9)
 if metric=='gross_recall':ax.axvline(-1,color='gray',ls=':',lw=.8)
axs[0].set_yticks([k*5+f for k in range(5) for f in range(4)],labels,fontsize=7);axs[0].invert_yaxis()
fig.suptitle('Existing within-fold paired 95% bootstrap intervals (complete leakage groups)',fontsize=11)
fig.text(.5,.005,'Centers are bootstrap means. Shared draws across seeds; overlapping folds are not pooled.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.025,1,.96));save(fig,'Fig03_slip_paired_intervals')
# future horizon full means
variants=['hold','ridge-K-VF','K-V','K-F','K-VF','F2-half']
fig,axs=plt.subplots(1,2,figsize=(10,3.8))
means=[]
for v in variants:
 for h in [1,5,10]:
  rr=[r for r in rq2 if r['variant']==v and int(r['horizon'])==h];assert len(rr)==12
  means.append({'variant':v,'horizon':h,'runs':12,'delta_mae_n':np.mean([num(r,'delta_mae_n') for r in rr]),'absolute_future_mae_n':np.mean([num(r,'absolute_future_mae_n') for r in rr])})
for ax,metric,title in zip(axs,['delta_mae_n','absolute_future_mae_n'],['(a) True force-change error','(b) Future absolute-force error']):
 for i,v in enumerate(variants):ax.plot([1,5,10],[r[metric] for r in means if r['variant']==v],marker=['o','s','^','v','D','P'][i],lw=1.4,color=colors[i],label=v)
 ax.set(xlabel='Prediction horizon (frames)',ylabel='MAE (N)',title=title,xticks=[1,5,10]);ax.legend(fontsize=7)
fig.suptitle('HTT future force: all four folds and three seeds; arithmetic run means',fontsize=10);fig.tight_layout();save(fig,'Fig04_future_horizons');table('RQ2_group_means.csv',means)
# future paired all key contrasts + all strata h10
fig,axs=plt.subplots(1,3,figsize=(11.5,5),sharey=True)
for ax,c,title in zip(axs,['K-VF-hold','K-VF-K-V','F2-half-K-VF'],['K-VF minus hold','K-VF minus K-V','Fixed 0.5 shrink minus K-VF']):
 for i,st in enumerate(['all','stable','transition','changing']):
  for f in range(1,5):
   z=next(r for r in c2 if r['contrast']==c and r['horizon']=='10' and r['stratum']==st and r['metric']=='delta_mae' and int(r['fold'])==f)
   y=i*5+f-1;ax.plot([num(z,'q025'),num(z,'q975')],[y,y],color=colors[i]);ax.plot(num(z,'mean'),y,'o',color=colors[i],ms=4)
 ax.axvline(0,color='gray',ls='--');ax.set(title=title,xlabel='Change in MAE (N); lower is better')
axs[0].set_yticks([i*5+f for i in range(4) for f in range(4)],[f'{s} / fold {f}' for s in ['all','stable','transition','changing'] for f in range(1,5)],fontsize=7);axs[0].invert_yaxis()
fig.suptitle('Horizon 10: existing paired 95% intervals; force strata are not slip labels',fontsize=10)
fig.tight_layout();save(fig,'Fig05_future_stratified_intervals')
# historic current force two domains
force=read('round21_final_evidence_synthesis/FORCE_EVIDENCE.csv');legacy=read('round21_final_evidence_synthesis/LEGACY_FORCE_TASK_REGRESSION.csv');table('historical_current_force.csv',force);table('historical_legacy_force.csv',legacy)
fig,axs=plt.subplots(1,2,figsize=(9,3.8))
for ax,rows,title in zip(axs,[[r for r in force if r['task']=='R10 adapted current force'],legacy],['(a) HTT slip-trial validation','(b) HTT historical force-task validation']):
 vs=list(dict.fromkeys(r['variant'] for r in rows));assert len(vs)==2,vs
 for i,v in enumerate(vs):
  ys=[num(next(r for r in rows if r['variant']==v and r['axis']==a),'mae_n') for a in ['shear_x','shear_y','normal']]
  ax.bar(np.arange(3)+(i-.5)*.34,ys,width=.34,color=colors[i],label=v)
  for j,y in enumerate(ys):ax.text(j+(i-.5)*.34,y+.025,f'{y:.3f}',ha='center',fontsize=8)
 ax.set(xticks=range(3),xticklabels=['shear x','shear y','normal'],ylabel='Current-force MAE (N)',title=title);ax.legend(fontsize=8);ax.margins(y=.18)
fig.suptitle('R10: domain adaptation benefit and old-task regression (not Sparsh regression)',fontsize=10);fig.tight_layout();save(fig,'Fig06_force_adaptation_and_regression')
# R8 full set no chosen best only
risk=read('round8_force_dynamics_event_time/results/reporting/RAW_RISK_ALL_GROUPS.csv');op=read('round8_force_dynamics_event_time/results/reporting/H3_OPERATIONAL_ALL_GROUPS.csv');table('historical_R8_all_horizons.csv',risk);table('historical_R8_H3_operating_points.csv',op)
groups=[r['group'] for r in op];short=['XYZ','XYZ+delta','hazard','concat','fusion','fusion+hazard','direct','state']
fig,axs=plt.subplots(1,3,figsize=(12,4.4))
for ax,metric,title,scale in zip(axs,['average_precision_mean','trial_false_alarm_rate_mean','event_recall_mean'],['(a) H3 raw AP','(b) Actual trial false-alarm rate','(c) Event recall'],[1,100,100]):
 rows=[next(r for r in (risk if metric.startswith('average') else op) if r['group']==g and r['horizon']=='3') for g in groups]
 y=[num(r,metric)*scale for r in rows];err=[num(r,metric.replace('_mean','_sample_sd'))*scale for r in rows]
 ax.bar(range(8),y,color=colors,yerr=err,capsize=2,error_kw={'lw':.8});ax.set(xticks=range(8),xticklabels=short,title=title,ylabel='AP' if scale==1 else '%');ax.tick_params(axis='x',labelrotation=65);ax.set_ylim(0,1 if scale==1 else (12 if metric.startswith('trial') else 100))
 if metric.startswith('trial'):ax.axhline(10,ls='--',color='gray',lw=1)
fig.suptitle('Historical R8: Sparsh-domain label-onset risk, H3; all eight candidates',fontsize=11)
fig.text(.5,.005,'Bars: three-seed means; error bars: seed SD (not independent-event confidence intervals).',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.025,1,.95));save(fig,'Fig07_historical_source_risk')
(P/'FIGURE_MANIFEST.json').write_text(json.dumps({'inputs':sources,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'figures':[{ 'path':str(f.relative_to(P)),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in sorted((P/'figures').glob('*'))],'exclusions':[],'statistical_note':'All folds/seeds retained; existing group bootstrap CI, no pooled significance. R8 SD is seed variation only.'},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'figures':len(list((P/'figures').glob('*.pdf'))),'source_files':len(sources),'new_training':0,'new_inference':0}))
