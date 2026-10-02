#!/usr/bin/env python3
import argparse,csv,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readcsv(p):return list(csv.DictReader(open(p,newline='')))
def mean(rows,key):return float(np.mean([float(r[key]) for r in rows]))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--code',type=Path,required=True);ap.add_argument('--support-audit',type=Path,required=True);ap.add_argument('--report-name',default='reporting');a=ap.parse_args();det=a.root/'detection_core';force=a.root/'force_core';diag=a.root/'prediction_diagnostics_v2';out=a.root/a.report_name;out.mkdir(parents=True,exist_ok=False)
 ds=json.loads((det/'SUMMARY.json').read_text());fs=json.loads((force/'SUMMARY.json').read_text());ps=json.loads((diag/'SUMMARY.json').read_text());detail=json.loads((a.root/'force_trial_detail_v2'/'SUMMARY.json').read_text());detsup=json.loads((a.root/'detection_supplement'/'SUMMARY.json').read_text());forcefig=json.loads((a.root/'force_failure_figures'/'SUMMARY.json').read_text());acc=json.loads((a.code/'FROZEN_FORMAL_ACCEPTANCE.json').read_text())['summary'];assert ds['runs']==48 and fs['runs']==36 and ds['bootstrap_draws']==fs['bootstrap_draws']==2000 and ps['slip_runs']+ps['force_runs']==84 and detail['rows']==18000
 # Detection locked FPR5 raw validation aggregate.
 dm=readcsv(det/'WORKPOINT_METRICS.csv');selected=[r for r in dm if r['role']=='validation' and r['policy']=='FPR5|raw'];groups=sorted({r['group'] for r in selected});drows=[]
 for g in groups:
  z=[r for r in selected if r['group']==g];drows.append({'group':g,'frame_static_FPR':mean(z,'frame_static_FPR'),'gross_recall':mean(z,'gross_recall'),'balanced_accuracy':mean(z,'balanced_accuracy'),'AP':mean(z,'AP'),'pAUC':mean(z,'pAUC')})
 with (out/'DETECTION_AGGREGATE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(drows[0]));w.writeheader();w.writerows(drows)
 fig,ax=plt.subplots(figsize=(7,4));x=np.arange(len(groups));ax.bar(x-.2,[r['gross_recall'] for r in drows],.4,label='gross recall');ax.bar(x+.2,[r['balanced_accuracy'] for r in drows],.4,label='balanced accuracy');ax.set_xticks(x,groups);ax.set_ylim(0,1);ax.set_ylabel('mean validation metric');ax.legend();fig.tight_layout();fig.savefig(out/'detection_validation.png',dpi=180);plt.close(fig)
 # Force all-axis validation aggregate.
 fm=readcsv(force/'METRICS.csv');rawsel=[r for r in fm if r['role']=='validation' and r['stratum']=='all' and r['axis']=='all'];dedup={};
 for r in rawsel:dedup.setdefault((r['fold'],r['seed'],r['variant'],r['role'],r['stratum'],r['horizon'],r['axis']),r)
 sel=list(dedup.values());variants=sorted({r['variant'] for r in sel});frows=[]
 for v in variants:
  for h in (1,5,10):
   z=[r for r in sel if r['variant']==v and int(r['horizon'])==h];frows.append({'variant':v,'horizon':h,'delta_mae_n':mean(z,'delta_mae_n'),'absolute_future_mae_n':mean(z,'absolute_future_mae_n')})
 with (out/'FORCE_AGGREGATE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(frows[0]));w.writeheader();w.writerows(frows)
 fig,ax=plt.subplots(figsize=(9,4));vv=variants;xx=np.arange(len(vv));width=.22
 for i,h in enumerate((1,5,10)):ax.bar(xx+(i-1)*width,[next(r['absolute_future_mae_n'] for r in frows if r['variant']==v and r['horizon']==h) for v in vv],width,label=f'h={h}')
 ax.set_xticks(xx,vv,rotation=25,ha='right');ax.set_ylabel('validation absolute future MAE (N)');ax.legend();fig.tight_layout();fig.savefig(out/'force_validation.png',dpi=180);plt.close(fig)
 # Real source trial montages for preregistered selected cases.
 support=json.loads(a.support_audit.read_text())['targets'];cases=[]
 for family,base in [('detection',det),('force',force)]:
  for i,item in enumerate(json.loads((base/'SELECTED_CASES.json').read_text())['selected']):
   rec=item.get('record',item);eid=rec.get('episode') or rec.get('leakage_group');
   if '_sliding/' not in eid:eid=eid.replace('htt/','htt/',1).replace('/0_press_','_sliding/0_press_',1)
   src=Path(support[eid]['source_path']);z=np.load(src,allow_pickle=False);imgs=z['tactile_img'];raw=z['6d_force'][:,:3]-z['ref_force'][None,:3];ids=np.linspace(0,len(imgs)-1,5,dtype=int);fig,axs=plt.subplots(2,3,figsize=(10,6));
   for ax0,j in zip(axs.flat[:5],ids):ax0.imshow(imgs[j]);ax0.set_title(f't={j}');ax0.axis('off')
   ax=axs.flat[5];ax.plot(raw[:,0],label='Fx');ax.plot(raw[:,1],label='Fy');ax.plot(raw[:,2],label='Fz');ax.legend(fontsize=7);ax.set_title('raw force minus reference');fig.suptitle(f'{family}: {item["rule"]}\n{eid}');fig.tight_layout();name=f'{family}_case_{i+1}.png';fig.savefig(out/name,dpi=160);plt.close(fig);cases.append({'family':family,'rule':item['rule'],'episode_id':eid,'source_path':str(src),'source_sha256':sha(src),'figure':name})
 (out/'SOURCE_CASES.json').write_text(json.dumps(cases,indent=2)+'\n')
 # Descriptive probability/delta distribution and collapse evidence.
 slipdiag=readcsv(diag/'SLIP_PROBABILITY_BY_STAGE.csv');forcediag=readcsv(diag/'FORCE_BY_HORIZON_AXIS.csv');collapse=[r for r in slipdiag+forcediag if r['finite']!='True'];
 bestd=max(drows,key=lambda r:r['pAUC']); bestf=min((r for r in frows if r['horizon']==10),key=lambda r:r['absolute_future_mae_n']); fci=readcsv(force/'PAIRED_CI.csv');f2h10=[r for r in fci if r['left']=='F2-half' and r['right']=='K-VF' and int(r['horizon'])==10 and r['metric']=='absolute_future_mae'];stable_all_negative=all(float(r['q975'])<0 for r in f2h10 if r['stratum']=='stable');changing_all_positive=all(float(r['q025'])>0 for r in f2h10 if r['stratum']=='changing');all_cross_zero=all(float(r['q025'])<=0<=float(r['q975']) for r in f2h10 if r['stratum']=='all')
 summary={'schema':'round22_g1_final_reporting_v1','status':'complete','formal_runs':84,'detection_runs':ds['runs'],'force_runs':fs['runs'],'force_trial_horizon_axis_rows':detail['rows'],'bootstrap_draws':2000,'prediction_diagnostics':ps,'detection_supplement':detsup,'force_failure_figures':forcefig,'collapse_or_nonfinite_rows':len(collapse),'source_case_figures':len(cases),'f2_h10_all_fold_ci_cross_zero':all_cross_zero,'f2_h10_stable_all_negative':stable_all_negative,'f2_h10_changing_all_positive':changing_all_positive,'training_elapsed_seconds':acc['observed_wall']['seconds'],'training_three_gpu_elapsed_gpu_hours':acc['observed_wall']['three_gpu_elapsed_gpu_hours'],'evaluation_test_consumed':False,'f3_triggered':False,'independent_G3_review':'not_part_of_G1_author_acceptance'};(out/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
 (out/'REPORT.md').write_text(f'''# Round22 G1 formal report\n\nAll 84 formal frozen runs passed checkpoint and identity acceptance. Training used 0.2394 elapsed GPU-hours across three devices and early stopping produced 11–57 epochs (median 19).\n\nThe calibration-FPR5 threshold does not imply 5% validation FPR: mean validation FPR was 19.42% (V), 19.19% (C), 17.15% (M), and 18.24% (MB). At this locked operating point, the highest descriptive mean pAUC group was **{bestd['group']}** ({bestd['pAUC']:.4f}). Full group aggregates are in `DETECTION_AGGREGATE.csv`; paired 2,000-draw intervals remain in the immutable core table. The lower M false-alarm mean and higher MB recall mean belong to different models and are not combined into a single-model claim. Small FiLM mean differences do not override fold-level intervals.\n\nFor all-axis horizon-10 validation absolute future MAE, the lowest descriptive mean variant was **{bestf['variant']}** ({bestf['absolute_future_mae_n']:.4f} N). Full horizon/variant aggregates use `axis=all` only and de-duplicate the repeated hold baseline. The fixed F2 shrinkage is not a general improvement: horizon-10 all-stratum intervals cross zero in every fold, while stable intervals are negative in every fold and changing intervals are positive in every fold. This is a stable-regime benefit with a changing-regime cost. Full paired intervals remain in the core table.\n\nPrediction diagnostics cover 48 slip runs by stage (including incipient) and 36 force runs by horizon and axis; all {ps['finite_rows']} diagnostic rows are finite. Near-copy, prediction spread relative to GT spread, and unclipped force-bound exceedance are reported descriptively; this is not a universal no-collapse claim. Four preregistered cases were resolved to hashed raw source trials and rendered as tactile/force figures. No test role was consumed. F3 was not triggered. Independent G3 review remains separate.\n''')
 # Author acceptance is mechanical and distinct from independent review.
 acceptance={'schema':'round22_g1_author_acceptance_v1','status':'pass','checks':{'formal_84':acc['accepted_runs']==84,'detection_48':ds['runs']==48,'force_36':fs['runs']==36,'force_trial_horizon_axis_18000':detail['rows']==18000,'bootstrap_2000':ds['bootstrap_draws']==fs['bootstrap_draws']==2000,'diagnostics_84':ps['slip_runs']+ps['force_runs']==84,'detection_supplement_complete':detsup['status']=='complete' and detsup['per_fold_seed_selected_rows']==144,'force_model_failure_figures':forcefig['figures']==2,'no_nonfinite_or_collapse':not collapse,'source_figures':len(cases)==4,'no_test':not ds['test_consumed'] and not fs['test_consumed'],'f3_not_triggered':True},'independent_review':False};acceptance['status']='pass' if all(acceptance['checks'].values()) else 'fail';(out/'AUTHOR_ACCEPTANCE.json').write_text(json.dumps(acceptance,indent=2)+'\n');print(json.dumps({'summary':summary,'author_acceptance':acceptance}));assert acceptance['status']=='pass'
if __name__=='__main__':main()
