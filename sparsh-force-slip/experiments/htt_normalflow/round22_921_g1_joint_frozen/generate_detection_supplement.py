#!/usr/bin/env python3
import argparse,csv,json,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np,torch
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train_frozen as T

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return list(csv.DictReader(open(p,newline='')))
def write(p,rows):
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def curves(y,s):
 order=np.argsort(-s,kind='stable');y=y[order];s=s[order];ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1];tp=np.cumsum(y)[ends];fp=np.cumsum(1-y)[ends];P=y.sum();N=len(y)-P;tpr=np.r_[0,tp/P];fpr=np.r_[0,fp/N];precision=np.r_[1,tp/np.maximum(tp+fp,1)];recall=tpr;threshold=np.r_[np.inf,s[ends]];return fpr,tpr,precision,recall,threshold
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--inventory',type=Path,required=True);ap.add_argument('--core',type=Path,required=True);ap.add_argument('--support-audit',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();runs=[r for r in json.loads(a.inventory.read_text())['runs'] if r['package'] in ('E1','E2')];thresholds={(int(r['fold']),int(r['seed']),r['group']):float(r['threshold']) for r in read(a.core/'CALIBRATION_THRESHOLDS.csv') if r['policy']=='FPR5'};preds={};curve_rows=[]
 for rr in runs:
  out=Path(rr['output']);cm=json.loads((out/'COMMIT.json').read_text());ck=torch.load(out/cm['best']['path'],map_location='cpu',weights_only=False);d=torch.load(rr['data'],map_location='cpu',weights_only=False);r=d['roles']['validation'];n=ck['normalizer'];x=(r['x']-n['mean'])/n['std'];m=T.init_model(rr['group'],rr['seed']);m.load_state_dict(ck['model']);prob=T.infer(m,x,'cpu').numpy();stage=r['stage'].numpy();mask=stage!=1;fpr,tpr,prec,rec,thr=curves((stage[mask]==2).astype(int),prob[mask]);key=(rr['fold'],rr['seed'],rr['group']);preds[key]={'prob':prob,'stage':stage,'t':r['t'].numpy(),'episode':np.asarray(r['episode_id'],object),'data':d}
  for i in range(len(fpr)):curve_rows.append({'fold':rr['fold'],'seed':rr['seed'],'group':rr['group'],'point':i,'threshold':thr[i],'fpr':fpr[i],'tpr':tpr[i],'precision':prec[i],'recall':rec[i]})
 a.output.mkdir(parents=True,exist_ok=False);write(a.output/'ROC_PR_POINTS.csv',curve_rows)
 # Mean descriptive interpolated curves.
 grid=np.linspace(0,1,201);groups=('V','C','M','MB');fig,ax=plt.subplots(figsize=(6,5));figp,axp=plt.subplots(figsize=(6,5));figz,axz=plt.subplots(figsize=(6,5))
 for g in groups:
  rr=[r for r in curve_rows if r['group']==g];by=defaultdict(list)
  for r in rr:by[(r['fold'],r['seed'])].append(r)
  roc=[];pr=[]
  for z in by.values():
   x=np.asarray([float(q['fpr']) for q in z]);y=np.asarray([float(q['tpr']) for q in z]);roc.append(np.interp(grid,x,y));re=np.asarray([float(q['recall']) for q in z]);pc=np.asarray([float(q['precision']) for q in z]);ix=np.argsort(re);pr.append(np.interp(grid,re[ix],pc[ix]))
  mr=np.mean(roc,axis=0);mp=np.mean(pr,axis=0);ax.plot(grid,mr,label=g);axz.plot(grid[grid<=.2],mr[grid<=.2],label=g);axp.plot(grid,mp,label=g)
 for q,title,xlabel,ylabel in ((ax,'Validation ROC','FPR','Recall'),(axz,'Same-FPR recall (zoom)','FPR','Recall'),(axp,'Validation PR','Recall','Precision')):q.set_title(title);q.set_xlabel(xlabel);q.set_ylabel(ylabel);q.legend();q.grid(alpha=.2)
 fig.tight_layout();fig.savefig(a.output/'ROC.png',dpi=180);plt.close(fig);figz.tight_layout();figz.savefig(a.output/'SAME_FPR_RECALL.png',dpi=180);plt.close(figz);figp.tight_layout();figp.savefig(a.output/'PR.png',dpi=180);plt.close(figp)
 # Complete locked main-comparison candidate and per-fold/seed deterministic selections.
 tm=[r for r in read(a.core/'TRIAL_METRICS.csv') if r['role']=='validation' and r['policy']=='FPR5|raw'];cal={(r['group'],r['fold'],r['seed']):float(r['frame_static_FPR']) for r in read(a.core/'WORKPOINT_METRICS.csv') if r['role']=='calibration' and r['policy']=='FPR5|raw'};lookup={(r['fold'],r['seed'],r['group'],r['episode']):r for r in tm};cands=[]
 for fold in map(str,range(1,5)):
  for seed in map(str,(20260914,20260915,20260916)):
   episodes=sorted({k[3] for k in lookup if k[0]==fold and k[1]==seed})
   for comp in ('C-V','M-V','M-C','MB-M'):
    left,right=comp.split('-')
    for ep in episodes:
     l=lookup[fold,seed,left,ep];r=lookup[fold,seed,right,ep];lfp=float(l['static_fp'])/max(float(l['static_frames']),1);rfp=float(r['static_fp'])/max(float(r['static_frames']),1);lmiss=float(l['gross_frames'])-float(l['gross_tp']);rmiss=float(r['gross_frames'])-float(r['gross_tp']);lmig=lfp-cal[left,fold,seed];rmig=rfp-cal[right,fold,seed];cands.append({'fold':fold,'seed':seed,'episode':ep,'comparison':comp,'static_alarm_rate_increment':lfp-rfp,'gross_miss_increment':lmiss-rmiss,'calibration_migration_increment':lmig-rmig,'left_validation_static_fpr':lfp,'right_validation_static_fpr':rfp,'left_calibration_static_fpr':cal[left,fold,seed],'right_calibration_static_fpr':cal[right,fold,seed]})
 write(a.output/'MAIN_COMPARISON_CASE_CANDIDATES.csv',cands);selected=[]
 for fold in map(str,range(1,5)):
  for seed in map(str,(20260914,20260915,20260916)):
   for comp in ('C-V','M-V','M-C','MB-M'):
    z=[r for r in cands if r['fold']==fold and r['seed']==seed and r['comparison']==comp]
    for metric in ('static_alarm_rate_increment','gross_miss_increment','calibration_migration_increment'):
     pick=sorted(z,key=lambda r:(-float(r[metric]),r['episode']))[0];selected.append({'fold':fold,'seed':seed,'comparison':comp,'rule':f'largest_{metric}',**pick})
 write(a.output/'PER_FOLD_SEED_SELECTED_CASES.csv',selected)
 # Fixed representative cases with model error traces and raw aligned frames.
 support=json.loads(a.support_audit.read_text())['targets'];fixed=json.loads((a.core/'SELECTED_CASES.json').read_text())['selected'];figures=[]
 for i,item in enumerate(fixed):
  fold=int(item['fold']);seed=int(item['seed']);ep=item['episode'];left,right=item['comparison'].split('-');zl=preds[fold,seed,left];zr=preds[fold,seed,right];mask=zl['episode']==ep;t=zl['t'][mask];stage=zl['stage'][mask];pl=zl['prob'][mask];pr=zr['prob'][mask];tl=thresholds[fold,seed,left];tr=thresholds[fold,seed,right];src=Path(support[ep]['source_path']);raw=np.load(src,allow_pickle=False);imgs=raw['tactile_img'];err=((stage==0)&((pl>=tl)|(pr>=tr)))|((stage==2)&((pl<tl)|(pr<tr)));candidates=t[err] if err.any() else t;ids=np.unique(np.clip(np.quantile(candidates,[0,.5,1]).astype(int),0,len(imgs)-1));
  while len(ids)<3:ids=np.unique(np.r_[ids,np.linspace(0,len(imgs)-1,3,dtype=int)])[:3]
  fig=plt.figure(figsize=(11,7));gs=fig.add_gridspec(2,3,height_ratios=[1.2,1]);ax=fig.add_subplot(gs[0,:]);ax.plot(t,pl,label=f'{left} probability');ax.plot(t,pr,label=f'{right} probability');ax.axhline(tl,ls='--',label=f'{left} cal-FPR5 threshold');ax.axhline(tr,ls=':',label=f'{right} cal-FPR5 threshold');
  for sid,color,name in ((0,'#d9f0d3','static'),(1,'#fee08b','incipient'),(2,'#f4a582','gross')):
   ix=np.where(stage==sid)[0]
   if len(ix):ax.scatter(t[ix],np.full(len(ix),-.03),s=8,c=color,label=name)
  ax.scatter(t[(stage==0)&(pl>=tl)],pl[(stage==0)&(pl>=tl)],marker='x',c='red',label=f'{left} false alarm');ax.scatter(t[(stage==2)&(pl<tl)],pl[(stage==2)&(pl<tl)],marker='v',c='black',label=f'{left} miss');ax.set_ylim(-.08,1.03);ax.set_xlabel('native frame t');ax.set_ylabel('slip probability');ax.legend(ncol=4,fontsize=7)
  for j,k in enumerate(ids[:3]):q=fig.add_subplot(gs[1,j]);q.imshow(imgs[k]);q.set_title(f'source t={k}');q.axis('off')
  fig.suptitle(f'{item["rule"]}: {item["comparison"]} {ep}');fig.tight_layout();name=f'detection_model_failure_{i+1}.png';fig.savefig(a.output/name,dpi=180);plt.close(fig);figures.append({'figure':name,'source_path':str(src),'source_sha256':sha(src),'episode':ep,'comparison':item['comparison']})
 (a.output/'FAILURE_FIGURES.json').write_text(json.dumps(figures,indent=2)+'\n');summary={'schema':'round22_detection_supplement_v1','status':'complete','runs':48,'roc_pr_points':len(curve_rows),'main_candidate_rows':len(cands),'per_fold_seed_selected_rows':len(selected),'fixed_model_failure_figures':len(figures),'comparisons':['C-V','M-V','M-C','MB-M'],'thresholds':'locked calibration FPR5 only','descriptive_only':True,'test_consumed':False};(a.output/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()
