#!/usr/bin/env python3
"""Reviewer-requested diagnostics from frozen best checkpoints, accepted caches, and raw scores."""
from __future__ import annotations
import argparse,csv,json,hashlib,math
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import train as r18

def read(p): return list(csv.DictReader(Path(p).open()))
def write(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 fields=list(dict.fromkeys(k for x in rows for k in x))
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def envelope(curves,targets,kind):
 fpr=np.array([float(x['frame_static_FPR']) for x in curves]);rec=np.array([float(x['gross_recall']) for x in curves]);th=np.array([float(x['threshold']) for x in curves])
 ok=np.isfinite(fpr)&np.isfinite(rec);fpr,rec,th=fpr[ok],rec[ok],th[ok]
 order=np.lexsort((-rec,fpr));fpr,rec,th=fpr[order],rec[order],th[order]
 uf=[];ur=[];ut=[]
 for x in np.unique(fpr):
  ids=np.where(fpr==x)[0];j=ids[np.argmax(rec[ids])];uf.append(x);ur.append(rec[j]);ut.append(th[j])
 uf=np.asarray(uf);ur=np.maximum.accumulate(ur);ut=np.asarray(ut)
 if kind=='same_fpr':
  vals=np.interp(targets,uf,ur,left=ur[0],right=ur[-1]);return vals
 # Invert the monotone recall envelope, retaining the first/lowest FPR attaining a recall.
 uq=[];u_f=[]
 for y in np.unique(ur):
  uq.append(y);u_f.append(uf[np.where(ur==y)[0][0]])
 return np.interp(targets,np.asarray(uq),np.asarray(u_f),left=u_f[0],right=u_f[-1])
def alarm_states(t,scores,threshold,k):
 out=[];state=False;run=0;prev=None
 for ti,score in zip(t,scores):
  if prev is None or ti!=prev+1: state=False;run=0
  if score>=threshold:
   run+=1
   if run>=k:state=True
  else: state=False;run=0
  out.append(int(state));prev=ti
 return out
def film_supplement(prepared,formal,device,out):
 rows=[]
 for fold in range(1,5):
  for seed in r18.SEEDS:
   data=torch.load(prepared/f'p{fold}_s{seed}'/'prepared.pt',map_location='cpu',weights_only=False);ck=torch.load(formal/'M0'/f'p{fold}_s{seed}'/'best.pth',map_location='cpu',weights_only=False);m=r18.init_model('M0',seed);m.load_state_dict(ck['model']);m.to(device).eval();norm=ck['normalizer']
   for role in r18.ROLES:
    d=data['roles'][role];x=(d['x']-norm['mean'])/norm['std'];gs=[];bs=[];rs=[]
    with torch.no_grad():
     for ids in torch.arange(len(x)).split(1024):
      xx=x[ids].to(device);c=m.components(xx);base=m.visual_ln(xx[...,:192]);mod=(1+c['gamma'])*base+c['beta'];gs.append(c['gamma'].cpu().numpy());bs.append(c['beta'].cpu().numpy());rs.append((mod.norm(dim=-1)/(base.norm(dim=-1)+1e-12)).cpu().numpy())
    g=np.concatenate(gs);b=np.concatenate(bs);ratio=np.concatenate(rs)
    for ti in range(g.shape[1]):
     for fi in range(g.shape[2]):
      gv=g[:,ti,fi];bv=b[:,ti,fi];scale=1+gv
      rows.append({'group':'M0','fold':fold,'seed':seed,'role':role,'time_index':ti,'relative_history_step':ti-(g.shape[1]-1),'feature':fi,'endpoints':len(gv),'gamma_mean':gv.mean(),'gamma_std':gv.std(),'gamma_abs_q95':np.quantile(np.abs(gv),.95),'gamma_min':gv.min(),'gamma_max':gv.max(),'beta_mean':bv.mean(),'beta_std':bv.std(),'beta_abs_q95':np.quantile(np.abs(bv),.95),'beta_min':bv.min(),'beta_max':bv.max(),'scale_min':scale.min(),'scale_max':scale.max(),'norm_ratio_time_q95':np.quantile(ratio[:,ti],.95)})
 write(out/'diagnostics/FILM_FEATURE_TIME_ROLE.csv',rows);return rows
def curve_supplement(evaluation,out):
 rows=read(evaluation/'metrics/DESCRIPTIVE_CURVES.csv');by=defaultdict(list)
 for x in rows:by[(x['group'],int(x['fold']),int(x['seed']))].append(x)
 ftargets=np.linspace(0,.1,101);rtargets=np.linspace(0,1,1001);samef=[];samer=[]
 for (group,fold,seed),rr in sorted(by.items()):
  vals=envelope(rr,ftargets,'same_fpr')
  samef += [{'group':group,'fold':fold,'seed':seed,'target_static_FPR':x,'envelope_interpolated_gross_recall':y,'source_threshold_grid':'0:0.001:1','descriptive_only':True} for x,y in zip(ftargets,vals)]
  vals=envelope(rr,rtargets,'same_recall')
  samer += [{'group':group,'fold':fold,'seed':seed,'target_gross_recall':x,'envelope_interpolated_static_FPR':y,'source_threshold_grid':'0:0.001:1','descriptive_only':True} for x,y in zip(rtargets,vals)]
 write(out/'metrics/SAME_FPR_ENVELOPE.csv',samef);write(out/'metrics/SAME_RECALL_ENVELOPE.csv',samer)
 # Complete paired run deltas, never model selection.
 pf=[];pr=[]
 for fold in range(1,5):
  for seed in r18.SEEDS:
   for left,right in (('M0','C0'),('M0','V0'),('C0','V0')):
    l=[x for x in samef if x['group']==left and x['fold']==fold and x['seed']==seed];r=[x for x in samef if x['group']==right and x['fold']==fold and x['seed']==seed]
    pf += [{'fold':fold,'seed':seed,'left':left,'right':right,'target_static_FPR':x['target_static_FPR'],'gross_recall_difference':float(x['envelope_interpolated_gross_recall'])-float(y['envelope_interpolated_gross_recall']),'descriptive_only':True} for x,y in zip(l,r)]
    l=[x for x in samer if x['group']==left and x['fold']==fold and x['seed']==seed];r=[x for x in samer if x['group']==right and x['fold']==fold and x['seed']==seed]
    pr += [{'fold':fold,'seed':seed,'left':left,'right':right,'target_gross_recall':x['target_gross_recall'],'static_FPR_difference':float(x['envelope_interpolated_static_FPR'])-float(y['envelope_interpolated_static_FPR']),'descriptive_only':True} for x,y in zip(l,r)]
 write(out/'metrics/SAME_FPR_PAIRED_DIFFERENCES.csv',pf);write(out/'metrics/SAME_RECALL_PAIRED_DIFFERENCES.csv',pr);return samef,samer,pf,pr
def case_supplement(evaluation,prepared,out):
 (out/'cases').mkdir(parents=True,exist_ok=True);selected=json.loads((evaluation/'cases/SELECTED.json').read_text())['selected'];scores=read(evaluation/'scores/RAW_SCORES.csv');ths=read(evaluation/'metrics/CALIBRATION_THRESHOLDS.csv');rows=[]
 for ci,case in enumerate(selected,1):
  fold=int(case['fold']);seed=int(case['seed']);ep=case['episode'];data=torch.load(prepared/f'p{fold}_s{seed}'/'prepared.pt',map_location='cpu',weights_only=False);d=data['roles']['validation'];ids=np.asarray([i for i,e in enumerate(d['episode_id']) if e==ep]);ids=ids[np.argsort(d['t'][ids].numpy())];t=d['t'][ids].numpy();stage=d['stage'][ids].numpy();pred=d['x'][ids,-1,192:195].numpy();gt=d['y_current'][ids].numpy();hist=d['x'][ids,:,192:195].numpy()
  fig,ax=plt.subplots(3,1,figsize=(11,7),sharex=True,gridspec_kw={'height_ratios':[2,1.2,1.2]})
  for group,color in (('V0','#377eb8'),('C0','#4daf4a'),('M0','#e41a1c')):
   rr=sorted([x for x in scores if x['group']==group and int(x['fold'])==fold and int(x['seed'])==seed and x['role']=='validation' and x['episode']==ep],key=lambda x:int(x['t']));sv=np.asarray([float(x['score']) for x in rr]);tt=np.asarray([int(x['t']) for x in rr]);assert np.array_equal(tt,t);th=next(float(x['threshold']) for x in ths if x['group']==group and int(x['fold'])==fold and int(x['seed'])==seed and x['policy']=='FPR5');raw=alarm_states(t,sv,th,1);confirm=alarm_states(t,sv,th,2)
   ax[0].plot(t,sv,label=group,color=color);ax[0].axhline(th,color=color,ls='--',alpha=.35);ax[2].step(t,np.asarray(confirm)+({'V0':0,'C0':1.25,'M0':2.5}[group]),where='post',label=f'{group} confirm2',color=color)
   gap=np.r_[True,np.diff(t)!=1];seg_end=np.r_[np.diff(t)!=1,True];gross_left=np.zeros(len(t),dtype=int);gross_right=np.zeros(len(t),dtype=int);raw_right=np.zeros(len(t),dtype=int);confirm_right=np.zeros(len(t),dtype=int)
   starts=np.where(gap)[0];ends=np.where(seg_end)[0]
   for lo,hi in zip(starts,ends):
    if stage[lo]==2:
     q=lo
     while q<=hi and stage[q]==2:gross_left[q]=1;q+=1
    if stage[hi]==2:
     q=hi
     while q>=lo and stage[q]==2:gross_right[q]=1;q-=1
    for states,target in ((raw,raw_right),(confirm,confirm_right)):
     if states[hi]:
      q=hi
      while q>=lo and states[q]:target[q]=1;q-=1
   for j,idx in enumerate(ids):
    rows.append({'case_index':ci,'kind':case['kind'],'fold':fold,'seed':seed,'episode':ep,'group':group,'t':int(t[j]),'stage':int(stage[j]),'native_gap_or_left_reset':int(gap[j]),'observed_segment_right_edge':int(seg_end[j]),'gross_event_left_censored':int(gross_left[j]),'gross_event_right_boundary_censored':int(gross_right[j]),'score':sv[j],'original_calibration_FPR5_threshold':th,'raw_alarm':raw[j],'confirm2_alarm':confirm[j],'raw_alarm_run_right_censored':int(raw_right[j]),'confirm2_alarm_run_right_censored':int(confirm_right[j]),'pred_force_x':pred[j,0],'pred_force_y':pred[j,1],'pred_force_z':pred[j,2],'gt_force_x':gt[j,0],'gt_force_y':gt[j,1],'gt_force_z':gt[j,2],'force_error_mae_n':float(np.abs(pred[j]-gt[j]).mean()),'pred_force_history_xyz_json':json.dumps(hist[j].tolist(),separators=(',',':'))})
  ax[0].step(t,stage/2,where='mid',color='gray',alpha=.3,label='stage/2');ax[0].set_ylabel('score');ax[0].legend(ncol=4,fontsize=7);ax[0].grid(alpha=.2)
  for j,(lab,color) in enumerate((('x','#1b9e77'),('y','#d95f02'),('z','#7570b3'))):ax[1].plot(t,pred[:,j],color=color,label=f'pred {lab}');ax[1].plot(t,gt[:,j],color=color,ls=':',label=f'GT {lab}')
  ax[1].set_ylabel('force (N)');ax[1].legend(ncol=3,fontsize=7);ax[1].grid(alpha=.2);ax[2].set_yticks([.5,1.75,3]);ax[2].set_yticklabels(['V0','C0','M0']);ax[2].set_ylabel('confirm2 state');ax[2].set_xlabel('native endpoint t');ax[2].grid(alpha=.2);fig.suptitle(f"{case['kind']}: {ep}, p{fold}, {seed}");fig.tight_layout();fig.savefig(out/f'cases/case_{ci}_aligned_scores_force_alarm.svg',format='svg');plt.close(fig)
 write(out/'cases/CASE_ALIGNED_EVIDENCE.csv',rows);(out/'cases/CASE_EVIDENCE_NOTES.md').write_text('# Case evidence semantics\n\nAlarm state resets at the episode left edge and every native-frame gap. Gross runs already active at a segment left edge are marked left-censored; gross runs active at the observed right edge and alarm runs still active there are marked right-censored. `static_alarming_frames` in the main trial table is observed static alarm-frame exposure on retained support, not a complete uncensored alarm-duration estimate. Raw and confirm2 states use the unchanged model-specific calibration FPR5 threshold.\n');return rows
def main():
 p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--prepared-root',type=Path,required=True);p.add_argument('--formal-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 film=film_supplement(a.prepared_root,a.formal_root,a.device,a.output);sf,sr,pf,pr=curve_supplement(a.evaluation,a.output);cases=case_supplement(a.evaluation,a.prepared_root,a.output)
 files=sorted(a.output.rglob('*.csv'))+sorted(a.output.rglob('*.svg'))+sorted(a.output.rglob('*.md'));audit={'schema':'round18_reviewer_supplement_v1','status':'pass','source_evaluation_summary_sha256':sha(a.evaluation/'SUMMARY.json'),'original_thresholds_unchanged':True,'training_unchanged':True,'test_consumed':False,'counts':{'film_feature_time_role_rows':len(film),'same_fpr_rows':len(sf),'same_recall_rows':len(sr),'same_fpr_paired_rows':len(pf),'same_recall_paired_rows':len(pr),'case_aligned_rows':len(cases),'case_figures':len(list((a.output/'cases').glob('*.svg')))},'hashes':{str(x.relative_to(a.output)):sha(x) for x in files}}
 (a.output/'SUPPLEMENT_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
