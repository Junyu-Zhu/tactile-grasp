#!/usr/bin/env python3
"""Phase7 supplemental experiments for force/slip future-instability paper evidence.
All outputs are derived artifacts. Raw datasets under /vla1/zjy/tactile_datasets are never modified.
"""
from __future__ import annotations
import argparse, json, math, shutil, subprocess, sys, time
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    plt = None
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import phase2_b_multitask as p2
import phase3_2_world_model as wm
import phase5_friction_stability as p5
REPO=Path('/home/zjy/document/tactile-grasp'); WORKSPACE=REPO/'sparsh-force-slip'; REPORT=WORKSPACE/'reports/phase7'
PHASE4=Path('/vla1/zjy/sparsh_runs/force_slip_phase4'); PHASE5=Path('/vla1/zjy/sparsh_runs/force_slip_phase5'); PHASE6=Path('/vla1/zjy/sparsh_runs/force_slip_phase6')
P4_DEC='phase4_3_reuse_p3_features_20260517_171500'; P4_SEP='phase4_2_separate_features_20260517_171500'; P5_FRIC='phase5_2_friction_features_20260518_0035'; P6_FRIC='phase6_1_friction_features_20260518_0130'
HORIZONS=(1,3,5); BASE_FULL=list(p5.BASE_FULL); STATIC=list(p5.STATIC_FORCE_SLIP); FULL_Q=list(p5.FRICTION_MODES['full_plus_q']); EPS=1e-6

def now(): return datetime.now().isoformat(timespec='seconds')
def stamp(): return datetime.now().strftime('%Y%m%d_%H%M%S')
def write_json(p:Path,x:Any): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2,default=p2.json_default),encoding='utf-8')
def read_json(p:Path)->Any: return json.loads(p.read_text(encoding='utf-8'))
def fmt(x,d=4):
    if x is None: return 'n/a'
    try:
        v=float(x)
        if math.isnan(v) or math.isinf(v): return 'n/a'
        return f'{v:.{d}f}'
    except Exception: return str(x)
def load_feat(root:Path,run:str,split:str): return torch.load(root/run/'features'/f'{split}_features.pt',map_location='cpu',weights_only=False)
def aux(payload,names):
    schema=list(payload['aux_schema']); idx=[schema.index(n) for n in names if n in schema]
    return payload['aux'][:,idx].float().numpy() if idx else np.zeros((len(payload['metadata']),0),np.float32)
def force_derived(payload,source='pred'):
    f=payload['force_pred_n' if source=='pred' else 'force_gt_n'].float().numpy(); fx,fy,fz=f[:,0],f[:,1],f[:,2]
    fn=np.abs(fz); ft=np.sqrt(fx*fx+fy*fy); fmag=np.sqrt(fx*fx+fy*fy+fz*fz)
    return {'Fx':fx,'Fy':fy,'Fz':fz,'Fn':fn,'Ft':ft,'Fmag':fmag,'ratio':ft/(fn+EPS)}
def groups(payload): return np.array([p5.group_name(m) for m in payload['metadata']])
def indices_by_groups(payload,gs): return np.flatnonzero(np.isin(groups(payload),gs)).astype(np.int64)
def metric_get(d,path):
    cur=d
    for part in path.split('.'):
        if not isinstance(cur,dict) or part not in cur: return None
        cur=cur[part]
    return cur

def future_metrics(y,prob):
    y=y.astype(int); pred=(prob>=0.5).astype(int)
    out={'f1':float(f1_score(y,pred,zero_division=0)),'accuracy':float(accuracy_score(y,pred)),'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'positive_ratio':float(y.mean()),'positive_count':int(y.sum())}
    if len(np.unique(y))>1: out['auroc']=float(roc_auc_score(y,prob)); out['auprc']=float(average_precision_score(y,prob))
    else: out['auroc']=out['auprc']=None
    out['ece']=float(wm.calibration_error((1-y).astype(int),1-prob)); return out
def eval_probs(payload,idx,probs,name):
    fut=payload['future_slip'].numpy()[idx].astype(int); out={'condition':name,'n_samples':int(len(idx)),'horizons':list(HORIZONS)}
    for j,h in enumerate(HORIZONS): out[f'H{h}']=future_metrics(fut[:,j],probs[:,j] if probs.ndim>1 else probs)
    return out
def render_metric_table(rows,title):
    lines=[f'# {title}','',f'- generated_at: `{now()}`','- raw_data_modified: `False`','', '| condition | n | H1 F1 | H1 AUPRC | H1 AUROC | H1 ECE | H3 F1 | H3 AUPRC | H5 F1 | H5 AUPRC |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        h1,h3,h5=r.get('H1',{}),r.get('H3',{}),r.get('H5',{})
        lines.append(f"| {r.get('condition')} | {r.get('n_samples')} | {fmt(h1.get('f1') or h1.get('future_slip_f1'))} | {fmt(h1.get('auprc') or h1.get('future_slip_auprc'))} | {fmt(h1.get('auroc') or h1.get('future_slip_auroc'))} | {fmt(h1.get('ece') or h1.get('stability_calibration_error'))} | {fmt(h3.get('f1') or h3.get('future_slip_f1'))} | {fmt(h3.get('auprc') or h3.get('future_slip_auprc'))} | {fmt(h5.get('f1') or h5.get('future_slip_f1'))} | {fmt(h5.get('auprc') or h5.get('future_slip_auprc'))} |")
    return '\n'.join(lines)+'\n'
def plot_bar(rows,out,metric='f1',horizon='H5'):
    if plt is None: return None
    labels=[r['condition'] for r in rows]; vals=[]
    for r in rows:
        hh=r.get(horizon,{}); vals.append(hh.get(metric) if metric in hh else hh.get(f'future_slip_{metric}'))
    fig,ax=plt.subplots(figsize=(max(8,len(labels)*1.2),4)); ax.bar(range(len(labels)),vals); ax.set_xticks(range(len(labels)),labels,rotation=30,ha='right'); ax.set_ylabel(f'{horizon} {metric}'); ax.set_ylim(0,1.02); fig.tight_layout(); out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,dpi=160); plt.close(fig); return str(out)

def temporal_features(payload):
    p=payload['slip_probs'][:,1].float().numpy(); prev=np.zeros_like(p); delta=np.zeros_like(p); buckets={}
    for i,m in enumerate(payload['metadata']): buckets.setdefault((m['dataset'],str(m['trajectory'])),[]).append((int(m['sample']),i))
    for rows in buckets.values():
        last=None
        for _,i in sorted(rows): prev[i]=0.0 if last is None else p[last]; delta[i]=0.0 if last is None else p[i]-p[last]; last=i
    return np.stack([p,prev,delta,(p>=0.5).astype(float)],1)
def train_logistic(trainX,trainY,valX):
    probs=[]
    for j in range(trainY.shape[1]):
        y=trainY[:,j].astype(int)
        if len(np.unique(y))<2: probs.append(np.full(valX.shape[0],float(y.mean()))); continue
        clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,class_weight='balanced',solver='lbfgs',random_state=42)); clf.fit(trainX,y); probs.append(clf.predict_proba(valX)[:,1])
    return np.stack(probs,1)
def step1(args):
    out=REPORT/'step1_future_baselines'; out.mkdir(parents=True,exist_ok=True); tr=load_feat(PHASE5,P5_FRIC,'train'); va=load_feat(PHASE5,P5_FRIC,'val'); idx=np.arange(len(va['metadata'])); rows=[]
    pcur=va['slip_probs'][:,1].float().numpy(); rows.append(eval_probs(va,idx,np.tile(pcur[:,None],(1,3)),'current_slip_persistence'))
    ra=p5.risk_arrays(va,va.get('friction_thresholds_pred_train')); rows.append(eval_probs(va,idx,np.tile(ra['q'][:,None],(1,3)),'friction_q_threshold_score'))
    rows.append(eval_probs(va,idx,np.tile((ra['r_clipped']>=va['friction_thresholds_pred_train']['q_tau']).astype(float)[:,None],(1,3)),'friction_ratio_hard_threshold'))
    ytr=tr['future_slip'].numpy().astype(int); rows.append(eval_probs(va,idx,train_logistic(temporal_features(tr),ytr,temporal_features(va)),'slip_only_temporal_logreg'))
    ftr=force_derived(tr); fva=force_derived(va); Xtr=np.stack([ftr['Fn'],ftr['Ft'],ftr['Fmag'],ftr['ratio']],1); Xva=np.stack([fva['Fn'],fva['Ft'],fva['Fmag'],fva['ratio']],1)
    Xtr=np.concatenate([Xtr,aux(tr,['dFx_causal_N','dFy_causal_N','dFz_causal_N'])],1); Xva=np.concatenate([Xva,aux(va,['dFx_causal_N','dFy_causal_N','dFz_causal_N'])],1); rows.append(eval_probs(va,idx,train_logistic(Xtr,ytr,Xva),'force_only_logreg'))
    for cond,path in [('full_dynamics_existing',WORKSPACE/'reports/phase5/phase5_2_friction_future_head/phase5_2_full_dynamics_baseline_20260518_0035_report.json'),('full_dynamics_plus_q_existing',WORKSPACE/'reports/phase5/phase5_2_friction_future_head/phase5_2_full_plus_q_20260518_0035_report.json')]:
        d=read_json(path); r={'condition':cond,'n_samples':d['best_val']['n_samples']}
        for h in HORIZONS:
            src=d['best_val'][f'H{h}']; r[f'H{h}']={'f1':src.get('future_slip_f1'),'auprc':src.get('future_slip_auprc'),'auroc':src.get('future_slip_auroc'),'ece':src.get('stability_calibration_error')}
        rows.append(r)
    payload={'generated_at':now(),'phase':'phase7_step1_future_baselines','rows':rows,'feature_run_id':P5_FRIC,'raw_data_modified':False}; write_json(out/'step1_future_baselines.json',payload); (out/'step1_future_baselines.md').write_text(render_metric_table(rows,'Phase7 Step1 Future Baselines'),encoding='utf-8')
    figs=[plot_bar(rows,out/'step1_h5_f1.png','f1','H5'),plot_bar(rows,out/'step1_h5_auprc.png','auprc','H5')]; payload['figures']=[f for f in figs if f]; write_json(out/'step1_future_baselines.json',payload); print(out/'step1_future_baselines.md')

def train_head_phase7(root,run_id,condition,aux_names,train_idx,val_idx,report_dir,exp,thresholds=None,max_epochs=40,seed=42):
    return p5.train_future_head(root,run_id,exp,aux_names,train_idx,val_idx,HORIZONS,max_epochs,512,1e-3,512,0.1,seed,'disabled',report_dir,thresholds,condition)
def step2(args):
    out=REPORT/'step2_leave_one_geometry_out'; out.mkdir(parents=True,exist_ok=True); tr_f=load_feat(PHASE5,P5_FRIC,'train'); thresholds=tr_f.get('friction_thresholds_pred_train') or p5.threshold_from_train(tr_f,'pred')
    specs=[('separate_late_fusion',PHASE4,P4_SEP,STATIC,None),('decoupled_static',PHASE4,P4_DEC,STATIC,thresholds),('decoupled_dynamics',PHASE4,P4_DEC,BASE_FULL,thresholds),('decoupled_dynamics_friction',PHASE5,P5_FRIC,FULL_Q,thresholds)]
    rows=[]; st=args.stamp or stamp()
    for cond,root,run_id,aux_names,th in specs:
        tr=load_feat(root,run_id,'train'); va=load_feat(root,run_id,'val'); ti=indices_by_groups(tr,['sharp','sphere']); vi=indices_by_groups(va,['flat']); exp=f'phase7_step2_sharp_sphere_to_flat_{cond}_{st}'; report_json=out/f'{exp}_report.json'
        r=read_json(report_json) if report_json.exists() else train_head_phase7(root,run_id,cond,aux_names,ti,vi,out,exp,th,args.max_epochs,args.seed)
        r['heldout_split']={'train_groups':['sharp','sphere'],'test_groups':['flat']}; rows.append(r)
    prev=[]
    for name,path in [('flat+sharp_to_sphere',WORKSPACE/'reports/phase5/phase5_3_heldout_generalization/phase5_3_heldout_generalization_report.json'),('flat+sphere_to_sharp',WORKSPACE/'reports/phase6/phase6_2_heldout_sharp_generalization/phase6_2_heldout_sharp_generalization_report.json')]:
        if path.exists():
            d=read_json(path)
            for c in d['conditions']: prev.append({'split':name,'condition':c['condition'],'H3_f1':metric_get(c,'best_val.H3.future_slip_f1'),'H5_f1':metric_get(c,'best_val.H5.future_slip_f1'),'H5_auprc':metric_get(c,'best_val.H5.future_slip_auprc')})
    for c in rows: prev.append({'split':'sharp+sphere_to_flat','condition':c['condition'],'H3_f1':metric_get(c,'best_val.H3.future_slip_f1'),'H5_f1':metric_get(c,'best_val.H5.future_slip_f1'),'H5_auprc':metric_get(c,'best_val.H5.future_slip_auprc')})
    payload={'generated_at':now(),'phase':'phase7_step2_leave_one_geometry_out','heldout_split':{'train_groups':['sharp','sphere'],'test_groups':['flat']},'conditions':rows,'combined_leave_one_out':prev,'raw_data_modified':False}; write_json(out/'step2_leave_one_geometry_out.json',payload)
    lines=['# Phase7 Step2 Leave-One-Contact-Geometry-Out','','- raw_data_modified: `False`','','## New split: train sharp+sphere -> test flat','','| condition | current force RMSE | current slip F1 | H1 F1 | H3 F1 | H5 F1 | H5 AUPRC | best epoch |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        cur=r.get('current_metrics_val_subset',{}); lines.append(f"| {r['condition']} | {fmt(cur.get('force_rmse_mean_N'))} | {fmt(cur.get('slip_f1'))} | {fmt(metric_get(r,'best_val.H1.future_slip_f1'))} | {fmt(metric_get(r,'best_val.H3.future_slip_f1'))} | {fmt(metric_get(r,'best_val.H5.future_slip_f1'))} | {fmt(metric_get(r,'best_val.H5.future_slip_auprc'))} | {r.get('best_epoch')} |")
    lines += ['','## Combined three held-out splits','','| split | condition | H3 F1 | H5 F1 | H5 AUPRC |','|---|---|---:|---:|---:|']
    for r in prev: lines.append(f"| {r['split']} | {r['condition']} | {fmt(r['H3_f1'])} | {fmt(r['H5_f1'])} | {fmt(r['H5_auprc'])} |")
    (out/'step2_leave_one_geometry_out.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print(out/'step2_leave_one_geometry_out.md')

def load_future_probs(checkpoint:Path,root:Path):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False); payload=load_feat(root,ck['feature_run_id'],'val'); ds=p5.FeatureHeadDataset(payload,np.arange(len(payload['metadata'])),ck['aux_names'],tuple(ck['horizons'])); dl=DataLoader(ds,batch_size=1024,shuffle=False,num_workers=0); model=wm.MLP(ck['input_dim'],ck['hidden_dim'],len(ck['horizons']),ck['dropout']); model.load_state_dict(ck['model_state']); model.eval(); arr=[]
    with torch.no_grad():
        for b in dl: arr.append(torch.sigmoid(model(b['x'])).numpy())
    return payload,np.concatenate(arr,0)
def warning_stats(meta,current_slip,p_inst,thr):
    buckets={}
    for i,m in enumerate(meta): buckets.setdefault((m['dataset'],str(m['trajectory'])),[]).append((int(m['sample']),int(current_slip[i]),float(p_inst[i])))
    slip_tr=early=late=miss=false_alarm=non=0; leads=[]
    for rows in buckets.values():
        rows=sorted(rows); on=[s for s,sl,_ in rows if sl==1]; warns=[s for s,_,p in rows if p>=thr]
        if on:
            slip_tr+=1; onset=min(on); pre=[s for s in warns if s<onset]
            if pre: early+=1; leads.append(onset-min(pre))
            elif warns: late+=1
            else: miss+=1
        else: non+=1; false_alarm += 1 if warns else 0
    return {'threshold':thr,'trajectory_count':len(buckets),'slip_trajectories':slip_tr,'non_slip_trajectories':non,'detected_pre_slip_trajectories':early,'late_warning_trajectories':late,'missed_slip_trajectories':miss,'false_alarm_trajectories':false_alarm,'early_warning_recall':early/slip_tr if slip_tr else None,'false_alarm_rate':false_alarm/non if non else None,'late_warning_ratio':late/slip_tr if slip_tr else None,'missed_warning_ratio':miss/slip_tr if slip_tr else None,'lead_time_to_slip_onset_steps_mean':float(np.mean(leads)) if leads else None,'lead_time_to_slip_onset_steps_median':float(np.median(leads)) if leads else None,'lead_times':leads}
def step3(args):
    out=REPORT/'step3_early_warning_threshold_sweep'; out.mkdir(parents=True,exist_ok=True); rep=read_json(WORKSPACE/'reports/phase6/phase6_4_early_warning_analysis/phase6_4_early_warning_analysis_report.json'); payload,stable=load_future_probs(Path(rep['checkpoint']),PHASE6); p_inst=1-stable[:,0]; rows=[warning_stats(payload['metadata'],payload['current_slip'].numpy(),p_inst,t) for t in [0.3,0.4,0.5,0.6,0.7,0.8]]
    write_json(out/'step3_early_warning_threshold_sweep.json',{'generated_at':now(),'phase':'phase7_step3_early_warning_threshold_sweep','checkpoint':rep['checkpoint'],'rows':rows,'raw_data_modified':False})
    if plt:
        for key,ylabel in [('early_warning_recall','early recall'),('false_alarm_rate','false alarm rate'),('lead_time_to_slip_onset_steps_mean','mean lead time')]:
            fig,ax=plt.subplots(figsize=(5,3)); ax.plot([r['threshold'] for r in rows],[r[key] for r in rows],marker='o'); ax.set_xlabel('threshold'); ax.set_ylabel(ylabel); fig.tight_layout(); fig.savefig(out/f'step3_threshold_{key}.png',dpi=160); plt.close(fig)
        leads=[x for r in rows if abs(r['threshold']-0.5)<1e-6 for x in r['lead_times']]; fig,ax=plt.subplots(figsize=(5,3)); ax.hist(leads,bins=20); ax.set_xlabel('lead time steps @0.5'); ax.set_ylabel('trajectories'); fig.tight_layout(); fig.savefig(out/'step3_lead_time_hist_threshold_0p5.png',dpi=160); plt.close(fig)
    lines=['# Phase7 Step3 Early-Warning Threshold Sweep','','| threshold | early recall | false alarm rate | late ratio | missed ratio | mean lead | median lead |','|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows: lines.append(f"| {r['threshold']} | {fmt(r['early_warning_recall'])} | {fmt(r['false_alarm_rate'])} | {fmt(r['late_warning_ratio'])} | {fmt(r['missed_warning_ratio'])} | {fmt(r['lead_time_to_slip_onset_steps_mean'])} | {fmt(r['lead_time_to_slip_onset_steps_median'])} |")
    (out/'step3_early_warning_threshold_sweep.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print(out/'step3_early_warning_threshold_sweep.md')

def rmse_row(name,path,kind='aggregate'):
    d=read_json(Path(path)); m=d.get(kind,d); agg=m.get('aggregate',m) if isinstance(m,dict) and 'aggregate' in m else m; drv=agg.get('force_derived_rmse_N',{}); xyz=agg.get('force_rmse_xyz_N',[None,None,None])
    return {'condition':name,'Fx_RMSE':xyz[0],'Fy_RMSE':xyz[1],'Fz_RMSE':xyz[2],'Fn_RMSE':drv.get('Fn') or xyz[2],'Ft_RMSE':drv.get('Ft'),'Fmag_RMSE':drv.get('Fmag'),'force_RMSE_mean':agg.get('force_rmse_mean_N'),'slip_F1':agg.get('slip_f1')}
def step4(args):
    out=REPORT/'step4_force_axis_decomposition'; out.mkdir(parents=True,exist_ok=True); rows=[rmse_row('separate_baseline',WORKSPACE/'reports/phase3/phase3_1_20260516_154730/eval_cache/a_mae_allsource_val.json'),rmse_row('naive_shared_multitask',WORKSPACE/'reports/phase2/phase2_b_gsmini_20260513_163448/eval_cache/b_mae_allsource_val.json'),rmse_row('partially_shared_lambda_0.25',WORKSPACE/'reports/phase2/phase2_b_ps_lam025_gsmini_20260514_063052/eval_cache/b_mae_partially_shared_allsource_val.json'),rmse_row('consistency_decoder',WORKSPACE/'reports/phase2/phase2_c_consistency_gsmini_20260514_161845/eval_cache/c_mae_consistency_phase2_c_consistency_gsmini_20260514_161845_allsource_val.json'),rmse_row('decoupled_multitask',WORKSPACE/'reports/phase3/phase3_1_20260516_154730/eval_cache/decoupled_mae_phase3_1_decoupled_gsmini_20260516_154730_allsource_val.json'),rmse_row('joint_lightweight_force_slip_future',WORKSPACE/'reports/phase6/phase6_3_frozen_vs_joint_ablation/phase6_3_joint_lightweight_seed42_20260518_0154_report.json','current_metrics_val')]
    write_json(out/'step4_force_axis_decomposition.json',{'generated_at':now(),'phase':'phase7_step4_force_axis_decomposition','rows':rows,'raw_data_modified':False})
    if plt:
        labels=[r['condition'] for r in rows]; xs=np.arange(len(labels)); fig,ax=plt.subplots(figsize=(10,4)); w=.13
        for j,k in enumerate(['Fx_RMSE','Fy_RMSE','Fz_RMSE','Fn_RMSE','Ft_RMSE','Fmag_RMSE']): ax.bar(xs+(j-2.5)*w,[r.get(k) or 0 for r in rows],w,label=k)
        ax.set_xticks(xs,labels,rotation=25,ha='right'); ax.set_ylabel('RMSE (N)'); ax.legend(fontsize=7,ncol=3); fig.tight_layout(); fig.savefig(out/'step4_force_axis_decomposition.png',dpi=160); plt.close(fig)
    lines=['# Phase7 Step4 Force Per-Axis / Physical Decomposition','','| condition | Fx | Fy | Fz | Fn | Ft | Fmag | mean | slip F1 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows: lines.append(f"| {r['condition']} | {fmt(r['Fx_RMSE'])} | {fmt(r['Fy_RMSE'])} | {fmt(r['Fz_RMSE'])} | {fmt(r['Fn_RMSE'])} | {fmt(r['Ft_RMSE'])} | {fmt(r['Fmag_RMSE'])} | {fmt(r['force_RMSE_mean'])} | {fmt(r['slip_F1'])} |")
    (out/'step4_force_axis_decomposition.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print(out/'step4_force_axis_decomposition.md')

def step5(args):
    out=REPORT/'step5_multiseed_stability'; out.mkdir(parents=True,exist_ok=True); p4m=read_json(WORKSPACE/'reports/phase4/phase4_1_20260517_144424/phase4_1_mae_multiseed_report.json'); p6m=read_json(WORKSPACE/'reports/phase6/phase6_1_multiseed_stability/phase6_1_multiseed_stability_report.json'); arch=read_json(WORKSPACE/'reports/phase4/phase4_4_architecture_ablation_20260517/phase4_4_architecture_ablation_report.json')
    rows=[]
    for name,prefix in [('separate_baseline','separate'),('decoupled_multitask','decoupled')]: rows.append({'condition':name,'source':'phase4_1 n=3','force_rmse_mean_N':p4m['metric_summary'].get(f'{prefix}_force_rmse_mean_N'),'slip_f1':p4m['metric_summary'].get(f'{prefix}_slip_f1')})
    naive=next(r for r in arch['rows'] if r['condition']=='naive_shared_multitask'); rows.append({'condition':'naive_shared_multitask','source':'phase4_4 single seed only','force_rmse_mean_N':{'mean':naive['force_rmse_mean_N'],'std':0,'n':1},'slip_f1':{'mean':naive['slip_f1'],'std':0,'n':1},'gap':'only seed42 available; missing seeds are recorded, not fabricated'})
    for cond,key in [('full_dynamics_future_head','full_dynamics_baseline'),('full_dynamics_plus_q','full_plus_q')]: rows.append({'condition':cond,'source':'phase6_1 n=3','future':p6m['metric_summary'][key]})
    write_json(out/'step5_multiseed_stability.json',{'generated_at':now(),'phase':'phase7_step5_multiseed_stability','rows':rows,'raw_data_modified':False})
    lines=['# Phase7 Step5 Main Results Multi-Seed Stability','','| condition | source | force RMSE mean±std | slip F1 mean±std | H3 F1 mean±std | H5 F1 mean±std | H5 AUPRC mean±std | note |','|---|---|---:|---:|---:|---:|---:|---|']
    for r in rows:
        fr=r.get('force_rmse_mean_N') or {}; sf=r.get('slip_f1') or {}; fut=r.get('future') or {}
        lines.append(f"| {r['condition']} | {r['source']} | {fmt(fr.get('mean'))}±{fmt(fr.get('std'))} (n={fr.get('n','')}) | {fmt(sf.get('mean'))}±{fmt(sf.get('std'))} (n={sf.get('n','')}) | {fmt((fut.get('H3_f1') or {}).get('mean'))}±{fmt((fut.get('H3_f1') or {}).get('std'))} | {fmt((fut.get('H5_f1') or {}).get('mean'))}±{fmt((fut.get('H5_f1') or {}).get('std'))} | {fmt((fut.get('H5_auprc') or {}).get('mean'))}±{fmt((fut.get('H5_auprc') or {}).get('std'))} | {r.get('gap','')} |")
    (out/'step5_multiseed_stability.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print(out/'step5_multiseed_stability.md')

def step6(args):
    out=REPORT/'step6_case_visualization'; fig=out/'figures'; fig.mkdir(parents=True,exist_ok=True); copied=[]
    for s in [WORKSPACE/'reports/phase6/phase6_4_early_warning_analysis/figures',WORKSPACE/'reports/phase5/phase5_4_visualization_failure_cases/figures']:
        if s.exists():
            for p in sorted(s.glob('*.png')): dst=fig/p.name; shutil.copy2(p,dst); copied.append(str(dst))
    cases=[]; rep=WORKSPACE/'reports/phase6/phase6_4_early_warning_analysis/phase6_4_early_warning_analysis_report.json'
    if rep.exists(): cases=read_json(rep).get('cases',[])
    write_json(out/'step6_case_visualization.json',{'generated_at':now(),'phase':'phase7_step6_case_visualization','figures':copied,'cases':cases,'raw_data_modified':False})
    lines=['# Phase7 Step6 Failure / Success Case Visualization','',f'- copied_figures: `{len(copied)}`','','## Figures']+[f'- `{p}`' for p in copied]
    (out/'step6_case_visualization.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print(out/'step6_case_visualization.md')

def count_params(path,exclude_encoder=False):
    d=torch.load(path,map_location='cpu',weights_only=False); sd=d.get('state_dict') or d.get('model_state') or d.get('model') or d; total=0
    for k,v in sd.items():
        if exclude_encoder and str(k).startswith('encoder.'): continue
        if hasattr(v,'numel'): total+=int(v.numel())
    return total
def future_latency(checkpoint,n_iter=100):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False); model=wm.MLP(ck['input_dim'],ck['hidden_dim'],len(ck['horizons']),ck['dropout']).eval(); model.load_state_dict(ck['model_state']); x=torch.randn(512,ck['input_dim'])
    with torch.no_grad():
        for _ in range(10): model(x)
        t0=time.perf_counter()
        for _ in range(n_iter): model(x)
        dt=time.perf_counter()-t0
    return {'device':'cpu','batch_size':512,'iterations':n_iter,'ms_per_batch':dt/n_iter*1000,'ms_per_sample':dt/n_iter/512*1000}
def step7(args):
    out=REPORT/'step7_runtime_model_size'; out.mkdir(parents=True,exist_ok=True); ck_future=Path('/vla1/zjy/sparsh_runs/force_slip_phase6/phase6_1_friction_features_20260518_0130/heads/phase6_1_full_plus_q_seed42_20260518_0130/checkpoints/best.pth')
    rows=[{'model':'separate_force_head','params':count_params('/vla1/zjy/sparsh_runs/experiments/2026.05.12_04-39_phase1_gsmini_20260512_043331_mae_force_gsmini_20260512_043652/checkpoints/epoch-0051.pth')},{'model':'separate_slip_head','params':count_params('/vla1/zjy/sparsh_runs/experiments/2026.05.13_01-21_phase1_gsmini_20260512_043331_mae_slip_allsource_diag_gsmini_20260513_012000/checkpoints/epoch-0051.pth')},{'model':'naive_shared_total','params':count_params('/vla1/zjy/sparsh_runs/force_slip_phase2/phase2_b_gsmini_20260513_163448/mae_shared_multitask/checkpoints/best_f1.pth')},{'model':'naive_shared_downstream_excl_encoder','params':count_params('/vla1/zjy/sparsh_runs/force_slip_phase2/phase2_b_gsmini_20260513_163448/mae_shared_multitask/checkpoints/best_f1.pth',True)},{'model':'decoupled_total','params':count_params('/vla1/zjy/sparsh_runs/force_slip_phase2/phase3_1_decoupled_gsmini_20260516_154730/mae_decoupled_multitask/checkpoints/epoch-0030.pth')},{'model':'decoupled_downstream_excl_encoder','params':count_params('/vla1/zjy/sparsh_runs/force_slip_phase2/phase3_1_decoupled_gsmini_20260516_154730/mae_decoupled_multitask/checkpoints/epoch-0030.pth',True)},{'model':'future_head_full_plus_q','params':count_params(str(ck_future))}]
    lat=future_latency(ck_future); payload={'generated_at':now(),'phase':'phase7_step7_runtime_model_size','parameter_rows':rows,'future_head_latency':lat,'notes':['CPU latency measured for future head on cached features; encoder extraction time is cache-based/not remeasured to avoid new raw-data passes.','GPU memory was not actively stressed; use nvidia-smi during training for deployment-grade numbers.'],'raw_data_modified':False}; write_json(out/'step7_runtime_model_size.json',payload)
    lines=['# Phase7 Step7 Runtime / Model Size / Inference Cost','','## Parameter counts','','| model | parameters |','|---|---:|']+[f"| {r['model']} | {r['params']} |" for r in rows]+['','## Future head cached-feature latency',f"- device: `{lat['device']}`",f"- batch_size: `{lat['batch_size']}`",f"- ms_per_batch: `{fmt(lat['ms_per_batch'],6)}`",f"- ms_per_sample: `{fmt(lat['ms_per_sample'],6)}`",'','## Notes']+[f"- {n}" for n in payload['notes']]
    (out/'step7_runtime_model_size.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print(out/'step7_runtime_model_size.md')

def provenance(args):
    out=REPORT/'provenance'; out.mkdir(parents=True,exist_ok=True)
    def cmd(c):
        try: return subprocess.check_output(c,cwd=REPO,text=True).strip()
        except Exception as e: return f'ERROR: {e}'
    p={'generated_at':now(),'repo':str(REPO),'branch':cmd(['git','branch','--show-current']),'commit':cmd(['git','rev-parse','HEAD']),'status_short':cmd(['git','status','--short']),'raw_dataset':'/vla1/zjy/tactile_datasets (read-only by protocol)','feature_runs':{'phase4_decoupled':str(PHASE4/P4_DEC),'phase5_friction':str(PHASE5/P5_FRIC),'phase6_friction':str(PHASE6/P6_FRIC)},'raw_data_modified':False}
    write_json(out/'phase7_provenance.json',p); (out/'phase7_provenance.md').write_text('# Phase7 Provenance\n\n```json\n'+json.dumps(p,indent=2)+'\n```\n',encoding='utf-8'); print(out/'phase7_provenance.md')
def summary(args):
    REPORT.mkdir(parents=True,exist_ok=True); sections=[]; payload={'generated_at':now(),'phase':'phase7_summary','steps':{},'raw_data_modified':False}
    for step in ['step1_future_baselines','step2_leave_one_geometry_out','step3_early_warning_threshold_sweep','step4_force_axis_decomposition','step5_multiseed_stability','step6_case_visualization','step7_runtime_model_size']:
        d=REPORT/step; mds_all=sorted(d.glob('*.md')); preferred=d/f'{step}.md'; mds=[preferred] if preferred.exists() else mds_all; js=sorted(d.glob('*.json')); payload['steps'][step]={'dir':str(d),'md':[str(x) for x in mds_all],'json':[str(x) for x in js],'complete':bool(mds_all and js)}; sections.append(f'## {step}\n'); sections.append((mds[0].read_text(encoding='utf-8')[:6000]+'\n') if mds else '- missing\n')
    write_json(REPORT/'phase7_summary.json',payload); text='# Phase7 Summary\n\n- generated_at: `'+now()+'`\n- raw_data_modified: `False`\n\n'+'\n'.join(sections); (REPORT/'phase7_summary.md').write_text(text,encoding='utf-8'); (REPORT/'phase7_all_results.md').write_text(text,encoding='utf-8')
    tr=WORKSPACE/'reports/training_results.md'; old=tr.read_text(encoding='utf-8') if tr.exists() else ''; marker='## 10. Phase7 supplemental experiments'; block=marker+'\n\nPhase7 added future baselines, completed the sharp+sphere→flat held-out geometry split, early-warning threshold sweep, force-axis decomposition, multi-seed aggregation, case visualizations, and runtime/model-size reporting. See `reports/phase7/phase7_all_results.md`. Raw datasets were not modified.\n'
    if marker not in old: tr.write_text(old.rstrip()+'\n\n'+block,encoding='utf-8')
    print(REPORT/'phase7_summary.md')

def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='cmd',required=True)
    sub.add_parser('provenance').set_defaults(func=provenance); sub.add_parser('step1').set_defaults(func=step1)
    p=sub.add_parser('step2'); p.add_argument('--stamp',default=None); p.add_argument('--max-epochs',type=int,default=40); p.add_argument('--seed',type=int,default=42); p.set_defaults(func=step2)
    for s,f in [('step3',step3),('step4',step4),('step5',step5),('step6',step6),('step7',step7),('summary',summary)]: sub.add_parser(s).set_defaults(func=f)
    p=sub.add_parser('all'); p.add_argument('--step2-epochs',type=int,default=40); p.set_defaults(func=None)
    args=ap.parse_args()
    if args.cmd=='all':
        provenance(args); step1(args); args.max_epochs=args.step2_epochs; args.seed=42; args.stamp=None; step2(args); step3(args); step4(args); step5(args); step6(args); step7(args); summary(args)
    else: args.func(args)
if __name__=='__main__': main()
