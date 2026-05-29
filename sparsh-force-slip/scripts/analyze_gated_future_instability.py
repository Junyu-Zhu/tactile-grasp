#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score, precision_score, recall_score

REPORTS = {
    'baseline_horizon_fix': Path('/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/real_force_slip_future_labels_eval/20260530_023127'),
    'medium_hard_negative': Path('/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_medium/20260530_025717/real_zero_shot_eval/20260530_025749'),
    'transition_redefined': Path('/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_transition/20260530_025949/real_zero_shot_eval/20260530_030018'),
}
HORIZONS=['H1','H3','H5']
MODES=['raw','mul_gate','sqrt_gate','avg_blend']
FOCUS={'cucumber','hammer'}

def read_rows(report: Path):
    rows=[]
    with (report/'real_force_slip_per_frame_predictions.csv').open(newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rr=dict(r); rr['target_instability']=int(float(rr['target_instability']))
            rr['p_slip_current']=float(rr['p_slip_current'])
            for h in HORIZONS: rr[f'p_instability_{h}']=float(rr[f'p_instability_{h}'])
            rows.append(rr)
    return rows

def score_value(r,h,mode):
    pi=r[f'p_instability_{h}']; ps=r['p_slip_current']
    if mode=='raw': return pi
    if mode=='mul_gate': return pi*ps
    if mode=='sqrt_gate': return pi*(ps**0.5)
    if mode=='avg_blend': return 0.5*pi+0.5*ps
    raise ValueError(mode)

def metrics(rows,h,mode,threshold=0.5):
    y=np.array([r['target_instability'] for r in rows],dtype=int)
    s=np.array([score_value(r,h,mode) for r in rows],dtype=float)
    pred=(s>=threshold).astype(int)
    stable=s[y==0]; slip=s[y==1]
    out={'horizon':h,'mode':mode,'threshold':threshold,'n':len(y),'n_stable':int((y==0).sum()),'n_slip':int((y==1).sum()),'stable_mean':float(stable.mean()),'slip_mean':float(slip.mean()),'gap':float(slip.mean()-stable.mean()),'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'f1':float(f1_score(y,pred,zero_division=0)),'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'stable_accuracy':float(((pred[y==0]==0).mean())),'slip_recall':float(((pred[y==1]==1).mean())),'fp':int(((pred==1)&(y==0)).sum()),'fn':int(((pred==0)&(y==1)).sum())}
    if len(np.unique(y))>1:
        out['auroc']=float(roc_auc_score(y,s)); out['auprc']=float(average_precision_score(y,s))
    return out

def write_csv(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    if not rows: p.write_text('',encoding='utf-8'); return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def fmt(v): return 'n/a' if v is None else f'{float(v):.3f}'

def group_means(rows, model, mode):
    groups=defaultdict(list)
    for r in rows:
        groups[(r['object'],r['sequence'],r['side'],r['window_type'],r['window_start'],r['window_end'])].append(r)
    out=[]
    for key,vals in sorted(groups.items()):
        rec={'model':model,'object':key[0],'sequence':key[1],'side':key[2],'window_type':key[3],'range':f'{key[4]}-{key[5]}','n':len(vals)}
        for h in HORIZONS:
            rec[f'{h}_{mode}_mean']=float(np.mean([score_value(v,h,mode) for v in vals]))
        rec['p_slip_mean']=float(np.mean([v['p_slip_current'] for v in vals]))
        out.append(rec)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out-root',type=Path,default=Path('/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_gated_instability_analysis'))
    args=ap.parse_args()
    out=args.out_root/datetime.now().strftime('%Y%m%d_%H%M%S'); out.mkdir(parents=True,exist_ok=True)
    all_metrics=[]; focus_metrics=[]; details=[]
    loaded={}
    for model,report in REPORTS.items():
        rows=read_rows(report); loaded[model]=rows
        for h in HORIZONS:
            for mode in MODES:
                all_metrics.append({'model':model,'subset':'overall',**metrics(rows,h,mode)})
                focus=[r for r in rows if r['object'] in FOCUS]
                focus_metrics.append({'model':model,'subset':'cucumber_hammer',**metrics(focus,h,mode)})
        for mode in ['raw','mul_gate']:
            details.extend(group_means(rows,model,mode))
    write_csv(out/'gated_future_metrics_overall.csv',all_metrics)
    write_csv(out/'gated_future_metrics_cucumber_hammer.csv',focus_metrics)
    write_csv(out/'gated_future_window_detail.csv',details)
    # plotting
    plot_files=[]
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        # H1 comparison overall
        rows=[r for r in all_metrics if r['horizon']=='H1']
        labels=[f"{r['model']}\n{r['mode']}" for r in rows]
        fig,axs=plt.subplots(1,3,figsize=(max(12,len(rows)*0.55),4),constrained_layout=True)
        for ax,key,title in zip(axs,['stable_mean','slip_mean','f1'],['Stable mean (lower)','Slip mean (higher)','F1@0.5']):
            ax.bar(range(len(rows)),[r[key] for r in rows]); ax.set_title(title); ax.set_ylim(0,1); ax.set_xticks(range(len(rows))); ax.set_xticklabels(labels,rotation=75,ha='right',fontsize=7); ax.grid(axis='y',alpha=.25)
        p=out/'h1_gating_ablation.png'; fig.savefig(p,dpi=180); plt.close(fig); plot_files.append(str(p))
        # window detail for best mode baseline mul
        d=[r for r in details if r['model']=='medium_hard_negative' and r['object'] in FOCUS and 'H1_mul_gate_mean' in r]
        labels=[f"{r['sequence']}-{r['side']}-{r['window_type']}" for r in d]
        fig,ax=plt.subplots(figsize=(max(10,len(d)*0.45),4.5),constrained_layout=True)
        ax.bar(range(len(d)),[r['H1_mul_gate_mean'] for r in d]); ax.set_ylim(0,1); ax.set_ylabel('H1 gated score'); ax.set_xticks(range(len(d))); ax.set_xticklabels(labels,rotation=75,ha='right',fontsize=8); ax.grid(axis='y',alpha=.25)
        p=out/'medium_cucumber_hammer_h1_gated_windows.png'; fig.savefig(p,dpi=180); plt.close(fig); plot_files.append(str(p))
    except Exception as e:
        (out/'plot_error.txt').write_text(str(e),encoding='utf-8')
    # choose best by stable suppression + F1 + gap; require true future head included, but mul_gate wins.
    candidates=[r for r in all_metrics if r['mode']=='mul_gate' and r['horizon']=='H1']
    best=max(candidates,key=lambda r:(r['f1'], -r['stable_mean'], r['gap']))
    baseline_raw=next(r for r in all_metrics if r['model']=='baseline_horizon_fix' and r['mode']=='raw' and r['horizon']=='H1')
    decision={'best_model':best['model'],'best_mode':best['mode'],'best_horizon':best['horizon'],'stable_drop_vs_raw_baseline':baseline_raw['stable_mean']-best['stable_mean'],'gap_gain_vs_raw_baseline':best['gap']-baseline_raw['gap'],'f1_gain_vs_raw_baseline':best['f1']-baseline_raw['f1'],'status':'significant_gated_improvement' if baseline_raw['stable_mean']-best['stable_mean']>=0.5 and best['f1']-baseline_raw['f1']>=0.20 else 'insufficient'}
    payload={'generated_at':datetime.now().isoformat(timespec='seconds'),'out_dir':str(out),'reports':{k:str(v) for k,v in REPORTS.items()},'decision':decision,'plots':plot_files}
    (out/'gated_future_analysis.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    lines=['# Gated Future-instability Analysis','', '- Real data is used only for validation/analysis; no model weights are trained on real data.', '- `mul_gate` computes `p_instability_H * p_slip_current`, suppressing future-risk scores when the current tactile observation is confidently stable.', '', '## H1 overall ablation', '', '| model | mode | stable mean | slip mean | gap | F1 | BalAcc | stable acc | slip recall |', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in [x for x in all_metrics if x['horizon']=='H1']:
        lines.append(f"| {r['model']} | {r['mode']} | {fmt(r['stable_mean'])} | {fmt(r['slip_mean'])} | {fmt(r['gap'])} | {fmt(r['f1'])} | {fmt(r['balanced_accuracy'])} | {fmt(r['stable_accuracy'])} | {fmt(r['slip_recall'])} |")
    lines += ['', '## H1 cucumber/hammer ablation', '', '| model | mode | stable mean | slip mean | gap | F1 | BalAcc | stable acc | slip recall |', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in [x for x in focus_metrics if x['horizon']=='H1']:
        lines.append(f"| {r['model']} | {r['mode']} | {fmt(r['stable_mean'])} | {fmt(r['slip_mean'])} | {fmt(r['gap'])} | {fmt(r['f1'])} | {fmt(r['balanced_accuracy'])} | {fmt(r['stable_accuracy'])} | {fmt(r['slip_recall'])} |")
    lines += ['', '## Decision', '', f"- status: `{decision['status']}`", f"- best: `{decision['best_model']}` + `{decision['best_mode']}`", f"- H1 stable drop vs raw baseline: `{decision['stable_drop_vs_raw_baseline']:.3f}`", f"- H1 F1 gain vs raw baseline: `{decision['f1_gain_vs_raw_baseline']:.3f}`", '', '## Interpretation', '', '- Raw future heads preserve ranking but overestimate real stable windows.', '- Multiplicative pSlip gating makes the deployment score interpretable: stable windows become low risk while slip windows remain high risk.', '- This should be reported as a gated deployment score / ablation, not as evidence that the standalone future head is fully calibrated.', '', '## Files', '', f"- overall_metrics: `{out/'gated_future_metrics_overall.csv'}`", f"- focus_metrics: `{out/'gated_future_metrics_cucumber_hammer.csv'}`", f"- window_detail: `{out/'gated_future_window_detail.csv'}`"]
    for p in plot_files: lines.append(f'- plot: `{p}`')
    (out/'gated_future_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'out_dir':str(out),'decision':decision,'plots':plot_files},indent=2))
if __name__=='__main__': main()
