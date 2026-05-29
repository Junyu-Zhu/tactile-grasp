#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math, sys, subprocess
from pathlib import Path
from datetime import datetime
from typing import Any
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
SCRIPT_DIR=Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path: sys.path.insert(0,str(SCRIPT_DIR))
import phase_future_head_transition as tr
import phase3_2_world_model as wm

REPO=Path('/home/zjy/document/tactile-grasp')
WORKSPACE=REPO/'sparsh-force-slip'
REAL_EVAL=WORKSPACE/'scripts/evaluate_real_force_slip_model_test.py'
QUICK=WORKSPACE/'scripts/quick_real_future_analysis.py'
REPORT_ROOT=WORKSPACE/'reports/future_head_original_val_calibration'
RUN_ROOT=Path('/vla1/zjy/sparsh_runs/force_slip_future_calibrated')
PREV_REAL=WORKSPACE/'reports/real_force_slip_future_labels_eval/20260530_023127'
HORIZONS=[1,3,5]

def write_json(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8')
def write_csv(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    if not rows: p.write_text('',encoding='utf-8'); return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
def read_metrics_csv(path):
    out={}
    with path.open(newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            out[r['score']]={k:(float(v) if v not in ('','None') else None) for k,v in r.items() if k!='score'}
    return out

def logits_from_p(p):
    p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))

def run_real(ckpt,root):
    cmd=f"cd {REPO} && source /home/zjy/miniconda3/etc/profile.d/conda.sh && conda activate sparsh && export PYTHONPATH=/home/zjy/document/sparsh:. && CUDA_VISIBLE_DEVICES=0 python {REAL_EVAL} --window-source future_labels --batch-size 64 --stage2-checkpoint {ckpt} --report-root {root}"
    subprocess.run(['bash','-lc',cmd],check=True)
    return sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p:p.stat().st_mtime)[-1]
def run_quick(source,root):
    cmd=f"cd {REPO} && source /home/zjy/miniconda3/etc/profile.d/conda.sh && conda activate sparsh && python {QUICK} --source-report {source} --output-root {root}"
    subprocess.run(['bash','-lc',cmd],check=True)
    return sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p:p.stat().st_mtime)[-1]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--stamp',default=None)
    args=ap.parse_args()
    st=args.stamp or datetime.now().strftime('%Y%m%d_%H%M%S')
    report=REPORT_ROOT/st; run=RUN_ROOT/st
    report.mkdir(parents=True,exist_ok=True); run.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    schema=ckpt.get('input_schema') or ['z','aux']
    state=ckpt['model_state']; in_dim=int(state['net.1.weight'].shape[1]); hidden=int(state['net.1.weight'].shape[0])
    model=wm.MLP(in_dim,hidden,3,dropout=float(ckpt.get('dropout',0.1))).to(device); model.load_state_dict(state); model.eval()
    val=tr.load_features('val'); X=tr.make_x(val,schema); P=tr.probs(model,X,device); Y=tr.instability_target(val).numpy().astype(int)
    thresholds=[]
    for j,h in enumerate(HORIZONS):
        candidates=[]
        for t in np.linspace(0.05,0.995,190):
            pred=(P[:,j]>=t).astype(int)
            bal=balanced_accuracy_score(Y[:,j],pred)
            f1=f1_score(Y[:,j],pred,zero_division=0)
            stable_acc=((pred[Y[:,j]==0]==0).mean() if (Y[:,j]==0).any() else 0)
            recall=((pred[Y[:,j]==1]==1).mean() if (Y[:,j]==1).any() else 0)
            # prefer stable suppression while keeping recall usable
            score=bal + 0.10*stable_acc + 0.02*f1 - max(0,0.70-recall)*0.25
            candidates.append({'horizon':f'H{h}','threshold':float(t),'balanced_accuracy':float(bal),'f1':float(f1),'stable_accuracy':float(stable_acc),'recall':float(recall),'score':float(score)})
        best=max(candidates,key=lambda r:r['score']); thresholds.append(best)
    # Adjust stable-logit final bias so p_inst_new>=0.5 iff p_inst_raw>=threshold.
    new_state={k:v.clone() for k,v in state.items()}
    bias_key='net.7.bias'
    if bias_key not in new_state: raise SystemExit(f'{bias_key} missing')
    shifts=[]
    for r in thresholds:
        t=r['threshold']; shifts.append(float(-logits_from_p(1.0-t))) # L stable shift = -logit(1-t)
    new_state[bias_key]=new_state[bias_key]+torch.tensor(shifts,dtype=new_state[bias_key].dtype)
    ckpt['model_state']=new_state
    ckpt['calibration']={'type':'original_val_bias_shift','thresholds':thresholds,'stable_logit_bias_shift':shifts,'real_data_used':False,'source_checkpoint':str(args.checkpoint)}
    out_ckpt=run/'checkpoints'/'best_original_val_calibrated.pth'; out_ckpt.parent.mkdir(parents=True,exist_ok=True); torch.save(ckpt,out_ckpt)
    real_dir=run_real(out_ckpt, report/'real_zero_shot_eval')
    quick_dir=run_quick(real_dir, report/'quick_analysis')
    prev=read_metrics_csv(PREV_REAL/'real_force_slip_binary_metrics.csv'); new=read_metrics_csv(real_dir/'real_force_slip_binary_metrics.csv')
    h1_old=prev['p_instability_H1']; h1_new=new['p_instability_H1']
    decision={'stable_drop_H1':h1_old['stable_mean']-h1_new['stable_mean'],'gap_gain_H1':h1_new['slip_minus_stable_gap']-h1_old['slip_minus_stable_gap'],'f1_gain_H1':h1_new['f1_at_0p5']-h1_old['f1_at_0p5']}
    decision['status']='significant_calibrated_improvement' if decision['stable_drop_H1']>=0.10 and decision['gap_gain_H1']>=0.05 and decision['f1_gain_H1']>=0.10 else 'insufficient_calibrated_improvement'
    payload={'generated_at':datetime.now().isoformat(timespec='seconds'),'source_checkpoint':str(args.checkpoint),'calibrated_checkpoint':str(out_ckpt),'report_dir':str(report),'real_report_dir':str(real_dir),'quick_analysis_dir':str(quick_dir),'thresholds':thresholds,'bias_shifts':shifts,'previous_real_metrics':prev,'new_real_metrics':new,'decision':decision}
    write_json(report/'original_val_calibration_report.json',payload)
    lines=['# Original-validation Calibrated Future Head','',f'- source_checkpoint: `{args.checkpoint}`',f'- calibrated_checkpoint: `{out_ckpt}`','- calibration data: original flat/sharp/sphere validation only; real data remains validation-only.','', '## Chosen original-val thresholds','','| horizon | threshold | bal acc | F1 | stable acc | recall |','|---|---:|---:|---:|---:|---:|']
    for r in thresholds: lines.append(f"| {r['horizon']} | {r['threshold']:.3f} | {r['balanced_accuracy']:.3f} | {r['f1']:.3f} | {r['stable_accuracy']:.3f} | {r['recall']:.3f} |")
    lines += ['', '## Real comparison', '', '| score | old stable | old slip | new stable | new slip | old gap | new gap | old F1 | new F1 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for score in ['p_instability_H1','p_instability_H3','p_instability_H5']:
        o=prev[score]; n=new[score]
        lines.append(f"| {score} | {o['stable_mean']:.3f} | {o['slip_mean']:.3f} | {n['stable_mean']:.3f} | {n['slip_mean']:.3f} | {o['slip_minus_stable_gap']:.3f} | {n['slip_minus_stable_gap']:.3f} | {o['f1_at_0p5']:.3f} | {n['f1_at_0p5']:.3f} |")
    lines += ['', f"## Decision: {decision['status']}", '', json.dumps(decision,indent=2)]
    (report/'original_val_calibration_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'report_dir':str(report),'calibrated_checkpoint':str(out_ckpt),'real_report_dir':str(real_dir),'quick_analysis_dir':str(quick_dir),'decision':decision},indent=2))
if __name__=='__main__': main()
