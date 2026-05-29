#!/usr/bin/env python3
"""Diagnose and retrain Stage-II future-instability heads without using real data for fitting.

Real banana/cucumber/hammer/lemon tactile data is used only once at the end for
zero-shot validation. All training, model selection, and optional calibration are
based on cached flat/sharp/sphere features.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import phase3_2_world_model as wm  # noqa: E402
import phase2_b_multitask as p2  # noqa: E402

REPO = Path('/home/zjy/document/tactile-grasp')
WORKSPACE = REPO / 'sparsh-force-slip'
FEATURE_RUN = Path('/vla1/zjy/sparsh_runs/force_slip_phase4/phase_lambda_best_future_features_20260528_000000')
OLD_HEAD = FEATURE_RUN / 'heads/phase_lambda_best_full_20260528_000000/checkpoints/best.pth'
RUN_ROOT = Path('/vla1/zjy/sparsh_runs/force_slip_future_head_horizon_fix')
REPORT_ROOT = WORKSPACE / 'reports/future_head_horizon_fix'
REAL_DATASET = Path('/vla1/zjy/tactile_dataset/real-force-slip-model-test-dataset')
REAL_EVAL_SCRIPT = WORKSPACE / 'scripts/evaluate_real_force_slip_model_test.py'
AUX_NAMES = ['Fn_pred_N', 'Ft_pred_N', 'Ft_over_Fn_pred', 'p_slip_current', 'dFx_causal_N', 'dFy_causal_N', 'dFz_causal_N']
GEOMS = ('flat', 'sharp', 'sphere')


def stamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def jdefault(obj: Any) -> Any:
    if isinstance(obj, Path): return str(obj)
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, torch.Tensor): return obj.detach().cpu().tolist()
    return str(obj)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=jdefault), encoding='utf-8')


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8'); return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def geom_of_dataset(name: str) -> str:
    s = name.lower()
    for g in GEOMS:
        if s.startswith(g + '_') or s.startswith(g + '/') or g in s.split('_'):
            return g
    return 'unknown'


def load_features(split: str) -> dict[str, Any]:
    return torch.load(FEATURE_RUN / 'features' / f'{split}_features.pt', map_location='cpu', weights_only=False)


def feature_x(payload: dict[str, Any]) -> torch.Tensor:
    return torch.cat([payload['z'].float(), payload['aux'].float()], dim=1)


def mask_by_geoms(payload: dict[str, Any], include: set[str] | None = None, exclude: set[str] | None = None) -> np.ndarray:
    flags=[]
    for m in payload['metadata']:
        g=geom_of_dataset(m['dataset'])
        ok=True
        if include is not None: ok = g in include
        if exclude is not None and g in exclude: ok = False
        flags.append(ok)
    return np.asarray(flags, dtype=bool)


class FeatureDataset(Dataset):
    def __init__(self, payload: dict[str, Any], mask: np.ndarray | None = None, sample_weights: np.ndarray | None = None, label_smoothing: float = 0.0) -> None:
        X=feature_x(payload)
        Y=payload['future_slip'].float()  # instability labels H1/H3/H5
        if mask is not None:
            idx=torch.tensor(np.where(mask)[0], dtype=torch.long)
            X=X[idx]; Y=Y[idx]
            if sample_weights is not None: sample_weights=sample_weights[mask]
        self.x=X
        self.y_inst=Y
        self.y_stable=1.0-Y
        if label_smoothing>0:
            eps=float(label_smoothing)
            self.y_stable = self.y_stable*(1-eps) + 0.5*eps
        self.sample_weights=torch.tensor(sample_weights, dtype=torch.float32) if sample_weights is not None else torch.ones(len(self.x))
    def __len__(self) -> int: return int(self.x.shape[0])
    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {'x': self.x[i], 'y_stable': self.y_stable[i], 'y_inst': self.y_inst[i], 'w': self.sample_weights[i]}


def hard_negative_weights(payload: dict[str, Any]) -> np.ndarray:
    aux=payload['aux'].float().numpy()
    fut=payload['future_slip'].float().numpy()
    # hard negatives: no future slip for all horizons, but high force/ratio/current pSlip.
    fn, ft, ratio, pslip = aux[:,0], aux[:,1], aux[:,2], aux[:,3]
    score = np.nan_to_num(fn) + np.nan_to_num(ft) + np.nan_to_num(ratio) + np.nan_to_num(pslip)
    thr=np.quantile(score, 0.75)
    neg=(fut.sum(axis=1)==0)
    w=np.ones(len(score), dtype=np.float32)
    w[neg & (score>=thr)] = 2.0
    return w


def load_head(path: Path, input_dim: int, hidden_dim: int = 512, dropout: float = 0.1, device: torch.device | str = 'cpu') -> torch.nn.Module:
    payload=torch.load(path, map_location='cpu', weights_only=False)
    state=payload['model_state']
    if 'net.1.weight' in state:
        hidden_dim=int(state['net.1.weight'].shape[0])
        input_dim=int(state['net.1.weight'].shape[1])
    horizons=payload.get('horizons',[1,3,5])
    model=wm.MLP(input_dim, hidden_dim, len(horizons), dropout=dropout).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def probs_from_model(model: torch.nn.Module, X: torch.Tensor, device: torch.device, batch_size: int=4096) -> np.ndarray:
    out=[]; model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            logits=model(X[i:i+batch_size].to(device))
            p_inst=1.0-torch.sigmoid(logits)  # model predicts stable prob
            out.append(p_inst.cpu())
    return torch.cat(out, dim=0).numpy()


def binary_metrics(y: np.ndarray, p: np.ndarray, prefix: str='') -> dict[str, Any]:
    y=y.astype(int); pred=(p>=0.5).astype(int)
    out={
        f'{prefix}n': int(len(y)), f'{prefix}positive_ratio': float(y.mean()) if len(y) else None,
        f'{prefix}mean_stable_label_score': float(p[y==0].mean()) if (y==0).any() else None,
        f'{prefix}mean_inst_label_score': float(p[y==1].mean()) if (y==1).any() else None,
        f'{prefix}std_stable_label_score': float(p[y==0].std()) if (y==0).any() else None,
        f'{prefix}std_inst_label_score': float(p[y==1].std()) if (y==1).any() else None,
        f'{prefix}gap': (float(p[y==1].mean()-p[y==0].mean()) if (y==0).any() and (y==1).any() else None),
        f'{prefix}saturation_ge_0p99': float((p>=0.99).mean()) if len(p) else None,
        f'{prefix}stable_saturation_ge_0p99': float((p[y==0]>=0.99).mean()) if (y==0).any() else None,
        f'{prefix}inst_saturation_ge_0p99': float((p[y==1]>=0.99).mean()) if (y==1).any() else None,
        f'{prefix}f1': float(f1_score(y,pred,zero_division=0)),
        f'{prefix}accuracy': float(accuracy_score(y,pred)),
        f'{prefix}balanced_accuracy': float(balanced_accuracy_score(y,pred)),
        f'{prefix}precision': float(precision_score(y,pred,zero_division=0)),
        f'{prefix}recall': float(recall_score(y,pred,zero_division=0)),
    }
    if len(np.unique(y))>1:
        out[f'{prefix}auroc']=float(roc_auc_score(y,p)); out[f'{prefix}auprc']=float(average_precision_score(y,p))
    else:
        out[f'{prefix}auroc']=None; out[f'{prefix}auprc']=None
    return out


def eval_prediction_matrix(payload: dict[str, Any], probs: np.ndarray, mask: np.ndarray | None, tag: str) -> list[dict[str, Any]]:
    y=payload['future_slip'].numpy().astype(int)
    metas=payload['metadata']
    if mask is None: mask=np.ones(len(y), dtype=bool)
    rows=[]
    for subset_name, subset_mask in [('all', mask), *[(g, mask & mask_by_geoms(payload, include={g})) for g in GEOMS]]:
        if not subset_mask.any(): continue
        for j,h in enumerate([1,3,5]):
            m=binary_metrics(y[subset_mask,j], probs[subset_mask,j])
            rows.append({'tag': tag, 'subset': subset_name, 'horizon': f'H{h}', **m})
    return rows


def train_one(train_payload: dict[str, Any], val_payload: dict[str, Any], train_mask: np.ndarray, val_mask: np.ndarray, exp: str, loss_mode: str, out_dir: Path, epochs: int, batch_size: int, lr: float, hidden_dim: int, dropout: float, device: torch.device) -> dict[str, Any]:
    smooth = 0.05 if 'smooth' in loss_mode else 0.0
    base_weights=hard_negative_weights(train_payload)
    train_ds=FeatureDataset(train_payload, train_mask, base_weights, label_smoothing=smooth)
    val_X=feature_x(val_payload)
    val_y=val_payload['future_slip'].numpy().astype(int)
    input_dim=int(train_ds.x.shape[1])
    model=wm.MLP(input_dim, hidden_dim, 3, dropout=dropout).to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    pos_weight=None
    if 'balanced' in loss_mode:
        y_st=train_ds.y_stable.numpy()
        pos=y_st.sum(axis=0); neg=len(y_st)-pos
        pos_weight=torch.tensor(neg/np.maximum(pos,1.0), dtype=torch.float32, device=device)
    sampler=None
    loader=DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    best_score=-1e9; best_state=None; history=[]
    for epoch in range(1, epochs+1):
        model.train(); losses=[]
        for b in loader:
            x=b['x'].to(device); target=b['y_stable'].to(device); sw=b['w'].to(device)
            logits=model(x)
            loss_raw=F.binary_cross_entropy_with_logits(logits,target,pos_weight=pos_weight,reduction='none')
            # hard-negative weights apply per sample, all horizons.
            loss=(loss_raw.mean(dim=1)*sw).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),10.0); opt.step()
            losses.append(float(loss.detach().cpu()))
        probs=probs_from_model(model, val_X, device)
        rows=eval_prediction_matrix(val_payload, probs, val_mask, 'val')
        all_rows=[r for r in rows if r['subset']=='all']
        mean_auprc=float(np.mean([r['auprc'] for r in all_rows if r['auprc'] is not None]))
        stable_sat=float(np.mean([r['stable_saturation_ge_0p99'] for r in all_rows if r['stable_saturation_ge_0p99'] is not None]))
        score=mean_auprc - 0.10*stable_sat
        rec={'epoch':epoch,'loss':float(np.mean(losses)),'mean_val_auprc':mean_auprc,'mean_val_stable_saturation':stable_sat,'selection_score':score}
        history.append(rec)
        if score>best_score:
            best_score=score; best_state={k:v.detach().cpu() for k,v in model.state_dict().items()}; best_epoch=epoch
    model.load_state_dict(best_state)
    ckpt=out_dir/'heads'/exp/'checkpoints'/'best.pth'; ckpt.parent.mkdir(parents=True, exist_ok=True)
    probs=probs_from_model(model,val_X,device)
    val_rows=eval_prediction_matrix(val_payload, probs, val_mask, exp)
    payload={
        'experiment_name': exp, 'loss_mode': loss_mode, 'best_epoch': best_epoch, 'best_selection_score': best_score,
        'model_state': model.state_dict(), 'horizons':[1,3,5], 'aux_names': AUX_NAMES, 'input_dim': input_dim, 'hidden_dim': hidden_dim, 'dropout': dropout,
        'feature_run': str(FEATURE_RUN), 'target': 'stable logits; p_instability=1-sigmoid(logits)', 'raw_data_modified': False,
    }
    torch.save(payload, ckpt)
    write_json(out_dir/'heads'/exp/'history.json', history)
    write_csv(out_dir/'heads'/exp/'val_metrics.csv', val_rows)
    return {'experiment_name':exp,'loss_mode':loss_mode,'checkpoint':str(ckpt),'best_epoch':best_epoch,'best_score':best_score,'val_metrics':val_rows,'history':history}


def summarize_metrics(rows: list[dict[str, Any]], tag: str) -> dict[str, Any]:
    all_rows=[r for r in rows if r.get('subset')=='all']
    return {
        'tag': tag,
        'mean_auprc': float(np.mean([r['auprc'] for r in all_rows if r.get('auprc') is not None])) if all_rows else None,
        'mean_auroc': float(np.mean([r['auroc'] for r in all_rows if r.get('auroc') is not None])) if all_rows else None,
        'mean_f1': float(np.mean([r['f1'] for r in all_rows])) if all_rows else None,
        'mean_balanced_accuracy': float(np.mean([r['balanced_accuracy'] for r in all_rows])) if all_rows else None,
        'mean_stable_saturation_ge_0p99': float(np.mean([r['stable_saturation_ge_0p99'] for r in all_rows if r.get('stable_saturation_ge_0p99') is not None])) if all_rows else None,
        'H1_auprc': next((r.get('auprc') for r in all_rows if r.get('horizon')=='H1'), None),
        'H3_auprc': next((r.get('auprc') for r in all_rows if r.get('horizon')=='H3'), None),
        'H5_auprc': next((r.get('auprc') for r in all_rows if r.get('horizon')=='H5'), None),
    }


def render_report(payload: dict[str, Any]) -> str:
    def fmt(v):
        if v is None: return 'n/a'
        try: return f'{float(v):.4f}'
        except Exception: return str(v)
    lines=['# Future Head Horizon Fix Report','',f"- generated_at: `{payload['generated_at']}`",f"- report_dir: `{payload['report_dir']}`",f"- feature_run: `{payload['feature_run']}`",f"- old_head: `{payload['old_head']}`",'- real data policy: banana/cucumber/hammer/lemon data was used only for final zero-shot validation, not training, calibration, threshold fitting, or model selection.','']
    lines += ['## Label audit','', '- Existing cached labels are future-horizon labels generated from `labels[sample+1:sample+h+1]`; they are not direct copies of current slip labels.', '- H1/H3/H5 therefore mean whether slip/instability occurs within the next 1/3/5 sampled steps in the original flat/sharp/sphere dataset.', '']
    lines += ['## Original-data diagnostics and trained heads','','| tag | mean AUPRC | mean AUROC | mean F1 | mean BalAcc | stable saturation>=0.99 | ckpt |','|---|---:|---:|---:|---:|---:|---|']
    for r in payload['selection_rows']:
        lines.append(f"| {r['tag']} | {fmt(r.get('mean_auprc'))} | {fmt(r.get('mean_auroc'))} | {fmt(r.get('mean_f1'))} | {fmt(r.get('mean_balanced_accuracy'))} | {fmt(r.get('mean_stable_saturation_ge_0p99'))} | `{r.get('checkpoint','')}` |")
    lines += ['', f"- selected_best_head: `{payload['best_head']['checkpoint']}`", f"- selected_reason: `{payload['best_head']['selection_reason']}`", '']
    lines += ['## Real zero-shot summary','','| object | windows | accuracy | stable_acc | inst_acc | stable H1 sat old/new | H1 stable mean old/new | H1 inst mean old/new |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in payload.get('real_object_summary',[]):
        lines.append(f"| {r['object']} | {r['n_windows']} | {fmt(r['new_accuracy'])} | {fmt(r['new_stable_accuracy'])} | {fmt(r['new_inst_accuracy'])} | {fmt(r['old_H1_stable_sat'])}/{fmt(r['new_H1_stable_sat'])} | {fmt(r['old_H1_stable_mean'])}/{fmt(r['new_H1_stable_mean'])} | {fmt(r['old_H1_inst_mean'])}/{fmt(r['new_H1_inst_mean'])} |")
    lines += ['', '## Files', '']
    for k,v in payload['files'].items(): lines.append(f'- {k}: `{v}`')
    return '\n'.join(lines)+'\n'


def run_real_eval(stage2_ckpt: Path, report_root: Path) -> Path:
    cmd = f"cd {REPO} && source /home/zjy/miniconda3/etc/profile.d/conda.sh && conda activate sparsh && export PYTHONPATH=/home/zjy/document/sparsh:. && CUDA_VISIBLE_DEVICES=0 python {REAL_EVAL_SCRIPT} --batch-size 32 --stage2-checkpoint {stage2_ckpt} --report-root {report_root}"
    subprocess.run(['bash','-lc',cmd], check=True)
    dirs=sorted([p for p in report_root.iterdir() if p.is_dir()], key=lambda p:p.stat().st_mtime)
    return dirs[-1]


def read_real_window_table(path: Path) -> list[dict[str, Any]]:
    with path.open(newline='', encoding='utf-8') as f: return list(csv.DictReader(f))


def real_compare(old_dir: Path, new_dir: Path, out_dir: Path) -> tuple[list[dict[str,Any]], list[dict[str,Any]]]:
    old=read_real_window_table(old_dir/'real_force_slip_sequence_side_window_summary.csv')
    new=read_real_window_table(new_dir/'real_force_slip_sequence_side_window_summary.csv')
    key=lambda r:(r['sequence'],r['side'],r['window_type'],r['window_index'],r['window_start'],r['window_end'])
    old_map={key(r):r for r in old}; rows=[]
    for r in new:
        o=old_map.get(key(r),{})
        target=1 if r['window_type']=='slip' else 0
        # default future prediction uses H1>=0.5, no real threshold fitting.
        new_pred=1 if float(r['p_instability_H1_mean'])>=0.5 else 0
        old_pred=1 if o and float(o['p_instability_H1_mean'])>=0.5 else None
        rec={'object':r['object'],'sequence':r['sequence'],'side':r['side'],'window_type':r['window_type'],'window_start':r['window_start'],'window_end':r['window_end'],'target_instability':target,'old_pred':old_pred,'new_pred':new_pred}
        for h in [1,3,5]:
            rec[f'old_H{h}_mean']=float(o.get(f'p_instability_H{h}_mean','nan')) if o else None
            rec[f'new_H{h}_mean']=float(r[f'p_instability_H{h}_mean'])
            rec[f'old_H{h}_sat_ge_0p99']=1 if o and float(o.get(f'p_instability_H{h}_mean','nan'))>=0.99 else 0
            rec[f'new_H{h}_sat_ge_0p99']=1 if float(r[f'p_instability_H{h}_mean'])>=0.99 else 0
        rows.append(rec)
    summary=[]
    for obj in sorted({r['object'] for r in rows}):
        vals=[r for r in rows if r['object']==obj]
        stable=[r for r in vals if r['target_instability']==0]; inst=[r for r in vals if r['target_instability']==1]
        def acc(vs, col): return sum(1 for x in vs if x[col]==x['target_instability'])/len(vs) if vs else None
        def mean(vs,col): return float(np.mean([x[col] for x in vs if x[col] is not None])) if vs else None
        summary.append({'object':obj,'n_windows':len(vals),'new_accuracy':acc(vals,'new_pred'),'new_stable_accuracy':acc(stable,'new_pred'),'new_inst_accuracy':acc(inst,'new_pred'),'old_accuracy':acc(vals,'old_pred'),'old_stable_accuracy':acc(stable,'old_pred'),'old_inst_accuracy':acc(inst,'old_pred'),'old_H1_stable_sat':mean(stable,'old_H1_sat_ge_0p99'),'new_H1_stable_sat':mean(stable,'new_H1_sat_ge_0p99'),'old_H1_stable_mean':mean(stable,'old_H1_mean'),'new_H1_stable_mean':mean(stable,'new_H1_mean'),'old_H1_inst_mean':mean(inst,'old_H1_mean'),'new_H1_inst_mean':mean(inst,'new_H1_mean')})
    write_csv(out_dir/'real_zero_shot_old_vs_new_window_table.csv', rows)
    write_csv(out_dir/'real_zero_shot_object_summary.csv', summary)
    return rows, summary


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--stamp', default=None)
    ap.add_argument('--epochs', type=int, default=16)
    ap.add_argument('--batch-size', type=int, default=2048)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--hidden-dim', type=int, default=512)
    ap.add_argument('--dropout', type=float, default=0.1)
    args=ap.parse_args()
    st=args.stamp or stamp()
    report_dir=REPORT_ROOT/st
    run_dir=RUN_ROOT/st
    report_dir.mkdir(parents=True, exist_ok=True); run_dir.mkdir(parents=True, exist_ok=True)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train=load_features('train'); val=load_features('val')
    X_val=feature_x(val); input_dim=int(X_val.shape[1])
    old_model=load_head(OLD_HEAD, input_dim, device=device)
    old_probs=probs_from_model(old_model,X_val,device)
    old_rows=eval_prediction_matrix(val,old_probs,None,'old_head_val_all')
    write_csv(report_dir/'old_head_val_diagnostics.csv', old_rows)
    selection_rows=[{**summarize_metrics(old_rows,'old_head_val_all'),'checkpoint':str(OLD_HEAD)}]
    experiments=[]
    # All-source train/val plus heldout geometry train/eval splits.
    split_specs=[('all_source', None, None), ('flat_sharp_to_sphere', {'flat','sharp'}, {'sphere'}), ('flat_sphere_to_sharp', {'flat','sphere'}, {'sharp'}), ('sharp_sphere_to_flat', {'sharp','sphere'}, {'flat'})]
    loss_modes=['baseline_bce','balanced_bce','balanced_bce_smooth']
    for split_name, train_geoms, val_geoms in split_specs:
        tr_mask=mask_by_geoms(train, include=train_geoms) if train_geoms is not None else np.ones(len(train['metadata']), dtype=bool)
        va_mask=mask_by_geoms(val, include=val_geoms) if val_geoms is not None else np.ones(len(val['metadata']), dtype=bool)
        # old diagnostics on the same split
        rows=eval_prediction_matrix(val, old_probs, va_mask, f'old_{split_name}')
        write_csv(report_dir/f'old_{split_name}_diagnostics.csv', rows)
        for mode in loss_modes:
            exp=f'{split_name}_{mode}'
            res=train_one(train,val,tr_mask,va_mask,exp,mode,run_dir,args.epochs,args.batch_size,args.lr,args.hidden_dim,args.dropout,device)
            experiments.append({**res,'split_name':split_name,'train_geoms':sorted(train_geoms) if train_geoms else ['flat','sharp','sphere'],'eval_geoms':sorted(val_geoms) if val_geoms else ['flat','sharp','sphere']})
            summary={**summarize_metrics(res['val_metrics'],exp),'checkpoint':res['checkpoint'],'loss_mode':mode,'split_name':split_name}
            selection_rows.append(summary)
    write_csv(report_dir/'model_selection_summary.csv', selection_rows)
    # Select only by original-data metrics: prefer heldout all mean AUPRC, penalize stable saturation.
    candidates=[r for r in selection_rows if r.get('checkpoint') and 'old_head' not in r['tag']]
    def score(r): return float(r.get('mean_auprc') or 0) - 0.15*float(r.get('mean_stable_saturation_ge_0p99') or 0)
    best=max(candidates,key=score)
    best_ckpt=Path(best['checkpoint'])
    real_root=report_dir/'real_zero_shot_eval'
    new_real_dir=run_real_eval(best_ckpt, real_root/'new_best_head')
    old_real_dir=Path('/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/real_force_slip_model_test/20260529_215727')
    real_rows, real_summary=real_compare(old_real_dir,new_real_dir,report_dir)
    payload={'generated_at':datetime.now().isoformat(timespec='seconds'),'report_dir':str(report_dir),'run_dir':str(run_dir),'feature_run':str(FEATURE_RUN),'old_head':str(OLD_HEAD),'device':str(device),'label_audit':{'future_label_logic':'phase3_2 FutureIndexedDataset uses labels[sample+1:sample+h+1], not current slip copy','horizons':[1,3,5]},'selection_rows':selection_rows,'experiments':experiments,'best_head':{**best,'selection_reason':'selected on original flat/sharp/sphere validation/heldout metrics only; real data not used'},'real_new_report_dir':str(new_real_dir),'real_old_report_dir':str(old_real_dir),'real_object_summary':real_summary,'files':{'model_selection_summary':str(report_dir/'model_selection_summary.csv'),'real_window_table':str(report_dir/'real_zero_shot_old_vs_new_window_table.csv'),'real_object_summary':str(report_dir/'real_zero_shot_object_summary.csv')}}
    write_json(report_dir/'future_head_horizon_fix_report.json', payload)
    (report_dir/'future_head_horizon_fix_report.md').write_text(render_report(payload),encoding='utf-8')
    print(json.dumps({'report_dir':str(report_dir),'best_checkpoint':str(best_ckpt),'real_new_report_dir':str(new_real_dir),'real_object_summary':real_summary},indent=2,default=jdefault))

if __name__=='__main__':
    main()
