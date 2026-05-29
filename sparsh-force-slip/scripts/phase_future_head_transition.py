#!/usr/bin/env python3
"""Transition-redefined contact-instability head training.

Real banana/cucumber/hammer/lemon data is validation-only. Training and model
selection use cached flat/sharp/sphere features only. This full-version pass
redefines the Stage-II target from future-only slip to current-or-future contact
instability: y_H = current_slip OR future_slip_within_H. This keeps slip windows
positive while making high-force but non-slipping stable contacts explicit
negative examples.
"""
from __future__ import annotations

import argparse, csv, json, math, subprocess, sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import phase3_2_world_model as wm  # noqa: E402

REPO = Path('/home/zjy/document/tactile-grasp')
WORKSPACE = REPO / 'sparsh-force-slip'
FEATURE_RUN = Path('/vla1/zjy/sparsh_runs/force_slip_phase4/phase_lambda_best_future_features_20260528_000000')
OLD_MEDIUM_BASE = Path('/vla1/zjy/sparsh_runs/force_slip_future_head_horizon_fix/20260529_223000/heads/sharp_sphere_to_flat_balanced_bce_smooth/checkpoints/best.pth')
RUN_ROOT = Path('/vla1/zjy/sparsh_runs/force_slip_future_transition')
REPORT_ROOT = WORKSPACE / 'reports/future_head_transition'
REAL_EVAL_SCRIPT = WORKSPACE / 'scripts/evaluate_real_force_slip_model_test.py'
QUICK_ANALYSIS_SCRIPT = WORKSPACE / 'scripts/quick_real_future_analysis.py'
AUX_NAMES = ['Fn_pred_N', 'Ft_pred_N', 'Ft_over_Fn_pred', 'p_slip_current', 'dFx_causal_N', 'dFy_causal_N', 'dFz_causal_N']
HORIZONS = [1, 3, 5]
GEOMS = ('flat', 'sharp', 'sphere')
PREV_REAL_REPORT = WORKSPACE / 'reports/real_force_slip_future_labels_eval/20260530_023127'


def now_stamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def jdefault(o: Any) -> Any:
    if isinstance(o, Path): return str(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, torch.Tensor): return o.detach().cpu().tolist()
    return str(o)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=jdefault), encoding='utf-8')


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


def load_features(split: str) -> dict[str, Any]:
    return torch.load(FEATURE_RUN / 'features' / f'{split}_features.pt', map_location='cpu', weights_only=False)


def geom_of_dataset(name: str) -> str:
    s=name.lower()
    for g in GEOMS:
        if s.startswith(g+'_') or s.startswith(g+'/') or g in s.split('_'):
            return g
    return 'unknown'


def mask_by_geoms(payload: dict[str, Any], include: set[str] | None = None) -> np.ndarray:
    return np.asarray([geom_of_dataset(m['dataset']) in include if include is not None else True for m in payload['metadata']], dtype=bool)


def prev_z(payload: dict[str, Any]) -> torch.Tensor:
    z=payload['z'].float()
    metas=payload['metadata']
    index={(m['dataset'], str(m['trajectory']), int(m['sample'])): i for i,m in enumerate(metas)}
    out=torch.empty_like(z)
    missing=0
    for i,m in enumerate(metas):
        key=(m['dataset'], str(m['trajectory']), int(m['sample'])-1)
        j=index.get(key)
        if j is None:
            out[i]=z[i]; missing += 1
        else:
            out[i]=z[j]
    return out


def make_x(payload: dict[str, Any], schema: list[str]) -> torch.Tensor:
    z=payload['z'].float()
    aux=payload['aux'].float()
    pz=prev_z(payload)
    parts=[]
    for name in schema:
        if name == 'z': parts.append(z)
        elif name in {'z_delta_prev', 'delta_z_prev'}: parts.append(z - pz)
        elif name == 'aux': parts.append(aux)
        else: raise ValueError(name)
    return torch.cat(parts, dim=1)



def instability_target(payload: dict[str, Any]) -> torch.Tensor:
    """Current-or-future instability target for H1/H3/H5.

    Existing cached labels are future-only. For deployment-style contact
    instability, a currently slipping frame is already unstable for all horizons,
    while a currently stable frame is positive only if slip appears inside the
    future horizon.
    """
    future = payload['future_slip'].float()
    current = payload['current_slip'].float().view(-1, 1).expand_as(future)
    return torch.maximum(future, current)

def hard_weights(payload: dict[str, Any], multiplier: float=4.0, q: float=0.70) -> np.ndarray:
    aux=payload['aux'].float().numpy()
    fut=instability_target(payload).numpy()
    fn, ft, ratio, pslip = aux[:,0], aux[:,1], aux[:,2], aux[:,3]
    # Robust normalized contact-intensity score; high but future-stable samples are hard negatives.
    raw=np.nan_to_num(fn) + np.nan_to_num(ft) + np.nan_to_num(ratio) + 2.0*np.nan_to_num(pslip)
    thr=np.quantile(raw, q)
    neg=(fut.sum(axis=1)==0)
    w=np.ones(len(raw), dtype=np.float32)
    w[neg & (raw>=thr)] = multiplier
    return w


class FeatDS(Dataset):
    def __init__(self, payload: dict[str, Any], schema: list[str], mask: np.ndarray, sample_weight: np.ndarray, smooth: float):
        idx=torch.tensor(np.where(mask)[0], dtype=torch.long)
        self.x=make_x(payload, schema)[idx]
        y_inst=instability_target(payload)[idx]
        self.y_stable=1.0-y_inst
        if smooth:
            self.y_stable = self.y_stable*(1.0-smooth) + 0.5*smooth
        self.w=torch.tensor(sample_weight[mask], dtype=torch.float32)
    def __len__(self): return int(self.x.shape[0])
    def __getitem__(self, i): return self.x[i], self.y_stable[i], self.w[i]


def probs(model: torch.nn.Module, X: torch.Tensor, device: torch.device, batch: int=4096) -> np.ndarray:
    outs=[]; model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch):
            logits=model(X[i:i+batch].to(device))
            outs.append((1.0-torch.sigmoid(logits)).cpu())
    return torch.cat(outs, dim=0).numpy()


def m_binary(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    y=y.astype(int); pred=(p>=0.5).astype(int)
    out={
        'n': int(len(y)), 'positive_ratio': float(y.mean()) if len(y) else None,
        'stable_mean': float(p[y==0].mean()) if (y==0).any() else None,
        'inst_mean': float(p[y==1].mean()) if (y==1).any() else None,
        'gap': float(p[y==1].mean()-p[y==0].mean()) if (y==0).any() and (y==1).any() else None,
        'stable_sat_ge_0p99': float((p[y==0]>=0.99).mean()) if (y==0).any() else None,
        'inst_sat_ge_0p99': float((p[y==1]>=0.99).mean()) if (y==1).any() else None,
        'f1': float(f1_score(y,pred,zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(y,pred)),
    }
    if len(np.unique(y))>1:
        out['auroc']=float(roc_auc_score(y,p)); out['auprc']=float(average_precision_score(y,p))
    else:
        out['auroc']=None; out['auprc']=None
    return out


def eval_probs(payload: dict[str, Any], P: np.ndarray, mask: np.ndarray | None, tag: str) -> list[dict[str, Any]]:
    Y=instability_target(payload).numpy().astype(int)
    if mask is None: mask=np.ones(len(Y), dtype=bool)
    rows=[]
    subsets=[('all', mask)] + [(g, mask & mask_by_geoms(payload,{g})) for g in GEOMS]
    for subset, smask in subsets:
        if not smask.any(): continue
        for j,h in enumerate(HORIZONS):
            rows.append({'tag': tag, 'subset': subset, 'horizon': f'H{h}', **m_binary(Y[smask,j], P[smask,j])})
    return rows


def summarize(rows: list[dict[str, Any]], tag: str, ckpt: str | None=None) -> dict[str, Any]:
    a=[r for r in rows if r['subset']=='all']
    rec={'tag': tag, 'checkpoint': ckpt}
    for k in ['auprc','auroc','f1','balanced_accuracy','stable_mean','inst_mean','gap','stable_sat_ge_0p99']:
        vals=[r[k] for r in a if r.get(k) is not None]
        rec['mean_'+k]=float(np.mean(vals)) if vals else None
    return rec


def train_one(train: dict[str, Any], val: dict[str, Any], exp: dict[str, Any], run_dir: Path, report_dir: Path, args, device: torch.device) -> dict[str, Any]:
    schema=exp['schema']
    tr_mask=mask_by_geoms(train, set(exp['train_geoms']))
    va_mask=mask_by_geoms(val, set(exp['eval_geoms'])) if exp.get('eval_geoms') else np.ones(len(val['metadata']), dtype=bool)
    weights=hard_weights(train, exp.get('hard_multiplier',1.0), exp.get('hard_q',0.70)) if exp.get('hard_negative') else np.ones(len(train['metadata']), dtype=np.float32)
    ds=FeatDS(train, schema, tr_mask, weights, exp.get('smooth',0.0))
    loader=DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    X_val=make_x(val, schema)
    model=wm.MLP(int(ds.x.shape[1]), args.hidden_dim, 3, dropout=args.dropout).to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # stable target imbalance per horizon
    y_np=ds.y_stable.numpy(); pos=y_np.sum(axis=0); neg=len(y_np)-pos
    pos_weight=torch.tensor(neg/np.maximum(pos,1.0), dtype=torch.float32, device=device) if exp.get('balanced') else None
    best_score=-1e9; best_state=None; best_rows=None; history=[]
    for epoch in range(1,args.epochs+1):
        model.train(); losses=[]
        for x,y,w in loader:
            x=x.to(device); y=y.to(device); w=w.to(device)
            loss_raw=F.binary_cross_entropy_with_logits(model(x), y, pos_weight=pos_weight, reduction='none')
            loss=(loss_raw.mean(dim=1)*w).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); opt.step()
            losses.append(float(loss.detach().cpu()))
        P=probs(model, X_val, device)
        rows=eval_probs(val, P, va_mask, exp['name'])
        summ=summarize(rows, exp['name'])
        score=(summ.get('mean_auprc') or 0.0) - 0.10*(summ.get('mean_stable_sat_ge_0p99') or 0.0) - 0.05*(summ.get('mean_stable_mean') or 0.0)
        rec={'epoch': epoch, 'loss': float(np.mean(losses)), 'selection_score': score, **summ}
        history.append(rec)
        if score > best_score:
            best_score=score; best_state={k:v.detach().cpu() for k,v in model.state_dict().items()}; best_rows=rows; best_epoch=epoch
    model.load_state_dict(best_state)
    ckpt=run_dir/'heads'/exp['name']/'checkpoints'/'best.pth'
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    payload={'experiment_name': exp['name'], 'model_state': model.state_dict(), 'horizons': HORIZONS, 'aux_names': AUX_NAMES, 'input_schema': schema, 'hidden_dim': args.hidden_dim, 'dropout': args.dropout, 'target': 'current_or_future_instability: current_slip OR future_slip_H; stable logits; p_instability=1-sigmoid(logits)', 'training': exp, 'best_epoch': best_epoch, 'feature_run': str(FEATURE_RUN), 'raw_data_modified': False}
    torch.save(payload, ckpt)
    write_json(run_dir/'heads'/exp['name']/'history.json', history)
    write_csv(report_dir/f'{exp["name"]}_val_metrics.csv', best_rows)
    return {**exp, 'checkpoint': str(ckpt), 'best_epoch': best_epoch, 'best_selection_score': best_score, 'val_metrics': best_rows, **summarize(best_rows, exp['name'], str(ckpt))}


def load_head_eval(path: Path, payload: dict[str, Any], device: torch.device) -> list[dict[str, Any]]:
    ckpt=torch.load(path, map_location='cpu', weights_only=False)
    schema=ckpt.get('input_schema') or ['z','aux']
    state=ckpt['model_state']; hidden=int(state['net.1.weight'].shape[0]); in_dim=int(state['net.1.weight'].shape[1])
    model=wm.MLP(in_dim, hidden, 3, dropout=float(ckpt.get('dropout',0.1))).to(device)
    model.load_state_dict(state); model.eval()
    P=probs(model, make_x(payload, schema), device)
    return eval_probs(payload, P, None, path.parent.parent.name)


def run_real_eval(ckpt: Path, report_root: Path) -> Path:
    cmd=f"cd {REPO} && source /home/zjy/miniconda3/etc/profile.d/conda.sh && conda activate sparsh && export PYTHONPATH=/home/zjy/document/sparsh:. && CUDA_VISIBLE_DEVICES=0 python {REAL_EVAL_SCRIPT} --window-source future_labels --batch-size 64 --stage2-checkpoint {ckpt} --report-root {report_root}"
    subprocess.run(['bash','-lc',cmd], check=True)
    dirs=sorted([p for p in report_root.iterdir() if p.is_dir()], key=lambda p:p.stat().st_mtime)
    return dirs[-1]


def run_quick(source: Path, out_root: Path) -> Path | None:
    if not QUICK_ANALYSIS_SCRIPT.exists(): return None
    cmd=f"cd {REPO} && source /home/zjy/miniconda3/etc/profile.d/conda.sh && conda activate sparsh && python {QUICK_ANALYSIS_SCRIPT} --source-report {source} --output-root {out_root}"
    subprocess.run(['bash','-lc',cmd], check=True)
    dirs=sorted([p for p in out_root.iterdir() if p.is_dir()], key=lambda p:p.stat().st_mtime)
    return dirs[-1]


def read_metrics_csv(path: Path) -> dict[str, dict[str, float]]:
    out={}
    with path.open(newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            out[r['score']]={k:(float(v) if v not in ('', 'None') else None) for k,v in r.items() if k!='score'}
    return out


def render_report(payload: dict[str, Any]) -> str:
    def fmt(v):
        if v is None: return 'n/a'
        try: return f'{float(v):.4f}'
        except Exception: return str(v)
    lines=['# Transition-redefined Future-instability Head Report','',f"- generated_at: `{payload['generated_at']}`",f"- report_dir: `{payload['report_dir']}`",f"- run_dir: `{payload['run_dir']}`",'- real-data policy: real banana/cucumber/hammer/lemon data is validation-only; no training, checkpoint selection, or final threshold selection uses it.','']
    lines += ['## Original flat/sharp/sphere validation selection','','| tag | schema | hard neg | mean AUPRC | mean AUROC | stable mean | gap | stable sat>=0.99 | ckpt |','|---|---|---:|---:|---:|---:|---:|---:|---|']
    for r in payload['selection_rows']:
        lines.append(f"| {r['tag']} | {','.join(r.get('schema', []))} | {r.get('hard_negative')} | {fmt(r.get('mean_auprc'))} | {fmt(r.get('mean_auroc'))} | {fmt(r.get('mean_stable_mean'))} | {fmt(r.get('mean_gap'))} | {fmt(r.get('mean_stable_sat_ge_0p99'))} | `{r.get('checkpoint')}` |")
    lines += ['', '## Real future-label validation comparison','','| score | previous stable | previous slip | new stable | new slip | previous gap | new gap | previous F1@0.5 | new F1@0.5 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for score in ['p_instability_H1','p_instability_H3','p_instability_H5']:
        old=payload['previous_real_metrics'].get(score,{}); new=payload['new_real_metrics'].get(score,{})
        lines.append(f"| {score} | {fmt(old.get('stable_mean'))} | {fmt(old.get('slip_mean'))} | {fmt(new.get('stable_mean'))} | {fmt(new.get('slip_mean'))} | {fmt(old.get('slip_minus_stable_gap'))} | {fmt(new.get('slip_minus_stable_gap'))} | {fmt(old.get('f1_at_0p5'))} | {fmt(new.get('f1_at_0p5'))} |")
    lines += ['', '## Decision','', f"- improvement_status: **{payload['decision']['status']}**", f"- reason: {payload['decision']['reason']}", '']
    lines += ['## Files','']
    for k,v in payload['files'].items(): lines.append(f'- {k}: `{v}`')
    return '\n'.join(lines)+'\n'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stamp', default=None)
    ap.add_argument('--epochs', type=int, default=18)
    ap.add_argument('--batch-size', type=int, default=2048)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--hidden-dim', type=int, default=512)
    ap.add_argument('--dropout', type=float, default=0.1)
    args=ap.parse_args()
    st=args.stamp or now_stamp()
    report_dir=REPORT_ROOT/st; run_dir=RUN_ROOT/st
    report_dir.mkdir(parents=True, exist_ok=True); run_dir.mkdir(parents=True, exist_ok=True)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train=load_features('train'); val=load_features('val')
    experiments=[
        {'name':'transition_hard_aux_balanced','schema':['z','aux'],'hard_negative':True,'hard_multiplier':4.0,'hard_q':0.70,'balanced':True,'smooth':0.03,'train_geoms':['flat','sharp','sphere'],'eval_geoms':['flat','sharp','sphere']},
        {'name':'transition_hard_delta_balanced','schema':['z','z_delta_prev','aux'],'hard_negative':True,'hard_multiplier':4.0,'hard_q':0.70,'balanced':True,'smooth':0.03,'train_geoms':['flat','sharp','sphere'],'eval_geoms':['flat','sharp','sphere']},
    ]
    rows=[]
    # baseline old horizon-fix checkpoint on original val for reference
    if OLD_MEDIUM_BASE.exists():
        base_rows=load_head_eval(OLD_MEDIUM_BASE, val, device)
        rows.append({**summarize(base_rows, 'horizon_fix_best', str(OLD_MEDIUM_BASE)), 'schema':['z','aux'], 'hard_negative':'previous', 'val_metrics': base_rows})
        write_csv(report_dir/'horizon_fix_best_val_metrics.csv', base_rows)
    for exp in experiments:
        rows.append(train_one(train,val,exp,run_dir,report_dir,args,device))
    # Select with original validation only.
    candidates=[r for r in rows if r['tag'] != 'horizon_fix_best']
    def sel(r): return (r.get('mean_auprc') or 0) - 0.10*(r.get('mean_stable_sat_ge_0p99') or 0) - 0.05*(r.get('mean_stable_mean') or 0)
    best=max(candidates, key=sel)
    write_csv(report_dir/'transition_model_selection_summary.csv', [{k:v for k,v in r.items() if k!='val_metrics'} for r in rows])
    real_dir=run_real_eval(Path(best['checkpoint']), report_dir/'real_zero_shot_eval')
    quick_dir=run_quick(real_dir, report_dir/'quick_analysis')
    prev_metrics=read_metrics_csv(PREV_REAL_REPORT/'real_force_slip_binary_metrics.csv')
    new_metrics=read_metrics_csv(real_dir/'real_force_slip_binary_metrics.csv')
    h1_old=prev_metrics['p_instability_H1']; h1_new=new_metrics['p_instability_H1']
    stable_drop=(h1_old['stable_mean'] or 0) - (h1_new['stable_mean'] or 0)
    gap_gain=(h1_new['slip_minus_stable_gap'] or 0) - (h1_old['slip_minus_stable_gap'] or 0)
    f1_gain=(h1_new['f1_at_0p5'] or 0) - (h1_old['f1_at_0p5'] or 0)
    ok = stable_drop >= 0.10 and gap_gain >= 0.05 and f1_gain >= 0.10
    decision={'status':'significant_transition_improvement' if ok else 'insufficient_transition_improvement', 'stable_drop_H1': stable_drop, 'gap_gain_H1': gap_gain, 'f1_gain_H1': f1_gain, 'reason': 'H1 stable mean dropped by >=0.10, gap improved by >=0.05, and F1@0.5 improved by >=0.10.' if ok else 'Transition redefinition did not reduce real stable-window future-instability enough; further optimization is still required.'}
    payload={'generated_at': datetime.now().isoformat(timespec='seconds'), 'report_dir': str(report_dir), 'run_dir': str(run_dir), 'device': str(device), 'feature_run': str(FEATURE_RUN), 'selection_rows': [{k:v for k,v in r.items() if k!='val_metrics'} for r in rows], 'best_head': best, 'real_report_dir': str(real_dir), 'quick_analysis_dir': str(quick_dir) if quick_dir else None, 'previous_real_metrics': prev_metrics, 'new_real_metrics': new_metrics, 'decision': decision, 'files': {'model_selection_summary': str(report_dir/'transition_model_selection_summary.csv'), 'real_report': str(real_dir/'real_force_slip_model_test_report.md'), 'quick_analysis': str(quick_dir/'quick_real_future_analysis.md') if quick_dir else None}}
    write_json(report_dir/'future_head_transition_report.json', payload)
    (report_dir/'future_head_transition_report.md').write_text(render_report(payload), encoding='utf-8')
    print(json.dumps({'report_dir': str(report_dir), 'best_checkpoint': best['checkpoint'], 'real_report_dir': str(real_dir), 'quick_analysis_dir': str(quick_dir) if quick_dir else None, 'decision': decision}, indent=2, default=jdefault))

if __name__ == '__main__':
    main()
