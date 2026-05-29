#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math, os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

SCORES = ['p_slip_current', 'p_instability_H1', 'p_instability_H3', 'p_instability_H5']
H_SCORES = ['p_instability_H1', 'p_instability_H3', 'p_instability_H5']
FOCUS_OBJECTS = {'cucumber', 'hammer'}
ORIG_VAL_THRESHOLDS = {
    'p_slip_current': 0.5,
    'p_instability_H1': 0.848,
    'p_instability_H3': 0.728,
    'p_instability_H5': 0.784,
}
SWEEP_THRESHOLDS = [0.50,0.60,0.70,0.75,0.80,0.85,0.90,0.92,0.94,0.95,0.96,0.97,0.98,0.99,0.995]


def to_float(v: Any) -> float | None:
    if v in (None, ''):
        return None
    try:
        x = float(v)
        if math.isnan(x): return None
        return x
    except Exception:
        return None


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows=[]
    with path.open(newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rr=dict(r)
            rr['target_instability']=int(float(rr['target_instability']))
            rr['frame_index']=int(float(rr['frame_index']))
            for k in SCORES:
                rr[k]=to_float(rr.get(k))
            rows.append(rr)
    return rows


def mean(xs):
    vals=[x for x in xs if x is not None]
    return sum(vals)/len(vals) if vals else None


def fmt(v, n=3):
    return 'n/a' if v is None else f'{float(v):.{n}f}'


def auroc(y, s):
    pos=[x for yy,x in zip(y,s) if yy==1]
    neg=[x for yy,x in zip(y,s) if yy==0]
    if not pos or not neg: return None
    total=0.0
    for p in pos:
        total += sum(p>n for n in neg) + 0.5*sum(p==n for n in neg)
    return total/(len(pos)*len(neg))


def auprc(y, s):
    if not y or sum(y)==0 or sum(y)==len(y): return None
    order=sorted(range(len(s)), key=lambda i: -s[i])
    tp=0; fp=0; precisions=[]
    for i in order:
        if y[i]==1:
            tp+=1; precisions.append(tp/(tp+fp))
        else:
            fp+=1
    return sum(precisions)/len(precisions) if precisions else None


def metrics(rows, score, thr=0.5):
    vals=[r for r in rows if r.get(score) is not None]
    y=[r['target_instability'] for r in vals]
    s=[float(r[score]) for r in vals]
    if not y: return {'score': score, 'threshold': thr, 'n': 0}
    pred=[1 if x>=thr else 0 for x in s]
    tp=sum(p==1 and yy==1 for p,yy in zip(pred,y)); fp=sum(p==1 and yy==0 for p,yy in zip(pred,y))
    tn=sum(p==0 and yy==0 for p,yy in zip(pred,y)); fn=sum(p==0 and yy==1 for p,yy in zip(pred,y))
    prec=tp/(tp+fp) if tp+fp else None
    rec=tp/(tp+fn) if tp+fn else None
    f1=2*prec*rec/(prec+rec) if prec is not None and rec is not None and (prec+rec) else None
    tpr=rec
    tnr=tn/(tn+fp) if tn+fp else None
    bal=(tpr+tnr)/2 if tpr is not None and tnr is not None else None
    stable=[x for x,yy in zip(s,y) if yy==0]
    slip=[x for x,yy in zip(s,y) if yy==1]
    return {
        'score': score, 'threshold': thr, 'n': len(y), 'n_stable': len(stable), 'n_slip': len(slip),
        'stable_mean': mean(stable), 'slip_mean': mean(slip),
        'gap': (mean(slip)-mean(stable) if stable and slip else None),
        'accuracy': (tp+tn)/len(y), 'balanced_accuracy': bal,
        'precision': prec, 'recall': rec, 'f1': f1,
        'stable_accuracy': tnr, 'slip_recall': tpr,
        'auroc': auroc(y,s), 'auprc': auprc(y,s),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
    }


def groupby(rows, keys):
    g=defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in keys)].append(r)
    return g


def write_csv(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8'); return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def object_window_rows(rows):
    out=[]
    for key, vals in sorted(groupby(rows, ['object','window_type']).items()):
        rec={'object':key[0], 'window_type':key[1], 'n_frames':len(vals), 'n_windows':len(set((v['sequence'],v['side'],v['window_start'],v['window_end']) for v in vals))}
        for s in SCORES:
            rec[s+'_mean']=mean([v[s] for v in vals])
        out.append(rec)
    return out


def sequence_detail_rows(rows):
    out=[]
    for key, vals in sorted(groupby(rows, ['sequence','object','side','window_type','window_start','window_end']).items()):
        rec={'sequence':key[0], 'object':key[1], 'side':key[2], 'window_type':key[3], 'range':f'{key[4]}-{key[5]}', 'n_frames':len(vals)}
        for s in SCORES:
            rec[s+'_mean']=mean([v[s] for v in vals])
        out.append(rec)
    return out


def try_plots(out: Path, rows: list[dict[str, Any]], sweep: list[dict[str, Any]], detail: list[dict[str, Any]]):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        (out/'plot_error.txt').write_text(str(e), encoding='utf-8')
        return []
    files=[]
    # Distribution box-like histograms
    fig, axs = plt.subplots(1, 4, figsize=(16,3.8), constrained_layout=True)
    for ax, score in zip(axs, SCORES):
        stable=[r[score] for r in rows if r['target_instability']==0 and r[score] is not None]
        slip=[r[score] for r in rows if r['target_instability']==1 and r[score] is not None]
        ax.hist(stable, bins=30, alpha=0.55, label='stable', density=True)
        ax.hist(slip, bins=30, alpha=0.55, label='slip', density=True)
        ax.set_title(score); ax.set_xlim(0,1); ax.grid(alpha=0.25)
        if score in ORIG_VAL_THRESHOLDS: ax.axvline(ORIG_VAL_THRESHOLDS[score], color='k', ls='--', lw=1)
    axs[0].legend()
    p=out/'score_distributions_stable_vs_slip.png'; fig.savefig(p, dpi=180); plt.close(fig); files.append(str(p))

    # Threshold sweep balanced accuracy
    fig, ax = plt.subplots(figsize=(7,4), constrained_layout=True)
    for score in H_SCORES:
        xs=[r['threshold'] for r in sweep if r['score']==score and r['subset']=='overall']
        ys=[r['balanced_accuracy'] for r in sweep if r['score']==score and r['subset']=='overall']
        ax.plot(xs, ys, marker='o', label=score.replace('p_instability_',''))
    ax.set_xlabel('threshold'); ax.set_ylabel('balanced accuracy'); ax.set_ylim(0,1); ax.grid(alpha=0.25); ax.legend()
    p=out/'future_threshold_sweep_balanced_accuracy.png'; fig.savefig(p, dpi=180); plt.close(fig); files.append(str(p))

    # focus detail bar
    focus=[d for d in detail if d['object'] in FOCUS_OBJECTS]
    labels=[f"{d['sequence']}-{d['side']}-{d['window_type']}" for d in focus]
    if focus:
        fig, ax = plt.subplots(figsize=(max(10, len(focus)*0.45),4.5), constrained_layout=True)
        x=range(len(focus)); width=0.22
        for off, score in [(-width,'p_slip_current'), (0,'p_instability_H1'), (width,'p_instability_H5')]:
            ax.bar([i+off for i in x], [d[score+'_mean'] for d in focus], width=width, label=score)
        ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=75, ha='right', fontsize=8)
        ax.set_ylim(0,1.05); ax.grid(axis='y', alpha=0.25); ax.legend()
        p=out/'cucumber_hammer_window_scores.png'; fig.savefig(p, dpi=180); plt.close(fig); files.append(str(p))
    return files


def render_md(payload):
    lines=[]
    lines += ['# Quick Real Future-instability Analysis', '', f"- generated_at: `{payload['generated_at']}`", f"- source_report: `{payload['source_report']}`", f"- output_dir: `{payload['output_dir']}`", '- real-data policy: this is **analysis only**. The real banana/cucumber/hammer/lemon dataset is not used for model training, checkpoint selection, or final threshold selection.', '- threshold sweep on real data is marked diagnostic; original-validation thresholds are reported separately.', '']
    lines += ['## Score separation summary', '', '| subset | score | n | stable mean | slip mean | gap | AUROC | AUPRC | F1@0.5 | BalAcc@0.5 |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for subset in ['overall','cucumber_hammer']:
        for m in payload['summary_metrics'][subset]:
            lines.append(f"| {subset} | {m['score']} | {m['n']} | {fmt(m['stable_mean'])} | {fmt(m['slip_mean'])} | {fmt(m['gap'])} | {fmt(m['auroc'])} | {fmt(m['auprc'])} | {fmt(m['f1'])} | {fmt(m['balanced_accuracy'])} |")
    lines += ['', '## Original-validation thresholds applied to real data', '', '| subset | score | threshold | stable acc | slip recall | balanced acc | F1 | FP | TN |', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for subset in ['overall','cucumber_hammer']:
        for m in payload['orig_threshold_metrics'][subset]:
            lines.append(f"| {subset} | {m['score']} | {m['threshold']:.3f} | {fmt(m['stable_accuracy'])} | {fmt(m['slip_recall'])} | {fmt(m['balanced_accuracy'])} | {fmt(m['f1'])} | {m['fp']} | {m['tn']} |")
    lines += ['', '## Diagnostic best thresholds on real data (not for model selection)', '', '| subset | score | best threshold | balanced acc | F1 | stable acc | slip recall |', '|---|---|---:|---:|---:|---:|---:|']
    for rec in payload['diagnostic_best_thresholds']:
        lines.append(f"| {rec['subset']} | {rec['score']} | {rec['threshold']:.3f} | {fmt(rec['balanced_accuracy'])} | {fmt(rec['f1'])} | {fmt(rec['stable_accuracy'])} | {fmt(rec['slip_recall'])} |")
    lines += ['', '## Cucumber/Hammer window means', '', '| sequence | side | window | range | n | pSlip | H1 | H3 | H5 |', '|---|---|---|---|---:|---:|---:|---:|---:|']
    for d in payload['focus_sequence_detail']:
        lines.append(f"| {d['sequence']} | {d['side']} | {d['window_type']} | {d['range']} | {d['n_frames']} | {fmt(d['p_slip_current_mean'])} | {fmt(d['p_instability_H1_mean'])} | {fmt(d['p_instability_H3_mean'])} | {fmt(d['p_instability_H5_mean'])} |")
    lines += ['', '## Main observations', '', '- `p_slip_current` remains the most deployment-ready signal: it has a large stable/slip score gap and good F1 at the default 0.5 threshold.', '- The future-instability head has useful ranking separation, but H1/H3/H5 raw probabilities remain shifted upward on real images; this is why stable windows still look unstable at threshold 0.5.', '- Cucumber and hammer slip windows are clearer/longer, so they are better for qualitative figures. However hammer stable windows still produce high future-instability scores, highlighting the remaining domain-calibration gap.', '- For paper use without training on real data, report AUROC/AUPRC/gap and original-validation threshold results, not a threshold tuned on real validation labels.', '']
    lines += ['## Generated files', '']
    for k,v in payload['files'].items(): lines.append(f'- {k}: `{v}`')
    return '\n'.join(lines)+'\n'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source-report', type=Path, required=True)
    ap.add_argument('--output-root', type=Path, required=True)
    args=ap.parse_args()
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    out=args.output_root/stamp; out.mkdir(parents=True, exist_ok=True)
    rows=load_rows(args.source_report/'real_force_slip_per_frame_predictions.csv')
    focus=[r for r in rows if r['object'] in FOCUS_OBJECTS]
    summary={'overall':[metrics(rows,s,0.5) for s in SCORES], 'cucumber_hammer':[metrics(focus,s,0.5) for s in SCORES]}
    orig={'overall':[metrics(rows,s,ORIG_VAL_THRESHOLDS[s]) for s in SCORES], 'cucumber_hammer':[metrics(focus,s,ORIG_VAL_THRESHOLDS[s]) for s in SCORES]}
    sweep=[]
    for subset_name, subset_rows in [('overall',rows),('cucumber_hammer',focus)]:
        for s in SCORES:
            for t in SWEEP_THRESHOLDS:
                m=metrics(subset_rows,s,t); m['subset']=subset_name; sweep.append(m)
    best=[]
    for subset in ['overall','cucumber_hammer']:
        for s in H_SCORES:
            candidates=[r for r in sweep if r['subset']==subset and r['score']==s]
            best.append(max(candidates, key=lambda r: (r.get('balanced_accuracy') or -1, r.get('f1') or -1)))
    obj=object_window_rows(rows); detail=sequence_detail_rows(rows); focus_detail=[d for d in detail if d['object'] in FOCUS_OBJECTS]
    write_csv(out/'threshold_sweep_metrics.csv', sweep)
    write_csv(out/'object_window_means.csv', obj)
    write_csv(out/'sequence_window_means.csv', detail)
    write_csv(out/'cucumber_hammer_window_means.csv', focus_detail)
    plot_files=try_plots(out, rows, sweep, detail)
    payload={'generated_at':datetime.now().isoformat(timespec='seconds'), 'source_report':str(args.source_report), 'output_dir':str(out), 'summary_metrics':summary, 'orig_threshold_metrics':orig, 'diagnostic_best_thresholds':best, 'focus_sequence_detail':focus_detail, 'files':{'threshold_sweep_csv':str(out/'threshold_sweep_metrics.csv'), 'object_window_means_csv':str(out/'object_window_means.csv'), 'sequence_window_means_csv':str(out/'sequence_window_means.csv'), 'cucumber_hammer_window_means_csv':str(out/'cucumber_hammer_window_means.csv'), 'markdown':str(out/'quick_real_future_analysis.md'), 'plots':plot_files}}
    (out/'quick_real_future_analysis.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    (out/'quick_real_future_analysis.md').write_text(render_md(payload), encoding='utf-8')
    print(json.dumps({'output_dir':str(out), 'plots':plot_files, 'diagnostic_best_thresholds':best}, indent=2))

if __name__ == '__main__':
    main()
