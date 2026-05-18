#!/usr/bin/env python3
from __future__ import annotations
import csv, json
from pathlib import Path
ROOT=Path('/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase7/step1_future_baselines')
p=ROOT/'step1_future_baselines.json'
d=json.loads(p.read_text(encoding='utf-8'))
cols=['condition','n_samples']
for h in [1,3,5]:
    for m in ['f1','auprc','auroc','ece']:
        cols.append(f'H{h}_{m}')
rows=[]
for r in d['rows']:
    row={'condition':r['condition'],'n_samples':r.get('n_samples')}
    for h in [1,3,5]:
        m=r.get(f'H{h}',{})
        for k in ['f1','auprc','auroc','ece']:
            row[f'H{h}_{k}']=m.get(k) or m.get(f'future_slip_{k}') or m.get('stability_calibration_error' if k=='ece' else k)
    rows.append(row)
with (ROOT/'step1_future_baselines_full_metrics.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(rows)

def fmt(x):
    if x is None: return 'n/a'
    return f'{float(x):.4f}'
lines=['\n## Full H1/H3/H5 metric table','','| condition | H1 F1 | H1 AUPRC | H1 AUROC | H1 ECE | H3 F1 | H3 AUPRC | H3 AUROC | H3 ECE | H5 F1 | H5 AUPRC | H5 AUROC | H5 ECE |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for row in rows:
    vals=[row['condition']]+[fmt(row[c]) for c in cols[2:]]
    lines.append('| '+' | '.join(vals)+' |')
lines += ['', f'- CSV: `{ROOT / "step1_future_baselines_full_metrics.csv"}`']
md=ROOT/'step1_future_baselines.md'
text=md.read_text(encoding='utf-8')
if '## Full H1/H3/H5 metric table' not in text:
    text=text.rstrip()+'\n'+'\n'.join(lines)+'\n'
md.write_text(text,encoding='utf-8')
d['full_metrics_csv']=str(ROOT/'step1_future_baselines_full_metrics.csv')
p.write_text(json.dumps(d,indent=2,default=str),encoding='utf-8')
print(ROOT/'step1_future_baselines_full_metrics.csv')
