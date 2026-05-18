#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from pathlib import Path
ROOT=Path('/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase7/step2_leave_one_geometry_out')
p=ROOT/'step2_leave_one_geometry_out.json'
d=json.loads(p.read_text(encoding='utf-8'))
def get(o,path):
    cur=o
    for part in path.split('.'):
        if not isinstance(cur,dict): return None
        cur=cur.get(part)
    return cur
cols=['split','condition','current_force_RMSE','current_slip_F1','best_epoch']
for h in [1,3,5]:
    for m in ['future_slip_f1','future_slip_auprc','future_slip_auroc']:
        cols.append(f'H{h}_{m.replace("future_slip_","")}')
rows=[]
for r in d.get('conditions',[]):
    row={'split':'sharp+sphere_to_flat','condition':r.get('condition'),'current_force_RMSE':get(r,'current_metrics_val_subset.force_rmse_mean_N'),'current_slip_F1':get(r,'current_metrics_val_subset.slip_f1'),'best_epoch':r.get('best_epoch')}
    for h in [1,3,5]:
        row[f'H{h}_f1']=get(r,f'best_val.H{h}.future_slip_f1')
        row[f'H{h}_auprc']=get(r,f'best_val.H{h}.future_slip_auprc')
        row[f'H{h}_auroc']=get(r,f'best_val.H{h}.future_slip_auroc')
    rows.append(row)
with (ROOT/'step2_sharp_sphere_to_flat_full_metrics.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(rows)

def fmt(x): return 'n/a' if x is None else f'{float(x):.4f}'
lines=['\n## New split full H1/H3/H5 metrics','','| condition | current force RMSE | current slip F1 | H1 F1 | H1 AUPRC | H1 AUROC | H3 F1 | H3 AUPRC | H3 AUROC | H5 F1 | H5 AUPRC | H5 AUROC | best epoch |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    vals=[r['condition'],fmt(r['current_force_RMSE']),fmt(r['current_slip_F1']),fmt(r['H1_f1']),fmt(r['H1_auprc']),fmt(r['H1_auroc']),fmt(r['H3_f1']),fmt(r['H3_auprc']),fmt(r['H3_auroc']),fmt(r['H5_f1']),fmt(r['H5_auprc']),fmt(r['H5_auroc']),str(r['best_epoch'])]
    lines.append('| '+' | '.join(vals)+' |')
lines += ['', f'- CSV: `{ROOT / "step2_sharp_sphere_to_flat_full_metrics.csv"}`']
md=ROOT/'step2_leave_one_geometry_out.md'
text=md.read_text(encoding='utf-8')
if '## New split full H1/H3/H5 metrics' not in text:
    text=text.rstrip()+'\n'+'\n'.join(lines)+'\n'
md.write_text(text,encoding='utf-8')
d['new_split_full_metrics_csv']=str(ROOT/'step2_sharp_sphere_to_flat_full_metrics.csv')
p.write_text(json.dumps(d,indent=2,default=str),encoding='utf-8')
print(ROOT/'step2_sharp_sphere_to_flat_full_metrics.csv')
