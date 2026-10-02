#!/usr/bin/env python3
"""Render verified R9 HTT evaluation with all seed curves and bounded conclusions."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'evaluation'))
import evaluate as e

def main():
 p=argparse.ArgumentParser();p.add_argument('--evaluation-dir',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-synthetic',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 summary=json.loads((a.evaluation_dir/'summary.json').read_text())
 if summary.get('synthetic',False) and not a.allow_synthetic:raise ValueError('Synthetic results forbidden in formal reporting')
 if summary['status']!='complete' or summary['new_runs']!=36 or summary['manifest_sha256']!=e.sha(a.manifest):raise ValueError('Incomplete/mismatched evaluation')
 for name,h in summary['output_hashes'].items():
  if e.sha(a.evaluation_dir/name)!=h:raise ValueError('Evaluation artifact changed')
 rows=e.readcsv(a.evaluation_dir/'summary_metrics.csv');ci=e.readcsv(a.evaluation_dir/'paired_ci.csv');roc=e.readcsv(a.evaluation_dir/'roc_points.csv');fail=e.readcsv(a.evaluation_dir/'failure_cases.csv');manifest=json.loads(a.manifest.read_text());new=[r for r in manifest['runs'] if not r.get('historical',False)];folds=sorted({r['fold'] for r in new})
 for mode in ('roc','pr','low_fpr'):
  fig,axes=plt.subplots(4,3,figsize=(14,15),squeeze=False)
  for i,fold in enumerate(folds):
   for j,seed in enumerate(e.SEEDS):
    ax=axes[i,j]
    for group in e.GROUPS:
     rr=[r for r in roc if r['group']==group and r['fold']==fold and int(r['seed'])==seed and r['role']=='validation'];xx=[float(r['gross_recall'] if mode=='pr' else r['static_fpr']) for r in rr];yy=[float(r['precision'] if mode=='pr' else r['gross_recall']) for r in rr];ax.plot(xx,yy,label=group)
    ax.set_title(f'{fold} / {seed}');ax.set_xlabel('Recall' if mode=='pr' else 'Static FPR');ax.set_ylabel('Precision' if mode=='pr' else 'Gross recall');ax.set_xlim(0,.1 if mode=='low_fpr' else 1);ax.set_ylim(0,1);ax.grid(alpha=.2)
  axes[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(a.output/f'{mode}_all_seeds.png',dpi=140);plt.close(fig)
 # Representative failures fixed by the evaluator, all runs retained in its table.
 selected=[]
 for group in e.GROUPS:
  for kind in ('false_positive','missed_event'):
   candidates=[r for r in fail if r['group']==group and r['kind']==kind]
   if not candidates:continue
   if kind=='false_positive':key=lambda r:(-float(r['fp'])/(float(r['fp'])+float(r['tn'])),-float(r['false_starts']),r['fold'],r['seed'],r['episode'])
   else:key=lambda r:(float(r['hits'])/float(r['events']),-(float(r['events'])-float(r['hits'])),r['fold'],r['seed'],r['episode'])
   item=sorted(candidates,key=key)[0];run=next(r for r in new if r['group']==group and r['fold']==item['fold'] and str(r['seed'])==item['seed']);rr=e.load_rows(run['predictions']['validation'],run['endpoints']['validation'],'validation');cal=e.load_rows(run['predictions']['calibration'],run['endpoints']['calibration'],'calibration');threshold=e.choose(cal)['FPR0.05'];rr=[r for r in rr if r['episode']==item['episode']];fig,ax=plt.subplots(figsize=(10,3));ax.plot([r['t'] for r in rr],[r['score'] for r in rr],label='p gross');ax.axhline(threshold,color='r',linestyle='--',label='cal FPR5 threshold');ax.step([r['t'] for r in rr],[r['stage']/2 for r in rr],alpha=.4,label='stage / 2');ax.set_xlabel('Frame');ax.set_ylim(-.05,1.1);ax.set_title(f"{group} {kind}: {item['episode']}");ax.legend();fig.tight_layout();filename=f'failure_{group}_{kind}.png';fig.savefig(a.output/filename,dpi=140);plt.close(fig);selected.append({**item,'figure':filename})
 e.csvout(a.output/'representative_cases.csv',selected)
 natural=e.readcsv(a.evaluation_dir/'metrics.csv');natural=[float(r['positive_prevalence']) for r in natural if r['group']=='V_temporal' and r['role']=='validation' and r['point']=='fixed_0.5' and r['rule']=='raw'];prevalence=float(np.mean(natural))
 text=['# 第九轮 HTT 当前滑移检测结果'+(' — SYNTHETIC SMOKE ONLY' if summary.get('synthetic') else ''),'', '36次新训练，四折×三个种子×三组；以下为重叠开发折的描述性均值，不是独立盲测。validation用于checkpoint选择。',f'共同validation端点自然gross占比均值为{prevalence:.2%}（static/gross主任务）；AP需结合该占比、实际误报和召回解释。','', '|组|工作点|FPR|Gross召回|BA|AP|事件召回|','|---|---|---:|---:|---:|---:|---:|']
 for group in e.GROUPS:
  for point in ('fixed_0.5','maxBA','FPR0.01','FPR0.05','FPR0.10'):
   vals=[]
   for metric in ('static_fpr','gross_recall','balanced_accuracy','AP','event_recall'):
    r=next(r for r in rows if r['group']==group and r['point']==point and r['rule']=='raw' and r['metric']==metric);vals.append(f"{float(r['mean']):.4f}" if r['mean'] else 'NA')
   text.append('|'+ '|'.join([group,point]+vals)+'|')
 text+=['','## 主要观察','F_history相对视觉容量对照只有小幅平均收益，配对证据不支持跨折普遍改善。F_delta相对完整力历史未形成稳定额外收益，部分工作点降低误报同时损失召回。校准约束迁移到validation后明显失效，低误报部署目标尚未达成。此处的折级区间计数用于定位异质性，不以投票方式证明整体显著。','','## 配对证据','每折对三个seed使用同一完整泄漏组bootstrap抽样，先逐seed计算再平均；不合并四折为独立样本。']
 for candidate,base in e.PAIRS:
  for metric,point in (('pAUC','ranking'),('gross_recall','FPR0.05'),('static_fpr','FPR0.05')):
   rr=[r for r in ci if r['candidate']==candidate and r['base']==base and r['metric']==metric and r['point']==point and r['rule']=='raw'];positive=sum(bool(r['ci_lower']) and float(r['ci_lower'])>0 for r in rr);negative=sum(bool(r['ci_upper']) and float(r['ci_upper'])<0 for r in rr);undefined=sum(r['status']!='available' for r in rr);text.append(f'- {candidate} − {base}，{point}/{metric}：{positive}/4折区间完全正，{negative}/4折完全负。FPR差为负较好，召回/pAUC差为正较好；未定义区间{undefined}/4折；其余不等于证明无效。')
 text+=['','## 解释边界','- confirm2为预固定规则，沿用raw校准阈值，不保证校准或validation FPR约束。','- 事件延迟排除左删失段，仅是当前gross检测；不代表future提前预警。','- 力来自同一图像，显式差分是归纳偏置，不是额外传感器信息。','- 本轮force_diagnostics已完成同一slip NPZ的索引/尺度核验，并使用同试次GT力做诊断；没有跨basename配对。该核验不额外证明硬件同步。详见[同试次力诊断](/home/zjy/Documents/grasp/tactile_grasp/sparsh-force-slip/experiments/htt_normalflow/round9_htt_temporal_force_fusion/force_diagnostics/results/SUMMARY_ZH.md)。','- 原域future稳健性和独立数据支持审计由本轮对应工作包单独交付，不能由HTT检测结果代替。','- 第八轮未来状态预测未超简单基线的负向证据仍有效；本轮不建立世界模型或实物成功率结论。']
 (a.output/'HTT_SUMMARY_ZH.md').write_text('\n'.join(text)+'\n')
 e.js(a.output/'REPORT_AUDIT.json',dict(status='pass',synthetic=summary.get('synthetic',False),evaluation_summary_sha256=e.sha(a.evaluation_dir/'summary.json'),source_sha256=e.sha(__file__),new_runs=36,outputs={p.name:e.sha(p) for p in a.output.iterdir() if p.is_file() and p.name!='REPORT_AUDIT.json'}))
if __name__=='__main__':main()
