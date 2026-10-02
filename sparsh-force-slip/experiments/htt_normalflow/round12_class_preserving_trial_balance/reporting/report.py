#!/usr/bin/env python3
"""Render verified R12 HTT evaluation with all seed curves and bounded conclusions."""
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
 if summary['status']!='complete' or (summary['new_runs']!=24 or summary['reused_runs']!=48) or summary['manifest_sha256']!=e.sha(a.manifest):raise ValueError('Incomplete/mismatched evaluation')
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
 # Each subplot retains all seeds separately; no pooled distribution hides fold migration.
 tail=e.readcsv(a.evaluation_dir/'static_tail_distribution.csv');transfer=e.readcsv(a.evaluation_dir/'threshold_migration.csv')
 for mode in ('tail_q99','actual_FPR5'):
  fig,axes=plt.subplots(4,3,figsize=(15,15),squeeze=False)
  for i,fold in enumerate(folds):
   for j,seed in enumerate(e.SEEDS):
    ax=axes[i,j]
    for group in e.GROUPS:
     rr=tail if mode=='tail_q99' else [r for r in transfer if r['point']=='FPR0.05']
     yy=[float(next(r for r in rr if r['group']==group and r['fold']==fold and int(r['seed'])==seed and r['role']==role)['q99' if mode=='tail_q99' else 'static_fpr']) for role in e.ROLES]
     ax.plot(e.ROLES,yy,marker='o',label=group)
    ax.set_title(f'{fold} / {seed}');ax.set_ylabel(mode);ax.set_ylim(0,1);ax.grid(alpha=.2)
  axes[0,0].legend(fontsize=6);fig.tight_layout();fig.savefig(a.output/f'{mode}_all_seeds.png',dpi=140);plt.close(fig)
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
 natural=e.readcsv(a.evaluation_dir/'metrics.csv');natural=[float(r['positive_prevalence']) for r in natural if r['group']=='V_original' and r['role']=='validation' and r['point']=='fixed_0.5' and r['rule']=='raw'];prevalence=float(np.mean(natural))
 text=['# 第十二轮 HTT 当前滑移检测结果'+(' — SYNTHETIC SMOKE ONLY' if summary.get('synthetic') else ''),'', '24次新训练＋48次复用对照，四折×三个种子×六组；以下为重叠开发折的描述性均值，不是独立盲测。validation用于checkpoint选择。',f'共同validation端点自然gross占比均值为{prevalence:.2%}（static/gross主任务）；AP需结合该占比、实际误报和召回解释。','', '|组|工作点|FPR|Gross召回|BA|AP|事件召回|','|---|---|---:|---:|---:|---:|---:|']
 for group in e.GROUPS:
  for point in ('fixed_0.5','maxBA','FPR0.01','FPR0.05','FPR0.10'):
   vals=[]
   for metric in ('static_fpr','gross_recall','balanced_accuracy','AP','event_recall'):
    r=next(r for r in rows if r['group']==group and r['point']==point and r['rule']=='raw' and r['metric']==metric);vals.append(f"{float(r['mean']):.4f}" if r['mean'] else 'NA')
   text.append('|'+ '|'.join([group,point]+vals)+'|')
 text+=['','## 配对证据','以下区间按完整泄漏组抽样，三个seed共享同一组抽样；不合并四折为独立样本。不以区间计数投票证明整体显著。']
 for candidate,base in e.PAIRS:
  for metric,point in (('pAUC','ranking'),('gross_recall','FPR0.05'),('static_fpr','FPR0.05')):
   rr=[r for r in ci if r['candidate']==candidate and r['base']==base and r['metric']==metric and r['point']==point and r['rule']=='raw'];positive=sum(bool(r['ci_lower']) and float(r['ci_lower'])>0 for r in rr);negative=sum(bool(r['ci_upper']) and float(r['ci_upper'])<0 for r in rr);undefined=sum(r['status']!='available' for r in rr);text.append(f'- {candidate} − {base}，{point}/{metric}：{positive}/4折区间完全正，{negative}/4折完全负。FPR差为负较好，召回/pAUC差为正较好；未定义区间{undefined}/4折；其余不等于证明无效。')
 text+=['','## 解释边界','- 新A/B分别比较原始训练和第十一轮试次平衡版本；新B−新A检验同训练策略下预测力贡献。本轮不新增残差结构。','- 所有FPR工作点只在calibration选择，表中为validation实际FPR，不能把1%校准约束表述为验证1%。','- confirm2沿用raw校准阈值；未重新选择规则。事件延迟排除左删失，仅表示当前gross检测，不是future提前量。','- AP需要结合自然gross占比；高事件覆盖可能伴随持续误报。','- HTT slip标签部分依据力规则，力监督适配后的提升仍不构成独立物理滑移验证。','- 预测力源于同一图像；本轮不能证明实物泛化或世界模型有效。','- 力误差和检测改善的关系需结合force诊断及视觉共同难度对照，不由相关性推断因果。']
 (a.output/'HTT_SUMMARY_ZH.md').write_text('\n'.join(text)+'\n')
 e.js(a.output/'REPORT_AUDIT.json',dict(status='pass',synthetic=summary.get('synthetic',False),evaluation_summary_sha256=e.sha(a.evaluation_dir/'summary.json'),source_sha256=e.sha(__file__),new_runs=24,reused_runs=48,outputs={p.name:e.sha(p) for p in a.output.iterdir() if p.is_file() and p.name!='REPORT_AUDIT.json'}))
if __name__=='__main__':main()
