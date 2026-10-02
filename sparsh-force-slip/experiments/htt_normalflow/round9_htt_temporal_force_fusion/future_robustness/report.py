"""Describe fixed-threshold robustness, retaining all three seeds."""
from pathlib import Path
import csv,json,sys,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(sys.argv[1])
analysis_audit=json.loads((p/'AUDIT.json').read_text())
assert analysis_audit['status']=='pass'
for name in ['metrics.csv','strata.csv','paired_ci.csv']:
 assert hashlib.sha256((p/name).read_bytes()).hexdigest()==analysis_audit['output_hashes'][name]
rows=list(csv.DictReader((p/'metrics.csv').open()));chosen=[r for r in rows if r['operating_point']=='trial_FA_0.10']
groups=sorted({r['group'] for r in chosen});kinds=['unperturbed','force_bias_plus_0p1_fit_std','force_bias_minus_0p1_fit_std','missing_base_penultimate_hold']
lines=['# 第九轮原域future可靠性重评','', '复用第八轮三个P3模型及全部三个种子，H1/H3/H5与严格稳定端点不变。下表为trial_FA_0.10工作点（calibration试次误报约束10%），由第八轮calibration确定，未重新选择规则或阈值。误报为实际开发评价试次误报比例。','', '|模型|扰动|H|事件召回均值±种子SD|试次误报均值±种子SD|命中事件平均提前帧|','|---|---|---|---|---|---|']
summary=[]
for g in groups:
 for k in kinds:
  for h in [1,3,5]:
   rr=[r for r in chosen if r['group']==g and r['intervention']==k and int(r['horizon'])==h]
   assert len(rr)==3 and {int(r['seed']) for r in rr}=={20260914,20260915,20260916}
   values={n:np.asarray([float(r[n]) for r in rr]) for n in ['event_recall','trial_false_alarm_rate','mean_lead_frames']}
   summary.append(dict(group=g,intervention=k,horizon=h,**{n:float(np.mean(v)) for n,v in values.items()}))
   e,f,l=values.values();lines.append(f'|{g}|{k}|{h}|{e.mean():.4f} ± {e.std(ddof=1):.4f}|{f.mean():.4f} ± {f.std(ddof=1):.4f}|{np.nanmean(l):.3f}|')
fig,axs=plt.subplots(1,2,figsize=(12,4))
for ax,metric in zip(axs,['event_recall','trial_false_alarm_rate']):
 for i,g in enumerate(groups):
  vals=[next(r[metric] for r in summary if r['group']==g and r['intervention']==k and r['horizon']==3) for k in kinds]
  ax.plot(range(4),vals,'o-',label=g)
 ax.set_xticks(range(4),['original','+0.1 std','-0.1 std','hold prev'],rotation=15);ax.set_ylabel(metric);ax.set_title('H3; R8 trial_FA_0.10 threshold');ax.grid(alpha=.2)
axs[0].legend(fontsize=7);fig.tight_layout();fig.savefig(p/'robustness_H3.png',dpi=150);plt.close(fig)
lines+=['','## 解释边界','','- 力偏置在所有历史位置相同，因此真实差分不变；这里测量预测力零点偏移敏感性，不代表接触力物理干预。','- 历史缺失为倒数第2基础输出因果保持，当前输出不改，差分由改变后的历史重新构造。','- 配对区间在paired_ci.csv，使用完整leakage_group重采样200次，种子在每次重采样内部平均；记录有效重复数及不可用状态。','- 逐试次见trial_metrics.csv；距起点分层只报告帧指标，不从裁剪后的窗口伪造告警起点。单类分层BA/F1/AP留空，不能比较其AP优劣。','- stable_prefix_length以首次标签起点的原始帧索引为代理，并非独立标注接触时长；batch_proxy也不是已核实物体身份。','- 当前已验收缓存未提供第二个经审计的早期无接触参考，因此参考替换不触发。','- reused_intervention_metrics.csv和reused_fixed_multiplicative_gate_diagnostic.csv复用第八轮原规则诊断；不重新校准。','- 外层数据已反复参与开发。结果不是独立盲测、物理因果证据或无标记GSmini实物成功率。','', '![H3稳定性](robustness_H3.png)']
strata=list(csv.DictReader((p/'strata.csv').open()))
def avg_stratum(group,level,key):
 rr=[r for r in strata if r['group']==group and r['intervention']=='unperturbed' and r['horizon']=='3' and r['stratum']=='batch_proxy' and r['level']==level]
 return float(np.mean([float(r[key]) for r in rr]))
levels=sorted({r['level'] for r in strata if r['stratum']=='batch_proxy'})
differences=[avg_stratum('P3_fusion_hazard',v,'event_recall')-avg_stratum('P3_concat_independent',v,'event_recall') for v in levels]
lines+=['','## 分层与扰动结果说明','',f'- H3事件召回相对容量拼接对照，在{len(levels)}个批次代理中{sum(v>1e-12 for v in differences)}个提高、{sum(abs(v)<=1e-12 for v in differences)}个持平、{sum(v< -1e-12 for v in differences)}个降低。这是描述性批次统计，不能替代物体级独立验证或多重比较校正。','- H3融合hazard在固定小力偏置下，召回及试次误报变化较小；历史缺失因果保持在三个模型上均增加H3试次误报；这不是统计显著性声明。该结论限于本次预注册扰动幅度和位置，不能推断任意噪声/丢帧都稳健。','- 距滑移起点1帧的检出明显高于2–3帧区间；短时预警仍主要集中于临近滑移，不能解释为长提前量稳定预测。','- 无观测起点分层的稳定负端点支持很少，不能单独据此声称稳定无滑移场景性能充分。']
(p/'SUMMARY_ZH.md').write_text('\n'.join(lines)+'\n')
(p/'REPORT_AUDIT.json').write_text(json.dumps({'status':'pass','analysis_audit_sha256':hashlib.sha256((p/'AUDIT.json').read_bytes()).hexdigest(),'all_three_seeds':True,'threshold_refit':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'files':{n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in ['SUMMARY_ZH.md','robustness_H3.png','metrics.csv','paired_ci.csv']}},indent=2)+'\n')
