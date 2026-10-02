#!/usr/bin/env python3
"""Build descriptive R11 delivery tables from all verified runs, never best seed."""
import argparse,csv,json,hashlib,statistics as st,math
from pathlib import Path
GROUPS=['V_original','F_history_original','V_balanced','F_history_balanced','F_residual_balanced']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
 with p.open() as f:return list(csv.DictReader(f))
def csvout(p,rr):
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def mean(rr,k):
 vals=[float(x[k]) for x in rr if x[k] not in ('','None') and math.isfinite(float(x[k]))];return st.mean(vals) if vals else None

def main(a):
 o=a.root;out=o/'reporting';out.mkdir(parents=True,exist_ok=True)
 checks=['formal_delivery/TRAINING_AUDIT.json','analysis/AUDIT.json','cross_model_corrections/AUDIT.json','current_reporting/REPORT_AUDIT.json']
 for p in checks:assert json.loads((o/p).read_text())['status']=='pass'
 assert json.loads((o/'formal_delivery/TRAINING_AUDIT.json').read_text())['expected_count']==36
 ev=json.loads((o/'current_evaluation/summary.json').read_text());assert ev['status']=='complete' and ev['runs']==60 and ev['new_runs']==36 and ev['reused_runs']==24 and not ev['synthetic']
 rr=rows(o/'current_evaluation/metrics.csv');current=[]
 for g in GROUPS:
  for point in ['fixed_0.5','maxBA','FPR0.01','FPR0.05','FPR0.10']:
   x=[r for r in rr if r['group']==g and r['role']=='validation' and r['rule']=='raw' and r['point']==point];assert len(x)==12
   current.append(dict(group=g,point=point,**{k:mean(x,k) for k in ['static_fpr','gross_recall','balanced_accuracy','macro_f1','AP','pAUC','positive_prevalence','false_starts_per_trial','event_recall','mean_delay']}))
 csvout(out/'CURRENT_KEY_RESULTS.csv',current)
 ss=rows(o/'analysis/metrics.csv');sensitivity=[]
 for g in GROUPS[2:]:
  for intervention in sorted({r['intervention'] for r in ss if r['group']==g}):
   x=[r for r in ss if r['group']==g and r['intervention']==intervention and r['point']=='FPR0.05' and r['rule']=='raw'];assert len(x)==12
   sensitivity.append(dict(group=g,intervention=intervention,**{k:mean(x,k) for k in ['static_fpr','gross_recall','balanced_accuracy']}))
 csvout(out/'SENSITIVITY_KEY_RESULTS.csv',sensitivity)
 b=json.loads((o/'benchmark/F_residual_balanced.json').read_text());assert b['status']=='complete' and b['parity']['pass'] and all(b['frozen'].values());cost=[]
 for mode,reps in b['measurements'].items():
  cost.append(dict(mode=mode,median_ms=st.median(x for r in reps for x in r['samples_ms']),min_repeat_median_ms=min(r['median_ms'] for r in reps),max_repeat_median_ms=max(r['median_ms'] for r in reps),peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in reps),head_parameters=b['parameter_counts']['head'],full_parameters=b['total_deployment_parameters']))
 csvout(out/'DEPLOYMENT_COST.csv',cost)
 ci=[r for r in rows(o/'current_evaluation/paired_ci.csv') if r['point']=='ranking' and r['metric']=='pAUC'];assert len(ci)==20
 csvout(out/'PAUC_FOLD_CI.csv',ci)
 t=['# 第十一轮：稳定负例覆盖与力条件残差融合验证','',
 '36/36次正式训练与60模型统一评价完成。A/B/C各四折三个seed，另复用R9视觉与R10新力融合各12次。MAE、R10 force、R3视觉上游冻结，没有新增force/future/NormalFlow训练。',
 '', '## 训练改变和归因范围','',
 'A=V_balanced，B=F_history_balanced，C=F_residual_balanced。旧对照V_original/F_history_original分别沿R9视觉和R10新力管线。A/B结构和同seed初始化与历史逐值一致；C完整A视觉分支后增加小力GRU与加性logit残差，总148908参数，比B总143299多3.91%，B有效参数143203，A143179。C的base和修正共同训练，修正也读视觉，不可将全部残差效果归力。',
 '权重w=N/(J*K_j*n_jc)使每试次等贡献、试次内已有类别均分。四折18–20个train试次在共同t>=13后没有static，全局static权重因此从旧50%变为30.39/32.35/31.37/31.13%。这不是新增稳定状态，也不是单纯提高负类权重。train归一化仍沿旧逐帧统计，只改loss贡献。',
 '五个对照分别检验B对旧F训练改变、A对旧V训练改变、B对A力输入、C对B结构、C对A整体力结构。未训练原策略C，不能估计完整训练策略×结构交互效应；A/B参数接近且C约多4%，容量并非严格相同。',
 '', '## 统一开发评价','',
 '以下均为validation四折×三seed描述性均值；不是12个独立试验的总体置信结论。阈值只由calibration确定，validation同时用于checkpoint选择。自然gross占比高，AP不能替代低FPR性能。',
 '', '|组|工作点|实际FPR|gross召回|BA|macro-F1|AP|低FPR面积|','|---|---|---:|---:|---:|---:|---:|---:|']
 for r in current:t.append(f"|{r['group']}|{r['point']}|{r['static_fpr']:.2%}|{r['gross_recall']:.2%}|{r['balanced_accuracy']:.4f}|{r['macro_f1']:.4f}|{r['AP']:.4f}|{r['pAUC']:.4f}|")
 t+=['','低FPR面积为积分TPR@FPR[0,.1]/.1，不是McClish标准化AUC。实际calibration工作点与validation同FPR/同召回描述曲线分开，后者不构成可部署阈值。完整seed波动、连续告警/检出/延迟、删失/never-alarm及incipient分布见current_evaluation/current_reporting。',
 '', '## 逐折低FPR排序差异与配对区间','',
 '|candidate − base|fold|差值|95%区间|','|---|---|---:|---|']
 for r in ci:t.append(f"|{r['candidate']} − {r['base']}|{r['fold']}|{float(r['difference']):.4f}|[{float(r['ci_lower']):.4f}, {float(r['ci_upper']):.4f}]|")
 t+=['','200次完整泄漏组配对重采样，同次抽样共享三个seed；各fold分开，不池化重叠fold/相邻帧，不把区间数量当全局显著性投票。区间跨零不等于证明无效；一个seed或一个fold改善也不等于稳定增益。',
 '', '## 力与残差诊断','',
 '|组|诊断|实际FPR|gross召回|','|---|---|---:|---:|']
 for r in sensitivity:t.append(f"|{r['group']}|{r['intervention']}|{r['static_fpr']:.2%}|{r['gross_recall']:.2%}|")
 t+=['','表中采用raw/FPR5。force屏蔽、因果lag1、residual_off均保持完整模型原cal阈值；base_only_independent_calibration由base自身cal选阈值单列，不能混为固定阈值消融。base不是独立A。扰动可能分布外，不构成物理因果实验。',
 'analysis提供残差正负、幅度、|r|>=1.9饱和、试次/probe/观察序列位置及纠错误杀；cross_model_corrections另比较C对A/B各自cal-FPR5完整管线的对齐帧纠错、引入错误和共同错误。各自阈值不同，不能把该比较孤立解释为结构因果效应。代表图保留全部seed及成功、伤害、持续误报案例。',
 '', '## 端到端成本与适用边界','',
 '|模式|中位数ms|repeat中位数范围ms|峰值分配字节|头参数|完整参数|','|---|---:|---|---:|---:|---:|']
 for r in cost:t.append(f"|{r['mode']}|{r['median_ms']:.3f}|{r['min_repeat_median_ms']:.3f}–{r['max_repeat_median_ms']:.3f}|{r['peak_allocated_bytes']}|{r['head_parameters']}|{r['full_parameters']}|")
 t+=['','固定首折seed20260914、既定验证试次t13，cold9base/stream1newbase。包含图像预处理/H2D/MAE/force/任务头，排除磁盘IO、权重加载、传感采集及机器人控制。共享GPU波动保留；cached训练吞吐不能当部署速度，约1亿参数完整系统与约15万参数新增头分开报告。',
 'HTT滑移标签部分依赖力规则；R5初始化历史使用validation、本轮融合也由validation选checkpoint，均为重叠折开发评价。没有无标记GSmini实物成功率、新future或世界模型有效性证据。',
 '', '## 交付','',
 '结论与下一阶段建议见CONCLUSIONS_ZH.md；完整协议、训练权重、配置、日志、best/latest索引、真实提交点中断恢复/冻结、独立审查及SHA同步见根目录DELIVERY/ACCEPTANCE/REPRODUCE/FINAL_STATUS与LOCAL_SYNC_PROOF。大型缓存及checkpoint留服务器，不自动启动下一轮。']
 (out/'SUMMARY_ZH.md').write_text('\n'.join(t)+'\n')
 sources=[o/'current_evaluation/metrics.csv',o/'current_evaluation/paired_ci.csv',o/'current_evaluation/summary.json',o/'analysis/metrics.csv',o/'benchmark/F_residual_balanced.json',*[o/p for p in checks],Path(__file__)]
 audit={'status':'pass','sources':{str(p):sha(p) for p in sources},'outputs':{p.name:sha(p) for p in out.iterdir() if p.name in ['SUMMARY_ZH.md','CURRENT_KEY_RESULTS.csv','SENSITIVITY_KEY_RESULTS.csv','DEPLOYMENT_COST.csv','PAUC_FOLD_CI.csv']}}
 (out/'REPORT_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps({'status':'pass','outputs':len(audit['outputs'])}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args())
