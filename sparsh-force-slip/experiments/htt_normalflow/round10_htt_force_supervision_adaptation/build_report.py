#!/usr/bin/env python3
"""Consolidate accepted R10 evidence without fitting, selection, or new inference."""
import argparse,csv,hashlib,json,statistics as st
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return list(csv.DictReader(Path(p).open()))
def csvout(p,r):
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)
def main(a):
 o=a.root;dest=o/'reporting';dest.mkdir(exist_ok=True)
 names=['formal_delivery/FORCE_TRAINING_AUDIT.json','formal_delivery/TRAINING_AUDIT.json','current_evaluation/summary.json','current_evaluation/metrics.csv','current_evaluation/paired_ci.csv','current_reporting/REPORT_AUDIT.json','force_aggregate/AUDIT.json','force_aggregate/per_run_summary.csv','force_aggregate/paired_group_ci.csv','force_report/REPORT_AUDIT.json','current_sensitivity/AUDIT.json','current_sensitivity/metrics.csv','association/AUDIT.json','association/correlations.csv','association/figures/PLOT_AUDIT.json','benchmark/F_history_new.json','smoke/SMOKE_AUDIT.json','smoke/OLD_FORCE_PARITY.json']
 sources={str(o/n):sha(o/n) for n in names};sources[str(Path(__file__).resolve())]=sha(__file__)
 for n in ['formal_delivery/FORCE_TRAINING_AUDIT.json','formal_delivery/TRAINING_AUDIT.json']:
  j=json.loads((o/n).read_text());assert j['status']=='pass' and j['expected_count']==12 and len(j['accepted_runs'])==12
 j=json.loads((o/'current_evaluation/summary.json').read_text());assert j['status']=='complete' and j['runs']==36 and j['new_runs']==12 and j['reused_runs']==24
 for n in ['force_aggregate/AUDIT.json','current_sensitivity/AUDIT.json','association/AUDIT.json']:assert json.loads((o/n).read_text())['status']=='pass'
 rr=rows(o/'current_evaluation/metrics.csv');current=[]
 for g in ['V_temporal','F_history_old','F_history_new']:
  for point in ['fixed_0.5','maxBA','FPR0.01','FPR0.05','FPR0.10']:
   x=[r for r in rr if r['group']==g and r['role']=='validation' and r['rule']=='raw' and r['point']==point];assert len(x)==12
   current.append({'group':g,'point':point,**{k:st.mean(float(r[k]) for r in x) for k in ['static_fpr','gross_recall','balanced_accuracy','macro_f1','AP','positive_prevalence','pAUC','false_starts_per_trial','event_recall','mean_delay']}})
 csvout(dest/'CURRENT_KEY_RESULTS.csv',current)
 ff=rows(o/'force_aggregate/per_run_summary.csv');force=[]
 for task in ['slip_force','old_force_regression']:
  for axis in ['shear_x','shear_y','normal']:
   for variant in ['old','new']:
    x=[r for r in ff if r['task']==task and r['axis']==axis and r['variant']==variant and r['role']=='validation' and r['population']=='all' and r['aggregation']=='complete_trial_macro'];assert len(x)==12
    force.append({'task':task,'axis':axis,'variant':variant,**{k:st.mean(float(r[k]) for r in x) for k in ['mae','rmse','bias','delta5_mae','prediction_std','target_std']}})
 csvout(dest/'FORCE_KEY_RESULTS.csv',force)
 ss=rows(o/'current_sensitivity/metrics.csv');sensitivity=[]
 for g in ['V_temporal','F_history_old','F_history_new']:
  for kind in ['unperturbed','force_fit_mean_zero','force_causal_lag1']:
   x=[r for r in ss if r['group']==g and r['intervention']==kind and r['point']=='FPR0.05' and r['rule']=='raw'];assert len(x)==12
   sensitivity.append({'group':g,'intervention':kind,**{k:st.mean(float(r[k]) for r in x) for k in ['static_fpr','gross_recall']}})
 csvout(dest/'SENSITIVITY_KEY_RESULTS.csv',sensitivity)
 b=json.loads((o/'benchmark/F_history_new.json').read_text());assert b['status']=='complete' and b['parity']['pass'] and all(b['frozen'].values());cost=[]
 for mode,reps in b['measurements'].items():
  cost.append({'mode':mode,'median_ms':st.median(x for r in reps for x in r['samples_ms']),'min_repeat_median_ms':min(r['median_ms'] for r in reps),'max_repeat_median_ms':max(r['median_ms'] for r in reps),'peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in reps),'head_parameters':b['parameter_counts']['head'],'full_parameters':b['total_deployment_parameters']})
 csvout(dest/'DEPLOYMENT_COST.csv',cost)
 t=['# 第十轮：力监督适配与低误报滑移检测收益验证','',
 '24/24次正式训练及36模型统一评价完成：12次force适配、12次新力F_history融合；复用第九轮V与旧力F_history各12次。没有更换网络、更新MAE或新增future训练。',
 '', '## 结论先行','',
 'HTT slip试次validation三轴力MAE均下降（2.369→0.859、1.568→0.822、1.717→1.117 N），但专用HTT旧force任务三轴MAE均上升，出现明显任务迁移代价。力误差与slip收益必须分开判断。本轮新力管线的低误报排序面积均值略有改善，但没有稳定转化为更好的校准告警：FPR5%工作点误报下降伴随召回下降，FPR1%工作点到validation后实际误报仍约14.7%。不能宣称低误报目标已解决。专用HTT旧force任务的回归结果也必须与slip域适配效果一起报告。',
 '', '## 1. Force适配改变了什么','',
 '以下仅展示外部validation：各折三个种子的完整试次宏均值再作描述性平均；不是把重叠折看作独立样本。单位N、native shear_x/shear_y/normal。原始GT来自同NPZ，目标参考相对后clip±20；表中统一t>=13，lag5完整。',
 '', '|任务|轴|版本|MAE N|RMSE N|偏置 N|lag5变化MAE N|预测std / GTstd N|','|---|---|---|---:|---:|---:|---:|---:|']
 for r in force:t.append(f"|{r['task']}|{r['axis']}|{r['variant']}|{r['mae']:.4f}|{r['rmse']:.4f}|{r['bias']:.4f}|{r['delta5_mae']:.4f}|{r['prediction_std']:.4f} / {r['target_std']:.4f}|")
 t+=['','完整fit/selection/validation/calibration、逐stage、逐轴与配对CI见force_report和force_aggregate。trial内std的平均不等于所有帧拼接std。适配从旧R5完整force分支开始，保留旧输出normalization，网络未增加；力误差改善不能单独证明slip收益。',
 '新增force仅在outer train的fit组学习、internal selection选checkpoint；R5初始化历史曾使用validation选择，因此不是从未接触validation的全新盲测。内部selection在融合阶段仍属于原outer train，阶段角色不同，不混用为外部评价。',
 'old_force_regression明确指专用HTT旧force任务，不是Sparsh原域。后者坐标、参考和量程未统一，本轮不直接套用或合并物理误差，也不宣称Sparsh性能保持。短试次缺乏t>=13端点已逐项排除，不填零；初版仅补排除元数据的修订保留，数值CSV未改变。',
 '', '## 2. 更准确的力是否改善当前slip','',
 '所有工作点的阈值仅在calibration确定，表中为validation实际结果。四折×三种子的描述性均值；validation同时用于融合checkpoint选择。',
 '', '|组|工作点|实际FPR|gross召回|BA|macro-F1|AP|','|---|---|---:|---:|---:|---:|---:|']
 for r in current:t.append(f"|{r['group']}|{r['point']}|{r['static_fpr']:.2%}|{r['gross_recall']:.2%}|{r['balanced_accuracy']:.4f}|{r['macro_f1']:.4f}|{r['AP']:.4f}|")
 t+=['','FPR5%工作点，新力相对旧力实际FPR约从19.32%降至18.79%，但gross召回从84.58%降至82.72%。相对V也有误报和召回的取舍；不能只展示误报下降。FPR1%工作点新力召回约74.74%，实际FPR约14.68%，仍未满足低误报部署目标。',
 'pAUC（积分TPR在FPR[0,.1]除以.1）均值V约0.4818、旧力0.5359、新力0.5494。新力−旧力仅p1折区间完全正，其余跨零；新力−V仅p3折区间完全正，其余跨零。不能将这些区间数量作全局显著性投票，也不能把跨零等同于证明无效。详见全部seed和逐折CI。',
 'gross自然占比约90.90%，AP约0.987不能独立证明低误报性能。事件检测延迟以帧数报告，并保留左删失及实际事件支持；当前gross最终检出不是future提前预警，static误报不能重新命名成提前预测。',
 '', '## 3. 输入敏感性与失败案例','', '|组|诊断|实际FPR|gross召回|','|---|---|---:|---:|']
 for r in sensitivity:t.append(f"|{r['group']}|{r['intervention']}|{r['static_fpr']:.2%}|{r['gross_recall']:.2%}|")
 t+=['','两项扰动都保持原校准阈值。V预测逐值不变，支持无力泄漏；新力模型屏蔽力后有一定退化，说明它在使用力，但没有证明这种使用能普遍改善检测。因果错位只用历史值，屏蔽/错位可能分布外，不能当物理因果干预。',
 'association按同fold/seed/试次比较力误差和slip变化，三seed先按完整试次平均，再分折描述相关；V保留共同难度对照。四类预先规定的代表案例与三轴GT/新旧预测、概率和阈值见association/figures。相关性不是force通路因果贡献。47/50个折×试次记录平均三轴MAE改善，但误报变化的相关方向跨折混合。p4 / htt/p3_sliding/0_press_13 的MAE从3.305降至1.346 N，V/旧力/新力static FPR仍均100%；p1 / htt/p2_sliding/0_press_74 的FPR从18.18%降至9.09%，同时gross漏报从3.56%升至5.48%。这些是预先规则选出的描述性案例，不能替代全量结果。',
 '', '## 4. 成本、验收与边界','', '|模式|中位数ms|各repeat中位数范围ms|峰值分配字节|新增时序头参数|完整模型参数|','|---|---:|---:|---:|---:|---:|']
 for r in cost:t.append(f"|{r['mode']}|{r['median_ms']:.3f}|{r['min_repeat_median_ms']:.3f}–{r['max_repeat_median_ms']:.3f}|{r['peak_allocated_bytes']}|{r['head_parameters']}|{r['full_parameters']}|")
 t+=['','同第九轮图像预处理＋MAE＋force＋时序头协议，固定fold1/seed20260914验证试次；冷启动9base，流式1新base+8缓存。排除磁盘IO、权重加载、传感采集与机器人控制。共享GPU条件和原始波动保留；不同轮次或GPU的小时间差不解释为架构速度改善。fresh/cache逐块与风险输出误差通过固定容差，所有部署模块冻结。',
 '完整系统约1.01亿参数；轻量只能形容新增约14.3万参数的时序头。HTT滑移阶段标签部分依据力规则，与测得的力GT不构成独立物理证据；公开开发结果不能替代无标记GSmini实物验证，也不证明世界模型。',
 '', '## 5. 下一阶段建议：先保留对照，不急于扩大网络','',
 '1. 保留force分支作为力估计与物理辅助研究分支，但目前不足以把“力融合稳定提高低误报slip”作为已成立主贡献。论文应同时呈现V、旧力、新力和不利种子。',
 '2. 力误差与检测收益分离后，下一步优先检查稳定负例覆盖、标签/阶段边界及阈值跨试次迁移。若做新实验，提前固定一个采样或校准方案，并保留现有网络与独立评价角色；不按本轮validation反复调参。',
 '3. 若需要同时保留专用force任务性能，可在下一轮考虑仅train角色的混合监督或保持旧任务约束，作为单独对照；本轮没有启动。',
 '4. 暂不根据本轮结果直接增加复杂融合层、更换MAE或重训future。未来是否修改结构，应由受控实验明确剩余瓶颈，并保留无力/容量对照。',
 '5. 本轮不提供新的future证据，既有原域future结论及状态预测负向结果保持原边界；独立滑移起点和传感器域验证仍缺失。',
 '', '## 交付入口','',
 '协议与24运行清单：根目录及formal_delivery；全部模型/日志：formal和pairs；完整检测评价/CI/ROC/PR：current_evaluation/current_reporting；force逐试次/旧任务回归/CI：force_evaluation/force_aggregate/force_report；敏感性：current_sensitivity；失败关联：association；成本：benchmark。',
 '最终独立审查、恢复/冻结证据、精确checkpoint索引、同步证明与复现入口见DELIVERY.md、ACCEPTANCE.md、REPRODUCE.md、FINAL_STATUS.json及LOCAL_SYNC_PROOF.json。大型缓存和checkpoint留服务器。']
 (dest/'SUMMARY_ZH.md').write_text('\n'.join(t)+'\n')
 (dest/'REPORT_AUDIT.json').write_text(json.dumps({'status':'pass','formal_runs':24,'comparison_models':36,'sources':sources,'output_hashes':{p.name:sha(p) for p in dest.iterdir() if p.is_file() and p.name!='REPORT_AUDIT.json'},'no_new_training_or_selection':True},indent=2)+'\n')
 print(json.dumps({'status':'pass','outputs':5}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args())
