#!/usr/bin/env python3
"""Consolidate accepted R9 evidence; no model or threshold selection."""
import argparse,csv,hashlib,json,statistics as st
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return list(csv.DictReader(p.open()))
def writecsv(p,rs):
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def main(a):
 o=a.root;dest=o/'reporting';dest.mkdir(exist_ok=True)
 files=['formal_delivery/TRAINING_AUDIT.json','formal_delivery/CHECKPOINT_INDEX.csv','current_evaluation/summary.json','current_evaluation/metrics.csv','current_evaluation/paired_ci.csv','current_reporting/REPORT_AUDIT.json','current_reporting/HTT_SUMMARY_ZH.md','future_robustness/analysis/AUDIT.json','future_robustness/analysis/SUMMARY_ZH.md','force_diagnostics/SUPPORT_AUDIT.json','force_diagnostics/REPORT_AUDIT.json','force_diagnostics/SUMMARY_ZH.md','force_diagnostics/ASSOCIATION_AUDIT.json','force_diagnostics/ASSOCIATION_ZH.md','current_sensitivity/AUDIT.json','current_sensitivity/metrics.csv','prepare/support/SUPPORT_AUDIT.json','BUDGET_UPDATE.json']
 for g in ['V_temporal','F_history','F_delta']:files.append(f'benchmark/{g}.json')
 sources={str(o/name):sha(o/name) for name in files};sources[str(Path(__file__).resolve())]=sha(Path(__file__).resolve())
 training=json.loads((o/'formal_delivery/TRAINING_AUDIT.json').read_text());evaluation=json.loads((o/'current_evaluation/summary.json').read_text());sensitivity=json.loads((o/'current_sensitivity/AUDIT.json').read_text())
 if training['status']!='pass' or len(training['accepted_runs'])!=36 or evaluation['status']!='complete' or evaluation['new_runs']!=36 or sensitivity['status']!='pass':raise ValueError('incomplete inputs')
 metric=rows(o/'current_evaluation/metrics.csv');ci=rows(o/'current_evaluation/paired_ci.csv');sens=rows(o/'current_sensitivity/metrics.csv')
 table=[]
 for g in ['V_temporal','F_history','F_delta']:
  for point in ['fixed_0.5','FPR0.01','FPR0.05','FPR0.10']:
   rr=[r for r in metric if r['group']==g and r['role']=='validation' and r['rule']=='raw' and r['point']==point]
   if len(rr)!=12:raise ValueError('missing fold/seed')
   table.append({'group':g,'point':point,**{k:st.mean(float(r[k]) for r in rr) for k in ['static_fpr','gross_recall','balanced_accuracy','macro_f1','AP','positive_prevalence','pAUC','event_recall','events','left_censored','mean_delay','false_starts_per_trial']}})
 writecsv(dest/'KEY_CURRENT_WORKPOINTS.csv',table)
 costs=[]
 for g in ['V_temporal','F_history','F_delta']:
  b=json.loads((o/f'benchmark/{g}.json').read_text())
  if b['status']!='complete' or not b['parity']['pass'] or not all(b['frozen'].values()):raise ValueError('benchmark failed')
  for mode,vals in b['measurements'].items():
   samples=[t for v in vals for t in v['samples_ms']];costs.append({'group':g,'mode':mode,'median_ms':st.median(samples),'min_repeat_median_ms':min(v['median_ms'] for v in vals),'max_repeat_median_ms':max(v['median_ms'] for v in vals),'peak_allocated_bytes':max(v['peak_allocated_bytes'] for v in vals),'head_parameters':b['parameter_counts']['head'],'total_parameters':b['total_deployment_parameters']})
 writecsv(dest/'DEPLOYMENT_COST.csv',costs)
 stable=[]
 for g in ['V_temporal','F_history','F_delta']:
  for kind in ['unperturbed','force_fit_mean_zero','force_causal_lag1']:
   rs=[r for r in sens if r['group']==g and r['intervention']==kind and r['point']=='FPR0.05' and r['rule']=='raw']
   if len(rs)!=12:raise ValueError('sensitivity grid')
   stable.append({'group':g,'intervention':kind,**{k:st.mean(float(r[k]) for r in rs) for k in ['static_fpr','gross_recall','event_recall','mean_delay']}})
 writecsv(dest/'CURRENT_FORCE_SENSITIVITY.csv',stable)
 text=['# 第九轮综合结论：HTT当前检测与原域future可靠性','',
 '全部36次正式训练完成，36个新模型和48个历史参照完成共同端点评价；未新增future/force/NormalFlow训练，未消费test或采集实物。结果不支持“时序力差分已普遍解决HTT低误报检测”。尚未解决的是阈值跨角色迁移和稳定负例区分；另观察到预测力幅度压缩及动态误差，其对失败的因果贡献尚未确定。','',
 '## 1. 当前slip：力历史只有有限且不稳定的收益','',
 '以下为四折×三种子的描述性均值。它们不是12次独立数据试验；validation同时用于checkpoint选择。FPR工作点只在calibration选择，表中是实际validation静态误报率。','',
 '|组|校准工作点|实际FPR|gross召回|BA|AP|gross自然占比|', '|---|---|---:|---:|---:|---:|---:|']
 for r in table:
  text.append(f"|{r['group']}|{r['point']}|{100*r['static_fpr']:.2f}%|{100*r['gross_recall']:.2f}%|{r['balanced_accuracy']:.4f}|{r['AP']:.4f}|{100*r['positive_prevalence']:.2f}%|")
 text+=['','在calibration FPR5%工作点，F-history相对V的召回约增0.51个百分点，实际FPR约降0.65个百分点；F-delta相对F-history的召回反而约降3.36个百分点，而FPR只降约0.23个百分点。不能把减少告警本身当检测改善。',
 'F-history−V的低FPR排序面积仅1/4折配对区间完全为正；F-delta−F-history没有折获得稳定正向pAUC区间，1/4折为负。各折区间详见current_evaluation/paired_ci.csv；这不是通过数区间给显著性投票，也不意味着跨0就证明无效。',
 'calibration FPR1%到validation仍变为约14%，FPR5%变为约19–20%。AP约0.986必须和约90.90%的gross自然占比一起读，不能据此称低误报性能优秀。',
 'raw FPR5%下各组事件召回均值接近1，但每折仅约4.25个非左删失事件、约8.25个左删失段；检出事件平均延迟约12.5–13.3帧。长期gross最终被检出不等于及时检测，更不是滑移前预警。误报案例中存在static阶段分数已接近1并持续告警的试次。',
 '', '## 2. 力误差与输入敏感性', '',
 '新的同文件审计确认：HTT slip NPZ自身包含6d_force/ref_force。此前复用的force-task缓存没有导出这项GT，不代表原始slip数据没有力。现已按同NPZ帧顺序、参考相对与原生尺度核验，只用于诊断，没有跨basename拼接或重新训练。GT来源/裁剪语义及逐轴结果见force_diagnostics。',
 'validation剪切x MAE约2.07–2.79N，预测标准差约0.55–0.66N而GT约2.34–3.01N；剪切y亦有幅度压缩。normal总体幅度较接近，但部分折存在偏置。说明专用force试次训练的力模型在slip试次上并非无误差的物理输入。',
 '固定FPR5%工作点的输入诊断：','', '|组|扰动|实际FPR|gross召回|','|---|---|---:|---:|']
 for r in stable:text.append(f"|{r['group']}|{r['intervention']}|{100*r['static_fpr']:.2f}%|{100*r['gross_recall']:.2f}%|")
 text+=['','V对力扰动逐位不变，支持视觉无力泄漏。F组屏蔽力后仅小幅退化，当前模型对该力表征的依赖有限；力置均值、因果错位均可能分布外，不能当物理因果实验。',
 '力误差关联按完整试次先跨三个seed平均、再各折描述，共96项相关；无static的试次不强填误报率。部分折力误差与误报正相关，但V也呈相似关系，可能反映共同场景难度，不能归因于force通路；gross漏报关联跨折没有一致方向。',
 '', '## 3. 原域future：已取得的收益保留，但历史缺失增加误报','',
 '第八轮三个固定候选、全部三个seed完成27份新扰动预测及原始输出一致性复核。所有阈值/规则来自第八轮，未重新校准或训练。1188工作点、2160分层记录、54项配对CI保留全部种子与不可达工作点。',
 'H3融合+hazard原始事件召回78.04%、试次误报5.26%；预设±0.1 fit标准差力偏置下变化较小。倒数第二历史输出缺失并以前值保持后，召回约79.41%，试次误报升到6.82%；其他两个候选也出现误报增加。这不是缺失输入带来性能提升。',
 '分层仅能确认dataset batch proxy，不能假定为独立物体；预测集中临近标签滑移。没有已核验的第二张早期无接触参考，参考替换未触发。该项有明确证据而非静默跳过。',
 '', '## 4. 现有独立数据支持仍不足','',
 '复用第六轮身份与支持审计，HTT严格future事件支持仍不足，Sparsh原域仍是重复开发数据，NormalFlow没有独立slip起点，ToucHD下载未操作或检查。没有新增独立future-slip盲测，不通过重划开发集改变这一事实。',
 '', '## 5. 成本与可复现性','', '|组|模式|中位数ms|新增头参数|完整驻留模型参数|峰值分配字节|','|---|---|---:|---:|---:|---:|']
 for r in costs:text.append(f"|{r['group']}|{r['mode']}|{r['median_ms']:.3f}|{r['head_parameters']}|{r['total_parameters']}|{r['peak_allocated_bytes']}|")
 text+=['','固定fold1/seed20260914/validation首试次t13，同一物理GPU串行测量。流式含一组新图像预处理、MAE和所需任务头，保留八个历史base；冷启动处理九个base。排除文件IO、权重加载、传感采集和机器人控制。fresh/cache逐块误差及风险误差均通过预先固定容差，全部上游冻结。V没有执行force；共享GPU下小时间差不作结构速度优劣结论。完整系统约0.94–1.01亿参数，“轻量”只指新增约14.3万参数的头。',
 '', '## 6. 论文证据与下一阶段建议（不自动执行）','',
 '1. 原域future的等历史力差分与一致事件风险是目前更清晰的正向证据；本轮补充了受控扰动下的可靠性边界。',
 '2. HTT当前力融合的改善仍不稳，必须保留视觉容量对照和本轮负向差分结果，不能将原域future收益外推为当前检测普遍改善。',
 '3. 下一步优先核查专用force任务与slip试次的力监督分布差异，并设计仅train-role、保持独立校准/评价的域内force适配对照。应预先固定方案，并同步检查稳定负例覆盖与阈值迁移，不能按本轮validation反复调参。',
 '4. 因HTT slip标签本身受力规则影响，即便后续力监督适配改善，也需独立事件起点或新域来验证物理可传递性。现有数据无法替代真实无标记GSmini验证。',
 '5. 第八轮状态预测不及简单基线的负向结论继续保留；本轮不支持已验证世界模型、动作后果建模或实物成功率。',
 '', '## 交付与修订','',
 '完整运行/权重：formal_delivery；当前全折全seed指标/CI/失败曲线：current_evaluation与current_reporting；原域稳健性：future_robustness/analysis；力GT及关联：force_diagnostics；敏感性：current_sensitivity；端到端：benchmark；支持审计：prepare/support。',
 '训练数值协议未改变。验收脚本补强输出、角色与checkpoint交叉绑定；历史预测reader仅修复frame/t列名兼容，保留旧源码和派生manifest。冻结评价协议原文的GT合同表述另有独立澄清文档，源哈希已恢复原锁，未追溯改动训练/阈值规则。',
 '全部36次使用同一冻结来源规则、三固定seed和共同t>=13原始历史。checkpoint真实中断恢复逐值一致；未消费test。独立最终审查与逐文件同步证明见最终DELIVERY/FINAL_STATUS及LOCAL_SYNC_PROOF。']
 (dest/'SUMMARY_ZH.md').write_text('\n'.join(text)+'\n')
 outputs={p.name:sha(p) for p in dest.iterdir() if p.is_file() and p.name!='REPORT_AUDIT.json'}
 (dest/'REPORT_AUDIT.json').write_text(json.dumps({'status':'pass','sources':sources,'output_hashes':outputs,'formal_runs':36,'historical_comparisons':48,'training_protocol_changed':False,'test_role_consumed':False},indent=2)+'\n')
 print(json.dumps({'status':'pass','outputs':len(outputs)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args())
