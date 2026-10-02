#!/usr/bin/env python3
"""Build the Chinese paper-evidence report from completed formal analyses."""
import argparse,csv,json,statistics
from pathlib import Path

def read(p):return json.loads(p.read_text())
def avg(xs):return statistics.fmean(xs)
def tab(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows])
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--code',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 force=read(a.root/'analysis_force/summary/summary.json');source=read(a.root/'analysis_future/source/evaluation.json');htt=read(a.root/'analysis_future/htt/evaluation.json');current=read(a.root/'analysis_summary/summary.json');audit=read(a.root/'formal_delivery/TRAINING_AUDIT.json')
 if any(d['status']!='complete' for d in [force,source,htt,current]) or audit['status']!='pass':raise ValueError('Final analyses not complete')
 desc=list(csv.DictReader((a.root/'analysis_summary/descriptive_summary.csv').open()))
 raw={r['model']:r for r in desc if r['operating_point']=='fixed_0.5' and r['mode']=='raw'}
 seq={r['model']:r for r in desc if r['operating_point']=='fpr_0.05' and r['mode']=='sequential'}
 if len(seq)!=4:raise ValueError('Expected4 sequential5% summaries')
 names=['mae-r3-b','V','F-old','F-adapt']
 ct=[]
 for n in names:
  r=raw[n];ct.append([n,f"{float(r['partial_tpr_auc_0_0p1_mean']):.4f}±{float(r['partial_tpr_auc_0_0p1_std']):.4f}",f"{float(r['balanced_accuracy_mean']):.4f}",f"{float(r['macro_f1_mean']):.4f}",f"{float(r['average_precision_mean']):.4f}",f"{100*float(r['static_fpr_mean']):.2f}%",f"{100*float(r['gross_recall_mean']):.2f}%"])
 st=[]
 for n in names:
  r=seq[n];st.append([n,f"{100*float(r['static_fpr_mean']):.2f}%",f"{100*float(r['gross_recall_mean']):.2f}%",f"{float(r['false_alarm_starts_per_trial_mean']):.4f}",f"{float(r['uncensored_mean_detection_delay_frames_mean']):.2f}"])
 sr=[]
 for h in [1,3,5]:
  for n in ['z_p_slip','z_p_slip_force','z_p_slip_force_pred_delta']:
   r=next(r for r in source['seed_aggregates'] if r['horizon']==h and r['variant']==n and r['method']=='raw');v=r['validation']
   sr.append([h,n,f"{v['average_precision']['mean']:.4f}±{v['average_precision']['std']:.4f}",f"{v['brier']['mean']:.4f}",f"{100*v['fpr']['mean']:.2f}%",f"{100*v['recall']['mean']:.2f}%"])
 hr=[]
 for n in ['risk','base','full_state']:
  for op in ['fixed_0.5','max_ba']:
   r=next(r for r in htt['seed_aggregates'] if r['variant']==n and r['method']=='raw' and r['operating_point']==op);v=r['validation']
   hr.append([n,op,f"{v['average_precision']['mean']:.4f}",f"{v['balanced_accuracy']['mean']:.4f}",f"{100*v['fpr']['mean']:.2f}%",f"{100*v['recall']['mean']:.2f}%"])
 state=[]
 for i,h in enumerate([1,3,8]):
  rr=htt['state_evaluation'];assert len({r['same_eligible_frame_identity_sha256'] for r in rr})==1;learn=[r['learned_full_state'][i] for r in rr]
  state.append([h,f"{rr[0]['persistence_zero_residual'][i]['mse_raw']:.6f}",f"{rr[0]['train_only_ridge'][i]['mse_raw']:.6f}",f"{avg(r['mse_raw'] for r in learn):.6f}",f"{avg(r['r2_raw'] for r in learn):.4f}",f"{100*avg(r['variance_replication_ratio'] for r in learn):.2f}%"])
 timing=[]
 for n,r in force['e2e'].items():
  c,s=r['measurement']['cold'],r['measurement']['streaming'];timing.append([n,f"{r['total_unique_parameters']/1e6:.3f}",f"{c['median_ms']:.2f}",f"{s['median_ms']:.2f}",f"{s['p10_ms']:.2f}–{s['p90_ms']:.2f}",f"{s['peak_allocated_bytes']/1024**2:.1f}"])
 sensitivity=[]
 manifest=read(a.code/'analysis_force/ANALYSIS_MANIFEST.json')
 for variant in ['F-old','F-adapt']:
  rr=[read(Path(j['acceptance_path'])) for j in manifest['jobs'] if j['id'].startswith('sensitivity_'+variant) and j['id'].endswith('_validation')]
  if len(rr)!=12:raise ValueError('Sensitivity seed coverage')
  sensitivity.append([variant]+[f"{avg(r['metrics'][mode]['partial_tpr_auc_0_0p1'] for r in rr):.4f}" for mode in ['original','masked_zero_train_standardized','mismatched_half_cycle']])
 f=force['force_aggregate']['validation'];old=force['source_current_raw_evaluation']['historical_old_head_fresh_regression']['aggregate']['axis']['rmse_n'];fn=[m['aggregate']['prediction_invariant_mean_n']['Fn'] for m in force['source_current_raw_evaluation']['models'].values()]
 lines=['# 第五轮正式实验：力条件化滑移检测与未来接触状态预测','',
 '## 结论先行','',
 '本轮完成66/66正式训练、48份当前检测评价、82项力与部署分析，以及原域/HTT两套future评价。全量checkpoint工程审计通过。结果支持保留力监督与预测力差分作为方法方向，但尚不支持“稳定低误报、保持召回的跨域滑移检测”，也不支持可靠轻量世界模型或实物成功率声明。', '',
 '较明确的正向证据是原域future中的预测力差分。HTT当前检测出现低误报排序改善，但折间差异和召回代价仍明显；HTT首次滑移预警与状态预测仅能作为探索性结果。没有删除负向种子或追加调参。','',
 '## 1. 实验范围与工程验收','',
 '四折×三个固定种子20260914/15/16：force监督适配12、V/F-old/F-adapt融合36；原域future三组×三种子9；HTT future三组×三种子9。共140个依赖作业，均完成。没有新增编码器、NormalFlow训练、test-role或实物采集。', '',
 'MAE冻结；官方权重SHA与seed42构造的完整状态再次独立核验，匹配冻结缓存（官方缺失的register token沿用相同固定初始化）；HTT force仅接受真实力监督；融合阶段force预测与对应R3-B分支冻结，只训练新增模块。正式结果绑定预先固定的inventory、源码、输入哈希及scheduler receipts。准备阶段含实际中断恢复实验；正式阶段66个best checkpoint再次按对应结构strict-load并检查数值与训练审计，全部通过。未发生OOM或正式训练失败重试。分析阶段的MAE-B计时入口发现分支调用错误，单独修复并记录代码修订与未受影响结果的复用证明，不涉及神经重训。', '',
 '沿用2026-09-15 01:55:53 +08:00启动预算和2026-09-25 01:55:53 +08:00截止。训练于9月15日完成，没有重计十天，也不为填满期限追加研究。实测预算见RUNTIME_BUDGET_FINAL.json；最后两天停止派发规则未被触发。', '',
 '## 2. 当前滑移：有方向性收益，仍存在误报—召回代价','',
 '全部组共同使用t≥10范围，static vs gross主任务；incipient不作主二分类标签。下表为12个折/种子开发运行的描述性均值，pAUC附跨运行标准差，不能把12次当独立样本计算推断区间。BA=(static正确率+gross召回率)/2。pAUC是FPR∈[0,0.1]的TPR面积除以0.1。','',
 tab(['模型','pAUC均值±SD','BA@0.5','macro-F1@0.5','AP','实际FPR@0.5','gross召回@0.5'],ct),'',
 f"主评价正类gross自然占比均值为{100*float(raw['V']['positive_prevalence_mean']):.2f}%，因此高AP不能单独证明低误报能力。严格t≥10结果不能直接与此前全帧R3汇总数值横向比较。", '',
 'F-old和F-adapt均通过预注册的“相对V至少3/4折pAUC均值为正”方向准则。但F-adapt每折差值分别为−0.00113、+0.14400、+0.02793、+0.07959；四折仅6/12个配对种子差值为正，p1三个种子均负。多数按完整泄漏组、且三个种子共享重采样的95%区间跨零。该准则是方向性筛选，不能改写为显著、稳定或普遍提升。', '',
 '以下工作点仅在calibration上选择FPR≤5%的阈值与连续确认/滞回参数，并原样应用validation。这里的5%是校准约束，实际验证误报率如下：','',
 tab(['模型','validation实际FPR','gross帧召回','每试次误报告警起点','非左删失事件检出延迟均值/帧'],st),'',
 'F-adapt降低误报的同时损失了gross帧召回，尚未达到“降低误报并保持召回”的完整目标。在1%校准工作点还有1个F-adapt运行永不告警，必须保留其召回代价。5%工作点各模型的可观察起点事件最终均检出，并不等于无延迟或提前预警：每折只有5–7个非左删失事件，另有6–7个在t=10时已经滑移的片段。原含左删失事件的指标保留为告警覆盖率；onset召回和延迟另报。', '',
 '代表失败曲线仍显示某些static阶段长期高置信告警。完整混淆矩阵、逐试次分布、三种子波动、配对区间与所有失败案例均保留，见analysis_summary和各formal/evaluation。', '',
 '## 3. 力监督适配与条件输入','',
 f"HTT validation上12个适配头的三轴RMSE均值分别为{', '.join(f'{x:.3f}' for x in f['mean_axis_rmse_n'])} N；相对train均值常数基线的RMSE改善分别为{', '.join(f'{x:.3f}' for x in f['mean_trainmean_axis_improvement_n'])} N。Fn/Ft/Fmag RMSE为{f['mean_Fn_rmse_n']:.3f}/{f['mean_Ft_rmse_n']:.3f}/{f['mean_Fmag_rmse_n']:.3f} N，全部非塌缩检查通过。", '',
 f"validation目标至少一轴触及裁剪的帧比例，跨运行均值为{100*f['mean_target_any_saturation_fraction']:.2f}%。误差、偏置、预测幅值、五帧变化、饱和子集和逐试次/泄漏组区间都在force分析中报告。这些数字对应参考相对且裁剪的目标，不是无限量程的绝对物理力误差。", '',
 f"原域通过同一次图像编码复核当前已核验旧权重，在14,920样本上三轴RMSE为{', '.join(f'{x:.5f}' for x in old)} N。HTT适配头在原域的Fn均值范围为{min(fn):.3f}–{max(fn):.3f} N，输出分布有明显变化；由于符号轴、参考定义与量程未统一，不把适配头与原域真值的差异当成可比逐轴物理回归误差。旧历史特征缓存缺少生成时权重哈希，只作旁证。", '',
 tab(['条件组','原始力条件pAUC','屏蔽条件pAUC','角色内错配pAUC'],sensitivity),'',
 '屏蔽或错配降低了平均低误报排序表现，说明力条件确实被模型使用；但干预本身改变输入分布，不能单独证明物理因果或实物有效。', '',
 '因此应保留原域旧力头，HTT适配头作为具有明确目标语义的域内版本，不能直接用新头替换全部部署场景。预测力来自同一图像，是监督归纳偏置；Fn/Ft比值不能直接称作摩擦系数。力屏蔽与角色内错配指标、相对原始输入pAUC变化见sensitivity_metric_changes.csv；是否有益以相对V的对照判断。force_slip_association.csv保留全部12个折/种子的力误差—slip增益对应，两个任务不是同批试次，不作试次级因果相关推断。', '',
 '## 4. future：预测力差分有原域价值，HTT提前预警证据不足','',
 '原域维持原future-any-slip目标和MLP结构，训练及部署输入均为预测力/预测力差分。三个种子的固定0.5与阈值无关结果如下；没有独立calibration，禁止用val重新调阈值。', '',
 tab(['窗口','输入组','AP均值±seed SD','Brier','FPR@0.5','召回@0.5'],sr),'',
 '在H5，当前slip概率基线AP为0.9353，Z+pSlip为0.9482，加入预测力差分达到0.9771；Brier也优于不含力差分的组。单独加入力值的收益并不一致。固定乘法门控将差分组H5召回降至约78.23%，因此应将raw future风险与当前检测分开报告。这是future-any-slip窗口分类证据，不能据此声称稳定抓取阶段已有真实提前预警。原域标签不足以独立验证事件起点，所以未报告毫秒或事件提前量。', '',
 'HTT H8只使用首次gross前的static端点：train210帧/35正，calibration29帧/7正，validation39帧/6正。validation合格端点集中于2个试次，12个onset仅2个具有可评价提前窗口，其余10个删失。', '',
 tab(['架构','阈值规则','AP','BA','validation实际FPR','召回'],hr),'',
 '三种架构的raw AP均0.6775，排序相同；加入力条件或状态辅助监督没有形成可辨别的排序提升。全部9个learned模型在固定0.5乘法门控下零告警。这是该门控/阈值下的单类决策退化，不表示raw概率恒定或数值错误。calibration-maxBA重新校准较低的门控分数可恢复部分召回，但会产生很高实际FPR；两种工作点不能混称。低误报工作点的完整结果见HTT评价JSON与报告。仅2个可观察事件上的2/2、8帧结果受H8目标窗口及极少事件支持限制，不能作为独立8帧提前能力，也不能当作8帧实物保证。', '',
 '## 5. 未来特征状态与轻量化边界','',
 '预测目标是冻结MAE pooled特征的未来残差，三窗口联合输出。保持基线为零残差；ridge仅在train拟合，使用相同四步z、pSlip和五维force条件。下表汇总三个种子，误差是特征空间误差。', '',
 tab(['窗口','保持MSE','train-only ridge MSE','full-state MSE','full-state R²','预测/真实残差方差'],state),'',
 '部分窗口MSE优于保持和线性基线，但R²接近零或为负、预测残差方差远低于真实变化，不能仅凭低MSE声称可靠状态建模。三窗口归一化和std已独立重算，参数及输出有限；这属于预测表达不足的负向结果，不是把数值错误计作完成。当前证据不足以使用“已验证的轻量世界模型”作为论文主结论。可以准确描述为“冻结触觉编码器上的轻量接触风险与状态预测模块”。', '',
 tab(['路径','实际前向参数/M','cold中位数/ms','streaming中位数/ms','streaming P10–P90/ms','streaming峰值/MiB'],timing),'',
 '延迟包括驻内存原始图像预处理、编码器及任务头，排除相机采集、NPZ读取、初始化和历史缓存建立。cold F-adapt需重建当前/滞后编码；cold full future需8次编码，streaming只处理新帧并复用已有历史。参数量只统计实际前向路径的唯一参数；峰值显存是该评测进程驻留布局的分配峰值，保守包含辅助副本，不是理论最小显存。每路径预热10、测量50次，共享GPU条件与每次时间记录保留，不能把缓存训练吞吐当作以上部署速度，也不等于相机到控制器闭环延迟。新增融合模块仅38,146参数（V多965），完整系统还包含MAE及原检测/力头；“冻结”本身不代表整个系统轻量。', '',
 '## 6. 论文证据与下一阶段建议','',
 '可支持：①在冻结视觉表征上通过真实力监督建立具有明确目标语义的力条件模块；②HTT低误报排序有方向性收益，但要同时公开召回代价和跨折不确定性；③原域部署一致的预测力差分改善future-any-slip的较远窗口。', '',
 '尚不能支持：所有域统一力标定、验证集达到1%/5%低误报、真实无标记GSmini抓取成功率、稳定抓取前可靠预警、完整世界模型。HTT标签部分由力规则形成，validation参与checkpoint选择，四折重复用于开发，以上限制必须写入论文。', '',
 '下一阶段优先把当前检测与预警决策解耦，保留本轮raw future差分分支作为原域证据；不要默认乘以当前slip概率。当前检测若继续优化，应先针对跨试次static覆盖和校准迁移失败提出有独立评价角色的新协议，再考虑更复杂融合。力头保留域内版本与坐标/量程元数据，先解决输出语义与部署输入一致性。HTT的预警与世界模型部分宜降为探索性附录，不能为了论文亮点隐藏负结果。本轮不自动启动这些后续训练。', '',
 '## 7. 产物与复现','',
 f'服务器大型产物根目录：`{a.root}`。全部best/latest、配置、日志和输入身份见CHECKPOINT_INDEX.csv、RUN_STATUS.csv、冻结RUN_INVENTORY.json及每运行scheduler receipt。正式入口见FORMAL_REPRODUCE.md。', '',
 '独立审查记录与同步证明分别保存；最终同步通过前不标记goal完成。小型报告、图表、配置、必要代码与评价明细归档到formal_delivery_bundle，逐文件SHA256见归档manifest及LOCAL_SYNC_PROOF_FORMAL.json。大型缓存和checkpoint留在服务器。','']
 (a.output/'SUMMARY_ZH.md').write_text('\n'.join(lines))
 inv=read(a.code/'RUN_INVENTORY.json');status=read(a.root/'formal_scheduler_state.json');index=[];runs=[]
 for job in inv['jobs']:
  runs.append({'id':job['id'],'kind':job['kind'],'status':status['jobs'][job['id']],'acceptance_path':job['acceptance_path'],'log_path':job['log_path']})
  if job['kind']!='neural':continue
  s=read(Path(job['acceptance_path']));best=s.get('best_checkpoint',s.get('artifacts',{}).get('best',{}).get('path'));sha=s.get('best_checkpoint_sha256',s.get('artifacts',{}).get('best',{}).get('sha256'))
  index.append({'id':job['id'],'variant':s.get('variant','force'),'fold':s.get('fold','source_or_fixed_htt'),'seed':s.get('seed'),'best_checkpoint':best,'best_sha256':sha,'latest_checkpoint':str(Path(best).with_name('latest.pth')),'summary':job['acceptance_path'],'log':job['log_path']})
 for name,rows in [('CHECKPOINT_INDEX.csv',index),('RUN_STATUS.csv',runs)]:
  with (a.output/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps({'status':'complete','report':str(a.output/'SUMMARY_ZH.md'),'neural_runs':len(index),'all_jobs':len(runs)}))
if __name__=='__main__':main()
