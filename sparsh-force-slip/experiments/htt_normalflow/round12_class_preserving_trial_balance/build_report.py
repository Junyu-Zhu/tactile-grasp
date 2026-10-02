#!/usr/bin/env python3
"""R12 full-grid descriptive report; no best-seed selection or threshold changes."""
import argparse,csv,json,hashlib,math,statistics as st
from pathlib import Path
GROUPS=('V_original','F_history_original','V_balanced','F_history_balanced','V_class_trial_balanced','F_class_trial_balanced')
SEEDS=(20260914,20260915,20260916)
PAIRS=(('V_class_trial_balanced','V_original'),('F_class_trial_balanced','F_history_original'),('V_class_trial_balanced','V_balanced'),('F_class_trial_balanced','F_history_balanced'),('F_class_trial_balanced','V_class_trial_balanced'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return list(csv.DictReader(Path(p).open()))
def dump(p,rr):
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def mean(rr,k):
 vals=[float(r[k]) for r in rr if r[k] not in ('','None') and math.isfinite(float(r[k]))];return st.mean(vals) if vals else None
def fmt(x,pct=False):return 'NA' if x is None else f'{x:.2%}' if pct else f'{x:.4f}'
def grid(rr,group):assert {(r['fold'],int(r['seed'])) for r in rr}=={(f'htt_leave_p{f}',s) for f in range(1,5) for s in SEEDS} and len(rr)==12,group

def main(a):
 o=a.root;out=o/'reporting';out.mkdir(parents=True,exist_ok=True);sources={}
 def artifact(p):p=Path(p);sources[str(p)]=sha(p);return p
 def receipt(relative,status='pass'):
  p=artifact(o/relative);r=json.loads(p.read_text());assert r['status']==status
  for field in ('outputs','output_hashes'):
   for name,h in r.get(field,{}).items():
    q=p.parent/name;assert sha(q)==h;sources[str(q)]=h
  return r
 tr=receipt('formal_delivery/TRAINING_AUDIT.json');assert tr['expected_count']==24 and tr['test_role_consumed'] is False
 ev=receipt('current_evaluation/summary.json','complete');assert not ev['synthetic'] and (ev['runs'],ev['new_runs'],ev['reused_runs'])==(72,24,48)
 receipt('current_reporting/REPORT_AUDIT.json');sens=receipt('current_sensitivity/AUDIT.json');assert sens['runs']==24
 unc=receipt('calibration_uncertainty/AUDIT.json');assert unc['runs']==72 and unc['draws_per_fold']==200 and not unc['synthetic'] and not unc['validation_consumed'] and not unc['deployment_threshold_changed']
 case=receipt('fixed_case/AUDIT.json');assert case['models']==18 and case['metrics']==36
 b=receipt('benchmark/F_class_trial_balanced.json','complete');assert b['group']=='F_class_trial_balanced' and b['parity']['pass'] and all(b['frozen'].values())
 rr=read(o/'current_evaluation/metrics.csv');current=[]
 for g in GROUPS:
  for point in ('fixed_0.5','maxBA','FPR0.01','FPR0.05','FPR0.10'):
   x=[r for r in rr if r['group']==g and r['role']=='validation' and r['point']==point and r['rule']=='raw'];grid(x,g);current.append(dict(group=g,point=point,**{k:mean(x,k) for k in ('static_fpr','gross_recall','balanced_accuracy','macro_f1','AP','pAUC','positive_prevalence','false_starts_per_trial','event_recall','mean_delay')}))
 dump(out/'CURRENT_KEY_RESULTS.csv',current)
 ss=read(o/'current_sensitivity/metrics.csv');sensitivity=[]
 for g in GROUPS[-2:]:
  for intervention in ('unperturbed','force_fit_mean_zero','force_causal_lag1'):
   x=[r for r in ss if r['group']==g and r['intervention']==intervention and r['point']=='FPR0.05' and r['rule']=='raw'];grid(x,g);sensitivity.append(dict(group=g,intervention=intervention,**{k:mean(x,k) for k in ('static_fpr','gross_recall','balanced_accuracy')}))
 dump(out/'SENSITIVITY_KEY_RESULTS.csv',sensitivity)
 ci=[r for r in read(o/'current_evaluation/paired_ci.csv') if r['point']=='ranking' and r['metric']=='pAUC'];assert len(ci)==20 and {(r['candidate'],r['base']) for r in ci}==set(PAIRS);dump(out/'PAUC_FOLD_CI.csv',ci)
 uu=read(o/'calibration_uncertainty/threshold_uncertainty.csv');assert len(uu)==216;dump(out/'CALIBRATION_UNCERTAINTY_PER_RUN.csv',uu);us=[]
 for g in GROUPS:
  for fpr in (.01,.05,.1):
   x=[r for r in uu if r['group']==g and float(r['fpr_constraint'])==fpr];grid(x,g);us.append(dict(group=g,fpr_constraint=fpr,valid_fraction=mean(x,'valid_fraction'),never_alarm_fraction_among_valid=mean(x,'never_alarm_fraction'),invalid_no_static=sum(int(r['invalid_no_static']) for r in x),invalid_no_gross=sum(int(r['invalid_no_gross']) for r in x)))
 dump(out/'CALIBRATION_UNCERTAINTY_DESCRIPTIVE.csv',us)
 cc=read(o/'fixed_case/fixed_case_metrics.csv');assert len(cc)==36;dump(out/'FIXED_CASE_PER_SEED.csv',cc)
 cost=[]
 for mode,reps in b['measurements'].items():
  assert len(reps)==3 and all(len(r['samples_ms'])==30 for r in reps);cost.append(dict(mode=mode,median_ms=st.median(x for r in reps for x in r['samples_ms']),min_repeat_median_ms=min(r['median_ms'] for r in reps),max_repeat_median_ms=max(r['median_ms'] for r in reps),peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in reps),head_parameters=b['parameter_counts']['head'],full_parameters=b['total_deployment_parameters']))
 dump(out/'DEPLOYMENT_COST.csv',cost)
 t=['# 第十二轮：保留类别总贡献的试次平衡验证','','24次新训练、72模型统一评价：每组四折×三个固定种子，六组完整纳入。新A=V_class_trial_balanced，新B=F_class_trial_balanced。历史组为R9视觉V_original、R10新力F_history_original、R11 V_balanced/F_history_balanced。没有新增残差结构、force、future、NormalFlow或编码器训练。','','## 训练改变与归因','','新权重w=N/(2·J_c·n_jc)：static/gross各占总损失贡献50%，类别内具有该类的试次等贡献，再按该试次类内帧数均分。仅改变现有样本贡献，不能补充缺失接触状态。与R11试次整体均分不同：R11部分试次没有static，导致全局static贡献低于50%。归一化、网络与部署输入沿用原协议，新头按固定seed重新初始化。','','五个配对：新A−原V、新B−原F检验整体训练改变；新A−R11V、新B−R11F检验两种完整权重方案的差异，不能全部归因于static总权重变化；新B−新A检验相同训练策略下的预测力增益。没有通过追加网络复杂度解释结果。','','## 当前滑移检测','','表中为validation四折三seed的描述性均值，不是12次独立总体试验。validation用于checkpoint选择；校准阈值来自calibration并原样应用。自然gross占比应结合AP解释。','','|组|工作点|实际FPR|gross召回|BA|macro-F1|AP|低FPR面积|','|---|---|---:|---:|---:|---:|---:|---:|']
 for r in current:t.append('|'+ '|'.join([r['group'],r['point'],fmt(r['static_fpr'],True),fmt(r['gross_recall'],True)]+[fmt(r[k]) for k in ('balanced_accuracy','macro_f1','AP','pAUC')])+'|')
 t+=['','低FPR面积为TPR在FPR[0,.1]积分除以.1，非McClish标准化AUC。同FPR/同召回validation包络仅描述，不能替代calibration可执行工作点。单纯告警减少或AP提高不能证明低误报改善。完整逐seed、连续告警、事件延迟、删失、never-alarm见current_evaluation/current_reporting。','','## 逐折排序差异与配对区间','','|candidate−base|fold|差值|95%区间|','|---|---|---:|---|']
 for r in ci:t.append(f"|{r['candidate']}−{r['base']}|{r['fold']}|{fmt(float(r['difference']))}|[{r['ci_lower']}, {r['ci_upper']}]|")
 t+=['','200次完整泄漏组配对重采样，同fold抽样共享三个seed。重叠fold不合并作独立样本，区间跨零不是证明无效，也不以区间数量投票。','','## 校准阈值的不确定性','','额外200次仅calibration完整组重采样，对所有模型和seed共享fold内同一组抽样。无static或无gross的抽样分别标无效；没有读取validation概率来选择阈值，没有改变主工作点。以下有效率/永不告警比例均是12运行描述性均值，不是部署置信保证。','','|组|FPR约束|有效抽样比例|有效抽样中never-alarm比例|无static次数|无gross次数|','|---|---:|---:|---:|---:|---:|']
 for r in us:t.append('|'+ '|'.join([r['group'],fmt(r['fpr_constraint'],True),fmt(r['valid_fraction'],True),fmt(r['never_alarm_fraction_among_valid'],True),str(r['invalid_no_static']),str(r['invalid_no_gross'])])+'|')
 t+=['','各运行阈值q2.5/50/97.5原样保存在CALIBRATION_UNCERTAINTY_PER_RUN.csv。禁止跨fold合并这些区间为单个阈值或选择更有利抽样。少量校准试次的有效率与阈值不稳定性是诊断，不证明阈值迁移已经解决。','','## 力敏感性与固定历史失败案例','','|新组|干预|实际FPR|gross召回|','|---|---|---:|---:|']
 for r in sensitivity:t.append(f"|{r['group']}|{r['intervention']}|{fmt(r['static_fpr'],True)}|{fmt(r['gross_recall'],True)}|")
 t+=['','raw/FPR5下，预测力均值屏蔽与因果lag1均沿用原calibration阈值；视觉组须逐值不变。扰动可能分布外，不能由敏感性推断物理因果。','','固定案例为此前已披露的htt/p3_sliding/0_press_13、fold4：六组所有三个seed的曲线和阈值见fixed_case/fixed_case_all_seeds.png，FIXED_CASE_PER_SEED.csv保留逐seed误报/漏检和类别适用性。若无gross，gross指标为NA，不是零漏检。案例不能代表总体。','','复用R11已审视觉邻域证据，核验历史来源、输出和新模型prepared身份。该邻域诊断是R11训练后有限补充，仅固定单fold单seed；不扩充为三seed覆盖结论。static/gross候选密度不均、相邻帧相关、固定表征限制均保留，不能证明物理状态覆盖或唯一根因。详见fixed_case/COVERAGE_REUSE_ZH.md。','','## 端到端成本','','|模式|中位数ms|repeat中位范围ms|峰值字节|头参数|完整参数|','|---|---:|---|---:|---:|---:|']
 for r in cost:t.append(f"|{r['mode']}|{r['median_ms']:.3f}|{r['min_repeat_median_ms']:.3f}–{r['max_repeat_median_ms']:.3f}|{r['peak_allocated_bytes']}|{r['head_parameters']}|{r['full_parameters']}|")
 t+=['','新B固定fold1/seed20260914、t13，cold9base/stream1newbase，每模式90测量。包含图像预处理、传输、MAE、force和任务头，排除磁盘IO、权重载入、传感采集及机器人控制；共享GPU波动保留。小任务头参数与完整系统参数分别报告。','','## 科学边界与交付','','HTT滑移标签部分依赖力规则；历史初始化与本轮checkpoint选择涉及开发validation，不能称独立盲测。没有无标记GSmini实物成功率、新future或世界模型有效性证据。本轮结果即使不改善也有效，不能为正向结果反复改协议。','','最终解释、下一阶段建议与验收状态由根目录CONCLUSIONS_ZH/DELIVERY/FINAL_STATUS/LOCAL_SYNC_PROOF给出；本报告生成通过不等于goal已完成。所有checkpoint与缓存留服务器，不自动启动下一轮。']
 (out/'SUMMARY_ZH.md').write_text('\n'.join(t)+'\n');artifact(Path(__file__));audit=dict(status='pass',runs=72,new_runs=24,reused_runs=48,all_seeds_included=True,calibration_uncertainty_diagnostic_only=True,sources=sources,outputs={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='REPORT_AUDIT.json'});(out/'REPORT_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n');print('R12 report PASS')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args())
