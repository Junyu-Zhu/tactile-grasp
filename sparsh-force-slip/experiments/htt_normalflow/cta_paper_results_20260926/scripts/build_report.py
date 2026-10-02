from pathlib import Path
import csv,json,hashlib,shutil
from collections import defaultdict
import numpy as np
P=Path(__file__).resolve().parents[1];R=P.parent
sources=[]
def read(path):
 p=R/path;sources.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()});return list(csv.DictReader(p.open()))
def num(x,k):return float(x[k])
def md(headers,rows):return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])
def link(rel,label):return f'[{label}]({R/rel})'
g1=read('round24_921_g3_final_review/tables/RQ1_GROUP_SUMMARY.csv');future=read('cta_paper_results_20260926/tables/RQ2_group_means.csv');ci=read('cta_paper_results_20260926/tables/RQ2_paired_CI.csv')
f=read('round22_921_g1_joint_frozen/formal_evaluation/force_core/METRICS.csv')
force=read('round21_final_evidence_synthesis/FORCE_EVIDENCE.csv');legacy=read('round21_final_evidence_synthesis/LEGACY_FORCE_TASK_REGRESSION.csv')
r8=read('round8_force_dynamics_event_time/results/reporting/H3_OPERATIONAL_ALL_GROUPS.csv');raw=read('round8_force_dynamics_event_time/results/reporting/RAW_RISK_ALL_GROUPS.csv')
condition=[]
for st in ['all','stable','transition','changing']:
 for v in ['hold','K-V','K-F','K-VF','F2-half']:
  rows=[r for r in f if r['role']=='validation' and r['axis']=='all' and r['stratum']==st and r['horizon']=='10' and r['variant']==v]
  dedup={}
  for r in rows:
   key=(r['fold'],r['seed']);val=(num(r,'delta_mae_n'),num(r,'absolute_future_mae_n'))
   if key in dedup:assert np.allclose(val,dedup[key],atol=1e-8)
   dedup[key]=val
  assert len(dedup)==12,(st,v,len(dedup))
  condition.append({'stratum':st,'variant':v,'runs':len(dedup),'delta_mae_n':np.mean([x[0] for x in dedup.values()]),'absolute_future_mae_n':np.mean([x[1] for x in dedup.values()])})
with (P/'tables/CONDITIONAL_H10_ALL_STRATA.csv').open('w') as h:
 w=csv.DictWriter(h,fieldnames=list(condition[0]));w.writeheader();w.writerows(condition)
def dettab(stage):
 order=['V','C','M','MB'] if stage=='G1_frozen' else ['A','B','C','D']
 a=sorted([r for r in g1 if r['stage']==stage],key=lambda r:order.index(r['group']))
 return md(['组别','pAUC','AP','实际static FPR','gross召回','BA','macro-F1'],[[r['group'],f"{num(r,'pAUC'):.4f}",f"{num(r,'AP'):.4f}",f"{num(r,'frame_static_FPR')*100:.2f}%",f"{num(r,'gross_recall')*100:.2f}%",f"{num(r,'balanced_accuracy'):.4f}",f"{num(r,'macro_f1'):.4f}"] for r in a])
futuretable=md(['方法','变化MAE：1/5/10帧（N）','未来绝对力MAE：1/5/10帧（N）'],[[v,' / '.join(f"{num(r,'delta_mae_n'):.4f}" for r in future if r['variant']==v),' / '.join(f"{num(r,'absolute_future_mae_n'):.4f}" for r in future if r['variant']==v)] for v in ['hold','ridge-K-VF','K-V','K-F','K-VF','F2-half']])
ct=md(['片段','保持','K-V','K-F','K-VF','0.5收缩'],[[st]+[f"{next(r['delta_mae_n'] for r in condition if r['stratum']==st and r['variant']==v):.4f}" for v in ['hold','K-V','K-F','K-VF','F2-half']] for st in ['all','stable','transition','changing']])
ft=md(['对照域/轮次','方法','剪切x MAE','剪切y MAE','法向MAE'],[[domain,v]+[f"{num(next(r for r in rows if r['variant']==v and r['axis']==axis),'mae_n'):.4f}" for axis in ['shear_x','shear_y','normal']] for domain,rows in [('R10 HTT滑移试次',[r for r in force if r['task']=='R10 adapted current force']),('R10 HTT历史专用力任务',legacy),('R16 HTT迁移路线',[r for r in force if r['task']=='R16 pretransfer comparison'])] for v in dict.fromkeys(r['variant'] for r in rows)])
risk=md(['R8方法','H3 AP','实际试次误告警率','事件召回','命中事件平均提前帧'],[[r['group'],f"{num(next(a for a in raw if a['group']==r['group'] and a['horizon']=='3'),'average_precision_mean'):.4f}",f"{num(r,'trial_false_alarm_rate_mean')*100:.2f}%",f"{num(r,'event_recall_mean')*100:.2f}%",f"{num(r,'mean_lead_frames_mean'):.3f}"] for r in r8])
text='''# CTA论文实验结果

更新：2026-09-26。本文档是已有实验的证据核查与论文实验材料重组，不是新训练结果。完整保留有效负向结果；新增训练0次，不新增模型推理或阈值搜索。图中英文便于用于论文，正文给出中文图注；同名PDF/SVG为矢量稿，PNG供预览。

## 1. 总结意见：可以建立的论文论证

论文可以围绕“预测力表征在触觉检测与短期状态预测中的作用及边界”展开。最稳妥的结论不是力信息普遍提升全部任务，而是：

1. **当前滑移检测**：在冻结MAE、相同HTT开发协议与固定校准规则下，特征级FiLM出现小幅平均误报—召回收益；在末两块微调的完整因子对照中，当前力辅助监督改善平均排序，但未形成跨折稳定低误报优势。
2. **近期接触状态**：轻量当前状态模型能学习部分未来力变化；优势主要存在于真实力变化片段。固定收缩改善稳定片段，却损害变化片段；额外预测力输入相对视觉的增量没有可靠跨折支持。
3. **历史补充证据**：Sparsh原域R8的等原始历史对照中，显式预测力差分提高标签级未来风险排序和部分窗口的事件召回。这是另一数据域、另一任务，不能证明HTT的未来力预测模块已带来早滑收益。

可采用的题目方向为“力辅助触觉滑移检测与轻量短期接触状态预测：方法与受控评估”。若继续用“力条件化”作标题，应说明辅助监督与条件输入是两个因素；不宜使用“可靠滑移预警”“掉落预测”或“完整世界模型”作已实现能力。

## 2. 代码、数据与网络身份审计

<!-- AUDIT_BLOCK -->

必须区分三类情况：**误用checkpoint或数据**是实现错误；**目标、归一化、历史不同却直接排名**是比较错误；**同协议下结果负向**是有效研究结果。后者不能靠删数据解决。

已知历史修正与不可混比事项：

- R6相同GRU步数不等于同原始图像历史，R7统一至t−13…t；R7力值槽位为Fn/Ft/比值，而差分含有符号XYZ，因此纯差分归因优先采用R8完整XYZ历史对照。
- 历史pAUC截断在重复FPR处的实现曾被修正。R22所有检测组用修正后的内部selection规则重训；不能只重算旧模型分数就宣称旧选择轨迹等价。
- G2 checkpoint位于R22准备的formal/A–D目录，不代表它们是G1冻结模型。身份应由配置、训练模块、摘要与checkpoint实际内容确定，不能看目录名。
- G2的47次去重编码评价＋1次原始窗口前向回退有预设selection复现门依据。极小浮点差异改变离散排序时，恢复原始计算路径而非放宽验收；未改变模型或数据。
- G1与G2同构head初始化规则一致；真正混杂包括G1对195维fit标准化、G2仅标准化force、编码器更新和训练/shuffle路径差异。G1 M与G2 D还混合辅助监督，不能拿两者均值差独立归因微调。
- R10的旧任务回退域是HTT专用历史force任务，**不是Sparsh原域**；R8/P4预测的是冻结力表征，R22/F1预测的是真实力变化，不得混用。

## 3. 任务、数据和统计口径

|数据/协议|用途|不能据此声称|
|---|---|---|
|HTT开发四折|当前力、static/gross检测、真实未来力变化|独立盲测、无标记GSmini实物泛化；其阶段标签部分依赖力规则|
|Sparsh原域连续序列|历史R6–R9标签起点风险|独立物理滑移起点、HTT未来力增强风险已验证|
|ToucHD-Force Mini路线|R16力预适配及HTT迁移|跨域力轴误差可直接合并；新数据必然提升下游|
|NormalFlow|R4未来冻结特征及相对运动辅助|真实slip标签或物理力预测|
|DeformableObjectsGrasping|可用性审计|外域性能或实物验证已完成|

最新主实验四折×三个种子20260914/15/16；四折重叠，种子不是独立物理重复。每个run先计算再等权汇总；配对区间沿用完整泄漏组重采样并保留全部种子，不对12次运行计算伪独立显著性。报告中的fold只是既有开发协议索引，不自动等同某个独立未见物体。

当前检测主要标签为static/gross，incipient不并入稳定负例。FPR=FP/(FP+TN)，gross召回=TP/(TP+FN)，BA=(1−FPR+召回)/2。pAUC为FPR∈[0,0.1]的ROC面积除0.1，不是工作点准确率。AP须结合gross高自然占比解释。下表阈值来自calibration-FPR5，原样应用validation；5%是校准目标而非验证保证。连续误告警、事件检出、延迟和左/右删失保留在完整源表，长gross段最终检出不等于提前预警。

近期预测目标为参考相对且逐轴裁剪±20 N的真实力：Δf=GT(t+h)−GT(t)，h∈{1,5,10}。预测绝对力=冻结预测当前力＋预测变化。保持基线预测变化为0；真实当前力保持只可作不可部署理想参考。误差满足e_future=e_anchor+e_delta，MSE含交叉项，MAE不能简单相加。

## 4. 当前接触力：支持域内适配，但不保证下游收益

以下均为对应轮次域内描述性均值，单位N。R10新旧对照在本轮同目标/数据内解释；R16 H/T_H是另一公平路线实验，不与R10行跨轮排名。

'''+ft+'''

![图6 当前力适配与旧任务回退](CTA论文实验结果_assets/figures/Fig06_force_adaptation_and_regression.png)

**图6图注。** 左：R10 HTT滑移试次域内力适配降低逐轴MAE。右：同一适配在HTT历史专用力任务上的回退，保留所有轴。该回退不代表已证实Sparsh原域回退。误差对应裁剪目标，不能与不同坐标/量程/划分的公开论文绝对数字直接比较。

因此可以说“域内力监督改善当前力估计”，不能由此推出“更准的力必然改善slip”。R16预适配的剪切误差略低、法向略高；其FPR5工作点H与T_H的误报/召回分别为18.54%/78.08%、18.06%/76.77%，收益伴随召回代价。

## 5. RQ1：力条件化与当前力监督的作用

### 5.1 冻结编码器：完整结构对照

V为视觉；C为普通预测力拼接；M为特征级FiLM；MB为唯一预设有界FiLM。使用同一MAE、R10预测力来源和共同历史，未删折或种子。

'''+dettab('G1_frozen')+'''

M−V平均FPR降低2.27个百分点、召回提高0.28个百分点、pAUC提高0.0029。这个方向支持限定为“本HTT开发协议下的平均收益”。但pAUC折1区间为正、折3为负，不能进一步声称跨折稳定优越。C未显示可靠的普通拼接增量；MB提高平均召回的同时，相对M提高了误报，限制调制幅度并未解决全部问题。

### 5.2 微调编码器：完整2×2因子对照

A为视觉滑移；B增加训练期当前GT力辅助监督；C增加部署预测力FiLM；D同时增加两者。所有组只微调视觉MAE末两块及规定新模块。B/D辅助头在部署移除，原R10物理force路径冻结。B不能叫“训练完全无力信息”，也不能说它部署时读取预测力。

'''+dettab('G2_finetuned')+'''

B−A的pAUC均值增加0.1042，平均gross召回提高约1.68个百分点，FPR仅下降约0.12个百分点。可写“当前力监督改变了表征学习并改善部分平均检测指标”；不能用辅助头MAE改善代替低误报主问题。D的pAUC最高，但其召回81.61%低于B的88.36%，不是全部指标上的赢家。

四项比较B−A、D−C、C−A、D−B达到工程点参考的折数分别为0/4、1/4、1/4、0/4。工程参考为FPR至少降1个百分点且召回损失不超过1个百分点；这只是点估计判据，四项均未获得严格跨折CI支持。D−C折4达到点参考，但pAUC下降；不得称为全面改善。已有辅助梯度同向证据只描述局部优化，不证明泛化机制。

![图1 误报召回权衡](CTA论文实验结果_assets/figures/Fig01_slip_operating_points.png)

**图1图注。** 每个面板为独立管线，圆点保留全部四折三种子，菱形为12次运行描述均值；灰线为calibration的5%目标。实际validation误报远高于目标，不能将灰线当验证保证。两个面板不能用于纯微调效应排名。

![图2 全部运行排序指标](CTA论文实验结果_assets/figures/Fig02_slip_pauc_all_runs.png)

**图2图注。** normalized pAUC与全部fold/seed点。点的分散同时含开发折难度与随机种子差异；不把这些点视为独立物理重复，也不因低分删除折。

![图3 配对区间](CTA论文实验结果_assets/figures/Fig03_slip_paired_intervals.png)

**图3图注。** 轮内比较的完整泄漏组配对95%区间；点为bootstrap均值，不必与原始run差值完全一致。FPR/召回差单位为百分点，召回−1 pp虚线仅为原工程容差。相邻帧和重叠折不池化；图中包含不利区间。

### 5.3 可以限定前提，但不能删掉不利结果

|限定方式|可以写的结论|不能做的扩展|
|---|---|---|
|冻结MAE＋既定HTT开发角色＋calibration-FPR5|FiLM有小幅平均误报—召回收益|不称跨折稳定或新物体部署可靠|
|相同微调管线＋当前力辅助监督|B相对A平均排序、召回改善|不称5%低误报目标已实现|
|事后观察某折达到工程参考|明确标“事后、单折点估计”，同时列所有折|不能将其改写为预注册适用物体集合|
|真实力变化片段条件|按原物理门槛分层报告未来变化能力|不能推理时偷看未来GT后选择模型|
|确认的损坏/错配/协议违规样本|可登记排除理由、分母及统一重评|不能只排掉正确产生的高误报试次|

这类写法缩小的是**结论适用范围**，不是隐匿观察。论文主文可以集中展示机制最清楚的受控对照，其他候选放完整补充材料；但正文仍须明确主要负向发现。

## 6. RQ2：近期接触力变化预测

### 6.1 全窗口、全部输入对照

K-V只用当前基础视觉表示；K-F只用当前预测力；K-VF同时用二者。基础状态本身有t与t−5图像依赖，不是单张原图。当前状态重复两次进入轻量GRU，不增加真实历史。输出直接联合预测多窗口变化，不是递归世界模型。三组未来绝对力均加同一冻结预测当前力，故K-V仅在变化模块中无力输入。

'''+futuretable+'''

K-VF相对保持在1/5/10帧的变化MAE均值分别下降约2.2%/4.9%/6.8%；同信息线性基线未达到K-VF的平均变化误差。这支持有限的非零未来变化预测能力。K-VF相对K-V的增量很小，h10四折区间均跨零；K-F单独输入未超过保持。不能声称力输入是预测改善的必要原因。

未来绝对力要单独解释：K-VF在h1为0.9688 N，略差于保持0.9649 N；在h10变化误差略小于K-V，绝对力误差反而略大。这是当前锚点误差与变化预测误差共同作用的结果。

![图4 未来变化与绝对力](CTA论文实验结果_assets/figures/Fig04_future_horizons.png)

**图4图注。** 六种方法的全部窗口与12个运行均值；左为真实力变化MAE，右为部署可得锚点形成的未来绝对力MAE。两图不得混称同一误差，也不以跨窗口平均掩盖h1反例。窗口以帧计，无毫秒或物理提前量保证。

### 6.2 条件性证据：变化片段能力与稳定片段代价

原协议按h10三轴变化最大绝对值划分：≤0.25 N为stable，≥1 N为changing，中间为transition。以下保留全部片段，而非删除稳定片段。单位为h10真实变化MAE（N），每格四折三种子等权均值。

'''+ct+'''

![图5 条件性配对证据](CTA论文实验结果_assets/figures/Fig05_future_stratified_intervals.png)

**图5图注。** h10所有预定义片段、所有折的配对区间。K-VF相对保持在changing片段四折区间均为负，在stable片段均为正；因此可支持“变化片段有局部预测收益，同时稳定片段退化”。F2收缩相对K-VF在stable四折改善、changing四折恶化。这里分层用未来GT，仅为离线解释，不能声称部署时已能识别这些片段并无代价切换策略。力稳定不等于static滑移标签。

这是论文可以强化的真实结论：模型在变化发生的片段确实学习到部分变化信息，但并未同时解决稳定保持和动态响应。额外预测力输入收益仍有限，不能通过只展示changing中最好的折来宣称全面力增益。

## 7. 必须补回的历史正向证据：原域future-slip

“F3未触发”只说明最新HTT方案没有完成未来状态增强风险训练，**不是所有历史future-slip都未做或无收益**。R7–R9已在Sparsh原域重复开发outer开展数据集标签起点预测。以下R8 H3保留全部八个候选、三种子均值；工作点由calibration试次误报≤10%规则确定，实际outer误报逐组报告。

'''+risk+'''

![图7 原域标签风险](CTA论文实验结果_assets/figures/Fig07_historical_source_risk.png)

**图7图注。** R8全部候选的H3排序、实际试次误告警和事件召回。误差棒为三个种子的样本标准差，不是独立试次CI；不能与HTT当前检测的帧FPR混比。配对CI见原始`PAIRED_KEY_COMPARISONS.csv`，其单位为完整泄漏组。

更干净的归因是同完整XYZ历史下C_xyz_delta−B_xyz：H3 AP增加0.3231，95%配对区间[0.2682,0.3645]；事件召回增加0.1882，[0.1293,0.2449]。试次误报差−0.0097的区间[−0.0395,0.0193]跨零，不能说低误报改善已被同样确认。H1事件召回差区间跨零；H5召回改善伴随误报点估计上升，不能仅选H3后宣称所有窗口都改善。

可以在论文补充中写：“在Sparsh原域、等原始历史的开发对照下，显式力差分表征提高了标签级短期风险排序和部分窗口事件召回。”同时必须注明：

- 这不是HTT F3，不是“先预测真实未来力，再用该未来力改善slip”链路验证。
- 起点来自数据集标签，不是独立物理仪器；没有动作干预、无标记GSmini实物或外部盲测。
- R8的融合hazard候选是多模块整体方案；容量和损失不同的对照不可归因到单一新算子。其H3均值提前约1.885帧不等于可靠提前3帧，H5也不等于提前5帧。
- R8状态辅助未超过保持/线性状态基线，R9扰动显示缺失历史可能增加误报。早期正向结果不能掩盖这些负向发现。

## 8. 其他结构尝试与历史实验总览

|轮次/阶段|做过的内容|结论与在论文中的位置|
|---|---|---|
|早期/R1–R2|原模型、数据、输入和标签审计|方法来源与问题定位；本轮不重新作全部实体审计|
|R3|冻结MAE，新slip头与旧头继续训练|BA@0.5：旧迁移0.4644、新头0.8822、旧头适配0.8586；支持域内头适配，非力融合归因|
|R4|MAE/DINO/I-JEPA，几何消融，NormalFlow与早期future|BA@0.5 0.8822/0.8745/0.8673；NormalFlow GRU只在H3略优线性，不作slip真值证据|
|R5–R6|force适配、力融合、原域/HTT future及严格支持审计|力融合低误报收益常伴召回损失；R6原始图像历史不等，差分归因由后续修正|
|R7–R9|等历史、完整XYZ、差分、hazard、时序融合与扰动|原域future有正向开发证据；HTT当前检测仍不稳定；参考上节完整对照|
|R10|HTT滑移试次真实力监督适配|域内当前力改善，旧HTT力任务回退；未自动转化稳定slip收益|
|R11–R12|试次平衡、保留类别权重、力logit残差|仍无可靠跨折低误报增益；残差不优于完整视觉容量对照|
|R13|试次级阈值校准与连续确认|calibration约束可满足，validation迁移失败；更严格告警常损失召回和增加延迟|
|R14|视觉/拼接/双路绝对未来力|平均MAE V1.6715、拼接1.2311、双路1.3174 N；保持1.1949 N。拼接优于视觉但未胜保持|
|R15|当前锚定残差、当前与九步历史|未来MAE1.1656/1.1653 N稍胜保持1.1949；真实变化误差0.6065/0.6128反而高于保持0.5935；不可只写动力学改善|
|R16|ToucHD力预适配→HTT|部分剪切误差改善，法向/召回代价保留；各路线需自己的保持基线|
|R17|共享时序/未来力辅助slip|共享J未来MAE1.6837高于独立F1.3114 N，检测亦退化；负迁移证据|
|R18–R19|冻结FiLM与有限微调|旧端点/选择轨迹结果仅历史背景，不与G1/G2直接排名|
|R20|K当前、D历史、视觉/力状态转移|单折K的h10变化MAE0.9139，保持0.9702；稳定片段退化，不能写四折世界模型验证|
|R21|前阶段证据收尾|厘清力回退域、比较与Q2/Q4边界，无新增训练|
|R22/G1|完整冻结检测与K输入/收缩验证|84次训练；FiLM平均局部收益，未来变化收益有片段依赖|
|R23/G2|48次当前GT辅助监督×FiLM微调|平均排序改善未形成跨折可靠低误报优势|
|R24/G3|综合证据与独审|3项有界报告修订闭合；不代表全部研究假设成立|

历史数值仅供其轮内受控对照解释。R14/R15/R17目标、归一化、误差汇总和上游可能不同，不能从表中选最低MAE拼成“最终模型”。R16采用ToucHD域内坐标，不能将两域力误差放一起求平均。

## 9. 论文组织建议与可直接使用的结论

建议正文采用：①身份/数据/角色与任务定义；②力估计作为支撑功能；③冻结力条件化与微调辅助监督两个独立消融；④真实未来力变化及简单基线；⑤稳定/变化片段和校准迁移；⑥限制。原域future单列历史补充实验；如果论文篇幅有限，R3–R20过程性结构探索可放补充材料，但保留负向概述与完整来源。

推荐主图为图1、图3、图4、图5；图2是全部fold/seed透明性补图，图6是force支撑证据，图7是独立数据域风险探索补图。每张图都同时交付PDF/SVG/PNG，不应只裁出有利曲线或面板。图3/5在双栏论文中适合通栏，避免缩至单栏导致CI标签不可读。

可用实验结论段：

> 在HTT开发协议下，特征级力条件化在冻结视觉表示上表现出小幅平均误报—召回收益，当前力辅助监督则改善了有限微调模型的平均排序表现。然而，逐折配对结果和固定校准工作点显示，这些收益尚未转化为跨折稳定的低误报优势。轻量当前状态预测器能够学习部分短期力变化，其收益主要位于变化片段，并伴随稳定片段的误差代价；预测力输入相对视觉表示的额外收益仍不确定。另在Sparsh原域的等历史标签风险任务中，显式力差分获得排序及部分窗口事件召回的开发证据。这些结果说明力监督表征具有任务相关价值，同时揭示了校准迁移、片段权衡和独立验证不足等限制。

不能写“加入力稳定优于所有视觉方法”“达到5%部署误报率”“已经可靠预测掉落”“用未来力证明世界模型有效”。FiLM、GRU、残差和hazard均不因使用而成为原创算子；可讨论的贡献是受控的力条件化/监督设计、任务接口与系统性证据，而非算子发明。

## 10. 文件与复现

正文来自完整CSV和固定汇总；条件性统计是本次事后整理，不是新预注册发现。`CTA论文实验结果_assets/tables/`保留全部96个当前检测run、216条未来力窗口run、既有CI、所有条件片段、R8完整候选与历史force表。未丢弃有效不利记录，排除列表为空（如审计发现技术问题则在下列审计中登记）。

图源：`CTA论文实验结果_assets/scripts/build_figures.py`；正文生成：`build_report.py`。仓库工作目录为`experiments/htt_normalflow/cta_paper_results_20260926`；服务器审计位于其中`audit/`。重新生成使用本机`/home/zjy/miniconda3/envs/sparsh/bin/python`，依赖numpy/matplotlib和标准库，不下载新包。`FIGURE_MANIFEST.json`与`REPORT_SOURCES.json`记录来源哈希，最终交付清单记录全部文件SHA。

主要证据入口：

'''
refs=[('round24_921_g3_final_review/SUMMARY_ZH.md','R24综合结果'),('round24_921_g3_final_review/INDEPENDENT_RECHECK.json','R24独立复核'),('round22_921_g1_joint_frozen/ROOT_G1_ACCEPTANCE.json','G1验收'),('round23_921_g2_force_aux_finetune/ROOT_G2_ACCEPTANCE.json','G2验收'),('round21_final_evidence_synthesis/SUMMARY_ZH.md','R21历史证据综合'),('round8_force_dynamics_event_time/results/reporting/SUMMARY_ZH.md','R8原域完整风险对照'),('round9_htt_temporal_force_fusion/results/reporting/SUMMARY_ZH.md','R9扰动与HTT检测'),('round16_touchd_force_transfer/results/reporting/SUMMARY_ZH.md','R16预适配'),('round13_trial_level_alarm_calibration/results/reporting/SUMMARY_ZH.md','R13校准迁移')]
text+='\n'.join('- '+link(x,y) for x,y in refs)+'\n'
audit=P/'audit/AUDIT_SUMMARY_ZH.md'
text=text.replace('<!-- AUDIT_BLOCK -->',audit.read_text() if audit.exists() else '服务器实体审计进行中；最终结论待审计结果写入，不能据此草稿宣称全部代码正确。')
text=text.replace('CTA论文实验结果_assets/', '')
(P/'CTA论文实验结果.md').write_text(text)
refs += [('round3_adaptation/SUMMARY_ZH.md','R3'),('round4_comprehensive/SUMMARY_ZH.md','R4'),('round5_force_conditioned_slip/formal_delivery/SUMMARY_ZH.md','R5'),('round6_future_validation/SUMMARY_ZH.md','R6'),('round7_temporal_fairness_event_warning/SUMMARY_ZH.md','R7'),('round11_stable_negative_force_residual/results/reporting/SUMMARY_ZH.md','R11'),('round12_class_preserving_trial_balance/results/reporting/SUMMARY_ZH.md','R12'),('round14_htt_future_force_dual/results/final_report/SUMMARY_ZH.md','R14'),('round15_htt_future_force_residual_history/results/final_report/SUMMARY_ZH.md','R15'),('round17_htt_shared_temporal_multitask/results/final_report/SUMMARY_ZH.md','R17')]
for rel,_ in refs:
 path=R/rel;sources.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
(P/'REPORT_SOURCES.json').write_text(json.dumps({'inputs':sources,'exclusions':[],'note':'Historical reported figures attributed, not all re-inferred. Current main tables complete. No unfavorable valid observations removed.'},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'report':str(P/'CTA论文实验结果.md'),'bytes':len(text.encode()),'conditional_rows':len(condition),'audit_integrated':audit.exists()},ensure_ascii=False))
