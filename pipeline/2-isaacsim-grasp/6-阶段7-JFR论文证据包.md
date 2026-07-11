# 6 — 阶段7：Journal of Field Robotics 论文证据包

## 目标

把前面阶段的技术报告整理成 JFR 投稿需要的系统证据，而不是只展示离线模型指标。

## 论文建议主线

> 一个触觉 force-slip-future 风险模型如何被校准并嵌入闭环抓取控制，在扰动和材料变化下实现早期纠偏；RL 是在已验证 bridge 上的增益层，而不是主贡献的唯一来源。

## 必须准备的表格

1. 模型前向与 runtime 表；
2. 标签有效性与数据规模表；
3. risk score calibration 表；
4. 确定性 controller 对比表；
5. RL ablation 表，如 RL 被尝试；
6. small real validation 表，如真实 smoke 完成；
7. failure-mode table。

## 必须准备的图

1. 系统框图：SPARSH tactile input -> force/slip/future risk -> risk gate -> controller/RL；
2. score-to-action timeline：warning、intervention、slip onset、drop/recovery；
3. risk calibration curves；
4. controller comparison plots；
5. sim perturbation setup；
6. representative failure cases；
7. sim-to-real score shift，如真实数据可用。

## 必须写清楚的限制

- 当前 force-slip 模型只是进入闭环系统的一部分；
- proxy label 不能支撑 metric-valid claim；
- raw future risk 在实物上可能饱和；
- RL 只有在 deterministic bridge 之后才有意义；
- 小规模真实验证如果样本量不足，只能作为 smoke/feasibility，不作为强统计结论。

## 产出文件

- `paper_evidence_checklist.md`
- `figures_to_generate.md`
- `tables_to_generate.md`
- `failure_cases.md`
- JFR contribution/limitation draft

## 通过条件

- 每个论文 claim 都能追溯到某个阶段报告；
- 没有把 smoke/proxy 结果写成 metric-valid；
- main contribution、ablation、failure analysis、limitations 都已列出；
- CICAI 工作与 JFR 扩展的关系清晰：CICAI 是感知模型基础，JFR 是闭环系统扩展。

## 最终推荐投稿叙事

1. Existing tactile force-slip-future perception model；
2. IsaacSim label-valid and calibrated deployment bridge；
3. Early-correction deterministic closed-loop proof；
4. RL residual improvement；
5. Small real-robot feasibility；
6. Failure analysis and deployment boundaries。
