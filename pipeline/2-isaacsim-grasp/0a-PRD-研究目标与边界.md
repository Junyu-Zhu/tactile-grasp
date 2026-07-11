# 0a — PRD：研究目标、边界与最终产出

## 1. 总目标

基于已经完成的 `sparsh-force-slip` 离线训练结果，在 IsaacSim/IsaacLab 中构建一个**触觉感知到闭环抓取控制**的研究流水线：

1. 让现有 force/slip/future-instability 模型能在 IsaacSim tactile frames 上前向运行；
2. 确认仿真中的 force/slip 标签是否具备 metric-valid 价值；
3. 把模型输出变成控制器可用的风险信号，例如：
   - `p_slip_current`
   - `p_instability_H1/H3/H5`
   - `p_slip_current * p_instability_H1`
   - force threshold / `Ft/(Fn+eps)`
4. 先做确定性早期纠偏控制，再做 RL 增益；
5. 最终形成面向 Journal of Field Robotics 的系统型论文证据。

## 2. 核心论文主张

建议主张不是“又训练了一个更高分的 slip classifier”，而是：

> 已有触觉 force/slip/future-risk 模型可以被组织成一个 perception-control bridge，用于在滑移发生前触发抓取力/姿态/动作调整，从而提升闭环抓取鲁棒性。

JFR 更看重系统可靠性、真实/仿真闭环效果、失败分析和部署边界。因此主线应是：

`触觉风险预测 -> 控制信号/安全门控 -> 早期纠偏 -> 仿真扰动泛化 -> 小规模真实验证`

## 3. 非目标

本阶段不要做：

- 端到端 VLA；
- 直接声明当前 force-slip 结果已经证明 IsaacSim 闭环抓取成功；
- 使用 world XY proxy、success label、contact onset 或 release/drop 当作 slip metric ground truth；
- 直接把 `q`、`tau`、`Ft/Fn` 写成真实物理摩擦系数；
- 在确定性 early-correction baseline 之前启动 RL；
- 只凭离线分类指标写 JFR 主张。

## 4. 已有证据如何使用

### 支持启动 IsaacSim 验证的证据

`sparsh-force-slip/reports/training_results.md` 和 `phase7_summary.md` 显示：

- MAE + decoupled multitask 是当前较稳路线；
- force RMSE 和 slip F1 支撑继续做模型前向验证；
- full dynamics / full+q future heads 在 H1/H3/H5 上较强；
- runtime 足够轻，可以进入 simulation-side experiments。

### 不能直接当作闭环成功的证据

之前 sim bridge 报告中：

- `FORMAT_BRIDGE_PASS: PASS`
- `FORCE_LABEL_VALIDITY: sim-valid`
- `SLIP_LABEL_VALIDITY: proxy`
- `METRIC_VALIDITY: limited-proxy`
- 当时没有用 checkpoint 跑真实 model-forward metrics

所以现有结果只能说明“值得开始 IsaacSim 验证”，不能说明“已经能闭环抓取”。

### 实物结果给出的约束

实物测试显示：

- `p_slip_current` 比 raw future risk 更稳定；
- raw future risk 在稳定窗口可能过高/饱和；
- gated score，例如 `p_slip_current * p_instability_H`，更适合作为部署候选。

因此后续控制不能直接盲用 raw future head。

## 5. 验收标准总览

完成本计划至少应有：

1. `inventory_report.{json,md}`：checkpoint、schema、artifact root 已冻结；
2. `model_forward_smoke_report.{json,md}`：IsaacSim 数据可前向跑通；
3. `label_validity_report.{json,md}`：force/slip 标签有效性明确；
4. `risk_calibration_report.{json,md}`：选出控制器 risk score；
5. `closed_loop_baseline_report.{json,md}`：确定性 early-correction baseline 有对比结果；
6. `rl_report.{json,md}`：如尝试 RL，必须证明相对确定性 baseline 有增益或诚实报告负结果；
7. `real_smoke_readiness_report.md`：真实 UR5/Robotiq/GSmini 小样本验证协议准备完成；
8. JFR 证据清单：主表、消融、失败案例、timeline 图、限制说明。
