# IsaacSim 力-滑移闭环抓取执行计划（整理版）

本目录把前一版 RALPLAN 计划改写为中文，并按**实际执行顺序**拆成阶段文件。

## 你应该先做哪一个？

**先做：`1-阶段0-盘点与接口冻结.md`。**

原因：后续所有仿真、校准、闭环和 RL 都依赖同一套 checkpoint、输入张量格式、IsaacSim 数据根目录、输出报告目录和日志字段。如果这些没有先冻结，后面会出现“跑通了但不可复现 / 指标不可比 / 标签不可用”的问题。

## 文件编号规则

- `0a/0b`：先读的背景约束文件，不是直接执行任务。
- `1, 2, 3...`：推荐执行顺序，数字越小越先做。
- 同一数字下的 `a/b`：可以并行推进，但必须满足各自退出条件后才能进入下一组。
- 每个阶段都要产出 `json/csv + md`，并明确标注结果是 `SMOKE_ONLY`、`PROXY_ONLY`、`MVP_SMOKE`、`EVAL_READY` 还是 `METRIC_VALID`。

## 高效执行顺序

| 顺序 | 文件 | 是否可并行 | 目标 |
| --- | --- | --- | --- |
| 0a | `0a-PRD-研究目标与边界.md` | 先读 | 明确论文/系统目标、非目标和验收标准 |
| 0b | `0b-测试规范-门控验收.md` | 先读 | 明确每阶段怎么验证、哪些报告必须生成 |
| 1 | `1-阶段0-盘点与接口冻结.md` | 必须最先执行 | 冻结 checkpoint、输入输出 schema、artifact 目录 |
| 2a | `2a-阶段1-模型前向烟测.md` | 可与 2b 并行 | 先确认 IsaacSim tactile frame 能通过 force/slip/future 模型前向运行 |
| 2b | `2b-阶段2-标签有效性v2.md` | 可与 2a 并行 | 把 slip/force 标签从 proxy 变成可度量的 sim-valid 标签 |
| 3 | `3-阶段3-风险校准与控制信号选择.md` | 依赖 2a+2b | 选择控制器真正使用的 risk score，而不是盲用 raw future risk |
| 4 | `4-阶段4-确定性闭环早期纠偏.md` | 依赖 3 | 先用规则/MPC/CEM 证明 tactile perception 能带来 early correction |
| 5a | `5a-阶段5-RL改进层.md` | 可与 5b 并行 | 在确定性 baseline 通过后，用 RL 做增益层 |
| 5b | `5b-阶段6-实物小样本验证准备.md` | 可与 5a 并行 | 准备真实 UR5/Robotiq/GSmini 小样本验证协议，先不贸然上硬件 |
| 6 | `6-阶段7-JFR论文证据包.md` | 可提前建模板，最终依赖 4/5 | 汇总 JFR 投稿所需图表、消融、失败案例和限制 |

## 最重要的顺序约束

1. **不能先做 RL。** 只有 `4-阶段4-确定性闭环早期纠偏.md` 通过后，才进入 `5a`。
2. **不能用 proxy slip label 报 metric-valid 指标。** `2b` 没通过前，`2a` 的结果只能算 smoke。
3. **不能直接信任 raw future risk。** 实物结果已经说明 raw future risk 可能饱和，必须在 `3` 中比较 `p_slip_current`、future risk、force threshold 和 gated score。
4. **每次 IsaacSim/IsaacLab 测试结束后**，默认清理当前用户下残留的 Isaac Sim 后台进程，除非明确要求保留。

## 一句话路线

`盘点接口 -> 模型前向烟测 + 标签有效性并行 -> 风险校准 -> 确定性早期纠偏 -> RL 增益 + 实物准备并行 -> JFR 证据包`
