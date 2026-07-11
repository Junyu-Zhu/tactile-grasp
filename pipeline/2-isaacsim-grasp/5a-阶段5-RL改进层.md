# 5a — 阶段5：RL 改进层（可与 5b 并行）

## 前置条件

`4-阶段4-确定性闭环早期纠偏.md` 已通过。若确定性 baseline 未通过，不要启动 RL。

## 目标

让 RL 在已有 perception-control bridge 和确定性 baseline 之上做增益，而不是从 raw tactile pixels 端到端学习。

## 推荐 RL 形式

优先级从高到低：

1. residual RL：在确定性 controller action 上学习小残差；
2. offline/model-based RL：利用已有 trial 数据减少在线试错；
3. constrained RL：加入 over-force 和安全限制；
4. 不建议第一版直接 raw tactile end-to-end RL。

## Observation 建议

优先使用 bridge features：

- `p_slip_current`
- selected/gated risk score
- `p_instability_H1/H3/H5`
- `Fn/Ft/Fmag/Ft_over_Fn`
- gripper width
- recent action history
- object/contact metadata，如可用

## Reward 建议

必须 bounded，并包含：

- 成功抓取/无 drop 正奖励；
- slip/drop 惩罚；
- over-force 惩罚；
- action chatter 惩罚；
- early intervention 成功奖励；
- 不鼓励无限增大夹持力。

## 对比实验

至少对比：

1. deterministic baseline；
2. RL without future-risk features；
3. RL with current slip only；
4. RL with full bridge features；
5. residual RL vs direct RL。

## 产出文件

- `rl_report.json`
- `rl_report.md`
- training curves
- policy variance across seeds
- ablation tables

## 通过条件

RL 必须相对 deterministic baseline 提升至少一个关键指标，同时不显著恶化 safety：

- drop/slip reduction；
- early intervention success；
- lead time；
- over-force；
- action smoothness；
- sample efficiency。

如果 RL 没有提升，也可以作为 negative result，但必须诚实报告。

## 完成后做什么

RL 结果进入：

- `6-阶段7-JFR论文证据包.md`
