# Multitask Training Phase Analysis

> 文件名按用户指定保留为 `multitask_training_pahse.md`。本文整理 Phase2-B shared multitask decoder 当前训练结果、force 退化原因分析、consistency decoder 是否可能改善，以及后续尝试优先级。

## 1. 当前背景

Phase2-B 目标是验证一个共享 downstream 表示能否替代 Phase1/Phase2-A 的 separate baseline。

当前 B 模型形式：

```text
frozen Sparsh encoder
  ↓
shared attentive pooler
  ↓
shared MLP trunk
  ├── force head: Fx, Fy, Fz
  └── slip head: no-slip / slip logits
```

训练 loss：

```text
L = L_force + lambda_slip * L_slip
```

其中：

- `L_force`: Smooth L1 loss for absolute `Fx,Fy,Fz` force regression。
- `L_slip`: Cross entropy loss for slip/no-slip classification。
- 当前 `lambda_slip = 1.0`。
- Sparsh encoder frozen，只训练 downstream multitask decoder。

当前 run：

```text
phase2_b_gsmini_20260513_163448
```

报告位置：

```text
/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase2/phase2_b_gsmini_20260513_163448/phase2_b_diagnostic_report.md
```

## 2. 当前训练结果概览

Phase2-B 使用 validation-selected `best_f1.pth` 作为 B gate 判断 checkpoint。

| encoder | A force RMSE mean N | B force RMSE mean N | force RMSE 变化 | A slip F1 | B slip F1 | gate |
|---|---:|---:|---:|---:|---:|---|
| DINOv2 | 0.0436 | 0.0566 | +29.85% | 0.9477 | 0.9693 | hard fail |
| MAE | 0.0322 | 0.0364 | +12.88% | 0.9731 | 0.9784 | hard fail |

结论：

- B 的 slip F1 确实提升了。
- 但 force RMSE 相比 A 增加超过 Phase2 gate 的 10% hard fail 阈值。
- 因此当前 B 不能作为正式优于 A 的方法。
- 若直接进入 C consistency decoder，C 只能是 diagnostic-only，不能作为 formal improvement claim。

## 3. Force 退化的具体表现

### 3.1 DINOv2

| axis | A RMSE N | B RMSE N | 变化 |
|---|---:|---:|---|
| Fx | 0.0437 | 0.0590 | 变差 |
| Fy | 0.0521 | 0.0618 | 变差 |
| Fz | 0.0350 | 0.0491 | 变差 |
| mean | 0.0436 | 0.0566 | +29.85% |

DINOv2 是三个轴都退化，说明 shared multitask 对 force regression 的负迁移比较明显。

### 3.2 MAE

| axis | A RMSE N | B RMSE N | 变化 |
|---|---:|---:|---|
| Fx | 0.0369 | 0.0402 | 略差 |
| Fy | 0.0345 | 0.0333 | 略好 |
| Fz | 0.0253 | 0.0357 | 明显变差 |
| mean | 0.0322 | 0.0364 | +12.88% |

MAE 的主要问题集中在 `Fz`，即 normal force 方向退化导致 gate hard fail。

## 4. 为什么 multitask 的 force 可能不如 separate baseline

### 4.1 Shared trunk 更偏向 slip classification

当前 B 是：

```text
shared pooler/trunk
  ├── force head
  └── slip head
```

force estimation 和 slip detection 对表征的需求不同：

- force estimation 需要连续、精细的形变幅值和轴向标定，尤其是 `Fz`。
- slip detection 更关心是否发生接触状态或剪切变化的分类边界。

shared trunk 可能学到更有利于 slip 分类的特征，但损失了 force regression 对连续幅值和轴向语义的精度。

### 4.2 Loss 没有做多任务平衡

当前 loss 是：

```text
L = L_force + 1.0 * L_slip
```

但 Smooth L1 force loss 和 Cross Entropy slip loss 的数值尺度、梯度尺度不一定匹配。可能出现：

- slip classification 对 shared representation 的梯度影响更强；
- force head 被迫使用一个更适合分类而不是回归的 trunk；
- 最终表现为 slip F1 提升，但 force RMSE 上升。

### 4.3 Checkpoint 选择偏向 slip F1

当前 gate 使用 `best_f1.pth`，即 slip F1 最优的 checkpoint。

这个选择对 slip detection 有利，但不一定对 force estimation 最优。MAE 尤其明显：

- `best_f1.pth` force RMSE mean N: 0.0364，触发 hard fail。
- `epoch-0051.pth` final force RMSE mean N: 0.0343，更接近 warning band。

这说明 MAE 存在 slip 最优和 force 最优 checkpoint 不完全一致的问题。

### 4.4 B force head 与 Sparsh 原 force decoder 不完全等价

Sparsh separate force baseline 是单任务 decoder，force path 独占下游表示。

Phase2-B 中 force head 和 slip head 共用 pooler/trunk。即使参数量不一定更少，force 也不再独占表征，因此更容易被 slip 任务干扰。

### 4.5 当前问题不是训练步数太少

W&B 上看到的几十个 step 是因为脚本按 epoch 级别 log，而不是 batch-level log。

实际训练量：

| encoder | epochs | 每 epoch train batches | 总 train batches / optimizer steps |
|---|---:|---:|---:|
| DINOv2 B | 51 | 686 | 34,986 |
| MAE B | 51 | 686 | 34,986 |

因此 force RMSE 不满足 gate 不应归因于只训练了几十步。

## 5. Consistency decoder 是否可能改善结果

### 5.1 可能改善的部分

Phase2-C consistency decoder 的目标是让：

```text
Ft/Fn force ratio
```

和：

```text
slip probability
```

更加一致。

因此它可能改善：

- contradiction rate；
- monotonic calibration error；
- high `Ft/Fn` 区域 slip recall；
- low `Ft/Fn` 区域 false alarm。

### 5.2 不太可能直接修复 force RMSE

Phase2-C 计划中的第一版 consistency loss 是：

```text
q = sigmoid(alpha * (Ft_pred / (Fn_pred + eps) - tau))
L_cons = BCE(p_slip, q.detach())
```

关键点是：

```text
q.detach()
```

这意味着 consistency loss 主要约束 slip head，而不是直接通过 consistency loss 更新 force prediction。force RMSE 仍主要依赖 `L_force`。

所以如果 B 本身 force regression 已经退化，C 更可能让 slip probability 对当前 force ratio 更一致，而不是让 `Fx,Fy,Fz` 更准确。

### 5.3 当前建议

当前 B 已经触发 hard fail，因此：

```text
C 可以跑，但只能作为 diagnostic-only。
```

不建议直接把当前 B 接到正式 C 并声称方法优于 A。

## 6. 后续尝试优先级

### 优先级 1：先做 checkpoint selection 分析，不重新训练

当前报告按 `best_f1.pth` 选 B checkpoint。建议新增 multi-objective selector：

```text
先筛选 force RMSE increase <= 10%
再从满足条件的 checkpoint 中选 slip F1 最高者
```

或定义：

```text
score = slip_f1 - k * force_rmse_mean_N
```

推荐先检查所有已保存 checkpoint：

```text
epoch-0005, epoch-0010, ..., epoch-0050, epoch-0051, best_f1
```

目的：

- 不重训，最快验证是否只是 checkpoint 选择导致 gate fail。
- MAE 最可能通过该方式从 hard fail 变为 warning/pass。
- DINOv2 即使仍失败，也能确认退化是否稳定。

### 优先级 2：调整 multitask loss 权重

当前：

```text
L = L_force + 1.0 * L_slip
```

建议尝试小网格：

| run | lambda_force | lambda_slip | 目的 |
|---|---:|---:|---|
| B-w1 | 2 | 1 | 提高 force 权重 |
| B-w2 | 5 | 1 | 更强保护 force |
| B-w3 | 1 | 0.5 | 降低 slip 干扰 |
| B-w4 | 1 | 0.2 | 更明显降低 slip 梯度 |

目标：

- force RMSE increase 控制在 <=10%；
- slip F1 不比 A 下降超过 2 个百分点；
- 若满足，则可以重新考虑正式进入 C。

### 优先级 3：改成 partially shared decoder

当前完全共享 trunk：

```text
shared pooler/trunk
  ├── force head
  └── slip head
```

建议改成 partially shared：

```text
shared pooler
  ├── force-specific trunk → force head
  └── slip-specific trunk  → slip head
```

或：

```text
shared shallow trunk
  ├── force-specific MLP
  └── slip-specific MLP
```

目的：减少 slip classification 对 force regression 的负迁移。

### 优先级 4：force-pretrained B 初始化

用 Phase1 force baseline 初始化 B 的 force path：

```text
force baseline pooler/probe → B force path
```

可尝试两种策略：

1. 冻结 force path，只训练 slip head；
2. force path 低学习率微调，slip head 较高学习率训练。

目的：保住 force 轴向标定和幅值精度，再引入 slip 任务。

### 优先级 5：再做 C diagnostic

如果 B 仍未过 gate，但想验证 consistency 思路，可以跑 C diagnostic-only。

C diagnostic 重点看：

- contradiction rate 是否下降；
- monotonic calibration error 是否下降；
- high `Ft/Fn` slip recall 是否保持；
- low `Ft/Fn` false alarm 是否不上升。

但在 B hard fail 前提下，不能把 C 写成 formal improvement。

### 优先级 6：检查 force scale / axis / output activation 细节

建议再做一次 B-specific sanity check：

- B 和 A 是否使用同一 `force_scale`；
- `Fx,Fy,Fz` 是否都恢复到 Newton 后再比较；
- `Fz` 符号、裁剪、normal force 语义是否一致；
- B force output 的 `tanh` 是否限制过强；
- MAE 的 Fz 退化是否和 output activation 或 shared trunk 有关。

尤其 MAE 问题集中在 `Fz`，值得单独分析。

## 7. 推荐执行顺序

建议按以下顺序推进：

1. **做 checkpoint selector 分析**：不重训，检查是否存在满足 force gate 的 B checkpoint。
2. **做 loss-weight 小网格**：如果 selector 不能过 gate，再重训少量配置。
3. **做 partially shared decoder**：如果 loss reweight 仍失败，再改结构。
4. **B 过 gate 后再正式跑 C**。
5. **如果 B 不过 gate但想验证物理一致性，C 只能 diagnostic-only**。

## 8. 总结

当前 Phase2-B 的主要结论是：

```text
shared multitask decoder 提升了 slip F1，但牺牲了 force regression 精度。
```

这更像是多任务共享表示带来的负迁移，而不是训练步数不足。

Consistency decoder 可能改善 force-slip 一致性指标，但不太可能直接修复 force RMSE。因此推荐先从 checkpoint selection、loss reweight 和 partially shared decoder 入手。
