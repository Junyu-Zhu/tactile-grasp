# Pipeline 1: Joint Force/Slip Decoder with Force-Slip Consistency

## 目标

设计一个基于 Sparsh DINO/MAE encoder 的联合 force/slip 解码器，同时预测接触力与滑移状态，并通过 force-slip consistency 或 friction-inspired loss 提高力估计与滑移预测之间的物理一致性。

该 pipeline 暂不包含闭环控制验证，重点放在离线 perception-level 任务：

- force estimation；
- slip detection；
- force-slip consistency；
- DINO 与 MAE 表征对比；
- frozen encoder 与 partial finetuning 对比。

## 代码组织与同步执行约定

后续所有 force-slip 阶段相关的实现代码、配置、训练入口、评估脚本、split/manifest 生成脚本和运行说明，统一放在：

```text
服务器项目根目录: /home/zjy/document
服务器仓库: /home/zjy/document/tactile_grasp
force-slip 代码目录: /home/zjy/document/tactile_grasp/sparsh-force-slip
工作分支: sparsh-force-slip
```

`pipeline/1-force-slip-decoder/` 只保存 pipeline 文档和执行约束，不放训练实现代码。

代码修改与训练必须遵循 **服务器修改 → 服务器运行 → phase 完成后整体 commit**：

```bash
ssh zjy-4090
cd /home/zjy/document/tactile_grasp
git switch sparsh-force-slip  # 若分支尚不存在，由用户/维护者在服务器上创建
cd sparsh-force-slip
# edit code/configs and run force-slip training/evaluation here
```

`git push` 和 `git pull` 由用户本人手动操作，pipeline/agent 不自动执行。相关任务的 commit 均在 `sparsh-force-slip` 分支上进行；每完成一个 phase（例如 Phase 1、Phase 2）后，必须在该分支进行一次覆盖该 phase 代码、配置、文档和实验记录的整体 commit，提交信息按 AGENTS.md 的 Lore Commit Protocol 填写。

## 核心研究问题

1. 联合 force/slip decoder 是否优于独立 force decoder 与 slip decoder？
2. 引入物理启发的一致性约束后，是否能减少 force 与 slip 预测之间的矛盾？
3. Sparsh DINO 与 MAE encoder 在 force regression、slip classification 与 consistency 指标上是否表现不同？
4. consistency loss 是否能在跨域或少样本条件下提供更稳定的归纳偏置？

## 输入与输出

### 输入

- tactile image 或 tactile image sequence；
- Sparsh DINO 或 MAE encoder 输出的 embedding；
- 可选 proprioceptive 信息，例如 gripper pose、contact state、time index。

### 输出

建议统一为：

```text
Fn      : normal force
Ft      : tangential/shear force magnitude
Fmag    : total force magnitude
p_slip  : slip probability
```

若原始 force label 是三轴力：

```text
Fn   = abs(Fz)
Ft   = sqrt(Fx^2 + Fy^2)
Fmag = sqrt(Fx^2 + Fy^2 + Fz^2)
```

## 模型结构

推荐最小结构：

```text
Tactile image
    ↓
Sparsh encoder: DINO / MAE
    ↓
Shared projection trunk
    ↓
 ┌───────────────┬────────────────┐
 Force head      Slip head
 Fn, Ft, Fmag     p_slip
```

### Baseline A: Separate Decoders

```text
Sparsh embedding → force decoder
Sparsh embedding → slip decoder
```

目的：作为非联合建模基线。

### Baseline B: Shared Multi-task Decoder

```text
Sparsh embedding → shared trunk → force head + slip head
```

目的：验证共享 tactile representation 是否提升下游任务。

### Proposed C: Consistency Multi-task Decoder

```text
Sparsh embedding → shared trunk → force head + slip head
                                  ↓
                       force-slip consistency loss
```

目的：通过物理启发约束提高预测一致性。

### Optional D: Learnable Friction-inspired Decoder

学习一个隐式摩擦阈值或 effective friction：

```text
mu_eff = h(z)
p_slip = sigmoid(alpha * (Ft - mu_eff * Fn))
```

该版本建议作为增强实验，不作为第一版主实现。

## Loss 设计

### Force loss

推荐使用 SmoothL1 或 MSE：

```text
L_force = SmoothL1([Fn, Ft, Fmag]_pred, [Fn, Ft, Fmag]_gt)
```

### Slip loss

推荐使用 BCE 或 class-balanced BCE：

```text
L_slip = BCE(p_slip, slip_gt)
```

若 slip 类别不平衡，可使用 focal loss。

### Force-slip consistency loss

从预测 force 构造 slip tendency：

```text
r = Ft_pred / (Fn_pred + eps)
q = sigmoid(alpha * (r - tau))
```

其中：

- `r` 是 shear-normal ratio；
- `tau` 是 friction-like threshold；
- `alpha` 控制 transition sharpness；
- `q` 是由 force 预测诱导出的 slip tendency。

一致性损失：

```text
L_cons = |p_slip - q|
```

或：

```text
L_cons = BCE(p_slip, q.detach())
```

建议第一版使用 detach，避免 slip head 反向干扰 force head 过强。

### 总损失

```text
L = L_force + lambda_slip * L_slip + beta_cons * L_cons
```

推荐从以下权重开始 grid search：

```text
lambda_slip ∈ {0.5, 1.0, 2.0}
beta_cons   ∈ {0.01, 0.05, 0.1, 0.2}
```

## 实验设计

### Encoder 设置

至少比较：

1. Sparsh DINO frozen encoder；
2. Sparsh MAE frozen encoder；
3. DINO partial finetuning；
4. MAE partial finetuning。

优先顺序：

```text
frozen encoder → partial finetuning → full finetuning
```

### Decoder 对照

| 编号 | 方法 | 目的 |
|---|---|---|
| A | separate force/slip decoders | 非联合 baseline |
| B | shared multi-task decoder | 多任务 baseline |
| C | joint decoder + consistency loss | 主方法 |
| D | friction-inspired learnable threshold | 增强版本 |

### 数据设置

可先在单域内验证：

1. Real GSmini train/test；
2. Sim cube train/test。

之后接入 sim2real pipeline 做跨域验证。

## 评价指标

### Force estimation

- MAE；
- RMSE；
- R²；
- normal force error；
- tangential force error；
- force direction cosine similarity，如果有三轴力。

### Slip detection

- Accuracy；
- Precision；
- Recall；
- F1；
- AUROC；
- AUPRC；
- slip onset detection latency，如果有时序标签。

### Force-slip consistency

建议加入独立 consistency 指标，否则方法优势可能无法充分体现：

1. slip-force agreement：高 `Ft/Fn` 区域 slip recall；
2. low-ratio false alarm：低 `Ft/Fn` 区域 slip false positive rate；
3. consistency calibration error：`p_slip` 与 shear-normal ratio 单调性；
4. contradiction rate：预测高 slip 但低 shear-normal ratio，或预测无 slip 但高 shear-normal ratio 的比例。

## 风险与应对

### 风险 1: Force label 没有明确 normal/shear 分量

应对：

- 优先确认三轴 force label；
- 若只有 force magnitude，则退化为 weak consistency；
- 在论文中明确该设置不能构成严格 friction cone 约束。

### 风险 2: Slip label 与 force label 时间不同步

应对：

- 使用 temporal window 平滑 slip 标签；
- 比较 current slip 与 future slip；
- 加入 slip onset tolerance window。

### 风险 3: Consistency loss 降低 force 或 slip 主指标

应对：

- 把 consistency 指标作为额外评价维度；
- 使用较小 `beta_cons`；
- 对 `q` 使用 detach；
- 只在高置信 force 区域启用 consistency loss。

## 最小可交付版本

1. 统一 force/slip 输出定义：`Fn, Ft, Fmag, p_slip`；
2. 实现 separate decoder baseline；
3. 实现 shared multi-task decoder；
4. 实现 consistency loss；
5. 完成 DINO vs MAE frozen encoder 对照；
6. 输出 force、slip、consistency 三类指标表格。

## 论文贡献表述建议

> We propose a physics-inspired joint force-slip decoder on top of pretrained Sparsh tactile representations. By enforcing consistency between predicted shear-normal force ratio and slip probability, the model improves physical plausibility of tactile force and slip perception without requiring closed-loop control.
