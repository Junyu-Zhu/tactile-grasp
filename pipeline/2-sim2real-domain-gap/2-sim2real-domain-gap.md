# Pipeline 2: Sim-to-Real Domain Gap for Tactile Force/Slip Decoding

## 目标

解决真实 GSmini 数据与 tactile_grasp 仿真 cube force/slip 数据之间的标签语义、输入分布和任务场景不一致问题，建立一个可比较、可诊断、可适配的 sim-to-real / real-to-sim tactile force-slip evaluation pipeline。

该 pipeline 的重点不是宣称完全消除 sim-to-real gap，而是：

1. 统一 real/sim force-slip label schema；
2. 量化 real/sim domain gap；
3. 建立跨域训练与测试矩阵；
4. 评估轻量 domain adaptation 方法；
5. 为 joint force/slip decoder 提供跨域验证场景。

## 核心研究问题

1. 官方 GSmini 数据与仿真 cube 数据在 force/slip 标签语义上能否映射到统一表示？
2. Sparsh DINO/MAE encoder 的 tactile representation 是否具有 real-to-sim 或 sim-to-real 迁移能力？
3. domain gap 主要来自输入图像分布、force label 定义、slip label 定义，还是任务场景差异？
4. 少量目标域样本或 feature alignment 是否能显著提升跨域 force/slip decoding？
5. physics-inspired joint decoder 是否比普通 decoder 更能抵抗 domain gap？

## 数据来源

### Real domain

官方 GelSight force 数据集：

```text
/home/zjy/Documents/dataset1/sparsh/tactile_datasets/Gelsight-mini/gelsight-force-estimation
```

已知用途：

- force-estimation 训练；
- slip-detection 训练；
- Sparsh DINO/MAE downstream decoder 训练。

### Sim domain

仿真 cube force/slip 数据建议位置：

```text
/home/zjy/Documents/grasp/tactile_grasp/sim_dataset
```

应包含或派生：

- tactile observation；
- normal force；
- tangential/shear force；
- object/contact relative velocity；
- slip label；
- optional grasp state metadata。

## Domain Gap 分解

### 1. 输入图像分布差异

Real GSmini：

- 真实传感器噪声；
- 光照变化；
- marker pattern；
- 弹性体真实形变；
- 复杂接触纹理。

Sim cube：

- 渲染或合成 tactile image；
- 接触边界更干净；
- 噪声模式缺失或不真实；
- 形变模型可能简化。

### 2. Force label 语义差异

Real 可能提供：

```text
Fx, Fy, Fz in sensor frame
```

Sim 可能提供：

```text
contact normal force
contact tangential force
net contact force
per-contact force
object-frame force
gripper-frame force
```

必须统一到公共表示：

```text
Fn   : normal force
Ft   : tangential/shear force magnitude
Fmag : total force magnitude
```

### 3. Slip label 语义差异

Real slip 可能来自：

- 数据集原始 slip label；
- tactile image displacement；
- 人工或规则标签；
- force/velocity threshold。

Sim slip 可从以下信号派生：

- object relative velocity；
- contact tangential velocity；
- friction cone violation；
- tangential displacement threshold。

建议区分：

```text
slip_observed : dataset original label
slip_physical : derived physical slip label
```

### 4. 任务场景差异

Real GSmini 可能是单传感器压滑实验；Sim cube 是机器人抓取过程。两者任务上下文不同，decoder 直接迁移可能不稳定。

## 统一 Label Schema

建议统一导出每条样本的 metadata：

```text
sample_id
domain: real | sim
timestamp or frame_index
image_path or tensor_path
Fx, Fy, Fz optional
Fn
Ft
Fmag
slip_observed optional
slip_physical optional
split: train | val | test
object_id optional
friction optional
contact_state optional
```

### Force 映射

如果有三轴力：

```text
Fn   = abs(Fz)
Ft   = sqrt(Fx^2 + Fy^2)
Fmag = sqrt(Fx^2 + Fy^2 + Fz^2)
```

如果仿真提供 contact normal/tangential force：

```text
Fn = contact_normal_force
Ft = contact_tangential_force_norm
Fmag = sqrt(Fn^2 + Ft^2)
```

### Slip 映射

仿真中推荐定义：

```text
slip_physical = 1 if contact_tangential_velocity > threshold
```

可选加强：

```text
slip_physical = 1 if tangential_displacement_over_window > threshold
```

真实数据保留原始标签，同时尽量派生 weak physical slip proxy。

## 实验矩阵

核心跨域矩阵：

| Train Domain | Test Domain | 用途 |
|---|---|---|
| Real GSmini | Real GSmini | real in-domain baseline |
| Sim cube | Sim cube | sim in-domain baseline |
| Real GSmini | Sim cube | real-to-sim zero-shot |
| Sim cube | Real GSmini | sim-to-real zero-shot |
| Real + Sim | Real / Sim | mixed-domain training |
| Real + few-shot Sim | Sim | target adaptation |
| Sim + few-shot Real | Real | target adaptation |

Few-shot target ratios：

```text
1%, 5%, 10%, 20%
```

## 方法路线

### Stage 1: Zero-shot transfer diagnostic

目标：先量化 domain gap，不做复杂适配。

设置：

- train on real, test on sim；
- train on sim, test on real；
- DINO vs MAE；
- separate decoder vs joint decoder。

输出：

- 跨域性能下降幅度；
- 哪个任务更难迁移：force 还是 slip；
- 哪个 encoder 更稳健。

### Stage 2: Feature distribution analysis

分析 Sparsh embedding 的 real/sim 分布差异：

- t-SNE / UMAP；
- feature mean/std distance；
- MMD distance；
- covariance distance；
- per-domain nearest-neighbor mixing ratio。

目的：解释跨域性能差异。

### Stage 3: Lightweight domain adaptation

优先使用稳定、低工程风险方法。

#### 方法 A: Feature normalization alignment

- real/sim mean-std normalization；
- domain-specific LayerNorm / BatchNorm；
- feature whitening。

#### 方法 B: CORAL alignment

对齐 real/sim feature covariance：

```text
L_coral = || C_real - C_sim ||_F^2
```

#### 方法 C: MMD alignment

最小化 real/sim embedding distribution discrepancy。

#### 方法 D: Few-shot target adaptation

使用少量目标域标注样本微调 decoder 或 adapter：

```text
1%, 5%, 10%, 20% target labels
```

### Stage 4: Combine with joint force/slip decoder

最终模型：

```text
Tactile image
    ↓
Sparsh DINO/MAE encoder
    ↓
Domain alignment module
    ↓
Joint force/slip decoder
    ↓
Force-slip consistency loss
```

总损失：

```text
L = L_force + lambda_slip * L_slip + beta_cons * L_cons + gamma_domain * L_domain
```

其中：

- `L_force`：force regression；
- `L_slip`：slip classification；
- `L_cons`：force-slip consistency；
- `L_domain`：CORAL/MMD/normalization alignment loss。

## 评价指标

### In-domain 指标

- force MAE / RMSE；
- slip F1 / AUROC / AUPRC；
- consistency metrics。

### Cross-domain 指标

- zero-shot performance drop；
- target adaptation gain；
- few-shot sample efficiency；
- real-to-sim vs sim-to-real asymmetry；
- DINO vs MAE transfer gap。

### Domain gap 指标

- feature MMD；
- covariance distance；
- UMAP/t-SNE domain separation；
- linear domain classifier accuracy。

如果 domain classifier 很容易区分 real/sim，说明 representation 中仍有强 domain-specific 信息。

## 风险与应对

### 风险 1: Real 与 Sim 图像差异过大，zero-shot 几乎失败

应对：

- 不把 zero-shot 成功作为唯一论文目标；
- 将 zero-shot 作为 domain gap diagnostic；
- 强调 few-shot adaptation 与 feature alignment 的提升；
- 报告 feature distribution analysis。

### 风险 2: Force label 无法严格对齐

应对：

- 使用公共表示 `Fn, Ft, Fmag`；
- 明确坐标系转换；
- 对无法对齐的数据只做 weak comparison；
- 在论文中避免过度声称绝对 force 一致。

### 风险 3: Slip label 定义不一致

应对：

- 区分 `slip_observed` 与 `slip_physical`；
- 对 sim 使用 velocity/displacement threshold；
- 对 real 保留原始标签并说明语义差异；
- 使用 tolerance window 处理时序误差。

### 风险 4: Domain adaptation 方法过复杂导致工程失控

应对：

推荐优先级：

```text
zero-shot diagnostic
→ feature normalization
→ few-shot decoder adaptation
→ CORAL/MMD
→ partial finetuning
```

暂不建议第一阶段使用：

- CycleGAN；
- adversarial domain discriminator；
- diffusion tactile translation；
- closed-loop control。

## 最小可交付版本

1. 统一 real/sim dataset schema；
2. 完成 force/slip label mapping；
3. 建立 real-to-sim 与 sim-to-real evaluation split；
4. 运行 DINO/MAE zero-shot transfer；
5. 输出 domain gap diagnostic 图表；
6. 完成 few-shot target adaptation；
7. 与 joint force/slip decoder pipeline 汇合。

## 论文贡献表述建议

> We establish a unified real/sim tactile force-slip evaluation protocol and quantify the domain gap between real GelSight data and simulated grasping data. Based on pretrained Sparsh representations, we evaluate zero-shot transfer, few-shot adaptation, and lightweight feature alignment for cross-domain tactile force and slip decoding.
