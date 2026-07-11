# Sparsh Force-Slip 与 World-Model-Style 未来状态预测方案

## 0. 依据范围

本文只基于同目录下的 `1-force-slip-decoder.md` 整理，不额外假设其他 pipeline 文件中的实现细节。

源文档给出的核心设定是：

- 输入：tactile image 或 tactile image sequence；
- 表征：Sparsh DINO 或 MAE encoder 输出 embedding；
- 可选输入：gripper pose、contact state、time index 等 proprioceptive 信息；
- 输出：`Fn, Ft, Fmag, p_slip`；
- 模型：Sparsh encoder + shared projection trunk + force head + slip head；
- 主要约束：force-slip consistency / friction-inspired loss；
- 当前边界：先做离线 perception-level 任务，暂不包含闭环控制验证。

因此，下面的方案也保持离线验证边界：先把 joint force/slip decoder 扩展成一个能预测短期未来状态的 tactile latent predictor，用它判断 grasp 是否稳定；闭环控制只作为后续方向，不作为第一版交付目标。

## 0.1 论文与代码索引

下面的文献用于给本文中的“表征、融合、未来预测、稳定性判断”提供依据。注意：这些论文不是说可以直接照搬到触觉任务，而是说明某类建模选择在视觉/触觉/控制中已有先例。

| 用途 | 论文/方法 | 链接 | GitHub / 项目 |
|---|---|---|---|
| 触觉 SSL 表征 | Sparsh: Self-supervised touch representations for vision-based tactile sensing | https://proceedings.mlr.press/v270/higuera25a.html / https://sparsh-ssl.github.io/ | https://github.com/facebookresearch/sparsh |
| DINO 触觉/视觉 encoder 依据 | Emerging Properties in Self-Supervised Vision Transformers | https://arxiv.org/abs/2104.14294 | https://github.com/facebookresearch/dino |
| DINOv2 frozen feature 依据 | DINOv2: Learning Robust Visual Features without Supervision | https://arxiv.org/abs/2304.07193 | https://github.com/facebookresearch/dinov2 |
| MAE encoder 依据 | Masked Autoencoders Are Scalable Vision Learners | https://arxiv.org/abs/2111.06377 | https://github.com/facebookresearch/mae |
| JEPA 表征预测 | Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture | https://arxiv.org/abs/2301.08243 | https://github.com/facebookresearch/ijepa |
| 视频 latent feature prediction | Revisiting Feature Prediction for Learning Visual Representations from Video / V-JEPA | https://arxiv.org/abs/2404.08471 | https://github.com/facebookresearch/jepa |
| 原始 world model 思路 | World Models | https://arxiv.org/abs/1803.10122 | https://worldmodels.github.io/ / https://github.com/hardmaru/WorldModelsExperiments |
| latent dynamics + planning | Learning Latent Dynamics for Planning from Pixels / PlaNet | https://arxiv.org/abs/1811.04551 | https://planetrl.github.io/ / https://github.com/google-research/planet |
| RSSM / imagined rollout | Mastering Diverse Domains through World Models / DreamerV3 | https://arxiv.org/abs/2301.04104 | https://github.com/danijar/dreamerv3 |
| decoder-free latent MPC | TD-MPC: Temporal Difference Learning for Model Predictive Control | https://arxiv.org/abs/2203.04955 | https://github.com/nicklashansen/tdmpc |
| scalable latent MPC | TD-MPC2: Scalable, Robust World Models for Continuous Control | https://arxiv.org/abs/2310.16828 | https://tdmpc2.com/ / https://github.com/nicklashansen/tdmpc2 |
| frozen feature world model | Back to the Features: DINO as a Foundation for Video World Models / DINO-World | https://arxiv.org/abs/2507.19468 | 项目页：https://baldassarrefe.github.io/publication/dino-world-2025/；未确认公开代码 |
| JEPA latent planning | Learning from Reward-Free Offline Data: A Case for Planning with Latent Dynamics Models / PLDM | https://arxiv.org/abs/2502.14819 | https://latent-planning.github.io/ / https://github.com/vladisai/PLDM |
| robot implicit world modeling | FLARE: Robot Learning with Implicit World Modeling | https://arxiv.org/abs/2505.15659 | https://research.nvidia.com/labs/gear/flare/；未确认公开代码 |
| VLA + latent world model | VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model | https://arxiv.org/abs/2602.10098 | https://github.com/ginwind/VLA-JEPA |
| feature-wise 条件融合 | FiLM: Visual Reasoning with a General Conditioning Layer | https://arxiv.org/abs/1709.07871 | https://github.com/ethanjperez/film |
| gated multimodal fusion | Gated Multimodal Units for Information Fusion | https://arxiv.org/abs/1702.01992 | 未确认官方代码 |
| 多模态融合分类依据 | Multimodal Machine Learning: A Survey and Taxonomy | https://arxiv.org/abs/1705.09406 | 综述，无官方代码要求 |
| 触觉 force/slip 联合任务先例 | Force estimation and slip detection/classification for grip control using a biomimetic tactile sensor | https://is.mpg.de/publications/su2015force-f48108e4-e831-4481-ba33-38e6b0fc615f | 未确认公开代码 |
| 触觉稳定性/slip 预测 | Tactile-Driven Grasp Stability and Slip Prediction | https://www.mdpi.com/2218-6581/8/4/85 | 未确认公开代码 |
| incipient slip / safety margin | Learning to estimate incipient slip with tactile sensing to gently grasp objects | https://doi.org/10.1109/ICRA57147.2024.10611517 | 未确认公开代码 |

## 1. 核心判断

你的想法是可行的，而且和 world model 的思想是同构的：

```text
world model:
    state_t + action_t -> state_{t+1} / future outcome

Sparsh force-slip:
    tactile latent_t + force_t + slip_t + optional proprio_t
        -> tactile latent_{t+1} / force_{t+1} / slip_{t+1} / stability
```

不过需要一个关键修正：**不建议只用 slip 去预测下一帧状态**。Slip 是未来不稳定性的强信号，但它不是完整状态。更稳妥的状态应该包含：

```text
s_t = [z_t, Fn_t, Ft_t, Fmag_t, p_slip_t, optional proprio_t]
```

其中 `z_t` 是 Sparsh embedding，`Fn/Ft/Fmag` 提供接触力状态，`p_slip` 提供接触稳定性风险，optional proprio 提供夹爪/接触上下文。

第一版应做成：

```text
输入:
    tactile image sequence 或当前 tactile image
    Sparsh embedding z_t
    optional proprio_t

输出:
    当前感知: Fn_t, Ft_t, Fmag_t, p_slip_t
    未来预测: z_{t+1}, Fn_{t+1}, Ft_{t+1}, Fmag_{t+1}, p_slip_{t+1}
    稳定性: p_stable 或 p_future_slip
```

这样既保留了原 pipeline 的 force estimation 和 slip detection，又加入了 world-model-style 的 future state prediction。

## 2. 统一输出定义

建议把原文档中的输出扩展为一个统一结构：

```text
current:
    Fn_t
    Ft_t
    Fmag_t
    p_slip_t

future:
    z_pred_{t+1}
    Fn_pred_{t+1}
    Ft_pred_{t+1}
    Fmag_pred_{t+1}
    p_slip_pred_{t+1}

stability:
    p_stable_t
    risk_score_t
```

其中：

```text
r_t = Ft_t / (Fn_t + eps)
q_t = sigmoid(alpha * (r_t - tau))
```

`q_t` 是由 force 诱导出的 slip tendency；`p_slip_t` 是 slip head 的预测。稳定性可以先定义为：

```text
risk_score_t = w1 * p_slip_t + w2 * p_slip_pred_{t+1} + w3 * q_t
p_stable_t  = 1 - sigmoid(risk_score_t)
```

第一版可以不用学习复杂稳定性标签，而是先用 future slip 和 force-slip contradiction 构造弱监督或评价指标。

### 论文依据

- `z_t -> z_{t+1}` 这种 latent future prediction 直接对应 PlaNet、DreamerV3、TD-MPC/TD-MPC2、DINO-World、PLDM 的共同思想：不一定重建像素，而是在 compact latent 或 foundation feature space 里预测未来。
- 选择 Sparsh embedding 作为 `z_t`，依据来自 Sparsh 本身，以及 DINO/MAE/I-JEPA/V-JEPA 这些 Sparsh 支持或相近的 SSL 表征学习方法。
- 用 `Fn/Ft/p_slip` 作为稳定性判断依据，和 Force estimation and slip detection、Tactile-Driven Grasp Stability and Slip Prediction、Learning to estimate incipient slip 这些触觉 grasp stability / slip 论文一致。

## 3. 融合方法一：State-Augmented Multi-task Decoder

这是最小改动版本，适合第一阶段实现。

### 结构

```text
tactile image_t
    ↓
Sparsh encoder
    ↓
z_t
    ↓
shared trunk
    ├── force head: Fn_t, Ft_t, Fmag_t
    ├── slip head : p_slip_t
    └── stability head: p_stable_t / risk_score_t
```

### 融合方式

把不同表征拼成一个当前状态：

```text
h_t = MLP([z_t, Fn_t, Ft_t, Fmag_t, p_slip_t, q_t])
```

然后从 `h_t` 输出稳定性：

```text
p_stable_t = sigmoid(MLP(h_t))
```

### 优点

- 与原文档的 shared multi-task decoder 最接近；
- 不强依赖序列数据；
- 能直接比较 separate decoder、shared decoder、consistency decoder；
- 适合作为 world-model-style 方案的 baseline。

### 缺点

- 它没有真正预测 `state_{t+1}`；
- 更像 stability classifier，不是完整 world model。

### 适用定位

作为 `Baseline E: state-augmented stability decoder`。

### 论文/代码依据

- **Sparsh: Self-supervised touch representations**：提供 frozen 或 partial-finetuned tactile encoder 的直接依据。链接：https://proceedings.mlr.press/v270/higuera25a.html，代码：https://github.com/facebookresearch/sparsh。
- **Multimodal Machine Learning: A Survey and Taxonomy**：支持 early fusion / representation fusion / co-learning 这类把 `z_t, force, slip, proprio` 统一成状态向量的分类框架。链接：https://arxiv.org/abs/1705.09406。
- **Gated Multimodal Units for Information Fusion**：如果不想简单 concat，可以把 `force/slip/q` 对 `z_t` 的影响做成 gated fusion。链接：https://arxiv.org/abs/1702.01992。
- **FiLM**：如果希望用 `p_slip_t` 或 `q_t` 调制 Sparsh latent，可用 feature-wise affine modulation。链接：https://arxiv.org/abs/1709.07871，代码：https://github.com/ethanjperez/film。

## 4. 融合方法二：Slip-Conditioned Latent Dynamics

这是最贴近你想法的版本：用 slip 风险影响下一帧 tactile latent 的预测。

### 结构

```text
tactile image_t
    ↓
Sparsh encoder
    ↓
z_t
    ↓
force/slip heads
    ↓
Fn_t, Ft_t, Fmag_t, p_slip_t
    ↓
latent dynamics head
    ↓
z_pred_{t+1}
```

动态模型：

```text
z_pred_{t+1} = f_dyn([z_t, Fn_t, Ft_t, Fmag_t, p_slip_t, optional proprio_t])
```

如果有连续 tactile image sequence，则可以用同一个 Sparsh encoder 得到监督目标：

```text
z_gt_{t+1} = SparshEncoder(image_{t+1})
L_dyn = SmoothL1(z_pred_{t+1}, stopgrad(z_gt_{t+1}))
```

### 融合方式

`p_slip_t` 不单独作为状态，而是作为 dynamics gate：

```text
g_t = sigmoid(MLP([p_slip_t, q_t]))
z_pred_{t+1} = z_t + g_t * Delta_slip(z_t) + (1 - g_t) * Delta_stick(z_t)
```

直觉：

- `p_slip_t` 低：下一帧接触状态应接近当前稳定接触；
- `p_slip_t` 高：下一帧 latent 更可能发生明显变化；
- `Ft/Fn` 高但 `p_slip` 低：这是 force-slip contradiction，需要被 loss 惩罚。

### 输出

```text
Fn_t, Ft_t, Fmag_t, p_slip_t
z_pred_{t+1}
Fn_pred_{t+1}, Ft_pred_{t+1}, Fmag_pred_{t+1}, p_slip_pred_{t+1}
p_stable_t
```

未来 force/slip 可以从 `z_pred_{t+1}` 再接一组共享 head：

```text
future force/slip head(z_pred_{t+1})
```

### 优点

- 明确对应 world model 的 `state_t -> state_{t+1}`；
- 能验证 slip 是否真的提供未来预测信息；
- 与源文档的 tactile image sequence 输入兼容；
- 可以离线训练，不需要先接闭环控制。

### 缺点

- 需要相邻帧或短序列；
- 如果 slip/force label 时间不同步，会直接污染未来预测；
- 如果没有 action/proprio，模型学到的是被动接触演化，不是完整 action-conditioned dynamics。

### 适用定位

作为主方案：`Proposed E: slip-conditioned Sparsh latent world model`。

### 论文/代码依据

- **World Models**：用 compressed latent state 和 recurrent dynamics 预测未来，是本文 “tactile latent world model” 类比的最早来源之一。链接：https://arxiv.org/abs/1803.10122，项目：https://worldmodels.github.io/，代码：https://github.com/hardmaru/WorldModelsExperiments。
- **PlaNet**：从像素学习 latent dynamics，并在 latent space 规划；对应这里的 `z_t -> z_{t+1}`。链接：https://arxiv.org/abs/1811.04551，项目：https://planetrl.github.io/，代码：https://github.com/google-research/planet。
- **TD-MPC / TD-MPC2**：强调 decoder-free、task-oriented latent dynamics；对应这里“不先生成下一帧 tactile image，而预测 Sparsh latent / force / slip”。TD-MPC：https://arxiv.org/abs/2203.04955，代码：https://github.com/nicklashansen/tdmpc；TD-MPC2：https://arxiv.org/abs/2310.16828，项目：https://tdmpc2.com/，代码：https://github.com/nicklashansen/tdmpc2。
- **DINO-World**：直接支持“冻结 foundation encoder，然后训练未来 feature predictor”的设计；把 DINOv2 feature 换成 Sparsh feature，就是本文最接近的迁移模板。链接：https://arxiv.org/abs/2507.19468，项目页：https://baldassarrefe.github.io/publication/dino-world-2025/。
- **PLDM**：用 JEPA latent dynamics 做 offline planning，说明 latent prediction 可以服务于下游目标，而不必生成像素。链接：https://arxiv.org/abs/2502.14819，项目：https://latent-planning.github.io/，代码：https://github.com/vladisai/PLDM。

## 5. 融合方法三：Recurrent Belief State Fusion

如果输入是 tactile image sequence，而不只是单帧，建议引入 belief state。

### 结构

```text
image_{t-k:t}
    ↓
Sparsh encoder per frame
    ↓
z_{t-k:t}
    ↓
GRU / temporal transformer
    ↓
b_t
    ├── current force/slip heads
    ├── future force/slip heads
    └── stability head
```

其中：

```text
b_t = recurrent belief state
```

`b_t` 融合了历史 tactile latent、force trend、slip trend，比单帧 `z_t` 更接近 world model 中的 hidden state。

### 输出

```text
Fn_t, Ft_t, Fmag_t, p_slip_t
Fn_pred_{t+1:t+H}, Ft_pred_{t+1:t+H}, p_slip_pred_{t+1:t+H}
p_stable_{t:t+H}
```

### Loss

```text
L = L_force_current
  + lambda_slip * L_slip_current
  + gamma_future * L_future
  + beta_cons * L_cons
```

其中：

```text
L_future = SmoothL1(force_pred_{t+1}, force_gt_{t+1})
         + BCE(p_slip_pred_{t+1}, slip_gt_{t+1})
```

### 优点

- 能自然处理 slip onset latency；
- 对 slip label 与 force label 的小时间偏差更鲁棒；
- 更适合验证 grasp 是否稳定，因为稳定性本身是时序概念。

### 缺点

- 数据组织更复杂；
- 训练时要处理 sequence window；
- 评价指标要加入 future slip、latency、稳定窗口。

### 适用定位

作为第二阶段主方法：`Proposed F: recurrent tactile belief world model`。

### 论文/代码依据

- **PlaNet**：使用 deterministic + stochastic transition 处理部分可观测像素控制任务；触觉接触同样存在部分可观测性，历史窗口比单帧更可靠。链接：https://arxiv.org/abs/1811.04551，代码：https://github.com/google-research/planet。
- **DreamerV3**：通过 world model imagined rollout 训练行为；虽然本文第一版不做闭环控制，但 `b_t -> future force/slip` 的时序 belief state 可以借鉴 RSSM/imagined rollout 的验证方式。链接：https://arxiv.org/abs/2301.04104，代码：https://github.com/danijar/dreamerv3。
- **V-JEPA**：不重建像素，而预测视频片段的 latent representation；对应 tactile sequence 中预测未来 Sparsh feature。链接：https://arxiv.org/abs/2404.08471，代码：https://github.com/facebookresearch/jepa。
- **Learning to estimate incipient slip with tactile sensing to gently grasp objects**：说明 slip 前兆和 safety margin 是时序触觉信号中的关键预测对象。链接：https://doi.org/10.1109/ICRA57147.2024.10611517。

## 6. 融合方法四：Contact Mode Latent Fusion

这是更物理化的版本，把接触状态拆成离散模式。

### 接触模式

```text
mode_t ∈ {
    no_contact,
    stick,
    pre_slip,
    slip
}
```

模型输出：

```text
p(mode_t)
p(mode_{t+1})
Fn_t, Ft_t, Fmag_t
p_slip_t
p_stable_t
```

### 融合方式

```text
h_t = MLP([z_t, Fn_t, Ft_t, Fmag_t, p_slip_t, q_t])
p(mode_t) = softmax(mode_head(h_t))
p(mode_{t+1}) = transition_head([h_t, p(mode_t)])
```

`pre_slip` 是关键，因为它能表达“还没滑，但 force ratio 已经危险”的状态。

### 优点

- 比单一 `p_slip` 更可解释；
- 适合与 friction-inspired loss 结合；
- 能把 stability 判断变成模式转移问题。

### 缺点

- 需要 mode label，或者用 weak rule 构造伪标签；
- 如果标签质量不好，模式分类会不稳定。

### 适用定位

作为增强实验：`Optional G: contact-mode transition model`。

### 论文/代码依据

- **Force estimation and slip detection/classification for grip control using a biomimetic tactile sensor**：把 force estimation、slip detection/classification 和 grip control 连起来，是本文 `force/slip -> stability` 的直接触觉任务先例。链接：https://is.mpg.de/publications/su2015force-f48108e4-e831-4481-ba33-38e6b0fc615f。
- **Tactile-Driven Grasp Stability and Slip Prediction**：把 grasp stability 与 slip prediction 作为学习任务处理，支持 `stable / unstable / slip` 这类 contact mode 或 stability label。链接：https://www.mdpi.com/2218-6581/8/4/85。
- **Learning to estimate incipient slip with tactile sensing to gently grasp objects**：把“距离滑移的安全裕度”作为控制变量，支持本文增加 `pre_slip` mode，而不是只做二分类 slip。链接：https://doi.org/10.1109/ICRA57147.2024.10611517。

## 7. 融合方法五：Consistency-Energy Fusion

这个版本不显式预测下一帧 latent，而是把 force/slip 的矛盾作为能量或风险。

### 结构

```text
z_t
    ↓
force/slip heads
    ↓
Fn_t, Ft_t, Fmag_t, p_slip_t
    ↓
consistency energy
    ↓
risk_score_t
```

定义：

```text
r_t = Ft_t / (Fn_t + eps)
q_t = sigmoid(alpha * (r_t - tau))
E_t = |p_slip_t - q_t| + q_t + p_slip_t
```

稳定性：

```text
p_stable_t = sigmoid(-E_t)
```

也可以让模型学习：

```text
E_t = MLP([z_t, Fn_t, Ft_t, Fmag_t, p_slip_t, q_t])
```

### 优点

- 与原文档中的 consistency loss 完全一致；
- 不需要下一帧标签；
- 可以作为所有方法的统一风险指标。

### 缺点

- 不是严格 world model；
- 只能判断当前接触是否危险，不能真正 rollout 未来。

### 适用定位

作为风险估计 baseline：`Baseline H: consistency-energy stability score`。

### 论文/代码依据

- **Force estimation and slip detection/classification for grip control using a biomimetic tactile sensor**：支持 force 与 slip 同时估计，并把 slip detection 用于 grip control。链接：https://is.mpg.de/publications/su2015force-f48108e4-e831-4481-ba33-38e6b0fc615f。
- **Learning to estimate incipient slip with tactile sensing to gently grasp objects**：用 safety margin 估计接触距离 frictional limit 的距离；这和本文 `q_t = sigmoid(alpha * (Ft/Fn - tau))` 作为风险能量的想法一致。链接：https://doi.org/10.1109/ICRA57147.2024.10611517。
- **Tactile-Driven Grasp Stability and Slip Prediction**：支持用触觉预测稳定/滑移，而不是只回归力。链接：https://www.mdpi.com/2218-6581/8/4/85。

## 8. 推荐实验路线

### Stage 1: 保持原 pipeline，增加统一输出

先实现源文档已有的三类模型：

```text
A: separate force/slip decoders
B: shared multi-task decoder
C: consistency multi-task decoder
```

然后新增：

```text
E0: shared multi-task decoder + p_stable_t
```

目的：确认 `Fn/Ft/Fmag/p_slip/q` 是否已经能预测稳定性。

### Stage 2: 单步未来预测

新增：

```text
E1: slip-conditioned latent dynamics
```

训练目标：

```text
z_t -> z_{t+1}
z_pred_{t+1} -> Fn_{t+1}, Ft_{t+1}, Fmag_{t+1}, p_slip_{t+1}
```

关键对照：

```text
without slip input:
    z_t + force_t -> future

with slip input:
    z_t + force_t + p_slip_t -> future
```

如果加入 `p_slip_t` 后 future slip / future force / stability 指标提升，说明 slip 确实提供了 world-model-style 未来状态信息。

对应论文依据：

- **DINO-World**：frozen feature + future predictor；
- **TD-MPC2**：latent dynamics 服务于控制/规划，而不是像素重建；
- **PLDM**：JEPA latent dynamics 可以用于 offline planning；
- **V-JEPA**：latent representation prediction 可作为自监督时序目标。

### Stage 3: 多步稳定性窗口

新增：

```text
F: recurrent belief state fusion
```

输出：

```text
p_slip_pred_{t+1:t+H}
p_stable_{t:t+H}
```

稳定 grasp 可以先定义为：

```text
stable if:
    max p_slip_pred_{t+1:t+H} < threshold
    and Fn_pred stays in valid range
    and force-slip contradiction rate is low
```

### Stage 4: 物理解释增强

最后尝试：

```text
G: contact-mode transition model
H: consistency-energy fusion
```

这两个方法更适合写论文分析，因为它们能解释为什么模型判断 grasp 稳定或不稳定。

## 9. 损失函数建议

在源文档已有 loss 上增加 future loss。

### 当前感知 loss

```text
L_force = SmoothL1([Fn, Ft, Fmag]_pred, [Fn, Ft, Fmag]_gt)
L_slip  = BCE(p_slip, slip_gt)
```

### 一致性 loss

```text
r = Ft_pred / (Fn_pred + eps)
q = sigmoid(alpha * (r - tau))
L_cons = BCE(p_slip, q.detach())
```

### 未来状态 loss

```text
L_dyn = SmoothL1(z_pred_{t+1}, stopgrad(z_gt_{t+1}))
```

如果有下一帧 force/slip label：

```text
L_future_force = SmoothL1(force_pred_{t+1}, force_gt_{t+1})
L_future_slip  = BCE(p_slip_pred_{t+1}, slip_gt_{t+1})
```

### 总 loss

```text
L = L_force
  + lambda_slip * L_slip
  + beta_cons * L_cons
  + gamma_dyn * L_dyn
  + gamma_future * (L_future_force + L_future_slip)
```

建议起始权重：

```text
lambda_slip  = 1.0
beta_cons    = 0.01 or 0.05
gamma_dyn    = 0.1
gamma_future = 0.5
```

先让 force/slip 主任务稳定，再逐步提高 future loss 权重。

### 损失设计的论文依据

- `L_dyn = SmoothL1/L2(z_pred, stopgrad(z_gt))`：对应 I-JEPA、V-JEPA、PLDM、DINO-World 的 latent/feature prediction 思路。
- `L_future_force + L_future_slip`：对应触觉任务中 force estimation 与 slip detection 的联合预测先例。
- `q.detach()`：工程上用于避免 consistency 分支过强干扰 force head；思想上接近 teacher/target stop-gradient 的 SSL 设计，例如 DINO、I-JEPA、V-JEPA。

## 10. 评价指标

除源文档中的 force、slip、consistency 指标外，新增 future-state 指标。

### Future force/slip

- `Fn_{t+1}` MAE；
- `Ft_{t+1}` MAE；
- `p_slip_{t+1}` F1 / AUROC / AUPRC；
- slip onset prediction latency。

### Latent prediction

- `z_pred_{t+1}` 与 `z_gt_{t+1}` 的 cosine similarity；
- latent SmoothL1；
- 用 `z_pred_{t+1}` 解码出的 force/slip 是否优于不用 slip 的 baseline。

### Stability

- future slip recall；
- stable window accuracy；
- contradiction rate；
- low-ratio false alarm；
- high-ratio missed slip。

最重要的对照是：

```text
z_t + force_t
vs
z_t + force_t + p_slip_t
vs
z_t + force_t + p_slip_t + consistency q_t
```

如果第三个最好，说明 force-slip consistency 不只是正则项，而是在构造更好的未来状态表征。

### 指标依据

- force/slip 主指标参考 tactile force/slip 论文中的 force error、slip classification、stability/slip prediction 设置。
- latent prediction 指标参考 JEPA/V-JEPA/DINO-World：关注 future representation 是否保留下游可用信息，而不要求像素级复原。
- stability 指标参考 Tactile-Driven Grasp Stability and Slip Prediction 与 incipient slip safety margin：重点看是否提前发现未来 slip 或接触安全裕度不足。

## 11. 可行性分析

### 可行点

1. 源文档已经允许 tactile image sequence，因此可以自然构造 `t -> t+1` 监督。
2. 源文档已经定义了 `Fn/Ft/Fmag/p_slip`，这正好能组成接触状态。
3. 源文档已有 force-slip consistency，能作为稳定性判断的物理先验。
4. Sparsh embedding 可以作为 latent state，避免直接预测下一帧 tactile image 的高难度问题。
5. 离线验证即可完成第一轮实验，不需要直接进入闭环控制。

### 主要风险

1. **Slip 不是完整 dynamics state**：只用 slip 预测下一帧会丢失接触几何、力大小、历史趋势。
2. **缺少 action/proprio 时，模型不是完整 action-conditioned world model**：它只能预测当前接触自然演化，不能完整预测夹爪动作后的变化。
3. **force/slip 时间不同步会影响 future loss**：源文档已经指出需要 temporal window 和 onset tolerance。
4. **稳定 grasp 标签可能不存在**：第一版应先用 future slip 和 force consistency 构造可评价的 proxy。
5. **consistency loss 可能损害主指标**：应使用小 `beta_cons` 和 `q.detach()`，并保留无 consistency 对照。

### 结论

该想法可行，但第一版应表述为：

> 基于 Sparsh tactile latent 的 force-slip predictive representation，而不是完整机器人 world model。

更准确的研究目标是：

```text
Can joint force/slip perception improve short-horizon tactile state prediction
and grasp stability estimation?
```

这比直接宣称构建完整 world model 更稳，也更容易通过离线实验验证。

### 与论文的对应关系

| 本文想法 | 对应论文依据 | 为什么相关 |
|---|---|---|
| Sparsh latent 作为触觉状态 | Sparsh、DINO、MAE、I-JEPA、V-JEPA | 说明 frozen/SSL encoder 能提供可迁移表征 |
| 不预测 tactile image，只预测 latent | TD-MPC、TD-MPC2、DINO-World、PLDM | 避免高维像素生成，把能力集中到任务相关 latent |
| `z_t -> z_{t+1}` | World Models、PlaNet、DreamerV3、DINO-World | world model 的核心就是学习未来状态演化 |
| slip 作为 dynamics gate | tactile slip / incipient slip 论文 + FiLM/GMU | slip 是接触状态变化信号，可作为条件调制或门控 |
| `Ft/Fn` consistency | force/slip grip-control 论文 + incipient slip safety margin | shear-normal ratio 和 friction margin 是稳定性物理先验 |
| future slip 判断 grasp stability | Tactile-Driven Grasp Stability and Slip Prediction | 抓取稳定性可以作为触觉序列预测任务 |

## 12. 最推荐的第一版方法

建议第一版只做一个主方法和两个关键对照。

### 对照 1：Shared Multi-task Decoder

```text
z_t -> Fn_t, Ft_t, Fmag_t, p_slip_t
```

### 对照 2：Shared Multi-task + Consistency

```text
z_t -> Fn_t, Ft_t, Fmag_t, p_slip_t
L_cons = BCE(p_slip_t, q_t.detach())
```

### 主方法：Slip-Conditioned Sparsh Latent Predictor

```text
z_t
Fn_t, Ft_t, Fmag_t, p_slip_t, q_t
    ↓
predict z_{t+1}
    ↓
predict Fn_{t+1}, Ft_{t+1}, Fmag_{t+1}, p_slip_{t+1}
    ↓
estimate p_stable_t
```

主 claim：

```text
Joint force-slip perception provides a predictive tactile state representation.
Slip probability and force-slip consistency improve short-horizon future tactile
state prediction and offline grasp stability estimation.
```

### 第一版主方法对应参考

- 表征 backbone：Sparsh + DINO/MAE/I-JEPA/V-JEPA。
- future predictor：DINO-World / PLDM / TD-MPC2。
- 时序 belief 可选增强：PlaNet / DreamerV3 / V-JEPA。
- slip/stability 物理解释：Force estimation and slip detection、Tactile-Driven Grasp Stability and Slip Prediction、Learning to estimate incipient slip。
- 融合模块可选实现：concat baseline、GMU gate、FiLM modulation、cross-attention/temporal transformer。

## 13. 最小交付清单

1. 保留原输出：`Fn, Ft, Fmag, p_slip`；
2. 新增 `q = sigmoid(alpha * (Ft/Fn - tau))`；
3. 新增 `risk_score` 和 `p_stable`；
4. 新增 `z_pred_{t+1}`；
5. 新增 future force/slip heads；
6. 新增 `L_dyn` 与 `L_future`；
7. 做三组对照：
   - no consistency；
   - consistency only；
   - consistency + future prediction；
8. 报告四类指标：
   - force estimation；
   - slip detection；
   - force-slip consistency；
   - future-state / stability prediction。

## 14. 推荐命名

可以把该方向命名为：

```text
Sparsh-ForceSlip-Predictor
```

或更贴近 world model：

```text
Sparsh Tactile Latent World Model
```

如果论文表述更谨慎，推荐：

```text
Predictive Force-Slip Representation for Tactile Grasp Stability
```
