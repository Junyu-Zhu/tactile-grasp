---
title: "Force-Slip Decoder Phase 3 建议：轻量 World-Model 化稳定性预测"
lang: zh-CN
geometry: margin=1in
mainfont: "Noto Sans CJK SC"
CJKmainfont: "Noto Sans CJK SC"
---

# Force-Slip Decoder Phase 3 建议：轻量 World-Model 化稳定性预测

> 本文档基于同目录下的 `1-force-slip-decoder.md`、`force-slip-decoder-phase1.md`、`force-slip-decoder-phase2.md` 和当前 phase1/diagnostic 报告整理，目标不是替代 A/B/C baseline，而是在**尽量少改架构**的前提下，为后续实验增加一条具有 world model 味道的稳定性预测路径。

## 1. 当前结论与推荐方向

从现有结果看，当前系统已经具备继续向“稳定性预测”推进的条件：

- force 轴语义已基本明确，`Fz` 可作为 normal axis；
- trajectory-level split 已冻结，避免 sample-level 泄漏；
- slip 对齐审计通过，flat/sharp/sphere 的 slip 标签与 shear 变化具备一致性；
- A/B/C 的门禁已经定义，主指标和 consistency diagnostic 都有统一口径；
- 最新 all-source slip report 表明：all-source 可作为诊断分支，但**不应自动替代 sphere-only 主基线**；
- MAE 与 DINOv2 的对比显示，MAE 在当前设置下整体更强，适合作为第一版主线 backbone。

**推荐结论：**

> 在保持 Sparsh encoder 冻结、共享 decoder 结构基本不变的前提下，优先增加一个 **one-step latent predictor + stability head**，把当前的 force/slip 联合建模升级为“未来稳定性预测”。

这比直接上完整 RSSM / Dreamer / diffusion 更轻，也比单纯 current slip classifier 更有 world model 特征。

---

## 2. 后续实验建议：推荐推进顺序

### 2.1 第一优先级：锁定 A/B/C 基线

先确保下列事项保持不变：

1. 轨迹级划分固定，train/val/test 不混轨迹；
2. force 指标统一在 Newton 单位上计算；
3. `Fn = abs(Fz_N)`、`Ft = sqrt(Fx_N^2 + Fy_N^2)` 作为默认物理映射；
4. B/C 的主指标门禁继续沿用当前定义；
5. C 的 consistency 只在 B 不触发 hard fail 时作为正式候选。

**目的：**避免后续 world model 扩展把 baseline 解释搞乱。

### 2.2 第二优先级：轻量 one-step future predictor

推荐实验命名为：

- `E1: shared decoder + one-step latent predictor`
- 或 `E1: stability-aware latent predictor`

核心是把当前状态扩展为：

```text
s_t = [z_t, Fn_t, Ft_t, p_slip_t, Δforce_t, optional proprio_t]
```

其中 `Δforce_t` 是强烈建议加入的条件项，因为它天然接近 action surrogate / transition cue。

### 2.3 第三优先级：再考虑 recurrent belief state

如果 one-step predictor 已经能带来稳定性收益，再升级为：

- GRU belief state
- temporal transformer belief state
- multi-step rollout

这一步更像完整 world model，但不建议作为第一版起点。

---

## 3. 推荐网络架构

### 3.1 方案 A：最小改动版稳定性打分头

适合先验证“force + slip 是否足以形成稳定性 proxy”。

```text
tactile image_t
    ↓
Sparsh encoder (frozen)
    ↓
z_t
    ↓
shared trunk
    - force head → Fn_t, Ft_t, Fmag_t
    - slip head  → p_slip_t
    - stability head → p_stable_t / risk_t
```

推荐输入给 stability head 的特征：

```text
h_t = MLP([z_t, Fn_t, Ft_t, p_slip_t, q_t, Δforce_t])
```

其中：

```text
r_t = Ft_t / (Fn_t + eps)
q_t = sigmoid(alpha * (r_t - tau))
```

### 3.2 方案 B：推荐主线，one-step latent predictor

这是当前最推荐的结构。

```text
tactile image_t                 tactile image_{t+1}
    ↓                                 ↓
Sparsh encoder (frozen)         Sparsh encoder (frozen, target)
    ↓                                 ↓
z_t                           z_{t+1}^target
    ↓
shared trunk
    - force head → Fn_t, Ft_t
    - slip head  → p_slip_t
    - future head → ẑ_{t+1}, p̂_slip_{t+1}, ŝ_{t+1}
```

**核心做法：**

- 当前分支负责 force/slip；
- 未来分支负责下一时刻 latent 与 stability；
- 未来 latent 的监督来自下一帧 Sparsh embedding；
- future slip / stability 都从未来 latent 再解码得到。

### 3.3 方案 C：后续增强版 recurrent belief

```text
z_{t-k:t}, Fn_{t-k:t}, Ft_{t-k:t}, p_slip_{t-k:t}
    ↓
GRU / causal transformer
    ↓
b_t
    - current heads
    - future heads
```

适用于后续想做多步预测、滑移前兆、或更像 Dreamer 风格的 rollout。

---

## 4. 推荐训练流程

### Step 1：固定数据与评估口径

- 固定 trajectory-level split；
- 固定 force 反归一化逻辑；
- 固定 `Ft/Fn` 的 train/val 阈值；
- 固定 slip 评价阈值与 contradiction rate 定义。

### Step 2：先跑稳定性 proxy baseline

先只训练：

- force head
- slip head
- stability head

目标是先判断：

> 当前状态 + force/slip 是否能直接预测未来稳定性。

### Step 3：加入 one-step latent predictor

训练目标建议写成：

```text
L = L_force
  + λ_slip * L_slip
  + λ_dyn  * SmoothL1(ẑ_{t+1}, stopgrad(z_{t+1}))
  + λ_future_slip * BCE(p̂_slip_{t+1}, slip_{t+1})
  + λ_stab * BCE(ŝ_t, y_stable_t)
```

其中：

```text
y_stable_t = 1 - max(slip_{t+1:t+H})
```

### Step 4：做小步消融

最少建议做以下消融：

1. only force；
2. force + slip；
3. force + slip + stability head；
4. force + slip + Δforce；
5. force + slip + future latent；
6. force + slip + future latent + stability head。

---

## 5. 需要注意的点

### 5.1 不要把 stability 直接写成 grasp success

当前数据没有显式 grasp success label，因此论文里更稳妥的写法是：

- future slip-free probability
- instability risk
- grasp stability proxy

### 5.2 不要一开始就上完整 RSSM / diffusion

这些方法更“重”，但第一版很可能把时间花在工程和调参上。

当前最优折中是：

> **共享 decoder + one-step latent predictor + stability head**

### 5.3 `delta_force` 很值得用

现有数据里已经有 `delta_force`，这相当于免费的 temporal cue。  
如果不把它用进去，world model 的味道会弱很多。

### 5.4 仍然要保留 force-slip consistency

建议继续保留：

```text
r = Ft / (Fn + eps)
q = sigmoid(alpha * (r - tau))
```

因为这能让 current consistency 和 future stability 两条线同时成立。

### 5.5 encoder 建议优先 MAE 主线

基于当前 report：

- MAE 当前整体更强；
- DINOv2 all-source 更适合作为诊断分支；
- 如果只选一个主线，优先 MAE frozen。

---

## 6. 推荐评估指标

### 当前任务指标

- Force：RMSE / MAE
- Slip：Accuracy / Precision / Recall / F1 / AUROC / AUPRC
- Consistency：contradiction rate、monotonic calibration error、high-ratio slip recall、low-ratio false alarm

### 未来预测指标

- Future slip AUROC / F1
- Future stability AUROC / AUPRC
- Lead time to slip onset
- Stability calibration error
- Multi-step roll-out error（若后续加入）

---

## 7. 推荐的论文叙事

可以把这一阶段写成：

> 我们在 Sparsh 表征上构建了一个轻量级触觉世界模型。模型不仅联合预测当前接触力与滑移状态，还通过一个 one-step latent predictor 预测下一时刻的触觉潜状态，并基于未来滑移概率输出抓取稳定性评分。

这条叙事比“普通 slip classifier”更强，也比“完整控制 world model”更容易落地。

---

## 8. 一句话总结

**最推荐的 Phase 3 路线：**

> 保持 A/B/C baseline 不动，在共享 decoder 上加一个 one-step latent predictor 和 stability head，用 `delta_force` 作为 transition cue，把当前 force-slip 联合感知升级为未来抓取稳定性预测。
