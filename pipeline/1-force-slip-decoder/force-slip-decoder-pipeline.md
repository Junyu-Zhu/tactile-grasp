# Force-Slip Decoder 远程训练执行流程

本文是对 `1-force-slip-decoder.md` 的执行版总结，已按当前约定改为 **服务器训练优先** 的流程。

核心约定：

> 后续所有 force-slip 相关代码修改和训练都在服务器 `zjy-4090` 上进行。服务器项目根目录固定为 `/home/zjy/document`，`tactile_grasp` 仓库位于 `/home/zjy/document/tactile_grasp`，force-slip 代码位于 `/home/zjy/document/tactile_grasp/sparsh-force-slip`。相关任务统一在 `tactile_grasp` 的 `sparsh-force-slip` 分支上工作并 commit；`git push` 和 `git pull` 由用户本人手动操作。每完成一个 phase 后，必须在该分支进行一次整体 commit。

---

## 1. 总体目标

该 pipeline 的目标是基于 Sparsh tactile encoder，训练一个 force/slip 联合感知模型：

```text
tactile image / image sequence
    ↓
Sparsh encoder
    ↓
decoder / task head
    ↓
force prediction + slip prediction
```

实验上分为三类模型：

```text
A: separate force/slip baseline
B: shared multitask decoder
C: shared decoder + force-slip consistency loss
```

推荐主线仍然是：

```text
服务器项目路径与分支确认
  ↓
远程环境与路径确认
  ↓
远程数据加载冒烟测试
  ↓
force 轴语义与 Newton 单位确认
  ↓
trajectory-level 数据划分
  ↓
A: separate force/slip baseline 重评或重训
  ↓
B: shared multitask decoder
  ↓
C: shared decoder + force-slip consistency loss
  ↓
训练完成后按需 scp checkpoint 到本地测试
```

不要一开始直接训练 C。必须先固定 force 单位、normal 轴、数据划分和 baseline，否则 C 的提升可能不可比较或不可解释。

---

## 2. 服务器代码、分支与训练资源约定

### 2.1 服务器连接方式

训练服务器：

```bash
ssh zjy-4090
```

后续训练、重评 baseline、导出 checkpoint、整理训练日志，默认都在该服务器上完成。

### 2.2 服务器项目路径与 force-slip 代码目录

服务器项目根目录：

```text
/home/zjy/document
```

服务器上的 `tactile_grasp` 仓库：

```text
/home/zjy/document/tactile_grasp
```

force-slip 代码目录：

```text
/home/zjy/document/tactile_grasp/sparsh-force-slip
```

关于 force-slip 阶段的代码必须集中在 `sparsh-force-slip/`：

- decoder/model/head 实现；
- dataloader 或 dataset wrapper；
- split / manifest / preprocessing 诊断脚本；
- train / eval / metrics 入口；
- Hydra/config 或 shell 启动脚本；
- runbook、实验记录模板和结果整理脚本。

`pipeline/1-force-slip-decoder/` 只保存流程文档和约束，不作为训练代码目录。

### 2.3 分支、commit 与手动 push/pull 规则

后续 force-slip 任务统一在 `tactile_grasp` 仓库的 `sparsh-force-slip` 分支上进行：

```bash
ssh zjy-4090
cd /home/zjy/document/tactile_grasp
git status
git switch sparsh-force-slip
```

若分支尚不存在，由用户/维护者在服务器仓库中创建 `sparsh-force-slip` 分支。相关任务的 commit 均在该分支上进行，提交信息按 AGENTS.md 的 Lore Commit Protocol 填写。

`git push` 和 `git pull` 改为用户本人手动操作：pipeline、训练脚本或 agent 不自动 push/pull，也不把 push/pull 作为训练前后默认步骤。若需要同步远端代码或发布结果，由用户手动执行并记录对应 commit。

新增 phase 级提交规则：

- 每完成一个 phase（例如 Phase 1、Phase 2 或后续新增 phase）并完成该 phase 的验证后，必须进行一次整体 commit。
- 该整体 commit 应覆盖本 phase 的代码、config、pipeline 文档、运行脚本、实验记录模板、指标汇总或 source 记录等相关改动。
- 若 phase 未通过验证，不提交“完成”性质的 commit；可按需要提交明确标注 diagnostic / failed-attempt 的记录性 commit。

### 2.4 服务器运行入口

后续训练、重评 baseline、导出 checkpoint、整理训练日志，默认都在服务器上从以下目录执行：

```bash
ssh zjy-4090
cd /home/zjy/document/tactile_grasp
git switch sparsh-force-slip
cd sparsh-force-slip
```

若服务器上已有 `sparsh` conda 环境，训练和评估应优先使用该环境。实际执行前需要在服务器上确认：

```bash
which python
python -V
python -c "import torch, hydra, omegaconf; print(torch.__version__)"
```

### 2.5 服务器上的数据、base model 和训练权重

训练所需资源统一放在：

```text
/vla1/zjy/
```

约定结构：

```text
/vla1/zjy/tactile_datasets      # 各类 tactile datasets
/vla1/zjy/sparsh_models         # Sparsh base encoder models
/vla1/zjy/sparsh_runs           # 已完成或后续训练输出
```

其中：

- 训练数据集放在 `/vla1/zjy/tactile_datasets`；
- Sparsh base model 放在 `/vla1/zjy/sparsh_models`；
- 已完成训练权重和后续训练结果放在 `/vla1/zjy/sparsh_runs`。

本地路径如 `/home/zjy/Documents/dataset1/sparsh/...` 可以作为历史参考，但不再作为训练默认路径。

### 2.6 推荐远程 paths 配置

服务器上应使用指向 `/vla1/zjy` 的 paths 配置，例如：

```yaml
data_root: /vla1/zjy/tactile_datasets
encoder_checkpoint_root: /vla1/zjy/sparsh_models
log_dir: /vla1/zjy/sparsh_runs/experiments
tacbench_dir: /vla1/zjy/sparsh_runs/tacbench

output_dir: ${hydra:runtime.output_dir}
work_dir: ${hydra:runtime.cwd}
```

若仓库中已有类似 `config/paths/zjy_4090.yaml`，后续训练命令应显式使用：

```text
paths=zjy_4090
```

---

## 3. 本地与远程的职责划分

### 3.1 远程负责

服务器 `zjy-4090` 负责：

- 在 `/home/zjy/document/tactile_grasp` 的 `sparsh-force-slip` 分支上进行 force-slip 代码修改、提交和实验记录；
- 从 `/home/zjy/document/tactile_grasp/sparsh-force-slip` 运行 force-slip 训练/评估；
- 数据加载冒烟测试；
- force 轴语义统计；
- trajectory-level split 生成；
- A/B/C 模型训练；
- baseline 重评；
- checkpoint 保存；
- 训练日志与指标表生成；
- 每个 phase 完成并验证后执行一次整体 commit。

### 3.2 本地负责

本地 `/home/zjy/Documents/grasp/tactile_grasp` 负责：

- 可保存从服务器同步回来的 pipeline 文档副本、分析结论和 checkpoint；
- 保存从服务器拷贝回来的 checkpoint；
- 做轻量测试、集成测试或 `tactile_grasp` 侧的验证；
- 不作为后续 force-slip 代码开发和训练的默认位置。

### 3.3 用户手动 Git 操作

以下操作由用户本人手动执行，不由 pipeline、训练脚本或 agent 自动执行：

- `git pull`
- `git push`
- 远端分支发布、同步和冲突处理

phase 完成后的整体 commit 仍应发生在服务器 `/home/zjy/document/tactile_grasp` 的 `sparsh-force-slip` 分支上；如当时由用户手动接管提交，执行者必须至少整理好待提交文件、验证结果和符合 Lore Commit Protocol 的提交信息草案。

### 3.4 checkpoint 回传策略

训练完成后，再按需要从服务器拷贝 checkpoint 到本地。

示例：

```bash
scp zjy-4090:/vla1/zjy/sparsh_runs/experiments/<run_name>/checkpoints/<ckpt>.pth \
    /home/zjy/Documents/grasp/tactile_grasp/checkpoints/force_slip_decoder/
```

如果需要连同配置和指标一起保存，建议同时拷贝：

```bash
scp zjy-4090:/vla1/zjy/sparsh_runs/experiments/<run_name>/config.yaml \
    /home/zjy/Documents/grasp/tactile_grasp/checkpoints/force_slip_decoder/<run_name>-config.yaml

scp zjy-4090:/vla1/zjy/sparsh_runs/experiments/<run_name>/wandb/*/files/wandb-summary.json \
    /home/zjy/Documents/grasp/tactile_grasp/checkpoints/force_slip_decoder/<run_name>-summary.json
```

本地测试前必须记录 checkpoint 来源：

```text
server: zjy-4090
remote_run: /vla1/zjy/sparsh_runs/experiments/<run_name>
checkpoint: checkpoints/<ckpt>.pth
config: config.yaml
```

---

## 4. 数据加载冒烟测试

数据加载冒烟测试指：

> 只检查数据能不能被 dataloader 正常读出来，字段和张量形状是否符合预期；不训练模型，也不声称模型效果。

该测试应在服务器上执行。

预期至少确认：

```text
image: [6, 320, 240]
force: [3]
delta_force: [3]
slip_label: scalar
force_scale: [3]
```

需要检查的数据主线是 GSmini force/slip 数据，例如：

```text
/vla1/zjy/tactile_datasets/Gelsight-mini/gelsight-force-estimation
```

如果路径或目录命名与实际服务器不同，以 `/vla1/zjy/tactile_datasets` 下的实际目录为准，并把最终路径写入实验记录。

---

## 5. force 轴语义和 Newton 单位确认

这是整个 pipeline 的关键前置条件。

### 5.1 为什么必须确认单位

`VisionForceSlipDataset` 返回的 `force` 通常是归一化后的 force：

```text
force_norm = force_N / force_scale
```

因此后续所有物理量必须先反归一化：

```text
force_N = force_norm * force_scale
```

不能直接在归一化 force 上计算 `Ft/Fn`，否则 shear-normal ratio 会被不同轴的 scale 扭曲。

### 5.2 normal 轴不能直接假设

原始文档默认：

```text
Fn = abs(Fz)
Ft = sqrt(Fx^2 + Fy^2)
Fmag = sqrt(Fx^2 + Fy^2 + Fz^2)
```

但正式实验前必须先验证 GSmini 数据中哪个轴对应 normal loading。

服务器上需要统计：

1. `Fx/Fy/Fz` 在 flat / sharp / sphere 各 batch 中的分布；
2. 哪个轴随接触加载最稳定变化；
3. slip 发生前后各轴变化；
4. normal axis 映射是否与数据文档一致。

如果确认 `Fz` 是 normal，则使用：

```text
Fn = abs(Fz_N)
Ft = sqrt(Fx_N^2 + Fy_N^2)
```

如果不是，则替换为正确轴映射。

如果无法确认稳定 normal 轴，则 C 阶段不能做严格 friction-style consistency，只能降级为 signed-force multitask diagnostic。

---

## 6. trajectory-level 数据划分

正式比较不能依赖普通 sample-level random split。

原因是 force/slip 数据是时间序列。同一条滑动轨迹中的相邻帧非常相似，如果随机拆到 train 和 test，会造成数据泄漏。

正式实验应使用：

```text
trajectory-level split
```

即按整条 trajectory 划分：

```text
train trajectories
val trajectories
test trajectories
```

Sparsh dataset 中已有：

```text
idx2traj
traj2idx
```

可以利用这些信息做 deterministic split。

服务器上应持久化保存：

- split 文件；
- 每个 split 的 trajectory 数量；
- 每个 split 的 sample 数量；
- slip 正负样本数量；
- force 分布；
- `Ft/Fn` 分布。

这些 split 文件也应随 run 一起保存在：

```text
/vla1/zjy/sparsh_runs/experiments/<run_name>/
```

---

## 7. A — separate force/slip baseline

A 是独立基线：

```text
Sparsh encoder → force decoder
Sparsh encoder → slip decoder
```

force 和 slip 各自训练、各自评估，下游 decoder 不共享。

A 的作用是提供后续 B 和 C 的公平对照。

服务器上已有的 DINOv2 / MAE force/slip 训练结果可以作为参考，但正式 A baseline 必须在新的 trajectory-level split 和统一指标脚本下重评或重训。

### 7.1 force 指标

应报告：

```text
Fx / Fy / Fz RMSE
Fx / Fy / Fz MAE
Fn RMSE / MAE
Ft RMSE / MAE
Fmag RMSE / MAE
```

所有 force 指标都应在 Newton 单位下计算。

### 7.2 slip 指标

应报告：

```text
Accuracy
Precision
Recall
F1
AUROC
AUPRC
```

### 7.3 初始一致性诊断

即使 A 是 separate baseline，也可以计算诊断指标：

```text
高 Ft/Fn 区域 slip recall
低 Ft/Fn 区域 false alarm
p_slip 与 Ft/Fn 的单调关系
contradiction rate
```

这些指标用于后续判断 B/C 是否真的改善了 force-slip 物理一致性。

---

## 8. B — shared multitask decoder

B 是共享多任务 decoder。

中文理解为：

> force 和 slip 共用同一个 Sparsh encoder 后的下游共享层，然后分别接 force head 和 slip head。

结构：

```text
tactile image / image sequence
    ↓
Sparsh encoder, frozen
    ↓
shared pooler / shared trunk
    ↓
 ┌──────────────┬──────────────┐
 force head      slip head
 Fx,Fy,Fz        slip logits
```

B 的 loss 是：

```text
L = L_force + lambda_slip * L_slip
```

B 不加入 consistency loss。

B 要回答的问题是：

> 仅仅让 force 和 slip 共享 tactile representation，是否已经比 separate decoders 更好？

第一版建议 force head 仍输出 signed force：

```text
Fx, Fy, Fz
```

然后在评估时派生：

```text
Fn, Ft, Fmag
```

这样更容易和 A 的 force-only baseline 对齐。

B 的警戒线：

```text
force RMSE 增加超过 5–10%
slip F1 下降超过 1–2 个百分点
```

如果越过警戒线，需要标记为明显退化，并分析是否值得继续做 C。

---

## 9. C — consistency decoder

C 是在 B 的基础上加入 force-slip consistency loss。

直觉是：

```text
Ft/Fn 高 → 更容易 slip
Ft/Fn 低 → 不应该频繁 slip
```

计算方式：

```text
r = Ft_pred / (Fn_pred + eps)
q = sigmoid(alpha * (r - tau))
```

其中：

- `r`：预测出来的切向力 / 法向力比例；
- `tau`：类似摩擦阈值；
- `alpha`：控制 sigmoid 变化陡峭程度；
- `q`：由 force 预测推导出的 slip tendency。

第一版推荐：

```text
L_cons = BCE(p_slip, q.detach())
```

这样主要约束 slip head，不让 consistency loss 过强地干扰 force regression。

总损失：

```text
L = L_force + lambda_slip * L_slip + beta_cons * L_cons
```

### 9.1 参数选择

`tau` 和 `alpha` 只能用 train/val 选择，不能在 test set 上调。

建议预设：

```text
tau: train Ft/Fn 分布的 50%、65%、80% 分位数
alpha: 5, 10, 20
beta_cons: 0.01, 0.05, 0.1
```

所有最终选择都要写入 run report。

### 9.2 C 是否成功的判断标准

C 不能只看 consistency 变好，还必须保证 force/slip 主指标不明显变差。

主任务警戒线：

```text
force RMSE 不增加超过 5–10%
slip F1 不下降超过 1–2 个百分点
```

一致性成功条件：

```text
contradiction rate 相对 B 下降 ≥ 10%
或 monotonic calibration error 下降 ≥ 5%
```

同时：

```text
高 Ft/Fn 区域 slip recall 不下降超过 1 个百分点
低 Ft/Fn 区域 false alarm 不上升超过 1 个百分点
```

如果 C 达不到这些条件，则不能称为成功，应记录为 diagnostic 或 negative result。

---

## 10. Phase5 仿真数据定位

`tactile_grasp` 中已有 Phase5 cube force/slip export，可以作为辅助诊断。

但该数据的 force 语义是：

```text
[tangent_a_n, tangent_b_n, normal_n] = [0, 0, normal_force_n]
```

即切向力为 0。

因此 Phase5 sim 只能用于：

- 数据加载检查；
- normal force 诊断；
- slip valid mask 检查；
- 仿真数据通路验证。

不能用于证明：

- shear-normal consistency；
- friction cone consistency；
- sim2real 效果；
- 论文级方法提升。

如果后续需要用 Phase5 sim 做服务器侧诊断，应先把相应数据同步到 `/vla1/zjy` 下，并在报告中明确标注为 diagnostic-only。

---

## 11. 风险处理

### 风险 1：force label 没有明确 normal/shear 分量

处理方式：

- Real GSmini：可通过三轴 force 和 axis semantic validation 解决；
- Phase5 sim：不能解决，因为切向力为 0，只能降级为 weak diagnostic；
- 若无法确认 normal 轴，不做 friction-style `Ft/Fn` claim。

### 风险 2：slip label 与 force label 时间不同步

处理方式：

- 使用 `slip_horizon` 做 current/future window；
- 在 train/val 上 sweep：

```text
slip_horizon ∈ {0, 1, 2, 3, 5}
```

- 比较 F1、AUROC、AUPRC、onset latency、contradiction rate；
- test set 只评估，不调参。

### 风险 3：consistency loss 降低 force 或 slip 主指标

处理方式：

- B 作为无 consistency 的 fallback；
- C 使用小 `beta_cons`；
- C 使用 `q.detach()`；
- 必要时只在高置信 force 区域启用 consistency；
- 若主指标越过警戒线，则不把 C 称为成功。

---

## 12. 服务器训练后的本地测试流程

训练完成后，不建议立即把所有 run 目录拷回本地。推荐只拷贝需要测试的最小集合：

```text
checkpoint
config.yaml
split 文件
metrics summary
必要的模型定义版本信息
```

推荐本地保存结构：

```text
tactile_grasp/checkpoints/force_slip_decoder/<run_name>/
  checkpoint.pth
  config.yaml
  split.json
  metrics.json
  source.txt
```

`source.txt` 中记录：

```text
server: zjy-4090
remote_project_root: /home/zjy/document
remote_repo: /home/zjy/document/tactile_grasp
remote_branch: sparsh-force-slip
remote_code_dir: /home/zjy/document/tactile_grasp/sparsh-force-slip
remote_run: /vla1/zjy/sparsh_runs/experiments/<run_name>
remote_checkpoint: checkpoints/<ckpt>.pth
git_commit: <commit used on server>
copied_at: <timestamp>
```

本地测试只用于确认 checkpoint 能否被加载、推理接口是否能与 `tactile_grasp` 对接；正式训练指标仍以服务器上的统一评估结果为准。

---

## 13. 最小交付物

第一阶段最小交付应包括：

1. 服务器 paths config、`sparsh-force-slip` 分支状态和 phase 级整体 commit 记录；
2. 服务器数据加载冒烟测试结果；
3. force 轴语义和单位报告；
4. trajectory-level split 文件；
5. A baseline 指标表；
6. B shared multitask decoder 指标表；
7. C consistency decoder 指标表；
8. Phase5 sim diagnostic-only 报告；
9. DINOv2 / MAE 对照表；
10. 需要本地测试时，从服务器 scp 回来的 checkpoint 和对应 config。

推荐最终表格结构：

```text
DINOv2 + A separate
DINOv2 + B shared multitask
DINOv2 + C consistency

MAE + A separate
MAE + B shared multitask
MAE + C consistency
```

---

## 14. 后续扩展

只有当 C 在主指标不明显退化的情况下改善 consistency，才进入扩展阶段：

```text
partial encoder finetuning
learnable friction threshold / mu_eff
high-confidence consistency mask
sim2real diagnostic
```

如果 C 不成立，则保留 B 作为可行 deliverable，并记录 fixed-threshold consistency 在当前标签设置下不成立。

---

## 15. 一句话总结

> 后续训练和代码修改统一在 `zjy-4090` 上进行；服务器项目根目录为 `/home/zjy/document`，仓库为 `/home/zjy/document/tactile_grasp`，force-slip 代码从 `/home/zjy/document/tactile_grasp/sparsh-force-slip` 运行。相关任务统一在 `sparsh-force-slip` 分支上 commit；`git push` 和 `git pull` 由用户本人手动操作；每完成一个 phase 后必须进行一次整体 commit。数据、base model、训练输出仍使用 `/vla1/zjy`。先用真实 GSmini 数据固定 force 轴、Newton 单位、轨迹级划分和 A baseline；再训练 B；最后谨慎训练 C。训练完成后，再按需把 checkpoint 从服务器 scp 回本地做测试。Phase5 仿真数据只作为辅助诊断，不能作为 friction consistency 的主证据。
