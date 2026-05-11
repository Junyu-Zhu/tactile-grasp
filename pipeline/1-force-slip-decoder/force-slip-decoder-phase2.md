# Force-Slip Decoder Phase 1/2 执行步骤

> 本文件从 RALPLAN 审查通过的计划中拆分生成。B/C 实现代码必须先在本地 `/home/zjy/Documents/grasp/tactile_grasp/sparsh-force-slip` 修改、提交并 push 到 GitHub，再在 `zjy-4090` 的 `/documents/tactile_grasp` pull；训练/评估从 `/documents/tactile_grasp/sparsh-force-slip` 执行，远程资源位于 `/vla1/zjy/`。

## 3. Phase 2：实现/训练 B 与 C、结果选择与本地回传

### Step 1：实现并训练 B shared multitask decoder
- **目标**：验证共享下游表示是否优于 separate baseline。
- **远程执行内容**：
  - 若需要修改 B 的模型、config、训练入口或指标脚本，必须先在本地 `sparsh-force-slip/` 修改并 push；服务器只执行 `git pull --ff-only origin main` 后运行。
  - 进入服务器运行目录：`ssh zjy-4090 && cd /documents/tactile_grasp && git pull --ff-only origin main && cd sparsh-force-slip`。
  - Sparsh encoder frozen；下游使用 shared pooler/trunk + force head + slip head。
  - force head 输出 signed `Fx,Fy,Fz`；评估时派生 `Fn/Ft/Fmag`。
  - loss：`L = L_force + lambda_slip * L_slip`，不加 consistency loss。
- **产物**：B 模型代码/config、训练日志、checkpoint、B 指标表。
- **验证标准**：相对 A，force RMSE 增加不超过 **10%**，且 slip F1 下降不超过 **2 个百分点**；若 force RMSE 增加处于 **5%–10%** 或 slip F1 下降处于 **1–2 个百分点**，标记为 warning band。若 B 相对 A 的 force RMSE 增加 **>10%** 或 slip F1 下降 **>2 个百分点**，这是进入 C 的 hard fail 硬门禁：C 只能作为 diagnostic run，不允许作为 formal claim 或最终方法改进结论。

### Step 2：B 诊断与超参定版
- **目标**：为 C 的 consistency 参数选择提供 train/val 依据。
- **远程执行内容**：
  - 复用 A 的 diagnostic 指标，比较 B 与 A。
  - 从 train/val 的 `Ft/Fn` 分布确定 C 的候选 `tau`：50%、65%、80% 分位数。
  - 固定 `alpha ∈ {5,10,20}`、`beta_cons ∈ {0.01,0.05,0.1}` 搜索范围。
  - 将 `slip_horizon` 纳入 train/val sweep，或在进入正式 test 前基于数据采样率、标签定义和 train/val 诊断固定唯一取值；禁止使用 test set 调整 `slip_horizon`。
- **产物**：B diagnostic report、C sweep plan、候选参数表、`slip_horizon` 选择依据。
- **验证标准**：`tau/alpha/beta_cons/slip_horizon` 只由 train/val 决定；test set 不参与调参。

### Step 3：实现并训练 C consistency decoder
- **目标**：在 B 基础上加入 force-slip consistency，检验物理一致性是否改善。
- **远程执行内容**：
  - 若需要修改 C 的 consistency loss、超参 sweep、metrics 或 launch script，必须先在本地 `sparsh-force-slip/` 修改并 push；服务器 pull 后再运行。
  - 先检查 B→C 硬门禁：若 B 相对 A force RMSE 增加 **>10%** 或 slip F1 下降 **>2 个百分点**，本步骤只能运行 diagnostic C，不进入正式 A/B/C claim；若 B 落入 warning band，则可继续 C，但报告中必须标记风险。
  - 计算 `r = Ft_pred / (Fn_pred + eps)`。
  - 计算 `q = sigmoid(alpha * (r - tau))`。
  - 第一版使用 `L_cons = BCE(p_slip, q.detach())`，主要约束 slip head，避免过强干扰 force regression。
  - 总损失：`L = L_force + lambda_slip * L_slip + beta_cons * L_cons`。
  - 在 train/val 上 sweep `tau/alpha/beta_cons` 以及已纳入 sweep 的 `slip_horizon`；若 `slip_horizon` 已按 Step 2 固定，则记录固定依据。选择最优候选后只在 test 上最终评估。
- **产物**：C 模型代码/config、sweep 结果、checkpoint、C 指标表。
- **验证标准**：主指标不触发 hard fail：force RMSE 增加不超过 **10%**，slip F1 下降不超过 **2 个百分点**；若落入 warning band，最终报告必须标记风险。若 B 已触发硬门禁，C 结果必须标记为 diagnostic-only，不得写成 formal improvement claim。

### Step 4：C 成功判定与模型选择
- **目标**：决定最终交付 A/B/C 哪个 checkpoint。
- **远程执行内容**：
  - 比较 A/B/C 的主指标与 consistency diagnostic。
  - C 成功需满足：按统一 metrics 脚本定义的 contradiction rate 相对 B 下降 **≥10%**，或 monotonic calibration error 下降 **≥5%**。
  - 同时高 `Ft/Fn` 区域 slip recall 下降不超过 **1 个百分点**；低 `Ft/Fn` 区域 false alarm 上升不超过 **1 个百分点**。
- **产物**：最终模型选择报告、A/B/C 汇总表、negative result 记录（如适用）。
- **验证标准**：若 C 主指标越线或 consistency 未达标，不称为成功；保留 B 或 A 作为可交付 fallback。

### Step 5：checkpoint 回传与本地集成冒烟测试
- **目标**：只回传必要产物，验证能被 `tactile_grasp` 加载和调用。
- **远程执行内容**：
  - 从 `/vla1/zjy/sparsh_runs/experiments/<run_name>/` 选择 checkpoint、config、split、metrics summary、模型版本信息。
  - `scp` 到本地：`tactile_grasp/checkpoints/force_slip_decoder/<run_name>/`。
  - 本地记录 `source.txt`：server、remote_repo、remote_code_dir、remote_run、remote_checkpoint、git_commit、copied_at。
- **产物**：本地 checkpoint 包、source.txt、本地加载/推理 smoke test 日志。
- **验证标准**：checkpoint 可加载；单 batch/样例推理接口可与 `tactile_grasp` 对接；正式指标仍以远程统一评估为准。

## 2b. 统一指标脚本与硬门禁定义

正式 A/B/C 比较必须使用同一个远程 metrics 脚本，且在 run config 或 metrics config 中固定以下定义，避免执行时解释不一致。

### B/C 主指标门禁

- **warning band（需要记录风险但可继续）**：相对 A 或 B，force RMSE 增加 `5%–10%`，或 slip F1 下降 `1–2 个百分点`。
- **hard fail（正式结论失败）**：相对 A 或 B，force RMSE 增加 `>10%`，或 slip F1 下降 `>2 个百分点`。
- **B→C 硬门禁**：若 B 相对 A 触发 hard fail，则 C 只能作为 diagnostic-only run，不能作为 formal A/B/C claim。
- **C 成功主任务条件**：C 相对 B 不触发 hard fail；若落入 warning band，最终报告必须标注风险并优先解释主指标退化来源。

### consistency diagnostic 固定定义

- `Ft/Fn` 全部使用 Newton 单位下的 force，并使用 train/val 分布确定阈值；test set 只按已冻结阈值评估。
- **high Ft/Fn 区域**：`Ft/Fn >= train_val_ratio_p80`；**low Ft/Fn 区域**：`Ft/Fn <= train_val_ratio_p20`。
- **slip 概率阈值**：`p_slip >= 0.5` 判为 slip，`p_slip < 0.5` 判为 no-slip；如需改阈值，只能在 train/val 上固定并写入 config。
- **contradiction rate**：
  - high-ratio contradiction：high Ft/Fn 区域中模型判为 no-slip 的比例；
  - low-ratio contradiction：low Ft/Fn 区域中模型判为 slip 的比例；
  - 总 contradiction rate 为两者按样本数加权平均。
- **monotonic calibration error**：按已冻结的 `Ft/Fn` bin 边界把样本分成 10 个等频 bin，计算每个 bin 的平均 `Ft/Fn` 和平均 `p_slip`；若 bin 的平均 `Ft/Fn` 递增但平均 `p_slip` 下降，则把下降幅度累计为 violation，最终除以 bin 数得到归一化误差。A/B/C 共用同一 bin 边界。
- **高低 ratio 辅助指标**：报告 high Ft/Fn 区域 slip recall、low Ft/Fn 区域 false alarm；C 相对 B 的 high-ratio slip recall 下降不得超过 1 个百分点，low-ratio false alarm 上升不得超过 1 个百分点。
