# Force-Slip Decoder Phase 1/2 执行步骤


> 本文件从 RALPLAN 审查通过的计划中拆分生成。当前执行约定：force-slip 代码修改、派生数据生成、训练和评估统一在 `zjy-4090` 执行；服务器项目根目录 `/home/zjy/document`，仓库 `/home/zjy/document/tactile-grasp`，代码目录 `/home/zjy/document/tactile-grasp/sparsh-force-slip`；相关任务在 `sparsh-force-slip` 分支上进行；`git push`/`git pull` 由用户本人手动操作；每完成一个 phase 后必须进行一次整体 commit；远程资源位于 `/vla1/zjy/`。

## 2. Phase 1：数据、环境、验证、基线可比性准备

### Step 1：远程环境与路径确认
- **目标**：确认训练和代码修改只在 `zjy-4090` 执行，并固定服务器项目路径、`sparsh-force-slip` 分支、force-slip 代码目录、数据、模型、输出路径。
- **远程执行内容**：
  - `ssh zjy-4090 && cd /home/zjy/document/tactile-grasp`。
  - 确认当前分支为 `sparsh-force-slip`；若分支尚不存在，由用户/维护者在服务器仓库中创建。
  - 进入 `/home/zjy/document/tactile-grasp/sparsh-force-slip` 后再运行数据检查、训练和评估命令。
  - `git push` 和 `git pull` 由用户本人手动操作；pipeline/agent 不自动执行。
  - 检查 `which python`、`python -V`、`torch/hydra/omegaconf` import。
  - 确认 `/vla1/zjy/tactile_datasets`、`/vla1/zjy/sparsh_models`、`/vla1/zjy/sparsh_runs` 存在。
  - 建立或确认 `paths=zjy_4090` 指向 `/vla1/zjy`。
- **产物**：环境记录、服务器路径记录、分支状态记录、paths config、资源目录清单。
- **验证标准**：服务器仓库位于 `/home/zjy/document/tactile-grasp`；当前分支为 `sparsh-force-slip`；训练入口位于 `/home/zjy/document/tactile-grasp/sparsh-force-slip`；训练命令可显式使用 `paths=zjy_4090`；日志/输出默认进入 `/vla1/zjy/sparsh_runs`。

### Step 2：数据加载冒烟测试与 training data manifest/hash
- **目标**：确认官方 Sparsh 下游训练主线读取 `dataset_gelsight_*.pkl`，不是 `org_dataset_gelsight_*.pkl`；同时把“实际训练加载了什么”固化为可审计 manifest。
- **远程执行内容**：
  - 对 GSmini force/slip 数据执行 dataloader smoke test。
  - 检查字段和 shape：`image [6,320,240]`、`force [3]`、`delta_force [3]`、`slip_label scalar`、`force_scale [3]`。
  - 在训练入口或 datamodule 初始化处输出并保存 data manifest：
    - 实际加载的 `.pkl` 文件绝对路径列表；
    - 每个文件的 size、mtime、sha256/hash；
    - 每个文件样本数、总样本数；
    - 是否显式排除 `org_dataset_gelsight_*.pkl`；
    - 使用的 dataset config、split manifest 路径、预处理 config。
  - `org_dataset_gelsight_*.pkl` 仅登记为视觉域/实机域差异诊断用；官方训练默认数据仍为 `dataset_gelsight_*`。
- **产物**：smoke test 日志、样本字段/shape 表、`data_manifest.json/yaml`、使用的数据文件清单。
- **验证标准**：至少一个 train/val batch 可稳定读取；字段、shape、dtype 与训练脚本预期一致；manifest 能证明正式 A/B/C 的训练数据来自 `dataset_gelsight_*` 且未混入 `org_dataset_gelsight_*`。

### Step 2b：GSmini 图像预处理/颜色域一致性门禁
- **目标**：在不改变官方训练默认数据的前提下，评估 `dataset_gelsight`、`org_dataset_gelsight`、本地/实机 `tactile_rgb` 或代表性样例图像之间的颜色域与预处理一致性，作为实机部署风险门禁。
- **远程/本地执行内容**：
  - 对以下来源抽样：`dataset_gelsight_*`、`org_dataset_gelsight_*`、本地或实机采集的 `tactile_rgb`、必要样例图像。
  - 所有来源必须经过同一条预处理链：`load_sample_from_buf(io_buf, bg)` 内部完成 background diff、必要时 rotate/crop；随后使用 `get_resize_transform([320,240])` 做 resize/ToTensor。这里的 `remove_bg` 指通过传入背景 `bg` 做差分，不是另一个独立函数。
  - 对预处理后张量统一记录：通道顺序（RGB/BGR 或 six-channel frame ordering）、shape、dtype、range/min/max、per-channel mean/std、关键分位数、直方图或颜色分布摘要。
  - 明确比较 `dataset_gelsight` 与 `org_dataset_gelsight`、实机 `tactile_rgb` 的差异，并标注可能影响部署的颜色域/背景去除/旋转裁剪风险。
  - 结论仅用于 deployment risk / domain-shift diagnostic；不得因此把官方 A/B/C 默认训练数据切换为 `org_dataset_gelsight_*`，除非另起实验并重新记录 manifest。
- **产物**：`preprocess_domain_report.md`、统计表/直方图、代表性预处理前后样例图。
- **验证标准**：报告能复现实验代码、样本来源和预处理参数；若实机/本地样例与 `dataset_gelsight_*` 通道顺序、shape、range 或均值方差/直方图显著不一致，必须在模型交付中标记实机部署风险，不做 sim2real/real deployment 正式 claim。

### Step 3：force 轴语义与 Newton 单位确认
- **目标**：确认 `force_norm * force_scale = force_N`，并确定 normal 轴。
- **远程执行内容**：
  - 统计 flat/sharp/sphere 等 batch 的 `Fx/Fy/Fz` 分布。
  - 对 slip 前后各轴变化做分布和时序诊断。
  - 验证哪个轴随接触加载最稳定变化；若确认为 `Fz`，使用 `Fn=abs(Fz_N)`、`Ft=sqrt(Fx_N^2+Fy_N^2)`。
- **产物**：axis/unit report、force 分布图/表、最终 axis mapping。
- **验证标准**：所有 force 指标以 Newton 单位计算；normal 轴有证据支持。若无法确认 normal 轴，C 降级为 signed-force multitask diagnostic，不能做 friction-style claim。

### Step 4：trajectory-level split 生成、冻结与训练入口强制消费
- **目标**：避免时间序列 sample-level random split 泄漏；正式 A/B/C 禁止走默认 sample-level `random_split`。
- **远程执行内容**：
  - 基于 Sparsh dataset 的 `idx2traj` / `traj2idx` 做 deterministic split。
  - 固化 train/val/test trajectory 列表和对应 sample index manifest。
  - 修改/配置训练入口，使 datamodule/dataloader 必须显式读取 split manifest；若未提供 split manifest 或回落到默认 sample-level random split，正式 A/B/C run 应 fail-fast。
  - 在每个正式 run 日志中输出 dataloader 实际使用的 split manifest path、train/val/test index 数量、trajectory id 摘要和 hash。
  - 统计每个 split 的 trajectory 数、sample 数、slip 正负样本数、force 分布、`Ft/Fn` 分布。
- **产物**：split 文件、split summary、dataloader-index audit、分布统计，随 run 保存到 `/vla1/zjy/sparsh_runs/experiments/<run_name>/`。
- **验证标准**：同一 trajectory 不跨 split；split 文件可复现加载；test set 只评估不调参；正式 A/B/C 的 dataloader index 全部来自 split manifest，日志/审计脚本能证明没有使用默认 sample-level `random_split`。

### Step 5：A separate force/slip baseline 重评/重训
- **目标**：在冻结 split 和统一指标下建立公平对照。
- **远程执行内容**：
  - 对 DINOv2/MAE 等 encoder 的 separate force decoder 与 slip decoder 重评或重训。
  - 使用统一指标脚本输出：
    - force：`Fx/Fy/Fz RMSE/MAE`、`Fn RMSE/MAE`、`Ft RMSE/MAE`、`Fmag RMSE/MAE`，全部 Newton 单位；
    - slip：Accuracy、Precision、Recall、F1、AUROC、AUPRC；
    - diagnostic：高 `Ft/Fn` slip recall、低 `Ft/Fn` false alarm、`p_slip` 与 `Ft/Fn` 单调关系、contradiction rate。
- **产物**：A baseline checkpoints、metrics table、diagnostic report、DINOv2/MAE 对照表初版。
- **验证标准**：A 的评估可复现；所有指标来自同一 trajectory split；能作为 B/C 的比较基线。

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
