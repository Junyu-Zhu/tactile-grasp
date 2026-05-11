# Force-Slip Decoder Phase 1/2 规划草案

Revision note: 已根据 Architect/Critic ITERATE 反馈补强 GSmini 图像预处理/颜色域一致性门禁、训练数据 manifest/hash、trajectory-level split 实际消费验收、B→C 硬门禁、slip_horizon train/val 定版约束，并固定 B/C 门禁阈值、consistency diagnostic 指标定义和 GSmini 预处理链表述；本版新增 force-slip 代码集中到 `sparsh-force-slip/`，并强制采用“本地修改 → GitHub push → 服务器 pull → 服务器运行”的同步流程。

> 基于 `tactile_grasp/pipeline/1-force-slip-decoder/force-slip-decoder-pipeline.md` 与已确认约束生成。force-slip 代码统一在本地 `/home/zjy/Documents/grasp/tactile_grasp/sparsh-force-slip` 修改并 push 到 GitHub，再在服务器 `zjy-4090` 的 `/documents/tactile_grasp` pull；训练/评估从 `/documents/tactile_grasp/sparsh-force-slip` 执行。远程资源统一使用 `/vla1/zjy/{tactile_datasets,sparsh_models,sparsh_runs}`。本地保留 pipeline 文档、分析结论、必要本地集成测试和按需回传的 checkpoint。

## 1. RALPLAN-DR summary

### 原则
1. **先证据、后训练**：先确认环境、数据、force 轴语义、Newton 单位和 trajectory-level split，再训练 B/C。
2. **可比性优先**：A/B/C 必须使用同一 split、同一指标脚本、同一 Newton 反归一化逻辑。
3. **C 不抢跑**：未确认 normal 轴、单位、轨迹级划分和 A baseline 前，不训练 consistency decoder。
4. **主指标不可牺牲**：C 只有在 force/slip 主指标不明显退化且 consistency 改善时才算成功。
5. **代码先本地、运行在远程**：force-slip 代码只在本地 `sparsh-force-slip/` 开发，经 GitHub 同步后由服务器 pull 并运行，禁止服务器长期保留未同步改动。
6. **本地轻量化**：训练、评估、日志整理在远程；本地只做 checkpoint 加载和 tactile_grasp 集成冒烟测试。

### Top 3 决策驱动
1. **实验可信度**：避免 sample-level 泄漏、归一化 force 误用、normal 轴误判导致不可解释提升。
2. **阶段风险控制**：用 A baseline 和 B multitask 作为 C 的前置 sanity check 与 fallback。
3. **资源与复现管理**：代码版本由 GitHub commit 固化；所有 run、split、metrics、checkpoint 统一落在 `/vla1/zjy/sparsh_runs`，便于追踪和回传。

### 可选方案与取舍
- **方案 A：直接训练 C consistency decoder**  
  - 优点：最快得到“方法结果”。  
  - 缺点：若 normal 轴/单位/split/baseline 未固定，提升不可比较，且可能是数据泄漏或尺度错误。  
  - 结论：**拒绝作为起点**。
- **方案 B：Phase 1 固定数据与 A baseline，Phase 2 再训练 B/C**  
  - 优点：可解释、可复现、风险最小；失败时也能留下 A/B 有效交付物。  
  - 缺点：前期准备较多。  
  - 结论：**推荐主线**。
- **方案 C：先只做 B，不做 C**  
  - 优点：实现简单，可验证共享表示是否有收益。  
  - 缺点：无法回答 force-slip consistency 是否有帮助。  
  - 结论：可作为 Phase 2 中 C 失败或主指标退化时的 fallback。

## 2. Phase 1：数据、环境、验证、基线可比性准备

### Step 1：远程环境与路径确认
- **目标**：确认训练只在 `zjy-4090` 执行，并固定本地/远程代码同步方式、force-slip 代码目录、数据、模型、输出路径。
- **远程执行内容**：
  - 本地开发目录固定为 `/home/zjy/Documents/grasp/tactile_grasp/sparsh-force-slip`；所有 force-slip 模型、config、训练入口和指标脚本先在本地修改。
  - 本地修改完成后从 `/home/zjy/Documents/grasp/tactile_grasp` 执行 `git status`、`git add sparsh-force-slip pipeline/1-force-slip-decoder`、按 AGENTS.md 的 Lore Commit Protocol 执行 `git commit`、`git push origin main`。
  - 服务器执行：`ssh zjy-4090 && cd /documents/tactile_grasp && git status && git pull --ff-only origin main && cd sparsh-force-slip`。
  - 后续数据检查、训练、评估命令都从 `/documents/tactile_grasp/sparsh-force-slip` 运行。
  - 检查 `which python`、`python -V`、`torch/hydra/omegaconf` import。
  - 确认 `/vla1/zjy/tactile_datasets`、`/vla1/zjy/sparsh_models`、`/vla1/zjy/sparsh_runs` 存在。
  - 建立或确认 `paths=zjy_4090` 指向 `/vla1/zjy`。
- **产物**：环境记录、GitHub commit/pull 记录、paths config、资源目录清单。
- **验证标准**：服务器代码来自 `/documents/tactile_grasp` 的 GitHub pull；训练入口位于 `/documents/tactile_grasp/sparsh-force-slip`；训练命令可显式使用 `paths=zjy_4090`；日志/输出默认进入 `/vla1/zjy/sparsh_runs`。

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

## 3. Phase 2：实现/训练 B 与 C、结果选择与本地回传

### Step 1：实现并训练 B shared multitask decoder
- **目标**：验证共享下游表示是否优于 separate baseline。
- **远程执行内容**：
  - 若需要修改 B 的模型、config、训练入口或指标脚本，必须先在本地 `/home/zjy/Documents/grasp/tactile_grasp/sparsh-force-slip` 修改并 push；服务器只在 `/documents/tactile_grasp` pull 后运行。
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

## 4. 不应做 / 暂缓做的事项

- 不在 Phase 1 之前直接训练 C。
- 不使用 sample-level random split 做正式比较；正式 A/B/C 必须由训练入口实际消费 trajectory-level split manifest。
- 不在归一化 force 上计算 `Ft/Fn` 或 consistency loss。
- 不假设 `Fz` 一定是 normal 轴；必须先验证。
- 不把 `org_dataset_gelsight_*.pkl` 作为默认训练数据；它只用于视觉域/实机域差异诊断。
- 不用 Phase5 sim 证明 shear-normal consistency、friction cone consistency 或 sim2real 提升；Phase5 仅 diagnostic-only。
- 不把 test set 用于 `tau/alpha/beta_cons/slip_horizon` 调参。
- 不直接在服务器 `/documents/tactile_grasp/sparsh-force-slip` 长期开发或留下未 push/pull 同步的代码。
- 暂缓 partial encoder finetuning、learnable friction threshold、high-confidence mask、sim2real claim，直到 C 在主指标不退化下改善 consistency。

## 5. 最小验收标准

1. 远程 `zjy-4090` 环境、`/documents/tactile_grasp` repo、`/documents/tactile_grasp/sparsh-force-slip` 运行目录和 `/vla1/zjy` 路径配置可复现；代码修改记录包含本地 commit、GitHub push 和服务器 `git pull --ff-only` 记录。
2. `dataset_gelsight_*.pkl` 数据加载冒烟测试通过，关键字段/shape 符合预期，并产出记录 pkl 列表、size/hash、样本数与 `org_dataset_gelsight_*` 排除状态的 data manifest。
3. GSmini 图像预处理/颜色域一致性报告完成，覆盖 `dataset_gelsight`、`org_dataset_gelsight`、本地/实机 `tactile_rgb` 或样例图像在同一预处理链后的通道顺序、shape、range、均值方差/直方图，并标注实机部署风险。
4. force 已反归一化到 Newton；normal 轴有明确证据和 axis mapping。
5. trajectory-level split 冻结并保存，训练入口实际消费 split manifest；dataloader index 审计确认无 trajectory 跨 train/val/test 泄漏且未回落到 sample-level random split。
6. A baseline 在统一 split/指标脚本下完成重评或重训。
7. B 完成训练和评估；若相对 A force RMSE 增加 >10% 或 slip F1 下降 >2pp，C 被硬性降级为 diagnostic-only，不做 formal claim；若落入 5%–10% / 1–2pp warning band，必须在报告中记录风险。
8. C 仅在 Phase 1 与 B 通过后作为正式实验训练；`tau/alpha/beta_cons/slip_horizon` 只用 train/val 定版；成功必须满足固定主指标门禁和统一 consistency diagnostic 改善阈值。
9. 最终交付包含远程 run 路径、checkpoint/config/split/metrics/source 记录；本地 checkpoint 加载和推理冒烟测试通过。
