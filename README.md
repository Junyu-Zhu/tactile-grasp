# tactile-grasp

基于视觉触觉的接触力估计、力条件化滑移检测与短期接触状态预测研究代码。
仓库同时保留部分 UR5 抓取仿真与触觉采集代码；本文档以 `sparsh-force-slip/` 为主要入口。

> 当前发布的是研究源码。数据集、预训练权重、训练后的模型、特征缓存、运行日志、实验记录及内部 pipeline 不随最新代码版本分发。部分入口依赖原实验生成的协议和清单，因此尚不是下载后即可完整复现实验的一键运行包。

## 1. 研究内容

- **当前接触力估计**：从视觉触觉表示估计三轴接触力，为下游提供预测力信息。
- **当前滑移检测**：比较视觉基线、视觉—力普通拼接、特征级 FiLM 和有界 FiLM，分析 static 误报与 gross 召回的权衡。
- **有限编码器微调**：比较视觉训练、当前真实力辅助监督、FiLM 及二者组合。辅助监督不等同于修改原物理力输出路径。
- **短期接触状态预测**：利用当前基础视觉表示、预测力或其组合，预测未来 1、5、10 帧的三轴力变化。

不同实验保存在独立轮次目录中。它们是对照方案，不是默认串联成一个部署模型；未来力预测也不直接等同于未来滑移或掉落预测。

## 2. 目录结构

```text
tactile-grasp/
├── sparsh-force-slip/
│   ├── scripts/                 # 早期 force/slip/future 训练与分析源码
│   ├── runbooks/                # 保留的 shell 启动脚本
│   └── experiments/htt_normalflow/
│       ├── adapters.py          # 数据适配
│       ├── prepare_splits.py    # 数据角色划分工具
│       ├── round18_htt_force_conditioned_film/
│       ├── round19_htt_partial_encoder_finetuning/
│       ├── round20_htt_contact_state_transition/
│       ├── round22_921_g1_joint_frozen/
│       ├── round23_921_g2_force_aux_finetune/
│       └── ...                  # 其他历史方案与评价工具
├── protac/                      # 抓取、感知与采集相关模块
├── isaacsim_test/               # 仿真相关脚本和检查
├── environment/                 # 机器人与场景资源
└── PYTHON_CODE_LAYOUT.md        # 父仓库代码布局说明
```

部分轮次直接导入早期轮次模块，请保持目录结构，不要只复制单个训练文件。
`experiments` 中的 `.py`、`.sh` 是源码；同目录下的运行结果与记录由 `.gitignore` 排除。

## 3. 数据集及其用途

| 数据集或数据来源 | 在本研究中的用途 | 使用边界 |
|---|---|---|
| **HTT** | 当前力监督适配、static/gross 滑移检测、未来接触力变化预测 | 主要使用试次级隔离的开发协议；incipient 不进入主要二分类监督 |
| **Sparsh / TacBench 相关 Gelsight-mini 数据** | 早期接触力估计和滑移任务，包括 `gelsight-force-estimation`、`object_slide` | 需按具体任务检查输入、标签和划分，不能直接与 HTT 指标混排 |
| **ToucHD-Force** | 在经过核验的传感器路线进行力监督预适配，再评价向 HTT 的迁移 | 不直接混合两域的力坐标、零点或误差 |
| **NormalFlow** | 历史阶段的接触状态、冻结特征及运动辅助预测探索 | 不把运动或位姿直接解释为滑移标签；不是当前 HTT 检测模型的必需训练数据 |
| **ToucHD-Mani** | 数据可用性讨论和后续研究候选 | 不属于本次主要训练方案的必需输入，不宣称已完成其滑移监督训练 |
| **DeformableObjectsGrasping** | 外域真实物体数据可用性审计 | 审计不等同于外域性能验证 |
| **自行采集的稳定/滑移触觉序列** | 有限实物补充分析 | 无真实力记录的序列不用于报告物理力精度 |

数据和权重应从各自发布方获得，并遵守原始使用条款。相关入口：

- [Sparsh 官方项目](https://github.com/facebookresearch/sparsh)：预训练触觉表示与下游任务说明。
- [AnyTouch2 / ToucHD 项目](https://github.com/GeWu-Lab/AnyTouch2)：ToucHD 系列数据与相关说明。

本仓库不重新分发上述数据。预训练编码器采用 Sparsh 路线，主要后续实验使用 MAE；历史代码也包含 DINO、I-JEPA 等编码器对照。

### 建议的数据布局

将数据和产物放在仓库之外，例如：

```text
<storage>/
├── tactile_dataset/
│   ├── HTT-dataset/
│   ├── NormalFlow-dataset/
│   └── ToucHD-Force/
├── tactile_datasets/Gelsight-mini/
│   ├── gelsight-force-estimation/
│   └── object_slide/
├── sparsh_models/               # 预训练权重
└── sparsh_runs/                 # 特征缓存、训练模型和结果
```

这只是推荐布局，不是自动生效的环境配置。历史脚本仍可能含原机器的绝对路径，运行前需要检查参数默认值、配置和上游依赖。

## 4. 环境准备

原训练环境记录为：Python **3.9.25**、PyTorch **2.7.0+cu128**、CUDA runtime **12.8**。
这些版本是原实验环境信息，不代表所有代码已在其他机器验证。GPU 驱动需兼容所安装的 PyTorch CUDA 构建。

核心 Python 依赖包括：`torch`、`torchvision`、`numpy`、`scipy`、`scikit-learn`、`matplotlib`、`Pillow`、`opencv-python`，以及独立的 Sparsh `tactile_ssl` 包及其依赖。

```bash
git clone https://github.com/Junyu-Zhu/tactile-grasp.git
cd tactile-grasp

# 在已配置好兼容 PyTorch 和 Sparsh 依赖的 Python 环境中：
export XFORMERS_DISABLED=1
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import tactile_ssl; print(tactile_ssl.__file__)"
```

Sparsh 请按官方仓库安装。原服务器对其训练入口、环境及信号处理有本地适配，
这些外部仓库修改不会自动随本仓库克隆获得。当前没有声称提供完整、可移植的依赖锁文件。
Isaac Sim、Isaac Lab 和机器人控制依赖仅属于对应仿真模块，不是阅读或使用缓存检测头源码的前提。

## 5. 主要训练与评价入口

下列路径均相对 `sparsh-force-slip/experiments/htt_normalflow/`。

| 任务 | 入口 | 说明 |
|---|---|---|
| 数据适配、划分 | `adapters.py`、`prepare_splits.py` | 核验原始字段、试次、标签和角色 |
| 冻结编码器检测头 | `round22_921_g1_joint_frozen/train_frozen.py` | V：视觉；C：拼接；M：FiLM；MB：有界 FiLM |
| 短期力变化预测 | `round22_921_g1_joint_frozen/train_f1.py` | K-V、K-F、K-VF 三组输入对照 |
| 有限编码器微调 | `round22_921_g1_joint_frozen/train_e3.py` | G2 的训练实现仍位于 R22 准备目录 |
| 微调任务队列 | `round23_921_g2_force_aux_finetune/launch_g2_queue.py` | 依赖运行清单及授权身份，不应直接套用历史路径 |
| 冻结检测评价 | `round22_921_g1_joint_frozen/evaluate_frozen_detection.py` | 固定阈值与 calibration 规则 |
| 微调模型预测、评价 | `round23_921_g2_force_aux_finetune/predict_e3.py`、`evaluate_e3.py` | 使用匹配输入和上游身份的模型 |

### 运行顺序

1. 按目标实验准备原始数据、预训练编码器和所需 force/visual 上游权重。
2. 核验试次与泄漏组，生成 fit、selection、calibration、validation 角色及完整端点。
3. 使用对应准备代码生成特征缓存、支持清单和协议；归一化只使用允许的训练数据。
4. 先做小规模 smoke，确认输入、目标、梯度、冻结边界和恢复行为，再执行正式配置。
5. 由 selection 选择 checkpoint、calibration 确定阈值，在 validation 报告开发评价结果。

不要伪造缺失的协议、清单或身份摘要来绕过检查，也不要用测试标签选择模型。

### 已准备缓存后的命令示例

以下命令展示真实 CLI 参数，**不是原始数据到训练的一键命令**。
`DATA`、`SUPPORT`、`OUT` 必须替换为已生成且与代码 schema、fold、seed 一致的路径。
示例没有启用 `--formal`；正式派发还需要入口规定的锁定协议和授权文件。

```bash
EXP=sparsh-force-slip/experiments/htt_normalflow/round22_921_g1_joint_frozen

# 冻结编码器：力条件 FiLM 当前滑移检测
python "$EXP/train_frozen.py" \
  --data "$DATA" --support-inventory "$SUPPORT" \
  --output "$OUT" --group M --fold 1 --seed 20260914 --device cuda:0

# 短期接触状态：视觉＋预测力输入
# 此处 DATA、SUPPORT、OUT 应切换为未来状态任务自己的缓存、清单和输出目录。
python "$EXP/train_f1.py" \
  --data "$DATA" --support-inventory "$SUPPORT" \
  --output "$OUT" --group K-VF --fold 1 --seed 20260914 --device cuda:0
```

完整对照需保留协议规定的全部折、种子和组别；单条示例不代表完整实验。
原实验的配置、划分、运行清单、授权与结果没有随本次源码公开，重现实验前仍需准备相应材料。

## 6. 评价约定

- 当前检测主要比较 **static 与 gross**。incipient 单独分析，不默认将其作为可靠提前预警标签。
- 报告 pAUC、AP、实际 static FPR、gross 召回、BA、macro-F1，以及逐试次连续误告警与事件检出。
- calibration 上的误报约束不保证 validation 达到同样误报率。
- 短期状态任务区分真实力变化、预测变化、当前力估计误差和未来绝对力误差；比较保持与线性等基线。
- HTT 相关实验使用的参考相对、逐轴裁剪至 ±20 N 的目标是实验约定，不代表传感器通用量程或各数据集统一力定义。
- 开发折存在重叠和历史开发暴露；不能将这些结果表述为独立盲测或真实机器人成功率。

## 7. Git 与资源管理

最新版本只跟踪源码和使用说明；数据、权重、训练记录、pipeline 等通过 `.gitignore` 排除。
被停止跟踪的文件仍保留在原机器磁盘。原 Git 历史按项目所有者要求保留，
**旧提交中可能仍能找到此前提交的日志、报告和计划**；忽略规则只约束后续版本。

查看组件说明：[sparsh-force-slip/README.md](sparsh-force-slip/README.md)。
上游代码、数据集及权重遵守各自许可证；本次整理不擅自新增或改变它们的授权。
