# UR5 + Robotiq + DIGIT / GelSight Mini + GraspNet + Sparsh + RL 抓取训练 Pipeline

## 1. 项目目标

目标是在 **UR5e 机械臂 + Robotiq 夹爪** 平台上，结合：

- **GraspNet**：提供初始抓取位姿
- **Sparsh**：提供视觉式触觉表征与触觉下游任务能力
- **强化学习（RL）**：在抓取过程中实时调节夹爪力与闭合动作

实现以下能力：

1. 在 **YCB 物体** 上完成稳定抓取
2. 抓取过程中通过触觉实时估计接触状态、力和滑移风险
3. 在不掉落的前提下尽量使用**最小必要夹持力**
4. 对柔软物体降低过压和抓坏风险

本项目首版采用 **“先仿真打通，再迁移实机”** 的路线。

---

## 2. 系统总体流程

整体控制链路定义如下：

1. **场景感知**
   - 采集 RGB-D 图像
   - 使用 GraspNet 生成候选 6D 抓取位姿

2. **抓取初始化**
   - 选择一个候选 grasp pose
   - 控制 UR5e 将 Robotiq 移动到预抓取位姿并执行闭合

3. **触觉感知**
   - 从 DIGIT 或 GelSight Mini 实时读取触觉图像流
   - 使用 Sparsh backbone 编码触觉观测
   - 通过任务头输出：
     - 力估计
     - 滑移检测
     - 可选：接触场 / force-field / 稳定性分数

4. **闭环调节**
   - RL 策略接收：
     - GraspNet 初始抓取位姿信息
     - 当前夹爪状态
     - Sparsh 触觉特征
     - 力估计结果
     - 滑移检测结果
   - RL 输出：
     - 夹爪闭合增量
     - 夹爪目标力增量
     - 保持 / 放松 / 释放动作

5. **任务目标**
   - 物体成功提起并保持稳定
   - 减少滑落
   - 减少过度夹紧

---

## 3. 传感器路线划分

本项目按两条路线规划，但**首版实验默认一次只安装一种触觉传感器**。

### 3.1 DIGIT 路线

DIGIT 是视觉式触觉传感器。基于 Sparsh 开源资源，DIGIT 路线可直接支持：

- Sparsh backbone 初始化
- force-field 可视化 / 解码
- 力估计
- 滑移检测
- 位姿估计（in-hand pose / contact state）

适合的角色：

- 接触力变化估计
- 剪切趋势判断
- 抓取过程中的在手状态感知

### 3.2 GelSight Mini 路线

GelSight Mini 也是视觉式触觉传感器。基于 Sparsh 开源资源，GelSight Mini 路线可直接支持：

- Sparsh backbone 初始化
- force-field 可视化 / 解码
- 力估计
- 滑移检测

适合的角色：

- 局部接触面积与形变感知
- 接触法向/切向变化判断
- 抓取中的失稳预警

### 3.3 重要说明

- **DIGIT 不是 GelSight 系列**，两者只能算同类“视觉式触觉传感器”
- Sparsh README 明确支持的传感器族包括：
  - **DIGIT**
  - **GelSight'17**
  - **GelSight Mini**
- repo 中的部分 grasp / textile 任务配置是基于 **GelSight'17（marker 版）**，**不是 GelSight Mini**

因此：

- **DIGIT 做抓取任务**：必须准备 DIGIT 自己的抓取数据
- **GelSight Mini 做抓取任务**：也必须准备 GelSight Mini 自己的抓取数据
- 不能直接把 GelSight'17 的抓取数据当成 DIGIT / Mini 的最终目标数据

---

## 4. Sparsh 可复用模型与能力

## 4.1 推荐默认 backbone

首版默认选用：

- **facebook/sparsh-dino-base**

推荐原因：

- Sparsh 仓库中的 demo 和下游实验参考较完整
- DINO 路线的工程参考较多
- 后续迁移到力估计、滑移检测和闭环控制更方便

如需对比实验，可加入：

- `facebook/sparsh-mae-small`
- `facebook/sparsh-mae-base`
- `facebook/sparsh-dino-small`
- `facebook/sparsh-dino-base`
- `facebook/sparsh-dinov2-base`
- `facebook/sparsh-ijepa-small`
- `facebook/sparsh-ijepa-base`
- `facebook/sparsh-vjepa-small`
- `facebook/sparsh-vjepa-base`

## 4.2 现成可直接利用的任务权重

### DIGIT

- **facebook/sparsh-digit-forcefield-decoder**

用途：

- 实时 force-field 可视化
- 接触区域与剪切方向理解
- 可作为后续 grasp RL 的辅助观测或调试工具

### GelSight Mini

- **facebook/sparsh-gelsight-forcefield-decoder**

用途：

- 实时 force-field 可视化
- 接触法向 / 剪切分布辅助判断
- 可作为后续 grasp RL 的辅助观测或调试工具

## 4.3 Repo 已有下游任务支持

根据 `facebookresearch/sparsh` 的 task 配置：

### DIGIT 已支持的下游任务

- **T1** 力估计
- **T2** 滑移检测
- **T3** 位姿估计
- **forcefield demo**

### GelSight 已支持的下游任务

- **T1** 力估计
- **T2** 滑移检测
- **T4** 抓取稳定性（基于 GelSight'17）
- **T6** textile recognition（基于 GelSight'17）
- **forcefield demo**

### 关键结论

对于本项目：

- **DIGIT**：Sparsh 提供了很强的辅助感知基础，但**没有现成 DIGIT grasp stability 公开权重/数据**
- **GelSight Mini**：Sparsh 提供了 force/slip 基础，但 **repo 的 grasp stability 对应 GelSight'17，不是 Mini**
- 因此两条路线都需要围绕你自己的抓取任务做**额外微调和数据采集**

---

## 5. 公开数据集准备清单

以下数据集分为：

1. **无标注预训练 / 域适配数据**
2. **监督下游数据**
3. **可作为抓取 warm-start 的近似数据**

## 5.1 DIGIT 相关数据

### 无标注 / 预训练 / continued pretraining

- **YCB-Slide**
- **facebook/touch-slide**

用途：

- DIGIT 域适配
- continued pretraining
- 让 Sparsh backbone 适应你的 DIGIT 光照、背景、安装角度、夹爪结构

### 监督下游数据

- **facebook/digit-force-estimation**
- **facebook/digit-pose-estimation**

用途：

- 力估计头训练
- 位姿估计头训练
- 辅助形成抓取中 contact state encoder

### DIGIT 路线建议

必须准备：

- `touch-slide`
- YCB-Slide
- `digit-force-estimation`
- `digit-pose-estimation`
- **自采 DIGIT grasp 数据**（必须）

## 5.2 GelSight Mini / GelSight 相关数据

### 无标注 / 预训练 / continued pretraining

- **Touch and Go**（GelSight'17）
- **ObjectFolder-Real**（README 中说明可提取 GelSight Mini tactile images）

用途：

- GelSight / GelSight Mini 域适配
- continued pretraining

### 监督下游数据

- **facebook/gelsight-force-estimation**

用途：

- 力估计训练
- 滑移检测辅助训练

### 抓取 warm-start 数据

- **Feeling of Success**（GelSight'17）

用途：

- grasp stability 任务结构参考
- 可用作初始抓取判别器 warm-start

注意：

- Feeling of Success 使用的是 **GelSight'17**，不是 GelSight Mini
- 只能用来 warm-start，**不能替代你的 GelSight Mini 自采抓取数据**

### GelSight Mini 路线建议

必须准备：

- ObjectFolder-Real（提取 Mini 相关 tactile 图像）
- `gelsight-force-estimation`
- Feeling of Success（仅作为 warm-start 可选）
- **自采 GelSight Mini grasp 数据**（必须）

---

## 6. 必须自采的抓取数据

这是本项目最关键的数据部分。

## 6.1 为什么必须自采

Sparsh 开源资源虽然提供了：

- force estimation
- slip detection
- pose estimation
- force-field decoder

但没有直接覆盖你当前目标场景：

- UR5 + Robotiq
- YCB 抓取
- DIGIT 或 GelSight Mini
- 以最小夹持力完成稳定抓取
- 对柔软物体抑制破坏

因此必须构建**你自己的抓取时序数据集**。

## 6.2 自采数据应包含的标签与字段

每个抓取 episode 至少包含：

### 场景与物体信息

- `episode_id`
- `object_id`
- `object_name`
- `object_category`
- `object_material`
- `is_soft_object`
- `ycb_split`（train / val / test）

### 视觉输入

- `rgb_path`
- `depth_path`
- `camera_intrinsics`
- `graspnet_candidates` 或选中的 `grasp_pose`

### 机器人与控制状态

- `ee_pose`
- `gripper_width`
- `gripper_command`
- `gripper_force_command`（若可得）
- `joint_positions`
- `timestamp`

### 触觉数据

- `sensor_type`（digit / gelsight_mini）
- `tactile_stream`
- `before_frame`
- `during_frame`
- `after_frame`
- `bg_frame` / `no_contact_frame`

### 任务标签

- `success_label`（成功抓起并保持）
- `stable_label`
- `slip_label`
- `drop_label`
- `drop_time`
- `damage_label`（柔软物体是否被压坏/明显变形）
- `min_required_force_proxy`（可选）

## 6.3 推荐数据组织方式

首版建议组织成与 Sparsh 现有 loader 尽量接近的形式。

### GelSight Mini

优先参考 `vision_based_grasp_probes.py` 的 grasp 数据组织思想：

- `before`
- `during`
- `after`
- `is_gripping`

这样最容易复用 Sparsh 的抓取 loader 思路。

### DIGIT

因为 repo 没有现成的 DIGIT grasp loader，建议：

- 新建与 `vision_based_grasp_probes.py` 相似的数据格式
- 新写一个 `digit_grasp` loader
- 也保留 `before / during / after + is_gripping` 语义

---

## 7. 训练路线（推荐 4 阶段）

## Phase A：continued pretraining / 域适配

目的：

- 让 Sparsh backbone 适配你的传感器、光照、背景、夹爪安装方式、YCB 接触分布

### DIGIT

数据：

- YCB-Slide
- `facebook/touch-slide`
- 自采无标注 DIGIT 接触流

### GelSight Mini

数据：

- ObjectFolder-Real（Mini tactile 图像）
- Touch and Go（辅助）
- 自采无标注 GelSight Mini 接触流

输出：

- 适配你硬件域的 Sparsh encoder checkpoint

## Phase B：监督辅助任务训练

目的：先把触觉表征变成可控、可解释的接触状态估计器。

### DIGIT

训练：

- force estimation head
- slip detection head
- pose estimation head（推荐）

### GelSight Mini

训练：

- force estimation head
- slip detection head

默认策略：

- **冻结 encoder，只训练轻量 task head**
- 当数据量足够时再尝试部分解冻 encoder

输出：

- 可在抓取过程中实时输出力 / 滑移 / 位姿线索的监督模型

## Phase C：抓取监督学习

目的：构建抓取稳定性和掉落风险判别器。

输入：

- 触觉时序特征
- 夹爪状态
- grasp pose 信息
- 可选视觉特征

输出：

- success / failure
- stable / unstable
- drop risk
- optional damage risk

建议：

- backbone 先冻结
- 先训练二分类 grasp head / stability head
- 再决定是否联合训练 encoder

## Phase D：RL 闭环抓取训练

目的：使用触觉信号实现抓取过程中的实时夹持力调节。

### 状态输入

- GraspNet 选中的初始抓取位姿
- 当前夹爪宽度
- 当前夹爪目标力 / 控制命令
- Sparsh encoder 特征
- 力估计输出
- 滑移检测输出
- 可选视觉特征

### 动作输出

首版默认限制为夹爪控制，不做机械臂 6D 动作重规划：

- `delta_gripper_close`
- `delta_gripper_force`
- `hold`
- `release`

### 奖励设计

首版建议：

- 成功抓取并保持：正奖励
- 中途掉落：大惩罚
- 过大夹持力：惩罚
- 柔软物体损伤：大惩罚
- 在满足稳定抓取前提下的低力抓取：额外奖励

### 首版边界

首版不做：

- 复杂末端重抓取
- 机械臂 6D 连续在线重规划
- 多传感器融合策略学习

首版先做：

- **GraspNet 初始位姿 + 触觉闭环夹爪力调整**

---

## 8. 仿真到实机路线

## 8.1 仿真阶段目标

作用：

- 打通整体软件链路
- 验证 UR5 + Robotiq + grasp pose execution + RL 环路
- 验证状态机、奖励函数、动作空间

注意：

- 仿真中的触觉不能替代真实触觉分布
- 仿真主要用于控制框架验证，不是最终触觉学习来源

## 8.2 实机阶段目标

作用：

- 采集真实触觉数据
- 做 continued pretraining 与辅助任务训练
- 完成最终 grasp 微调与 RL 闭环验证

## 8.3 推荐顺序

1. 在仿真中打通 UR5 + Robotiq + GraspNet + RL 状态机
2. 上真实 DIGIT / GelSight Mini 采集无标注接触数据
3. 训练 Sparsh 域适配模型
4. 训练 force/slip/pose 等辅助任务头
5. 采集自定义 grasp 数据
6. 训练 grasp 稳定性判别头
7. 实机 RL 闭环微调

---

## 9. 首版默认实现决策

为减少后续实现歧义，首版默认如下：

- 传感器路线：**DIGIT 版 / GelSight Mini 版分开做**
- 一次实验只装一种传感器
- backbone 默认：`facebook/sparsh-dino-base`
- 先冻结 encoder，只训练 probe / decoder / policy head
- 抓取任务以 **YCB 物体** 为主
- 柔软物体实验作为第二阶段扩展
- RL 只控制夹爪相关动作，不做 6D 末端实时重规划
- GraspNet 只负责初始抓取位姿提议

---

## 10. YCB 抓取实验设计

## 10.1 实验对象

首版使用 YCB 中适合平行夹爪抓取的物体，覆盖：

- 刚性物体
- 尺寸差异较大的物体
- 表面摩擦特性不同的物体

后续扩展：

- 柔软物体
- 易碎物体
- 外形不规则物体

## 10.2 对照组

至少设计以下对照：

1. **视觉-only baseline**
   - GraspNet + 固定夹持策略

2. **视觉 + Sparsh 辅助监督**
   - 使用力估计 / 滑移检测做规则控制

3. **视觉 + Sparsh + RL**
   - 使用 RL 实时调力

4. **不同 backbone 对比（可选）**
   - DINO vs DINOv2 vs MAE

## 10.3 评价指标

- grasp success rate
- lift-and-hold success rate
- drop rate
- slip detection precision / recall
- force estimation error
- average gripping force
- peak gripping force
- soft-object damage rate（针对柔软物体）
- 每类物体上的成功率与平均力

---

## 11. 里程碑建议

## Milestone 1：工具链打通

- UR5 + Robotiq 仿真链路可运行
- GraspNet 候选抓取位姿可生成
- 机器人可执行基本抓取动作

## Milestone 2：触觉接入

- DIGIT / GelSight Mini 实时图像流可接入
- background / no-contact 采样流程可用
- 能运行 Sparsh backbone 推理

## Milestone 3：辅助任务训练

- DIGIT：力 / slip / pose 至少完成两项
- GelSight Mini：力 / slip 至少完成两项
- 结果达到可用精度

## Milestone 4：抓取监督建模

- 自采 grasp 数据集构建完成
- success / stable / slip / drop 标签可用
- grasp stability head 可训练

## Milestone 5：RL 闭环

- 完成视觉 + 触觉 + RL 抓取闭环
- 在 YCB 上优于固定夹力 baseline
- 在不降低成功率前提下减少平均夹持力

---

## 12. 当前应优先准备的内容

### 12.1 模型

优先下载：

- `facebook/sparsh-dino-base`
- `facebook/sparsh-digit-forcefield-decoder`
- `facebook/sparsh-gelsight-forcefield-decoder`

可选对比：

- `facebook/sparsh-dinov2-base`
- `facebook/sparsh-mae-base`
- `facebook/sparsh-ijepa-base`
- `facebook/sparsh-vjepa-base`

### 12.2 数据

#### DIGIT

- `facebook/touch-slide`
- YCB-Slide
- `facebook/digit-force-estimation`
- `facebook/digit-pose-estimation`
- 自采 DIGIT grasp 数据

#### GelSight Mini

- ObjectFolder-Real
- `facebook/gelsight-force-estimation`
- Feeling of Success（warm-start 可选）
- 自采 GelSight Mini grasp 数据

### 12.3 软件实现优先级

1. GraspNet 初始抓取位姿链路
2. DIGIT / GelSight Mini 采集接口
3. Sparsh backbone 推理接口
4. 力估计 / slip 检测任务头
5. grasp 数据采集与标注管线
6. RL 闭环训练

---

## 13. 参考资源

### GitHub

- Sparsh 仓库：
  - https://github.com/facebookresearch/sparsh

### Hugging Face 模型

- Sparsh collection：
  - https://huggingface.co/collections/facebook/sparsh-67167ce57566196a4526c328
- `facebook/sparsh-dino-base`
- `facebook/sparsh-digit-forcefield-decoder`
- `facebook/sparsh-gelsight-forcefield-decoder`

### Hugging Face 数据集

- `facebook/touch-slide`
- `facebook/digit-force-estimation`
- `facebook/digit-pose-estimation`
- `facebook/gelsight-force-estimation`
- `facebook/sparsh-x-dataset`（可作为后续多模态触觉扩展参考）
- `facebook/sparsh-skin-dataset`（可作为后续触觉自监督扩展参考）

### 外部数据资源

- YCB-Slide
- Touch and Go
- ObjectFolder-Real
- Feeling of Success

---

## 14. 最终结论

对于当前项目，最现实、最稳妥的路线是：

- **视觉侧**：用 GraspNet 提供初始抓取位姿
- **触觉侧**：用 Sparsh backbone 提供通用触觉表征
- **监督侧**：先训练力估计 / 滑移检测 / 位姿估计等辅助任务
- **抓取侧**：基于你自己的 DIGIT / GelSight Mini 抓取数据训练 grasp stability 与 drop risk 模型
- **控制侧**：最后通过 RL 实现抓取过程中的实时力调整

换句话说，Sparsh 已经提供了非常好的**触觉基础模型与中间能力**，但你要做的 **UR5 + Robotiq + YCB + 最小力稳定抓取** 仍然必须依赖：

1. 目标传感器域适配
2. 自采抓取数据
3. 任务级微调
4. 闭环控制训练
