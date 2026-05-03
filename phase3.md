# Phase 3：最小 scripted grasp/contact protocol + tactile logging integration

## 目标
这一阶段只回答一个问题：

> **在已经稳定的 `UR5e + Robotiq + connector + GSmini` embodiment 上，能否构建一个最小、可重复、可记录的 contact/grasp trial protocol，并把 tactile frames 与机器人状态对齐保存下来，为后续 Sparsh 适配提供干净数据？**

本阶段**不做**：
- Sparsh inference / adaptation
- closed-loop control
- GraspNet
- RL
- 大规模 object generalization

---

## 依赖
开始本阶段前，必须满足：
- `tactile_grasp/phase1.md` 全部通过
- `tactile_grasp/phase2.md` 全部通过
- tactile image 已稳定输出，且 no-contact / contact 可区分

---

## Source of Truth
- Canonical robot source：`tactile_grasp/environment/ur5_robotiq_GSmini/urdf/ur5_robotiq_GSmini.urdf`
- Phase 3 主执行脚本（建议新建）：`tactile_grasp/ur5_phase3_data_collection.py`
- Phase 3 trial runner（建议新建）：`tactile_grasp/ur5_phase3_trial_runner.py`
- 输出目录（建议）：`tactile_grasp/artifacts/phase3_*`

原则：**Phase 3 只做 protocol + logging，不做学习。**

---

## Step 1：锁定 Phase 3 范围
### 要做
- 确认本阶段目标仅为：
  - 最小 contact/grasp trial protocol
  - tactile logging integration

### 完成标准
- 能用一句话描述本阶段目标

### 禁止
- 不允许提前进入 Sparsh / adaptation / control

---

## Step 2：锁定 trial schema
### 要做
先定义每条 trial 必须保存的字段。

### 推荐字段
- `trial_id`
- `object_id`
- `sensor_id`
- `seed`
- `phase_name`
- `timestamp_start`
- `timestamp_end`
- `action_stage`
- `joint_state`
- `gripper_state`
- `tactile_frame_id`
- `contact_onset`
- `success_label`
- `failure_reason`

### 完成标准
- trial schema 有固定字段清单
- 后续所有 trial 都按统一格式写盘

### 禁止
- 边做边改字段定义

---

## Step 3：选定最小 object set 和 protocol variant
### 要做
- 先选 1 个 object
- 稳定后最多扩到 3 个 object
- 先定义最小 protocol variant：
  - contact-only
  - contact-hold
  - contact-hold-micro-lift

### 推荐顺序
- 先从 `contact-hold` 开始
- 稳定后再加 `micro-lift`

### 完成标准
- object list 固定
- protocol variant list 固定

### 禁止
- 一开始就做大规模 object suite

---

## Step 4：实现 deterministic trial runner
### 要做
实现固定顺序 trial：
1. reset
2. move to pre-grasp
3. close / contact
4. hold
5. optional micro-lift
6. release
7. end trial

### 完成标准
- trial runner 可以重复执行
- stage 顺序稳定不乱

### 失败 fallback
- 如果 lift 不稳定，先退化到 `contact-hold`

### 禁止
- 不允许加入任何 learning / policy logic

---

## Step 5：接入 tactile logging + robot state logging
### 要做
每条 trial 同步保存：
- tactile frames
- joint states
- gripper state
- action stage
- timestamp
- trial id
- object id

### 推荐目录结构
```text
tactile_grasp/artifacts/phase3/
  trial_0001/
    meta.json
    tactile/
    robot_state.json
    frame_map.csv
```

### 完成标准
- 能从磁盘上重建单条 trial 的时序
- tactile frame 和 action stage 可以一一对应

### 禁止
- 不能只存图片
- 不能只存状态

---

## Step 6：验证 alignment 和标签质量
### 要做
- 检查 tactile frame 与 action stage 是否错位
- 检查 contact_onset 是否可标注
- 检查 success/failure 标签是否合理

### 完成标准
- 对任意一条 trial，都能明确说出：
  - 何时接触
  - 何时保持
  - 何时释放
  - 是否成功

### 失败 fallback
- 如果时间戳不稳，先降采样，但不能丢失对齐信息

### 禁止
- alignment 不稳前，不准扩 trial 数量

---

## Step 7：跑一小批 clean trial
### 要做
- 先跑 10 条
- 稳定后扩到 20–30 条
- 检查：
  - 坏 trial 比例
  - 图片丢帧
  - 状态记录缺失

### 完成标准
- 有一小批干净、可复用 trial
- 后续可直接供 Phase 4 使用

### 失败 fallback
- 如果 trial 成功率低，先缩回 contact-hold，不强求 micro-lift

---

## Step 8：Phase 3 Review
### Exit Criteria
必须全部通过：
- [ ] trial schema 已锁定
- [ ] object set 已锁定（1–3 个）
- [ ] deterministic trial runner 可重复执行
- [ ] tactile frame 与 robot state 成功对齐
- [ ] contact_onset / success_label / failure_reason 可记录
- [ ] 数据能落盘到固定结构
- [ ] 一小批 clean trial 可复用

### 规则
- 任一项失败，就继续修 Phase 3
- **不允许**带着数据对齐问题进入 Phase 4

---

## 本阶段需要保存的产物
- trial schema 文档
- data directory 结构
- 运行命令
- 10–30 条 clean trial
- 对齐检查截图/表格
- 失败 trial 列表及原因

---

## Phase 3 的一句话总结

Phase 3 只构建 **deterministic contact/grasp trial protocol + tactile/robot-state 对齐日志**：先从 cube 开始，稳定后扩展到 `YcbChipsCan` 与 `YcbCrackerBox`，为 Phase 4/Sparsh 留下干净、可重放的数据，不在本阶段加入学习、策略或闭环控制。

## 当前锁定的 Phase 3 schema（v1）

代码 source-of-truth：`ur5_phase3_schema.py`。

每条 trial 的 `meta.json` 必须包含固定字段：

```text
trial_id
object_id
sensor_id
seed
phase_name
timestamp_start
timestamp_end
action_stage
joint_state
gripper_state
tactile_frame_id
contact_onset
success_label
failure_reason
```

本阶段固定 action stage 顺序：

```text
reset -> pre_grasp -> contact_close -> hold -> micro_lift -> release -> end_trial
```

固定 protocol variants：

```text
contact_only
contact_hold
contact_hold_micro_lift
```

固定落盘结构：

```text
tactile_grasp/artifacts/phase3/
  trial_schema.json
  phase3_review.json
  <trial_id>/
    meta.json
    tactile/
    robot_state.json
    frame_map.csv
```
**先把最小的 contact/grasp trial protocol 和 tactile logging 做到稳定、可对齐、可复用，再开始任何 Sparsh 或闭环控制工作。**
