# Phase 4：Cube-only Phase3 artifacts → Sparsh force/slip bridge and cube-trial optimization

## 目标

Phase 4 当前收窄为一个 **分步式 format bridge + smoke test + cube-trial optimization 规划阶段**：

> **先把当前仿真中已经采集的 cube Phase3 artifacts 转成 Sparsh force estimation / slip detection 可消费的格式，完成 dataloader / model-forward 连通性验证；随后再通过有设计地增加 cube trials，逐步补足 force labels 与 slip labels，使后续 force/slip evaluation 变得可信。**

本阶段不追求立刻得到可信 force/slip 指标，而是按阶段明确：

1. 当前数据能否接上 Sparsh 数据管线。
2. 当前 labels 哪些只是 placeholder / proxy，不能用于指标。
3. 后续需要怎样采更多 cube trials，才能让 force/slip labels 逐步有效。
4. 满足什么 gate 后，才允许进入真正 force/slip evaluation 或 adaptation。

---

## Scope Freeze

### 只考虑

- object：`cube`
- data source：`tactile_grasp/artifacts/phase3/phase3_cube_*`
- Sparsh tasks：
  - force estimation
  - slip detection
- 目标：
  - 将 Phase3 cube artifacts 映射 / 转换为 Sparsh-like force/slip 数据格式。
  - 用当前数据完成 dataloader / forward smoke test。
  - 规划更多 cube trials 的采集方式，用于补齐 force/slip labels。

### 不考虑

- forcefield demo / forcefield decoder。
- `chips_can` / `cracker_box`。
- 实物 GelSight Mini 数据采集。
- Sparsh adaptation / fine-tuning。
- closed-loop control。
- GraspNet / RL。
- 把当前 Phase3 `contact_onset` / `success_label` 直接当成真实 force/slip labels。

---

## Source of Truth

### Phase3 cube artifacts

- `tactile_grasp/artifacts/phase3/phase3_cube_*/meta.json`
- `tactile_grasp/artifacts/phase3/phase3_cube_*/robot_state.json`
- `tactile_grasp/artifacts/phase3/phase3_cube_*/frame_map.csv`
- `tactile_grasp/artifacts/phase3/phase3_cube_*/tactile/*_tactile_rgb.npy`
- optional preview：`*_tactile_rgb.png`

当前 artifacts 目录下可见 cube trials：

```text
phase3_cube_0001
phase3_cube_0002
phase3_cube_0003
```

> 注：`phase3.md` 中曾记录 10 条 cube clean trials，但当前 artifacts 目录可见的是 3 条 cube trials。因此 Phase4 当前以实际存在 artifacts 为准。

### Sparsh force/slip entry points

- force config：`sparsh/config/experiment/downstream_task/force/gelsight_dino.yaml`
- slip config：`sparsh/config/experiment/downstream_task/slip/gelsight_dino.yaml`
- shared data config：`sparsh/config/data/gelsight_force.yaml`
- force/slip dataset：`sparsh/tactile_ssl/data/vision_based_forces_slip_probes.py`
- force tester：`sparsh/tactile_ssl/test/test_t1_force.py`
- slip tester：`sparsh/tactile_ssl/test/test_t2_slip.py`
- dataset helper：`sparsh/tactile_ssl/data/digit/utils.py`

---

# 当前 Phase3 cube artifact 格式

每条 cube trial：

```text
tactile_grasp/artifacts/phase3/phase3_cube_0001/
  meta.json
  robot_state.json
  frame_map.csv
  tactile/
    left_frame_000001_tactile_rgb.npy
    left_frame_000001_tactile_rgb.png
    left_frame_000001_camera_depth.npy
    right_frame_000002_tactile_rgb.npy
    ...
```

## `frame_map.csv` 关键字段

```text
frame_id
timestamp
action_stage
side
sensor_id
tactile_rgb_path
camera_depth_path
camera_rgb_path
robot_state_index
contact_detected
success_label
failure_reason
```

## `meta.json` 关键字段

```text
trial_id
object_id
sensor_id
protocol_variant
success_label
contact_onset
alignment_summary
output_files
```

## `robot_state.json` 关键字段

```text
trial_id
samples[]
  index
  timestamp
  action_stage
  joint_state
  gripper_state
  object_state
  contact_state
  tactile_frame_ids
```

`contact_state` 当前提供：

```text
contact_detected
contact_sides
geometry_contact_sides
force_contact_sides
force_by_side_n
max_force_n
geometry
```

重要观察：当前 cube artifacts 有 tactile frames、stage、contact flag、robot-state alignment；但 `force_by_side_n` 示例为 0.0，且没有明确 slip event label。因此当前数据足够做 **格式连通性验证**，不足以做可信 force/slip 指标。

---

# Sparsh force/slip 期望格式

`gelsight_force.yaml` 默认使用：

```text
_target_: tactile_ssl.data.vision_based_forces_slip_probes.VisionForceSlipDataset
path_dataset: ${paths.data_root}/tacbench_data/T1_force/gelsight/
out_format: concat_ch_img
num_frames: 2
frame_stride: 5
```

Sparsh loader 期望：

```text
<path_dataset>/<dataset_name>/dataset_slip_forces.pkl
<path_dataset>/<dataset_name>/dataset_gelsight*   # pickle list of tactile images / arrays / buffers
```

`dataset_slip_forces.pkl` 至少需要：

```python
{
  "in_contact": np.ndarray,
  "trajectories": {
    trajectory_id: {
      "indexes": np.ndarray,
      "forces": np.ndarray,      # [T, 3]
      "slip_label": np.ndarray,  # [T], 0/1
    }
  }
}
```

Force task sample：

```python
sample["image"]
sample["force"]
```

Slip task sample：

```python
sample["image"]
sample["slip_label"]
sample["delta_force"]
```

---

# Phase3 → Sparsh 映射可行性

## 可直接映射

| Phase3 | Sparsh-compatible view | 可行性 |
| --- | --- | --- |
| `*_tactile_rgb.npy` | `dataset_gelsight_cube.pkl` image list | 高 |
| `frame_map.csv.frame_id` | image index / metadata | 高 |
| `frame_map.csv.action_stage` | trajectory stage metadata | 高 |
| `frame_map.csv.contact_detected` | `in_contact` | 中-高：contact flag，不等于 slip label |
| `robot_state.samples[].tactile_frame_ids` | frame → state alignment | 高 |
| `meta.success_label` | trial-level metadata | 中：不能用于 force/slip metric |

## 缺失 / 不可靠

| Sparsh expected | 当前状态 | 影响 |
| --- | --- | --- |
| true `forces[:, 3]` | `force_by_side_n` 示例为 0.0；缺少可靠三轴 force label | 不能做可信 force metric |
| true `slip_label` | 当前 `contact_hold` protocol 没有受控 slip event | 不能做可信 slip metric |
| true `delta_force` | 依赖 force 差分；force 不可靠时 delta_force 也不可靠 | 不能做可信 slip metric |
| enough trial diversity | 当前 cube trials 数量少，且多为 clean success/contact_hold | 不足以评估 classifier |

---

# Phase 4 分步操作计划

## Step 1 — 当前 cube artifacts inventory

### 目标

确认当前能用于 bridge 的 cube 数据到底有哪些。

### 操作

- 枚举 `tactile_grasp/artifacts/phase3/phase3_cube_*`。
- 对每条 trial 检查：
  - `meta.json`
  - `robot_state.json`
  - `frame_map.csv`
  - tactile RGB `.npy` / `.png`
- 统计：
  - trial 数量。
  - 每条 trial 的 frame count。
  - left/right sensor frame count。
  - stage 分布：reset / pre_grasp / contact_close / hold / release / end_trial。
  - contact frames vs no-contact frames。
  - 是否存在非零 `force_by_side_n`。

### 输出

```text
tactile_grasp/artifacts/phase4_sparsh_cube/inventory.json
tactile_grasp/artifacts/phase4_sparsh_cube/inventory.csv
```

### 完成标准

- 明确当前 cube 数据量。
- 明确当前 force label 是否可用。
- 明确当前 slip label 是否缺失。

---

## Step 2 — 生成当前数据的 Sparsh-compatible derived dataset

### 目标

不修改 Phase3 raw artifacts，派生一个 Sparsh-like 数据视图。

### 输出目录

```text
tactile_grasp/artifacts/phase4_sparsh_cube/cube_phase3_bridge/
  dataset_gelsight_cube.pkl
  dataset_slip_forces.pkl
  manifest.csv
  README.md
```

### 操作

1. 读取所有 `phase3_cube_* / frame_map.csv`。
2. 读取对应 tactile RGB `.npy`。
3. 生成 `dataset_gelsight_cube.pkl`：
   - pickle list。
   - 每个元素为 RGB ndarray 或 Sparsh `load_sample_from_buf` 可接受格式。
4. 生成 `dataset_slip_forces.pkl`：
   - `in_contact` 来自 `frame_map.csv.contact_detected` 或 `robot_state.contact_state.contact_detected`。
   - `trajectories[trial_id]["indexes"]` 指向 `dataset_gelsight_cube.pkl` 中的 image index。
   - `forces` 初期只允许：
     - placeholder zeros，或
     - 明确标记的 geometry/contact proxy，或
     - 后续修复 sim contact force 后替换为真实 sim force。
   - `slip_label` 初期只允许：
     - all-zero no-slip placeholder，或
     - 后续 controlled-slip protocol 生成的真实仿真 label。
5. 生成 `manifest.csv`，保留 provenance：
   - trial_id
   - side
   - frame_id
   - source_path
   - action_stage
   - contact_detected
   - robot_state_index
   - force_label_source：`placeholder` / `geometry_proxy` / `sim_contact_sensor`
   - slip_label_source：`placeholder` / `controlled_slip_protocol`
   - label_valid_for_metrics：true/false

### 完成标准

- Sparsh loader 能找到：
  - `dataset_slip_forces.pkl`
  - `dataset_gelsight*`
- 每个 derived sample 都能追溯回 Phase3 原始 frame。
- README 明确写出：当前 labels 是否有效，哪些只是 placeholder。

---

## Step 3 — Force estimation dataloader / forward smoke

### 目标

验证 Sparsh force estimation 模块能否消费 derived cube dataset。

### 操作

- 使用 `sparsh/config/experiment/downstream_task/force/gelsight_dino.yaml`。
- override：
  - `data.dataset.config.path_dataset=tactile_grasp/artifacts/phase4_sparsh_cube/`
  - dataset name 指向 `cube_phase3_bridge`。
- 先只做 dataloader instantiate。
- 如果 checkpoint / environment 可用，再做 model forward。

### 输出状态

Force lane 必须输出以下之一：

- `PASS_FORWARD`：dataloader + model forward 成功。
- `PASS_DATALOADER_ONLY`：dataloader 成功，但 checkpoint / model / environment 阻塞。
- `BLOCKED-with-exact-contract`：明确缺失哪个文件、字段、shape、checkpoint 或 package。

### 重要限制

如果 `forces` 使用 placeholder / proxy：

- 不允许报告可信 RMSE / correlation。
- 只能记录为 `METRIC_VALIDITY=invalid` 或 `proxy-only`。

---

## Step 4 — Slip detection dataloader / forward smoke

### 目标

验证 Sparsh slip detection 模块能否消费 derived cube dataset。

### 操作

- 使用 `sparsh/config/experiment/downstream_task/slip/gelsight_dino.yaml`。
- 复用 `cube_phase3_bridge`。
- 先只做 dataloader instantiate。
- 如果 checkpoint / environment 可用，再做 model forward。

### 输出状态

Slip lane 必须输出以下之一：

- `PASS_FORWARD`：dataloader + model forward 成功。
- `PASS_DATALOADER_ONLY`：dataloader 成功，但 checkpoint / model / environment 阻塞。
- `BLOCKED-with-exact-contract`：明确缺失哪个文件、字段、shape、checkpoint 或 package。

### 重要限制

如果 `slip_label` 是 all-zero placeholder：

- 不允许报告可信 balanced accuracy / F1。
- 不允许把 no-slip placeholder 当作真实 slip detection 结论。
- 只能记录为 `METRIC_VALIDITY=invalid`。

---

## Step 5 — 多采 cube trials：bridge robustness batch

### 目标

在不改变 label 语义的情况下，先增加 cube 数据覆盖，让 bridge / dataloader 更稳。

### 适合采集

继续采 `contact_hold` / clean cube trials，但加入轻微扰动：

- cube 初始位置微扰。
- left / right / both contact coverage。
- 不同 tactile sampling frequency。
- 不同 stage length。
- 不同 frame_stride 对应的连续帧。

### 作用

能优化：

- 数据格式转换稳定性。
- dataloader robustness。
- before/contact/hold/release 阶段覆盖。
- 左右 sensor coverage。

不能解决：

- true force label 缺失。
- true slip label 缺失。
- force/slip metric 无效。

### 完成标准

- bridge dataset 能稳定处理更多 cube trials。
- manifest 能覆盖多 trial / 多 stage / 多 side。
- force/slip metric 仍保持 invalid，除非 labels 被后续步骤补齐。

---

## Step 6 — 多采 cube trials：force-oriented protocol

### 目标

让 force labels 变得可用，而不是继续使用 0 force placeholder。

### 推荐 protocol variants

```text
no_contact
light_contact
medium_contact
firm_contact
over_contact
```

### 需要记录

- tactile frames。
- gripper close target / finger joint position。
- contact state。
- contact force。
- force side：left / right / both。
- normal / shear force components。
- force label source：`sim_contact_sensor` / `proxy`。

### 关键要求

优先修复 / 验证仿真 contact force logging：

- `force_by_side_n` 不能一直为 0。
- 最好能得到三轴 force 或至少可解释的 normal force。
- 如果只能使用 penetration / AABB overlap 估计 force，必须标记为 `proxy-only`。

### 完成标准

- 至少存在多个 force magnitude levels。
- force labels 非零且随 close level / contact state 有合理变化。
- force label source 被记录。
- 才允许把 force lane 从 `smoke only` 提升到 `limited/proxy evaluation`。

---

## Step 7 — 多采 cube trials：slip-oriented protocol

### 目标

让 slip labels 有真实事件来源，而不是 all-zero placeholder。

### 推荐 protocol variants

```text
stable_hold
low_force_slip
lift_then_slip
release_before_drop
external_disturbance_slip
friction_reduced_slip
```

### slip label 定义建议

参考 Sparsh 的二分类 slip 语义，但由仿真状态生成：

```text
slip_label = 1
if contact_detected
and tangential relative motion between cube and fingertip/gelpad exceeds threshold
within a short horizon window
```

可用信号：

- cube pose over time。
- fingertip / gelpad pose over time。
- object 与 gripper 的相对切向位移。
- object 下滑 / 旋转 / 掉落。
- contact side。
- gripper close target / contact force。

### 需要记录

- slip onset time。
- slip_label per frame or per sample。
- no-slip / slip class balance。
- delta_force，如果 force labels 已可用。
- label threshold 与 horizon。

### 完成标准

- 同时有 slip 与 no-slip 样本。
- slip onset 可复现。
- slip label 不再来自 `contact_detected`，而来自相对滑动判据。
- 才允许 slip lane 从 `smoke only` 提升到 `limited/proxy evaluation`。

---

## Step 8 — Label validity gate

### 目标

决定当前 dataset 到底只能 smoke，还是可以开始 limited evaluation。

### Gate 状态

```text
FORMAT_BRIDGE_PASS / BLOCKED
FORCE_SMOKE_PASS / BLOCKED
SLIP_SMOKE_PASS / BLOCKED
FORCE_LABEL_VALIDITY = invalid / proxy / sim-valid
SLIP_LABEL_VALIDITY = invalid / proxy / sim-valid
METRIC_VALIDITY = invalid / limited-proxy / limited-sim
```

### 进入真正 evaluation 的最低条件

Force evaluation 最低条件：

- force labels 非零。
- force labels 与 contact / close level 单调或可解释相关。
- label source 不再是纯 placeholder。
- sample 数量覆盖多个 force levels。

Slip evaluation 最低条件：

- 同时存在 slip / no-slip labels。
- slip onset 由仿真相对运动判据生成。
- slip labels 可复现。
- 最好有 delta_force 或明确说明无 delta_force 的替代方式。

---

## Step 9 — 总结与下一阶段决策

### 输出

```text
tactile_grasp/artifacts/phase4_sparsh_cube/summary.md
tactile_grasp/artifacts/phase4_sparsh_cube/bridge_report.json
```

### 必须回答

- 当前 Phase3 cube artifacts 是否能转成 Sparsh-like 格式？
- Force dataloader 是否能跑？
- Force model forward 是否能跑？
- Slip dataloader 是否能跑？
- Slip model forward 是否能跑？
- 当前 force/slip labels 是 invalid / proxy / sim-valid？
- 是否需要继续采 bridge robustness batch？
- 是否需要采 force-oriented cube trials？
- 是否需要采 slip-oriented cube trials？

---

# Updated Exit Criteria

Phase4 完成必须全部满足：

- [ ] 只使用 cube 数据。
- [ ] 不考虑 forcefield。
- [ ] 不处理 `chips_can` / `cracker_box`。
- [ ] 不采集实物数据。
- [ ] 生成或明确规划 `phase4_sparsh_cube/cube_phase3_bridge`。
- [ ] Force estimation lane 完成 dataloader / forward smoke，或给出 exact blocker。
- [ ] Slip detection lane 完成 dataloader / forward smoke，或给出 exact blocker。
- [ ] 当前 placeholder / proxy labels 被明确标记，不能用于可信 metrics。
- [ ] 多采 cube trials 的下一步被分成：
  - bridge robustness batch，
  - force-oriented protocol，
  - slip-oriented protocol。
- [ ] 明确 label validity gate，说明什么时候能从 smoke test 进入 limited evaluation。
- [ ] 不做 Sparsh adaptation / fine-tuning / closed-loop control / GraspNet / RL。

---

# 可行性最终判断

**可行，但第一阶段成功标准必须是“连通性验证”，不是“模型性能评估”。**

当前 cube Phase3 artifacts 已足够支持：

- tactile RGB frame 读取。
- trial / side / stage / contact flag 组织。
- `dataset_gelsight*` image pickle 构造。
- `dataset_slip_forces.pkl` 结构壳构造。
- Sparsh force/slip dataloader smoke。
- 在 checkpoint 可用时尝试 model forward。

当前 cube Phase3 artifacts 不足以支持：

- 可信 force estimation metric。
- 可信 slip detection metric。
- 证明 Sparsh 在 cube grasp 上有效。

要优化，不能只“多采同样 clean contact_hold trials”。需要分步采：

1. **Bridge robustness batch**：让格式转换和 dataloader 稳。
2. **Force-oriented cube trials**：让 force labels 有非零、可解释来源。
3. **Slip-oriented cube trials**：让 slip labels 有受控事件来源。

只有当 force/slip labels 通过 validity gate 后，Phase4/Phase5 才能开始 limited evaluation 或后续 adaptation planning。

---

# 一句话总结

**Phase4 先把现有 cube 仿真数据接到 Sparsh force/slip 数据管线上；当前 labels 只允许 smoke / placeholder，不做性能结论；随后通过 bridge robustness、force-oriented、slip-oriented 三类 cube trials 逐步补齐有效 labels，再决定是否进入 limited evaluation。**
