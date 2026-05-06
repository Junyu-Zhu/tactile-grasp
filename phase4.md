# Phase 4：Cube-only Phase3 artifacts → Sparsh force/slip bridge feasibility

## 目标

Phase 4 当前收窄为一个 **format bridge / smoke-test 阶段**：

> **只使用当前仿真中已经采集的 cube Phase3 artifacts，不采集实物数据，不考虑 forcefield，把 Phase3 cube tactile/logging 数据整理成尽量接近 Sparsh force estimation / slip detection 的输入格式，并验证 Sparsh 现有模块能否与这些仿真数据连通。**

本阶段优先回答“能不能接上数据管线”，不是回答“模型效果是否可信”。

---

## Scope Freeze

### 只考虑

- object：`cube`
- data source：`tactile_grasp/artifacts/phase3/phase3_cube_*`
- Sparsh task：
  - force estimation
  - slip detection
- 目标：
  - 分析 Phase3 artifact 格式与 Sparsh force/slip loader 格式差异。
  - 生成或规划一个 derived Sparsh-compatible view。
  - 尝试让 Sparsh 现有 force/slip 模块完成 dataloader + model forward / smoke test。

### 不考虑

- forcefield demo / forcefield decoder。
- chips_can / cracker_box。
- 实物 GelSight Mini 数据采集。
- Sparsh adaptation / fine-tuning。
- closed-loop control。
- GraspNet / RL。
- 把当前 Phase3 labels 伪装成真实 force/slip ground truth。

---

## Source of Truth

### Phase3 cube artifacts

- `tactile_grasp/artifacts/phase3/phase3_cube_*/meta.json`
- `tactile_grasp/artifacts/phase3/phase3_cube_*/robot_state.json`
- `tactile_grasp/artifacts/phase3/phase3_cube_*/frame_map.csv`
- `tactile_grasp/artifacts/phase3/phase3_cube_*/tactile/*_tactile_rgb.npy`
- optional preview：`*_tactile_rgb.png`

当前检查到的 cube trials：

```text
phase3_cube_0001
phase3_cube_0002
phase3_cube_0003
```

> 注：之前 `phase3.md` 曾记录 10 条 cube clean trials，但当前 artifacts 目录下可见的是 3 条 cube trials。因此 Phase4 当前以实际存在的 cube artifacts 为准。

### Sparsh force/slip entry points

- force config：`sparsh/config/experiment/downstream_task/force/gelsight_dino.yaml`
- slip config：`sparsh/config/experiment/downstream_task/slip/gelsight_dino.yaml`
- shared data config：`sparsh/config/data/gelsight_force.yaml`
- combined force/slip dataset：`sparsh/tactile_ssl/data/vision_based_forces_slip_probes.py`
- force tester：`sparsh/tactile_ssl/test/test_t1_force.py`
- slip tester：`sparsh/tactile_ssl/test/test_t2_slip.py`
- Sparsh dataset loading helper：`sparsh/tactile_ssl/data/digit/utils.py`

---

# 当前 Phase3 artifact 格式

每条 cube trial 的结构：

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

其中 `contact_state` 当前有：

```text
contact_detected
contact_sides
geometry_contact_sides
force_contact_sides
force_by_side_n
max_force_n
geometry
```

重要观察：当前 cube artifact 中可看到 `contact_detected` / geometry contact，但 `force_by_side_n` 示例为 0.0。这意味着当前数据有接触事件与触觉图像变化，但没有可直接作为 force estimation ground truth 的可靠非零 force label。

---

# Sparsh force/slip 期望格式

当前 `gelsight_force.yaml` 默认使用：

```text
_target_: tactile_ssl.data.vision_based_forces_slip_probes.VisionForceSlipDataset
path_dataset: ${paths.data_root}/tacbench_data/T1_force/gelsight/
out_format: concat_ch_img
num_frames: 2
frame_stride: 5
```

`VisionForceSlipDataset` 通过 `load_dataset_forces(config, dataset_name, sensor)` 读取：

```text
<path_dataset>/<dataset_name>/dataset_slip_forces.pkl
<path_dataset>/<dataset_name>/dataset_gelsight*   # pickle list of images / arrays / buffers
```

## `dataset_slip_forces.pkl` 至少需要

```python
{
  "in_contact": np.ndarray,        # per-frame contact flag
  "trajectories": {
    trajectory_id: {
      "indexes": np.ndarray,      # indexes into dataset_gelsight image list
      "forces": np.ndarray,       # shape [T, 3], force labels
      "slip_label": np.ndarray,   # shape [T], 0/1 labels
      # some loaders may use delta_forces in older paths
    }
  }
}
```

## Force task loader output

Force tester expects dataloader samples like：

```python
sample["image"]
sample["force"]
```

`TestForceSL` reports RMSE / correlation only if `force` is true label.

## Slip task loader output

Slip tester expects dataloader samples like：

```python
sample["image"]
sample["slip_label"]
sample["delta_force"]
```

`TestSlipSL` reports balanced accuracy / classification metrics / delta-force metrics only if `slip_label` and `delta_force` are meaningful labels.

---

# Phase3 → Sparsh 格式映射

## 可直接映射的部分

| Phase3 | Sparsh-compatible view | 可行性 |
| --- | --- | --- |
| `*_tactile_rgb.npy` | `dataset_gelsight_cube.pkl` image list | 高 |
| `frame_map.csv.frame_id` | image index / frame metadata | 高 |
| `frame_map.csv.action_stage` | trajectory stage metadata | 高 |
| `frame_map.csv.contact_detected` | `in_contact` | 中-高：可作为 contact flag，不等于 slip label |
| `robot_state.samples[].tactile_frame_ids` | frame → state alignment | 高 |
| `meta.success_label` | trial-level grasp success metadata | 中：不用于 force/slip metric |

## 缺失或不可靠的部分

| Sparsh expected | 当前 Phase3 cube artifact 状态 | 影响 |
| --- | --- | --- |
| true `forces[:, 3]` | 当前 `force_by_side_n` 示例为 0.0；缺少可靠三轴 force label | 不能做可信 force metric |
| true `slip_label` | 当前 protocol 是 contact_hold，没有明确 slip event label | 不能做可信 slip metric |
| true `delta_force` | 可由 forces 差分得到，但 forces 本身不可靠 | 不能做可信 slip delta-force metric |
| sufficient class diversity | 当前 cube trials 少，且多为 clean success/contact_hold | 不足以评估 slip classifier |

---

# 可行性分析

## 结论概览

| 目标 | 可行性 | 说明 |
| --- | --- | --- |
| 把 cube Phase3 tactile frames 整理成 Sparsh-like image list | 高 | `.npy`/`.png` 可转为 pickle image list，路径和 stage 可由 `frame_map.csv` 组织 |
| 构建 `dataset_slip_forces.pkl` 的结构壳 | 高 | `in_contact` 和 `trajectories.indexes` 可由 frame_map / robot_state 生成 |
| 跑通 Sparsh dataloader smoke | 中-高 | 需要生成符合 loader 预期的 pickle 和 config override；不依赖实物数据 |
| 跑通 Sparsh model forward smoke | 中 | 取决于 checkpoint / package / config 是否可用；数据形状可适配 |
| 获得可信 force estimation metrics | 低 | 当前 artifacts 缺少真实 force label，`force_by_side_n` 似乎为 0 |
| 获得可信 slip detection metrics | 低 | 当前 artifacts 缺少真实 slip events / slip labels |
| 用当前数据证明 Sparsh 在 cube grasp 上有效 | 不可行 | 当前只能证明“连接上”，不能证明模型有效 |

## 推荐判断

本阶段是可行的，但应定义为：

> **Sparsh force/slip data-contract bridge + forward smoke test**

而不是：

> Sparsh force/slip evaluation

因为当前仿真 artifacts 有足够的信息构建 Sparsh-like 数据容器，但没有足够真实监督信号支撑 force/slip 指标。

---

# 推荐技术路线

## Option A — Recommended：Derived Sparsh-compatible dataset view

不修改原始 Phase3 artifacts，另建 derived 输出：

```text
tactile_grasp/artifacts/phase4_sparsh_cube/
  cube_phase3_bridge/
    dataset_gelsight_cube.pkl
    dataset_slip_forces.pkl
    manifest.csv
    README.md
```

### 生成内容

1. `dataset_gelsight_cube.pkl`
   - pickle list。
   - 每个元素为 RGB image array 或 Sparsh `load_sample_from_buf` 能接受的 buffer / ndarray。
   - 数据来自 `phase3_cube_*/tactile/*_tactile_rgb.npy`。

2. `dataset_slip_forces.pkl`
   - 结构匹配 Sparsh loader。
   - `in_contact` 来自 `frame_map.csv.contact_detected` 或 `robot_state.contact_state.contact_detected`。
   - `trajectories[trial_id]["indexes"]` 指向对应 image list index。
   - `forces` 暂时只能是：
     - placeholder zeros，或
     - geometry/contact proxy，或
     - 如果后续修复 sim contact sensor，则使用真实 sim force。
   - `slip_label` 暂时只能是：
     - all-zero no-slip placeholder，或
     - 后续专门仿真 slip protocol 生成。

3. `manifest.csv`
   - 保留所有 provenance：
     - trial_id
     - side
     - frame_id
     - source_path
     - action_stage
     - contact_detected
     - robot_state_index
     - label_source：`placeholder` / `sim_contact_sensor` / `geometry_proxy`
     - label_valid_for_metrics：true/false

### 为什么推荐

- 不破坏 Phase3 raw artifacts。
- 最大限度复用 Sparsh 原 loader/config。
- 明确区分“格式连通”和“监督指标有效”。
- 后续如果有真实 sim force/slip labels，只需替换 derived label 字段。

---

## Option B：写自定义 PyTorch Dataset adapter

新增一个 dataset 类，直接读取 Phase3 `frame_map.csv` / `robot_state.json`，输出：

```python
{
  "image": ...,
  "force": ...,
  "delta_force": ...,
  "slip_label": ...,
}
```

### 优点

- 不需要伪装成 `dataset_slip_forces.pkl`。
- 更清晰表达 Phase3 native schema。

### 缺点

- 需要改 Sparsh config / 新增 dataset 类。
- 与“格式修改为与 Sparsh 对应”的目标相比，侵入更大。

当前不推荐作为第一步；可以作为 Option A 不足时的 fallback。

---

# Phase 4 Task List（更新版）

## Step 1 — 锁定 cube-only scope

### 要做

- 只枚举 `tactile_grasp/artifacts/phase3/phase3_cube_*`。
- 确认每条 cube trial 的：
  - `meta.json`
  - `robot_state.json`
  - `frame_map.csv`
  - tactile RGB frames
- 不处理 `chips_can` / `cracker_box`。

### 完成标准

- 生成 cube trial inventory。
- 明确当前可用 cube trial 数量。

---

## Step 2 — 分析 Phase3 cube artifact → Sparsh force/slip contract

### 要做

- 读取 `frame_map.csv`。
- 读取 `robot_state.json`。
- 统计：
  - frame count per trial / side / stage。
  - contact frames vs no-contact frames。
  - 是否存在非零 force readings。
  - 是否存在可用 slip label。

### 完成标准

- 形成 contract gap report：
  - image：PASS
  - in_contact：PASS / approximate
  - force：BLOCKED or proxy-only
  - slip_label：BLOCKED or placeholder-only
  - delta_force：BLOCKED unless force usable

---

## Step 3 — 生成 derived Sparsh-compatible cube dataset

### 要做

输出目录：

```text
tactile_grasp/artifacts/phase4_sparsh_cube/cube_phase3_bridge/
```

生成：

```text
dataset_gelsight_cube.pkl
dataset_slip_forces.pkl
manifest.csv
README.md
```

### 完成标准

- Sparsh loader 能找到：
  - `dataset_slip_forces.pkl`
  - `dataset_gelsight*`
- `manifest.csv` 能追溯每个 sample 到 Phase3 原始 frame。
- label validity 在 README / manifest 中明确标注。

---

## Step 4 — 配置 Sparsh force estimation smoke

### 要做

- 使用 `gelsight_dino` force config。
- 将 `path_dataset` override 到 derived cube bridge root。
- 将 `list_datasets_test` / `test.data.dataset_name` 指向 `cube_phase3_bridge`。
- 尝试 dataloader instantiate。
- 如果 checkpoint 可用，尝试 model forward。

### 完成标准

Force lane 必须输出：

- `PASS_FORWARD`：dataloader + model forward 成功。
- `PASS_DATALOADER_ONLY`：dataloader 成功，checkpoint/model 阻塞。
- `BLOCKED-with-exact-contract`：明确缺少哪一个契约。

### 注意

即使 forward 成功，当前也只能叫 smoke test，不能叫可信 force evaluation。

---

## Step 5 — 配置 Sparsh slip detection smoke

### 要做

- 使用 `gelsight_dino` slip config。
- 复用同一个 derived cube bridge dataset。
- 尝试 dataloader instantiate。
- 如果 checkpoint 可用，尝试 model forward。

### 完成标准

Slip lane 必须输出：

- `PASS_FORWARD`：dataloader + model forward 成功。
- `PASS_DATALOADER_ONLY`：dataloader 成功，checkpoint/model 阻塞。
- `BLOCKED-with-exact-contract`：明确缺少哪一个契约。

### 注意

当前 cube contact_hold 数据没有真实 slip labels；不能把 all-zero placeholder 结果当成 slip detection 结论。

---

## Step 6 — 总结连通性与下一步

### 要做

总结：

- Phase3 cube → Sparsh format bridge 是否成功。
- Force dataloader / forward 是否成功。
- Slip dataloader / forward 是否成功。
- 哪些 labels 是 placeholder / proxy。
- 是否需要后续仿真补采：
  - nonzero contact force。
  - controlled slip events。
  - label validity checks。

### 完成标准

明确给出：

- `FORMAT_BRIDGE_PASS / BLOCKED`
- `FORCE_SMOKE_PASS / BLOCKED`
- `SLIP_SMOKE_PASS / BLOCKED`
- `METRIC_VALIDITY = invalid / limited / valid`

---

# Updated Exit Criteria

Phase4 完成必须全部满足：

- [ ] 只使用 cube Phase3 artifacts。
- [ ] 不包含 forcefield 路线。
- [ ] 不处理 chips_can / cracker_box。
- [ ] 生成或明确规划 derived Sparsh-compatible cube dataset。
- [ ] Force estimation lane 完成 dataloader / forward smoke 或给出 exact blocker。
- [ ] Slip detection lane 完成 dataloader / forward smoke 或给出 exact blocker。
- [ ] 明确当前 force/slip metrics 是否有效；若使用 placeholder/proxy labels，必须标记为不可用于性能结论。
- [ ] 不采集实物数据。
- [ ] 不做 adaptation / fine-tuning / closed-loop control / GraspNet / RL。

---

# 可行性最终判断

**可行，但目标必须限定为“连通性验证”。**

当前 Phase3 cube artifacts 已经足够支持：

- 读取 tactile RGB frames。
- 按 trial/stage/side 建立 trajectories。
- 构造 Sparsh-like `dataset_gelsight*` image pickle。
- 构造 `dataset_slip_forces.pkl` 的结构壳。
- 尝试 Sparsh force/slip dataloader 与 model forward。

但当前 Phase3 cube artifacts 还不足以支持：

- 可信 force estimation metric。
- 可信 slip detection metric。
- 证明 Sparsh 在仿真 cube grasp 上“有效”。

原因是：

- force labels 缺失或当前为 0。
- slip labels 缺失。
- contact_hold protocol 本身缺少受控 slip event。
- 数据量仅 3 条 cube trials，远不足以做评估。

因此 Phase4 应以如下成功定义收尾：

> **成功 = 当前仿真 cube artifacts 可以被无损追溯地转换 / 映射成 Sparsh force/slip loader 可消费的格式，并完成 dataloader + forward smoke；所有 force/slip 指标只作为占位或无效指标记录，不作为论文/实验结论。**

---

# 下一阶段建议

如果 Phase4 bridge 成功，下一阶段再决定是否：

1. 在仿真中补充可用 force labels：修复 / 启用有效 contact force logging，而不是使用 0 force placeholder。
2. 设计 cube controlled-slip protocol：让 slip_label 有真实事件来源。
3. 再考虑小规模真实 GelSight Mini 数据采集。
4. 最后才进入 Sparsh adaptation / control。 
