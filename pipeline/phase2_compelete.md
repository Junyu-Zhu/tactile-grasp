# Phase2 完成状态汇总（UR5 + Robotiq + GSmini + TacEx）

> 文件名按用户要求保留为 `phase2_compelete.md`。Phase2 当前目标是：在 canonical UR5e + Robotiq + GSmini 环境中，完成 4 cm cube 侧向抓取、实时触觉显示、soft-center 对齐、稳定接触闭合与后续 Sparsh/触觉闭环接口预留。

## 1. 当前推荐启动命令

### 1.1 直接启动 GUI 仿真与实时触觉窗口

```bash
cd /home/zjy/Documents/grasp/tactile_grasp
conda activate tacex
python -u ur5_phase2_grasp_sim.py \
  --enable_cameras \
  --device cpu \
  --show_right_tactile \
  --success_mode contact_demo
```

说明：

- `--enable_cameras`：实时 GSmini/TacEx 触觉图像必须开启相机渲染。
- `--device cpu`：当前 reset/rigid pose 写入路径与 PhysX Direct GPU API 容易冲突，Phase2 grasp 默认建议 CPU。
- `--show_right_tactile`：同时显示左右 GSmini 触觉流。
- `--success_mode contact_demo`：当前默认成功标准，适合调试触觉接触与稳定闭合，不要求 10 cm lift/5 s hold。

### 1.2 只保留脚本合并触觉面板，不开 TacEx legacy 窗口

```bash
cd /home/zjy/Documents/grasp/tactile_grasp
conda activate tacex
python -u ur5_phase2_grasp_sim.py \
  --enable_cameras \
  --device cpu \
  --show_right_tactile \
  --success_mode contact_demo \
  --no_tacex_debug_windows
```

### 1.3 恢复旧严格成功标准

```bash
python -u ur5_phase2_grasp_sim.py \
  --enable_cameras \
  --device cpu \
  --show_right_tactile \
  --success_mode lift_hold
```

`lift_hold` 会恢复旧逻辑：要求配置的 object lift margin / hold 条件通过。

---

## 2. 已完成的功能模块

### 2.1 Canonical UR5e + Robotiq + GSmini 环境

相关文件：

- `environment/ur5_robotiq_GSmini/urdf/ur5_robotiq_GSmini.urdf`
- `environment/ur5_robotiq_GSmini/usd/ur5_robotiq_GSmini.usd`
- `ur5_phase1_scene.py`
- `ur5_phase1_reset.py`
- `ur5_phase1_control.py`

当前状态：

- canonical URDF 是机器人/夹爪/GSmini 装配链的 source of truth。
- generated USD 只是 Isaac Sim 派生资产，方便加载；后续不要把 generated USD 当成主源头手工改。
- Phase2 场景将原 YCB banana 替换为 4 cm cube，cube 默认位置用于左右 GSmini 侧向夹取。
- 已保留 Phase1 的 reset、pregrasp、joint target、mimic gripper 稳定写入逻辑。

### 2.2 Phase2 TacEx runtime sensor shell 挂载

相关文件：

- `ur5_phase2_mount.py`
- `ur5_phase2_tactile.py`
- `ur5_phase2_grasp_test.py`

当前状态：

- canonical URDF 负责可见 GSmini/Robotiq 结构与接触碰撞近似。
- Phase2 额外挂载隐藏的 TacEx `Case.usd` / `Gelpad_low_res.usd` runtime shell，用于提供 TacEx `GelSightMiniCfg` 需要的内部 `/Camera` prim。
- runtime shell mesh 被隐藏，并关闭其 physics/collision，避免重复不可见碰撞体影响夹爪闭合。
- camera clipping range 已设置为 `0.024 ~ 0.040 m`，比 TacEx 默认 far plane 更宽，适配当前 UR5/Robotiq/GSmini 装配偏差。

关键接口：

- `phase2_sensor_prim_paths()`：返回左右 TacEx case/gelpad/camera USD prim path。
- `mount_phase2_sensor_shells(...)`：挂载 TacEx runtime shell。
- `validate_phase2_sensor_camera_prims(...)` / `validate_phase2_sensor_mounts(...)`：验证 camera 与 mount 是否存在且位置正确。

### 2.3 TacEx GSmini 触觉配置与实时更新

相关文件：

- `ur5_phase2_tactile.py`
- `TacEx/source/tacex/tacex/gelsight_sensor.py`
- `TacEx/source/tacex/tacex/simulation_approaches/gpu_taxim/taxim_sim.py`

当前状态：

- 已对接 TacEx `GelSightMiniCfg`。
- 默认输出 `tactile_rgb`，可选 `camera_depth` / `camera_rgb`。
- `update_phase2_sensor(sensor, sim, dt=...)` 会先 render，再调用 TacEx sensor update。
- TacEx/Taxim 的基本路径是：camera depth → height_map → indentation_depth → tactile_rgb。

当前默认触觉输出：

```python
output = update_phase2_sensor(sensor, sim, dt=dt)
output["tactile_rgb"]   # live tactile RGB tensor
output.get("camera_depth")  # optional
output.get("height_map")    # TacEx internal output
```

### 2.4 实时触觉窗口与空白背景修复

相关文件：

- `ur5_phase2_grasp_test.py`

当前状态：

- 脚本自带 `TactileRgbPanel`，把 TacEx `tactile_rgb` tensor 镜像到 Isaac UI panel。
- 面板显示前会 normalize RGB，以避免 Taxim 低对比图像看起来像空白。
- 修复了“cube 已经接触，但实时窗口仍只有 GSmini 背景”的问题：
  - 如果 soft-link/cube 几何 AABB 已接触；
  - 但 TacEx camera-depth 渲染出来的 `tactile_rgb` 仍等于 no-contact 背景；
  - 则启用 contact-geometry imprint fallback：把接触几何转成 TacEx/Taxim height map，再渲染回 `tactile_rgb`。

新增 CLI：

```bash
--disable_tactile_contact_imprint
--tactile_contact_imprint_depth_mm 1.5
--tactile_contact_imprint_background_threshold 0.75
```

注意：

- 这个 fallback 不是随意造图，而是利用当前 canonical GSmini soft-link 与 cube 的接触几何，交给 TacEx/Taxim 渲染。
- 将来如果彻底修正 TacEx camera 与物理 soft gel/cube 的真实对齐，让 camera depth 能直接看到接触形变，可逐步关闭或删除 fallback。

### 2.5 Soft-center TCP 与一次 IK 初始目标

相关文件：

- `ur5_phase2_grasp_test.py`

当前状态：

- 已实现 `virtual_soft_center_tcp` 运行时几何层。
- 定义：左右 GSmini `soft_link` mesh AABB center 的平均点。
- IK 仍然解 `ee_link`，但目标位置通过 `ee_link -> virtual_soft_center_tcp` 偏移换算得到。
- 这样可以先由 cube 中心直接计算一个软垫中心对齐的 EE target，而不是完全靠手调 EE offset。

保留 refinement 的原因：

- Isaac/PhysX articulation settling、Robotiq mimic joints、接触解算、reset 后约束漂移都会让实际 soft_link 中心与理论 TCP 有毫米级误差。
- 所以当前方案是：
  1. 用 virtual soft-center TCP 计算初始 IK 目标；
  2. 到位后读取实际 soft_link/cube 几何误差；
  3. 用 bounded refinement 进行少量修正；
  4. 再 latch arm，进入 close。

相关参数：

```bash
--no_align_soft_center
--grasp_soft_center_z_offset 0.0
--soft_center_tolerance 0.005
--soft_center_refine_rounds 4
--soft_center_refine_steps 60
--max_soft_center_refine_step 0.025
```

### 2.6 Force-control close 与安全保护

相关文件：

- `ur5_phase2_grasp_test.py`

当前状态：

- 已配置左右 GSmini gelpad body 的 contact sensor，cube 作为 filter prim。
- close 过程中读取：
  - soft-link/cube 几何接触；
  - contact sensor force；
  - object z lift；
  - gripper joint error。
- 默认 close 会在两侧稳定 force 达标后停止，避免继续挤压 cube。
- 默认启用 object lift guard，防止夹爪闭合时把 cube 向上顶起。

关键参数：

```bash
--force_control_stable_force_threshold 0.5
--force_control_high_force_threshold 8.0
--force_control_stable_steps 1
--disable_force_control
--disable_stable_force_stop
--max_close_object_lift 0.001
--disable_close_object_lift_guard
```

### 2.7 成功标准调整

相关文件：

- `ur5_phase2_grasp_test.py`

当前默认成功模式：

```bash
--success_mode contact_demo
```

`contact_demo` 判定内容：

- soft-center preclose 对齐通过；
- direct motion 到位；
- close command 与 close settle 通过；
- stable two-sided force grasp 通过；
- close safety / object lift guard 通过；
- live tactile contact image change 通过。

旧成功模式：

```bash
--success_mode lift_hold
```

`lift_hold` 保留旧的严格 10 cm lift / hold 风格验证，用于后续真正抓起物体的实验。

当前验证 artifact：

- `artifacts/phase2_grasp_test_20260503T154023Z.json`
- 结果：`passed=true`
- 原因：`success (contact_demo)`
- 触觉变化证据：left/right contact frames 均通过，fallback imprint applied 后 `tactile_rgb` delta 非零。

### 2.8 Artifact 与日志

相关目录：

- `artifacts/`
- `log.md`

当前状态：

- 每次 grasp test 写 JSON/MD artifact。
- artifact 包含：
  - robot/cube 几何；
  - soft-center alignment；
  - close force history；
  - close safety；
  - tactile live stats；
  - contact imprint fallback stats；
  - success evaluation。

---

## 3. 将来修改时必须注意的点

### 3.1 不要把 generated USD 当 source of truth

- 主源头是 canonical URDF：
  - `environment/ur5_robotiq_GSmini/urdf/ur5_robotiq_GSmini.urdf`
- USD 是 Isaac 加载/转换用派生物。
- 如果修改装配链、link/joint、碰撞、mesh path，应优先改 URDF 和转换流程，而不是只改 generated USD。

### 3.2 TacEx runtime shell 不应变成第二套物理传感器

- runtime shell 的目的主要是提供 TacEx camera prim。
- shell mesh 应保持隐藏。
- shell collision/rigid-body 应保持关闭。
- 如果重新启用 shell mesh/collision，很容易产生：
  - 看不见的重复碰撞；
  - gripper close 被阻挡；
  - cube 被额外顶起；
  - tactile camera 被自身 mesh 遮挡。

### 3.3 相机与触觉必须开启 render/cameras

- 触觉窗口和 TacEx camera depth 依赖 Isaac camera pipeline。
- GUI 或 headless 验证都建议显式加：

```bash
--enable_cameras
```

- 没有这个参数时，可能出现触觉图像不更新、窗口空白或 AppLauncher/renderer 行为异常。

### 3.4 当前建议使用 CPU device

- 当前脚本会 reset scene、写 rigid pose/velocity、稳定 mimic gripper。
- 在当前 Isaac/IsaacLab 环境下，GPU PhysX Direct API 容易与这些写入冲突。
- 推荐：

```bash
--device cpu
```

### 3.5 Isaac/Kit 进程清理

测试结束后默认应清理当前用户下残留的：

- `isaacsim`
- `isaac-sim`
- `omni.kit`
- `kit`
- `SimulationApp`
- `ur5_phase2_grasp*`
- `conda run ... ur5_phase2...`

否则后续运行可能遇到 GPU/renderer/session 被占用或旧窗口残留。

### 3.6 不要误删 measured refinement

`virtual_soft_center_tcp` 可以提供理论一次 IK 目标，但不能替代 measured refinement。

原因：

- URDF fixed transform 只能描述静态几何；
- 实际仿真里 link pose 会受 drive、mimic、contact、settle 影响；
- 一次 IK 后的 soft_link center 仍可能偏几毫米到几厘米。

所以后续如果优化抓取控制，建议保留：

- 初始 soft-center TCP target；
- preclose measured correction；
- artifact 中的 soft-center error 记录。

### 3.7 contact imprint fallback 是过渡修复层

当前 tactile image 的 fallback 解决的是：物理几何已经接触，但 TacEx hidden camera path 仍看到背景。

后续如果要进一步追求真实触觉图像，应考虑：

1. 重新对齐 TacEx camera/case/gelpad 与 canonical URDF soft_link；
2. 排查 canonical GSmini mesh 是否遮挡了 TacEx camera；
3. 尽量让 TacEx camera depth 直接看到 cube/gelpad 接触；
4. 再关闭 `--disable_tactile_contact_imprint` 对比。

在真实 camera-depth 路径完全可靠前，不建议删除 fallback。

### 3.8 成功标准分层

- `contact_demo`：用于 Phase2 触觉接触闭环 bring-up。
- `lift_hold`：用于真正抓取并提起物体的严格验证。

不要把两者混在一起，否则会导致触觉闭环调试阶段被 lift/hold 失败误判为整体失败。

---

## 4. 后续结合 Sparsh / 触觉闭环的接口建议

### 4.1 触觉输入接口

当前最直接的输入来自 `update_phase2_sensor(...)`：

```python
output = update_phase2_sensor(sensor, sim, dt=dt)
tactile_rgb = output["tactile_rgb"]
camera_depth = output.get("camera_depth")
height_map = output.get("height_map")
```

建议后续 Sparsh 输入优先使用：

- `tactile_rgb`：和真实 GSmini 图像/模型输入最接近；
- 可选 `height_map` / `camera_depth`：用于调试、监督或几何 sanity check。

### 4.2 Sparsh 分析接口建议

建议新增一个独立 adapter，不要直接把 Sparsh 逻辑写死在 grasp 主循环里：

```python
class SparshTactileAnalyzer:
    def update(self, side: str, tactile_rgb, *, dt: float) -> dict:
        return {
            "stability_score": float,
            "slip_probability": float,
            "contact_patch": {...},
            "recommended_delta_close_rad": float,
            "recommended_delta_ee_world_m": [dx, dy, dz],
        }
```

推荐接入点：

- `_update_tactile_sensors(...)`：拿到最新 tactile frame 后做 Sparsh 推理；
- `_close_gripper_with_force_control(...)`：每个 close step 根据 Sparsh 输出修正 close target；
- lift/hold 阶段：根据 slip prediction 做微小 grip force 或 EE pose 修正。

### 4.3 控制闭环接口建议

当前 close loop 中已有这些可用信号：

- `force_by_side_n`
- `stable_grasp_detected`
- `high_force_detected`
- `object_lift_m`
- `soft_contact_geometry`
- `tactile_contact_imprint_stats`

后续控制建议分三层：

1. **几何层**：soft-center TCP + refinement，负责把软垫放到 cube 侧面中心；
2. **力控层**：现有 contact sensor force，负责避免过紧/过松；
3. **触觉语义层**：Sparsh 预测稳定性/滑移，负责微调力和姿态。

### 4.4 建议的实时修正策略

在 close 阶段：

```text
if slip_probability 高 and force 不高:
    小幅增加 close_rad
elif force 高 or object_lift 增加:
    减小 close_rad 或保持 last_safe_close_rad
elif contact_patch 偏离中心:
    小幅修正 EE x/z 或 wrist orientation
```

在 hold/lift 阶段：

```text
if slip_probability 高:
    先增加微小夹紧量
    若仍滑移，再做 EE 姿态/高度微调
if tactile patch 快速移动:
    降低 lift speed 或暂停 lift
```

### 4.5 Artifact 扩展建议

后续新增 Sparsh 后，建议 artifact 增加：

```json
"sparsh": {
  "enabled": true,
  "model_path": "...",
  "per_side_history_tail": [...],
  "max_slip_probability": 0.0,
  "min_stability_score": 1.0,
  "control_actions": [...]
}
```

这样后续可以直接从 JSON 复盘：

- 哪一帧预测滑移；
- 控制器做了什么修正；
- 修正后 force/tactile/lift 是否改善。

---

## 5. 推荐后续开发顺序

1. **先固定当前 Phase2 contact_demo 行为**
   - 保证 soft-center、stable close、触觉图像变化稳定通过。
2. **加入 Sparsh adapter，只做离线/旁路推理**
   - 先不控制 gripper，只记录 stability/slip 输出到 artifact。
3. **接入 close 阶段的微小夹紧修正**
   - 只允许很小的 `delta_close_rad`，并保留 high-force/object-lift guard。
4. **接入 lift/hold 阶段的 slip 修正**
   - 先降速/暂停，再微调 grip force，最后才考虑 EE 姿态修正。
5. **最后再挑战 `--success_mode lift_hold`**
   - contact_demo 通过只能说明触觉接触闭环正常；真正抓起 cube 仍需要更完整的 grip/lift 策略。

---

## 6. 当前关键文件索引

| 文件 | 作用 |
| --- | --- |
| `ur5_phase2_grasp_sim.py` | Phase2 grasp demo 兼容入口，实际运行 `ur5_phase2_grasp_test.py` |
| `ur5_phase2_grasp_test.py` | 主抓取脚本：soft-center TCP、IK/refinement、force close、tactile panel、success criteria |
| `ur5_phase2_mount.py` | TacEx runtime shell 挂载、camera prim path、mount validation |
| `ur5_phase2_tactile.py` | TacEx GSmini config、sensor init/update、触觉 bring-up helper |
| `ur5_phase1_scene.py` | 场景设计、cube/table/robot asset 加载 |
| `ur5_phase1_reset.py` | reset 与 mimic gripper 稳定状态写入 |
| `ur5_phase1_control.py` | UR5/Robotiq joint target 与 joint name 映射 |
| `artifacts/` | 每次运行的 JSON/MD/runtime log 证据 |

---

## 7. 当前验证记录

最近一次验证：

```text
artifacts/phase2_grasp_test_20260503T154023Z.json
passed: true
reason: success (contact_demo)
```

关键结果：

- soft-center preclose error 通过；
- stable two-sided contact force 通过；
- close safety 通过；
- tactile contact image change 通过；
- strict lift_hold 未通过是预期现象，因为当前默认目标不是 10 cm lift/hold，而是 Phase2 触觉接触闭环 bring-up。

