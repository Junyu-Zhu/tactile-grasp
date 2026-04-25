# Phase 1：Integrated UR5e + Robotiq + connector + GSmini Bring-up Checklist

## 目标
这一阶段只回答一个问题：

> **能否在 Isaac Sim 里稳定地加载、reset、控制 `UR5e + Robotiq + connector + GSmini` 这个 integrated embodiment（先不验证 tactile output）？**

本阶段**不做**：
- tactile image / tactile output 调试
- Sparsh 接入前的任何触觉算法验证
- Sparsh 接入
- closed-loop control
- GraspNet
- RL

---

## Source of Truth
- Canonical robot source：`tactile_grasp/environment/ur5_robotiq_GSmini/urdf/ur5_robotiq_GSmini.urdf`
- Phase 1 主脚本：`tactile_grasp/ur5_sim.py`
- Generated USD（当前 Phase 1 会生成/复用）：`tactile_grasp/environment/ur5_robotiq_GSmini/usd/ur5_robotiq_GSmini.usd`，只作为从 canonical URDF 派生出来的加载便利资产，不是 source-of-truth
- table 资产：`tactile_grasp/environment/table.usd`
- rigid object：`tactile_grasp/ycb_objects/YcbBanana/model.urdf`
- 已退役的旧 Phase 1 truth：`tactile_grasp/assets/ur5_usd/ur5_moveit.usd`
- 历史 URDF 参考（仅 legacy 对照）：`tactile_grasp/assets/ur5DHGS.urdf`、`tactile_grasp/assets/ur5RQGS.urdf`

原则：**Phase 1 只维护一个 canonical robot truth 和一个主入口，不并行改多个 UR5 脚本，也不再把 `ur5_moveit.usd` 当 source-of-truth。**

---

## Step 1：锁定 Phase 1 范围
### 要做
- 确认 Phase 0 contract 已接受
- 确认本阶段目标仅为：`stable integrated UR5e + Robotiq + connector + GSmini embodiment bring-up（excluding tactile-output work）`

### 完成标准
- 能用一句话描述本阶段目标

### 禁止
- 不允许提前进入传感器、Sparsh、控制逻辑

---

## Step 2：选定唯一入口脚本和 canonical robot truth
### 要做
- 确认 `ur5_sim.py` 为主执行脚本
- 记录 canonical robot URDF 路径
- 明确标注：后续若生成 USD，只能是从 canonical URDF 派生出来的 convenience asset
- 明确标注 `ur5_moveit.usd` 已退役，不再作为 source-of-truth

### 完成标准
- 文档里明确写出唯一 canonical robot truth 与唯一 Phase 1 入口

---

## Step 3：验证 canonical robot 导入与 secondary asset 路径
### 要做
- 用最简参数运行 `ur5_sim.py`
- 优先验证 `ur5_robotiq_GSmini.urdf` 这套 integrated embodiment 能被 Isaac Sim 正常加载
- 验证 table / banana 本地资产路径工作正常
- 记录 generated USD 只作为 derived loading convenience，并明确它来自 canonical URDF
- 若导入失败，先记录 canonical URDF 及其依赖 mesh 的导入错误

### 完成标准
- canonical robot 成功导入，或者已有一份明确的导入错误清单

### 失败 fallback
- 若 canonical URDF 路径有问题，先检查 `tactile_grasp/environment/ur5_robotiq_GSmini/urdf/ur5_robotiq_GSmini.urdf` 与其 mesh 引用
- 若存在 generated USD，再核对它是否与 canonical URDF 保持一致；不要倒过来把 USD 当 truth

---

## Step 4：拿到一个干净的 spawn
### 要做
- 成功生成 1 个 UR5e
- 加入 table
- 加入 1 个 rigid object
- 确保 integrated embodiment 的 base pose 合理

### 完成标准
- robot + table + object 同时存在且稳定
- 保存一张截图

---

## Step 5：验证 arm joint actuation
### 要做
- 确认 6 个 arm joints 都能被命令驱动
- 只做小范围 reach motion

### 完成标准
- 6 个 arm joints 都能到目标位并稳定收敛
- 记录 joint name 和异常关节

### 禁止
- arm 还不稳时，不准先调 gripper

---

## Step 6：验证 Robotiq 开合控制
### 要做
- 确认 gripper joint 响应正常
- 验证 mimic 逻辑或最小化单关节逻辑
- 连续测试 open/close
- 明确本步骤只验证 integrated embodiment 的运动学稳定性，不验证 tactile output

### 完成标准
- 连续 20 次 open/close 不出现卡死、炸关节、资产损坏
- 记录 gripper joint 名称与 target 值

### 失败 fallback
- 若完整 mimic 太不稳，先用最小单关节控制约定跑通

---

## Step 7：实现 deterministic reset
### 要做
- reset robot pose
- reset gripper pose
- reset object pose
- reset计数器/状态

### 完成标准
- 连续 20 次 reset 无漂移、无 corruption

### 禁止
- reset 不稳定前，不准进入下一阶段

---

## Step 8：加入固定 pre-grasp reach
### 要做
- 定义一个固定 pre-grasp pose
- 实现 reset -> pre-grasp -> reset 的稳定流程

### 完成标准
- 每次都能稳定移动到预抓取位
- 暂时不要求抓取成功

### 禁止
- 还没稳定到 pre-grasp 前，不准开始任何 tactile output / tactile image bring-up

---

## Step 9：Phase 1 Review
### Exit Criteria
必须全部通过：
- [x] 已选定唯一 source-of-truth 脚本
- [x] UR5e + Robotiq + connector + GSmini integrated embodiment 成功导入（tactile-output work 仍排除）
- [x] table + one object 稳定存在
- [x] arm joints 可控
- [x] gripper open/close 可重复
- [x] deterministic reset 20 次通过
- [x] robot 能稳定到 fixed pre-grasp pose

### 规则
- 任一项失败，就继续修 Phase 1
- **不允许**带着 Phase 1 的问题进入 Phase 2

---

## 本阶段需要保存的产物
- source-of-truth 记录（`tactile_grasp/environment/ur5_robotiq_GSmini/urdf/ur5_robotiq_GSmini.urdf` + `tactile_grasp/ur5_sim.py`）
- 运行命令（见 `tactile_grasp/artifacts/phase1_validation_report.md`）
- clean spawn 截图（可选后补）
- joint name mapping（见 `tactile_grasp/artifacts/phase1_validation_latest.log`）
- gripper target 记录（见 `tactile_grasp/artifacts/phase1_validation_latest.log`）
- reset 测试结果（见 `tactile_grasp/artifacts/phase1_validation_latest.log`）
- pre-grasp 截图或短视频（可选后补）
- 导入错误与修复日志（见 `tactile_grasp/artifacts/phase1_validation_latest.log`）

## 本次 Phase 1 验证结论
- 运行环境：`conda` 环境 `tacex`
- 通过命令：`--headless --regenerate-robot-usd --phase1-checks --gripper-cycles 20 --reset-trials 20 --pregrasp-trials 20`
- 结果：`[RESULT] Phase 1 validation PASSED.`
- 关键控制约定：gripper 以 `finger_joint` 为主控制关节，并协同控制 `left_inner_knuckle_joint` / `right_outer_knuckle_joint` / `right_inner_knuckle_joint` / `left_inner_finger_joint` / `right_inner_finger_joint`；当前 visible close target 采用 `0.45 rad`，内指关节按 Robotiq mimic 四连杆关系镜像到 `-0.45 rad`，使左右 `inner_finger` 上的 pad / GSmini 指尖保持平行闭合，而不是夹爪式斜向闭合
- Phase 1 mimic 稳定化约定：Isaac 的 URDF importer 会把 Robotiq 四连杆 mimic/闭链近似成开链 articulation，单靠 position drive 时内指关节会在约 `-0.16~-0.18 rad` 附近滞后，导致 GSmini 斜向闭合并触发 gripper demo 失败。因此 Phase 1 在 `step_scene()` 中对 gripper-only mimic joints 做有界的 kinematic stabilization（`0.02 rad/step`），只用于 bring-up/可视化控制稳定性，不代表后续 tactile contact dynamics 的最终物理模型。
- Phase 1 碰撞/挂载近似约定：为保证 integrated embodiment 在 Isaac Sim 中稳定导入和开合，当前保留 UR5/table/object 的正常碰撞，并把指尖有效接触近似为左右内指上的 pad box collision；robot articulation self-collision 关闭以避免 Robotiq 内部 mesh 自碰撞阻止 mimic 闭合；pad、GSmini connector/base/soft 作为 visual/collision 直接嵌入左右 `left_inner_finger` / `right_inner_finger`，不再通过 `inner_finger_pad` fixed joint 或单独 connector fixed-joint 刚体挂载，从而避免末端加速时 fixed-joint solver compliance 造成 pad/connector/GSmini 相对内指摆动。本阶段通过并不等价于 tactile 接触几何已完全保真
- 正式报告：`tactile_grasp/artifacts/phase1_validation_report.md`
- 完整日志：`tactile_grasp/artifacts/phase1_validation_latest.log`
- 2026-04-25 回归验证：`ur5_sim.py --headless --regenerate-robot-usd --phase1-checks --gripper-cycles 2 --reset-trials 1 --pregrasp-trials 1` 通过（arm / gripper / reset / pregrasp 全部 passed），且 `ur5_sim_test.py --headless --regenerate-robot-usd --gripper-cycles 2 --reset-demo-trials 0 --pregrasp-holds 0 --spawn-hold-steps 5` 完成 Step 4/5/6/7/8，无 `[DEMO] Gripper demo failed`。

---

## Phase 1 的一句话总结
**先证明 integrated UR5e + Robotiq + connector + GSmini embodiment 是稳定、可控、可 reset、可到预抓取位的（但暂不做 tactile output），再开始任何传感器输出和算法工作。**
