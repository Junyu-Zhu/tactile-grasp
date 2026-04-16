# Phase 1：UR5e 仿真本体 Bring-up Checklist

## 目标
这一阶段只回答一个问题：

> **能否在 Isaac Sim 里稳定地加载、reset、控制 `UR5e + Robotiq` 这个仿真 embodiment？**

本阶段**不做**：
- GelSight Mini 挂载
- tactile image 调试
- Sparsh 接入
- closed-loop control
- GraspNet
- RL

---

## Source of Truth
- 主脚本：`tactile_grasp/ur5_sim.py`
- 主 USD 配置：`tactile_grasp/assets/ur5_usd/ur5.py`
- 主 USD 资产：`tactile_grasp/assets/ur5_usd/ur5_moveit.usd`
- table 资产：`tactile_grasp/assets/ur5_usd/table.usd`
- rigid object：`tactile_grasp/ycb_objects/YcbBanana/model.urdf`
- 旧 URDF 参考：`tactile_grasp/assets/ur5DHGS.urdf`
- 备选 URDF 参考：`tactile_grasp/assets/ur5RQGS.urdf`

原则：**Phase 1 只维护一个主入口，不并行改多个 UR5 脚本。**

---

## Step 1：锁定 Phase 1 范围
### 要做
- 确认 Phase 0 contract 已接受
- 确认本阶段目标仅为：`stable UR5e + Robotiq embodiment bring-up`

### 完成标准
- 能用一句话描述本阶段目标

### 禁止
- 不允许提前进入传感器、Sparsh、控制逻辑

---

## Step 2：选定唯一入口脚本和资产路径
### 要做
- 确认 `ur5_sim.py` 为主执行脚本
- 记录主 USD 配置 / USD 资产路径，以及旧 URDF fallback 路径

### 完成标准
- 文档里明确写出唯一 source-of-truth

---

## Step 3：验证主资产导入与 fallback 路径
### 要做
- 用最简参数运行 `ur5_sim.py`
- 验证 `ur5.py` + `ur5_moveit.usd` 能被 Isaac Sim 正常加载
- 验证 table / banana 本地资产路径工作正常
- 若主 USD 路径失败，再记录旧 URDF fallback 的导入错误

### 完成标准
- 机器人成功导入，或者已有一份明确的导入错误清单

### 失败 fallback
- 若主 USD 路径有问题，先检查 `assets/ur5_usd/ur5.py` 与 `ur5_moveit.usd`
- 若仍不稳，再回退检查 `ur5DHGS.urdf` / `ur5RQGS.urdf`

---

## Step 4：拿到一个干净的 spawn
### 要做
- 成功生成 1 个 UR5e
- 加入 table
- 加入 1 个 rigid object
- 确保 base pose 合理

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
- 还没稳定到 pre-grasp 前，不准开始挂 GelSight Mini

---

## Step 9：Phase 1 Review
### Exit Criteria
必须全部通过：
- [x] 已选定唯一 source-of-truth 脚本
- [x] UR5e + Robotiq 成功导入
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
- source-of-truth 记录（`tactile_grasp/ur5_sim.py`）
- 运行命令（见 `tactile_grasp/artifacts/phase1_validation_report.md`）
- clean spawn 截图（可选后补）
- joint name mapping（见 `tactile_grasp/artifacts/phase1_validation_latest.log`）
- gripper target 记录（见 `tactile_grasp/artifacts/phase1_validation_latest.log`）
- reset 测试结果（见 `tactile_grasp/artifacts/phase1_validation_latest.log`）
- pre-grasp 截图或短视频（可选后补）
- 导入错误与修复日志（见 `tactile_grasp/artifacts/phase1_validation_latest.log`）

## 本次 Phase 1 验证结论
- 运行环境：`conda` 环境 `tacex`
- 通过命令：`--headless --phase1-checks --gripper-cycles 20 --reset-trials 20 --pregrasp-trials 20`
- 结果：`[RESULT] Phase 1 validation PASSED.`
- 正式报告：`tactile_grasp/artifacts/phase1_validation_report.md`
- 完整日志：`tactile_grasp/artifacts/phase1_validation_latest.log`

---

## Phase 1 的一句话总结
**先证明 UR5e + Robotiq 仿真本体是稳定、可控、可 reset、可到预抓取位的，再开始任何传感器和算法工作。**
