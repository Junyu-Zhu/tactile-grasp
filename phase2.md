# Phase 2：GelSight Mini 挂载与 Tactile Output Bring-up Checklist

## 目标
这一阶段只回答两个问题：

> **GelSight Mini 能否被正确挂到 Robotiq 上？**  
> **挂上之后，能否输出稳定、可解释的 tactile image？**

本阶段**不做**：
- Sparsh inference
- adaptation
- closed-loop control
- GraspNet
- RL

---

## 依赖
开始本阶段前，必须满足：
- `tactile_grasp/phase1.md` 全部通过
- integrated `UR5e + Robotiq + connector + GSmini` embodiment 已稳定导入、可 reset、可开合、可到 pre-grasp（但尚未做 tactile-output 验证）

---

## Source of Truth
- Canonical robot source：`tactile_grasp/environment/ur5_robotiq_GSmini/urdf/ur5_robotiq_GSmini.urdf`
- Generated USD：`tactile_grasp/environment/ur5_robotiq_GSmini/usd/ur5_robotiq_GSmini.usd`，作为 canonical URDF 的 derived loading asset 使用，不替代 truth
- 已退役的旧 Phase 1 truth：`tactile_grasp/assets/ur5_usd/ur5_moveit.usd`
- Phase 2 主执行脚本：`tactile_grasp/ur5_phase2_sim.py`
- 挂载 helper：`tactile_grasp/ur5_phase2_mount.py`
- GelSight Mini local mesh：`tactile_grasp/assets/meshes/gelsight_mini`
- connector mesh：`tactile_grasp/assets/meshes/gelsight_robotiq_connector`
- TacEx GelSight Mini 资产/装载格式参考：`TacEx/source/tacex_assets/tacex_assets/data/Sensors/GelSight_Mini`
- TacEx 传感器配置参考：`TacEx/source/tacex_assets/tacex_assets/sensors/gelsight_mini/gsmini_cfg.py`
- TacEx 挂载教程参考：`TacEx/docs/source/tutorials/creating_robot_asset_with_sensors.md`

原则：**Phase 2 在 canonical integrated embodiment 之上做挂载校核与 tactile image/output bring-up，不讨论算法。**

---

## Step 1：锁定 Phase 2 范围
### 要做
- 确认本阶段目标仅为：
  - GelSight Mini 挂载
  - tactile output bring-up

### 完成标准
- 能用一句话描述本阶段目标

### 禁止
- 不允许提前进入 Sparsh / control / adaptation

---

## Step 2：锁定挂载几何 source-of-truth
### 要做
- 明确采用 canonical `ur5_robotiq_GSmini.urdf` 作为唯一机器人/挂载几何 truth
- 确定 connector / sensor mesh 使用哪套本地路径
- 记录左/右传感器命名方案
- 若后续生成 robot USD，只把它记为 derived loading convenience，不把它写回 truth

### 完成标准
- 文档里明确写出 canonical 挂载 source-of-truth

### 禁止
- 不要并行维护多套挂载方案，也不要回退到 `ur5_moveit.usd` 作为 truth

---

## Step 3：先做单侧 GelSight Mini 挂载原型
### 要做
- 先选一个 finger 作为主调试侧
- 挂上一个 GelSight Mini case + connector
- 检查局部坐标系和几何位置
- 检查是否明显穿模

### 完成标准
- 单侧传感器挂载稳定
- reset 后位置不漂
- finger 运动时传感器跟随正常

### 失败 fallback
- 如果 connector 太复杂，先用简化固定 mount 占位

---

## Step 4：验证单侧挂载与 gripper 开合兼容
### 要做
- 在挂载单侧传感器后做 open/close 测试
- 观察是否：
  - 自碰撞异常
  - 关节卡死
  - sensor 位姿漂移

### 完成标准
- 单侧挂载后连续 20 次开合稳定

### 禁止
- 单侧都不稳定前，不准继续到 tactile image

---

## Step 5：接入 GelSightMiniCfg
### 要做
- 把 TacEx 的 `GelSightMiniCfg` 接到当前挂载的 sensor prim
- 先只要求输出最基本数据：
  - tactile_rgb
  - 必要时 camera_depth / camera_rgb

### 完成标准
- no-contact 条件下能拿到稳定 tactile image

### 禁止
- 没有 no-contact frame 前，不准做 contact 测试

---

## Step 6：验证 no-contact vs contact tactile image
### 要做
- 记录 no-contact tactile frame
- 记录 contact tactile frame
- 比较两者是否能被肉眼和程序上区分

### 完成标准
- no-contact 与 contact 有稳定、明显差异
- 多次重复结果一致

### 禁止
- no-contact/contact 区分不稳定前，不准进入双侧挂载

---

## Step 7：恢复双侧挂载
### 要做
- 在单侧稳定后，再挂另一侧
- 确认左右命名、坐标、prim path 清楚
- 再次做开合测试

### 完成标准
- 双侧挂载后 gripper 开合仍稳定
- 左右传感器都能输出 no-contact 图像

### 失败 fallback
- 若双侧不稳，可临时保留单侧继续 debug，但必须把双侧恢复列为优先修复项

---

## Step 8：做最小 tactile logging
### 要做
- 保存短序列 tactile frames：
  - no-contact
  - contact
  - 开合中间状态
- 保存 trial id / 时间戳

### 完成标准
- tactile frame 能稳定导出到磁盘
- 后续可直接给 Sparsh 或 debug 工具使用

---

## Step 9：Phase 2 Review
### Exit Criteria
必须全部通过：
- [ ] 已锁定唯一挂载几何 source-of-truth
- [ ] 单侧 GSmini 挂载稳定
- [ ] 单侧挂载后 gripper 开合稳定
- [ ] 能输出稳定 no-contact tactile image
- [ ] no-contact / contact frame 可区分
- [ ] 双侧挂载恢复成功，或已记录明确单侧 fallback
- [ ] tactile frames 可记录到磁盘

### 规则
- 任一项失败，就继续修 Phase 2
- **不允许**带着 Phase 2 的问题进入 Phase 3

---

## 本阶段需要保存的产物
- 左/右传感器 prim path
- 单侧/双侧挂载截图
- 开合测试结果
- no-contact / contact 样例图
- tactile logging 路径说明
- mount / collision / reset 问题与修复记录

---

## Phase 2 的一句话总结
**先在 canonical integrated UR5e + Robotiq + connector + GSmini embodiment 上把挂载校核清楚、把 tactile 图像跑通，再开始任何 Sparsh 和闭环控制工作。**


## 本次 Step 1/2/3 产物
- 结论报告：`tactile_grasp/artifacts/phase2_step1_3_report.md`
- 原始 JSON：`tactile_grasp/artifacts/phase2_step1_3_report.json`
- 主参考命名：left = `gelsight_connector_left` / `gelsight_mini_case_left` / `gelsight_mini_gelpad_left`
