# Phase 1 动作展示说明

## 为什么直接运行 `ur5_sim.py` 看不到明显动作
`ur5_sim.py` 的默认行为是 **稳定 spawn / 稳定 reset**，用于作为 Phase 1 的主入口。
因此直接运行：

```bash
./isaaclab.sh -p tactile_grasp/ur5_sim.py
```

默认只会让机器人保持在 reset 姿态，不会主动连续演示：
- Step 5 arm joint actuation
- Step 6 Robotiq 开合
- Step 8 fixed pre-grasp reach

这些功能**已经实现并通过测试**，但主要挂在：
- `ur5_sim.py --phase1-checks`
- `ur5_sim_test.py`（显式演示版）

## 实现位置
- `tactile_grasp/ur5_phase1_control.py`
- `tactile_grasp/ur5_phase1_reset.py`
- `tactile_grasp/ur5_phase1_checks.py`

## 推荐的可视化演示命令
```bash
./isaaclab.sh -p tactile_grasp/ur5_sim_test.py
```

或 headless smoke：
```bash
PYTHONPATH="/home/zjy/Documents/grasp/TacEx/IsaacLab/source/isaaclab:/home/zjy/Documents/grasp/TacEx/IsaacLab/source/isaaclab_assets:/home/zjy/Documents/grasp/TacEx/IsaacLab/source/isaaclab_tasks:/home/zjy/Documents/grasp/TacEx/IsaacLab/source/isaaclab_mimic:/home/zjy/Documents/grasp/TacEx/IsaacLab/source/isaaclab_rl" conda run -s -n tacex python -u /home/zjy/Documents/grasp/tactile_grasp/ur5_sim_test.py --headless --gripper-cycles 1 --pregrasp-holds 1
```

## 本次确认结论
- Step 5：已实现，已测试通过
- Step 6：已实现，已测试通过
- Step 8：已实现，已测试通过
- 之所以你直接运行 `ur5_sim.py` 没看到，是因为它默认不播放动作演示，只做稳定场景入口
