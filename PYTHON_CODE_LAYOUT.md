# ProTac-RL Python 功能边界

仓库根目录只保留稳定用户入口 `ur5_cube_tactile_grasp.py`。入口通过当前 Conda `tacex` 的 Python 解释器执行 `protac.baseline.cube_collect`，其余代码按职责组织在 `protac/`，不再用历史实施阶段命名生产文件。

## 生产与未来复用单元

```text
protac/
├── baseline/
│   ├── cube_collect.py        # 当前 cube 抓取/触觉数据基线入口
│   └── trial_runner.py        # reset→接近→夹持→保持→微抬升→释放编排
├── robot/
│   ├── asset.py               # URDF/USD、机器人配置、桌面和安装底座
│   ├── control.py             # UR5/Robotiq 关节与目标生成
│   └── reset.py               # 确定性机器人复位
├── tactile/
│   ├── gsmini_contract.py     # GSmini/TacEx 资产、标定、坐标与 prim 契约
│   ├── mount.py               # detached TacEx shell、相机挂载与同步
│   └── sensor.py              # TacEx cfg、实时更新和双窗口显示
├── grasp/
│   ├── contact_geometry.py    # 双侧接触、力和软表面几何
│   └── nominal_controller.py  # IK、闭合、保持、微抬升和释放原语
├── scene/
│   └── object_profiles.py     # 对象/协议 profile 与场景装配
├── compat/
│   ├── phase3_v1_schema.py    # 冻结 phase3_trial_v1 契约
│   └── phase3_v1_logger.py    # 冻结 Phase3 v1 对齐日志
├── eval/
│   ├── trial_validator.py     # trial artifact 质量门禁
│   └── trial_diagnostics.py   # trial 诊断与基线对比 CLI
└── data/
    ├── sparsh_bridge.py       # 旧 Phase3→SPARSH 只读桥接参考
    └── force_slip_export.py   # force/slip 标签、valid mask 与导出
```

`compat/` 保留旧 schema/日志是为了 G1 只读 adapter，不表示后续 RL 或预测器继续写入旧 runner。G2 之后仍应新建 ProTac collector/runtime。

## 测试与工具

- `isaacsim_test/test_*.py`：当前行为、契约、结构边界回归。
- `isaacsim_test/viewers/view_ur5_robotiq_GSmini_pybullet.py`：URDF 快速资产查看。
- 历史 Phase1/Phase2 demo 和旧 SPARSH smoke/effect 已删除；它们不再作为生产或未来门禁。
- `sparsh-force-slip/scripts/` 仍是独立短期稳定性预测研究工程，不与当前 Isaac runtime 混合。

## 执行方式

```bash
conda run -n tacex python ur5_cube_tactile_grasp.py --headless
conda run -n tacex python -m pytest -q
conda run -n tacex python -m protac.eval.trial_diagnostics --help
conda run -n tacex python -m protac.eval.trial_validator --help
conda run -n tacex python -m protac.data.sparsh_bridge --help
conda run -n tacex python -m protac.data.force_slip_export --help
```
