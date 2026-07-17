# Isaac Sim / ProTac 回归测试

该目录只保存当前仍有价值的自动化回归和资产查看器，不再保存历史 Phase1/Phase2 bring-up demo 或失效的 SPARSH smoke。

## 内容

- `test_cube_entrypoint.py`：GUI/headless 参数和模块启动行为。
- `test_protac_module_boundary.py`：根目录与 `protac/` 功能边界。
- `test_gsmini_tacex_contract.py`：GSmini/TacEx 资产、相机、挂载和触觉契约。
- `test_phase3_schema_contract.py`：冻结的 `phase3_trial_v1` 兼容契约。
- `test_phase3_review.py`、`test_phase3_diagnostics.py`：artifact review/diagnostics。
- `test_ur5_robotiq_gsmini_new_urdf.py`：URDF、mesh 和运行时引用契约。
- `viewers/view_ur5_robotiq_GSmini_pybullet.py`：快速查看 canonical URDF。

## 命令

```bash
conda run -n tacex python -m pytest -q
conda run -n tacex python -m pytest -q \
  -k 'not connector_visual_and_collision_use_scaled_new_adaptor'
conda run -n tacex python -m isaacsim_test.viewers.view_ur5_robotiq_GSmini_pybullet
```

## adaptor_4 正式机械基线

当前 URDF 是机械装配权威来源，左右 connector 的 visual/collision 均固定使用：

- 网格：`adaptor_4.STL`
- SHA-256：`ab10eaef867622f9107b43e7602e698c35191cf66376c3b2a6e3e70c277728e4`
- STL 尺寸：`31 × 26 × 75 mm`
- URDF 缩放：`0.001 0.001 0.001`
- mesh origin：`-0.017104848723 -0.015687201598 -0.053443920870 m`

回归测试同时锁定文件路径、二进制哈希、几何边界、左右安装、USD 合法文件名和 gelpad 前向接触关系。以后若修改 adaptor 网格、原点、缩放或固定关节，必须作为新的机械基线变更并重新完成 Isaac 导入、碰撞和抓取验收。

## 已知边界

- 旧 adaptor_3 网格仍作为历史 CAD 文件保留，但当前 URDF 不再引用。
- GUI 双触觉窗口仍需人工确认；自动门禁覆盖参数、契约和 headless 初始化。
