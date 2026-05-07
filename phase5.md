# Phase 5：Cube simulation force/slip label validity hardening before limited Sparsh evaluation

## 目标

Phase 5 承接 Phase 4 的 cube-only Sparsh force/slip bridge。Phase 4 已经证明：

- 当前 cube 仿真 tactile frames 可以被整理为 Sparsh loader 可读的 `dataset_gelsight*` + `dataset_slip_forces.pkl`。
- force / slip dataloader smoke 可以跑到 `PASS_DATALOADER_ONLY`。
- 但没有 Sparsh checkpoint，因此没有 model-forward metrics。
- 更重要的是，当前 labels 仍然只能支撑 `limited-proxy`：
  - 当前 force 3-vector 曾以 `[left_force, right_force, max_force]` 的 shape-compatible proxy 形式存在。
  - 当前 slip label 曾以 object XY motion / relative-motion proxy 形式生成。

Phase 5 只回答一个问题：

> **能否把 cube 仿真数据中的 force/slip labels 从“shape-compatible proxy”升级为具有明确物理语义、坐标系、有效 mask 与可复现判据的 simulation-valid labels，从而允许后续 limited Sparsh force/slip evaluation？**

Phase 5 不以跑出漂亮指标为目标。它的核心是 **label validity hardening**。

---

## Source of Truth

### Phase 4 文档与产物

- Phase4 plan：`tactile_grasp/phase4.md`
- Phase4 completion audit：`tactile_grasp/phase4_completion.md`
- Phase4 bridge script：`tactile_grasp/phase4_sim_dataset.py`
- Phase4 smoke script：`tactile_grasp/phase4_sparsh_smoke.py`
- Phase4 generated data root：`tactile_grasp/sim_dataset/phase4_sparsh_cube/`
- Phase4 collected data roots：
  - `tactile_grasp/sim_dataset/phase4_collected/bridge_robustness/`
  - `tactile_grasp/sim_dataset/phase4_collected/force_oriented/`
  - `tactile_grasp/sim_dataset/phase4_collected/slip_oriented/`

### Current Phase4 observed status

From `phase4_completion.md`:

```text
source trials: 6
frame rows: 1372
trajectories: 6
positive proxy slip labels: 218
max force label: 1.792631 N
force smoke: PASS_DATALOADER_ONLY
slip smoke: PASS_DATALOADER_ONLY
metric validity: limited-proxy
```

Important limitation:

```text
No Sparsh model-forward metrics were run because no Sparsh encoder/task checkpoint was supplied.
Force labels include real simulation contact-sensor values where available, but remaining samples may be geometry/placeholder labels.
Slip labels are simulation relative-motion proxy labels, not real-world controlled slip ground truth.
```

### Sparsh force/slip contract

Sparsh loader expects:

```text
<dataset_root>/<dataset_name>/dataset_slip_forces.pkl
<dataset_root>/<dataset_name>/dataset_gelsight*
```

`VisionForceSlipDataset` emits samples with:

```python
sample["image"]
sample["force"]
sample["delta_force"]
sample["slip_label"]
```

Therefore Phase5 labels must satisfy more than shape compatibility. They must carry defensible semantics.

---

## RALPLAN-DR Summary

### Principles

1. **Semantic labels before metrics**：force/slip arrays 仅 shape-compatible 不够；必须有坐标系、单位、方向、valid mask 与生成规则。
2. **Cube-only continuity**：Phase5 只延续 Phase4 cube-only 范围。
3. **Simulation-first**：本阶段只使用 Isaac Sim / IsaacLab 仿真数据，不采集实物 GelSight Mini 或 real robot 数据。
4. **Gate split**：必须分开 `FORMAT_GATE`、`LABEL_VALIDITY_GATE`、`EVAL_GATE`，避免把 dataloader smoke 误当评估。
5. **Proxy honesty**：`[left, right, max_force]` 和 world-XY motion slip proxy 不得升级为 metric-valid label。

### Decision Drivers

1. Phase4 已经完成 data bridge smoke，但结果是 `limited-proxy`，不是可信 force/slip evaluation。
2. Sparsh force/slip loader 把 `forces` / `[T, 3]` 当作可归一化的三维力向量，并从中计算 `force` / `delta_force`。
3. Slip label 必须来自 cube 相对 gripper/contact frame 的切向滑动，而不是 `contact_onset`、`success_label`、world XY proxy 或 intentional release。

### Options

#### Option A — 直接把 Phase4 proxy dataset 当成 Phase5 evaluation 数据

Pros:
- 最快。
- 当前 dataloader smoke 已经跑通。

Cons:
- `[left, right, max_force]` 不是物理 XYZ / contact-frame normal-shear force。
- object XY motion proxy 会混入夹爪整体运动、lift、release/drop。
- 指标会评价 proxy 假设，而不是模型能力。

Verdict：拒绝，只能作为 Phase4 smoke 背景。

#### Option B — 只补 force labels，再做 force-only limited evaluation

Pros:
- 直接解决 force label blocker。
- 工作量小于同时处理 slip。

Cons:
- slip detection 仍停留在 proxy。
- Phase5 无法完整承接 force/slip 双任务目标。

Verdict：可作为降级路径，但不是主线。

#### Option C — Recommended：force vector semantic gate + slip contact-frame gate + v2 export

Pros:
- 同时硬化 force 和 slip label 语义。
- 保持 cube-only、simulation-only、Sparsh-compatible。
- 通过 gate 后才允许 limited evaluation，避免误导。

Cons:
- 需要先修 instrumentation / label rule，而不是立即追求指标。

Recommendation：采用 Option C。

---

# Phase5 Objective

Phase5 的目标是：

> 在 cube-only simulation 范围内，验证并扩展 force/slip labels 的物理语义与有效性，生成 Sparsh-compatible v2 dataset，并只有在 `FORMAT_GATE`、`LABEL_VALIDITY_GATE`、`EVAL_GATE` 通过后，才允许 limited Sparsh force/slip evaluation。

---

## Scope

### In scope

- cube-only simulation trials。
- Phase4 generated bridge/report audit。
- Force vector semantic validation。
- Contact/gripper-frame slip label validation。
- New or revised cube force-oriented trials。
- New or revised cube slip-oriented trials。
- Sparsh-compatible v2 export。
- Label validity reports and masks。
- Limited evaluation gate, not adaptation。

### Non-goals

Phase5 明确不做：

- forcefield demo / forcefield decoder。
- `chips_can` / `cracker_box`。
- real GelSight Mini data collection。
- real robot data collection。
- Sparsh adaptation / fine-tuning。
- closed-loop control。
- GraspNet。
- RL。
- 将 Phase4 proxy label 直接升级为真实 evaluation label。
- 在 checkpoint / label gate 未通过时报告可信 Sparsh 指标。

---

# Gate model

Phase5 必须把三类 gate 分开记录。

## 1. FORMAT_GATE

目标：确认 Sparsh loader 能读数据。

PASS 条件：

- `dataset_gelsight*` 存在并可加载。
- `dataset_slip_forces.pkl` 存在并可加载。
- dataloader sample 包含：
  - `image`
  - `force`
  - `delta_force`
  - `slip_label`
- tensor shape 符合 Sparsh loader / model 输入要求。

注意：

> FORMAT_GATE 通过不代表 labels 有效。

## 2. LABEL_VALIDITY_GATE

目标：确认 force/slip labels 有物理语义与有效 mask。

必须分别给出：

```text
FORCE_LABEL_VALIDITY = invalid / proxy / sim-valid
SLIP_LABEL_VALIDITY = invalid / proxy / sim-valid
```

## 3. EVAL_GATE

目标：确认是否允许 limited Sparsh evaluation。

PASS 条件：

- FORMAT_GATE = PASS。
- FORCE_LABEL_VALIDITY 或 SLIP_LABEL_VALIDITY 至少对应任务为 `sim-valid`。
- checkpoint / model-forward path 可用。
- metric split / valid masks 定义清楚。
- metric set 不包含 placeholder/proxy samples。

注意：

> EVAL_GATE 未通过时，只允许 dataloader/model-forward smoke，不允许报告可信 metric。

---

# Step-by-step Task List

## Step 1 — Phase4 audit and proxy boundary freeze

### Goal

明确 Phase4 已经完成什么，以及哪些内容不能直接作为 Phase5 metric ground truth。

### Tasks

- 读取：
  - `tactile_grasp/phase4_completion.md`
  - `tactile_grasp/sim_dataset/phase4_sparsh_cube/bridge_report.json`
  - `tactile_grasp/sim_dataset/phase4_sparsh_cube/smoke_report.json`
  - `tactile_grasp/sim_dataset/phase4_sparsh_cube/cube_phase3_bridge/manifest.csv`
- 记录当前：
  - source trial count。
  - positive proxy slip count。
  - max force label。
  - force/slip smoke status。
  - metric validity。
- 冻结以下边界：
  - Phase4 `relative_motion_proxy` 不自动升级为 `sim-valid`。
  - `[left_force, right_force, max_force]` 不自动升级为 metric-valid force vector。

### Acceptance criteria

- `reports/phase4_audit.md` 写清 Phase4 proxy / smoke / blocker 状态。
- 所有 Phase4-derived labels 都有 `label_source` 与 `label_valid_for_metrics`。

---

## Step 2 — Force vector semantic gate

### Goal

验证或重定义 force label，使其不只是 `[T, 3]` shape，而是有物理意义。

### Required force semantics

`FORCE_LABEL_VALIDITY=sim-valid` 只有在以下内容全部定义并验证后才成立：

- coordinate frame：world / gripper / sensor / contact frame。
- axis order：`[fx, fy, fz]` 或 `[shear_x, shear_y, normal]`。
- sign convention：正方向定义。
- unit：Newton。
- normal/shear meaning：哪个轴是 normal，哪些轴是 shear。
- per-sample alignment：force sample 与 tactile frame 的 timestamp / robot_state_index 对齐。
- valid mask：哪些 samples 可以用于 metric。

### Explicitly invalid for metric

以下只能标为 `proxy` 或 `shape-compatible smoke`：

```text
[left_force, right_force, max_force]
```

除非它被重新定义为某个明确坐标系下的物理向量，并通过验证。

### Tasks

- 检查 Isaac / IsaacLab contact sensor 是否能提供 contact normal / force vector，而不只是 side scalar。
- 若只能得到 scalar normal force：定义 `[0, 0, normal_force_n]`，并明确 frame/axis/sign。
- 若能得到 full vector：记录 vector frame 与 transform。
- 对 no-contact / low / medium / high contact 做 sanity check。
- 生成 force label summary。

### Acceptance criteria

- 非接触样本 near zero。
- 接触样本非零。
- force magnitude 随 close pressure 或 contact condition 有合理变化。
- force vector 语义被写入 metadata。
- 无效 / proxy samples 被 mask 出 metric set。

---

## Step 3 — Force-oriented cube trials v2

### Goal

收集或重构能验证 force magnitude / axis semantics 的 cube trials。

### Trial families

```text
no_contact
light_contact
medium_contact
firm_contact
over_contact
lift_and_hold
contact_release
```

### Tasks

- 使用 cube-only scene。
- 控制 gripper close target / contact duration / lift stage。
- 每条 trial 保存：
  - tactile frames。
  - robot_state。
  - force vector。
  - force frame metadata。
  - force validity mask。
  - side/contact metadata。

### Acceptance criteria

- 至少存在两个 force regimes；推荐 low / medium / high。
- force values 与 protocol level 有可解释关系。
- label source 不再是纯 placeholder。
- 所有 metric-bearing samples 有 `valid_force=true`。

---

## Step 4 — Slip contact-frame semantic gate

### Goal

定义 slip label，使其来自 cube 相对 gripper/contact frame 的切向滑动，而不是 world XY proxy。

### Required slip semantics

`SLIP_LABEL_VALIDITY=sim-valid` 需要满足：

- 每帧保存或可重建：
  - `cube_pose_in_gripper_frame`，或
  - `cube_pose_in_contact_frame`。
- slip 判据基于 tangential relative displacement / velocity。
- normal approach / contact onset 不算 slip。
- intentional release 不算 slip。
- release/drop frames 必须 mask out，除非 protocol 明确标记为 slip event。
- positive / negative samples 都存在。
- threshold / horizon 被写入 metadata。

### Disallowed as sole slip labels

- `success_label`
- `contact_onset`
- trial-level failure reason
- tactile image appearance alone
- world XY displacement alone
- Phase4 `relative_motion_proxy` alone

### Proposed rule

```text
slip_label[t] = 1
if contact_established(t)
and stage in {hold, lift, perturbation}
and tangential_relative_displacement_or_velocity(cube, contact/gripper frame)
    exceeds threshold within horizon
and frame is not intentional release
```

### Acceptance criteria

- Rule produces both slip and no-slip windows。
- Label rule can be recomputed from saved states。
- Manual spot-check examples match expected cube motion。
- Ambiguous windows have `valid_slip=false`。

---

## Step 5 — Slip-oriented cube trials v2

### Goal

生成有受控 slip / no-slip 对照的 cube trials。

### Trial families

```text
stable_hold_no_slip
weak_grasp_downward_slip
lift_induced_slip
lateral_perturbation_slip
friction_reduced_slip
intentional_release_ignored
borderline_invalid
```

### Tasks

- 保存 cube pose、gripper/contact frame pose、relative pose / velocity。
- 标记 stages：approach / contact / hold / lift / perturbation / slip / release。
- 生成 frame-level slip labels 与 validity masks。
- release frame 默认 `valid_slip=false`，除非 protocol 明确将其作为 slip event。

### Acceptance criteria

- controlled positive slip events 存在。
- stable no-slip windows 存在。
- positive/negative class balance 足够做 smoke-level evaluation。
- slip onset 可追溯到 relative motion。

---

## Step 6 — Sparsh-compatible dataset export v2

### Goal

导出 v2 数据，使它既符合 Sparsh loader，又保留 label validity metadata。

### Heavy data policy

- 大型 `.npy` / `.pkl` 数据默认放在 ignored：

```text
tactile_grasp/sim_dataset/phase5_cube_force_slip/
```

- 可版本化/可审查的报告与 manifest 放在：

```text
tactile_grasp/artifacts/phase5_cube_force_slip/
```

### Export layout

```text
tactile_grasp/sim_dataset/phase5_cube_force_slip/sparsh_export_v2/
  cube_force_slip_v2/
    dataset_gelsight_cube_v2.pkl
    dataset_slip_forces.pkl
    sample_index.csv
    export_manifest.json
```

Reports:

```text
tactile_grasp/artifacts/phase5_cube_force_slip/
  README.md
  reports/
    phase4_audit.md
    force_vector_semantics.md
    force_trial_summary.md
    slip_contact_frame_rule.md
    slip_trial_summary.md
    label_validity_gate.md
    format_gate.md
    eval_gate.md
    phase5_review.md
```

### `dataset_slip_forces.pkl` recommended content

```python
{
    "in_contact": np.ndarray,
    "trajectories": {
        trajectory_id: {
            "indexes": np.ndarray,
            "forces": np.ndarray,       # [T, 3], metric-valid only if valid_force[t]
            "slip_label": np.ndarray,   # [T], metric-valid only if valid_slip[t]
            "valid_force": np.ndarray,  # [T], bool
            "valid_slip": np.ndarray,   # [T], bool
            "metadata": {
                "trial_id": str,
                "object_id": "cube",
                "protocol_variant": str,
                "force_frame": str,
                "force_axis_order": list[str],
                "force_sign_convention": str,
                "slip_frame": str,
                "slip_threshold": float,
                "slip_horizon": int,
                "source": "phase5_cube_force_slip"
            }
        }
    }
}
```

### Acceptance criteria

- FORMAT_GATE can pass with this export。
- Metadata can explain every metric-bearing label。
- Proxy / placeholder samples are present only if masked out from metric split。

---

## Step 7 — FORMAT_GATE

### Goal

确认 loader contract 仍然打通。

### Tasks

- Run Sparsh dataloader against v2 export。
- Confirm emitted keys and shapes：
  - `image`
  - `force`
  - `delta_force`
  - `slip_label`
- Record dataset length and valid sample counts。

### Acceptance criteria

- `FORMAT_GATE=PASS` if dataloader works。
- Otherwise exact blocker is documented。

---

## Step 8 — LABEL_VALIDITY_GATE

### Goal

决定 labels 是否允许进入 metric set。

### Force gate outcomes

```text
FORCE_LABEL_VALIDITY=invalid
FORCE_LABEL_VALIDITY=proxy
FORCE_LABEL_VALIDITY=sim-valid
```

`sim-valid` requires force-vector semantic gate to pass。

### Slip gate outcomes

```text
SLIP_LABEL_VALIDITY=invalid
SLIP_LABEL_VALIDITY=proxy
SLIP_LABEL_VALIDITY=sim-valid
```

`sim-valid` requires contact-frame slip semantic gate to pass。

### Combined metric validity

```text
METRIC_VALIDITY=invalid
METRIC_VALIDITY=limited-proxy
METRIC_VALIDITY=limited-sim
```

Only `limited-sim` permits limited metric reporting。

---

## Step 9 — EVAL_GATE and limited Sparsh evaluation

### Goal

只有在 FORMAT + LABEL gates 通过后，才允许 limited evaluation。

### Preconditions

- `FORMAT_GATE=PASS`
- target task label validity is `sim-valid`
- checkpoint / model forward path exists
- valid masks and split are defined
- metric set excludes proxy / placeholder samples

### Allowed outputs

- Force limited evaluation：
  - RMSE / correlation only on `valid_force=true` samples。
- Slip limited evaluation：
  - balanced accuracy / F1 only on `valid_slip=true` samples。
- All reports must be scoped as cube-simulation limited evaluation。

### Disallowed

- benchmark-quality claim。
- real-world generalization claim。
- adaptation / fine-tuning。
- closed-loop use。
- non-cube claim。

---

## Step 10 — Phase5 review

### Review outputs

- Phase4 audit verdict。
- Force vector semantic verdict。
- Slip contact-frame semantic verdict。
- Format gate verdict。
- Label validity verdict。
- Eval gate verdict。
- Known limitations。
- Phase6 recommendation。

### Acceptance criteria

- Review distinguishes smoke, proxy, sim-valid, and evaluation。
- Review states whether Phase6 may run broader cube evaluation or must return to instrumentation。
- Review does not recommend real data collection/adaptation unless deferred to a separate future phase。

---

# Exit Criteria

Phase5 is complete when：

- [ ] Phase4 proxy boundary audit is written。
- [ ] Force vector semantics are defined and validated, or rejected with exact blocker。
- [ ] `[left_force, right_force, max_force]` is not used as metric-valid force。
- [ ] Force-oriented cube trials v2 exist or an instrumentation blocker is documented。
- [ ] Slip contact-frame rule is defined and validated, or rejected with exact blocker。
- [ ] World-XY / Phase4 relative-motion proxy is not used as `sim-valid` slip label。
- [ ] Slip-oriented cube trials v2 include controlled no-slip/slip examples or a protocol blocker is documented。
- [ ] Sparsh-compatible v2 export exists or exact blocker is documented。
- [ ] FORMAT_GATE / LABEL_VALIDITY_GATE / EVAL_GATE are separately reported。
- [ ] Limited Sparsh metrics, if any, use only metric-valid samples。
- [ ] No forcefield / real data / adaptation / fine-tuning / closed-loop / GraspNet / RL / non-cube expansion is introduced。

---

# Execution handoff guidance

## Recommended sequential lane (`$ralph`)

Use `$ralph` if Phase5 is executed by one owner in strict gate order：

1. `executor`：audit Phase4 and freeze proxy boundaries。
2. `executor`：implement / repair force vector instrumentation and export metadata。
3. `executor`：implement / repair contact-frame slip label rule and masks。
4. `test-engineer` / `verifier`：run format, label-validity, and eval gates。
5. `writer`：produce Phase5 reports and Phase6 recommendation。

Suggested launch：

```text
$ralph execute tactile_grasp/phase5.md step by step; stop at any failed gate and document exact blocker
```

## Recommended parallel lane (`$team`)

Use `$team` only if Phase5 work is split across independent lanes：

| Lane | Agent type | Ownership | Reasoning |
|---|---|---|---|
| Force semantics | `executor` | contact sensor vector extraction, force metadata, force masks | high |
| Slip semantics | `executor` | gripper/contact-frame relative pose, slip rule, slip masks | high |
| Sparsh export/smoke | `executor` or `test-engineer` | v2 export layout, dataloader smoke, format report | medium |
| Verification | `verifier` | gate evidence, metric-valid sample checks, blocker audit | high |
| Documentation | `writer` | reports under `artifacts/phase5_cube_force_slip/` | medium |

Team verification path：

```text
1. Force lane and slip lane produce independent label-validity reports.
2. Export lane only consumes labels after validity metadata exists.
3. Verifier checks FORMAT_GATE before LABEL_VALIDITY_GATE, then EVAL_GATE.
4. Writer records whether Phase6 proceeds to broader cube-only evaluation or returns to instrumentation.
```

Available useful agent types：`planner`, `architect`, `executor`, `debugger`, `test-engineer`, `verifier`, `writer`。

---

# ADR：Phase5 as semantic label-validity hardening

## Decision

Phase5 will harden cube simulation force/slip labels by validating force vector semantics and contact-frame slip labels before allowing limited Sparsh evaluation.

## Drivers

- Phase4 already completed format bridge and dataloader smoke, but metric validity remains `limited-proxy`。
- Sparsh force/slip tasks require semantically meaningful `force`, `delta_force`, and `slip_label`。
- Proxy labels can easily produce misleading metrics if not gated。

## Alternatives considered

1. **Use Phase4 proxy dataset directly** — rejected because it evaluates proxy assumptions。
2. **Only fix force labels** — accepted only as fallback; incomplete for force/slip scope。
3. **Collect real data now** — rejected by current simulation-only scope。
4. **Begin adaptation** — rejected because label validity is not established。

## Consequences

- Phase5 may spend most effort on instrumentation and reports rather than headline metrics。
- Later limited evaluation will be more defensible。
- If force/slip semantics cannot be made valid in simulation, Phase6 should repair instrumentation rather than train/adapt models。

## Follow-ups

- If FORCE_LABEL_VALIDITY reaches `sim-valid` but slip remains proxy, Phase6 may do force-only limited evaluation while redesigning slip protocol。
- If SLIP_LABEL_VALIDITY reaches `sim-valid` but force remains proxy, Phase6 may do slip-only limited evaluation but must avoid delta-force claims。
- If both reach `sim-valid`, Phase6 may run broader cube-only limited evaluation before any object expansion or adaptation。

---

# 一句话总结

**Phase5 不再重复 Phase4 的 format bridge；它要把 cube 仿真 force/slip labels 从 proxy 提升到有坐标系、接触帧、valid mask 和语义 gate 的 simulation-valid labels，只有通过 FORMAT / LABEL / EVAL 三层 gate 后才允许 limited Sparsh evaluation。**
