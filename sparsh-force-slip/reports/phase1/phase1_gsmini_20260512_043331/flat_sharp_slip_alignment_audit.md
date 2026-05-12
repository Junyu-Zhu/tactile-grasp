# Flat/Sharp Slip Alignment Audit

- generated_at: `2026-05-13T00:52:37`
- run_id: `phase1_gsmini_20260512_043331`
- raw_root: `/vla1/zjy/tactile_datasets/Gelsight-mini/gelsight-force-estimation`
- derived_root: `/vla1/zjy/sparsh_runs/force_slip_phase1/phase1_gsmini_20260512_043331/derived_gsmini`

## Conclusion

- structural_pass: `True`
- derived_split_pass: `True`
- dataloader_smoke_pass: `True`
- official_delta_shear_positive_greater_than_negative: `True`
- recommendation: **flat/sharp 可以进入下一步 all-source slip 诊断训练；正式 baseline 仍建议等 all-source 训练指标审计后再升级。**

flat/sharp 的 `len(indexes)==len(labels)+5` 与官方 `indexes[5:]` 采样长度规则结构一致，官方 dataloader 可读取派生 split，所有 source trajectory 都被 train/val/test 覆盖，且 slip 正样本的 official delta shear 平均值高于负样本。

## Official sample-index semantics checked

Sparsh `VisionForceSlipDataset` uses `indexes[5:]` only to set dataset length; the actual sample id remains `0..len(indexes[5:])-1`. For a sample `s`, labels use `slip_label[s]` and images use `indexes[s]` plus temporal context `indexes[max(s-5,0)]` for the default two-frame setup.

## Source structure summary

| source | trajectories | relation | official samples | slip ratio | transitions | shear pos/neg | issues |
|---|---:|---|---:|---:|---:|---:|---:|
| `flat/batch_1` | 151 | len(indexes)==len(labels)+5:151 | 6732 | 0.4005 | 147 | 15.322 | 0 |
| `flat/batch_2` | 147 | len(indexes)==len(labels)+5:147 | 6313 | 0.4215 | 142 | 17.928 | 0 |
| `sharp/batch_1` | 153 | len(indexes)==len(labels)+5:153 | 11183 | 0.2585 | 148 | 40.665 | 0 |
| `sharp/batch_2` | 151 | len(indexes)==len(labels)+5:151 | 9725 | 0.2847 | 149 | 35.758 | 0 |

## Derived split consistency

| source | train ratio | val ratio | test ratio | missing raw traj | passed |
|---|---:|---:|---:|---:|---|
| `flat/batch_1` | 0.4033 | 0.3471 | 0.4529 | 0 | `True` |
| `flat/batch_2` | 0.4210 | 0.4488 | 0.3990 | 0 | `True` |
| `sharp/batch_1` | 0.2451 | 0.3057 | 0.2815 | 0 | `True` |
| `sharp/batch_2` | 0.2843 | 0.2974 | 0.2745 | 0 | `True` |

## Aggregate all-source slip balance

| split | sphere-only ratio | all-source ratio | all-source positives/total |
|---|---:|---:|---:|
| `train` | 0.1973 | 0.2401 | 16490/68685 |
| `val` | 0.1999 | 0.2468 | 3682/14920 |
| `test` | 0.2012 | 0.2460 | 3559/14468 |

## Visual examples

- `flat/batch_1` trajectory `1` sample `3`: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase1/phase1_gsmini_20260512_043331/flat_sharp_slip_alignment_examples/flat_batch_1_slip_alignment_event.png`
- `flat/batch_2` trajectory `1` sample `15`: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase1/phase1_gsmini_20260512_043331/flat_sharp_slip_alignment_examples/flat_batch_2_slip_alignment_event.png`
- `sharp/batch_1` trajectory `1` sample `11`: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase1/phase1_gsmini_20260512_043331/flat_sharp_slip_alignment_examples/sharp_batch_1_slip_alignment_event.png`
- `sharp/batch_2` trajectory `1` sample `21`: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase1/phase1_gsmini_20260512_043331/flat_sharp_slip_alignment_examples/sharp_batch_2_slip_alignment_event.png`

## Notes

- This audit does not modify raw data.
- `org_dataset_gelsight_*` is not used by the official downstream loader and is not part of this audit's training compatibility check.
- Passing this audit supports adding flat/sharp to a diagnostic all-source slip run. A full all-source training run is still recommended before upgrading it to the formal baseline.
