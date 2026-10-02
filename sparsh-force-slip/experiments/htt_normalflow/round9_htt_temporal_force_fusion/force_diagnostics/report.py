import csv,json,hashlib,sys
from pathlib import Path
import numpy as np
p=Path(sys.argv[1]);audit=json.loads((p/'SUPPORT_AUDIT.json').read_text());assert audit['status']=='pass'
for name,h in audit['output_hashes'].items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==h
rows=list(csv.DictReader((p/'per_trial_axis.csv').open()));summary=[]
for fold in sorted({r['fold'] for r in rows}):
 for seed in [20260914,20260915,20260916]:
  for role in ['train','validation','calibration']:
   for axis in ['shear_x','shear_y','normal']:
    rr=[r for r in rows if r['fold']==fold and int(r['seed'])==seed and r['role']==role and r['axis']==axis and r['population']=='primary']
    summary.append(dict(fold=fold,seed=seed,role=role,axis=axis,trials=len(rr),**{k:float(np.mean([float(r[k]) for r in rr])) for k in ['mae','rmse','bias','target_std','prediction_std','delta5_mae','target_clip_fraction']}))
with (p/'macro_trial_summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
lines=['# HTT同试次力预测诊断','','本轮核验101个已在四折开发角色使用的slip NPZ，均携带自身6d_force/ref_force；三轴与图像帧数一致。每个折的分析只使用该折train/validation/calibration，不读取该折test结果。相同物理试次可在其他重叠开发折出现，不能据此增加独立样本数。','','## 对历史表述的澄清','','第五轮data/PRECHECK.md已经记录全部101个slip归档有6d_force，只是force头监督限定于专门force任务，以免在力规则生成的slip试次上混入监督。此前“合同未缓存GT，因此slip试次没有力GT”的表述过宽。原历史文件不修改，本轮使用同一原始归档诊断，不跨force/slip同名文件配对。','','## Validation逐试次宏平均','','下表先每试次计算，再对试次均值、最后汇总三个种子。MAE/bias/delta5 MAE单位遵循HTT数据约定N，不与原域Sparsh三轴直接合并。','','|折|轴|MAE|偏置|lag5差分MAE|目标std|预测std|','|---|---|---|---|---|---|']
for fold in sorted({r['fold'] for r in summary}):
 for axis in ['shear_x','shear_y','normal']:
  rr=[r for r in summary if r['fold']==fold and r['axis']==axis and r['role']=='validation'];v=[np.mean([r[k] for r in rr]) for k in ['mae','bias','delta5_mae','target_std','prediction_std']]
  lines.append('|'+fold+'|'+axis+'|'+'|'.join(f'{x:.4f}' for x in v)+'|')
lines+=['','## 主要观察','','验证角色中，预测剪切力的时间标准差明显小于同试次参考相对目标：shear_x约0.55–0.66 N，对应目标2.34–3.01 N；shear_y也存在幅度压缩。normal的动态幅度更接近，但仍有逐折偏置。该现象支持检查从专门force任务到slip试次的力表征适配差异，尚不能单独证明当前slip误报的成因。','','## 解释限制','','- 目标严格为clip((6d_force-ref_force)[:,:3],-20,20)，来自最终contract；原始绝对力不直接用于误差。','- 只读NPZ小力数组、参考力和图像NPY头，不解码完整图片。沿用历史完整归档哈希，新增力数组哈希；没有重复全量原始文件哈希审计。','- 同归档帧序与长度支持索引对齐，不额外证明硬件同步精度、外参或仪器校准。','- slip bracket标签有力规则成分，预测力误差和滑移失败关联不能作为独立物理因果验证。','- 本轮未用这些目标重训force或slip，也未改变任何checkpoint/阈值。','- 全种子/角色/试次及clip诊断见per_trial_axis.csv，帧误差供后续精确episode/t关联，不能按文件basename关联。']
(p/'SUMMARY_ZH.md').write_text('\n'.join(lines)+'\n');(p/'REPORT_AUDIT.json').write_text(json.dumps(dict(status='pass',support_audit_sha256=hashlib.sha256((p/'SUPPORT_AUDIT.json').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),output_hashes={n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in ['SUMMARY_ZH.md','macro_trial_summary.csv']}),indent=2)+'\n')
