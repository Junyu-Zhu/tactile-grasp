#!/usr/bin/env python3
"""Chinese force-only evidence summary from accepted aggregation, not model selection."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 with Path(p).open() as f:return list(csv.DictReader(f))
def fmt(x):return '不可用' if x is None or not np.isfinite(x) else f'{x:.4f}'
def num(x):return float(x) if x not in ('',None) else float('nan')
def main():
 p=argparse.ArgumentParser();p.add_argument('--aggregate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 ap=a.aggregate/'AUDIT.json';audit=json.loads(ap.read_text());assert audit['status']=='pass' and audit['formal_run_pairs']==24
 for f,h in audit['output_hashes'].items():assert sha(a.aggregate/f)==h
 rows=read(a.aggregate/'per_run_summary.csv');ci=read(a.aggregate/'paired_group_ci.csv');out=[]
 out+=['# 第十轮力监督适配诊断','', '本报告比较同结构、同物理输出接口的旧R5力预测和第十轮适配结果。12个新force运行、12组旧力配对；另对12组旧force任务运行检查回归。指标按native shear_x、shear_y、normal分别报告，不合并不兼容坐标。','', '力监督使用slip试次自身的6d_force与ref_force，目标为clip((force−reference)前三轴,−20,20) N。监督从s=5开始；这里的统一诊断从t=13开始，lag5差分的历史也完整。HTT滑移阶段标签部分依据力规则；测得的力GT与滑移标签并非独立证据，所以此处不是独立物理滑移验证。','', '新增force训练只使用外层train中的fit组，checkpoint由train内部selection组选择；外层validation/calibration未参与本轮新增force训练或选择。仍继承R5原checkpoint已经使用外层validation进行历史选择的限制。R5输出normalization保持不变，fit统计只用于审计。','', '下表每个数为该折三个种子“完整试次宏均值”的描述性平均。std是每试次时间变化标准差再平均，不是拼接帧后的全局标准差。置信区间先在同试次内平均三个种子的new−old差，再按完整泄漏组作200次配对重采样，四折分别计算。']
 axes=('shear_x','shear_y','normal');folds=tuple(f'htt_leave_p{i}' for i in range(1,5))
 for task,roles,title in [('slip_force',('fit','selection','validation','calibration'),'Slip同试次力诊断'),('old_force_regression',('legacy_force_train','validation','calibration'),'旧force任务回归')]:
  out+=['',f'## {title}','']
  for role in roles:
   out+=[f'### {role}','','|折/轴|MAE旧→新 N|ΔMAE 95%组CI|lag5变化MAE旧→新 N|预测std旧→新 / GTstd N|','|---|---:|---:|---:|---:|']
   for fold in folds:
    for axis in axes:
     sel=[r for r in rows if (r['task'],r['role'],r['fold'],r['axis'],r['population'],r['aggregation'])==(task,role,fold,axis,'all','complete_trial_macro')]
     vals={}
     for variant in ('old','new'):
      rr=[r for r in sel if r['variant']==variant];assert {int(r['seed']) for r in rr}=={20260914,20260915,20260916}
      vals[variant]={m:float(np.mean([num(r[m]) for r in rr])) for m in ('mae','delta5_mae','prediction_std','target_std')}
     cc=next(r for r in ci if (r['task'],r['role'],r['fold'],r['axis'],r['population'],r['metric'])==(task,role,fold,axis,'all','mae'))
     out.append(f"|{fold[-2:]}/{axis}|{fmt(vals['old']['mae'])} → {fmt(vals['new']['mae'])}|[{fmt(num(cc['ci_lower']))}, {fmt(num(cc['ci_upper']))}]|{fmt(vals['old']['delta5_mae'])} → {fmt(vals['new']['delta5_mae'])}|{fmt(vals['old']['prediction_std'])} → {fmt(vals['new']['prediction_std'])} / {fmt(vals['new']['target_std'])}|")
 out+=['','## 解释与限制','','- 负的new−old MAE代表误差降低；置信区间跨零时，该折该轴不能明确判断方向。不同折重叠、三个轴相关，不把改善区间数量当独立投票或全局显著性。','- fit改善说明监督可拟合；selection受checkpoint选择影响；外层validation/calibration用于评估迁移，但仍是历史开发数据。','- 幅值更大不是自动更准确，应同时检查目标std、误差、偏置及lag5变化误差。完整逐试次表包含这些字段。','- 旧force任务错误增加说明适配可能损失原任务性能；不能只展示slip域改善。','- force误差改善是否转化为slip收益，必须结合V/旧力/新力固定工作点和相同试次关联，不能由本报告单独推断。','- 各stage的static/incipient/gross只是描述性切片；不能把force-rule来源标签当额外独立物理证据。','- 同NPZ目标裁剪比例与预测超范围比例在完整表内。原force缓存已裁剪，无法从该缓存恢复未裁剪比例，因此保留不可用，不填零。','','## 产物','',f'- `{a.aggregate}/per_run_summary.csv`：每run每角色/阶段/轴完整试次宏均值与帧加权统计。',f'- `{a.aggregate}/per_seed_trial_differences.csv`：各seed逐试次new−old差。',f'- `{a.aggregate}/seedmean_trial_differences.csv`：三个seed同试次平均差。',f'- `{a.aggregate}/paired_group_ci.csv`：逐折配对组CI。',f'- `{a.aggregate}/AUDIT.json`：24组完整性、哈希、角色与统计边界。']
 rp=a.output/'SUMMARY_ZH.md';rp.write_text('\n'.join(out)+'\n');proof={'status':'pass','source_sha256':sha(__file__),'aggregate_audit_sha256':sha(ap),'outputs':{'SUMMARY_ZH.md':sha(rp)},'scope':'force-only report; slip usefulness requires separate controlled evaluation'};(a.output/'REPORT_AUDIT.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
if __name__=='__main__':main()
