#!/usr/bin/env python3
"""Build the compact R16 result tables and Chinese evidence summary."""
import argparse,csv,json,statistics as st
from collections import defaultdict
from pathlib import Path

SEEDS=(20260914,20260915,20260916)
def read(path):
 with Path(path).open(newline='') as f:return list(csv.DictReader(f))
def avg(rows,key):return st.fmean(float(r[key]) for r in rows)
def pct(x):return f'{100*x:.2f}%'
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 force=[]
 for route in ('H','T_H'):
  for fold in range(1,5):
   for seed in SEEDS:
    force += [{**r,'route':route,'fold_id':fold,'seed_id':seed} for r in read(a.root/f'force/{route}_p{fold}_s{seed}/metrics.csv')]
 force_table=[]
 for route in ('H','T_H'):
  for axis in ('fx','fy','fz'):
   x=[r for r in force if r['route']==route and r['role']=='validation' and r['stage']=='all' and r['axis']==axis]
   force_table.append({'route':route,'axis':axis,'mae':avg(x,'mae'),'rmse':avg(x,'rmse'),'bias':avg(x,'bias'),'change_mae':avg(x,'change_mae')})
 slip=read(a.root/'slip/metrics.csv');slip_table=[]
 specs=(('fixed0.5_raw',lambda r:r['family']=='historical' and r.get('point')=='fixed0.5' and r.get('rule')=='raw'),('FPR0.05_raw',lambda r:r['family']=='historical' and r.get('point')=='FPR0.05' and r.get('rule')=='raw'),('macro_a05_k1',lambda r:r['family']=='trial_macro_static_FPR' and r.get('alpha')=='0.05' and r['k']=='1'),('any_a05_k1',lambda r:r['family']=='trial_any_static_alarm_rate' and r.get('alpha')=='0.05' and r['k']=='1'))
 for group in ('V','H','T_H'):
  for name,test in specs:
   x=[r for r in slip if r['group']==group and r['role']=='validation' and test(r)]
   slip_table.append({'group':group,'point':name,'frame_static_FPR':avg(x,'frame_static_FPR'),'trial_any_static_alarm_rate':avg(x,'trial_any_static_alarm_rate'),'gross_recall':avg(x,'gross_recall'),'balanced_accuracy':avg(x,'balanced_accuracy'),'event_recall':avg(x,'event_recall')})
 future=[]
 for route in ('H','T_H'):
  for fold in range(1,5):
   for seed in SEEDS:future += [{**r,'route':route} for r in read(a.root/f'future/{route}_p{fold}_s{seed}/metrics.csv')]
 future_table=[]
 for route in ('H','T_H'):
  for method in ('neural','predicted_current_persistence','fit_ridge_linear'):
   for horizon in ('1','5','10'):
    x=[r for r in future if r['route']==route and r['role']=='validation' and r['method']==method and r['horizon']==horizon and r['stratum']=='all']
    future_table.append({'route':route,'method':method,'horizon':int(horizon),'future_mae_axis_macro':avg(x,'future_mae'),'change_mae_axis_macro':avg(x,'change_mae')})
 paired=read(a.root/'bootstrap/PAIRED_CI.csv');key={'schema':'round16_key_results_v1','status':'complete','force':force_table,'slip':slip_table,'future':future_table,'paired_ci_rows':len(paired),'test_consumed':False}
 (a.output/'KEY_RESULTS.json').write_text(json.dumps(key,indent=2)+'\n')
 lines=['# R16：ToucHD Mini 力私有迁移到 HTT','','本轮按预注册完成75次神经训练；ToucHD与HTT保持域内坐标，未把两域轴误差直接合并。发布数据共81,493行，按compact行i与i−3形成80,783个有效双帧pair。','','## HTT力验证集（12个fold/seed运行的描述性均值）','','|路线|轴|MAE N|RMSE N|变化MAE N|偏差 N|','|---|---|---:|---:|---:|---:|']
 for r in force_table:lines.append(f"|{r['route']}|{r['axis']}|{r['mae']:.4f}|{r['rmse']:.4f}|{r['change_mae']:.4f}|{r['bias']:.4f}|")
 lines += ['','## 当前滑移验证集','','|组|工作点|帧static FPR|试次任意static告警|gross召回|BA|事件召回|','|---|---|---:|---:|---:|---:|---:|']
 for r in slip_table:lines.append(f"|{r['group']}|{r['point']}|{pct(r['frame_static_FPR'])}|{pct(r['trial_any_static_alarm_rate'])}|{pct(r['gross_recall'])}|{pct(r['balanced_accuracy'])}|{pct(r['event_recall'])}|")
 lines += ['','## 未来力验证集','','|路线|方法|horizon|未来MAE N|变化MAE N|','|---|---|---:|---:|---:|']
 for r in future_table:lines.append(f"|{r['route']}|{r['method']}|{r['horizon']}|{r['future_mae_axis_macro']:.4f}|{r['change_mae_axis_macro']:.4f}|")
 lines += ['','所有主比较保留逐fold/seed结果，并使用每fold 2,000次完整validation泄漏组共享抽样；重叠fold、窗口和seed未被当作独立物理样本。固定失败案例、完整R13规则族、未来误差交叉项和范围/复制诊断见评价目录。ToucHD目标仅是发布compact域内监督，官方按N使用不构成独立标定证明；传感器到腕/世界坐标、物理零点与data_fixed过程仍未知。']
 (a.output/'SUMMARY_ZH.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'status':'complete','paired_ci_rows':len(paired)}))
if __name__=='__main__':main()
