#!/usr/bin/env python3
import argparse,csv,json
from pathlib import Path
import numpy as np
import future_train as ft
def read(p):return list(csv.DictReader(open(p)))
def write(p,rows):
 with p.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 p=argparse.ArgumentParser();p.add_argument("--reporting",type=Path,required=True);p.add_argument("--current",type=Path,required=True);p.add_argument("--training-audit",type=Path,required=True);p.add_argument("--cost",type=Path,required=True);p.add_argument("--failure",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 rows=read(a.reporting/"all_metrics.csv");run=[]
 for g in ft.GROUPS:
  for f in range(1,5):
   for s in (20260914,20260915,20260916):
    for method in ("neural","predicted_current_persistence","fit_ridge_linear","gt_current_persistence_ideal_only"):
     for h in ft.HORIZONS:
      z=[x for x in rows if x["group"]==g and int(x["fold"])==f and int(x["seed"])==s and x["method"]==method and x["role"]=="validation" and x["stratum"]=="all" and int(x["horizon"])==h]
      run.append({"group":g,"fold":f,"seed":s,"method":method,"horizon":h,"future_mae_axis_mean":float(np.mean([float(x["future_mae"]) for x in z])),"future_rmse_axis_mean":float(np.mean([float(x["future_rmse"]) for x in z])),"change_mae_axis_mean":float(np.mean([float(x["change_mae"]) for x in z]))})
 write(a.output/"seed_fold_horizon.csv",run)
 desc=[]
 for g in ft.GROUPS:
  for method in ("neural","predicted_current_persistence","fit_ridge_linear","gt_current_persistence_ideal_only"):
   for h in ft.HORIZONS:
    z=[x["future_mae_axis_mean"] for x in run if x["group"]==g and x["method"]==method and x["horizon"]==h];desc.append({"group":g,"method":method,"horizon":h,"future_mae_mean":float(np.mean(z)),"seed_fold_descriptive_sd":float(np.std(z,ddof=1)),"runs":len(z)})
 write(a.output/"horizon_summary.csv",desc)
 strata=[]
 for g in ft.GROUPS:
  for st in ("stable","transitional","changing"):
   z=[float(x["future_mae"]) for x in rows if x["group"]==g and x["method"]=="neural" and x["role"]=="validation" and x["stratum"]==st]
   b=[float(x["future_mae"]) for x in rows if x["group"]==g and x["method"]=="predicted_current_persistence" and x["role"]=="validation" and x["stratum"]==st]
   strata.append({"group":g,"stratum":st,"neural_future_mae":float(np.mean(z)),"persistence_future_mae":float(np.mean(b)),"difference":float(np.mean(z)-np.mean(b)),"cells":len(z)})
 write(a.output/"strata_summary.csv",strata)
 def overall(g,m):return float(np.mean([float(x["future_mae"]) for x in rows if x["group"]==g and x["method"]==m and x["role"]=="validation" and x["stratum"]=="all"]))
 vals={g:{m:overall(g,m) for m in ("neural","predicted_current_persistence","fit_ridge_linear","gt_current_persistence_ideal_only")} for g in ft.GROUPS}
 cur=json.loads((a.current/"SUMMARY.json").read_text());ta=json.loads(a.training_audit.read_text());cost=json.loads(a.cost.read_text());failure=json.loads((a.failure/"FAILURE_CASE_AUDIT.json").read_text());collapse=read(a.reporting/"temporal_collapse_summary.csv");cf=read(a.current/"full_endpoints.csv");cc=read(a.current/"common_endpoints.csv");current_delta=[]
 for point in ("fixed_0.5","maxBA","FPR0.01","FPR0.05","FPR0.10"):
  for rule in ("raw","confirm2"):
   for metric in ("frame_static_FPR","gross_recall","balanced_accuracy","event_recall"):
    deltas=[float(y[metric])-float(x[metric]) for x,y in zip(cf,cc) if x["role"]=="validation" and x["point"]==point and x["rule"]==rule and x[metric] not in ("",None) and y[metric] not in ("",None)]
    current_delta.append({"point":point,"rule":rule,"metric":metric,"common_minus_full_mean":float(np.mean(deltas)),"min":float(np.min(deltas)),"max":float(np.max(deltas)),"runs":len(deltas)})
 write(a.output/"current_common_change.csv",current_delta);fpr5=next(x for x in current_delta if x["point"]=="FPR0.05" and x["rule"]=="raw" and x["metric"]=="frame_static_FPR")
 lines=["# 第十四轮：HTT未来参考相对力预测（P0/P1/P2）","","## 执行与边界",f"36/36个预注册future模型训练并通过训练审计；每组四折×三seed。当前slip没有新训练，共同端点评价复用48个冻结历史模型。test未读取。未来目标为逐轴裁剪±20 N的参考相对力，窗口为1/5/10帧。以下跨折与seed均为重叠开发数据的描述性统计，不是独立物理重复。","","## Validation未来力主结果","","|组|神经MAE (N)|预测当前力保持 (N)|fit线性 (N)|GT当前保持理想参考 (N)|","|---|---:|---:|---:|---:|"]
 for g in ft.GROUPS: lines.append(f"|{g}|{vals[g]['neural']:.4f}|{vals[g]['predicted_current_persistence']:.4f}|{vals[g]['fit_ridge_linear']:.4f}|{vals[g]['gt_current_persistence_ideal_only']:.4f}|")
 collapse_text="；".join(f"{x['group']} {int(float(x['collapse_flag_cells']))}/{int(float(x['episode_axis_horizon_cells']))}" for x in collapse if x["method"]=="neural")
 lines += ["","F_concat相对V的未来力MAE较低，但总体仍未超过部署一致的预测当前力保持基线。F_dual总体不优于F_concat，也未超过保持。由此只能支持“预测力历史改善纯视觉神经预测”这一受限观察，不能声称新的时序模型优于简单保持，更不能声称完整世界模型或future-slip能力。","","四折完整泄漏组bootstrap中，F_concat−V的MAE差四折区间均为负；F_dual−F_concat三折区间为正、一折跨0。CI按折分别解释，不将重叠折当独立样本。逐seed/折/窗口见`seed_fold_horizon.csv`，分窗口与变化分层见对应表。","","## 变化、复制与固定失败案例",f"塌缩诊断按validation每试次×轴×窗口计算时间标准差；仅当预测std≤0.05 N且GT std≥0.25 N时标记。神经模型标记计数为：{collapse_text}。GT近常数单列，不能用跨轴/跨窗口总体std排除塌缩；完整结果见`temporal_diagnostics.csv`与`temporal_collapse_summary.csv`。",f"结果后描述性失败案例按固定不利规则从{failure['candidate_count']}个完整试次候选中选择：最大“神经MAE−预测当前保持MAE”。选中{failure['selected']['group']} fold{failure['selected']['fold']} seed{failure['selected']['seed']} `{failure['selected']['episode_id']}`，差值{failure['selected']['neural_minus_persistence_n']:.3f} N。全候选、数值曲线及t/t+10源触觉图均保留；该案例不是预注册选择，也不是总体证据。","","## 当前slip共同端点",f"复算保留48个历史模型。历史5工作点×raw/confirm2全/共同端点各{cur['historical_full_metric_rows']}行；R13两类新策略×alpha×k各{cur['r13_new_policy_rows_each_scope']}行，confirm4 reference各{cur['r13_confirm4_rows_each_scope']}行。全部复用保存阈值，不重新拟合或搜索；先在完整原时间线生成alarm/starts，再mask共同端点，共{cur['state_first_parity_checks']}次逐值一致。所有静态/gross/event分母与左右删失字段均保留。validation历史FPR5/raw共同端点相对全端点帧FPR描述性平均增加{100*fpr5['common_minus_full_mean']:.2f}个百分点，范围{100*fpr5['min']:.2f}至{100*fpr5['max']:.2f}个百分点，说明共同窗口本身会改变报告分母，不能解释为模型变化。","","## 成本",f"新增future头参数为{cost['future_heads']['V']['parameters']}/{cost['future_heads']['F_concat']['parameters']}/{cost['future_heads']['F_dual']['parameters']}，FP32权重约{cost['future_heads']['V']['parameter_memory_bytes_fp32']/1024:.1f}/{cost['future_heads']['F_concat']['parameter_memory_bytes_fp32']/1024:.1f}/{cost['future_heads']['F_dual']['parameter_memory_bytes_fp32']/1024:.1f} KiB；batch1缓存头GPU中位延迟为{cost['future_heads']['V']['head_latency_ms_median']:.3f}/{cost['future_heads']['F_concat']['head_latency_ms_median']:.3f}/{cost['future_heads']['F_dual']['head_latency_ms_median']:.3f} ms。完整绝对预测管线参数约为{cost['future_heads']['V']['complete_absolute_pipeline_parameters']:,}/{cost['future_heads']['F_concat']['complete_absolute_pipeline_parameters']:,}/{cost['future_heads']['F_dual']['complete_absolute_pipeline_parameters']:,}。完整管线memory仅给FP32 weights-only算术，未重测运行峰值；头延迟/显存不含预处理、MAE、视觉/force分支、IO或控制。","","## 限制","","上游checkpoint与外层validation有历史开发暴露；四折重叠；预测力来自同一触觉图像；HTT标签部分由力规则产生；没有可信物理时间戳、独立test、实物闭环或动作条件。R10只证明同一NPZ中的力/参考数组与帧数一致，不能替代独立硬件时间同步证明。ToucHD-Force 145个文件共366,070,997,896 bytes已完成本地原始文件与服务器副本全量SHA256一致性核验，但P3不在本轮范围，未使用该数据训练。"]
 (a.output/"SUMMARY_ZH.md").write_text("\n".join(lines)+"\n")
 result={"schema":"round14_report_v1","status":"complete","training_runs":ta["accepted_count"],"current_models":cur["models"],"main_values":vals,"hashes":{p.name:ft.sha(p) for p in a.output.iterdir() if p.is_file() and p.name!="REPORT_AUDIT.json"}};ft.atomic_json(result,a.output/"REPORT_AUDIT.json");print(json.dumps(result))
if __name__=="__main__":main()
