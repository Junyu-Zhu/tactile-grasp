#!/usr/bin/env python3
import argparse,csv,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
import torch
import multitask_train as mt
def read(p):return list(csv.DictReader(open(p)))
def write(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with p.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def f(x):
 try:return float(x)
 except:return float("nan")
def mean(rows,key):
 x=np.asarray([f(r[key]) for r in rows]);return float(np.nanmean(x))
def weighted(rows,key):
 n=np.asarray([f(r["n"]) for r in rows]);v=np.asarray([f(r[key]) for r in rows]);return float(np.nansum(n*v)/np.nansum(n))
def pooled_rmse(rows):return float(math.sqrt(weighted(rows,"future_error_sq")))
def savefig(p):plt.tight_layout();plt.savefig(p,format="svg");plt.close()
def main():
 p=argparse.ArgumentParser();p.add_argument("--evaluation",type=Path,required=True);p.add_argument("--prepared-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--cost",type=Path,required=True);p.add_argument("--r14-current",type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);s=read(a.evaluation/"slip/metrics.csv");fm=read(a.evaluation/"future/metrics.csv");detail=read(a.evaluation/"future/trial_axis_horizon.csv");curves=read(a.evaluation/"slip/descriptive_matched_curves.csv");ci=read(a.evaluation/"bootstrap/PAIRED_CI.csv");th=read(a.evaluation/"slip/thresholds.csv");sel=json.loads((a.evaluation/"cases/SELECTED.json").read_text());cost=json.loads(a.cost.read_text())
 slip=[]
 for g in ("S","J"):
  for fold in range(1,5):
   for seed in mt.SEEDS:
    r=[x for x in s if x["group"]==g and int(x["fold"])==fold and int(x["seed"])==seed and x["role"]=="validation" and x["policy"]=="FPR0.05|raw"][0];slip.append({k:r[k] for k in ("group","fold","seed","frame_static_FPR","gross_recall","balanced_accuracy","macro_f1","AP","pAUC","positive_prevalence","false_starts_per_trial","static_alarming_frames","event_recall","mean_detected_event_delay","left_censored_events","right_boundary_events")})
 write(a.output/"SLIP_RUN_RESULTS.csv",slip)
 future=[]
 for g in ("F","J"):
  for fold in range(1,5):
   for seed in mt.SEEDS:
    rr=[x for x in fm if x["group"]==g and int(x["fold"])==fold and int(x["seed"])==seed and x["role"]=="validation" and x["predictor"]==g and x["stratum"]=="all"];future.append({"group":g,"fold":fold,"seed":seed,"future_mae":weighted(rr,"future_mae"),"future_rmse":pooled_rmse(rr),"deployed_change_mae":weighted(rr,"deployed_change_mae"),"copy_fraction_0p01n":weighted(rr,"copy_fraction_0p01n"),"prediction_variance":weighted(rr,"prediction_variance")})
 write(a.output/"FUTURE_RUN_RESULTS.csv",future)
 baselines=[]
 for pred in ("predicted_current_persistence","linear_ridge","ground_truth_current_persistence_ideal"):
  for fold in range(1,5):
   for seed in mt.SEEDS:
    rr=[x for x in fm if x["group"]=="F" and int(x["fold"])==fold and int(x["seed"])==seed and x["role"]=="validation" and x["predictor"]==pred and x["stratum"]=="all"];baselines.append({"predictor":pred,"fold":fold,"seed":seed,"future_mae":weighted(rr,"future_mae"),"future_rmse":pooled_rmse(rr),"deployed_change_mae":weighted(rr,"deployed_change_mae")})
 write(a.output/"BASELINE_RESULTS.csv",baselines)
 decomposition=[]
 for pred in ("F","J","predicted_current_persistence","linear_ridge","ground_truth_current_persistence_ideal"):
  rr=[x for x in detail if x["predictor"]==pred and x["stratum"]=="all"]
  decomposition.append({"predictor":pred,"n":sum(int(x["n"]) for x in rr),**{k:(pooled_rmse(rr) if k=="future_rmse" else weighted(rr,k)) for k in ("future_mae","future_rmse","true_change_abs_mean","predicted_deployed_change_abs_mean","deployed_change_mae","current_estimate_mae","anchor_sq","residual_minus_change_sq","cross_2_anchor_residual","future_error_sq","prediction_variance","target_variance","copy_fraction_0p01n","prediction_abs_ge20_fraction","target_at_clip_fraction")}})
 write(a.output/"FUTURE_ERROR_DECOMPOSITION.csv",decomposition)
 strata=[]
 for pred in ("F","J"):
  for sn in ("stable","transition","changing"):
   rr=[x for x in detail if x["predictor"]==pred and x["stratum"]==sn];strata.append({"predictor":pred,"stratum":sn,"n":sum(int(x["n"]) for x in rr),"future_mae":weighted(rr,"future_mae"),"future_rmse":pooled_rmse(rr),"deployed_change_mae":weighted(rr,"deployed_change_mae")})
 write(a.output/"FUTURE_STRATUM_RESULTS.csv",strata)
 summary=[]
 for g in ("S","J"):
  rr=[x for x in slip if x["group"]==g];summary.append({"task":"slip","model":g,**{k:mean(rr,k) for k in ("frame_static_FPR","gross_recall","balanced_accuracy","macro_f1","AP","pAUC","false_starts_per_trial","event_recall")}})
 for g in ("F","J"):
  rr=[x for x in future if x["group"]==g];summary.append({"task":"future","model":g,**{k:mean(rr,k) for k in ("future_mae","future_rmse","deployed_change_mae","copy_fraction_0p01n","prediction_variance")}})
 write(a.output/"HEADLINE_SUMMARY.csv",summary)
 heterogeneity=[]
 for unit in ("fold","seed"):
  vals=sorted({x[unit] for x in future},key=int)
  for v in vals:
   fv=mean([x for x in future if int(x[unit])==int(v) and x["group"]=="F"],"future_mae");jv=mean([x for x in future if int(x[unit])==int(v) and x["group"]=="J"],"future_mae");sv=mean([x for x in slip if int(x[unit])==int(v) and x["group"]=="S"],"pAUC");pj=mean([x for x in slip if int(x[unit])==int(v) and x["group"]=="J"],"pAUC");heterogeneity.append({"unit":unit,"value":v,"J_minus_F_future_mae_n":jv-fv,"J_minus_S_pAUC":pj-sv})
 write(a.output/"FOLD_SEED_HETEROGENEITY.csv",heterogeneity)
 matched=[]
 for kind,xkey,ykey,points in (("same_fpr","target_fpr","gross_recall",(.01,.05,.10)),("same_recall","target_recall","static_fpr",(.50,.75,.90))):
  for g in ("S","J"):
   for point in points:
    rr=[x for x in curves if x["kind"]==kind and x["group"]==g and math.isclose(f(x[xkey]),point,abs_tol=1e-9)];matched.append({"kind":kind,"group":g,xkey:point,ykey:mean(rr,ykey),"runs":len(rr),"descriptive_only":True,"calibration_thresholds_changed":False})
 write(a.output/"DESCRIPTIVE_MATCHED_OPERATING_POINTS.csv",matched)
 # Fold plots with units and complete legends.
 for metric,ylabel,name in (("pAUC","pAUC over FPR [0, 0.1]","slip_pauc_by_fold.svg"),("frame_static_FPR","Validation static FPR","slip_fpr_by_fold.svg"),("gross_recall","Validation gross recall","slip_recall_by_fold.svg")):
  for g,c in (("S","#377eb8"),("J","#e41a1c")):
   vals=[mean([x for x in slip if x["group"]==g and int(x["fold"])==fold],metric) for fold in range(1,5)];plt.plot(range(1,5),vals,"o-",label=g,color=c)
  plt.xlabel("HTT development fold");plt.ylabel(ylabel);plt.xticks(range(1,5));plt.grid(alpha=.25);plt.legend(title="Model");savefig(a.output/name)
 for g,c in (("F","#4daf4a"),("J","#e41a1c")):
  vals=[mean([x for x in future if x["group"]==g and int(x["fold"])==fold],"future_mae") for fold in range(1,5)];plt.plot(range(1,5),vals,"o-",label=g,color=c)
 plt.axhline(mean([x for x in baselines if x["predictor"]=="predicted_current_persistence"],"future_mae"),ls="--",color="black",label="Predicted-current persistence (run mean)");plt.xlabel("HTT development fold");plt.ylabel("Absolute future-force MAE (N)");plt.xticks(range(1,5));plt.grid(alpha=.25);plt.legend(title="Predictor");savefig(a.output/"future_mae_by_fold.svg")
 # CI forest, each fold kept separate.
 metrics=sorted(set(x["metric"] for x in ci));fig,axs=plt.subplots(1,4,figsize=(16,4),sharey=False)
 for ax,metric in zip(axs,metrics):
  rr=[x for x in ci if x["metric"]==metric];y=np.arange(1,5);m=np.array([f(x["mean"]) for x in rr]);lo=np.array([f(x["q025"]) for x in rr]);hi=np.array([f(x["q975"]) for x in rr]);ax.errorbar(m,y,xerr=[m-lo,hi-m],fmt="o",capsize=3);ax.axvline(0,color="black",lw=1);ax.set_yticks(y);ax.set_yticklabels([f"Fold {i}" for i in y]);ax.set_title(metric.replace("_","\n"),fontsize=8);ax.set_xlabel("Paired difference")
 plt.tight_layout();plt.savefig(a.output/"paired_ci_forest.svg",format="svg");plt.close()
 # Selected case curves and tactile source montages.
 case_rows=[]
 for n,case in enumerate(sel["selected"],1):
  fold=int(case["fold"]);seed=int(case["seed"]);ep=case["episode"];data=torch.load(a.prepared_root/f"p{fold}_s{seed}"/"prepared.pt",map_location="cpu",weights_only=False);d=data["roles"]["validation"];ids=np.asarray([i for i,e in enumerate(d["episode_id"]) if e==ep]);t=d["t"][ids].numpy();order=np.argsort(t);ids=ids[order];t=t[order]
  if case["kind"] in ("J_minus_S","S_minus_J"):
   for g,c in (("S","#377eb8"),("J","#e41a1c")):
    z=np.load(a.evaluation/"predictions"/g/f"p{fold}_s{seed}"/"validation.npz");mask=z["episode_id"]==ep;tt=z["t"][mask];oo=np.argsort(tt);score=z["score"][mask][oo];threshold=f([x for x in th if x["group"]==g and int(x["fold"])==fold and int(x["seed"])==seed and x["policy"]=="FPR0.05"][0]["threshold"]);plt.plot(tt[oo],score,label=f"{g} score",color=c);plt.axhline(threshold,ls="--",color=c,alpha=.7,label=f"{g} cal-FPR5 threshold")
   stage=d["stage"][ids].numpy();plt.step(t,stage/2,where="mid",color="gray",alpha=.35,label="Stage/2");plt.ylabel("Slip probability / stage scale");kind=case["kind"]
  else:
   y=d["y"][ids].numpy()
   for g,c in (("F","#4daf4a"),("J","#e41a1c")):
    z=np.load(a.evaluation/"predictions"/g/f"p{fold}_s{seed}"/"validation.npz");mask=z["episode_id"]==ep;tt=z["t"][mask];oo=np.argsort(tt);err=np.abs(z["future"][mask][oo]-y).mean((1,2));plt.plot(tt[oo],err,label=f"{g} mean |error|",color=c)
   plt.ylabel("Future-force absolute error (N)");kind=case["kind"]
  plt.xlabel("Native frame index t");plt.title(f"{kind}: {ep}, fold {fold}, seed {seed}");plt.grid(alpha=.25);plt.legend(fontsize=7);savefig(a.output/f"case_{n}_{kind}.svg");case_rows.append(case)
  parent=torch.load(data["provenance"]["parent_prepared"],map_location="cpu",weights_only=False);rec=next(x for x in parent["episodes"] if x["episode_id"]==ep);src=np.load(rec["source_path"]);frames=src["tactile_img"];pick=[int(t[0]),int(t[len(t)//2]),int(t[-1])];ims=[]
  ref=np.asarray(src["ref_frame"]);ref=(255*(ref-ref.min())/(ref.max()-ref.min()+1e-12)).astype(np.uint8);ims.append(("reference",ref))
  for q in pick:ims.append((f"t={q}",frames[q]))
  canvas=Image.new("RGB",(224*4,250),"white");draw=ImageDraw.Draw(canvas)
  for j,(label,im) in enumerate(ims):canvas.paste(Image.fromarray(im).convert("RGB"),(224*j,26));draw.text((224*j+4,5),label,fill="black")
  canvas.save(a.output/f"case_{n}_{kind}_source_frames.png")
 write(a.output/"SELECTED_CASES.csv",case_rows)
 hist={"schema":"round17_historical_reference_v1","scope":"accepted R12/R13 models already re-evaluated by Round14 on the same complete endpoints; original and common-endpoint results retained","files":{x.name:{"path":str(x),"sha256":mt.sha(x)} for x in a.r14_current.glob("*.csv")}};mt.atomic_json(hist,a.output/"HISTORICAL_REFERENCE.json")
 ss={x["model"]:x for x in summary if x["task"]=="slip"};fs={x["model"]:x for x in summary if x["task"]=="future"};persistence=mean([x for x in baselines if x["predictor"]=="predicted_current_persistence"],"future_mae");dec={x["predictor"]:x for x in decomposition};strat={(x["predictor"],x["stratum"]):x for x in strata};folds=[x for x in heterogeneity if x["unit"]=="fold"];seeds=[x for x in heterogeneity if x["unit"]=="seed"]
 lines=["# 第十七轮结果总结","",f"36/36正式运行完成；test-role未读取。S/J在相同共同端点和冻结的内部selection滑移规则下比较，F由内部future MAE选择，J的future结果来自同一个滑移导向checkpoint。","",f"- 滑移排序：S的validation pAUC均值为{float(ss['S']['pAUC']):.4f}，J为{float(ss['J']['pAUC']):.4f}；AP分别为{float(ss['S']['AP']):.4f}和{float(ss['J']['AP']):.4f}。",f"- calibration FPR5%迁移到validation后，J把gross recall从{float(ss['S']['gross_recall']):.4f}提高到{float(ss['J']['gross_recall']):.4f}，同时static FPR从{float(ss['S']['frame_static_FPR']):.4f}升到{float(ss['J']['frame_static_FPR']):.4f}，false starts/trial从{float(ss['S']['false_starts_per_trial']):.4f}升到{float(ss['J']['false_starts_per_trial']):.4f}。描述性同FPR/同召回曲线另列于DESCRIPTIVE_MATCHED_OPERATING_POINTS.csv及两张validation曲线图；它们没有改变calibration阈值。",f"- 未来力（先逐run汇总再取均值）：F的validation绝对MAE为{float(fs['F']['future_mae']):.4f} N，J为{float(fs['J']['future_mae']):.4f} N，预测当前力保持基线为{persistence:.4f} N。逐试次×窗口×轴结果见future/trial_axis_horizon.csv；RMSE均由合并平方误差后开方计算。",f"- 以下误差分解和strata仅使用全部validation开发评价端点，按端点数目加权描述汇总；它们不是独立样本统计，也不是headline的逐run等权汇总。当前力锚点误差平方项两组相同，均为{float(dec['F']['anchor_sq']):.4f} N²。F的变化残差平方项为{float(dec['F']['residual_minus_change_sq']):.4f} N²、交叉项为{float(dec['F']['cross_2_anchor_residual']):.4f} N²；J分别为{float(dec['J']['residual_minus_change_sq']):.4f}和{float(dec['J']['cross_2_anchor_residual']):.4f} N²。负交叉项表示部分误差抵消，但J更大的变化残差仍把总未来误差平方从{float(dec['F']['future_error_sq']):.4f}推到{float(dec['J']['future_error_sq']):.4f} N²。真实绝对变化均值为{float(dec['F']['true_change_abs_mean']):.4f} N，F/J预测相对部署锚点的绝对变化为{float(dec['F']['predicted_deployed_change_abs_mean']):.4f}/{float(dec['J']['predicted_deployed_change_abs_mean']):.4f} N。",f"- 复制、塌缩和饱和诊断没有给J提供替代解释：F/J在0.01 N内复制部署锚点的比例为{100*float(dec['F']['copy_fraction_0p01n']):.2f}%/{100*float(dec['J']['copy_fraction_0p01n']):.2f}%，逐trial-cell预测方差加权均值为{float(dec['F']['prediction_variance']):.4f}/{float(dec['J']['prediction_variance']):.4f} N²（目标{float(dec['F']['target_variance']):.4f} N²），预测|force|≥20 N比例为{100*float(dec['F']['prediction_abs_ge20_fraction']):.3f}%/{100*float(dec['J']['prediction_abs_ge20_fraction']):.3f}%，目标位于±20 N裁剪边界的比例为{100*float(dec['F']['target_at_clip_fraction']):.3f}%。后者是目标裁剪比例，不是预测裁剪。",f"- 负迁移覆盖fit-only定义的稳定与变化端点：stable上F/J MAE为{float(strat[('F','stable')]['future_mae']):.4f}/{float(strat[('J','stable')]['future_mae']):.4f} N（J−F {float(strat[('J','stable')]['future_mae'])-float(strat[('F','stable')]['future_mae']):+.4f}），changing上为{float(strat[('F','changing')]['future_mae']):.4f}/{float(strat[('J','changing')]['future_mae']):.4f} N（J−F {float(strat[('J','changing')]['future_mae'])-float(strat[('F','changing')]['future_mae']):+.4f}）。",f"- 折间异质性明显：J−F future MAE四折为"+", ".join(f"{f(x['J_minus_F_future_mae_n']):+.4f}" for x in folds)+" N；J−S pAUC四折为"+", ".join(f"{f(x['J_minus_S_pAUC']):+.4f}" for x in folds)+"。三个seed的J−F future MAE为"+", ".join(f"{f(x['J_minus_F_future_mae_n']):+.4f}" for x in seeds)+" N。配对CI按折保留这种异质性，不把重叠折、seed、窗口或帧当独立样本。","- 训练时S/J共同冻结选择器按逐样本累计pAUC且未合并同分ties；最终评价使用标准tie合并ROC。已知最大复核例S/p1_s20260915为0.86364234对0.86364854，差6.20e-6，而冻结best与次佳的implemented差为0.0051356。历史checkpoint保持不变；没有保存所有epoch预测，因此不推断标准ROC重选是否处处相同。","- 成本收益只针对冻结上游输出之后的新时序模块：J相对独立S+F少22,809个新参数（49.47%）。冻结MAE/force上游本来也能供两个独立头共享，所以该数字不代表完整部署参数或端到端时延减半。","- 结论：本轮固定共享多任务方案不保留为性能升级。它在部分折提高召回，但总体低FPR排序下降、误报代价上升，并损害未来力；新增时序模块的参数共享收益不足以抵消性能结果。","","边界：这是HTT开发评价，冻结上游历史上有validation暴露；标签部分依赖力规则；没有独立盲测、实物闭环或因果证明。"]
 (a.output/"SUMMARY_ZH.md").write_text("\n".join(lines)+"\n");audit={"schema":"round17_report_v2","status":"complete","tables":len(list(a.output.glob("*.csv"))),"svgs":len(list(a.output.glob("*.svg"))),"pngs":len(list(a.output.glob("*.png"))),"future_rmse_definition":"sqrt(endpoint-count-weighted future_error_sq)","trial_axis_horizon":True,"descriptive_matched_curves":True,"selector_tie_disclosed":True,"cost_scope":"new temporal modules only","cost_sha256":mt.sha(a.cost),"historical_reference":True};mt.atomic_json(audit,a.output/"REPORT_AUDIT.json");print(json.dumps(audit,indent=2))
if __name__=="__main__":main()
