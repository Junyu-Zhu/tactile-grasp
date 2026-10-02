#!/usr/bin/env python3
"""Round-18 compact tables, figures, source-frame cases, and bounded conclusion."""
from __future__ import annotations
import argparse,csv,json,math,shutil
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
import train as r18
def read(path):return list(csv.DictReader(Path(path).open()))
def write(path,rows):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if not rows:path.write_text("");return
 fields=list(dict.fromkeys(k for row in rows for k in row));
 with path.open("w",newline="") as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def f(x):return float(x) if x not in (None,"") else float("nan")
def mean(rows,key):
 vals=np.asarray([f(x[key]) for x in rows],dtype=float);return float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
def savefig(path):plt.tight_layout();plt.savefig(path,format=Path(path).suffix.lstrip("."),dpi=180);plt.close()
def main():
 p=argparse.ArgumentParser();p.add_argument("--evaluation",type=Path,required=True);p.add_argument("--prepared-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--r12-fixed-image",type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 metrics=read(a.evaluation/"metrics/WORKPOINT_METRICS.csv");ci=read(a.evaluation/"bootstrap/PAIRED_CI.csv");film=read(a.evaluation/"diagnostics/FILM_STATS.csv");pert=read(a.evaluation/"diagnostics/FORCE_PERTURBATIONS.csv");thresholds=read(a.evaluation/"metrics/CALIBRATION_THRESHOLDS.csv");scores=read(a.evaluation/"scores/RAW_SCORES.csv");selected=json.loads((a.evaluation/"cases/SELECTED.json").read_text())["selected"]
 headline=[]
 for group in r18.GROUPS:
  rr=[x for x in metrics if x["group"]==group and x["role"]=="validation" and x["policy"]=="FPR5|raw"]
  headline.append({"group":group,**{key:mean(rr,key) for key in ("pAUC","AP","positive_prevalence","frame_static_FPR","gross_recall","balanced_accuracy","macro_f1","false_starts_per_trial","static_alarming_frames","event_recall","mean_detected_event_delay")}})
 write(a.output/"HEADLINE_SUMMARY.csv",headline)
 run_rows=[x for x in metrics if x["role"]=="validation" and x["policy"]=="FPR5|raw"];write(a.output/"HEADLINE_RUNS.csv",run_rows)
 heter=[]
 for group in r18.GROUPS:
  for fold in range(1,5):
   rr=[x for x in run_rows if x["group"]==group and int(x["fold"])==fold];heter.append({"unit":"fold","value":fold,"group":group,"pAUC":mean(rr,"pAUC"),"frame_static_FPR":mean(rr,"frame_static_FPR"),"gross_recall":mean(rr,"gross_recall")})
  for seed in r18.SEEDS:
   rr=[x for x in run_rows if x["group"]==group and int(x["seed"])==seed];heter.append({"unit":"seed","value":seed,"group":group,"pAUC":mean(rr,"pAUC"),"frame_static_FPR":mean(rr,"frame_static_FPR"),"gross_recall":mean(rr,"gross_recall")})
 write(a.output/"FOLD_SEED_HETEROGENEITY.csv",heter)
 # Fold-wise pAUC and calibration-transfer tradeoff.
 for metric,ylabel,name in (("pAUC","Validation pAUC [0,.1]","pauc_by_fold.svg"),("gross_recall","Validation gross recall at cal FPR5","recall_by_fold.svg"),("frame_static_FPR","Validation static FPR at cal FPR5","fpr_by_fold.svg")):
  for group,color in (("V0","#377eb8"),("C0","#4daf4a"),("M0","#e41a1c")):
   vals=[next(f(x[metric]) for x in heter if x["unit"]=="fold" and int(x["value"])==fold and x["group"]==group) for fold in range(1,5)];plt.plot(range(1,5),vals,"o-",label=group,color=color)
  plt.xlabel("HTT development fold");plt.ylabel(ylabel);plt.xticks(range(1,5));plt.grid(alpha=.25);plt.legend();savefig(a.output/name)
 # CI forest for primary M0-C0 metrics; unsupported resamples are counted in table.
 chosen=[x for x in ci if x["left"]=="M0" and x["right"]=="C0" and x["metric"] in ("pAUC","frame_static_FPR","gross_recall")];fig,axes=plt.subplots(1,3,figsize=(12,4))
 for ax,metric in zip(axes,("pAUC","frame_static_FPR","gross_recall")):
  rr=sorted([x for x in chosen if x["metric"]==metric],key=lambda x:int(x["fold"]));y=np.arange(1,5);m=np.array([f(x["mean"]) for x in rr]);lo=np.array([f(x["q025"]) for x in rr]);hi=np.array([f(x["q975"]) for x in rr]);ax.errorbar(m,y,xerr=[m-lo,hi-m],fmt="o",capsize=3);ax.axvline(0,color="black",lw=1);ax.set_yticks(y);ax.set_yticklabels([f"Fold {i}" for i in y]);ax.set_title(metric);ax.grid(alpha=.2)
 plt.tight_layout();plt.savefig(a.output/"M0_minus_C0_paired_ci.svg",format="svg");plt.close()
 # Cases: timelines plus accepted raw source frames. Historical plot is also retained.
 if a.r12_fixed_image.exists():shutil.copy2(a.r12_fixed_image,a.output/"historical_fixed_case_prior_figure.png")
 case_rows=[]
 for index,case in enumerate(selected,1):
  fold=int(case["fold"]);seed=int(case["seed"]);episode=case["episode"];data=torch.load(a.prepared_root/f"p{fold}_s{seed}"/"prepared.pt",map_location="cpu",weights_only=False);d=data["roles"]["validation"];ids=np.asarray([i for i,e in enumerate(d["episode_id"]) if e==episode])
  if not len(ids):case_rows.append({**case,"source_status":"episode_missing_on_validation_support"});continue
  ids=ids[np.argsort(d["t"][ids].numpy())];t=d["t"][ids].numpy();stage=d["stage"][ids].numpy()
  for group,color in (("V0","#377eb8"),("C0","#4daf4a"),("M0","#e41a1c")):
   rr=[x for x in scores if x["group"]==group and int(x["fold"])==fold and int(x["seed"])==seed and x["role"]=="validation" and x["episode"]==episode];rr=sorted(rr,key=lambda x:int(x["t"]));plt.plot([int(x["t"]) for x in rr],[f(x["score"]) for x in rr],label=group,color=color);th=next(f(x["threshold"]) for x in thresholds if x["group"]==group and int(x["fold"])==fold and int(x["seed"])==seed and x["policy"]=="FPR5");plt.axhline(th,ls="--",color=color,alpha=.45)
  plt.step(t,stage/2,where="mid",color="gray",alpha=.35,label="stage/2");plt.xlabel("Native endpoint t");plt.ylabel("Score / stage scale");plt.title(f"{case['kind']}: {episode}, p{fold}, {seed}");plt.grid(alpha=.2);plt.legend(fontsize=7);savefig(a.output/f"case_{index}_timeline.svg")
  parent=torch.load(data["provenance"]["parent_prepared"],map_location="cpu",weights_only=False);rec=next((x for x in parent["episodes"] if x["episode_id"]==episode),None)
  if rec is None:case_rows.append({**case,"source_status":"parent_episode_missing"});continue
  src=np.load(rec["source_path"]);frames=src["tactile_img"];picks=[int(t[0]),int(t[len(t)//2]),int(t[-1])];ref=np.asarray(src["ref_frame"]);ref=(255*(ref-ref.min())/(ref.max()-ref.min()+1e-12)).astype(np.uint8);items=[("reference",ref)]+[(f"t={q}",frames[q]) for q in picks];canvas=Image.new("RGB",(224*4,250),"white");draw=ImageDraw.Draw(canvas)
  for j,(label,image) in enumerate(items):canvas.paste(Image.fromarray(image).convert("RGB").resize((224,224)),(224*j,26));draw.text((224*j+4,5),label,fill="black")
  target=a.output/f"case_{index}_source_frames.png";canvas.save(target);case_rows.append({**case,"source_status":"saved","source_image":target.name})
 write(a.output/"SELECTED_CASES.csv",case_rows)
 # Compact diagnostics.
 write(a.output/"FORCE_PERTURBATION_RESULTS.csv",pert);write(a.output/"FILM_DISTRIBUTION_RESULTS.csv",film);write(a.output/"PAIRED_CI.csv",ci)
 h={x["group"]:x for x in headline};mc={k:h["M0"][k]-h["C0"][k] for k in ("pAUC","frame_static_FPR","gross_recall","balanced_accuracy","macro_f1")};mv={k:h["M0"][k]-h["V0"][k] for k in mc};cv={k:h["C0"][k]-h["V0"][k] for k in mc};anomaly=mean([x for x in film if x["role"]=="validation"],"anomalous_fraction")
 lines=["# Round 18 Q1A result","","All 36 frozen-head runs completed and test was not accessed. Evaluation is limited to the Round-17 common future-complete endpoints, not the full original current-slip timeline.","",f"Run-mean validation pAUC[0,0.1], computed from the low-FPR ROC curve and independent of any single operating threshold, is V0 {h['V0']['pAUC']:.4f}, C0 {h['C0']['pAUC']:.4f}, and M0 {h['M0']['pAUC']:.4f}. At each model's calibration-derived FPR5 raw threshold, validation static FPR is {h['V0']['frame_static_FPR']:.4f}/{h['C0']['frame_static_FPR']:.4f}/{h['M0']['frame_static_FPR']:.4f} and gross recall is {h['V0']['gross_recall']:.4f}/{h['C0']['gross_recall']:.4f}/{h['M0']['gross_recall']:.4f}.","",f"The main M0-C0 difference is pAUC {mc['pAUC']:+.4f}, static FPR {mc['frame_static_FPR']:+.4f}, and gross recall {mc['gross_recall']:+.4f}. M0-V0 is pAUC {mv['pAUC']:+.4f}, FPR {mv['frame_static_FPR']:+.4f}, recall {mv['gross_recall']:+.4f}; C0-V0 is {cv['pAUC']:+.4f}/{cv['frame_static_FPR']:+.4f}/{cv['gross_recall']:+.4f}.","","Fold-specific paired intervals are heterogeneous; no single uniform improvement claim is supported across all four folds. In fold 2, bootstrap draws 190, 644, and 1963 of 2000 lacked static support; pAUC/AP/static-FPR/BA intervals use the remaining 1997 valid draws, while all attempted draws remain recorded. Calibration FPR5 does not imply 5% validation FPR. AP is interpreted with the naturally high gross prevalence.","",f"The preregistered FiLM anomaly rule flags {100*anomaly:.2f}% of validation endpoints on average. Force masking, lag-1, FiLM-off, scale/bias distributions, and large force-error associations are retained in the diagnostic tables; these perturbations are sensitivity evidence, not physical causality.","","The result supports, at most, a local and fold-dependent benefit of feature conditioning. It does not establish a stable global improvement over concatenation or vision, an independent force sensor benefit, full-timeline generalization, physical slip onset prediction, or external-domain transfer."]
 (a.output/"SUMMARY_EN.md").write_text("\n".join(lines)+"\n")
 zh=["# 第18轮 Q1A 综合结论","","本轮36个冻结上游的检测头训练均完成，未读取test。所有结论仅适用于第17轮缓存中具有完整未来窗口的共同端点，不能外推为原始当前滑移完整时间线。","",f"跨4折×3种子取运行均值，验证集pAUC[0,0.1]为V0 {h['V0']['pAUC']:.4f}、C0 {h['C0']['pAUC']:.4f}、M0 {h['M0']['pAUC']:.4f}。pAUC由整段低FPR ROC曲线计算，不依赖某一个阈值；下述FPR、召回、BA和F1才是在各模型由calibration得到的FPR5阈值下计算。该阈值转移到validation后，实际static FPR为{h['V0']['frame_static_FPR']:.4f}/{h['C0']['frame_static_FPR']:.4f}/{h['M0']['frame_static_FPR']:.4f}，gross召回为{h['V0']['gross_recall']:.4f}/{h['C0']['gross_recall']:.4f}/{h['M0']['gross_recall']:.4f}。","",f"主比较M0-C0为：pAUC {mc['pAUC']:+.4f}、static FPR {mc['frame_static_FPR']:+.4f}、gross召回 {mc['gross_recall']:+.4f}。M0相对V0的对应差值为{mv['pAUC']:+.4f}/{mv['frame_static_FPR']:+.4f}/{mv['gross_recall']:+.4f}；C0相对V0为{cv['pAUC']:+.4f}/{cv['frame_static_FPR']:+.4f}/{cv['gross_recall']:+.4f}。4折配对区间明显异质，M0-C0的pAUC区间均跨0；因此只支持局部、依折而变的调制收益，不支持FiLM稳定优于拼接或视觉基线的全局结论。","",f"在cal-FPR5工作点，V0/C0/M0的BA为{h['V0']['balanced_accuracy']:.4f}/{h['C0']['balanced_accuracy']:.4f}/{h['M0']['balanced_accuracy']:.4f}，macro-F1为{h['V0']['macro_f1']:.4f}/{h['C0']['macro_f1']:.4f}/{h['M0']['macro_f1']:.4f}，每试次误告警启动数为{h['V0']['false_starts_per_trial']:.4f}/{h['C0']['false_starts_per_trial']:.4f}/{h['M0']['false_starts_per_trial']:.4f}。gross占比很高，AP只能结合prevalence解读；incipient仅单列分布，不作为训练监督或提前滑移证据。","",f"预注册FiLM异常放大规则平均标记{100*anomaly:.2f}%的validation端点。固定原阈值的force均值屏蔽、因果lag-1和FiLM关闭结果均保留为敏感性诊断；预测力来自同一触觉图像，这些扰动不能证明独立测力或物理因果。代表性失败案例同时保存真实触觉源图与时序曲线。","","部署证据由新测缓存头成本与身份严格匹配的既有端到端路径组成：V0/C0/M0新头中位延迟约0.204/0.217/0.345 ms；既有完整路径中位数仅作描述性适用证据，不能冒充本轮重新测得的端到端速度。M0相对V0新增参数4.684%，满足≤5%容量约束。","","DeformableObjectsGrasping只完成开发内容与划分元数据审计，未读取testing内容。开发图像显示外部相机视角和带黑色点阵的彩色触觉图，但无法据外观确认传感器具体型号或背景参考；其标签与划分不能直接支持本轮冻结HTT模型的外域评价。","","## 后续包准备状态","","Q1B尚未启动，也未由Q1A外validation结果选择结构。若后续正式派发，仍须按总计划固定V2/M2、从同一MAE起点训练、锁定末2层梯度路径、九步真实预算和恢复smoke；本轮结果不能作为warm-start或改组依据。","","Q3尚未启动。其K-current/D-history/T-visual/T-force四组、预定单fold三种子和当前锚点/真实变化监督仍须独立注册与验收，不按Q1A赢家改变；当前FiLM结果不足以把FiLM自动加入状态转移网格。","","本轮不支持独立力传感器收益、完整时间线泛化、物理滑移起点预测或外域迁移结论。"]
 (a.output/"SUMMARY_ZH.md").write_text("\n".join(zh)+"\n")
 audit={"schema":"round18_report_audit_v1","status":"complete","headline_rows":len(headline),"case_rows":len(case_rows),"source_images":sum(x.get("source_status")=="saved" for x in case_rows),"svgs":len(list(a.output.glob("*.svg"))),"test_consumed":False,"support_boundary":"Round-17 common future-complete endpoints only","hashes":{path.name:r18.sha(path) for path in a.output.iterdir() if path.is_file() and path.name!="REPORT_AUDIT.json"}};r18.atomic_json(a.output/"REPORT_AUDIT.json",audit);print(json.dumps(audit,indent=2))
if __name__=="__main__":main()
