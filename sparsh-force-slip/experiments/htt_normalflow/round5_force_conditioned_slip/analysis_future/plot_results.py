#!/usr/bin/env python3
"""Render standalone Round-5 future evaluation figures from completed JSON artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS={"raw":"#1f77b4","raw_mul_pslip_fixed":"#d62728"}
MARKERS={"risk":"o","base":"s","full_state":"^"}
SOURCE_MARKERS={"z_p_slip":"o","z_p_slip_force":"s","z_p_slip_force_pred_delta":"^"}


def load_evaluation(path:Path,domain:str,allow_smoke:bool=False):
    payload=json.loads(path.read_text())
    if payload.get("status")!="complete" or payload.get("domain")!=domain:
        raise ValueError(f"invalid {domain} evaluation artifact")
    if not allow_smoke and payload.get("formal") is not True:
        raise ValueError("plotter refuses nonformal evaluation")
    return payload


def save(fig,output:Path,stem:str):
    output.mkdir(parents=True,exist_ok=True)
    fig.tight_layout(rect=(0,.16,1,.93))
    for suffix in ("svg","png"):
        fig.savefig(output/f"{stem}.{suffix}",dpi=180,bbox_inches="tight")
    plt.close(fig)


def plot_source(payload,output):
    learned=[r for r in payload["results"] if r.get("seed") is not None and r["method"] in COLORS]
    if not learned:raise ValueError("source artifact has no learned seed results")
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),sharex=True)
    for variant in sorted({r["variant"] for r in learned}):
        for method in COLORS:
            rows=[r for r in learned if r["variant"]==variant and r["method"]==method]
            if not rows:continue
            for metric,ax in zip(("average_precision","brier"),axes):
                for seed in sorted({r["seed"] for r in rows}):
                    points=sorted((r["horizon"],r["validation"][metric]) for r in rows if r["seed"]==seed)
                    ax.plot(*zip(*points),color=COLORS[method],marker=SOURCE_MARKERS[variant],alpha=.48,linewidth=1,
                            label=f"{variant} / {method}" if seed==min(r["seed"] for r in rows) else None)
    axes[0].set_ylabel("Average precision");axes[1].set_ylabel("Brier score")
    for ax in axes:ax.set_xlabel("Prediction horizon (frames)");ax.grid(alpha=.25);ax.set_xticks([1,3,5])
    method_handles=[plt.Line2D([],[],color=color,label=method) for method,color in COLORS.items()]
    variant_handles=[plt.Line2D([],[],marker=marker,linestyle="",color="black",label=variant) for variant,marker in SOURCE_MARKERS.items()]
    fig.legend(handles=method_handles+variant_handles,loc="lower center",bbox_to_anchor=(.5,.01),ncol=5,fontsize=8,frameon=False)
    fig.suptitle("Source future performance by horizon and seed",y=.98)
    save(fig,output,"source_ap_brier_by_horizon_seed")


def plot_htt(payload,output):
    learned=[r for r in payload["results"] if r.get("seed") is not None and r["method"] in COLORS]
    if not learned:raise ValueError("HTT artifact has no learned seed results")
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.5),sharex=True,sharey=True)
    ops=sorted({r["operating_point"] for r in learned});cmap=plt.get_cmap("viridis",len(ops));opcolor={op:cmap(i) for i,op in enumerate(ops)}
    for ax,method in zip(axes,COLORS):
        for row in learned:
            if row["method"]!=method:continue
            v=row["validation"]
            ax.scatter(v["fpr"],v["recall"],color=opcolor[row["operating_point"]],marker=MARKERS[row["variant"]],s=38,alpha=.75)
        ax.set_title(method);ax.set_xlabel("Observed validation FPR");ax.grid(alpha=.25);ax.set_xlim(-.02,1.02);ax.set_ylim(-.02,1.02)
    axes[0].set_ylabel("Observed validation recall")
    op_handles=[plt.Line2D([],[],marker="o",linestyle="",color=opcolor[o],label=o) for o in ops]
    variant_handles=[plt.Line2D([],[],marker=m,linestyle="",color="black",label=v) for v,m in MARKERS.items()]
    fig.legend(handles=op_handles+variant_handles,loc="lower center",bbox_to_anchor=(.5,.01),ncol=4,fontsize=8,frameon=False)
    fig.suptitle("HTT calibration-selected operating points applied to validation",y=.98)
    save(fig,output,"htt_validation_fpr_recall")


def plot_state(payload,output):
    records=payload.get("state_evaluation",[])
    if len(records)!=3:raise ValueError("expected three full-state seed records")
    horizons=[1,3,payload["horizon"]];fig,axes=plt.subplots(1,2,figsize=(10.5,4.2))
    for record in records:
        learned=record["learned_full_state"]
        axes[0].plot(horizons,[x["mse_raw"] for x in learned],marker="o",alpha=.65,label=f"learned seed {record['seed']}")
        axes[1].plot(horizons,[x["variance_replication_ratio"] for x in learned],marker="o",alpha=.65,label=f"learned seed {record['seed']}")
        if "persistence_zero_residual" in record:
            axes[0].plot(horizons,[x["mse_raw"] for x in record["persistence_zero_residual"]],"--",color="black",label="persistence")
            axes[1].plot(horizons,[x["variance_replication_ratio"] for x in record["persistence_zero_residual"]],"--",color="black",label="persistence")
            axes[0].plot(horizons,[x["mse_raw"] for x in record["train_only_ridge"]],":",color="#2ca02c",label="train-only ridge")
            axes[1].plot(horizons,[x["variance_replication_ratio"] for x in record["train_only_ridge"]],":",color="#2ca02c",label="train-only ridge")
    axes[0].set_ylabel("Raw residual MSE");axes[1].set_ylabel("Predicted / true variance")
    axes[1].axhline(1.,color="gray",linewidth=.8,alpha=.6)
    for ax in axes:ax.set_xlabel("State horizon (frames)");ax.set_xticks(horizons);ax.grid(alpha=.25)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,.01),ncol=3,fontsize=8,frameon=False)
    fig.suptitle("HTT latent-state residual prediction",y=.98)
    save(fig,output,"htt_state_error_variance_by_horizon_seed")


def main():
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--htt",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--allow-smoke",action="store_true");args=p.parse_args()
    source=load_evaluation(args.source,"source",args.allow_smoke);htt=load_evaluation(args.htt,"htt",args.allow_smoke)
    plot_source(source,args.output);plot_htt(htt,args.output);plot_state(htt,args.output)
    print(json.dumps({"status":"complete","figures":6,"output":str(args.output.resolve())}))


if __name__=="__main__":main()
