#!/usr/bin/env python3
import argparse,csv
from pathlib import Path
import matplotlib.pyplot as plt
def read(p):return list(csv.DictReader(open(p)))
def main():
 p=argparse.ArgumentParser();p.add_argument("--reporting",type=Path,required=True);p.add_argument("--final",type=Path,required=True);a=p.parse_args();a.final.mkdir(parents=True,exist_ok=True)
 s=read(a.reporting/"summary.csv");groups=("V","F_concat","F_dual");methods=("neural","predicted_current_persistence","fit_ridge_linear");labels=("Neural","Pred-current persistence","Fit ridge")
 fig,ax=plt.subplots(figsize=(8,4.6));w=.24
 for j,(m,l) in enumerate(zip(methods,labels)):
  y=[float(next(x for x in s if x["group"]==g and x["method"]==m and x["metric"]=="future_mae")["mean"]) for g in groups];ax.bar([i+(j-1)*w for i in range(3)],y,w,label=l)
 ax.set_xticks(range(3),groups);ax.set_ylabel("Validation future force MAE (N)");ax.set_title("Round 14 descriptive mean across folds, seeds, axes and horizons");ax.legend(frameon=False);ax.grid(axis="y",alpha=.25);fig.tight_layout();fig.savefig(a.final/"future_mae_overall.svg");plt.close(fig)
 h=read(a.final/"horizon_summary.csv");fig,ax=plt.subplots(figsize=(8,4.6))
 for g in groups:
  y=[float(next(x for x in h if x["group"]==g and x["method"]=="neural" and int(x["horizon"])==k)["future_mae_mean"]) for k in (1,5,10)];ax.plot((1,5,10),y,marker="o",label=g)
 pbase=[float(next(x for x in h if x["group"]=="V" and x["method"]=="predicted_current_persistence" and int(x["horizon"])==k)["future_mae_mean"]) for k in (1,5,10)];ax.plot((1,5,10),pbase,marker="s",linestyle="--",label="Pred-current persistence")
 ax.set_xticks((1,5,10));ax.set_xlabel("Horizon (frames)");ax.set_ylabel("Validation future force MAE (N)");ax.set_title("Horizon-resolved descriptive MAE");ax.legend(frameon=False);ax.grid(alpha=.25);fig.tight_layout();fig.savefig(a.final/"future_mae_by_horizon.svg");plt.close(fig)
if __name__=="__main__":main()
