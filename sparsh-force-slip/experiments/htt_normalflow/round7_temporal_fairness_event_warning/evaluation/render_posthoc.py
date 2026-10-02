#!/usr/bin/env python3
"""Display-only rendering from immutable Round-7 metric CSVs."""
from __future__ import annotations
import argparse,csv,hashlib,json,os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
GROUPS=("A_visual","B_force","C_force_delta","D_visual_delta");SEEDS=(20260914,20260915,20260916);COLORS=dict(zip(GROUPS,("tab:blue","tab:orange","tab:green","tab:red")));STYLES=("-","--",":")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(path,v):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f".tmp.{os.getpid()}");tmp.write_text(json.dumps(v,indent=2,sort_keys=True)+"\n");os.replace(tmp,path)
def main():
 p=argparse.ArgumentParser();p.add_argument("--analysis",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 trade=list(csv.DictReader(open(a.analysis/"outer_event_tradeoff_descriptive.csv")));rel=list(csv.DictReader(open(a.analysis/"reliability.csv")));outputs=[]
 fig,axes=plt.subplots(1,3,figsize=(15,4.5))
 for ax,h in zip(axes,(1,3,5)):
  for group in GROUPS:
   for i,seed in enumerate(SEEDS):
    rows=sorted((r for r in trade if int(r["horizon"])==h and r["group"]==group and r["rule"]=="raw" and int(r["seed"])==seed),key=lambda r:float(r["trial_false_alarm_rate"]))
    ax.plot([float(r["trial_false_alarm_rate"]) for r in rows],[float(r["event_recall"]) for r in rows],color=COLORS[group],ls=STYLES[i],alpha=.85,label=f"{group} s{str(seed)[-2:]}" if h==1 else None)
  ax.set(title=f"H{h} outer descriptive",xlabel="trial-any false alarm",ylabel="new-start event recall",xlim=(0,1),ylim=(0,1));ax.grid(alpha=.25)
 axes[0].legend(fontsize=6,ncol=2);fig.tight_layout();path=a.output/"event_recall_vs_trial_false_alarm_all_seeds.png";fig.savefig(path,dpi=180);plt.close(fig);outputs.append(path)
 for role in ("calibration","outer"):
  fig,axes=plt.subplots(1,3,figsize=(14,4))
  for ax,h in zip(axes,(1,3,5)):
   for method,color in (("selected_rule_raw","tab:blue"),("selected_rule_monotone_platt","tab:orange")):
    rows=[r for r in rel if r["role"]==role and int(r["horizon"])==h and r["method"]==method and r["n"]!="0"]
    x=[];y=[]
    for b in range(10):
     br=[r for r in rows if int(r["bin"])==b]
     if br:x.append(sum(float(r["mean_probability"])*int(r["n"]) for r in br)/sum(int(r["n"]) for r in br));y.append(sum(float(r["positive_fraction"])*int(r["n"]) for r in br)/sum(int(r["n"]) for r in br))
    ax.plot(x,y,"o-",color=color,label=method);ax.plot([0,1],[0,1],"k:",alpha=.5);ax.set(title=f"H{h} {role}",xlabel="mean probability",ylabel="positive fraction",xlim=(0,1),ylim=(0,1));ax.grid(alpha=.25)
  axes[-1].legend(fontsize=7);fig.tight_layout();path=a.output/f"reliability_{role}.png";fig.savefig(path,dpi=180);plt.close(fig);outputs.append(path)
 atomic(a.output/"RENDER_PROVENANCE.json",{"schema":"round7_display_only_render_v1","status":"complete","timing":"after formal metrics; display only; no metric, threshold, or scientific selection changed","sources":{str((a.analysis/"outer_event_tradeoff_descriptive.csv").resolve()):sha(a.analysis/"outer_event_tradeoff_descriptive.csv"),str((a.analysis/"reliability.csv").resolve()):sha(a.analysis/"reliability.csv"),str(Path(__file__).resolve()):sha(Path(__file__).resolve())},"outputs":{str(x.resolve()):sha(x) for x in outputs}})
if __name__=="__main__":main()
