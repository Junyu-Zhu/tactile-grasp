#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
GROUPS=("A_visual","B_force","C_force_delta","D_visual_delta");SEEDS=(20260914,20260915,20260916);COLORS=dict(zip(GROUPS,("tab:blue","tab:orange","tab:green","tab:red")));STYLES=("-","--",":")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--curves",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);rows=[];sources={}
 for g in GROUPS:
  path=a.curves/f"curves_{g}.csv.gz";sources[str(path.resolve())]=sha(path)
  with gzip.open(path,"rt") as f:rows+=list(csv.DictReader(f))
 outputs=[]
 for kind in ("roc","pr"):
  fig,axes=plt.subplots(1,3,figsize=(15,4.5))
  for ax,h in zip(axes,(1,3,5)):
   for g in GROUPS:
    for i,s in enumerate(SEEDS):
     rr=[r for r in rows if r["group"]==g and int(r["seed"])==s and int(r["horizon"])==h and r["population"]=="primary" and r["method"]=="selected_rule"]
     x=[float(r["fpr"] if kind=="roc" else r["recall"]) for r in rr];y=[float(r["recall"] if kind=="roc" else r["precision"]) for r in rr]
     ax.plot(x,y,color=COLORS[g],ls=STYLES[i],label=f"{g} s{str(s)[-2:]}" if h==1 else None)
   ax.set(title=f"H{h} primary outer",xlabel="frame FPR" if kind=="roc" else "frame recall",ylabel="frame recall" if kind=="roc" else "precision",xlim=(0,1),ylim=(0,1));ax.grid(alpha=.25)
  axes[0].legend(fontsize=6,ncol=2);fig.tight_layout();path=a.output/f"frame_{kind}_all_runs.png";fig.savefig(path,dpi=180);plt.close(fig);outputs.append(path)
 provenance={"schema":"round7_frame_curve_display_v1","status":"complete","display_only":True,"sources":sources,"outputs":{str(x.resolve()):sha(x) for x in outputs}};path=a.output/"FRAME_CURVE_RENDER_PROVENANCE.json";tmp=path.with_name(path.name+f".tmp.{os.getpid()}");tmp.write_text(json.dumps(provenance,indent=2,sort_keys=True)+"\n");os.replace(tmp,path)
if __name__=="__main__":main()
