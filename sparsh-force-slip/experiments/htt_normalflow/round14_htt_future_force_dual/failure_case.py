#!/usr/bin/env python3
"""Post-result descriptive failure case selected by a fixed unfavorable rule."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np,torch
import matplotlib.pyplot as plt
import future_train as ft
def main():
 p=argparse.ArgumentParser();p.add_argument("--evaluation",type=Path,required=True);p.add_argument("--force-support",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);rows=[];payloads={}
 for g in ft.GROUPS:
  for f in range(1,5):
   for s in (20260914,20260915,20260916):
    run=f"{g}_p{f}_s{s}";d=torch.load(a.evaluation/run/"predictions_validation.pt",map_location="cpu",weights_only=False);ids=np.array(d["episode_id"]);payloads[(g,f,s)]=d
    for ep in sorted(set(ids)):
     m=ids==ep;ne=float((d["predictions"]["neural"][m]-d["y"][m]).abs().mean());pe=float((d["predictions"]["predicted_current_persistence"][m]-d["y"][m]).abs().mean());rows.append({"group":g,"fold":f,"seed":s,"episode_id":ep,"endpoints":int(m.sum()),"neural_future_mae_n":ne,"persistence_future_mae_n":pe,"neural_minus_persistence_n":ne-pe})
 rows.sort(key=lambda r:(-r["neural_minus_persistence_n"],r["group"],r["fold"],r["seed"],r["episode_id"]));sel=rows[0];d=payloads[(sel["group"],sel["fold"],sel["seed"])];ids=np.array(d["episode_id"]);m=ids==sel["episode_id"];t=d["t"][m].numpy();gt=d["y"][m,2].numpy();ne=d["predictions"]["neural"][m,2].numpy();pe=d["predictions"]["predicted_current_persistence"][m,2].numpy()
 with (a.output/"failure_candidates.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 curve=[]
 for i,tt in enumerate(t):
  for j,axis in enumerate(("fx","fy","fz")):curve.append({"t":int(tt),"target_t_plus_10":float(gt[i,j]),"axis":axis,"neural":float(ne[i,j]),"predicted_current_persistence":float(pe[i,j])})
 with (a.output/"selected_failure_curve.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(curve[0]));w.writeheader();w.writerows(curve)
 fig,axs=plt.subplots(3,1,figsize=(10,7),sharex=True)
 for j,(ax,axis) in enumerate(zip(axs,("Fx","Fy","Fz"))):ax.plot(t,gt[:,j],label="GT(t+10)",lw=1.5);ax.plot(t,ne[:,j],label="Neural",lw=1);ax.plot(t,pe[:,j],label="Pred-current persistence",lw=1);ax.set_ylabel(axis+" (N)");ax.grid(alpha=.2)
 axs[0].legend(ncol=3,frameon=False);axs[-1].set_xlabel("Current endpoint t (frame)");fig.suptitle(f"Post-hoc fixed worst gap: {sel['group']} p{sel['fold']} s{sel['seed']} {sel['episode_id']}");fig.tight_layout();fig.savefig(a.output/"selected_failure_curve.svg");plt.close(fig)
 # Deterministically show the endpoint with the largest framewise gap inside the selected episode.
 frame_gap=np.abs(ne-gt).mean(1)-np.abs(pe-gt).mean(1);ii=int(np.argmax(frame_gap));source=json.load(open(a.force_support/f"fold_p{sel['fold']}.json"));entry=next(e for e in source["entries"] if e["episode_id"]==sel["episode_id"]);src=Path(entry["source_path"])
 with np.load(src,allow_pickle=False) as raw:images=np.asarray(raw["tactile_img"]);current=images[int(t[ii])];future=images[int(t[ii])+10]
 fig,axs=plt.subplots(1,2,figsize=(9,4));axs[0].imshow(current);axs[0].set_title(f"Current tactile frame t={int(t[ii])}");axs[1].imshow(future);axs[1].set_title(f"Source frame t+10={int(t[ii])+10}")
 for ax in axs:ax.axis("off")
 fig.suptitle(sel["episode_id"]);fig.tight_layout();fig.savefig(a.output/"selected_failure_source_frames.png",dpi=160);plt.close(fig)
 receipt={"schema":"round14_failure_case_v2","status":"complete","selection_timing":"post-result descriptive; not preregistered","fixed_rule":"Across every validation run and complete episode, select the largest episode mean(neural absolute future error - predicted-current persistence absolute future error), averaged over all endpoints, axes, and horizons; lexical identity tie-break. Within it, source frames use the earliest argmax framewise gap.","candidate_count":len(rows),"selected":{**sel,"source_endpoint_t":int(t[ii]),"source_future_t":int(t[ii])+10,"frame_gap_n":float(frame_gap[ii])},"source_prediction":"evaluation/<group>_p<fold>_s<seed>/predictions_validation.pt","source_npz":str(src),"source_npz_sha256":ft.sha(src),"outputs":{"candidates_sha256":ft.sha(a.output/"failure_candidates.csv"),"curve_csv_sha256":ft.sha(a.output/"selected_failure_curve.csv"),"curve_svg_sha256":ft.sha(a.output/"selected_failure_curve.svg"),"source_frames_png_sha256":ft.sha(a.output/"selected_failure_source_frames.png")},"interpretation":"One deterministic worst-gap case illustrates failure shape and is not population evidence or a selection target."};ft.atomic_json(receipt,a.output/"FAILURE_CASE_AUDIT.json");print(json.dumps(receipt))
if __name__=="__main__":main()
