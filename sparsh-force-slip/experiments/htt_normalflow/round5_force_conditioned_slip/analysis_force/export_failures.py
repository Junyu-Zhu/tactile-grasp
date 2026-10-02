#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from analysis_common import atomic_json, protocol, sha256, verify_receipt_inventory


def select(rows, stage, reverse, limit):
    candidates=sorted((r for r in rows if int(r["stage"])==stage),key=lambda r:float(r["probability"]),reverse=reverse)
    selected=[];seen=set()
    for row in candidates:
        if row["episode_id"] in seen:continue
        selected.append(row);seen.add(row["episode_id"])
        if len(selected)==limit:break
    return selected


def main():
    p=argparse.ArgumentParser();p.add_argument("--predictions",type=Path,required=True);p.add_argument("--training-summary",type=Path,required=True);p.add_argument("--inventory",type=Path,required=True);p.add_argument("--contract",type=Path,required=True);p.add_argument("--fold",required=True);p.add_argument("--seed",type=int,required=True);p.add_argument("--variant",choices=("V","F-old","F-adapt"),required=True);p.add_argument("--role",choices=("calibration","validation"),required=True);p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    cfg=protocol();contract=json.loads(args.contract.read_text());entries={e["episode_id"]:e for e in contract["entries"] if e["task"]=="slip"}
    summary=json.loads(args.training_summary.read_text());record=summary.get("predictions",{}).get(args.role,{})
    if (summary.get("status")!="complete" or summary.get("smoke") is not False or summary.get("variant")!=args.variant or summary.get("fold")!=args.fold or int(summary.get("seed",-1))!=args.seed
            or Path(record.get("path","")).resolve()!=args.predictions.resolve() or record.get("sha256")!=sha256(args.predictions.resolve())):
        raise ValueError("failure export requires a matching formal prediction summary")
    receipt_proof=verify_receipt_inventory(args.training_summary,args.inventory)
    rows=[]
    with args.predictions.open() as stream:
        for raw in csv.DictReader(stream):
            episode=raw.get("episode_id",raw.get("episode"));frame=int(raw.get("frame",raw.get("t",-1)));stage=int(raw["stage"]);prob=float(raw.get("probability",raw.get("p_slip","nan")))
            if episode not in entries or entries[episode]["roles_by_fold"].get(args.fold)!=args.role:raise ValueError("prediction role/task mismatch")
            if frame<cfg["slip_min_frame"]:continue
            if not np.isfinite(prob) or not 0<=prob<=1:raise ValueError("invalid probability")
            rows.append({"episode_id":episode,"frame":frame,"stage":stage,"probability":prob})
    chosen=[("false_positive",r) for r in select(rows,0,True,cfg["failure_export"]["maximum_per_kind"])]
    chosen += [("false_negative",r) for r in select(rows,2,False,cfg["failure_export"]["maximum_per_kind"])]
    output=args.output.resolve();image_dir=output/"images";image_dir.mkdir(parents=True,exist_ok=True);index=[]
    opened={}
    for kind,row in chosen:
        entry=entries[row["episode_id"]];source=Path(sorted(p for p in entry["source_files"] if "/processed/" in p)[0])
        if sha256(source)!=entry["source_files"][str(source)]:raise ValueError("source image hash mismatch")
        if source not in opened:opened[source]=np.load(source,allow_pickle=False)
        frame=np.asarray(opened[source]["tactile_img"][row["frame"]],dtype=np.uint8);image=Image.fromarray(frame).convert("RGB")
        canvas=Image.new("RGB",(image.width,image.height+32),"white");canvas.paste(image,(0,32));draw=ImageDraw.Draw(canvas);draw.text((5,8),f"{kind} p={row['probability']:.4f} t={row['frame']}",fill="black")
        key=f"{kind}_{len([x for x in index if x['kind']==kind]):02d}_{row['episode_id'].replace('/','__')}_t{row['frame']}.png";path=image_dir/key
        temporary=path.with_suffix(".png.tmp");canvas.save(temporary,format="PNG");os.replace(temporary,path)
        index.append({**row,"kind":kind,"image_path":str(path),"image_sha256":sha256(path),"source_path":str(source),"source_sha256":sha256(source),"selection":"at most one frame per episode"})
    for value in opened.values():value.close()
    payload={"status":"complete","format":"round5_failure_images_v1","fold":args.fold,"seed":args.seed,"variant":args.variant,"role":args.role,"prediction_sha256":sha256(args.predictions.resolve()),"training_summary_sha256":sha256(args.training_summary.resolve()),**receipt_proof,"contract_sha256":sha256(args.contract.resolve()),"selection_rules":cfg["failure_export"],"interpretation":"single-frame high-confidence error examples; they are not asserted to be the same frames as representative continuous-alarm probability curves","count":len(index),"entries":index}
    atomic_json(output/"manifest.json",payload);print(json.dumps({"status":"complete","count":len(index)},indent=2))


if __name__=="__main__":main()
