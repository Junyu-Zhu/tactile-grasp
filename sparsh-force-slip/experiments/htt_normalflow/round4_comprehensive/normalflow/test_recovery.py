#!/usr/bin/env python3
"""Prove exact tiny interrupted/resumed equivalence for C and D on CPU."""
import copy, json, os, tempfile
from pathlib import Path

import numpy as np
import torch

import run_normalflow as nf


def arrays(seed=7, n=48):
    rng=np.random.default_rng(seed); history=rng.normal(0,0.2,(n,4,768)).astype(np.float32)
    delta=rng.normal(0,0.01,(n,3,768)).astype(np.float32)
    motion=rng.normal(0,0.01,(n,3,6)).astype(np.float32)
    return {"history":history,"target":history[:,-1:, :]+delta,"motion":motion,
            "episode":np.asarray([f"e{i//8}" for i in range(n)]),"object":np.asarray(["o"]*n),"anchor":np.arange(n)}


def tensor_equal(a,b):
    if torch.is_tensor(a): return torch.equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(tensor_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(tensor_equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    torch.use_deterministic_algorithms(True); torch.set_num_threads(1)
    data=arrays(); stats=nf.standardizers(data); cfg=nf.Config(batch_size=16,max_epochs=2,patience=5)
    results={}
    with tempfile.TemporaryDirectory(prefix="nf_recovery_") as temp:
        root=Path(temp)
        for variant in ("C","D"):
            full=root/f"full_{variant}"; resumed=root/f"resumed_{variant}"
            nf.train_one(data,data,stats,variant,20260914,full,cfg,"cpu",2)
            nf.train_one(data,data,stats,variant,20260914,resumed,cfg,"cpu",1)
            nf.train_one(data,data,stats,variant,20260914,resumed,cfg,"cpu",2)
            left=torch.load(full/"latest.pt",map_location="cpu",weights_only=False)
            right=torch.load(resumed/"latest.pt",map_location="cpu",weights_only=False)
            results[variant]={"model_exact":tensor_equal(left["model"],right["model"]),
                              "optimizer_exact":tensor_equal(left["optimizer"],right["optimizer"]),
                              "history_exact":left["history"]==right["history"],
                              "epoch_equal":left["epoch"]==right["epoch"]==2,
                              "temporary_files_remaining":list(map(str,resumed.glob("*.tmp.*")))}
            results[variant]["pass"]=all(results[variant][k] for k in ("model_exact","optimizer_exact","history_exact","epoch_equal")) and not results[variant]["temporary_files_remaining"]
    payload={"status":"pass" if all(x["pass"] for x in results.values()) else "fail","device":"cpu",
             "comparison":"uninterrupted 2 epochs vs interrupted after epoch 1 then resumed to epoch 2","variants":results}
    target=Path(os.environ.get("NF_RECOVERY_PROOF","recovery_proof.json")); target.parent.mkdir(parents=True,exist_ok=True)
    tmp=target.with_name(target.name+f".tmp.{os.getpid()}"); tmp.write_text(json.dumps(payload,indent=2)+"\n"); os.replace(tmp,target)
    print(json.dumps(payload,indent=2)); raise SystemExit(payload["status"]!="pass")

if __name__=="__main__": main()
