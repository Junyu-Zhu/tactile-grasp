#!/usr/bin/env python3
"""Export E3 predictions from the actual fine-tuned visual suffix checkpoint."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, random, sys, tempfile
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
R22 = HERE.parent / "round22_921_g1_joint_frozen"

def load_train():
    spec = importlib.util.spec_from_file_location("r23_e3_train", R22 / "train_e3.py")
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
    return mod

tr = load_train()

def endpoint_hash(role):
    h = hashlib.sha256()
    for e, t in zip(role["episode_id"], role["t"].tolist()):
        h.update(str(e).encode()); h.update(b"\0"); h.update(str(int(t)).encode()); h.update(b"\n")
    return h.hexdigest()

def atomic_npz(path, arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".npz"); os.close(fd)
    try: np.savez_compressed(tmp, **arrays); os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def deduplicated_visual(model,data,role,device,chunk=128):
    """Encode every unique base time once, then reassemble locked nine-step histories."""
    r=data.d["roles"][role];out=torch.empty((len(r["t"]),9,192),dtype=torch.float32)
    with torch.inference_mode():
      for eid in sorted(set(r["episode_id"])):
        endpoint_ids=np.asarray([i for i,e in enumerate(r["episode_id"]) if e==eid],dtype=np.int64);times=r["t"][endpoint_ids].numpy().astype(np.int64);base=np.unique(np.concatenate([np.arange(t-8,t+1) for t in times]));lookup={int(t):i for i,t in enumerate(base.tolist())}
        if eid not in data.mm:data.mm[eid]=np.load(data.paths[eid],mmap_mode="r",allow_pickle=False)
        encoded=[]
        for ids in np.array_split(base,max(1,int(np.ceil(len(base)/chunk)))):
          z=torch.from_numpy(np.array(data.mm[eid][ids],copy=True)).to(device);encoded.append(model.visual(z).cpu())
        ev=torch.cat(encoded)
        for endpoint_id,t in zip(endpoint_ids.tolist(),times.tolist()):out[endpoint_id]=ev[torch.tensor([lookup[x] for x in range(int(t)-8,int(t)+1)])]
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument("--inventory",type=Path,required=True);p.add_argument("--run",required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cuda:0");p.add_argument("--smoke-one-batch",action="store_true");a=p.parse_args()
    inv=json.loads(a.inventory.read_text()); item=next((x for x in inv["runs"] if x["run"]==a.run),None)
    if item is None: raise ValueError("run absent from inventory")
    run=Path(item["output"]); summary=json.loads((run/"summary.json").read_text()); commit=json.loads((run/"COMMIT.json").read_text()); best=run/commit["best"]["path"]
    if summary.get("status")!="complete" or tr.sha(best)!=commit["best"]["sha256"]: raise ValueError("unaccepted best commit")
    ck=torch.load(best,map_location="cpu",weights_only=False); ident=ck["identity"]
    expected={"group":item["group"],"fold":item["fold"],"seed":item["seed"],"data_sha256":item["data_sha256"],"prefix_index_sha256":item["prefix_index_sha256"],"source_sha256":item["source_sha256"],"visual_checkpoint_sha256":item["visual_checkpoint_sha256"],"source_sha256_code":item["train_source_sha256"],"formal":True}
    bad={k:(ident.get(k),v) for k,v in expected.items() if ident.get(k)!=v}
    if bad: raise ValueError(f"identity mismatch: {bad}")
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    random.seed(item["seed"]);np.random.seed(item["seed"]);torch.manual_seed(item["seed"]);torch.cuda.manual_seed_all(item["seed"])
    data=tr.Data(item["data"],item["prefix_index"]); model=tr.Model(item["group"],item["seed"],item["source"],item["visual_checkpoint"]);model.load_state_dict(ck["model"],strict=True);model.to(a.device).eval();norm=ck["normalizer"]
    if a.smoke_one_batch:
      ids=torch.arange(min(tr.BATCH,len(data.d["roles"]["validation"]["t"])));z,f,_=data.batch("validation",ids);fn=((f-norm["force_mean"])/norm["force_std"]).to(a.device);z=z.to(a.device)
      with torch.inference_mode():direct=torch.sigmoid(model(z,fn));v=deduplicated_visual(model,data,"validation",a.device)[:len(ids)].to(a.device);x=torch.cat((v,fn),-1) if item["group"] in tr.FORCE_INPUT_GROUPS else torch.cat((v,torch.zeros_like(fn)),-1);via=torch.sigmoid(model.head.components(x)["logit"])
      err=float((direct-via).abs().max());result={"schema":"round23_e3_prediction_one_batch_smoke_v2","status":"pass" if err<=1e-5 else "fail","run":a.run,"batch":len(ids),"direct_vs_deduplicated_max_abs":err,"base_visual_chunk":128,"best_commit_sha256":commit["best"]["sha256"],"test_consumed":False};print(json.dumps(result));raise SystemExit(0 if err<=1e-5 else 1)
    arrays={}; counts={}; groups_with_force=set(tr.FORCE_INPUT_GROUPS)
    with torch.inference_mode():
      for role in tr.ROLES:
        normal=[];fmean=[];flag=[];foff=[];aux=[];aux_gt=[];physical_error=[];gammas=[];betas=[];ratios=[];roledata=data.d["roles"][role]
        allv=deduplicated_visual(model,data,role,a.device);allf=roledata["x"][:,:,192:195].clone();allgt=roledata["y_current"].clone()
        for ids in torch.arange(len(roledata["t"])).split(tr.BATCH):
          v=allv[ids].to(a.device);f=allf[ids];gt=allgt[ids];fn=((f-norm["force_mean"])/norm["force_std"]).to(a.device);zero=torch.zeros_like(fn); lag=torch.zeros_like(fn);lag[:,1:]=fn[:,:-1];x=torch.cat((v,fn),-1) if item["group"] in groups_with_force else torch.cat((v,zero),-1);c=model.head.components(x);c["visual"]=v;c["aux_force"]=model.aux(v[:,-1]) if model.aux is not None else None;normal.append(torch.sigmoid(c["logit"]).cpu())
          if item["group"] in groups_with_force:
            fmean.append(torch.sigmoid(model.head.components(torch.cat((v,zero),-1))["logit"]).cpu());flag.append(torch.sigmoid(model.head.components(torch.cat((v,lag),-1))["logit"]).cpu())
          else:
            # A/B inference contract zeroes force before the head; prove the direct same-v path once.
            direct=torch.sigmoid(model.head.components(torch.cat((v,zero),-1))["logit"]).cpu();base=torch.sigmoid(c["logit"]).cpu()
            if not torch.equal(direct,base):raise ValueError("non-force group isolation failure")
            fmean.append(base);flag.append(base)
          if item["group"] in groups_with_force:
            base=model.head.visual_ln(v);foff.append(torch.sigmoid(model.head.risk(model.head.gru(base)[0][:,-1]).squeeze(-1)).cpu());gammas.append(c["gamma"].cpu());betas.append(c["beta"].cpu());ratios.append((((1+c["gamma"])*base+c["beta"]).norm(dim=-1)/(base.norm(dim=-1)+1e-12)).cpu())
          if c["aux_force"] is not None:
            pred=c["aux_force"].cpu()*norm["aux_std"]+norm["aux_mean"];aux.append(pred);aux_gt.append(gt)
          physical_error.append((f[:,-1]-gt).abs().mean(1))
        arrays[f"{role}_score"]=torch.cat(normal).numpy();arrays[f"{role}_force_mean_score"]=torch.cat(fmean).numpy();arrays[f"{role}_force_lag1_score"]=torch.cat(flag).numpy();arrays[f"{role}_physical_force_mae_n"]=torch.cat(physical_error).numpy()
        if foff: arrays[f"{role}_film_off_score"]=torch.cat(foff).numpy()
        if gammas:
          g=torch.cat(gammas).numpy();b=torch.cat(betas).numpy();ratio=torch.cat(ratios).numpy();arrays[f"{role}_film_gamma"]=g;arrays[f"{role}_film_beta"]=b;arrays[f"{role}_film_norm_ratio"]=ratio
        if aux: arrays[f"{role}_aux_force_pred_n"]=torch.cat(aux).numpy();arrays[f"{role}_aux_force_gt_n"]=torch.cat(aux_gt).numpy()
        counts[role]={"endpoints":len(roledata["t"]),"endpoint_sha256":endpoint_hash(roledata)}
    # This proof makes using the original frozen visual cache detectable: predictions bind the
    # fine-tuned best commit and the loaded blocks10/11 tensor bytes.
    suffix={k:v for k,v in ck["model"].items() if k.startswith("blocks.")}
    suffix_sha=tr.state_hash(suffix)
    atomic_npz(a.output/"PREDICTIONS.npz",arrays)
    smask=data.d["roles"]["selection"]["stage"].eq(0)|data.d["roles"]["selection"]["stage"].eq(2);replayed=tr.low_fpr_auc(data.d["roles"]["selection"]["stage"][smask],torch.from_numpy(arrays["selection_score"])[smask]);selection_abs_error=abs(replayed-float(summary["best_metric"]))
    if selection_abs_error>2e-6:raise ValueError(f"selection replay mismatch {selection_abs_error}")
    out={"schema":"round23_e3_prediction_v1","status":"complete","run":a.run,"group":item["group"],"fold":item["fold"],"seed":item["seed"],"best_commit":str(best),"best_commit_sha256":commit["best"]["sha256"],"fine_tuned_suffix_state_sha256":suffix_sha,"checkpoint_model_loaded_strict":True,"selection_pAUC_replayed":replayed,"selection_best_metric":float(summary["best_metric"]),"selection_replay_abs_error":selection_abs_error,"prediction_path":str(a.output/"PREDICTIONS.npz"),"prediction_sha256":tr.sha(a.output/"PREDICTIONS.npz"),"counts":counts,"source_hashes":{"inventory":tr.sha(a.inventory),"predictor":tr.sha(__file__),"trainer":tr.sha(R22/"train_e3.py")},"test_consumed":False}
    tr.atomic_json(a.output/"SUMMARY.json",out);print(json.dumps(out,sort_keys=True))

if __name__=="__main__": main()
