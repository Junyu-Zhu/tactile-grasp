#!/usr/bin/env python3
"""Build one immutable FP32 cache at the output of MAE block 9 (0-based)."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, tempfile
from pathlib import Path
import numpy as np
os.environ.setdefault("XFORMERS_DISABLED", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
sys.path[:0]=[str(HERE.parent),str(REPO/"scripts")]
import adapters
import phase2_b_multitask as p2

def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def atomic_json(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(dir=p.parent,prefix=p.name+".",suffix=".tmp");os.close(fd)
 try:Path(t).write_text(json.dumps(x,indent=2,sort_keys=True)+"\n");os.replace(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)
def atomic_npy(p,x):
 p=Path(p);fd,t=tempfile.mkstemp(dir=p.parent,prefix=p.name+".",suffix=".tmp");os.close(fd)
 try:
  with open(t,"wb") as f:np.save(f,x,allow_pickle=False)
  os.replace(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)
def key(e):return hashlib.sha256(e.encode()).hexdigest()[:20]
def prefix(encoder,x):
 x=encoder.prepare_tokens_with_masks(x)
 for block in encoder.blocks[:10]:x=block(x)
 return x
def suffix(encoder,x):
 for block in encoder.blocks[10:]:x=block(x)
 return encoder.norm(x)[:,encoder.num_register_tokens:]

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--source",type=Path,required=True);ap.add_argument("--split",type=Path,required=True);ap.add_argument("--prepared",type=Path,nargs="+",required=True);ap.add_argument("--output",type=Path,required=True);ap.add_argument("--device",default="cuda:0");ap.add_argument("--batch",type=int,default=32);ap.add_argument("--limit",type=int);a=ap.parse_args()
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
 split=json.loads(a.split.read_text());rows={r["id"]:r for r in split["episodes"]}
 ids=set()
 for pp in a.prepared:
  d=torch.load(pp,map_location="cpu",weights_only=False)
  if "test" in d["roles"]:raise ValueError("test role forbidden")
  for role in ("fit","selection","calibration","validation"):ids.update(d["roles"][role]["episode_id"])
 ids=sorted(ids)[:a.limit]
 model,_=p2.load_b_checkpoint(a.source,torch.device(a.device));enc=model.encoder.eval().requires_grad_(False)
 a.output.mkdir(parents=True,exist_ok=True);(a.output/"episodes").mkdir(exist_ok=True)
 entries=[];parity=[]
 for j,eid in enumerate(ids,1):
  row=rows[eid];ep=adapters.load_htt(row["path"]);out=a.output/"episodes"/(key(eid)+".block09.npy");meta=out.with_suffix(".json")
  expected={"episode_id":eid,"source_path":row["path"],"frames":len(ep.images),"source_files":row["source_files"],"source_checkpoint_sha256":sha(a.source),"split_sha256":sha(a.split),"cut":"after_blocks_0_through_9_before_trainable_blocks_10_11","dtype":"float32","shape":[len(ep.images),301,768]}
  reusable=False
  if out.exists() and meta.exists():
   old=json.loads(meta.read_text());z=np.load(out,mmap_mode="r",allow_pickle=False);reusable=all(old.get(k)==v for k,v in expected.items()) and list(z.shape)==expected["shape"] and z.dtype==np.float32 and old.get("sha256")==sha(out)
  if not reusable:
   if meta.exists():raise RuntimeError(f"incompatible committed cache {eid}")
   out.unlink(missing_ok=True);chunks=[]
   for s in range(0,len(ep.images),a.batch):
    images=torch.stack([adapters.window(ep,t)["inputs"]["image"] for t in range(s,min(s+a.batch,len(ep.images)))]).to(a.device)
    with torch.inference_mode():z=prefix(enc,images)
    chunks.append(z.float().cpu().numpy())
   atomic_npy(out,np.concatenate(chunks))
  expected["sha256"]=sha(out);atomic_json(meta,expected)
  # Exact direct/split parity on first and last frames.
  mm=np.load(out,mmap_mode="r",allow_pickle=False)
  ix=sorted(set([0,len(ep.images)-1]))
  images=torch.stack([adapters.window(ep,t)["inputs"]["image"] for t in ix]).to(a.device)
  with torch.inference_mode():direct=enc(images);split_out=suffix(enc,torch.from_numpy(np.array(mm[ix],copy=True)).to(a.device))
  err=float((direct-split_out).abs().max().cpu());parity.append({"episode_id":eid,"frames":ix,"max_abs":err})
  if err>2e-5:raise RuntimeError(f"prefix parity {eid}: {err}")
  entries.append({**expected,"path":str(out)});print(json.dumps({"episode":j,"of":len(ids),"id":eid,"reused":reusable,"parity":err}),flush=True)
 result={"schema":"round19_block09_prefix_cache_v1","status":"complete" if a.limit is None else "smoke","entries":entries,"episodes":len(entries),"frames":sum(x["frames"] for x in entries),"bytes":sum(Path(x["path"]).stat().st_size for x in entries),"source":str(a.source),"source_sha256":sha(a.source),"split":str(a.split),"split_sha256":sha(a.split),"prepared":[{"path":str(p),"sha256":sha(p)} for p in a.prepared],"direct_suffix_parity":parity,"max_parity_abs":max(x["max_abs"] for x in parity)}
 atomic_json(a.output/"PREFIX_INDEX.json",result);print(json.dumps({k:result[k] for k in ("status","episodes","frames","bytes","max_parity_abs")},indent=2))
if __name__=="__main__":main()
