#!/usr/bin/env python3
"""Verify repaired convenience aliases against authoritative COMMIT payloads."""
import argparse,hashlib,json
from pathlib import Path
import torch
def state_hash(s):
 h=hashlib.sha256()
 for k,v in sorted(s.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();commit=json.loads((a.run/"COMMIT.json").read_text());checks={}
 for alias,key in (("latest.pth","latest"),("best.pth","best")):
  x=torch.load(a.run/alias,map_location="cpu",weights_only=False);y=torch.load(a.run/commit[key]["path"],map_location="cpu",weights_only=False);checks[alias]={"model_state_equal":state_hash(x["model"])==state_hash(y["model"]),"epoch_equal":x["epoch"]==y["epoch"],"identity_equal":x["identity"]==y["identity"]}
 summary=json.loads((a.run/"summary.json").read_text());out={"schema":"round22_e3_recovery_smoke_v2","status":"pass" if all(all(v.values()) for v in checks.values()) and summary["epochs"]==1 else "fail","commit_epoch":commit["epoch"],"summary_epochs":summary["epochs"],"no_extra_epoch":summary["epochs"]==1,"checks":checks,"note":"torch.save reserialization need not preserve file bytes; authoritative COMMIT file hashes were verified by train_e3 before loading, aliases are compared by model state, epoch and identity"};a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");print(json.dumps(out))
if __name__=="__main__":main()
