#!/usr/bin/env python3
"""Freeze hashes of the six original formal runs before recovery-code repair."""
import argparse, hashlib, json, os
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); p.add_argument("--source",type=Path,required=True); a=p.parse_args()
    files=[]
    for pattern in ("runs/*/*/best.pt","runs/*/*/latest.pt","runs/*/*/metrics.json","runs/*/*/history.json",
                    "config.json","baselines.json","train_only_standardizers.npz","summary.json","cache_audit.json"):
        files.extend(a.root.glob(pattern))
    payload={"format":"normalflow_pre_recovery_patch_v1","status":"complete","source":str(a.source.resolve()),
             "source_sha256":sha(a.source),"artifacts":{str(x.relative_to(a.root)):sha(x) for x in sorted(set(files))}}
    if len(list(a.root.glob("runs/*/*/metrics.json")))!=6: raise SystemExit("expected six formal runs")
    target=a.root/"formal_artifacts_pre_recovery_patch.json"; tmp=target.with_name(target.name+f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); os.replace(tmp,target); print(json.dumps({"status":"complete","artifacts":len(payload["artifacts"]),"source_sha256":payload["source_sha256"]},indent=2))
if __name__=="__main__": main()
