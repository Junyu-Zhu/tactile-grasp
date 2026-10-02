#!/usr/bin/env python3
"""Final integrity audit for the NormalFlow work package."""
import argparse, hashlib, json, os
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); p.add_argument("--code",type=Path,required=True); a=p.parse_args()
    frozen=json.loads((a.root/"formal_artifacts_pre_recovery_patch.json").read_text()); errors=[]
    for rel,expected in frozen["artifacts"].items():
        path=a.root/rel
        if not path.is_file() or sha(path)!=expected: errors.append(f"frozen artifact changed: {rel}")
    run_metrics=sorted((a.root/"runs").glob("*/*/metrics.json"))
    if len(run_metrics)!=6 or any(json.loads(x.read_text()).get("status")!="pass" for x in run_metrics): errors.append("six formal run metrics not complete")
    proofs={name:json.loads((a.root/name).read_text()) for name in ("recovery_proof.json","same_next_step_C.json","same_next_step_D.json")}
    if any(x.get("status")!="pass" for x in proofs.values()): errors.append("recovery/continuation proof failed")
    analysis=json.loads((a.root/"analysis.json").read_text())
    if analysis.get("status")!="complete" or analysis.get("runs")!=6: errors.append("analysis incomplete")
    checkpoint_map=[]
    for path in sorted((a.root/"runs").glob("*/*/best.pt")):
        checkpoint_map.append({"variant":path.parts[-3],"seed":int(path.parts[-2]),"path":str(path),"sha256":sha(path)})
    code_files=sorted(x for x in a.code.glob("*.py")) + [a.code/"PROTOCOL.md"]
    payload={"status":"pass" if not errors else "fail","errors":errors,"formal_runs":len(run_metrics),
             "original_formal_source_sha256":frozen["source_sha256"],"current_recovery_source_sha256":sha(a.code/"run_normalflow.py"),
             "original_artifacts_verified":len(frozen["artifacts"]),"proofs":{k:sha(a.root/k) for k in proofs},
             "code_sha256":{str(x):sha(x) for x in code_files},"checkpoints":checkpoint_map,
             "scope":{"train_objects":8,"validation_objects":["table","bead"],"test_objects_loaded":[],"neural_runs":6}}
    target=a.root/"FINAL_AUDIT.json"; tmp=target.with_name(target.name+f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); os.replace(tmp,target)
    checkpoints=a.root/"CHECKPOINTS.json"; checkpoints_tmp=checkpoints.with_name(checkpoints.name+f".tmp.{os.getpid()}")
    checkpoints_tmp.write_text(json.dumps(checkpoint_map,indent=2)+"\n"); os.replace(checkpoints_tmp,checkpoints)
    print(json.dumps({"status":payload["status"],"errors":errors,"formal_runs":len(run_metrics)},indent=2)); raise SystemExit(bool(errors))
if __name__=="__main__": main()
