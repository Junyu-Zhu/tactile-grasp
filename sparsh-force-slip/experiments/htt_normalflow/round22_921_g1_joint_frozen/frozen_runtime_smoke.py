#!/usr/bin/env python3
"""Bounded non-formal runtime/recovery smoke for frozen G1 entrypoints."""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path
import torch
import e3_recovery_smoke as R
def run(cmd,log,env):
 with Path(log).open("wb") as f:return subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env).returncode
def main():
 p=argparse.ArgumentParser();p.add_argument("--current-data",type=Path,required=True);p.add_argument("--future-data",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--python",default=sys.executable);p.add_argument("--device",default="cuda:0");a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);here=Path(__file__).resolve().parent;logs=a.output/"logs";logs.mkdir();env={**os.environ,"OMP_NUM_THREADS":"4","MKL_NUM_THREADS":"4","XFORMERS_DISABLED":"1"};gate=here/"G1_DISPATCH_GATE.json";protocol=here/"FROZEN_PROTOCOL_LOCK.json";results={};checks={}
 frozen_base=[a.python,str(here/"train_frozen.py"),"--data",str(a.current_data),"--fold","1","--seed","20260914","--device",a.device,"--max-epochs","1","--patience","10"]
 for g in ("V","C","M","MB"):
  out=a.output/"frozen"/g;cmd=frozen_base+["--group",g,"--output",str(out)];rc=run(cmd,logs/f"frozen_{g}.log",env);s=json.loads((out/"summary.json").read_text()) if rc==0 else {};results[f"frozen_{g}"]=s;checks[f"frozen_{g}_one_epoch_complete"]=rc==0 and s.get("status")=="complete" and s.get("epochs")==1
 c=a.output/"frozen_recovery"/"continuous";r=a.output/"frozen_recovery"/"resumed";base=[a.python,str(here/"train_frozen.py"),"--data",str(a.current_data),"--group","V","--fold","1","--seed","20260914","--device",a.device,"--max-epochs","2","--patience","10"]
 checks["frozen_recovery_continuous_complete"]=run(base+["--output",str(c)],logs/"frozen_recovery_continuous.log",env)==0
 checks["frozen_recovery_commit_created"]=run(base+["--output",str(r),"--interrupt-after-commit","0"],logs/"frozen_recovery_interrupt.log",env)==0 and json.loads((r/"COMMIT.json").read_text())["epoch"]==0
 checks["frozen_recovery_resume_complete"]=run(base+["--output",str(r)],logs/"frozen_recovery_resume.log",env)==0
 ccj=json.loads((c/"COMMIT.json").read_text());rcj=json.loads((r/"COMMIT.json").read_text());cc=torch.load(c/ccj["latest"]["path"],map_location="cpu",weights_only=False);rc=torch.load(r/rcj["latest"]["path"],map_location="cpu",weights_only=False);diff=R.nested_equal(cc,rc);checks["frozen_recovery_exact_state"]=not diff
 f1_base=[a.python,str(here/"train_f1.py"),"--data",str(a.future_data),"--fold","1","--seed","20260914","--device",a.device,"--max-epochs","1","--patience","10"]
 for g in ("K-V","K-F","K-VF"):
  out=a.output/"f1"/g;cmd=f1_base+["--group",g,"--output",str(out)];rcod=run(cmd,logs/f"f1_{g}.log",env);s=json.loads((out/"summary.json").read_text()) if rcod==0 else {};results[f"f1_{g}"]=s;checks[f"f1_{g}_one_epoch_complete"]=rcod==0 and s.get("status")=="complete" and s.get("epochs")==1
 ev=a.output/"f2_eval";ecmd=[a.python,str(here/"evaluate_f1_f2.py"),"--data",str(a.future_data),"--run",str(a.output/"f1"/"K-VF"),"--output",str(ev),"--device",a.device];erc=run(ecmd,logs/"f2_eval.log",env);ed=json.loads((ev/"evaluation.json").read_text()) if erc==0 else {};checks["f2_fixed_half_evaluation_entrypoint_complete"]=erc==0 and ed.get("status")=="entrypoint_smoke_complete" and ed.get("f2_coefficient")==.5 and ed.get("f2_selected_without_validation") is True
 for name,script,data,group in (("frozen",here/"train_frozen.py",a.current_data,"V"),("f1",here/"train_f1.py",a.future_data,"K-V")):
  out=a.output/f"MUST_NOT_FORMAL_{name}";cmd=[a.python,str(script),"--data",str(data),"--output",str(out),"--group",group,"--fold","1","--seed","20260914","--device",a.device,"--formal","--authorization",str(gate),"--protocol",str(protocol)];checks[f"{name}_formal_gate_blocks_before_output"]=run(cmd,logs/f"formal_block_{name}.log",env)!=0 and not out.exists()
 receipt={"schema":"round22_frozen_runtime_smoke_v1","status":"pass" if all(checks.values()) else "fail","formal_training_started":False,"checks":checks,"frozen_recovery_differences":diff,"results":results,"f2_evaluation":ed};R.dump(a.output/"RECEIPT.json",receipt);print(json.dumps(receipt));raise SystemExit(0 if receipt["status"]=="pass" else 1)
if __name__=="__main__":main()
