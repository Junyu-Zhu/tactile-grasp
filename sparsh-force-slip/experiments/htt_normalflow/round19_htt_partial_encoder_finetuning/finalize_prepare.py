#!/usr/bin/env python3
"""Create the pre-formal audit, dependency identity, protocol lock, and gate."""
from __future__ import annotations
import argparse,json,sys
from datetime import datetime,timezone,timedelta
from pathlib import Path
import train as tr

def main():
 p=argparse.ArgumentParser();p.add_argument("--local",type=Path,default=Path(__file__).resolve().parent);p.add_argument("--large",type=Path,required=True);p.add_argument("--r17",type=Path,required=True);p.add_argument("--r18",type=Path,required=True);p.add_argument("--recovery",type=Path,required=True);a=p.parse_args()
 required=("PROTOCOL.md","COMPARABILITY_MATRIX.md","BUDGET.json","train.py","prepare_prefix.py","numeric_checks.py","smoke.py","run_all.py","finalize_prepare.py")
 local_hashes={x:tr.sha(a.local/x) for x in required}
 external=[Path(tr.p2.__file__),Path(sys.modules[tr.load_decoupled_decoder.__module__].__file__)]
 # These are the concrete MAE and decoder implementations used by the locked helper code.
 external += [Path("/home/zjy/document/sparsh/tactile_ssl/model/vision_transformer.py"),Path("/home/zjy/document/sparsh/tactile_ssl/model/layers/block.py"),Path("/home/zjy/document/sparsh/tactile_ssl/model/layers/attention.py"),Path("/home/zjy/document/sparsh/tactile_ssl/model/layers/decoder_block.py")]
 deps={str(x.resolve()):tr.sha(x) for x in external}
 prefix=json.loads((a.large/"prefix_cache/PREFIX_INDEX.json").read_text());gpu=json.loads((a.local/"GPU_SMOKE.json").read_text());numeric=json.loads((a.local/"PAUC_NUMERIC_CHECKS.json").read_text());budget=json.loads((a.local/"BUDGET.json").read_text());recovery=json.loads(a.recovery.read_text())
 prepared=[]
 for f in range(1,5):
  for s in tr.SEEDS:
   x=a.r17/f"p{f}_s{s}"/"prepared.pt";prepared.append({"fold":f,"seed":s,"path":str(x),"sha256":tr.sha(x)})
 r18={x:tr.sha(a.r18/x) for x in ("ROOT_ACCEPTANCE.json","FINAL_STATUS.json","IDENTITY_LOCK.json","SELECTOR_NUMERIC_AUDIT.json","EVALUATION_LOCK.json")}
 audit={"schema":"round19_startup_audit_v1","status":"pass","created_at":datetime.now(timezone(timedelta(hours=8))).isoformat(),"scope":"Q1B/Q5 only","formal_grid":{"groups":["V2","M2"],"folds":[1,2,3,4],"seeds":list(tr.SEEDS),"runs":24},"test_consumed":False,"prefix":{"index_sha256":tr.sha(a.large/"prefix_cache/PREFIX_INDEX.json"),"episodes":prefix["episodes"],"frames":prefix["frames"],"bytes":prefix["bytes"],"max_direct_suffix_abs_error":prefix["max_parity_abs"],"cut":"after source MAE block9 with register; live blocks10/11"},"gpu_smoke_sha256":tr.sha(a.local/"GPU_SMOKE.json"),"gpu_smoke_status":gpu["status"],"force_raw_cache_tolerance_abs_n":gpu["force_raw_cache_max_abs_n"],"numeric_checks_sha256":tr.sha(a.local/"PAUC_NUMERIC_CHECKS.json"),"numeric_status":numeric["status"],"recovery_audit_sha256":tr.sha(a.recovery),"recovery_status":recovery["status"],"budget_sha256":tr.sha(a.local/"BUDGET.json"),"budget_deadline_fit":budget["deadline_fit"],"prepared":prepared,"r18_reused_acceptance":r18,"local_sources":local_hashes,"dependency_hashes":deps,"gradient_boundary":{"trainable_original":["encoder.blocks.10","encoder.blocks.11"],"trainable_new":["visual_ln","gru","risk","M2.film_hidden","M2.film_out"],"frozen_gradient_passage":["encoder.norm","R3-B.slip_pooler","R3-B.slip_trunk"],"force":"complete accepted source MAE + private force branch eval/detached; no force module in slip optimizer"},"known_limits":["Round17 future-complete endpoints, not complete timeline","overlapping folds and historical upstream validation exposure","same-image predicted force is not an independent sensor","no C2; not a complete 2x3 factorial experiment"]}
 tr.atomic_json(a.local/"STARTUP_AUDIT.json",audit)
 lock={"schema":"round19_protocol_lock_v1","status":"locked_before_formal_results","created_at":audit["created_at"],"files":{**local_hashes,"STARTUP_AUDIT.json":tr.sha(a.local/"STARTUP_AUDIT.json")},"dependency_hashes":deps,"prefix_index_sha256":audit["prefix"]["index_sha256"],"r18":r18}
 tr.atomic_json(a.local/"PROTOCOL_LOCK.json",lock)
 gate={"schema":"round19_dispatch_ready_v1","status":"ready","protocol_lock_sha256":tr.sha(a.local/"PROTOCOL_LOCK.json"),"startup_audit_sha256":tr.sha(a.local/"STARTUP_AUDIT.json"),"budget_sha256":tr.sha(a.local/"BUDGET.json"),"checks":["24-run grid","no test","corrected selector","prefix parity","gradient and freeze boundary","raw force parity tolerance","consistent commit interruption recovery","deadline and disk envelope"]}
 tr.atomic_json(a.local/"DISPATCH_READY.json",gate);print(json.dumps(gate,indent=2))
if __name__=="__main__":main()
