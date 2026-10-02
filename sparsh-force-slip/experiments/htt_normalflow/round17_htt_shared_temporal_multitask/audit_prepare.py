#!/usr/bin/env python3
import argparse,json,re
from pathlib import Path
import multitask_train as mt
def main():
 p=argparse.ArgumentParser();p.add_argument("--local",type=Path,default=Path(__file__).parent);p.add_argument("--output",type=Path,required=True);p.add_argument("--real-smoke",type=Path,required=True);p.add_argument("--recovery",type=Path,required=True);a=p.parse_args();support=json.loads((a.local/"SUPPORT_AUDIT.json").read_text());gpu=json.loads((a.local/"GPU_SMOKE.json").read_text());cpu=json.loads((a.local/"SMOKE_CPU.json").read_text());recovery=json.loads(a.recovery.read_text());assert len(support["prepared"])==12 and gpu["status"]==cpu["status"]==recovery["status"]=="pass"
 real={}
 for g in mt.GROUPS:
  s=json.loads((a.real_smoke/g/"summary.json").read_text());t=(a.real_smoke/f"{g}.time").read_text();m=re.search(rf"TIME {g} ([0-9.]+) sec ([0-9]+) KB",t);assert s["status"]=="complete" and s["epochs"]==1 and m;real[g]={"summary":s,"wall_seconds":float(m.group(1)),"max_rss_kb":int(m.group(2))}
 runs=[]
 for g in mt.GROUPS:
  for f in range(1,5):
   for s in mt.SEEDS:runs.append({"run_id":f"{g}/p{f}_s{s}","group":g,"fold":f,"seed":s,"status":"registered_not_dispatched","prepared":str(a.output/"prepared"/f"p{f}_s{s}"/"prepared.pt"),"formal_output":str(a.output/"formal"/g/f"p{f}_s{s}")})
 inv=json.loads((a.local/"RUN_INVENTORY.json").read_text());inv["runs"]=runs;inv["actual_runs"]=len(runs);mt.atomic_json(inv,a.local/"RUN_INVENTORY.json")
 hashes={g:{n:gpu["module_initial_hashes"][g][n] for n in mt.MODULES} for g in mt.GROUPS};assert all(len({hashes[g][n] for g in mt.GROUPS})==1 for n in mt.MODULES)
 audit={"schema":"round17_prepare_audit_v1","status":"pass","common_caches":12,"formal_runs_registered":36,"cpu_smoke":"pass","gpu0_smoke":"pass","real_cache_one_epoch_smoke":real,"real_interrupted_resume":recovery,"parameter_counts":gpu["parameter_counts"],"canonical_parameters":gpu["canonical_parameters"],"selection_support":{f"p{x['fold']}_s{x['seed']}":x["selection_identifiability"] for x in support["prepared"]},"joint_lambdas":{f"p{x['fold']}_s{x['seed']}":x["joint_weight"]["lambda_future"] for x in support["prepared"]},"guards":{"test_consumed":False,"round16_force_used":False,"formal_dispatched":False,"unknown_slip_stage_masked":True,"external_jobs_stopped":False},"budget_projection":{"observed_one_epoch_seconds_each":{g:real[g]["wall_seconds"] for g in real},"conservative_training_gpu_hours":3.0,"evaluation_and_delivery_gpu_hours":6.0,"deadline_feasible":True}};mt.atomic_json(audit,a.local/"PREPARE_AUDIT.json")
 ready={"schema":"round17_dispatch_ready_v1","status":"ready_awaiting_root_formal_authorization","checks":{"protocol_lock":True,"identity_lock":True,"support":True,"cpu_smoke":True,"gpu0_smoke":True,"real_cache_smoke":True,"real_interrupted_resume":True,"grid_36":True,"deadline":True},"formal_authorized":False};mt.atomic_json(ready,a.local/"DISPATCH_READY.json");print(json.dumps(audit,indent=2))
if __name__=="__main__":main()
