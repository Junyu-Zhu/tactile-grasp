#!/usr/bin/env python3
"""Persistent root-gated queue for the 84 frozen E1/E2/F1 candidates."""
import argparse,json
from pathlib import Path
import formal_queue as Q
def load_inventory(path):
 inv=json.loads(Path(path).read_text());runs=inv.get("runs",[]);issues=[];packages=tuple(inv.get("packages",()));allowed={"E1":("V","C","M"),"E2":("MB",),"F1":("K-V","K-F","K-VF")};expected={(p,g,f,s) for p in packages for g in allowed.get(p,()) for f in range(1,5) for s in Q.SEEDS};seen=[];outputs=[];cache={}
 def digest(p):
  p=Path(p);st=p.stat();k=(str(p.resolve()),st.st_size,st.st_mtime_ns)
  if k not in cache:cache[k]=Q.sha(p)
  return cache[k]
 if inv.get("schema")!="round22_frozen_run_inventory_v1" or not packages or any(p not in allowed for p in packages) or len(runs)!=len(expected):return inv,[{"error":"schema_packages_or_count"}]
 for r in runs:
  seen.append((r.get("package"),r.get("group"),r.get("fold"),r.get("seed")));outputs.append(r.get("output"))
  for p,h in ((r.get("data"),r.get("data_sha256")),(r.get("trainer"),r.get("trainer_sha256")),(r.get("protocol"),r.get("protocol_sha256"))):
   if not p or not Path(p).is_file() or digest(p)!=h:issues.append({"run":r.get("run"),"error":"missing_or_hash_mismatch","path":p});break
  try:locked_source=json.loads(Path(r["protocol"]).read_text())["sources"][Path(r["trainer"]).name]["sha256"]
  except (OSError,KeyError,TypeError,json.JSONDecodeError):locked_source=None
  if locked_source!=r.get("trainer_sha256"):issues.append({"run":r.get("run"),"error":"protocol_trainer_source_mismatch"})
  expected_deps={"support_inventory_sha256":digest(Path(__file__).with_name("PREPARE_CURRENT_SUPPORT.json" if r.get("package") in ("E1","E2") else "G1_PREPARE.json"))}
  if r.get("package") in ("E1","E2"):expected_deps["r18_source_sha256"]=digest(Path(__file__).parent.parent/"round18_htt_force_conditioned_film"/"train.py")
  if r.get("dependencies")!=expected_deps:issues.append({"run":r.get("run"),"error":"dependency_hash_mismatch"})
 if set(seen)!=expected or len(seen)!=len(set(seen)):issues.append({"error":"grid_not_unique_complete"})
 if len(outputs)!=len(expected) or len(outputs)!=len(set(outputs)):issues.append({"error":"outputs_not_unique"})
 return inv,issues
def accepted(r):
 out=Path(r["output"])
 try:s=json.loads((out/"summary.json").read_text());i=s["identity"];c=json.loads((out/"COMMIT.json").read_text());committed=all(Q.sha(out/c[k]["path"])==c[k]["sha256"] for k in ("latest","best"))
 except Exception:return False
 expected={"group":r["group"],"fold":r["fold"],"seed":r["seed"],"data_sha256":r["data_sha256"],"source_sha256":r["trainer_sha256"],"protocol_sha256":r["protocol_sha256"],"formal":True,**r.get("dependencies",{})};return s.get("status")=="complete" and committed and all(i.get(k)==v for k,v in expected.items())
def main():
 p=argparse.ArgumentParser();p.add_argument("--authorization",type=Path,required=True);p.add_argument("--inventory",type=Path,required=True);p.add_argument("--state",type=Path,required=True);p.add_argument("--lock",type=Path,required=True);p.add_argument("--python",required=True);p.add_argument("--gpu",nargs="+",default=["0","1","2"]);p.add_argument("--execute",action="store_true");a=p.parse_args();auth=json.loads(a.authorization.read_text());inv,issues=load_inventory(a.inventory)
 if a.execute and (auth.get("g1_formal_authorized") is not True or auth.get("budget_pass") is not True or ("E2" in inv.get("packages",[]) and auth.get("e2_formal_authorized") is not True) or issues):raise SystemExit("frozen formal dispatch blocked by authorization, budget, E2 applicability, or inventory")
 devices=Q.gpu_inventory(a.gpu);owner=Q.acquire(a.lock);ih=Q.sha(a.inventory)
 if not a.state.exists():Q.atomic(a.state,{"schema":"round22_frozen_queue_v1","created_at":Q.now().isoformat(),"inventory_sha256":ih,"stop_dispatch":Q.STOP.isoformat(),"concurrency_limit":len(devices),"quarantined_devices":[],"events":[],"runs":[{**r,"status":"registered_not_dispatched","attempts":0,"pid":None} for r in inv["runs"]]})
 else:
  st=json.loads(a.state.read_text());actual=[(r.get("run"),r.get("output")) for r in st.get("runs",[])];expected=[(r["run"],r["output"]) for r in inv["runs"]]
  if st.get("schema")!="round22_frozen_queue_v1" or st.get("inventory_sha256")!=ih or len(actual)!=len(inv["runs"]) or len(actual)!=len(set(actual)) or set(actual)!=set(expected):raise SystemExit("existing frozen queue state mismatch")
 if not a.execute:print(json.dumps({"schema":"round22_frozen_queue_machine_v1","status":"prepared_candidate_not_dispatched","ready":not issues and auth.get("g1_formal_authorized") is True and auth.get("budget_pass") is True and ("E2" not in inv.get("packages",[]) or auth.get("e2_formal_authorized") is True),"inventory_issues":issues,"runs":len(inv["runs"]),"packages":inv.get("packages"),"g1_formal_authorized":auth.get("g1_formal_authorized") is True,"e2_formal_authorized":auth.get("e2_formal_authorized") is True,"budget_pass":auth.get("budget_pass") is True,"inventory_sha256":ih,"authorization_sha256":Q.sha(a.authorization),"queue_source_sha256":Q.sha(__file__),"core_queue_source_sha256":Q.sha(Q.__file__),"gpus":devices,"stop_dispatch":Q.STOP.isoformat()}));return
 def build(r,uuid):
  cmd=[a.python,r["trainer"],"--data",r["data"],"--output",r["output"],"--group",r["group"],"--fold",str(r["fold"]),"--seed",str(r["seed"]),"--device","cuda:0","--formal","--authorization",str(a.authorization),"--protocol",r["protocol"]];return cmd,Path(__file__).with_name("logs")/f"formal_{r['package']}_{r['group']}_p{r['fold']}_s{r['seed']}.log"
 uuids=[x["uuid"] for x in devices];state=Q.run_queue(a.state,a.lock,uuids,Q.STOP,accepted,lambda r:Q.pid_matches(r.get("pid"),(Path(r["trainer"]).name,r["output"],"--formal")),build,lambda u:{"CUDA_VISIBLE_DEVICES":u,"XFORMERS_DISABLED":"1","CUBLAS_WORKSPACE_CONFIG":":4096:8","OMP_NUM_THREADS":"4","MKL_NUM_THREADS":"4"},launch_discover=lambda r:Q.discover_pids((Path(r["trainer"]).name,r["output"],"--formal")),owner=owner);print(json.dumps({s:sum(r["status"]==s for r in state["runs"]) for s in {r["status"] for r in state["runs"]}}))
if __name__=="__main__":main()
