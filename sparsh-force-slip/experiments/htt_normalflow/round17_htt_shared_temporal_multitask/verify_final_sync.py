#!/usr/bin/env python3
"""Read-only three-root verification against the frozen manifest."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import multitask_train as mt
REMOTE="""import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);m=json.load(open(root/'DELIVERY_MANIFEST.json'));bad=[];missing=[]
for rel,rec in m['files'].items():
 p=root/rel
 if not p.is_file():missing.append(rel);continue
 h=hashlib.sha256(p.read_bytes()).hexdigest()
 if h!=rec['sha256'] or p.stat().st_size!=rec['bytes']:bad.append(rel)
print(json.dumps({'root':str(root),'files':len(m['files']),'missing':missing,'bad':bad,'manifest_sha256':hashlib.sha256((root/'DELIVERY_MANIFEST.json').read_bytes()).hexdigest()}))
"""
def main():
 p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path(__file__).parent);p.add_argument("--host",default="zjy-4090");p.add_argument("--server-code",required=True);p.add_argument("--server-delivery",required=True);a=p.parse_args();m=json.loads((a.root/"DELIVERY_MANIFEST.json").read_text());local={"root":str(a.root),"files":len(m["files"]),"missing":[],"bad":[],"manifest_sha256":mt.sha(a.root/"DELIVERY_MANIFEST.json")}
 for rel,rec in m["files"].items():
  x=a.root/rel
  if not x.is_file():local["missing"].append(rel)
  elif mt.sha(x)!=rec["sha256"] or x.stat().st_size!=rec["bytes"]:local["bad"].append(rel)
 rem=[]
 for root in (a.server_code,a.server_delivery):
  q=subprocess.run(["ssh",a.host,"/home/zjy/miniconda3/envs/sparsh/bin/python","-",root],input=REMOTE,text=True,capture_output=True,check=True);rem.append(json.loads(q.stdout))
 allroots=[local]+rem;ok=all(not x["missing"] and not x["bad"] and x["manifest_sha256"]==local["manifest_sha256"] for x in allroots);proof={"schema":"round17_final_sync_proof_v1","status":"pass" if ok else "fail","manifest_sha256":local["manifest_sha256"],"roots":allroots}
 if not ok:raise AssertionError(proof)
 mt.atomic_json(proof,a.root/"FINAL_SYNC_PROOF.json")
 for root in (a.server_code,a.server_delivery):subprocess.run(["rsync","-a",str(a.root/"FINAL_SYNC_PROOF.json"),f"{a.host}:{root}/FINAL_SYNC_PROOF.json"],check=True)
 print(json.dumps(proof,indent=2))
if __name__=="__main__":main()
