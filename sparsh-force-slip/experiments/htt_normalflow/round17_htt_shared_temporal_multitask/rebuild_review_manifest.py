#!/usr/bin/env python3
"""Rebuild the local review manifest after report-only repair synchronization."""
import argparse,hashlib,json,os,tempfile
from pathlib import Path
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""):h.update(block)
 return h.hexdigest()
def atomic_json(obj,path):
 path=Path(path);fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=path.name+".",suffix=".tmp")
 try:
  with os.fdopen(fd,"w") as f:json.dump(obj,f,indent=2,sort_keys=True);f.write("\n")
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)

def main():
 p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path(__file__).parent);a=p.parse_args();root=a.root
 old=json.loads((root/"REVIEW_READY.json").read_text())
 files={str(x.relative_to(root)):{"sha256":sha(x),"bytes":x.stat().st_size} for x in root.rglob("*") if x.is_file() and x.name not in ("REVIEW_READY.json","review_ready.log") and "__pycache__" not in x.parts}
 old["schema"]="round17_review_ready_v2";old["status"]="ready_for_targeted_independent_recheck";old["report_repair_audit"]="REPORT_REPAIR_AUDIT.json";old["local_files"]=len(files);old["file_manifest"]=files
 atomic_json(old,root/"REVIEW_READY.json");print(json.dumps({k:v for k,v in old.items() if k!="file_manifest"},indent=2))
if __name__=="__main__":main()
