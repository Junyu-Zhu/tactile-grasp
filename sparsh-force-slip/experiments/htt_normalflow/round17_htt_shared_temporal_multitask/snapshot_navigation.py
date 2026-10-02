#!/usr/bin/env python3
"""Capture final navigation documents as independent delivery snapshots."""
import argparse,hashlib,json,shutil
from pathlib import Path
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--round",type=Path,default=Path(__file__).parent);p.add_argument("--plan",type=Path,required=True);p.add_argument("--current-state",type=Path,required=True);a=p.parse_args();dst=a.round/"navigation_snapshot";dst.mkdir(parents=True,exist_ok=True)
 copies=((a.plan,dst/"FINAL_RESEARCH_PLAN_SNAPSHOT.md"),(a.current_state,dst/"FINAL_CURRENT_STATE_SNAPSHOT.md"))
 for src,out in copies:shutil.copy2(src,out)
 index={"schema":"round17_navigation_snapshot_v1","status":"frozen_independent_copy","does_not_replace_pretraining_or_protocol_snapshots":True,"files":{out.name:{"source":str(src),"sha256":sha(out),"bytes":out.stat().st_size} for src,out in copies}}
 (dst/"INDEX.json").write_text(json.dumps(index,indent=2,sort_keys=True)+"\n");print(json.dumps(index,indent=2))
if __name__=="__main__":main()
