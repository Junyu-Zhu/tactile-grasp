#!/usr/bin/env python3
"""Build the deterministic Round-16 compact-delivery manifest after metadata freeze."""
import argparse,datetime,hashlib,json
from pathlib import Path

CONTROL={'DELIVERY_MANIFEST.json','FINAL_STATUS.json','FINAL_SYNC_PROOF.json'}
EXCLUDED_DIRS={'__pycache__','review_mirror_v1','review_mirror_v2'}
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--output',type=Path);a=p.parse_args();root=a.root.resolve();output=a.output or root/'DELIVERY_MANIFEST.json';files=[]
 for path in sorted(x for x in root.rglob('*') if x.is_file()):
  rel=path.relative_to(root);s=rel.as_posix()
  if s in CONTROL or any(part in EXCLUDED_DIRS for part in rel.parts):continue
  files.append({'relative_path':s,'bytes':path.stat().st_size,'sha256':sha(path)})
 receipt={'schema':'round16_delivery_manifest_v1','status':'complete','created_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'local_root':str(root),'file_count':len(files),'total_bytes':sum(x['bytes'] for x in files),'files':files,'self_referential_control_files_excluded_from_file_table':sorted(CONTROL),'excluded_noncanonical_working_copies':sorted(EXCLUDED_DIRS),'large_artifacts':'Formal checkpoints, cache shards, prediction payloads, and large diagnostic CSVs remain under the server output root and are indexed by ARTIFACT_INDEX.json, REVIEW_READY.json, and CHECKPOINT_INDEX.json.'}
 output.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':'complete','files':len(files),'bytes':receipt['total_bytes']}))
if __name__=='__main__':main()
