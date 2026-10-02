#!/usr/bin/env python3
"""Build the non-circular compact delivery manifest after independent review."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
EXCLUDE={'DELIVERY_MANIFEST.json','FINAL_SYNC_PROOF.json','FINAL_STATUS.json','ROOT_ACCEPTANCE.json'}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 review=json.loads((ROOT/'INDEPENDENT_FINAL_REVIEW.json').read_text());provenance=json.loads((ROOT/'ROOT_PROVENANCE.json').read_text())
 assert review['status']=='pass_with_disclosed_limitations' and review['blocking_findings_remaining']==0
 assert provenance['reviewer']['status']=='pass_with_disclosed_limitations'
 files={}
 for p in sorted(ROOT.rglob('*')):
  rel=str(p.relative_to(ROOT))
  if not p.is_file() or '__pycache__' in p.parts or p.name in EXCLUDE:continue
  files[rel]={'sha256':sha(p),'bytes':p.stat().st_size}
 manifest={'schema':'round18_delivery_manifest_v1','status':'complete','file_count':len(files),'bytes':sum(x['bytes'] for x in files.values()),'files':files,'excluded_cycle_closure':sorted(EXCLUDE),'formal_runs':36,'test_consumed':False}
 (ROOT/'DELIVERY_MANIFEST.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n');print(json.dumps({'files':len(files),'bytes':manifest['bytes'],'sha256':sha(ROOT/'DELIVERY_MANIFEST.json')},indent=2))
if __name__=='__main__':main()
