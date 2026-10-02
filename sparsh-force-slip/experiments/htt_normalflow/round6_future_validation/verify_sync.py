#!/usr/bin/env python3
"""Verify the immutable bundle on each host; optionally verify its origin files."""
import argparse,hashlib,json
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--verify-sources',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 m=a.bundle/'SHA256_MANIFEST.json';d=json.loads(m.read_text());bad=[]
 for x in d['files']:
  f=a.bundle/x['path']
  if not f.is_file() or sha(f)!=x['sha256']:bad.append(x['path'])
  if a.verify_sources:
   q=Path(x['source'])
   if not q.is_file() or sha(q)!=x['sha256']:bad.append('source:'+x['source'])
 out={'status':'pass' if not bad else 'fail','manifest_sha256':sha(m),'files_verified':len(d['files']),'bytes_verified':sum(x['bytes'] for x in d['files']),'source_files_verified':a.verify_sources,'mismatches':bad}
 a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out));return bool(bad)
if __name__=='__main__':raise SystemExit(main())
