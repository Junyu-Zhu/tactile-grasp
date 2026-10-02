#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();m=json.loads((a.root/'DELIVERY_MANIFEST.json').read_text());bad=[];missing=[]
 for rel,x in m['files'].items():
  q=a.root/rel
  if not q.exists():missing.append(rel)
  elif q.stat().st_size!=x['bytes'] or sha(q)!=x['sha256']:bad.append(rel)
 out={'root':str(a.root),'files':m['file_count'],'missing':missing,'bad':bad,'manifest_sha256':sha(a.root/'DELIVERY_MANIFEST.json'),'status':'pass' if not missing and not bad else 'fail'};print(json.dumps(out));assert out['status']=='pass'
if __name__=='__main__':main()
