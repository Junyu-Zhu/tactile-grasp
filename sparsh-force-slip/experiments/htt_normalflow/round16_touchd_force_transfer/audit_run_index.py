#!/usr/bin/env python3
"""Verify the locked 75-run checkpoint inventory without reading model tensors."""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--index',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--verify-hashes',action='store_true');a=p.parse_args()
 d=json.loads(a.index.read_text());entries=d['entries'];expected={'touchd':3,'htt_force':24,'slip':24,'future':24};counts=dict(Counter(x['stage'] for x in entries));assert len(entries)==75 and counts==expected and d['counts']==expected
 assert len({(x['stage'],x['run']) for x in entries})==75 and all('smoke' not in (x['run']+' '+x['summary']+' '+x['best']+' '+x['latest']).lower() for x in entries)
 checked=0
 for x in entries:
  for field in ('summary','best','latest'):
   path=Path(x[field]);assert path.is_file() and path.stat().st_size>0
   if a.verify_hashes:assert sha(path)==x[field+'_sha256']
   checked+=1
 result={'schema':'round16_run_index_audit_v1','status':'pass','formal_runs':75,'counts':expected,'unique_stage_run_keys':75,'artifacts_present':checked,'artifact_hashes_verified':bool(a.verify_hashes),'smoke_entries':0,'index_path':str(a.index),'index_sha256':sha(a.index)}
 a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
