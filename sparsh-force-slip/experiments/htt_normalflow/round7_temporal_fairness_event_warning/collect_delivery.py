#!/usr/bin/env python3
"""Immutable small-artifact delivery; no large checkpoints or dataset caches."""
import argparse,hashlib,json,os,shutil,subprocess,sys,tempfile
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--code',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 subprocess.run([sys.executable,str(a.code/'verify_completion.py'),'--root',str(a.results),'--pre-sync'],check=True)
 if a.output.exists():raise RuntimeError('Do not overwrite immutable delivery')
 allow={'.py','.md','.json','.jsonl','.csv','.yaml','.yml','.txt','.log','.png','.jpg','.svg','.sha256','.gz'};files=[]
 for root,prefix in [(a.code,'source'),(a.results,'results')]:
  for src in root.rglob('*'):
   if not src.is_file() or src.suffix not in allow or '__pycache__' in src.parts or a.output in src.parents or 'delivery_bundle' in src.parts or any(part.startswith('.'+a.output.name+'.staging-') for part in src.parts):continue
   if src.stat().st_size>100*1024**2:raise RuntimeError('Large small-artifact candidate: '+str(src))
   rel=Path(prefix)/src.relative_to(root);files.append((src,rel))
 a.output.parent.mkdir(parents=True,exist_ok=True)
 staging=Path(tempfile.mkdtemp(prefix='.'+a.output.name+'.staging-',dir=a.output.parent));records=[]
 try:
  for src,rel in sorted(files,key=lambda x:str(x[1])):
   dest=staging/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest);digest=sha(dest)
   if digest!=sha(src):raise RuntimeError('Changed during snapshot: '+str(src))
   records.append({'path':str(rel),'source':str(src),'bytes':dest.stat().st_size,'sha256':digest})
  m={'status':'snapshot_verified','files':records};(staging/'SHA256_MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n')
  if a.output.exists():raise RuntimeError('Do not overwrite immutable delivery')
  os.rename(staging,a.output)
 finally:
  if staging.exists():shutil.rmtree(staging)
 print(json.dumps({'files':len(records),'bytes':sum(x['bytes'] for x in records),'manifest_sha256':sha(a.output/'SHA256_MANIFEST.json')}))

if __name__=='__main__':main()
