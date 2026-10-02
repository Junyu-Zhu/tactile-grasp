"""Resumable read-only file content hashing; results persisted after each file."""
import argparse,datetime,hashlib,json,os,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--expected',required=True);p.add_argument('--output',required=True);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=True);journal=out/'hashes.jsonl'
previous={}
if journal.exists():
 for line in journal.read_text().splitlines():
  try:
   row=json.loads(line);previous[row['path']]=row
  except (ValueError,KeyError):pass
expected=json.loads(Path(a.expected).read_text());start=time.time();done=[];errors=[]
for item in expected:
 f=Path(a.root)/item['path']
 try:
  st=f.stat();old=previous.get(item['path'])
  if old and old.get('size')==st.st_size and old.get('mtime_ns')==st.st_mtime_ns and old.get('sha256') and st.st_size==item['size']:row=old
  else:
   if st.st_size!=item['size']:raise ValueError('size differs from expected manifest')
   h=hashlib.sha256()
   with f.open('rb') as stream:
    for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
   after=f.stat()
   if (st.st_size,st.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('input changed during hashing')
   row={'path':item['path'],'size':st.st_size,'mtime_ns':st.st_mtime_ns,'sha256':h.hexdigest()}
   with journal.open('a') as stream:stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
  done.append(row)
 except Exception as e:errors.append({'path':item['path'],'error':str(e)})
 status={'root':a.root,'complete':len(done),'expected':len(expected),'hashed_bytes':sum(x['size'] for x in done),'elapsed_s':round(time.time()-start,1),'errors':errors,'updated_at':datetime.datetime.now().astimezone().isoformat()}
 tmp=out/'progress.tmp';tmp.write_text(json.dumps(status,indent=2));tmp.replace(out/'progress.json')
(out/'FINAL.json').write_text(json.dumps({**status,'status':'pass' if len(done)==len(expected) and not errors else 'fail'},indent=2))
print(json.dumps(status),flush=True)
