"""Synchronize G3 small artifacts and verify exact bytes; excludes final receipts."""
import pathlib,json,hashlib,subprocess,datetime
P=pathlib.Path(__file__).resolve().parent
S='/home/zjy/document/tactile-grasp/sparsh-force-slip/experiments/htt_normalflow/'+P.name
excluded={'ROOT_FINAL_SYNC_PROOF.json','ROOT_G3_ACCEPTANCE.json'}
subprocess.run(['ssh','zjy-4090','mkdir -p '+S],check=True)
subprocess.run(['rsync','-a','--exclude=__pycache__/','--exclude=*.pyc','--exclude=ROOT_FINAL_SYNC_PROOF.json','--exclude=ROOT_G3_ACCEPTANCE.json',str(P)+'/', 'zjy-4090:'+S+'/'],check=True)
entries=[]
for f in sorted(P.rglob('*')):
 if not f.is_file() or f.name in excluded or '__pycache__' in f.parts or f.suffix=='.pyc':continue
 entries.append({'path':str(f.relative_to(P)),'server_path':S+'/'+str(f.relative_to(P)),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
code='import pathlib,hashlib,json; paths='+repr([e['server_path'] for e in entries])+'; print(json.dumps({p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in paths}))'
r=subprocess.run(['ssh','zjy-4090','python3 -'],input=code,text=True,capture_output=True,check=True)
remote=json.loads(r.stdout)
issues=[e['path'] for e in entries if e['sha256']!=remote.get(e['server_path'])]
o={'checked_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'passed':not issues,'file_count':len(entries),'entries':entries,'mismatches':issues,'excluded_receipts':sorted(excluded),'notes':'Final proof and acceptance copied and verified separately after manifest freeze.'}
(P/'ROOT_FINAL_SYNC_PROOF.json').write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'passed':o['passed'],'file_count':len(entries),'mismatches':issues}))
