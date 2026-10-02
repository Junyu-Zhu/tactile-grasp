#!/usr/bin/env python3
"""Read-only SHA verification of the local, server-code, and server-delivery mirrors."""
import argparse,hashlib,json,shlex,subprocess
from pathlib import Path
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def remote_inventory(host,root,rels):
 code="""import hashlib,json,sys\nfrom pathlib import Path\nr=Path(sys.argv[1]); out={}\nfor s in json.loads(sys.stdin.read()):\n p=r/s\n if not p.is_file(): out[s]=None; continue\n h=hashlib.sha256()\n with p.open('rb') as f:\n  for b in iter(lambda:f.read(1<<20),b''):h.update(b)\n out[s]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}\nprint(json.dumps(out))\n"""
 remote=shlex.join(['/home/zjy/miniconda3/envs/sparsh/bin/python','-c',code,str(root)])
 cp=subprocess.run(['ssh',host,remote],input=json.dumps(rels),text=True,capture_output=True,check=True);return json.loads(cp.stdout)
def main():
 p=argparse.ArgumentParser();p.add_argument('--local',type=Path,required=True);p.add_argument('--host',default='zjy-4090');p.add_argument('--server-code',required=True);p.add_argument('--server-delivery',required=True);p.add_argument('--output',type=Path);a=p.parse_args();m=json.loads((a.local/'DELIVERY_MANIFEST.json').read_text());rels=[x['relative_path'] for x in m['files']];expected={x['relative_path']:{'bytes':x['bytes'],'sha256':x['sha256']} for x in m['files']}
 local={s:{'bytes':(a.local/s).stat().st_size,'sha256':sha(a.local/s)} for s in rels};assert local==expected
 controls=[x for x in ('DELIVERY_MANIFEST.json','FINAL_STATUS.json') if (a.local/x).is_file()];allrels=rels+controls;remotes={name:remote_inventory(a.host,root,allrels) for name,root in [('server_code',a.server_code),('server_delivery',a.server_delivery)]}
 for inv in remotes.values():
  assert {s:inv[s] for s in rels}==expected
  for s in controls:assert inv[s]=={'bytes':(a.local/s).stat().st_size,'sha256':sha(a.local/s)}
 proof={'schema':'round16_final_sync_proof_v1','status':'pass','manifest_sha256':sha(a.local/'DELIVERY_MANIFEST.json'),'files_verified_per_root':len(rels),'roots':{'local':str(a.local.resolve()),'server_code':a.server_code,'server_delivery':a.server_delivery},'control_files_verified':controls,'self_exclusion':'FINAL_SYNC_PROOF.json is written only after verification and is therefore not an input to its own proof.'}
 out=a.output or a.local/'FINAL_SYNC_PROOF.json';out.write_text(json.dumps(proof,indent=2,sort_keys=True)+'\n');print(json.dumps(proof))
if __name__=='__main__':main()
