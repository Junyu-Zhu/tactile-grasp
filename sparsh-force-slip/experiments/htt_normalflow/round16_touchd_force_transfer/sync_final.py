#!/usr/bin/env python3
"""Sync a frozen compact delivery to the two server mirrors; dry-run by default."""
import argparse,json,shlex,subprocess,tempfile
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--local',type=Path,required=True);p.add_argument('--host',default='zjy-4090');p.add_argument('--server-code',required=True);p.add_argument('--server-delivery',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args();m=json.loads((a.local/'DELIVERY_MANIFEST.json').read_text());files=[x['relative_path'] for x in m['files']]+['DELIVERY_MANIFEST.json','FINAL_STATUS.json'];files += ['FINAL_SYNC_PROOF.json'] if (a.local/'FINAL_SYNC_PROOF.json').is_file() else []
 if not a.execute:print(json.dumps({'status':'dry_run','files':len(files),'destinations':[a.server_code,a.server_delivery]}));return
 subprocess.run(['ssh',a.host,shlex.join(['mkdir','-p',a.server_code,a.server_delivery])],check=True)
 with tempfile.NamedTemporaryFile('w') as f:
  f.write('\n'.join(files)+'\n');f.flush()
  for dest in (a.server_code,a.server_delivery):subprocess.run(['rsync','-a','--files-from',f.name,str(a.local)+'/','%s:%s/'%(a.host,dest)],check=True)
 print(json.dumps({'status':'synced','files':len(files),'destinations':[a.server_code,a.server_delivery]}))
if __name__=='__main__':main()
