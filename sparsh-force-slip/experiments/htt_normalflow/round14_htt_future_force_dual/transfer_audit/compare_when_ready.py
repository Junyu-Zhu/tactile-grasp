import hashlib,json,subprocess,time
from pathlib import Path
r=Path(__file__).resolve().parent
remote='/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round14_htt_future_force_dual/transfer_audit/server'
for _ in range(120):
 result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','zjy-4090',f'cat {remote}/FINAL.json'],capture_output=True,text=True)
 if result.returncode==0:
  final=json.loads(result.stdout);break
 time.sleep(60)
else:raise SystemExit('Timed out waiting for remote hash completion; resume compare script after repair')
(r/'server').mkdir(exist_ok=True)
for name in ['FINAL.json','progress.json','hashes.jsonl']:
 subprocess.run(['scp','-q',f'zjy-4090:{remote}/{name}',str(r/'server'/name)],check=True)
def read_journal(p):return {x['path']:x for x in map(json.loads,p.read_text().splitlines())}
a=read_journal(r/'local/hashes.jsonl');b=read_journal(r/'server/hashes.jsonl');expected={e['path']:e['size'] for e in json.loads((r/'expected.json').read_text())}
bad=[]
for p,n in expected.items():
 if p not in a or p not in b:bad.append({'path':p,'reason':'missing hash'})
 elif a[p]['size']!=n or b[p]['size']!=n or a[p]['sha256']!=b[p]['sha256']:bad.append({'path':p,'reason':'size or content hash mismatch'})
local_final=json.loads((r/'local/FINAL.json').read_text())
receipt={'schema':'round14_full_touchd_force_transfer_sha256_v1','status':'pass' if not bad and final['status']==local_final['status']=='pass' else 'fail','expected_files':len(expected),'total_bytes':sum(expected.values()),'compared_files':len(a.keys()&b.keys()),'mismatches':bad,'local_root':local_final['root'],'server_root':final['root'],'local_elapsed_s':local_final['elapsed_s'],'server_elapsed_s':final['elapsed_s'],'manifest_sha256':hashlib.sha256((r/'expected.json').read_bytes()).hexdigest(),'checked_at_unix':time.time(),'scope':'All expected Force files read fully and SHA256 compared local original vs server copy; not Mani; no mutation of data'}
(r/'TRANSFER_SHA256_PROOF.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
