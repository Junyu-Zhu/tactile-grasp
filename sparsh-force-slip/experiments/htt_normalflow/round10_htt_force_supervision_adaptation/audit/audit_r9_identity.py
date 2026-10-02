import pathlib,json,hashlib,subprocess,datetime
L=pathlib.Path('/home/zjy/Documents/grasp/tactile_grasp/sparsh-force-slip/experiments/htt_normalflow')
r=L/'round9_htt_temporal_force_fusion'; out=L/'round10_htt_force_supervision_adaptation/audit'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
 return h.hexdigest()
x=json.loads((r/'LOCAL_SYNC_PROOF.json').read_text()); assert sha(r/'LOCAL_SYNC_PROOF.json')==(r/'LOCAL_SYNC_PROOF.sha256').read_text().split()[0]
local=[]
for row in x['mirror_files']:
 h=sha(r/row['local']); assert h==row['sha256'],row['local'];local.append({'path':row['local'],'sha256':h})
critical=['FINAL_STATUS.json','PROTOCOL.md','training/protocol.json','results/reporting/SUMMARY_ZH.md','results/formal_delivery/TRAINING_AUDIT.json','reviews/INDEPENDENT_FINAL_REVIEW.json','results/prepare/PRECHECK.json','results/force_diagnostics/SUPPORT_AUDIT.json','results/current_sensitivity/AUDIT.json','results/current_evaluation/summary.json','results/FORMAL_INVENTORY.json','results/PROTOCOL_LOCK.json']
remote={row['server']:row['sha256'] for row in x['mirror_files'] if row['local'] in critical}
for row in x['origin_files']:
 if row['local'] in critical:remote[row['origin']]=row['sha256']
for p in sorted((r/'results/prepare').glob('p*_s*/audit.json')):
 a=json.loads(p.read_text());assert a['status']=='pass'
 for v in a['provenance'].values():
  if isinstance(v,dict) and 'path' in v and 'sha256' in v:remote[v['path']]=v['sha256']
 for path,h in a['provenance']['dependency_hashes'].items():remote[path]=h
script='''import pathlib,json,hashlib\nexpected=EXPECTED\nrows=[]\nfor p,want in expected.items():\n h=hashlib.sha256()\n with open(p,'rb') as f:\n  for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)\n got=h.hexdigest();assert got==want,(p,got,want)\n rows.append({'path':p,'sha256':got})\nprint(json.dumps(rows))\n'''.replace('EXPECTED',repr(remote))
proc=subprocess.run(['ssh','zjy-4090','/home/zjy/miniconda3/envs/sparsh/bin/python -'],input=script,text=True,capture_output=True,check=True);remote_rows=json.loads(proc.stdout)
status=json.loads((r/'FINAL_STATUS.json').read_text()); assert status['status']=='complete' and status['formal_training']['accepted']==36
review=json.loads((r/'reviews/INDEPENDENT_FINAL_REVIEW.json').read_text());assert review['status']=='pass'
assert sha(r/'results/reporting/SUMMARY_ZH.md')==review['summary_sha256']==status['report_sha256']
result={'status':'pass','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'r9_status':status,'local_proof_sha256':sha(r/'LOCAL_SYNC_PROOF.json'),'local_checked_count':len(local),'remote_checked_count':len(remote_rows),'local_checked':local,'remote_checked':remote_rows,'scope':'400 small delivered mirror files locally, critical live remote reports/protocols and all 12 force/visual upstream identities; token hash audit reused, no full token or raw image scan','inherited_selection_limit':'R5 force best checkpoint selected on outer validation RMSE. R3 visual and R9 fusion also used validation selection. R10 force internal train selection avoids NEW outer validation optimization only; development evaluation is not blind.','fair_reuse_requirements':['same t>=13 role/endpoint population','R9 V and F_history full 12 runs retained','same fusion architecture/init/loss/budget/selection/calibration','new predicted-force train-only normalizer allowed by same rule; input pipeline replacement rather than isolated parameter intervention'],'test_role_consumed':False}
(out/'R9_IDENTITY_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'pass','local':len(local),'remote':len(remote_rows)}))
