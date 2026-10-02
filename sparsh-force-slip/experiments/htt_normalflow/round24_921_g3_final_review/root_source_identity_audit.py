import pathlib,json,hashlib,datetime,subprocess
P=pathlib.Path(__file__).resolve().parent
R=P.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks=[];remote={}
sets=[('round22_921_g1_joint_frozen','G1_FINAL_SYNC_PROOF.json',['ROOT_G1_ACCEPTANCE.json','G1_FINAL_ARTIFACT_INDEX.json','SUMMARY_ZH.md','E0_F3_SUPPORT.json','FROZEN_PROTOCOL_LOCK.json','E3_PROTOCOL_LOCK.json','formal_evaluation/detection_core/WORKPOINT_METRICS.csv','formal_evaluation/detection_core/PAIRED_CI.csv','formal_evaluation/force_core/METRICS.csv','formal_evaluation/force_core/PAIRED_CI.csv']),('round23_921_g2_force_aux_finetune','ROOT_G2_SYNC_PROOF.json',['ROOT_G2_ACCEPTANCE.json','REPORT_ZH.md','AUTHOR_AUDIT.json','ARTIFACT_INDEX.json','ROOT_TRAINING_AUDIT.json','ROOT_ROLE_AUDIT.json','ROOT_EVALUATION_AUDIT.json','CHECKPOINT_INDEX.json'])]
for folder,proof,names in sets:
 base=R/folder; d=json.loads((base/proof).read_text())
 expected=d.get('local_server_sha256',{e['path']:e['sha256'] for e in d.get('entries',[])})
 mapping={e['path']:e['server_path'] for e in d.get('entries',[])}
 for name in names+[proof]:
  path=base/name;actual=sha(path); exp=expected.get(name)
  checks.append({'path':str(path),'sha256':actual,'historical_sha256':exp,'matches_historical':actual==exp if exp else None})
  if folder.startswith('round22') and name.startswith('formal_evaluation/'): mapping[name]='/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/'+folder+'/'+name
  remote[mapping.get(name,'/home/zjy/document/tactile-grasp/sparsh-force-slip/experiments/htt_normalflow/'+folder+'/'+name)]=actual
 acceptance=json.loads((base/names[0]).read_text())
 for n,h in acceptance.get('evidence_sha256',{}).items():
  assert sha(base/n)==h,n
code='import json,hashlib,pathlib; x=json.loads('+repr(json.dumps(remote))+'); print(json.dumps({p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in x}))'
res=subprocess.run(['ssh','zjy-4090','python3 -'],input=code,text=True,capture_output=True,check=False)
if res.returncode: raise RuntimeError(res.stderr)
actual=json.loads(res.stdout); mismatch=[p for p,h in remote.items() if actual[p]!=h]
out={'at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'passed':not mismatch and all(c['matches_historical'] is not False for c in checks),'checks':checks,'remote_mismatches':mismatch,'scope':'Selected small evidence only, no full checkpoint/data hashing. Mutable current plans are snapshotted separately; historical plan SHA remains a historical identity.'}
(P/'ROOT_SOURCE_IDENTITY_AUDIT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'passed':out['passed'],'files':len(checks),'mismatches':mismatch}))
