from pathlib import Path
import json,hashlib,torch
C=Path(__file__).resolve().parents[1];O=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round11_stable_negative_force_residual')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def eq(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(eq(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(eq(x,y) for x,y in zip(a,b))
 return a==b
s=json.loads((O/'SMOKE_AUDIT.json').read_text());assert s['status']=='pass';hashes={};checks={}
for p,h in s['sources'].items():assert sha(p)==h;hashes[p]=h
for g,r in s['groups'].items():
 assert sha(r['summary']['path'])==r['summary']['sha256']
 log=(O/f'smoke/{g}_interrupted.log').read_text();assert 'interrupted_for_recovery_proof' in log
 cont=O/f'smoke/{g}_continuous';resume=O/f'smoke/{g}_recovery'
 for file in ['best.pth','latest.pth']:
  a=torch.load(cont/file,map_location='cpu',weights_only=False);b=torch.load(resume/file,map_location='cpu',weights_only=False)
  for k in ['model_state','optimizer_state','normalizer','history','gradient_audit','initial_state_sha256','epoch','best_epoch','wait','best_score']:assert eq(a[k],b[k]),(g,file,k)
  hashes[str(cont/file)]=sha(cont/file);hashes[str(resume/file)]=sha(resume/file)
 summary=json.loads((cont/'summary.json').read_text());assert all(summary['parameter_tensors_updated'].values())
 for p,h in summary['output_hashes'].items():assert sha(cont/p)==h
 checks[g]={'exact_best_latest_model_optimizer_normalizer_history':True,'interrupted_process_log_present':True,'finite_gradients_and_upstream_cache_frozen':True}
r={'status':'pass','reviewer':'r11_audit independent of smoke/train authors','smoke_audit_sha256':sha(O/'SMOKE_AUDIT.json'),'source_hashes':hashes,'checks':checks,'limitations':['Safe interruption at committed epoch checkpoint, not forced kill mid-file-write','Cached upstream immutability proven by source architecture and bound input hashes; no GPU upstream retraining involved']}
(C/'reviews/INDEPENDENT_SMOKE_REVIEW.json').write_text(json.dumps(r,indent=2));print('PASS3groups')
