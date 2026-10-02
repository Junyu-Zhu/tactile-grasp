import json, hashlib, itertools
from pathlib import Path
from datetime import datetime,timezone
import torch
O=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round10_htt_force_supervision_adaptation')
out=Path(__file__).resolve().parents[1]/'reviews'
seen={}
def sha(p):
 p=str(p)
 if p not in seen:
  h=hashlib.sha256()
  with open(p,'rb') as f:
   for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
  seen[p]=h.hexdigest()
 return seen[p]
def verify(p,h):assert sha(p)==h,(p,sha(p),h)
proof=json.loads((out/'LOCAL_PRECHECK.json').read_text())
for item in proof['remote_key_files']:verify(item['path'],item['sha256'])
rows=[]
for f,s in itertools.product(range(1,5),[20260914,20260915,20260916]):
 tag=f'p{f}_s{s}'
 c=json.loads((O/f'formal/fusion/{tag}/config.json').read_text())
 for k in ['prepared','prepared_audit']:verify(**{'p':c[k]['path'],'h':c[k]['sha256']})
 d=torch.load(c['prepared']['path'],map_location='cpu',weights_only=False)
 assert d['fold']==f'htt_leave_p{f}' and d['seed']==s and d['experiment']=='round10_new_force'
 assert d['force_replacement_audit']['pass'] and not d['force_replacement_audit']['smoke']
 for k in ['force_checkpoint','visual_checkpoint','source_checkpoint','split','contract','cache_audit','force_prediction_manifest']:
  q=d['provenance'][k];verify(q['path'],q['sha256'])
 for q in d['force_replacement_audit']['proofs']:
  if not q['path'].endswith('.npy') and not q['path'].endswith('prepared.pt'):verify(q['path'],q['sha256'])
 roles=d['roles'];assert set(roles)=={'train','validation','calibration'}
 groups={k:set(v['leakage_group']) for k,v in roles.items()};eps={k:set(v['episode_id']) for k,v in roles.items()}
 for a,b in itertools.combinations(roles,2):assert not groups[a]&groups[b] and not eps[a]&eps[b]
 support={}
 for k,v in roles.items():
  assert v['x'].shape[1:]==(9,199) and int(v['t'].min())>=13 and torch.isfinite(v['x']).all()
  assert torch.all(v['x'][:,:5,198]==0) and torch.all(v['x'][:,5:,198]==1)
  support[k]={'trials':len(eps[k]),'groups':len(groups[k]),'endpoints':len(v['t']),'static':int((v['stage']==0).sum()),'incipient':int((v['stage']==1).sum()),'gross':int((v['stage']==2).sum()),'min_t':int(v['t'].min())}
 rows.append({'fold':f,'seed':s,'prepared':c['prepared'],'force':d['provenance']['force_checkpoint'],'visual':d['provenance']['visual_checkpoint'],'roles':support,'upstream_audit':d['upstream_audit']})
result={'status':'pass','created_at':datetime.now(timezone.utc).isoformat(),'runs':rows,'hashes':seen,'hashed_paths':len(seen),'no_test':True,'role_isolation':True,'full_token_rehash':False,'scope':'Reuses accepted full encoder token and raw audit; fresh checks of downstream prepared, force/visual/source checkpoints, contracts, split and critical small files. Not a new raw dataset audit.'}
(out/'REMOTE_PRECHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps({'status':'pass','runs':len(rows),'hashes':len(seen)}))
