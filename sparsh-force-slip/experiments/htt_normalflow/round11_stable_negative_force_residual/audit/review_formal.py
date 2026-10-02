import argparse,json,csv,math,hashlib,importlib.util,itertools
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
C=Path(__file__).resolve().parents[1];O=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round11_stable_negative_force_residual')
sp=importlib.util.spec_from_file_location('trainer',C/'training/train.py');tr=importlib.util.module_from_spec(sp);sp.loader.exec_module(tr);torch.set_num_threads(2)
seen={}
def sha(p):
 p=str(p)
 if p not in seen:
  h=hashlib.sha256()
  with open(p,'rb') as f:
   for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
  seen[p]=h.hexdigest()
 return seen[p]
def verify(p,h):assert sha(p)==h,p
p=argparse.ArgumentParser();p.add_argument('--first-fold',action='store_true');a=p.parse_args();A=O/('first_fold_acceptance' if a.first_fold else 'formal_delivery')/'TRAINING_AUDIT.json';d=json.loads(A.read_text());assert d['status']=='pass'
expected=set(itertools.product(tr.GROUPS,[f'htt_leave_p{f}' for f in ([1] if a.first_fold else range(1,5))],[20260914,20260915,20260916]));assert {(r['group'],r['fold'],r['seed']) for r in d['accepted_runs']}==expected
cache={};records=[]
for r in d['accepted_runs']:
 root=Path(r['training_summary']['path']).parent
 for q in [r['training_summary'],r['checkpoint'],*r['predictions'].values()]:verify(q['path'],q['sha256'])
 s=json.loads((root/'summary.json').read_text());cf=json.loads((root/'config.json').read_text());assert not s['smoke'] and not cf['smoke'];assert all(s[k]==r[k]==cf[k] for k in ['group','fold','seed'])
 for k,h in s['output_hashes'].items():verify(root/k,h)
 for k,h in cf['sources'].items():verify(k,h)
 pp=cf['prepared']['path'];verify(pp,cf['prepared']['sha256'])
 if pp not in cache:cache[pp]=torch.load(pp,map_location='cpu',weights_only=False)
 data=cache[pp];role=data['roles']['train'];stage=role['stage'];idx=torch.nonzero((stage==0)|(stage==2)).flatten();by=defaultdict(lambda:defaultdict(list))
 for i in idx.tolist():by[role['episode_id'][i]][int(stage[i])].append(i)
 N=len(idx);J=len(by);w=json.loads((root/'TRAIN_WEIGHTS.json').read_text());assert w['primary_frames']==N and len(w['records'])==sum(len(v) for v in by.values())
 for x in w['records']:
  ids=by[x['episode_id']][x['stage']];v=N/(J*len(by[x['episode_id']])*len(ids));assert x['n']==len(ids) and math.isclose(v,x['frame_weight'],rel_tol=1e-12);assert math.isclose(x['total_weight'],v*len(ids),rel_tol=1e-12)
 best=torch.load(root/'best.pth',map_location='cpu',weights_only=False);latest=torch.load(root/'latest.pth',map_location='cpu',weights_only=False);hist=latest['history'];chosen=max(range(len(hist)),key=lambda i:hist[i]['validation_low_fpr_auc'])+1
 assert chosen==best['epoch']==s['best_epoch'];assert hist==s['history'] and best['history']==hist[:chosen]
 for cp in [best,latest]:
  assert cp['config']==cf
  assert len(cp['optimizer_state']['state'])==sum(len(g['params']) for g in cp['optimizer_state']['param_groups'])
  assert all(float(v['step'])==math.ceil(N/cf['batch_size'])*cp['epoch'] for v in cp['optimizer_state']['state'].values())
  assert all(torch.isfinite(v).all() for v in cp['model_state'].values())
 x=role['x'][idx].double();mean=x[:,:,:198].reshape(-1,198).mean(0);std=x[:,:,:198].reshape(-1,198).std(0,unbiased=False);mean[195:198]=x[:,5:,195:198].reshape(-1,3).mean(0);std[195:198]=x[:,5:,195:198].reshape(-1,3).std(0,unbiased=False)
 assert torch.equal(mean.float(),best['normalizer']['mean']) and torch.equal(std.clamp_min(1e-6).float(),best['normalizer']['std'])
 tr.configure(r['seed']);model=tr.TemporalHead(r['group']);initial={k:v.clone() for k,v in model.state_dict().items()};model.load_state_dict(best['model_state']);model.eval();assert all(not torch.equal(initial[k],v) for k,v in best['model_state'].items())
 for rn,q in r['predictions'].items():
  rows=list(csv.DictReader(open(q['path'])));rr=data['roles'][rn];assert len(rows)==len(rr['t'])
  for i,z in enumerate(rows):
   assert z['episode_id']==rr['episode_id'][i] and int(z['t'])==int(rr['t'][i]) and int(z['stage'])==int(rr['stage'][i]);assert z['group']==r['group'] and int(z['seed'])==r['seed'];assert math.isfinite(float(z['p_slip'])) and 0<=float(z['p_slip'])<=1
  pick=torch.linspace(0,len(rows)-1,16).long();xx=rr['x'][pick].clone();xx[:,:,:198]=(xx[:,:,:198]-mean.float())/std.clamp_min(1e-6).float();xx[:,:5,195:198]=0
  with torch.no_grad():pred=torch.sigmoid(model(xx))
  assert torch.allclose(pred,torch.tensor([float(rows[i]['p_slip']) for i in pick]),atol=1e-5,rtol=1e-5)
 records.append({'group':r['group'],'fold':r['fold'],'seed':r['seed'],'epochs':len(hist),'best_epoch':chosen,'parameter_count':sum(v.numel() for v in model.parameters()),'prediction_sample_per_role':16});print(root.name+' independently verified',flush=True)
result={'status':'pass','scope':'first fold9 only' if a.first_fold else 'all36 formal trainings','run_count':len(records),'records':records,'hashes':seen,'training_audit_sha256':sha(A),'checks':['all output and source identities','independently recomputed trial-class weights and normalizer','best/latest optimizer step and earliest selection','all raw prediction endpoint/group/seed finite identities','16 evenly-spaced checkpoint predictions per role CPU','fresh parameter tensors changed'],'limits':['Architecture forward imported from independently reviewed trainer; not a second independent implementation','All probability identities checked but checkpoint forward parity samples16 endpoints per role','No new GPU jobs, raw images or test data read']}
name='INDEPENDENT_FIRST_FOLD_RESULTS.json' if a.first_fold else 'INDEPENDENT_TRAINING_RESULTS.json';(C/'reviews'/name).write_text(json.dumps(result,indent=2));print('PASS',len(records))
