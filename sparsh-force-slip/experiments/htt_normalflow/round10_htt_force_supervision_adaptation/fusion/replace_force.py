#!/usr/bin/env python3
"""Replace only predicted-force columns; exact R9 visual and endpoint identity."""
import argparse,copy,importlib.util,json,shutil
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('r10_fusion',HERE/'train.py');tr=importlib.util.module_from_spec(spec);spec.loader.exec_module(tr)
def proof(p):return {'path':str(Path(p).resolve()),'sha256':tr.sha(p)}
def main(a):
 old=torch.load(a.r9_data,map_location='cpu',weights_only=False);oa=json.loads(a.r9_data.with_name('audit.json').read_text());tr.validate_data(old,oa,a.r9_data,old['fold'],old['seed'])
 m=json.loads(a.predictions.read_text())
 if m.get('status')!='complete' or m['fold']!=old['fold'] or m['seed']!=old['seed']:raise ValueError('prediction manifest identity')
 if bool(m['smoke'])!=a.smoke:raise ValueError('smoke identity')
 proofs=[proof(a.r9_data),proof(a.r9_data.with_name('audit.json')),proof(a.predictions)]
 for name in ['force_checkpoint','force_summary','force_config']:
  p=m[name]
  if tr.sha(p['path'])!=p['sha256']:raise ValueError('force artifact identity')
  proofs.append(p)
 summary=json.loads(Path(m['force_summary']['path']).read_text())
 if summary['status']!='complete' or bool(summary['smoke'])!=a.smoke:raise ValueError('unaccepted force')
 config=json.loads(Path(m['force_config']['path']).read_text())
 for item in [summary,config]:
  if item['fold']!=old['fold'] or item['seed']!=old['seed'] or bool(item['smoke'])!=a.smoke:raise ValueError('force run identity mismatch')
 for name in ['force_checkpoint','force_config']:
  rec=m[name]
  if summary['output_hashes'].get(rec['path'],summary['output_hashes'].get(Path(rec['path']).name))!=rec['sha256']:raise ValueError('checkpoint/config not accepted by force summary')
 mapping={e['episode_id']:e for e in m['entries']}
 if len(mapping)!=len(m['entries']) or set(mapping)!={e['episode_id'] for e in old['episodes']}:raise ValueError('prediction episode coverage')
 data=copy.deepcopy(old);diagnostics=[]
 for e in data['episodes']:
  eid=e['episode_id'];r=mapping[eid];fp=Path(r['prediction_path'])
  if r['role']!=e['role']:raise ValueError('force prediction role mismatch')
  if tr.sha(fp)!=r['prediction_sha256']:raise ValueError('prediction hash')
  f=np.load(fp,allow_pickle=False)
  if f.shape!=tuple(e['force'].shape) or not np.isfinite(f).all() or len(f)!=r['frames']:raise ValueError('prediction shape/finite')
  e['force']=torch.from_numpy(f.astype(np.float32));role=data['roles'][e['role']]
  ix=torch.tensor([i for i,x in enumerate(role['episode_id']) if x==eid]);t=role['t'][ix];hist=t[:,None]+torch.arange(-8,1)[None,:]
  role['x'][ix,:,192:195]=e['force'][hist]
  role['x'][ix,5:,195:198]=e['force'][hist[:,5:]]-e['force'][hist[:,5:]-5]
  diagnostics.append({'episode_id':eid,'role':e['role'],'frames':len(f),'prediction':proof(fp)})
  proofs.append(proof(fp))
 for role in tr.ROLES:
  r=data['roles'][role];original=old['roles'][role]
  if not torch.equal(r['x'][:,:,:192],original['x'][:,:,:192]) or not torch.equal(r['x'][:,:,198],original['x'][:,:,198]):raise ValueError('visual/valid changed')
  for k in ['t','stage']:
   if not torch.equal(r[k],original[k]):raise ValueError('endpoint changed')
  for k in ['episode_id','leakage_group','probe']:
   if r[k]!=original[k]:raise ValueError('role identity changed')
 data['experiment']='round10_new_force';data['force_replacement_audit']={'pass':True,'smoke':a.smoke,'proofs':proofs,'all_visual_and_endpoints_exact':True,'force_from_predictions_only':True,'no_future_input':True,'source':proof(Path(__file__))}
 data['provenance']['round9_prepared']=proof(a.r9_data);data['provenance']['force_checkpoint']=m['force_checkpoint'];data['provenance']['force_prediction_manifest']=proof(a.predictions)
 a.output.mkdir(parents=True,exist_ok=True);dst=a.output/'prepared.pt'
 if dst.exists():
  existing=json.loads((a.output/'audit.json').read_text())
  if existing['replacement']!=data['force_replacement_audit']:raise ValueError('existing replacement provenance changed')
  tr.validate_data(torch.load(dst,map_location='cpu',weights_only=False),existing,dst,data['fold'],data['seed']);print('reused');return
 tr.atomic_save(dst,data);hashes={str(dst.resolve()):tr.sha(dst)}
 for role in tr.ROLES:
  source=a.r9_data.with_name(f'endpoints_{role}.csv');target=a.output/source.name;shutil.copyfile(source,target);hashes[str(target.resolve())]=tr.sha(target)
 audit={'status':'pass','fold':data['fold'],'seed':data['seed'],'output_hashes':hashes,'replacement':data['force_replacement_audit'],'counts':oa['counts'],'prediction_entries':diagnostics,'source':proof(Path(__file__))}
 tr.atomic_json(a.output/'audit.json',audit);tr.validate_data(data,audit,dst,data['fold'],data['seed']);print(json.dumps({'status':'pass','episodes':len(diagnostics)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--r9-data',type=Path,required=True);p.add_argument('--predictions',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');main(p.parse_args())
