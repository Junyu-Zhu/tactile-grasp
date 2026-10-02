#!/usr/bin/env python3
"""Fail-closed 36-run acceptance and downstream manifest builder."""
import argparse,csv,importlib.util,json,math
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('r9_train_audit',HERE/'fusion/train.py');tr=importlib.util.module_from_spec(spec);spec.loader.exec_module(tr)

def proof(path):return {'path':str(path.resolve()),'sha256':tr.sha(path)}

def run(a):
 inv=json.loads(a.inventory.read_text());out=a.output;out.mkdir(parents=True,exist_ok=True)
 expected={(g,f'htt_leave_p{f}',s) for g in ['F_history'] for f in range(1,5) for s in [20260914,20260915,20260916]}
 actual={(j['group'],j['fold'],j['seed']) for j in inv['jobs']}
 if len(inv['jobs'])!=12 or actual!=expected:raise ValueError('formal grid')
 for name,h in inv['source_hashes'].items():
  if tr.sha(name)!=h:raise ValueError('source changed '+name)
 for x in inv['required_checks']:
  if tr.sha(x['path'])!=x['sha256'] or json.loads(Path(x['path']).read_text()).get('status') not in ('pass','complete'):raise ValueError('prerequisite drift')
 accepted=[];records=[];index=[];manifests=[];norms={};initial={}
 protocol=json.loads(tr.PROTOCOL.read_text())
 for job in inv['jobs']:
  p=Path(job['output']);summary=json.loads((p/'summary.json').read_text());config=json.loads((p/'config.json').read_text())
  if summary['status']!='complete' or summary['smoke'] or config['smoke']:raise ValueError('unaccepted/smoke run')
  if summary.get('schema')!='round10_htt_fusion_summary_v1' or config.get('schema')!='round10_htt_fusion_run_v1':raise ValueError('run schema')
  if config['protocol']!=protocol or config['max_epochs']!=protocol['max_epochs'] or config['batch_size']!=protocol['batch_size']:raise ValueError('formal protocol drift')
  required_outputs={'best.pth','latest.pth','config.json',*[f'predictions_{r}.csv' for r in tr.ROLES]}
  if set(summary['output_hashes'])!=required_outputs:raise ValueError('incomplete output hash grid')
  if any(summary[k]!=job[k] or config[k]!=job[k] for k in ['group','fold','seed']):raise ValueError('run identity')
  if tr.sha(p/'config.json')!=summary['config_sha256']:raise ValueError('config hash')
  for name,h in summary['output_hashes'].items():
   if tr.sha(p/name)!=h:raise ValueError('output hash '+str(p/name))
  for name,h in config['sources'].items():
   if tr.sha(name)!=h or inv['source_hashes'].get(name)!=h:raise ValueError('run source binding')
  data_path=Path(job['data']);key=str(data_path)
  if key not in norms:
   data=torch.load(data_path,map_location='cpu',weights_only=False)
   audit=json.loads(data_path.with_name('audit.json').read_text());tr.validate_data(data,audit,data_path,job['fold'],job['seed'])
   rep=data.get('force_replacement_audit',{})
   if data.get('experiment')!='round10_new_force' or rep.get('pass') is not True or rep.get('smoke') is not False:raise ValueError('invalid formal force replacement')
   if audit.get('replacement')!=rep:raise ValueError('replacement audit mismatch')
   for rec in rep['proofs']:
    if tr.sha(rec['path'])!=rec['sha256']:raise ValueError('replacement source drift')
   identities={r:[(eid,int(t),int(s),g) for eid,t,s,g in zip(v['episode_id'],v['t'],v['stage'],v['leakage_group'])] for r,v in data['roles'].items()}
   norms[key]=(tr.fit_normalizer(data['roles']['train']),tr.sha(data_path),tr.sha(data_path.with_name('audit.json')),audit,identities)
  norm,data_hash,audit_hash,prepared_audit,identities=norms[key]
  if config['prepared']!=proof(data_path) or config['prepared_audit']!=proof(data_path.with_name('audit.json')):raise ValueError('data binding')
  best=torch.load(p/'best.pth',map_location='cpu',weights_only=False);latest=torch.load(p/'latest.pth',map_location='cpu',weights_only=False)
  if best['config']!=config or latest['config']!=config:raise ValueError('checkpoint config')
  if any(not torch.equal(best['normalizer'][k],norm[k]) or not torch.equal(latest['normalizer'][k],norm[k]) for k in norm):raise ValueError('normalizer not fit-only')
  hist=latest['history'];first=max(range(len(hist)),key=lambda i:hist[i]['validation_low_fpr_auc'])
  if hist!=summary['history'] or first+1!=summary['best_epoch'] or best['epoch']!=first+1 or latest['epoch']!=len(hist):raise ValueError('checkpoint selection')
  if best['history']!=hist[:first+1] or best['best_epoch']!=first+1 or latest['best_epoch']!=first+1 or best['best_score']!=summary['best_score'] or latest['best_score']!=summary['best_score']:raise ValueError('best/latest history inconsistency')
  if len(hist)>protocol['max_epochs'] or [r['epoch'] for r in hist]!=list(range(1,len(hist)+1)):raise ValueError('epoch sequence')
  if summary['best_score']!=hist[first]['validation_low_fpr_auc'] or not all(math.isfinite(r[k]) for r in hist for k in ('train_loss','validation_low_fpr_auc')):raise ValueError('finite history')
  tr.configure(job['seed']);model=tr.TemporalHead(job['group']);init=tr.state_hash(model.state_dict())
  if init!=summary['initial_state_sha256'] or init!=best['initial_state_sha256'] or init!=latest['initial_state_sha256']:raise ValueError('initialization identity')
  initial[(job['fold'],job['seed'],job['group'])]=init
  model.load_state_dict(best['model_state'],strict=True)
  if not all(torch.isfinite(v).all() for v in best['model_state'].values()) or not all(torch.isfinite(v).all() for v in latest['model_state'].values()):raise ValueError('nonfinite model')
  for opt in [best['optimizer_state'],latest['optimizer_state']]:
   if sum(len(g['params']) for g in opt['param_groups'])!=len(list(model.parameters())):raise ValueError('optimizer parameter count')
   for state in opt['state'].values():
    if any(torch.is_tensor(v) and not torch.isfinite(v).all() for v in state.values()):raise ValueError('nonfinite optimizer')
  for k in ('prepared_unchanged','optimizer_only_new_head','upstream_readonly_by_cache_architecture','checkpoint_roundtrip_exact'):
   if summary[k] is not True:raise ValueError('missing check '+k)
  if not all(summary['parameter_tensors_updated'].values()):raise ValueError('unupdated head')
  ga=summary['gradient_audit']
  if not ga['all_gradients_finite'] or not ga['visual_gradient_nonzero']:raise ValueError('gradient checks')
  if job['group']=='V_temporal' and (ga['force_gradient_absmax'] or ga['delta_gradient_absmax']):raise ValueError('force leakage')
  if job['group']=='F_history' and ga['delta_gradient_absmax']:raise ValueError('delta leakage')
  preds={};endpoints={}
  for role in tr.ROLES:
   pred=p/f'predictions_{role}.csv';endpoint=data_path.with_name(f'endpoints_{role}.csv');ps=list(csv.DictReader(pred.open()));es=list(csv.DictReader(endpoint.open()))
   if prepared_audit['output_hashes'].get(str(endpoint))!=tr.sha(endpoint):raise ValueError('endpoint audit identity')
   endpoint_identity=[(e['episode_id'],int(e['t']),int(e['stage']),e['leakage_group']) for e in es]
   if endpoint_identity!=identities[role]:raise ValueError('endpoint vs prepared identity')
   if len(ps)!=len(es):raise ValueError('prediction endpoint count')
   for x,y in zip(ps,es):
    if any(x[k]!=y[k] for k in ('episode_id','t','stage','role','fold','leakage_group')):raise ValueError('prediction endpoint identity')
    value=float(x['p_slip'])
    if not math.isfinite(value) or not 0<=value<=1:raise ValueError('prediction finite range')
   if len({x['p_slip'] for x in ps})<2:raise ValueError('constant model')
   preds[role]=proof(pred);endpoints[role]=proof(endpoint)
  rec={k:job[k] for k in ('group','fold','seed')};rec['group']='F_history_new';rec.update(training_summary=proof(p/'summary.json'),checkpoint=proof(p/'best.pth'),predictions=preds)
  accepted.append(rec);manifests.append({**rec,'endpoints':{k:v for k,v in endpoints.items() if k!='train'},'historical':False})
  records.append({'id':job['id'],'status':'pass','epochs':len(hist),'best_epoch':summary['best_epoch'],'parameters':summary['parameters'],'effective_parameters':summary['effective_parameters']})
  for name in ('best.pth','latest.pth','config.json'):index.append({**{k:job[k] for k in ('group','fold','seed')},'artifact':name,**proof(p/name)})
  print(job['id']+' pass',flush=True)
 # Same seed exact initialization and numeric budget must match inherited old-force baseline.
 r9root=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round9_htt_temporal_force_fusion')
 r9audit=json.loads((r9root/'formal_delivery/TRAINING_AUDIT.json').read_text())
 r9sync=json.loads((HERE.parent/'round9_htt_temporal_force_fusion/LOCAL_SYNC_PROOF.json').read_text())
 expected=[r for r in r9sync['origin_files'] if r['origin']==str(r9root/'formal_delivery/TRAINING_AUDIT.json')]
 if len(expected)!=1 or expected[0]['sha256']!=tr.sha(r9root/'formal_delivery/TRAINING_AUDIT.json') or r9audit['status']!='pass':raise ValueError('R9 audit identity')
 for job in inv['jobs']:
  oldpath=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round9_htt_temporal_force_fusion/formal')/f"F_history_p{job['fold'][-1]}_s{job['seed']}"
  record=next(r for r in r9audit['accepted_runs'] if r['group']=='F_history' and r['fold']==job['fold'] and r['seed']==job['seed'])
  if record['training_summary']!=proof(oldpath/'summary.json') or record['checkpoint']!=proof(oldpath/'best.pth'):raise ValueError('old R9 accepted model identity')
  old=json.loads((oldpath/'summary.json').read_text())
  if old['output_hashes']['config.json']!=tr.sha(oldpath/'config.json'):raise ValueError('old R9 config identity')
  oldconfig=json.loads((oldpath/'config.json').read_text())
  if old['initial_state_sha256']!=initial[(job['fold'],job['seed'],'F_history')] or oldconfig['protocol']!=protocol:raise ValueError('R9 init/protocol mismatch')
 audit={'schema':'round10_fusion_training_audit_v1','status':'pass','expected_count':12,'test_role_consumed':False,'accepted_runs':accepted,'records':records,'formal_inventory':proof(a.inventory),'same_R9_F_history_initialization':True,'smoke_real_recovery_bound_in_inventory':True,'source':proof(Path(__file__).resolve())}
 tr.atomic_json(out/'TRAINING_AUDIT.json',audit)
 with (out/'CHECKPOINT_INDEX.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(index[0]));w.writeheader();w.writerows(index)
 tr.atomic_json(out/'RUN_INVENTORY.json',{'status':'complete','runs':records,'accepted_count':12})
 print(json.dumps({'status':'pass','accepted_runs':12}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--inventory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
