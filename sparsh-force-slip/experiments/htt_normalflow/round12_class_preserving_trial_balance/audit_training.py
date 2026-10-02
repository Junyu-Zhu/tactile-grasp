#!/usr/bin/env python3
"""Fail-closed complete-grid acceptance and downstream manifest builder."""
import argparse,csv,importlib.util,json,math
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('r9_train_audit',HERE/'training/train.py');tr=importlib.util.module_from_spec(spec);spec.loader.exec_module(tr)

def proof(path):return {'path':str(path.resolve()),'sha256':tr.sha(path)}

def run(a):
 inv=json.loads(a.inventory.read_text());out=a.output;out.mkdir(parents=True,exist_ok=True)
 expected={(g,f'htt_leave_p{f}',s) for g in tr.GROUPS for f in ([1] if a.first_fold else range(1,5)) for s in [20260914,20260915,20260916]}
 actual={(j['group'],j['fold'],j['seed']) for j in inv['jobs']}
 if len(inv['jobs'])!=len(expected) or actual!=expected:raise ValueError('formal grid')
 for name,h in inv['source_hashes'].items():
  if tr.sha(name)!=h:raise ValueError('source changed '+name)
 for x in inv['required_checks']:
  if tr.sha(x['path'])!=x['sha256'] or json.loads(Path(x['path']).read_text()).get('status') not in ('pass','complete'):raise ValueError('prerequisite drift')
 accepted=[];records=[];index=[];manifests=[];norms={};initial={}
 protocol=json.loads(tr.PROTOCOL.read_text())
 for job in inv['jobs']:
  p=Path(job['output']);summary=json.loads((p/'summary.json').read_text());config=json.loads((p/'config.json').read_text())
  if summary['status']!='complete' or summary['smoke'] or config['smoke']:raise ValueError('unaccepted/smoke run')
  if summary.get('schema')!='round12_htt_fusion_summary_v1' or config.get('schema')!='round12_htt_fusion_run_v1':raise ValueError('run schema')
  if config['protocol']!=protocol or config['max_epochs']!=protocol['max_epochs'] or config['batch_size']!=protocol['batch_size']:raise ValueError('formal protocol drift')
  required_outputs={'best.pth','latest.pth','config.json','TRAIN_WEIGHTS.json',*[f'predictions_{r}.csv' for r in tr.ROLES]}
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
   norms[key]=(tr.fit_normalizer(data['roles']['train']),tr.sha(data_path),tr.sha(data_path.with_name('audit.json')),audit,identities,tr.trial_weights(data['roles']['train']),{role:tr.normalize(v['x'][:16],tr.fit_normalizer(data['roles']['train'])) for role,v in data['roles'].items()})
  norm,data_hash,audit_hash,prepared_audit,identities,weight_info,probes=norms[key]
  weights,weight_records=weight_info
  weight_saved=json.loads((p/'TRAIN_WEIGHTS.json').read_text())
  if weight_saved['records']!=weight_records or summary['loss_weights_sha256']!=tr.sha(p/'TRAIN_WEIGHTS.json'):raise ValueError('trial weights drift')
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
  initial_state={k:v.clone() for k,v in model.state_dict().items()}
  model.load_state_dict(best['model_state'],strict=True);model.eval()
  actual_changes={k:not torch.equal(initial_state[k],v) for k,v in best['model_state'].items()}
  if actual_changes!=summary['parameter_tensors_updated'] or not all(actual_changes.values()):raise ValueError('parameter update self-report mismatch')
  total=sum(p.numel() for p in model.parameters());inactive=96 if job['group']=='F_class_trial_balanced' else 0
  if summary['parameters']!=total or summary['structurally_inactive_parameters']!=inactive or summary['effective_parameters']!=total-inactive:raise ValueError('parameter count mismatch')
  if not all(torch.isfinite(v).all() for v in best['model_state'].values()) or not all(torch.isfinite(v).all() for v in latest['model_state'].values()):raise ValueError('nonfinite model')
  for saved in [best,latest]:
   opt=saved['optimizer_state'];expected_step=math.ceil(weight_saved['primary_frames']/protocol['batch_size'])*saved['epoch']
   if any(float(v['step'])!=expected_step for v in opt['state'].values()):raise ValueError('optimizer step mismatch')
   if sum(len(g['params']) for g in opt['param_groups'])!=len(list(model.parameters())):raise ValueError('optimizer parameter count')
   for state in opt['state'].values():
    if any(torch.is_tensor(v) and not torch.isfinite(v).all() for v in state.values()):raise ValueError('nonfinite optimizer')
  for k in ('prepared_unchanged','optimizer_only_new_head','upstream_readonly_by_cache_architecture','checkpoint_roundtrip_exact'):
   if summary[k] is not True:raise ValueError('missing check '+k)
  if not all(summary['parameter_tensors_updated'].values()):raise ValueError('unupdated head')
  ga=summary['gradient_audit']
  if not ga['all_gradients_finite'] or not ga['visual_gradient_nonzero']:raise ValueError('gradient checks')
  if job['group']=='V_class_trial_balanced' and (ga['force_gradient_absmax'] or ga['delta_gradient_absmax']):raise ValueError('force leakage')
  if job['group']=='F_class_trial_balanced' and ga['delta_gradient_absmax']:raise ValueError('delta leakage')
  preds={};endpoints={}
  for role in tr.ROLES:
   pred=p/f'predictions_{role}.csv';endpoint=data_path.with_name(f'endpoints_{role}.csv');ps=list(csv.DictReader(pred.open()));es=list(csv.DictReader(endpoint.open()))
   if prepared_audit['output_hashes'].get(str(endpoint))!=tr.sha(endpoint):raise ValueError('endpoint audit identity')
   endpoint_identity=[(e['episode_id'],int(e['t']),int(e['stage']),e['leakage_group']) for e in es]
   if endpoint_identity!=identities[role]:raise ValueError('endpoint vs prepared identity')
   if len(ps)!=len(es):raise ValueError('prediction endpoint count')
   for x,y in zip(ps,es):
    if x['group']!=job['group'] or int(x['seed'])!=job['seed']:raise ValueError('prediction group/seed')
    if any(x[k]!=y[k] for k in ('episode_id','t','stage','role','fold','leakage_group')):raise ValueError('prediction endpoint identity')
    value=float(x['p_slip'])
    if not math.isfinite(value) or not 0<=value<=1:raise ValueError('prediction finite range')
   probe_pred=tr.infer(model,probes[role],'cpu').double()
   if not torch.allclose(probe_pred,torch.tensor([float(x['p_slip']) for x in ps[:16]],dtype=torch.float64),atol=1e-5,rtol=1e-5):raise ValueError('checkpoint prediction mismatch')
   if len({x['p_slip'] for x in ps})<2:raise ValueError('constant model')
   preds[role]=proof(pred);endpoints[role]=proof(endpoint)
  rec={k:job[k] for k in ('group','fold','seed')};rec.update(training_summary=proof(p/'summary.json'),checkpoint=proof(p/'best.pth'),predictions=preds)
  accepted.append(rec);manifests.append({**rec,'endpoints':{k:v for k,v in endpoints.items() if k!='train'},'historical':False})
  records.append({'id':job['id'],'status':'pass','epochs':len(hist),'best_epoch':summary['best_epoch'],'parameters':summary['parameters'],'effective_parameters':summary['effective_parameters']})
  for name in ('best.pth','latest.pth','config.json'):index.append({**{k:job[k] for k in ('group','fold','seed')},'artifact':name,**proof(p/name)})
  print(job['id']+' pass',flush=True)
 # Historical A/B initialization is exact; no residual group is authorized.
 oldspec=importlib.util.spec_from_file_location('r10_init_audit',HERE.parent/'round10_htt_force_supervision_adaptation/fusion/train.py')
 old=importlib.util.module_from_spec(oldspec);oldspec.loader.exec_module(old)
 for seed in protocol['seeds']:
  for group,historical in [('V_class_trial_balanced','V_temporal'),('F_class_trial_balanced','F_history')]:
   tr.configure(seed);new=tr.TemporalHead(group);old.configure(seed);prior=old.TemporalHead(historical)
   if tr.state_hash(new.state_dict())!=tr.state_hash(prior.state_dict()):raise ValueError('historical init mismatch')
 audit={'schema':'round12_fusion_training_audit_v1','status':'pass','expected_count':len(expected),'test_role_consumed':False,'accepted_runs':accepted,'records':records,'formal_inventory':proof(a.inventory),'same_historical_AB_initialization':True,'smoke_real_recovery_bound_in_inventory':True,'source':proof(Path(__file__).resolve())}
 tr.atomic_json(out/'TRAINING_AUDIT.json',audit)
 with (out/'CHECKPOINT_INDEX.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(index[0]));w.writeheader();w.writerows(index)
 tr.atomic_json(out/'RUN_INVENTORY.json',{'status':'complete','runs':records,'accepted_count':len(expected)})
 print(json.dumps({'status':'pass','accepted_runs':len(expected)}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--first-fold',action='store_true');p.add_argument('--inventory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
