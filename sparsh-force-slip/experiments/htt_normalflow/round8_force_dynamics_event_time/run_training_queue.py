#!/usr/bin/env python3
"""Root-owned bounded shared-GPU queue; validates parents and completed receipts."""
import argparse,datetime,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def save(p,d):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)

def main():
 p=argparse.ArgumentParser();p.add_argument('--inventory',type=Path,required=True);p.add_argument('--execute-formal',action='store_true');p.add_argument('--max-parallel',type=int,default=3);p.add_argument('--gpus',default='0,1,2');a=p.parse_args()
 if not a.execute_formal:raise ValueError('Formal execution flag required')
 inv=json.loads(a.inventory.read_text());out=Path(inv['output_root']);out.mkdir(parents=True,exist_ok=True)
 lock=(out/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 expected={(g,s) for g in inv['groups'] for s in [20260914,20260915,20260916]}
 allowed={'B_xyz','C_xyz_delta','C_hazard','P3_concat_independent','P3_fusion_independent','P3_fusion_hazard','P4_direct','P4_state'}
 if not set(inv['groups']).issubset(allowed) or len(inv['runs'])>24:raise ValueError('Invalid group scope')
 if len(inv['runs'])!=len(expected) or {(x['group'],x['seed']) for x in inv['runs']}!=expected or len({x['output'] for x in inv['runs']})!=len(expected) or any(x['id']!=f"future_{x['group']}_{x['seed']}" for x in inv['runs']):raise ValueError('Expected exact planned group/seed identities')
 decision=json.loads(Path(inv['support_decision']).read_text())
 if decision.get('training_triggered') is not True:raise ValueError('Support gate did not pass')
 if any(g.startswith('P3_') for g in inv['groups']) and not decision.get('P3_triggered'):raise ValueError('P3 gate failed')
 if any(g.startswith('P4_') for g in inv['groups']) and not decision.get('P4_triggered'):raise ValueError('P4 gate failed')
 for raw,digest in inv['frozen_inputs'].items():
  if sha(Path(raw))!=digest:raise ValueError('Changed frozen input '+raw)
 smoke=json.loads(Path(inv['smoke_audit']).read_text())
 if smoke['status']!='pass':raise ValueError('Smoke audit not accepted')
 if not set(inv['groups']).issubset(smoke.get('groups',[])):raise ValueError('Planned group absent from smoke')
 if smoke['real_prepared_sha256']!=inv['prepared_sha256'] or smoke['trainer_sha256']!=sha(Path(inv['trainer'])) or smoke['training_protocol_sha256']!=sha(Path(inv['trainer']).with_name('protocol.json')):raise ValueError('Smoke input/code identity mismatch')
 state_path=out/'QUEUE_STATE.json';state=json.loads(state_path.read_text()) if state_path.exists() else {'status':'running','jobs':{},'attempts':{},'adjustments':[]}
 def verify(run):
  s=Path(run['output'])/'summary.json'
  if not s.exists():return False
  d=json.loads(s.read_text())
  if d.get('schema')!='round8_future_timeline_v1':raise ValueError('Wrong prediction summary schema')
  if d.get('status')!='complete' or d.get('formal') is not True:return False
  if d.get('group',d.get('variant'))!=run['group'] or d.get('seed')!=run['seed']:raise ValueError('Result identity mismatch')
  if d['run_config']['mode']!='formal' or d['run_config']['group']!=run['group'] or d['run_config']['seed']!=run['seed']:raise ValueError('Config identity mismatch')
  rc=d['run_config']
  if rc.get('schema')!='round8_force_dynamics_training_v1' or rc.get('protocol_sha')!=sha(Path(inv['trainer']).with_name('protocol.json')):raise ValueError('Wrong schema/protocol')
  if not rc.get('source_hashes') or any(sha(Path(p))!=h for p,h in rc['source_hashes'].items()):raise ValueError('Completed run source drift')
  expected_code=hashlib.sha256(json.dumps(rc['source_hashes'],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
  if rc.get('code_sha')!=expected_code:raise ValueError('Wrong source bundle identity')
  expected_head='discrete_hazard' if run['group'] in ('C_hazard','P3_fusion_hazard') else 'independent_sigmoid'
  if rc.get('head_type')!=expected_head or d.get('head_type')!=expected_head:raise ValueError('Wrong head identity')
  required=inv['required_run_checks']
  if not all(d['audit'].get(k) is True for k in required):raise ValueError('Required numeric/boundary check failed')
  for key in ['best','latest']:
   x=d['artifacts'][key]
   if sha(Path(x['path']))!=x['sha256']:raise ValueError('Artifact changed')
  if d['run_config']['horizons']!=inv['horizons'] or d['run_config']['prepared_sha']!=inv['prepared_sha256']:raise ValueError('Wrong horizons/data')
  for role,pops in d['artifacts']['predictions'].items():
   for pop,x in pops.items():
    if x.get('schema')!='round8_future_timeline_v1':raise ValueError('Wrong artifact schema')
    if sha(Path(x['path']))!=x['sha256']:raise ValueError('Prediction artifact changed')
  return True
 pending=[]
 for run in inv['runs']:
  if verify(run):state['jobs'][run['id']]='complete'
  else:pending.append(run);state['jobs'][run['id']]='pending'
 running={};gpuids=a.gpus.split(',');parallel=min(a.max_parallel,len(gpuids));deadline=datetime.datetime.fromisoformat('2026-09-23T01:55:53+08:00');logroot=out/'logs';logroot.mkdir(exist_ok=True)
 while pending or running:
  while pending and len(running)<parallel and not any(v=='failed' for v in state['jobs'].values()):
   if datetime.datetime.now(datetime.timezone.utc)>=deadline:
    state['status']='deadline_no_new_dispatch';save(state_path,state);raise RuntimeError('Last two days: no new training')
   used={x['gpu'] for x in running.values()};gpu=next(g for g in gpuids if g not in used);run=pending.pop(0);rid=run['id'];attempt=state['attempts'].get(rid,0)+1;state['attempts'][rid]=attempt
   command=[sys.executable,inv['trainer'],'--data',inv['prepared_data'],'--group',run['group'],'--seed',str(run['seed']),'--output',run['output'],'--device','cuda:0','--execute-formal']
   if (Path(run['output'])/'latest.pth').exists():command.append('--resume')
   elif Path(run['output']).exists() and any(Path(run['output']).iterdir()):
    rejected=Path(run['output']+f'.rejected_before_epoch_attempt{attempt}')
    if rejected.exists():raise ValueError('Refusing rejected-attempt overwrite')
    Path(run['output']).rename(rejected)
    state['adjustments'].append({'id':rid,'reason':'no restorable epoch; preserve failed own artifacts and restart same initialization','preserved':str(rejected)})
   logpath=logroot/f'{rid}.attempt{attempt}.log';stream=logpath.open('w');env=os.environ.copy();env.update({'CUDA_VISIBLE_DEVICES':gpu,'XFORMERS_DISABLED':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'})
   proc=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env);running[rid]={'proc':proc,'stream':stream,'run':run,'gpu':gpu,'log':logpath,'command':command,'start':time.time()};state['jobs'][rid]='running';save(state_path,state)
  time.sleep(1)
  for rid,x in list(running.items()):
   rc=x['proc'].poll()
   if rc is None:continue
   x['stream'].close();del running[rid];run=x['run'];receipt={'id':rid,'argv':x['command'],'gpu':x['gpu'],'started_unix':x['start'],'finished_unix':time.time(),'exit_code':rc,'log':str(x['log']),'inventory_sha256':sha(a.inventory)}
   if rc==0 and verify(run):state['jobs'][rid]='complete';receipt['status']='complete'
   elif 'out of memory' in x['log'].read_text().lower() and state['attempts'][rid]<3:
    parallel=1;pending.append(run);state['jobs'][rid]='oom_retry_pending';state['adjustments'].append({'id':rid,'reason':'OOM','max_parallel':1});receipt['status']='oom_retry'
   else:
    state['jobs'][rid]='failed';receipt['status']='failed'
   save(out/f'{rid}.receipt.attempt{state["attempts"][rid]}.json',receipt);save(state_path,state)
  if any(v=='failed' for v in state['jobs'].values()) and not running:
   state['status']='failed';save(state_path,state);raise RuntimeError('Repair failed runs before resume')
 state['status']='complete';save(state_path,state);print(json.dumps({'status':'complete','runs':len(state['jobs'])}))
if __name__=='__main__':main()
