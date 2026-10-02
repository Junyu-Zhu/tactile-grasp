#!/usr/bin/env python3
"""Accepted HTT image-to-current-slip cold/streaming benchmark."""
from __future__ import annotations
import argparse,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
os.environ.setdefault('XFORMERS_DISABLED','1');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;R9=HERE.parent;EXP=R9.parent
sys.path.insert(0,str(R9/'fusion'));import train
sys.path.insert(0,str(EXP/'round5_force_conditioned_slip/current'));from sources import load_r3_slip_branch
from models import FrozenSlipBranch,ForceAdapter
import sources,models,common
sys.path.insert(0,str(EXP));import adapters
p2=sources.p2

def verify(rec):
 p=Path(rec['path']);assert p.is_file() and train.sha(p)==rec['sha256'],str(p);return p

def assemble(bases,norm,group):
 raw=torch.zeros((1,9,199),device=bases[0][0].device)
 raw[:,:,:192]=torch.stack([b[0] for b in bases],1)
 if group!='V_temporal':
  raw[:,:,192:195]=torch.stack([b[1] for b in bases],1)
  raw[:,5:,195:198]=raw[:,5:,192:195]-raw[:,:4,192:195]
 raw[:,5:,198]=1
 return raw,train.normalize(raw,norm)

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--accepted-audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
 protocol=json.loads((HERE/'protocol.json').read_text());parity_source=EXP/'round8_force_dynamics_event_time/ANALYSIS_PROTOCOL.json';assert json.loads(parity_source.read_text())['e2e_parity_limits']==protocol['parity_limits'];train.configure(protocol['seed']);common.configure_determinism()
 audit=json.loads(a.accepted_audit.read_text());assert audit['schema']=='round10_fusion_training_audit_v1' and audit['status']=='pass' and audit['expected_count']==12 and audit['test_role_consumed'] is False
 assert len(audit['accepted_runs'])==12 and {(r['group'],r['fold'],r['seed']) for r in audit['accepted_runs']}=={('F_history_new',f'htt_leave_p{f}',s) for f in range(1,5) for s in [20260914,20260915,20260916]}
 ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);cfg=ck['config'];group=cfg['group'];assert group=='F_history';assert not cfg['smoke'] and cfg['fold']==protocol['fold'] and cfg['seed']==protocol['seed']
 matches=[r for r in audit['accepted_runs'] if r['group']=='F_history_new' and r['fold']==cfg['fold'] and r['seed']==cfg['seed']];assert len(matches)==1 and verify(matches[0]['checkpoint']).resolve()==a.checkpoint.resolve()
 for path,h in cfg['sources'].items():assert train.sha(path)==h
 assert verify(cfg['prepared']).resolve()==a.data.resolve();data=torch.load(a.data,map_location='cpu',weights_only=False)
 assert data.get('experiment')=='round10_new_force' and data['force_replacement_audit']['smoke'] is False
 for rec in data['force_replacement_audit']['proofs']:verify(rec)
 da=json.loads(verify(cfg['prepared_audit']).read_text());train.validate_data(data,da,a.data,cfg['fold'],cfg['seed'])
 provenance=data['provenance'];source=verify(provenance['source_checkpoint']);bp=verify(provenance['visual_checkpoint']);fp=verify(provenance['force_checkpoint'])
 r=data['roles']['validation'];idx=min(range(len(r['t'])),key=lambda i:(r['episode_id'][i],int(r['t'][i])));eid=r['episode_id'][idx];t=int(r['t'][idx]);assert t==13
 ep=next(e for e in data['episodes'] if e['episode_id']==eid);assert ep['role']=='validation'
 # Read only image/reference arrays of one allowed trajectory, outside timing.
 with np.load(ep['source_path'],allow_pickle=False) as raw:
  images=raw['tactile_img'];reference=raw['ref_frame']
 assert len(images)==len(ep['stage']) and np.isfinite(reference).all()
 # Preserve accepted raw source hash identity without inspecting any other episode.
 split=json.loads(verify(provenance['split']).read_text());source_row=next(e for e in split['episodes'] if e['id']==eid);assert eid in split['splits'][cfg['fold']]['validation']
 source_hash=source_row['source_files'][ep['source_path']];assert train.sha(ep['source_path'])==source_hash
 parent,_=p2.load_b_checkpoint(source,torch.device('cpu'));encoder=parent.encoder;visual=FrozenSlipBranch(load_r3_slip_branch(parent.decoder,bp))
 force=None;force_norm=None
 if group!='V_temporal':
  fc=torch.load(fp,map_location='cpu',weights_only=False);assert fc['config']['fold']==cfg['fold'] and fc['config']['seed']==cfg['seed'] and not fc['config']['smoke']
  force=ForceAdapter(parent.decoder);force.load_state_dict(fc['model_state']);force_norm={k:torch.as_tensor(fc['normalization'][k],device=a.device,dtype=torch.float32) for k in ('mean','std')}
 del parent
 head=train.TemporalHead(group);head.load_state_dict(ck['model_state']);norm={k:v.to(a.device) for k,v in ck['normalizer'].items()}
 modules={'encoder':encoder,'visual':visual,'head':head}
 if force is not None:modules['force']=force
 for m in modules.values():m.eval().requires_grad_(False).to(a.device)
 initial={n:train.state_hash(m.state_dict()) for n,m in modules.items()}
 gpu=lambda:subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv'],text=True) if a.device.startswith('cuda') else 'CPU'
 before=gpu()
 def base(s):
  assert s>=5
  image=torch.cat([adapters.preprocess(images[j],reference) for j in (s,s-5)]).unsqueeze(0).to(a.device)
  token=encoder(image);v=visual.forward_features(token)
  f=None if force is None else force(token)*force_norm['std']+force_norm['mean']
  return v,f
 def sync():
  if a.device.startswith('cuda'):torch.cuda.synchronize(a.device)
 with torch.inference_mode():
  bases=[base(s) for s in range(t-8,t+1)];raw,fresh=assemble(bases,norm,group);cache_raw=r['x'][idx:idx+1].to(a.device);cache=train.normalize(cache_raw,norm)
  # V does not read force inputs; mask unused force slots solely for input identity comparison.
  limits=protocol['parity_limits'];comparisons={}
  for name,start,end in [('visual',0,192),('force_xyz',192,195),('delta_xyz',195,198),('valid',198,199)]:
   if group=='V_temporal' and name in ('force_xyz','delta_xyz'):continue
   for kind,left,right in [('raw',raw,cache_raw),('normalized',fresh,cache)]:
    aa=left[:,:,start:end];bb=right[:,:,start:end];comparisons[f'{kind}_{name}']={'max_abs':float((aa-bb).abs().max()),'mean_abs':float((aa-bb).abs().mean()),'pass':bool(torch.allclose(aa,bb,atol=limits['input_atol'],rtol=limits['input_rtol']))}
  risk_error=float((torch.sigmoid(head(fresh))-torch.sigmoid(head(cache))).abs().max());parity={'blocks':comparisons,'risk_max_abs':risk_error,'limits':limits,'pass':all(x['pass'] for x in comparisons.values()) and risk_error<=limits['risk_max_abs']}
  if not parity['pass']:
   train.atomic_json(a.output,{'status':'failed_cache_parity','group':group,'parity':parity});raise RuntimeError('Fresh/cache mismatch; no benchmark accepted')
  measurements={}
  for mode in protocol['modes']:
   reps=[]
   for rep in range(protocol['repeats']):
    if a.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats(a.device)
    times=[]
    for step in range(protocol['warmup']+protocol['steps']):
     sync();start=time.perf_counter();current=[base(s) for s in range(t-8,t+1)] if mode=='cold_history' else bases[:-1]+[base(t)]
     _,xx=assemble(current,norm,group);prob=torch.sigmoid(head(xx));sync();elapsed=(time.perf_counter()-start)*1000
     if not torch.isfinite(prob).all():raise RuntimeError('nonfinite probability')
     if step>=protocol['warmup']:times.append(elapsed)
    reps.append({'repeat':rep,'samples_ms':times,'median_ms':float(np.median(times)),'p10_ms':float(np.quantile(times,.1)),'p90_ms':float(np.quantile(times,.9)),'peak_allocated_bytes':torch.cuda.max_memory_allocated(a.device) if a.device.startswith('cuda') else None})
   measurements[mode]=reps
 frozen={n:initial[n]==train.state_hash(m.state_dict()) for n,m in modules.items()};assert all(frozen.values())
 result={'status':'complete','group':group,'fold':cfg['fold'],'seed':cfg['seed'],'episode_id':eid,'t':t,'raw_image_range':[t-13,t],'role':'validation','parity':parity,'frozen':frozen,'force_executed':force is not None,'parameter_counts':{n:sum(p.numel() for p in m.parameters()) for n,m in modules.items()},'total_deployment_parameters':sum(p.numel() for m in modules.values() for p in m.parameters()),'measurements':measurements,'protocol':protocol,'protocol_sha256':train.sha(HERE/'protocol.json'),'parity_source_sha256':train.sha(parity_source),'accepted_audit_sha256':train.sha(a.accepted_audit),'checkpoint_sha256':train.sha(a.checkpoint),'prepared_sha256':train.sha(a.data),'source_npz_sha256':source_hash,'upstream':provenance,'source_hashes':{str(Path(m.__file__).resolve()):train.sha(m.__file__) for m in (train,sources,models,common,adapters,p2)},'benchmark_source_sha256':train.sha(__file__),'device':a.device,'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'gpu_name':torch.cuda.get_device_name(a.device) if a.device.startswith('cuda') else None,'gpu_before':before,'gpu_after':gpu()}
 train.atomic_json(a.output,result);print(json.dumps({'status':'complete','group':group,'parity':parity}),flush=True)
if __name__=='__main__':main()
