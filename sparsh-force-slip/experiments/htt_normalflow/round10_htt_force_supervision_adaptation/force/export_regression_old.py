#!/usr/bin/env python3
"""Export selected frozen force predictions in native N from complete MAE tokens."""
import argparse,json,sys,csv
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'round5_force_conditioned_slip/current'))
from common import sha256_file as _sha,atomic_json,tensor_state_sha256,configure_determinism
from models import ForceAdapter
from sources import load_decoupled_decoder

def sha(p):return _sha(Path(p))

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--batch-size',type=int,default=128);p.add_argument('--regression',action='store_true');p.add_argument('--old-r5',action='store_true');a=p.parse_args()
 configure_determinism();a.output.mkdir(parents=True,exist_ok=True)
 sp=a.run/'training_summary.json';su=json.loads(sp.read_text());cp=a.run/'config.json';ck=a.run/'best.pth'
 assert su['status']=='complete' and sha(ck)==su['best_checkpoint_sha256'] and sha(cp)==su['config_sha256']
 ckpt=torch.load(ck,map_location='cpu',weights_only=False);conf=ckpt['config'];assert conf==json.loads(cp.read_text()) and conf['manifest_sha256']==sha(a.manifest)
 assert ckpt['format']=='round10_htt_native_force_adapter_v1'
 if a.old_r5:
  assert a.regression
  ck=Path(conf['r5_checkpoint']);assert sha(ck)==conf['r5_checkpoint_sha256'];ckpt=torch.load(ck,map_location='cpu',weights_only=False);assert ckpt['format']=='round5_htt_native_force_adapter_v1'
 manifest=json.loads(a.manifest.read_text());assert manifest['fold']==conf['fold'];entries=manifest['entries']
 if a.regression:
  c=json.loads(Path(manifest['provenance']['contract']).read_text());assert sha(manifest['provenance']['contract'])==manifest['provenance']['contract_sha256']
  entries=[{**e,'outer_role':e['roles_by_fold'][conf['fold']]} for e in c['entries'] if e['task']=='force' and e['roles_by_fold'][conf['fold']] in ('train','validation','calibration')]
 assert sha(conf['source_checkpoint'])==conf['source_checkpoint_sha256']
 decoder,_=load_decoupled_decoder(Path(conf['source_checkpoint']));model=ForceAdapter(decoder).to(a.device);model.load_state_dict(ckpt['model_state'],strict=True);model.eval().requires_grad_(False)
 before=tensor_state_sha256(model.state_dict());mean=torch.as_tensor(ckpt['normalization']['mean'],device=a.device);std=torch.as_tensor(ckpt['normalization']['std'],device=a.device)
 done=a.output/'prediction_manifest.json'
 if done.exists():
  old=json.loads(done.read_text());assert old['force_checkpoint']['sha256']==sha(ck) and old['source_sha256']==sha(__file__)
  assert old.get('old_r5',False)==a.old_r5 and old['regression']==a.regression and old['fold']==conf['fold'] and old['seed']==conf['seed'] and old['smoke']==conf['smoke'] and old['manifest_sha256']==sha(a.manifest)
  assert {e['episode_id'] for e in old['entries']}=={e['episode_id'] for e in entries} and len(old['entries'])==len(entries)
  assert all(sha(e['prediction_path'])==e['prediction_sha256'] for e in old['entries']);print(json.dumps({'status':'reused','manifest':str(done)}));return
 outs=[];reused={};reuse_provenance=[]
 if a.old_r5:
  for role in ('validation','calibration'):
   dp=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round5_force_conditioned_slip/analysis_force/htt')/f'p{conf["fold"][-1]}_s{conf["seed"]}_{role}'
   mp=dp/'metrics.json';rp=dp/'predictions.csv'
   if not mp.exists():continue
   metrics=json.loads(mp.read_text());assert metrics['fold']==conf['fold'] and metrics['seed']==conf['seed'] and metrics['checkpoint_sha256']==sha(ck) and metrics['predictions_csv_sha256']==sha(rp)
   with rp.open() as f:
    for row in csv.DictReader(f):reused.setdefault(row['episode_id'],{})[int(row['frame'])]=[float(row[f'prediction_{i}']) for i in range(3)]
   reuse_provenance.append({'role':role,'metrics':str(mp),'metrics_sha256':sha(mp),'predictions':str(rp),'predictions_sha256':sha(rp)})
 with torch.inference_mode():
  for i,e in enumerate(entries):
   tok=np.load(e['token_path'],mmap_mode='r');assert tok.shape==(e['frames'],300,768)
   parts=[]
   if e['episode_id'] in reused:
    cached=reused[e['episode_id']];assert set(cached)==set(range(5,len(tok)))
    y=np.zeros((len(tok),3),np.float32);y[5:]=np.asarray([cached[j] for j in range(5,len(tok))],np.float32)
   else:
    for j in range(0,len(tok),a.batch_size):parts.append((model(torch.from_numpy(np.array(tok[j:j+a.batch_size],copy=True)).to(a.device))*std+mean).cpu().numpy())
    y=np.concatenate(parts).astype(np.float32)
   assert y.shape==(e['frames'],3) and np.isfinite(y).all()
   pp=a.output/f'{i:03d}.npy';np.save(pp,y);outs.append({'episode_id':e['episode_id'],'prediction_path':str(pp),'prediction_sha256':sha(pp),'frames':e['frames'],'role':e['outer_role'],'token_path':e['token_path']})
 assert tensor_state_sha256(model.state_dict())==before
 out={'status':'complete','fold':conf['fold'],'seed':conf['seed'],'smoke':conf['smoke'],'variant':'r5_regression_reuse' if a.old_r5 else 'round10_adapt','regression':a.regression,'old_r5':a.old_r5,'reused_prediction_provenance':reuse_provenance,'reused_pre5_fill':'zero alignment placeholder; never evaluated or consumed as valid base','source_sha256':sha(__file__),'manifest_sha256':sha(a.manifest),'force_checkpoint':{'path':str(ck),'sha256':sha(ck)},'force_summary':{'path':str(sp),'sha256':sha(sp)},'force_config':{'path':str(cp),'sha256':sha(cp)},'normalization':{k:np.asarray(v).tolist() for k,v in ckpt['normalization'].items()},'entries':outs,'frozen':True,'minimum_unpadded_base':5,'predictions_before_5_exported_for_alignment_only':True}
 atomic_json(done,out);print(json.dumps({'status':'complete','entries':len(outs),'manifest':str(done)}))
if __name__=='__main__':main()
