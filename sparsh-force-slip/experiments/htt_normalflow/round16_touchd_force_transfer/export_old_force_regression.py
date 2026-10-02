#!/usr/bin/env python3
"""Read-only R16 inference on the accepted R10 dedicated HTT force task."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from touchd_common import atomic_json,fresh_adapter,sha256,state_sha256

ROLES={"train","validation","calibration"}
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--support',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:1');p.add_argument('--batch-size',type=int,default=128);a=p.parse_args()
 summary=json.loads((a.run/'training_summary.json').read_text());config=json.loads((a.run/'config.json').read_text());checkpoint=a.run/'best.pth';saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
 assert summary['status']=='complete' and not summary['smoke'] and sha256(checkpoint)==summary['best_checkpoint_sha256'] and saved['config']==config
 support=json.loads(a.support.read_text());assert support['status']=='pass' and support['fold']==config['fold'] and sha256(a.support)==config['manifest_sha256']
 contract_path=Path(support['provenance']['contract']);assert sha256(contract_path)==support['provenance']['contract_sha256'];contract=json.loads(contract_path.read_text());assert contract['status']=='complete'
 split_path=Path(contract['split_manifest']);assert sha256(split_path)==contract['split_manifest_sha256'];split=json.loads(split_path.read_text());meta={e['id']:e for e in split['episodes']}
 entries=[]
 for e in contract['entries']:
  if e['task']!='force':continue
  role=e['roles_by_fold'][config['fold']]
  if role=='test':continue
  assert role in ROLES and e['force_target_definition']=='clip((6d_force - ref_force)[:, :3], -20 N, 20 N)' and e['token_shape']==[e['frames'],300,768]
  assert Path(e['token_path']).is_file() and Path(e['force_native_n_path']).is_file()
  entries.append({**e,'role':role,'leakage_group':meta[e['episode_id']]['leakage_group']})
 assert entries and all(e['role']!='test' for e in entries)
 private=torch.load(config['t_private']['path'],map_location='cpu',weights_only=False) if config['route']=='T_H' else None
 model=fresh_adapter(config['seed'],trunk_state=private).to(a.device);model.load_state_dict(saved['model_state']);model.eval().requires_grad_(False);before=state_sha256(model.state_dict())
 mean=torch.as_tensor(saved['normalization']['mean'],device=a.device);std=torch.as_tensor(saved['normalization']['std'],device=a.device);a.output.mkdir(parents=True,exist_ok=True);outs=[]
 with torch.inference_mode():
  for i,e in enumerate(entries):
   token=np.load(e['token_path'],mmap_mode='r');parts=[]
   for j in range(0,len(token),a.batch_size):parts.append((model(torch.from_numpy(np.array(token[j:j+a.batch_size],copy=True)).to(a.device))*std+mean).cpu().numpy())
   prediction=np.concatenate(parts).astype(np.float32);assert prediction.shape==(e['frames'],3) and np.isfinite(prediction).all()
   target=np.load(e['force_native_n_path'],mmap_mode='r');assert target.shape==prediction.shape and np.isfinite(target).all()
   path=a.output/f'{i:03d}.npy';np.save(path,prediction);outs.append({'episode_id':e['episode_id'],'role':e['role'],'leakage_group':e['leakage_group'],'frames':e['frames'],'prediction_path':str(path),'prediction_sha256':sha256(path),'target_path':e['force_native_n_path'],'target_sha256':e['force_native_n_sha256'],'token_path':e['token_path'],'token_sha256':e['token_sha256']})
 assert state_sha256(model.state_dict())==before
 result={'schema':'round16_old_htt_force_regression_export_v1','status':'complete','route':config['route'],'fold':config['fold'],'seed':config['seed'],'roles':sorted(ROLES),'test_consumed':False,'training_or_tuning':False,'target':'accepted R10 contract task=force native clip((6d_force-ref_force)[:3],-20,20) N','frame_start_for_evaluation':13,'source_sha256':sha256(Path(__file__)),'support':str(a.support),'support_sha256':sha256(a.support),'contract':str(contract_path),'contract_sha256':sha256(contract_path),'split_manifest':str(split_path),'split_manifest_sha256':sha256(split_path),'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),'config_sha256':sha256(a.run/'config.json'),'summary_sha256':sha256(a.run/'training_summary.json'),'frozen_before_after_bitwise':True,'entries':outs}
 atomic_json(a.output/'prediction_manifest.json',result);print(json.dumps({'status':'complete','route':config['route'],'fold':config['fold'],'seed':config['seed'],'episodes':len(outs)}))
if __name__=='__main__':main()
