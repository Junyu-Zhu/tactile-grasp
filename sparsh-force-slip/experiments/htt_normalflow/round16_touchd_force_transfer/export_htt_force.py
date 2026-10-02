#!/usr/bin/env python3
"""Export R16 H/T_H selected predictions for all non-test HTT episodes."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
from touchd_common import atomic_json,fresh_adapter,sha256,state_sha256

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--batch-size',type=int,default=128);a=p.parse_args()
 s=a.run/'training_summary.json';su=json.loads(s.read_text());cp=a.run/'config.json';ck=a.run/'best.pth';conf=json.loads(cp.read_text())
 if su['status']!='complete' or sha256(ck)!=su['best_checkpoint_sha256'] or conf!=torch.load(ck,map_location='cpu',weights_only=False)['config']:raise RuntimeError('run identity')
 if conf['manifest_sha256']!=sha256(a.manifest):raise RuntimeError('manifest identity')
 saved=torch.load(ck,map_location='cpu',weights_only=False);private=torch.load(conf['t_private']['path'],map_location='cpu',weights_only=False) if conf['route']=='T_H' else None
 model=fresh_adapter(conf['seed'],trunk_state=private).to(a.device);model.load_state_dict(saved['model_state']);model.eval().requires_grad_(False);before=state_sha256(model.state_dict())
 mean=torch.as_tensor(saved['normalization']['mean'],device=a.device);std=torch.as_tensor(saved['normalization']['std'],device=a.device)
 m=json.loads(a.manifest.read_text());a.output.mkdir(parents=True,exist_ok=True);outs=[]
 with torch.inference_mode():
  for i,e in enumerate(m['entries']):
   tok=np.load(e['token_path'],mmap_mode='r');parts=[]
   for j in range(0,len(tok),a.batch_size):parts.append((model(torch.from_numpy(np.array(tok[j:j+a.batch_size],copy=True)).to(a.device))*std+mean).cpu().numpy())
   y=np.concatenate(parts).astype(np.float32);pp=a.output/f'{i:03d}.npy';np.save(pp,y)
   outs.append({'episode_id':e['episode_id'],'prediction_path':str(pp),'prediction_sha256':sha256(pp),'frames':e['frames'],'role':e['outer_role'],'token_path':e['token_path']})
 if state_sha256(model.state_dict())!=before:raise RuntimeError('model changed')
 out={'status':'complete','fold':conf['fold'],'seed':conf['seed'],'route':conf['route'],'smoke':conf['smoke'],'variant':'round16_'+conf['route'],'regression':False,
      'source_sha256':sha256(Path(__file__)),'manifest_sha256':sha256(a.manifest),'force_checkpoint':{'path':str(ck),'sha256':sha256(ck)},
      'force_summary':{'path':str(s),'sha256':sha256(s)},'force_config':{'path':str(cp),'sha256':sha256(cp)},'normalization':{k:np.asarray(v).tolist() for k,v in saved['normalization'].items()},
      'entries':outs,'frozen':True,'minimum_unpadded_base':5,'predictions_before_5_exported_for_alignment_only':True}
 atomic_json(a.output/'prediction_manifest.json',out);print(json.dumps({'status':'complete','entries':len(outs)}))
if __name__=='__main__':main()
