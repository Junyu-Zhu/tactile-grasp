#!/usr/bin/env python3
"""Build the resumable 142-shard complete-token ToucHD Mini cache."""
from __future__ import annotations
import argparse,json,os,time,zipfile
from pathlib import Path
import numpy as np
import torch
from touchd_common import (PREPROCESS,SOURCE_CHECKPOINT,TARGET,atomic_json,load_source_model,
                           make_encoder_input,published_target,sha256,state_sha256)

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--batch-size',type=int,default=8);p.add_argument('--execute-formal',action='store_true');a=p.parse_args()
 if not a.execute_formal:raise SystemExit('formal cache requires --execute-formal')
 audit=json.loads(a.audit.read_text());assert audit['status']=='pass';compact=json.loads((a.data/'all_data_direction.json').read_text())
 roles={int(o):r for r,objects in audit['split']['roles'].items() for o in objects};a.output.mkdir(parents=True,exist_ok=True)
 model,payload=load_source_model(torch.device(a.device));before=state_sha256(model.encoder.state_dict());entries=[];began=time.monotonic()
 for number,key in enumerate(sorted(compact),1):
  values=compact[key]['gelsight'];n=len(values)-5;obj=int(key[3:6]);prefix=a.output/key;tp=prefix.with_suffix('.tokens.npy');yp=prefix.with_suffix('.targets.npy');mp=prefix.with_suffix('.json')
  identity={'key':key,'object':obj,'speed':int(key[-1]),'role':roles[obj],'samples':n,'row_start':5,'token_shape':[n,300,768],'token_dtype':'float16','target_shape':[n,3],'target_dtype':'float32','target':TARGET,'preprocess':PREPROCESS,'source_checkpoint_sha256':sha256(SOURCE_CHECKPOINT),'audit_sha256':sha256(a.audit)}
  if mp.exists():
   saved=json.loads(mp.read_text())
   if saved['identity']!=identity or sha256(tp)!=saved['token_sha256'] or sha256(yp)!=saved['target_sha256']:raise RuntimeError('stale cache shard '+key)
  else:
   tt=Path(str(tp)+'.partial');yt=Path(str(yp)+'.partial');tokens=np.lib.format.open_memmap(tt,mode='w+',dtype=np.float16,shape=(n,300,768));targets=np.lib.format.open_memmap(yt,mode='w+',dtype=np.float32,shape=(n,3))
   with zipfile.ZipFile(a.data/(key+'.zip')) as archive:
    for lo in range(5,len(values),a.batch_size):
     hi=min(len(values),lo+a.batch_size);x=torch.stack([make_encoder_input(archive,key,values,i) for i in range(lo,hi)]).to(a.device)
     with torch.inference_mode():z=model.encoder(x)
     if z.shape[1:]!=(300,768) or not torch.isfinite(z).all():raise RuntimeError('token interface')
     tokens[lo-5:hi-5]=z.cpu().numpy().astype(np.float16);targets[lo-5:hi-5]=torch.stack([published_target(values[i]) for i in range(lo,hi)]).numpy()
   tokens.flush();targets.flush();del tokens,targets;os.replace(tt,tp);os.replace(yt,yp)
   saved={'identity':identity,'token_path':str(tp),'token_sha256':sha256(tp),'target_path':str(yp),'target_sha256':sha256(yp)};atomic_json(mp,saved)
  entries.append(saved);atomic_json(a.output/'progress.json',{'status':'running','completed':number,'total':len(compact),'elapsed_seconds':time.monotonic()-began,'last':key})
  print(json.dumps({'completed':number,'total':len(compact),'key':key}),flush=True)
 if state_sha256(model.encoder.state_dict())!=before:raise RuntimeError('encoder changed')
 manifest={'schema':'round16_touchd_token_cache_v1','status':'complete','formal':True,'entries':entries,'samples':sum(e['identity']['samples'] for e in entries),'roles':audit['split']['roles'],'target':TARGET,'preprocess':PREPROCESS,'token_tail_shape':[300,768],'token_dtype':'float16','source_checkpoint':str(SOURCE_CHECKPOINT),'source_checkpoint_sha256':sha256(SOURCE_CHECKPOINT),'audit_path':str(a.audit.resolve()),'audit_sha256':sha256(a.audit),'encoder_frozen_bitwise':True,'elapsed_seconds':time.monotonic()-began,'source_epoch':payload.get('epoch')}
 atomic_json(a.output/'cache_manifest.json',manifest);atomic_json(a.output/'progress.json',{'status':'complete','completed':len(entries),'total':len(entries),'elapsed_seconds':manifest['elapsed_seconds'],'manifest_sha256':sha256(a.output/'cache_manifest.json')});print(json.dumps({'status':'complete','samples':manifest['samples'],'manifest':str(a.output/'cache_manifest.json')}))
if __name__=='__main__':main()
