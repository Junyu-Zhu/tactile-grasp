#!/usr/bin/env python3
"""Incremental cached-head cost; explicitly not end-to-end deployment speed."""
import csv
import argparse, importlib.util, json, os, time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
import numpy as np
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('r8_head_cost',HERE/'training/train.py')
train=importlib.util.module_from_spec(spec);spec.loader.exec_module(train)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 inv=json.loads((a.root/'FORMAL_INVENTORY.json').read_text())
 accepted=json.loads((a.root/'formal_delivery/PREDICTION_MANIFEST.json').read_text())
 proof=accepted['training_audit'];assert train.sha256(Path(proof['path']))==proof['sha256'] and json.loads(Path(proof['path']).read_text())['status']=='pass'
 assert train.sha256(a.root/'FORMAL_INVENTORY.json')==json.loads(Path(proof['path']).read_text())['inventory_sha256']
 index_proof=accepted['checkpoint_index'];assert train.sha256(Path(index_proof['path']))==index_proof['sha256']
 index=list(csv.DictReader(Path(index_proof['path']).open()))
 assert train.sha256(Path(inv['prepared_data']))==accepted['prepared']['sha256']
 payload=torch.load(inv['prepared_data'],map_location='cpu',weights_only=False)
 row=payload['timelines']['outer'];idx=min(range(len(row['t'])),key=lambda i:(row['episode_id'][i],int(row['t'][i])))
 subset={k:(v[idx:idx+1] if torch.is_tensor(v) else v) for k,v in row.items()}
 config=json.loads((HERE/'NUMERIC_PROTOCOL.json').read_text())['e2e'];results=[]
 for run in inv['runs']:
  if run['seed']!=20260914:continue
  summary=json.loads((Path(run['output'])/'summary.json').read_text());best=summary['artifacts']['best'];assert train.sha256(Path(best['path']))==best['sha256']
  matches=[r for r in index if r['group']==run['group'] and int(r['seed'])==20260914 and r['kind']=='best']
  assert len(matches)==1 and Path(matches[0]['path']).resolve()==Path(best['path']).resolve() and matches[0]['sha256']==best['sha256']
  state=torch.load(best['path'],map_location='cpu',weights_only=False)
  assert all(train.sha256(Path(p))==h for p,h in state['run_config']['source_hashes'].items())
  group=run['group'];model=train.make_model(group).eval().requires_grad_(False).cuda();model.load_state_dict(state['model_state'])
  before={k:train.tensor_sha(v) for k,v in model.state_dict().items()};x=train.assemble_inputs(payload,subset,group).cuda()
  batches=[]
  with torch.inference_mode():
   for repeat in range(config['repeats']):
    torch.cuda.reset_peak_memory_stats();values=[]
    for step in range(config['warmup']+config['steps']):
     torch.cuda.synchronize();start=time.perf_counter();pred=train.model_probabilities(model(x),group,[1,3,5])[0];torch.cuda.synchronize()
     if step>=config['warmup']:values.append((time.perf_counter()-start)*1000)
     assert torch.isfinite(pred).all()
    batches.append({'repeat':repeat,'samples_ms':values,'median_ms':float(np.median(values)),'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
  assert all(train.tensor_sha(v)==before[k] for k,v in model.state_dict().items())
  results.append({'group':group,'seed':20260914,'parameters':sum(v.numel() for v in model.parameters()),'checkpoint_sha256':best['sha256'],'head_frozen':True,'measurement':batches})
  del model,x,pred;torch.cuda.empty_cache()
 assert {r['group'] for r in results}==set(accepted['groups']) and len(results)==len(accepted['groups'])
 train.atomic_json(a.root/'benchmark/HEAD_ONLY_COST.json',{'accepted_manifest_sha256':train.sha256(a.root/'formal_delivery/PREDICTION_MANIFEST.json'),'checkpoint_index_sha256':index_proof['sha256'],'inventory_sha256':train.sha256(a.root/'FORMAL_INVENTORY.json'),'status':'complete','schema':'round8_incremental_head_cost_v1','rows':results,'protocol_sha256':train.sha256(HERE/'NUMERIC_PROTOCOL.json'),'source_sha256':train.sha256(Path(__file__)),'prepared_sha256':accepted['prepared']['sha256'],'device':torch.cuda.get_device_name(0),'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'torch':torch.__version__,'semantics':'single sample cached normalized features resident on GPU -> new future module only; excludes image processing, encoder, force/current-slip and feature assembly; not deployment latency; peak_allocated_bytes is total resident head+input+temporary allocation, not an incremental subtraction','episode':row['episode_id'][idx],'t':int(row['t'][idx])})
 print(json.dumps({'status':'complete','groups':len(results)}))
if __name__=='__main__':main()
