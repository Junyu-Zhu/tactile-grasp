#!/usr/bin/env python3
"""Verify exact ToucHD sharded-loader recovery before formal dispatch."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from touchd_common import atomic_json,sha256

def same(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return a==b

def main():
 p=argparse.ArgumentParser();p.add_argument('--continuous',type=Path,required=True);p.add_argument('--recovery',type=Path,required=True);p.add_argument('--fixture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 fields=('model_state','optimizer_state','rng_state','history','best_metric','best_epoch','wait','normalization')
 checks={}
 for name in ('best.pth','latest.pth'):
  left=torch.load(a.continuous/name,map_location='cpu',weights_only=False);right=torch.load(a.recovery/name,map_location='cpu',weights_only=False)
  checks[name]={field:same(left[field],right[field]) for field in fields}
  if not all(checks[name].values()):raise RuntimeError(f'nonexact sharded recovery {name}: {checks[name]}')
 summaries=[json.loads((root/'summary.json').read_text()) for root in (a.continuous,a.recovery)]
 if any(x['status']!='complete' or not x['smoke'] or not x['formal_sharded_cache'] or not all(x['component_changed'].values()) for x in summaries):raise RuntimeError('invalid sharded summary')
 result={'schema':'round16_sharded_resume_smoke_v1','status':'pass','formal':False,'weights_forbidden_formal':True,'real_formal_cache_shards':True,'epochs':2,'subprocess_interruption_after_epoch':1,'exact':checks,'fixture_sha256':sha256(a.fixture),'continuous_summary_sha256':sha256(a.continuous/'summary.json'),'recovery_summary_sha256':sha256(a.recovery/'summary.json')}
 atomic_json(a.output,result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
