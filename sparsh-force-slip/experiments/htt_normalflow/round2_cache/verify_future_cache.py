"""Independently reconstruct deployed future inputs and quantify delta shift."""
import argparse
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('XFORMERS_DISABLED','1')
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parents[1]/'scripts'))
from evaluate_real_force_slip_model_test import load_future_head

def main():
    p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    index=json.loads(Path(a.index).read_text())
    model,payload=load_future_head(Path(index['future_head']['path']),torch.device('cpu'))
    if payload.get('input_schema') or payload['horizons']!=[1,3,5]:
        raise ValueError('Verifier fixes the selected legacy z+aux schema')
    torch.set_num_threads(2)
    count=np.zeros(3,np.int64);n=0;checks=[]
    for row in index['episodes']:
        if row['domain']!='htt': continue
        with np.load(row['cache_path'],allow_pickle=False) as data:
            f=data['force_pred'];previous=np.maximum(np.arange(len(f))-5,0);delta=f-f[previous]
            count+=(np.abs(delta)>np.array([.8,.8,.4])).sum(axis=0);n+=len(f)
            if len(checks)>=4: continue
            # Reconstruct from the recorded deployment schema, independently
            # of build_cache.future_predictions. Ground-truth force is absent.
            z=data['z'];s=data['p_slip'];norm=np.abs(f[:,2]);tangent=np.sqrt(f[:,0]**2+f[:,1]**2)
            named={'Fn_pred_N':norm,'Ft_pred_N':tangent,'Ft_over_Fn_pred':tangent/(norm+1e-6),'p_slip_current':s,
                   'dFx_causal_N':delta[:,0],'dFy_causal_N':delta[:,1],'dFz_causal_N':delta[:,2]}
            x=torch.tensor(np.concatenate([z,np.stack([named[k] for k in payload['aux_names']],axis=1)],axis=1))
            with torch.inference_mode():
                expected=(1-torch.sigmoid(model(x[:8]))).numpy()
                independent=(1-torch.sigmoid(model(x[:1]))).numpy()
            if not np.allclose(expected,data['p_future'][:8],atol=1e-6,rtol=1e-5):
                raise ValueError('Cached future differs from independent deployment reconstruction')
            if not np.allclose(expected[:1],independent,atol=1e-6,rtol=1e-5):
                raise ValueError('Prediction depends on future batch rows')
            checks.append({'id':row['id'],'max_abs_error':float(np.max(np.abs(expected-data['p_future'][:8])))})
    out={'status':'passed','independent_checks':checks,'atol':1e-6,'rtol':1e-5,
         'historical_delta_clip_limits':[.8,.8,.4],'htt_frames':n,'predicted_delta_exceeds_historical_clip_counts':count.tolist(),
         'predicted_delta_exceeds_historical_clip_fraction':(count/max(n,1)).tolist(),
         'limit':'Recorded-force and predicted-force domains still differ even when inside historical clipping range'}
    Path(a.output).write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
