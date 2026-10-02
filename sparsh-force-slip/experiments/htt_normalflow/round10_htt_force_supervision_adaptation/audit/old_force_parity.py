import sys,json,hashlib
from pathlib import Path
import numpy as np
import torch
C=Path(__file__).resolve().parents[1];sys.path.insert(0,str(C.parent/'round5_force_conditioned_slip/current'))
from sources import load_decoupled_decoder
from models import ForceAdapter
from common import configure_determinism
configure_determinism()
O=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow');rows=[]
for f in range(1,5):
 for s in [20260914,20260915,20260916]:
  p=O/f'round9_htt_temporal_force_fusion/prepare/p{f}_s{s}/prepared.pt';d=torch.load(p,map_location='cpu',weights_only=False);prov=d['provenance'];ck=torch.load(prov['force_checkpoint']['path'],map_location='cpu',weights_only=False)
  decoder,_=load_decoupled_decoder(Path(prov['source_checkpoint']['path']));m=ForceAdapter(decoder).eval().requires_grad_(False).cuda();m.load_state_dict(ck['model_state'])
  ep=next(e for e in d['episodes'] if e['role']=='train');tok=np.load(ep['token_path'],mmap_mode='r');x=torch.from_numpy(np.array(tok[5:21],copy=True)).cuda()
  with torch.inference_mode():y=m(x).cpu()*torch.tensor(ck['normalization']['std'])+torch.tensor(ck['normalization']['mean'])
  error=float((y-ep['force'][5:21]).abs().max());assert torch.allclose(y,ep['force'][5:21],atol=1e-5,rtol=1e-5)
  rows.append({'fold':f,'seed':s,'episode':ep['episode_id'],'max_abs_N':error,'force_checkpoint':prov['force_checkpoint']});del m,decoder,d
  print(f,s,error,flush=True)
p=O/'round10_htt_force_supervision_adaptation/smoke/OLD_FORCE_PARITY.json';p.write_text(json.dumps({'status':'pass','rows':rows,'atol':1e-5,'rtol':1e-5,'scope':'16 train frames per fold seed unpadded s5..20','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
