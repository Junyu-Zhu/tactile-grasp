import importlib.util,json,hashlib
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[1]
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=load(root/'training/train.py','r11');old=load(root.parent/'round9_htt_temporal_force_fusion/training/train.py','r9')
checks={}
role={'stage':torch.tensor([0,0,2,2,2,0,0,1]),'episode_id':['a']*5+['b']*3}
w,rs=m.trial_weights(role)
assert torch.allclose(w,torch.tensor([.875,.875,7/12,7/12,7/12,1.75,1.75,0]));checks['handcomputed_two_trials_one_single_class']=True
counts={}
for seed in [20260914,20260915,20260916]:
 for new,prev in [('V_balanced','V_temporal'),('F_history_balanced','F_history')]:
  m.configure(seed);a=m.TemporalHead(new);m.configure(seed);b=old.TemporalHead(prev)
  assert a.state_dict().keys()==b.state_dict().keys()
  assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
  x=torch.randn(7,9,199);assert torch.equal(a(x),b(x));counts[new]=sum(p.numel() for p in a.parameters())
 m.configure(seed);a=m.TemporalHead('V_balanced');m.configure(seed);c=m.TemporalHead('F_residual_balanced')
 assert all(torch.equal(v,c.state_dict()[k]) for k,v in a.state_dict().items())
 x=torch.randn(7,9,199,requires_grad=True);z=c.components(x);assert torch.equal(a(x),z['base_logit']);assert z['residual_logit'].abs().max()<=2
 z['logit'].sum().backward();assert x.grad[:,:,192:195].abs().sum()>0 and x.grad[:,:,195:198].count_nonzero()==0
 counts['F_residual_balanced']=sum(p.numel() for p in c.parameters())
assert abs(counts['F_residual_balanced']/counts['F_history_balanced']-1)<.1
checks.update({'three_seed_A_B_exact_historical_init_forward':True,'three_seed_C_exact_A_base_init_forward':True,'C_force_connected_delta_disconnected_bounded_residual':True,'capacity_within10percent':True})
result={'status':'pass','checks':checks,'parameters':counts,'source_sha256':hashlib.sha256((root/'training/train.py').read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256((root/'training/protocol.json').read_bytes()).hexdigest(),'limits':['CPU checks do not replace formal smoke true interruption and upstream freeze checks','C visual base jointly trains and correction also sees visual hidden; residual effects alone cannot establish exclusively force contribution']}
(root/'reviews/INDEPENDENT_TRAINING_CPU.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
