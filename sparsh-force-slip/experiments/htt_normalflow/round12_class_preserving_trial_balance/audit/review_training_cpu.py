import importlib.util,json,hashlib
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[1]
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=load(root/'training/train.py','r12');old=load(root.parent/'round10_htt_force_supervision_adaptation/fusion/train.py','r10');r11=load(root.parent/'round11_stable_negative_force_residual/training/train.py','r11')
role={'stage':torch.tensor([0,0,2,2,2,0,0,1]),'episode_id':['a']*5+['b']*3};w,rs=m.trial_weights(role)
assert torch.allclose(w,torch.tensor([.875,.875,7/6,7/6,7/6,.875,.875,0]));counts={}
for seed in [20260914,20260915,20260916]:
 for new,prev,prev11 in [('V_class_trial_balanced','V_temporal','V_balanced'),('F_class_trial_balanced','F_history','F_history_balanced')]:
  m.configure(seed);a=m.TemporalHead(new);m.configure(seed);b=old.TemporalHead(prev);m.configure(seed);c=r11.TemporalHead(prev11)
  assert all(torch.equal(v,b.state_dict()[k]) and torch.equal(v,c.state_dict()[k]) for k,v in a.state_dict().items())
  x=torch.randn(7,9,199,requires_grad=True);assert torch.equal(a(x),b(x)) and torch.equal(a(x),c(x));a(x).sum().backward();assert x.grad[:,:,195:198].count_nonzero()==0
  if new.startswith('V'):assert x.grad[:,:,192:195].count_nonzero()==0
  else:assert x.grad[:,:,192:195].count_nonzero()>0
  counts[new]=sum(p.numel() for p in a.parameters())
res={'status':'pass','checks':{'handcomputed_two_trials_one_single_class':True,'all_three_seed_R10_R11_init_forward_exact':True,'V_no_force_F_force_connected_delta_zero':True},'parameters':counts,'source_sha256':hashlib.sha256((root/'training/train.py').read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256((root/'training/protocol.json').read_bytes()).hexdigest(),'limits':['CPU checks do not replace true process recovery smoke','Global classes50percent preserved; pertrial total contribution differs when classes missing']}
(root/'reviews/INDEPENDENT_TRAINING_CPU.json').write_text(json.dumps(res,indent=2));print(json.dumps(res))
