import importlib.util,json,hashlib
from pathlib import Path
import numpy as np,torch
from sklearn.metrics import roc_curve,auc
P=Path(__file__).resolve().parent.parent/'training/train.py'
s=importlib.util.spec_from_file_location('r9trainer_review',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def main():
 models={};counts={};inits={}
 for g in m.GROUPS:
  m.configure(20260914);model=m.TemporalHead(g).eval();models[g]=model;counts[g]=sum(p.numel() for p in model.parameters());inits[g]=m.state_hash(model.state_dict())
 assert inits['F_history']==inits['F_delta']
 for k in models['V_temporal'].gru.state_dict():
  assert torch.equal(models['V_temporal'].gru.state_dict()[k],models['F_delta'].gru.state_dict()[k])
 assert (max(counts.values())-min(counts.values()))/min(counts.values())<.005
 x=torch.randn(4,9,199);alt=x.clone();alt[:,:,192:198]+=torch.randn(4,9,6)*100
 assert torch.equal(models['V_temporal'](x),models['V_temporal'](alt))
 alt=x.clone();alt[:,:,195:198]+=100
 assert torch.equal(models['F_history'](x),models['F_history'](alt))
 assert not torch.equal(models['F_delta'](x),models['F_delta'](alt))
 rng=np.random.default_rng(100)
 for _ in range(100):
  y=rng.integers(0,2,100);p=np.round(rng.random(100),1)
  f,t,_=roc_curve(y,p,drop_intermediate=False);stop=np.searchsorted(f,.1,'right');a=auc(np.r_[f[:stop],.1],np.r_[t[:stop],np.interp(.1,f,t)])/.1
  assert abs(a-m.low_fpr_auc(y,p))<1e-12
 out={'status':'pass','trainer_sha256':m.sha(P),'protocol_sha256':m.sha(P.parent/'protocol.json'),'parameter_counts':counts,'init_hashes':inits,'checks':['Fhistory/Fdelta full initialization exact','V/F common GRU initialization exact','capacity gap below0.5%','V force/delta perturbation exact invariant','Fhistory delta exact invariant','Fdelta uses delta','pAUC100 tied-score cases match independent sklearn ROC integral'],'limitations':['real interruption recovery and frozen cache identity need smoke runtime','source visual features use accepted R3-B frozen trunk, not fresh MAE representation']}
 (Path(__file__).parent/'TRAINER_INDEPENDENT_REVIEW.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
