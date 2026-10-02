#!/usr/bin/env python3
"""CPU semantic checks for E1/E2/F1/F2 before any formal dispatch."""
import json
from pathlib import Path
import torch
import train_frozen as D
import train_f1 as F
def main():
 torch.set_num_threads(4);seed=20260914;x=torch.randn(8,9,195);force_perturbed=x.clone();force_perturbed[...,192:195]+=7
 models={g:D.init_model(g,seed) for g in D.GROUPS};m_out={g:models[g].components(x) for g in D.GROUPS}
 checks={
  "V_force_isolation":bool(torch.equal(models["V"](x),models["V"](force_perturbed))),
  "M_MB_initial_identity":bool(torch.equal(models["M"](x),models["MB"](x))),
  "MB_initial_identity_modulation":bool(torch.count_nonzero(m_out["MB"]["gamma"])==0 and torch.count_nonzero(m_out["MB"]["beta"])==0),
  "M_MB_parameter_count_equal":sum(p.numel() for p in models["M"].parameters())==sum(p.numel() for p in models["MB"].parameters()),
 }
 mb=models["MB"];opt=torch.optim.AdamW(mb.parameters(),lr=1e-3)
 before=mb.film_hidden.weight.detach().clone()
 for _ in range(3):opt.zero_grad();loss=mb(x).square().mean();loss.backward();opt.step()
 comp=mb.components(x);checks.update({"MB_early_film_parameter_updates_after_multistep":bool(not torch.equal(before,mb.film_hidden.weight)),"MB_gamma_bound":float(comp["gamma"].abs().max())<=.25,"MB_beta_bound":float(comp["beta"].abs().max())<=.5})
 fmods={g:F.init_model(g,seed) for g in F.GROUPS};checks["F1_parameter_contract"]=all(sum(p.numel() for p in fmods[g].parameters())==F.EXPECTED_PARAMS[g] for g in F.GROUPS);checks["F1_zero_delta_initialization"]=bool(all(torch.count_nonzero(fmods[g](x))==0 for g in F.GROUPS))
 vpert=x.clone();vpert[...,:192]+=5;fpert=x.clone();fpert[...,192:195]+=5
 checks.update({"K_V_ignores_force_dynamic_input":torch.equal(fmods["K-V"](x),fmods["K-V"](fpert)),"K_F_ignores_visual_dynamic_input":torch.equal(fmods["K-F"](x),fmods["K-F"](vpert)),"F2_fixed_coefficient":.5==0.5})
 cases=[("perfect",[2,2,0,0],[.9,.8,.7,.6],1.),("reversed",[2,2,0,0],[.1,.2,.8,.9],0.),("all_tied",[2,2,0,0],[.5]*4,.05),("half_recall_at_zero_fpr",[2,2,0,0],[.9,.1,.8,.7],.5),("exact_boundary_vertical",[2]*10+[0]*10,[.95]+[.85]*9+[.9]+[.1]*9,.1)];numeric=[]
 for name,y,s,expected in cases:
  value=D.low_fpr_auc(torch.tensor(y),torch.tensor(s));numeric.append({"case":name,"value":value,"expected":expected,"pass":abs(value-expected)<1e-7})
 checks["corrected_pAUC_five_cases"]=all(r["pass"] for r in numeric)
 out={"schema":"round22_frozen_semantic_smoke_v2","status":"pass" if all(checks.values()) else "fail","checks":checks,"pAUC_cases":numeric,"parameters_frozen":{g:sum(p.numel() for p in models[g].parameters()) for g in D.GROUPS},"parameters_f1":{g:sum(p.numel() for p in fmods[g].parameters()) for g in F.GROUPS},"formal_training_started":False};Path(__file__).with_name("FROZEN_SEMANTIC_SMOKE.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");print(json.dumps(out))
 if out["status"]!="pass":raise SystemExit(1)
if __name__=="__main__":main()
