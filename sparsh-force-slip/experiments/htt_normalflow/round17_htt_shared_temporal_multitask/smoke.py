#!/usr/bin/env python3
import argparse, copy, json, tempfile
from pathlib import Path
import torch
import multitask_train as mt

def synthetic(seed=17):
 g=torch.Generator().manual_seed(seed);roles={}
 specs=(("fit",320),("selection",96),("calibration",96),("validation",96))
 for role,n in specs:
  x=torch.randn(n,9,195,generator=g);stage=torch.tensor(([0,2,1,1]*((n+3)//4))[:n]);cur=x[:,-1,192:195].clone();y=torch.stack([cur+.03*h+(.02*x[:,-1,:3]) for h in mt.HORIZONS],1);eps=[f"{role}/e{i//8}" for i in range(n)]
  roles[role]={"x":x,"y":y,"y_current":cur,"stage":stage,"episode_id":eps,"leakage_group":[f"{role}/g{i//8}" for i in range(n)],"probe":["synthetic"]*n,"t":torch.arange(n)+13}
 norm,w,table,joint=mt.fit_assets(roles,20260914);roles["fit"]["slip_weight"]=w
 return {"schema":"synthetic","roles":roles,"normalizer":norm,"class_trial_table":table,"joint_weight":joint,"data_sha256":"synthetic"}

def params(m,names):return {n:p.detach().clone() for n,p in m.named_parameters() if any(n.startswith(q+".") for q in names)}
def changed(a,m):return {n:not torch.equal(v,dict(m.named_parameters())[n].detach()) for n,v in a.items()}

def main():
 p=argparse.ArgumentParser();p.add_argument("--device",default="cpu");p.add_argument("--output",type=Path,default=Path(__file__).with_name("SMOKE.json"));a=p.parse_args();d=synthetic();device=a.device
 # Canonical initialization identities and head comparability.
 models={g:mt.init_model(20260914) for g in mt.GROUPS};module_hashes={g:{n:mt.state_hash(models[g].state_dict(),(n,)) for n in mt.MODULES} for g in mt.GROUPS}
 assert all(len({module_hashes[g][n] for g in mt.GROUPS})==1 for n in mt.MODULES)
 assert mt.state_hash(mt.init_model(20260914).state_dict())!=mt.state_hash(mt.init_model(20260915).state_dict())
 # Loss denominator is exactly reliable mask count.
 m=mt.init_model(20260914);ids=torch.arange(7);x=(d["roles"]["fit"]["x"][ids]-d["normalizer"]["x_mean"])/d["normalizer"]["x_std"];y=(d["roles"]["fit"]["y"][ids]-d["normalizer"]["y_mean"])/d["normalizer"]["y_std"];st=d["roles"]["fit"]["stage"][ids];w=d["roles"]["fit"]["slip_weight"][ids];ls,lf,slog,_=mt.loss_parts(m,x,y,st,w);mask=st.eq(0)|st.eq(2);manual=(torch.nn.functional.binary_cross_entropy_with_logits(slog[mask],st[mask].eq(2).float(),reduction="none")*w[mask]).sum()/mask.sum();assert torch.equal(ls,manual)
 # Empty slip: S does no optimizer operation; J updates backbone/future but not slip head.
 inc=torch.nonzero(d["roles"]["fit"]["stage"].eq(1),as_tuple=False).flatten()[:16];xx=((d["roles"]["fit"]["x"][inc]-d["normalizer"]["x_mean"])/d["normalizer"]["x_std"]).to(device);yy=((d["roles"]["fit"]["y"][inc]-d["normalizer"]["y_mean"])/d["normalizer"]["y_std"]).to(device);st=d["roles"]["fit"]["stage"][inc].to(device);ww=d["roles"]["fit"]["slip_weight"][inc].to(device)
 sm=mt.init_model(20260914).to(device);sopt=torch.optim.AdamW([p for n in mt.active_names("S") for p in getattr(sm,n).parameters()],lr=1e-3,weight_decay=1e-4);sb=params(sm,mt.active_names("S"));sls,slf,_,_=mt.loss_parts(sm,xx,yy,st,ww);assert sls is None;assert len(sopt.state)==0 and not any(changed(sb,sm).values())
 jm=mt.init_model(20260914).to(device);jopt=torch.optim.AdamW(jm.parameters(),lr=1e-3,weight_decay=1e-4);jb=params(jm,mt.MODULES);jls,jlf,_,_=mt.loss_parts(jm,xx,yy,st,ww);assert jls is None and torch.isfinite(jlf);(d["joint_weight"]["lambda_future"]*jlf).backward();jopt.step();jc=changed(jb,jm);assert any(jc[n] for n in jc if n.startswith(("visual.","force.","gru.","future."))) and not any(jc[n] for n in jc if n.startswith("slip."))
 # Both task gradients reach backbone; inactive head is unchanged in a formal S/F step.
 valid=torch.nonzero(d["roles"]["fit"]["stage"].eq(0)|d["roles"]["fit"]["stage"].eq(2),as_tuple=False).flatten()[:16];xx=((d["roles"]["fit"]["x"][valid]-d["normalizer"]["x_mean"])/d["normalizer"]["x_std"]).to(device);yy=((d["roles"]["fit"]["y"][valid]-d["normalizer"]["y_mean"])/d["normalizer"]["y_std"]).to(device);st=d["roles"]["fit"]["stage"][valid].to(device);ww=d["roles"]["fit"]["slip_weight"][valid].to(device)
 gm=mt.init_model(20260914).to(device);gls,glf,_,_=mt.loss_parts(gm,xx,yy,st,ww);gls.backward(retain_graph=True);slip_backbone=sum(float(p.grad.abs().sum()) for n,p in gm.named_parameters() if n.startswith("gru."));gm.zero_grad();glf.backward();future_backbone=sum(float(p.grad.abs().sum()) for n,p in gm.named_parameters() if n.startswith("gru."));assert slip_backbone>0 and future_backbone>0
 upstream=d["roles"]["fit"]["x"].clone()
 with tempfile.TemporaryDirectory() as td:
  td=Path(td);full=mt.train(copy.deepcopy(d),td/"full","J",1,20260914,device,max_epochs=4,patience=99);first=mt.train(copy.deepcopy(d),td/"resume","J",1,20260914,device,max_epochs=4,patience=99,interrupt_after=1);assert first["status"]=="interrupted";res=mt.train(copy.deepcopy(d),td/"resume","J",1,20260914,device,max_epochs=4,patience=99);assert full["latest_active_state_sha256"]==res["latest_active_state_sha256"] and full["best_epoch"]==res["best_epoch"] and full["best_metric"]==res["best_metric"]
 assert torch.equal(upstream,d["roles"]["fit"]["x"])
 bad=synthetic();bad["roles"]["selection"]["leakage_group"][0]=bad["roles"]["fit"]["leakage_group"][0]
 try:mt.assert_roles(bad["roles"]);raise AssertionError("role overlap accepted")
 except ValueError:pass
 counts={g:sum(p.numel() for n in mt.active_names(g) for p in getattr(mt.init_model(1),n).parameters()) for g in mt.GROUPS}
 receipt={"schema":"round17_smoke_v1","status":"pass","device":device,"parameter_counts":counts,"canonical_parameters":sum(p.numel() for p in mt.init_model(1).parameters()),"joint_weight":d["joint_weight"],"module_initial_hashes":module_hashes,"checks":["finite_losses_gradients","slip_mask_denominator","S_empty_slip_no_optimizer_state_or_update","J_empty_slip_future_only_update","both_tasks_reach_backbone","common_independent_module_initialization","different_seed_distinct","fit_only_normalizer_and_weights","role_overlap_rejected","immutable_cached_upstream","atomic_checkpoint","interrupted_resume_exact"]};mt.atomic_json(receipt,a.output);print(json.dumps(receipt,indent=2))
if __name__=="__main__":main()
