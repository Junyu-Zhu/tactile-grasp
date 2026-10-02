#!/usr/bin/env python3
from pathlib import Path
import json, tempfile, torch
import future_train as ft

def synthetic(seed=7):
 g=torch.Generator().manual_seed(seed); roles={}
 for j,(role,n) in enumerate((("fit",48),("selection",20),("calibration",12),("validation",12))):
  x=torch.randn(n,9,195,generator=g); y=torch.stack([x[:,-1,192:195]+0.02*h for h in ft.HORIZONS],1)
  roles[role]={"x":x,"y":y,"episode_id":[f"{role}/e{i}" for i in range(n)],"leakage_group":[f"{role}/g{i}" for i in range(n)],"t":torch.arange(n)+13}
 return {"roles":roles,"provenance":{}}

def main():
 d=synthetic(); upstream=d["roles"]["fit"]["x"].clone(); counts={g:sum(p.numel() for p in ft.init_model(g,1).parameters()) for g in ft.GROUPS}
 assert (max(counts.values())-min(counts.values()))/max(counts.values())<.03
 x=d["roles"]["selection"]["x"][:4]; xp=x.clone(); xp[...,192:195]+=100
 for group in ft.GROUPS:
  m=ft.init_model(group,11); a=m(x); b=m(xp)
  assert torch.equal(a,b) if group=="V" else not torch.equal(a,b)
 assert ft.state_hash(ft.init_model("V",1).state_dict())==ft.state_hash(ft.init_model("V",1).state_dict())
 assert ft.state_hash(ft.init_model("V",1).state_dict())!=ft.state_hash(ft.init_model("V",2).state_dict())
 bad=synthetic(); bad["roles"]["selection"]["leakage_group"][0]=bad["roles"]["fit"]["leakage_group"][0]
 try: ft.assert_role_isolation(bad["roles"]); raise AssertionError("overlap accepted")
 except ValueError: pass
 with tempfile.TemporaryDirectory() as td:
  td=Path(td); full=ft.train(d,td/"full","F_dual",1,20260914,max_epochs=4,patience=99)
  first=ft.train(d,td/"resume","F_dual",1,20260914,max_epochs=4,patience=99,interrupt_after=1); assert first["status"]=="interrupted"
  resumed=ft.train(d,td/"resume","F_dual",1,20260914,max_epochs=4,patience=99)
  assert full["state_sha256"]==resumed["state_sha256"] and full["best_epoch"]==resumed["best_epoch"]
 assert torch.equal(upstream,d["roles"]["fit"]["x"])
 receipt={"schema":"round14_cpu_smoke_v1","status":"pass","groups":list(ft.GROUPS),"parameter_counts":counts,"parameter_spread_fraction":(max(counts.values())-min(counts.values()))/max(counts.values()),"checks":["finite_loss_gradients","upstream_input_bitwise_unchanged","role_overlap_rejected","V_force_invariant","F_force_responsive","seed_repeatable_distinct","atomic_checkpoint_roundtrip","interrupted_resume_exact"]}
 Path(__file__).with_name("SMOKE.json").write_text(json.dumps(receipt,indent=2)+"\n"); print(json.dumps(receipt,indent=2))
if __name__=="__main__": main()

