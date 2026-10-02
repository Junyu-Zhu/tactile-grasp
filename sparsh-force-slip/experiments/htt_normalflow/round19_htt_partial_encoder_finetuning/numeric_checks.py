#!/usr/bin/env python3
"""Regression checks for the repaired low-FPR selector and tie semantics."""
from __future__ import annotations
import importlib.util,json,math
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("r19_train",HERE/"train.py");m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def ref(stage,score,limit=.1):
 y=(np.asarray(stage)==2).astype(int);s=np.asarray(score,float);order=np.argsort(-s,kind="stable");y=y[order];s=s[order];ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1];tp=np.r_[0,np.cumsum(y)[ends]]/y.sum();fp=np.r_[0,np.cumsum(1-y)[ends]]/(1-y).sum();
 if fp[-1]<limit:fp=np.r_[fp,limit];tp=np.r_[tp,tp[-1]]
 elif not np.any(fp==limit):
  j=np.flatnonzero(fp>limit)[0];i=j-1;q=(limit-fp[i])/(fp[j]-fp[i]);fp=np.insert(fp,j,limit);tp=np.insert(tp,j,tp[i]+q*(tp[j]-tp[i]))
 keep=fp<=limit;return float(np.trapezoid(tp[keep],fp[keep])/limit)

cases={
 "vertical_at_zero":([2,2]+[0]*10,[1,.9]+[.8-i*.01 for i in range(10)]),
 "exact_boundary":([2,0,2]+[0]*9,[1,.9,.8]+[.7-i*.01 for i in range(9)]),
 "repeated_scores":([2,0,2,0]+[0]*8,[1,.9,.9,.9]+[.8-i*.01 for i in range(8)]),
 "all_tied":([2,0,2,0,0,0,0,0,0,0,0,0],[.5]*12),
}
rows=[]
for name,(stage,score) in cases.items():
 got=m.low_fpr_auc(torch.tensor(stage),torch.tensor(score));want=ref(stage,score);rows.append({"case":name,"got":got,"reference":want,"pass":math.isclose(got,want,abs_tol=1e-12,rel_tol=0)})
missing=[]
for stage in ([0,0],[2,2]):
 try:m.low_fpr_auc(torch.tensor(stage),torch.tensor([.1,.2]))
 except ValueError:missing.append(True)
 else:missing.append(False)
result={"schema":"round19_pauc_numeric_checks_v1","status":"pass" if all(r["pass"] for r in rows) and all(missing) else "fail","cases":rows,"no_class_support_rejected":all(missing),"tie_policy":"distinct-score groups; stable sort; earliest strict epoch maximum"}
(HERE/"PAUC_NUMERIC_CHECKS.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps(result,indent=2))
if result["status"]!="pass":raise SystemExit(1)
