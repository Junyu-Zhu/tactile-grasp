import ast,hashlib,json
from pathlib import Path
import numpy as np,torch
p=Path(__file__).resolve().parent;src=p/'train_e3.py';tree=ast.parse(src.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='low_fpr_auc');ns={'np':np,'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),str(src),'exec'),ns)
cases=[('perfect',[2,2,0,0],[.9,.8,.7,.6],1.),('reversed',[2,2,0,0],[.1,.2,.8,.9],0.),('all_tied',[2,2,0,0],[.5]*4,.05),('half_recall_at_zero_fpr',[2,2,0,0],[.9,.1,.8,.7],.5),('exact_boundary_vertical',[2]*10+[0]*10,[.95]+[.85]*9+[.9]+[.1]*9,.1)]
rows=[]
for name,y,s,expected in cases:
 value=ns['low_fpr_auc'](torch.tensor(y),torch.tensor(s));rows.append({'case':name,'value':value,'expected':expected,'pass':abs(value-expected)<1e-7})
result={'status':'pass' if all(r['pass'] for r in rows) else 'fail','source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'scope':'only pAUC arithmetic; not whole trainer acceptance','cases':rows};(p/'root_numeric_check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
