#!/usr/bin/env python3
"""Reproduce the read-only role isolation audit without consuming test data."""
import pathlib,json,itertools,torch,datetime
p=pathlib.Path(__file__).resolve().parent
inv=json.loads((p/'G2_RUN_INVENTORY.json').read_text());rows=[]
for path in sorted({r['data'] for r in inv['runs']}):
 d=torch.load(path,map_location='cpu',weights_only=False);roles=d['roles'];issues=[];counts={}
 if set(roles)!={'fit','selection','calibration','validation'}:issues.append('unexpected_roles')
 for a,b in itertools.combinations(roles,2):
  for k in ('episode_id','leakage_group'):
   overlap=set(roles[a][k])&set(roles[b][k])
   if overlap:issues.append({'roles':[a,b],'kind':k,'overlap':sorted(overlap)})
 for name,r in roles.items():
  counts[name]={'endpoints':len(r['t']),'trials':len(set(r['episode_id'])),'leakage_groups':len(set(r['leakage_group'])),'t_min':int(r['t'].min()),'stages':{str(k):int((r['stage']==k).sum()) for k in (0,1,2)}}
  if int(r['t'].min())<13:issues.append('history_endpoint_too_early')
  if not torch.isfinite(r['y_current']).all():issues.append('nonfinite_gt')
  if not torch.isfinite(r['x']).all():issues.append('nonfinite_input')
 rows.append({'data':path,'counts':counts,'issues':issues})
result={'at':datetime.datetime.now().astimezone().isoformat(),'passed':all(not r['issues'] for r in rows),'data_sets':rows,'scope':'12 shared four-arm data identities, role trial/leakage group separation, endpoint minimum and finite GT/inputs. Historical full alignment audit reused.'}
(p/'ROOT_ROLE_AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps({'passed':result['passed'],'datasets':len(rows),'issues':[r for r in rows if r['issues']]}))
