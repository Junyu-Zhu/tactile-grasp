"""Read-only F0 role support; reuse historical GT audits, no model inference."""
import argparse,hashlib,json
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=[]
for fold in range(1,5):
 path=a.run_root/'round17_htt_shared_temporal_multitask/prepared'/f'p{fold}_s20260914/prepared.pt'
 data=torch.load(path,map_location='cpu',weights_only=False)
 support_path=a.run_root/'round10_htt_force_supervision_adaptation/force_support'/f'fold_p{fold}.json'
 support=json.loads(support_path.read_text());entries={e['episode_id']:e for e in support['entries']}
 assert list(data['horizons'])==[1,5,10]
 groups=[]
 for role,r in data['roles'].items():
  assert role in ('fit','selection','calibration','validation')
  y=r['y'];yc=r['y_current'];t=r['t'];delta=y-yc[:,None,:]
  assert torch.isfinite(y).all() and torch.isfinite(yc).all()
  assert y.shape[1:]==(3,3) and int(t.min())>=13
  for ep,time in zip(r['episode_id'],t.tolist()):
   assert entries[ep]['role']==role and time+10<entries[ep]['frames']
  magnitude=delta[:,2,:].abs().amax(1)
  masks={'stable':magnitude<=.25,'transition':(magnitude>.25)&(magnitude<1),'changing':magnitude>=1}
  strata={name:{'endpoints':int(mask.sum()),'trials':len({e for e,keep in zip(r['episode_id'],mask.tolist()) if keep}),'leakage_groups':len({g for g,keep in zip(r['leakage_group'],mask.tolist()) if keep})} for name,mask in masks.items()}
  g=set(r['leakage_group']);groups.append(g)
  endpoint='\n'.join(f'{e}\t{int(time)}\t{group}' for e,time,group in zip(r['episode_id'],t,r['leakage_group']))
  rows.append({'fold':fold,'role':role,'source':str(path),'endpoints':len(t),'trials':len(set(r['episode_id'])),'leakage_groups':len(g),'stage_counts':{str(c):int(r['stage'].eq(c).sum()) for c in (0,1,2)},'strata':strata,'endpoint_sha256':hashlib.sha256(endpoint.encode()).hexdigest(),'current_boundary_fraction_by_axis':yc.abs().ge(20).float().mean(0).tolist(),'future_boundary_fraction_by_horizon_axis':y.abs().ge(20).float().mean(0).tolist(),'future_complete_checked':True})
 assert all(not groups[i]&groups[j] for i in range(len(groups)) for j in range(i))
result={'status':'pass','scope':'F0 fourfold firstseed support; other seeds same endpoint identity must be bound by trainer inventory','rows':rows,'test_consumed':False,'horizons':[1,5,10],'strata_definition':'max axis absolute GT change at h10: <=0.25N stable, >=1N changing, otherwise transition','limits':['Boundary fraction is occupancy at clipped +/-20N, not unobserved raw exceedance fraction.','Trial counts overlap strata and folds; never sum as independent events.','Historical same-NPZ GT audit is reused; this check validates role/time support, not a repeat full raw audit.','Cached nine-step history is not additional information in K: only current base output is read.'],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'status':'pass','folds':4,'role_rows':len(rows),'stable_trial_counts':{f"p{x['fold']}/{x['role']}":x['strata']['stable']['trials'] for x in rows}}))
