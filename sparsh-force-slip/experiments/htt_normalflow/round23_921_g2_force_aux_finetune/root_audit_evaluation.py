#!/usr/bin/env python3
"""Independent arithmetic check within the G2 author acceptance, not G3 review."""
import pathlib,json,csv,math,datetime,torch,numpy as np
P=pathlib.Path(__file__).resolve().parent;L=pathlib.Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round23_921_g2_force_aux_finetune');E=L/'evaluation'
inv=json.loads((P/'G2_RUN_INVENTORY.json').read_text())['runs']; work=list(csv.DictReader((E/'metrics/WORKPOINT_METRICS.csv').open()));ci=list(csv.DictReader((E/'bootstrap/PAIRED_CI.csv').open()));issues=[];checked=0;cached={}
assert len(work)==1920
for r in inv:
 path=r['data']
 if path not in cached:cached={path:torch.load(path,map_location='cpu',weights_only=False)}
 d=cached[path];npz=np.load(L/'predictions'/r['run']/'PREDICTIONS.npz',allow_pickle=False)
 own=[x for x in work if x['group']==r['group'] and int(x['fold'])==r['fold'] and int(x['seed'])==r['seed']]
 if len(own)!=40:issues.append([r['run'],'row_count'])
 for x in own:
  if x['k']!='1':continue
  role=x['role'];stage=d['roles'][role]['stage'].numpy();score=npz[role+'_score'];alarm=score>=float(x['threshold']);static=stage==0;gross=stage==2
  exp={'tn':int((static&~alarm).sum()),'fp':int((static&alarm).sum()),'fn':int((gross&~alarm).sum()),'tp':int((gross&alarm).sum())}
  for k,v in exp.items():
   if float(x[k])!=v:issues.append([r['run'],role,x['policy'],k,float(x[k]),v])
  fpr=exp['fp']/(exp['tn']+exp['fp']);rec=exp['tp']/(exp['tp']+exp['fn']);ba=(1-fpr+rec)/2
  for k,v in [('frame_static_FPR',fpr),('gross_recall',rec),('balanced_accuracy',ba)]:
   if not math.isclose(float(x[k]),v,abs_tol=1e-12):issues.append([r['run'],role,x['policy'],k])
  if role=='calibration' and x['policy'].startswith('FPR'):
   alpha=int(x['policy'].split('|')[0][3:])/100
   if fpr>alpha+1e-12:issues.append([r['run'],'calibration_constraint',x['policy'],fpr])
  checked+=1
 npz.close()
if len(ci)!=280 or any(int(x['attempted_draws'])!=2000 or int(x['valid_draws'])+int(x['invalid_draws'])!=2000 for x in ci):issues.append('CI_grid_or_draw_count')
grad=[z for f in (L/'gradients').rglob('*.json') for z in json.loads(f.read_text())['rows']]
if len(grad)!=72 or any(x['status']!='available' or not x['slip_finite'] or not x['aux_finite'] for x in grad):issues.append('gradient_numerics')
result={'at':datetime.datetime.now().astimezone().isoformat(),'passed':not issues,'raw_workpoint_rows_recomputed':checked,'total_workpoint_rows':len(work),'ci_rows':len(ci),'gradient_rows':len(grad),'issues':issues,'scope':'Raw confusion matrices, FPR/recall/BA from original exported scores; calibration constraints; CI grid; gradient finite. Continuous-event implementation reuses prior audited R13 and current evaluator smoke. G3 independent review remains pending.'}
(P/'ROOT_EVALUATION_AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
