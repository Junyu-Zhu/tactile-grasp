import pathlib,json,csv,hashlib,numpy as np,datetime
R=pathlib.Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round10_htt_force_supervision_adaptation');R5=R.parent/'round5_force_conditioned_slip'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(pathlib.Path(p).read_text())
def rows(p):
 with pathlib.Path(p).open() as f:return list(csv.DictReader(f))
s=load(R/'force_support/fold_p1.json');ce={e['episode_id']:e for e in load(s['provenance']['contract'])['entries']};support={e['episode_id']:e for e in s['entries']};checks=[]
for regression in [False,True]:
 d=R/'force_evaluation/p1_s20260914';d=d/'old_force_regression' if regression else d;au=load(d/'AUDIT.json')
 for name,h in au['output_hashes'].items():assert sha(d/name)==h
 table=rows(d/'per_trial_axis.csv');assert len(table)==au['rows'];assert len({r['episode_id'] for r in table})==sum(e['frames']>13 for e in ce.values() if e['task']==('force' if regression else 'slip') and e['roles_by_fold']['htt_leave_p1'] in ('train','validation','calibration'));assert au['test_consumed'] is False
 nm=R/'regression_predictions/new/p1_s20260914/prediction_manifest.json' if regression else R/'force_predictions/p1_s20260914/prediction_manifest.json';om=R/'regression_predictions/old/p1_s20260914/prediction_manifest.json' if regression else R5/'formal/force_predictions/adapt/p1_s20260914/prediction_manifest.json'
 assert sha(nm)==au['new_manifest_sha256'] and sha(om)==au['old_manifest_sha256'];mn=load(nm);mo=load(om)
 for manifest in [mn,mo]:
  for e in manifest['entries']:assert sha(e['prediction_path'])==e['prediction_sha256']
 eid=sorted({r['episode_id'] for r in table if r['role']=='validation'})[0];entry=ce[eid]
 if regression:target=np.load(entry['force_native_n_path']);raw=None;stages=np.full(len(target),-1)
 else:
  with np.load(support[eid]['source_path']) as z:raw=np.asarray(z['6d_force'],float)[:,:3]-np.asarray(z['ref_force'],float)[None,:3]
  target=np.clip(raw,-20,20);stages=np.load(entry['label_path']);assert np.array_equal(target.astype(np.float32),np.load(support[eid]['force_native_n_path']))
 values={};maxerr=0.;count=0
 for variant,manifest in [('old',mo),('new',mn)]:
  e=next(e for e in manifest['entries'] if e['episode_id']==eid);pred=np.load(e['prediction_path']);tt=np.arange(13,len(target))
  for row in [r for r in table if r['episode_id']==eid and r['variant']==variant]:
   pop=row['population'];mask=np.ones(len(tt),bool) if pop=='all' else np.isin(stages[tt],[0,2]) if pop=='primary' else stages[tt]=={'static':0,'incipient':1,'gross':2}[pop];idx=tt[mask];j=['shear_x','shear_y','normal'].index(row['axis']);assert len(idx)==int(row['n'])
   y=target[idx,j];p=pred[idx,j];err=p-y;dy=target[idx,j]-target[idx-5,j];dp=pred[idx,j]-pred[idx-5,j]
   rec={'mae':np.mean(abs(err)),'rmse':np.sqrt(np.mean(err**2)),'bias':np.mean(err),'prediction_std':np.std(p),'target_std':np.std(y),'prediction_mean_abs':np.mean(abs(p)),'target_mean_abs':np.mean(abs(y)),'delta5_mae':np.mean(abs(dp-dy)),'delta5_bias':np.mean(dp-dy),'prediction_delta5_std':np.std(dp),'target_delta5_std':np.std(dy),'prediction_outside_clip_fraction':np.mean(abs(p)>20)}
   if not regression:rec['target_clip_fraction']=np.mean(abs(raw[idx,j])>20)
   for key,v in rec.items():error=abs(float(row[key])-v);maxerr=max(maxerr,error);assert np.isclose(float(row[key]),v,atol=1e-6,rtol=1e-6),(key,row[key],v);count+=1
   if pop=='all' and j==0:values[variant]={'mae_shear_x':float(rec['mae']),'delta5_mae_shear_x':float(rec['delta5_mae'])}
  for row in [r for r in rows(d/'frame_errors.csv') if r['episode_id']==eid and r['variant']==variant]:
   t=int(row['t']);assert t>=13;mae=float(np.mean(abs(pred[t]-target[t])));delta=float(np.mean(abs((pred[t]-pred[t-5])-(target[t]-target[t-5]))));assert np.isclose(mae,float(row['force_mae']),atol=1e-6,rtol=1e-6) and np.isclose(delta,float(row['delta5_mae']),atol=1e-6,rtol=1e-6)
 checks.append({'task':'regression' if regression else 'slip_force','episode_id':eid,'role':'validation','numeric_fields_recomputed':count,'max_abs_error':float(maxerr),'examples':values,'audit_sha256':sha(d/'AUDIT.json'),'input_output_hashes_pass':True})
print(json.dumps({'status':'pass','scope':'partial: p1 seed20260914 only; not full12-run acceptance','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks':checks,'all_formal_results_accepted':False}))
