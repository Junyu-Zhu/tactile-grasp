import pathlib,csv,json,hashlib,numpy as np,datetime,math
R=pathlib.Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round10_htt_force_supervision_adaptation');D=R/'reporting'
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):
 with pathlib.Path(p).open() as f:return list(csv.DictReader(f))
def load(p):return json.loads(pathlib.Path(p).read_text())
def num(v):return float(v) if v not in ('',None) else float('nan')
def eq(x,y):return (math.isnan(x) and math.isnan(y)) or np.isclose(x,y,rtol=1e-10,atol=1e-12)
a=load(D/'REPORT_AUDIT.json');assert a['status']=='pass' and a['formal_runs']==24 and a['comparison_models']==36
for p,h in a['sources'].items():assert sha(p)==h
for p,h in a['output_hashes'].items():assert sha(D/p)==h
current=read(R/'current_evaluation/metrics.csv');force=read(R/'force_aggregate/per_run_summary.csv');sens=read(R/'current_sensitivity/metrics.csv');counts={}
for filename,source,filterkeys,fixed in [('CURRENT_KEY_RESULTS.csv',current,('group','point'),{'role':'validation','rule':'raw'}),('FORCE_KEY_RESULTS.csv',force,('task','axis','variant'),{'role':'validation','population':'all','aggregation':'complete_trial_macro'}),('SENSITIVITY_KEY_RESULTS.csv',sens,('group','intervention'),{'point':'FPR0.05','rule':'raw'})]:
 rows=read(D/filename);counts[filename]=len(rows)
 for row in rows:
  rr=[x for x in source if all(x[k]==row[k] for k in filterkeys) and all(x[k]==v for k,v in fixed.items())];assert len(rr)==12
  for k,v in row.items():
   if k in filterkeys:continue
   vals=np.array([num(x[k]) for x in rr]);vals=vals[np.isfinite(vals)];assert eq(num(v),float(vals.mean()) if len(vals) else float('nan')),(filename,row,k)
b=load(R/'benchmark/F_history_new.json')
for row in read(D/'DEPLOYMENT_COST.csv'):
 rr=b['measurements'][row['mode']];samples=np.array([x for r in rr for x in r['samples_ms']]);meds=[np.median(r['samples_ms']) for r in rr]
 assert eq(num(row['median_ms']),float(np.median(samples))) and eq(num(row['min_repeat_median_ms']),min(meds)) and eq(num(row['max_repeat_median_ms']),max(meds));assert int(row['peak_allocated_bytes'])==max(x['peak_allocated_bytes'] for x in rr) and int(row['full_parameters'])==b['total_deployment_parameters']
inv=load(R/'formal_delivery/ALL_RUN_INVENTORY.json');assert inv['status']=='pass' and inv['formal_training_count']==24 and inv['force']==inv['fusion']==12;grid={(k,f'htt_leave_p{f}',s) for k in ['force','fusion'] for f in range(1,5) for s in [20260914,20260915,20260916]};assert len(inv['runs'])==24 and {(r['kind'],r['fold'],r['seed']) for r in inv['runs']}==grid
idx=read(R/'formal_delivery/ALL_CHECKPOINT_INDEX.csv');assert len(idx)==72 and len({r['path'] for r in idx})==72
for row in inv['runs']:
 assert row['status']=='accepted' and pathlib.Path(row['log']).is_dir()
 for name,rec in row['artifacts'].items():assert sha(rec['path'])==rec['sha256'];ir=[x for x in idx if x['path']==rec['path']];assert len(ir)==1 and ir[0]['sha256']==rec['sha256'] and ir[0]['artifact']==name
print(json.dumps({'status':'pass','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'report_sources_checked':len(a['sources']),'report_output_hashes_checked':len(a['output_hashes']),'key_tables_all_numbers_recomputed':counts,'cost_sample_median_repeat_range_recomputed':True,'all24_run_inventory_and72_checkpoint_config_paths_verified':True,'report_sha256':sha(D/'SUMMARY_ZH.md'),'report_audit_sha256':sha(D/'REPORT_AUDIT.json'),'all_inventory_sha256':sha(R/'formal_delivery/ALL_RUN_INVENTORY.json'),'all_checkpoint_index_sha256':sha(R/'formal_delivery/ALL_CHECKPOINT_INDEX.csv')}))
