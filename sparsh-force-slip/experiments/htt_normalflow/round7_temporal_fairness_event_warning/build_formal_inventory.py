#!/usr/bin/env python3
import argparse,datetime,hashlib,json
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--code',type=Path,required=True);a=p.parse_args();r=a.root;c=a.code
 decision=json.loads((r/'audit/support_decision.json').read_text());assert decision['training_triggered']
 smoke=json.loads((r/'SMOKE_AUDIT.json').read_text());assert smoke['status']=='pass' and all(smoke['checks'].values())
 for n in ['reviews/PREPARE_INDEPENDENT_REVIEW.json','reviews/TRAINING_PIPELINE_INDEPENDENT_REVIEW.json']:
  d=json.loads((r/n).read_text());assert d['status']=='pass',n
 groups=['A_visual','B_force','C_force_delta']+(['D_visual_delta'] if decision['D_triggered'] else [])
 files=[c/n for n in ['training/train.py','training/protocol.json','NUMERIC_PROTOCOL.json','prepare/prepare.py','prepare/protocol.json','run_training_queue.py']]+[r/n for n in ['prepare/prepared.pt','prepare/PREPARE_AUDIT.json','audit/support_decision.json','audit/support_manifest.json','audit/history_dependency_audit.json','SMOKE_AUDIT.json','SMOKE_INTERRUPTION_EVIDENCE.json','SUPPORT_INDEPENDENT_REVIEW.json','reviews/PREPARE_INDEPENDENT_REVIEW.json','reviews/TRAINING_PIPELINE_INDEPENDENT_REVIEW.json']]
 frozen={str(p):sha(p) for p in files};prepared=r/'prepare/prepared.pt';assert frozen[str(prepared)]==smoke['data_sha256']
 runs=[{'id':f'future_{g}_{s}','group':g,'seed':s,'output':str(r/'formal'/f'future_{g}_{s}')} for g in groups for s in [20260914,20260915,20260916]]
 inv={'schema':'round7_formal_inventory_v1','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'output_root':str(r/'formal_queue'),'trainer':str(c/'training/train.py'),'prepared_data':str(prepared),'prepared_sha256':frozen[str(prepared)],'support_decision':str(r/'audit/support_decision.json'),'smoke_audit':str(r/'SMOKE_AUDIT.json'),'groups':groups,'horizons':decision['supported_horizons'],'required_run_checks':['finite_all_parameter_gradients','all_parameter_tensors_updated','active_input_columns_all_received_nonzero_data_gradient','inactive_input_columns_received_no_data_gradient','checkpoint_roundtrip_exact','selection_only_checkpoint_choice'],'frozen_inputs':frozen,'runs':runs}
 for q in [c/'FORMAL_INVENTORY.json',r/'FORMAL_INVENTORY.json']:
  if q.exists():raise ValueError('Do not overwrite immutable formal inventory')
  q.write_text(json.dumps(inv,indent=2)+'\n')
 files=[p for p in c.rglob('*') if p.is_file() and (p.name in ['PROTOCOL.md','NUMERIC_PROTOCOL.json','protocol.json'] or 'AMENDMENT' in p.name)]
 (r/'PROTOCOL_LOCK.json').write_text(json.dumps({'frozen_before_formal':True,'files':{str(p):sha(p) for p in files}},indent=2)+'\n')
 (r/'RUN_INVENTORY.json').write_text(json.dumps({'status':'formal_ready','training_runs_executed':0,'runs':[dict(x,status='pending') for x in runs]},indent=2)+'\n')
 print(json.dumps({'runs':len(runs),'inventory_sha256':sha(c/'FORMAL_INVENTORY.json'),'prepared_sha256':frozen[str(prepared)]}))
if __name__=='__main__':main()
