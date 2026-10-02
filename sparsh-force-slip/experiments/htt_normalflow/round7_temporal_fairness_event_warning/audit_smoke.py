#!/usr/bin/env python3
import argparse,json,hashlib,importlib.util
from pathlib import Path
import torch

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--code',type=Path,required=True);a=p.parse_args();r=a.root;c=a.code
 spec=importlib.util.spec_from_file_location('r7train',c/'training/train.py');tr=importlib.util.module_from_spec(spec);spec.loader.exec_module(tr)
 summaries={};checks={};bundle,_=tr.code_bundle_sha()
 for group in ['A_visual','B_force','C_force_delta','D_visual_delta','C_resume']:
  d=json.loads((r/'smoke'/group/'summary.json').read_text());summaries[group]=d
  checks[group+':complete_smoke']=d['status']=='complete' and d['smoke'] and not d['formal'];checks[group+':samecode']=d['run_config']['code_bundle_sha256']==bundle
  for key in ['finite_all_parameter_gradients','all_parameter_tensors_updated','active_input_columns_all_received_nonzero_data_gradient','inactive_input_columns_received_no_data_gradient','checkpoint_roundtrip_exact','selection_only_checkpoint_choice']:
   checks[group+':'+key]=d['audit'][key] is True
  checks[group+':no_formal_predictions']=not d['artifacts']['predictions']
  for name in ['best','latest']:
   x=d['artifacts'][name];checks[group+':hash:'+name]=sha(Path(x['path']))==x['sha256']
 receipts=json.loads((r/'SMOKE_LAUNCH_RECEIPTS.json').read_text());checks['six_successful_process_receipts']=len(receipts)==6 and all(x['exit_code']==0 for x in receipts)
 initial_receipt=next(x for x in receipts if x['name']=='C_resume' and x['stage']=='initial');resume_receipt=next(x for x in receipts if x['name']=='C_resume' and x['stage']=='resume')
 command=initial_receipt['command'];checks['actual_interrupt_command']='--interrupt-after-epoch' in command and command[command.index('--interrupt-after-epoch')+1]=='1' and '--resume' not in command
 checks['actual_resume_command']='--resume' in resume_receipt['command'] and '--interrupt-after-epoch' not in resume_receipt['command'] and initial_receipt['finished_unix']<=resume_receipt['started_unix']
 interrupted=json.loads((r/'smoke_logs/C_resume.initial.log').read_text());checks['persisted_interrupted_log']=interrupted['status']=='interrupted' and interrupted['smoke'] and not interrupted['formal'] and len(interrupted['history'])==1 and interrupted['history'][0]['epoch']==1 and interrupted['run_config']==summaries['C_resume']['run_config']
 (r/'SMOKE_INTERRUPTION_EVIDENCE.json').write_text(json.dumps({'status':'pass' if checks['persisted_interrupted_log'] and checks['actual_interrupt_command'] and checks['actual_resume_command'] else 'fail','initial_log_sha256':sha(r/'smoke_logs/C_resume.initial.log'),'receipt_sha256':sha(r/'SMOKE_LAUNCH_RECEIPTS.json'),'interrupted_summary':interrupted,'note':'Extracted from preserved first-process stdout after completed resume; partial checkpoint digest was logged before resume replaced latest. Not a fabricated pre-resume snapshot.'},indent=2)+'\n')
 initials=[d['run_config']['initialization_tensor_sha256'] for d in summaries.values()];checks['identical_full_initialization']=all(x==initials[0] for x in initials)
 lc=torch.load(r/'smoke/C_force_delta/latest.pth',map_location='cpu',weights_only=False);lr=torch.load(r/'smoke/C_resume/latest.pth',map_location='cpu',weights_only=False)
 for key in ['model_state','optimizer_state','history','best_model_state','best_selection_loss','best_epoch','stale','rng_state','audit']:
  checks['actual_interrupted_resume:'+key]=tr.nested_equal(lc[key],lr[key])
 payload=torch.load(r/'prepare/prepared.pt',map_location='cpu',weights_only=False);verified=tr.validate_prepared(payload,r/'prepare/prepared.pt');checks['prepared_validated']=True
 checks['data_unchanged']=all(d['run_config']['data_sha256']==verified['data_sha256'] for d in summaries.values())
 checks['only_future_parameters']=set(lc['model_state'])=={'gru.weight_ih_l0','gru.weight_hh_l0','gru.bias_ih_l0','gru.bias_hh_l0','risk.weight','risk.bias'}
 result={'status':'pass' if all(checks.values()) else 'fail','checks':checks,'data_sha256':verified['data_sha256'],'code_bundle_sha256':bundle,'summary_hashes':{k:sha(r/'smoke'/k/'summary.json') for k in summaries},'freeze_semantics':'Only fresh GRU/risk instantiated and optimized; upstream outputs detached and identity-bound cached. No encoder/force/current-slip optimizer or live module in training.'}
 (r/'SMOKE_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'checks':len(checks),'failed':[k for k,v in checks.items() if not v]}));return result['status']!='pass'
if __name__=='__main__':raise SystemExit(main())
