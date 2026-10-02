#!/usr/bin/env python3
"""Lock the complete 75-run R16 formal inventories after cache completion."""
import hashlib,json
from pathlib import Path
from touchd_common import atomic_json,sha256

C=Path(__file__).resolve().parent;HTT=C.parent;R=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow');O=R/'round16_touchd_force_transfer';PY='/home/zjy/miniconda3/envs/sparsh/bin/python'
SEEDS=(20260914,20260915,20260916)

def main():
 cache=O/'cache/cache_manifest.json';d=json.loads(cache.read_text());assert d['status']=='complete' and d['formal'] and d['samples']==80783 and d['encoder_frozen_bitwise']
 smoke=json.loads((O/'smoke/PIPELINE_SMOKE.json').read_text());assert smoke['status']=='pass' and not smoke['formal'] and smoke['smoke_inputs_forbidden_formal']
 pair_order=json.loads((C/'PAIR_ORDER_AUDIT.json').read_text());assert pair_order['status']=='pass' and pair_order['pairs']==80783
 sources=[C/x for x in ('PROTOCOL.md','EVALUATION_PROTOCOL.json','BUDGET.json','CACHE_LOCK.json','PAIR_ORDER_AUDIT.json','PREFORMAL_REVISION.json','SHARDED_RESUME_SMOKE.json','touchd_common.py','train_touchd_force.py','train_htt_force.py','export_htt_force.py','run_downstream_pair.py','run_stage.py','run_formal_pipeline.py','evaluate_force.py','evaluate_future_diagnostics.py','verify_sharded_resume.py')]
 sources += [HTT/'round10_htt_force_supervision_adaptation/fusion/replace_force.py',HTT/'round12_class_preserving_trial_balance/training/train.py',HTT/'round12_class_preserving_trial_balance/training/protocol.json',HTT/'round14_htt_future_force_dual/future_train.py',HTT/'round14_htt_future_force_dual/PROTOCOL.md',cache,O/'audit/TOUCHD_GELSIGHT_AUDIT.json',O/'smoke/PIPELINE_SMOKE.json']
 locked={str(p):sha256(p) for p in sources}
 tj=[]
 for seed in SEEDS:
  out=O/f'formal/touchd/s{seed}';tj.append({'id':f'T_s{seed}','command':[PY,str(C/'train_touchd_force.py'),'--cache',str(cache),'--output',str(out),'--seed',str(seed),'--device','cuda:0'],'receipt':str(out/'summary.json')})
 fj=[]
 for route in ('H','T_H'):
  for fold in range(1,5):
   for seed in SEEDS:
    out=O/f'formal/htt_force/{route}_p{fold}_s{seed}';cmd=[PY,str(C/'train_htt_force.py'),'--manifest',str(R/f'round10_htt_force_supervision_adaptation/force_support/fold_p{fold}.json'),'--route',route,'--fold',f'htt_leave_p{fold}','--seed',str(seed),'--output',str(out),'--device','cuda:0','--workers','2']
    if route=='T_H':cmd += ['--t-private',str(O/f'formal/touchd/s{seed}/transferable_private_state.pth')]
    fj.append({'id':f'{route}_p{fold}_s{seed}','command':cmd,'receipt':str(out/'training_summary.json')})
 pj=[]
 for route in ('H','T_H'):
  for fold in range(1,5):
   for seed in SEEDS:
    key=f'{route}_p{fold}_s{seed}';pj.append({'id':key,'command':[PY,str(C/'run_downstream_pair.py'),'--route',route,'--fold',str(fold),'--seed',str(seed)],'receipt':str(O/f'formal/pairs/{key}.json')})
 common={'schema':'round16_inventory_v1','status':'locked','source_hashes':locked,'cache_manifest_sha256':sha256(cache),'smoke_sha256':sha256(O/'smoke/PIPELINE_SMOKE.json'),'stop_new_dispatch':'2026-09-23T01:55:53+08:00','formal_only':True}
 atomic_json(O/'inventories/TOUCHD.json',{**common,'stage':'A_touchd','neural_runs':3,'jobs':tj})
 atomic_json(O/'inventories/HTT_FORCE.json',{**common,'stage':'B_htt_force','neural_runs':24,'jobs':fj,'requires':str(O/'queues/touchd/status.json')})
 atomic_json(O/'inventories/DOWNSTREAM.json',{**common,'stage':'C_D_downstream','neural_runs':48,'job_pairs':24,'jobs':pj,'requires':str(O/'queues/htt_force/status.json')})
 ready={'schema':'round16_dispatch_ready_v1','status':'ready_for_root_formal_decision','formal_runs':75,'inventory_counts':{'touchd':3,'htt_force':24,'current_slip':24,'future_force':24},'cache_manifest_sha256':sha256(cache),'protocol_sha256':sha256(C/'PROTOCOL.md'),'budget_sha256':sha256(C/'BUDGET.json'),'pipeline_smoke_sha256':sha256(O/'smoke/PIPELINE_SMOKE.json'),'inventory_hashes':{p.name:sha256(p) for p in (O/'inventories').glob('*.json')},'formal_authorized':False,'smoke_weights_reused':False}
 atomic_json(O/'DISPATCH_READY.json',ready);print(json.dumps(ready,indent=2))
if __name__=='__main__':main()
