#!/usr/bin/env python3
"""Bind accepted P0 artifacts and immutable inputs before formal dispatch."""
import argparse
import datetime
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('r8_inventory_train', HERE/'training/train.py')
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-root', type=Path, required=True)
    a = p.parse_args()
    root = a.output_root
    data = root/'prepare/prepared.pt'
    smoke_path = root/'SMOKE_AUDIT.json'
    smoke = json.loads(smoke_path.read_text())
    if smoke['status'] != 'pass' or smoke['real_prepared_sha256'] != train.sha256(data):
        raise ValueError('Unaccepted smoke/prepared')
    if smoke['trainer_sha256'] != train.sha256(HERE/'training/train.py') or smoke['training_protocol_sha256'] != train.sha256(HERE/'training/protocol.json'):
        raise ValueError('Trainer changed after smoke')
    review_path = root/'reviews/P0_INDEPENDENT_REVIEW.json'
    review = json.loads(review_path.read_text())
    if review.get('status') != 'pass':
        raise ValueError('Independent P0 review not passed')
    decision_path = root/'SUPPORT_DECISION.json'
    decision = json.loads(decision_path.read_text())
    groups = ['B_xyz','C_xyz_delta','C_hazard']
    if decision.get('P3_triggered'):
        groups += ['P3_concat_independent','P3_fusion_independent','P3_fusion_hazard']
    if decision.get('P4_triggered'):
        groups += ['P4_direct','P4_state']
    if not set(groups).issubset(smoke['groups']):
        raise ValueError('Planned group did not smoke')
    paths = [data,root/'prepare/PREPARE_AUDIT.json', root/'prepare/P4_SUPPORT_DECISION.json',smoke_path,review_path,decision_path]
    paths += [HERE/name for name in ('USER_REQUEST.md','ANALYSIS_PROTOCOL.json','UPSTREAM_IDENTITY.json','NUMERIC_PROTOCOL.json','PROTOCOL.md','training/train.py','training/protocol.json','prepare/prepare.py','prepare/protocol.json','evaluation/protocol.json','run_training_queue.py','build_inventory.py')]
    frozen = {str(path.resolve()):train.sha256(path) for path in paths}
    inv = {'schema':'round8_formal_inventory_v1','groups':groups,'horizons':[1,3,5],
           'output_root':str(root/'formal_queue'), 'trainer':str(HERE/'training/train.py'),
           'prepared_data':str(data),'prepared_sha256':train.sha256(data),
           'support_decision':str(decision_path),'smoke_audit':str(smoke_path), 'frozen_inputs':frozen,
           'required_run_checks':['finite_all_parameter_gradients','all_parameter_tensors_updated','active_input_columns_all_received_nonzero_data_gradient','inactive_input_columns_received_no_data_gradient','checkpoint_roundtrip_exact','selection_only_checkpoint_choice'],
           'runs':[{'id':f'future_{g}_{s}','group':g,'seed':s,'output':str(root/'formal'/f'future_{g}_{s}')} for g in groups for s in (20260914,20260915,20260916)]}
    destination = root/'FORMAL_INVENTORY.json'
    if destination.exists() and json.loads(destination.read_text()) != inv:
        raise ValueError('Existing inventory differs; preserve and diagnose')
    train.atomic_json(destination,inv)
    train.atomic_json(root/'PROTOCOL_LOCK.json',{'locked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'inventory_sha256':train.sha256(destination),'files':frozen,'formal_runs':len(inv['runs'])})
    print(json.dumps({'status':'locked','runs':len(inv['runs'])}))


if __name__ == '__main__':
    main()
