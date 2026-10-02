import sys,json,torch
from pathlib import Path
C=Path('/home/zjy/document/tactile-grasp/sparsh-force-slip/experiments/htt_normalflow/round17_htt_shared_temporal_multitask');sys.path.insert(0,str(C));import multitask_train as mt
torch.set_num_threads(4);d=torch.load('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round17_htt_shared_temporal_multitask/prepared/p4_s20260915/prepared.pt',map_location='cpu',weights_only=False);j=mt.fit_assets(d['roles'],20260915)[3];old=d['joint_weight'];print(json.dumps({'saved':old,'recomputed':j,'deltas':{k:j[k]-old[k] for k in ['slip_loss_initial','future_loss_initial','lambda_future']}},indent=2))
