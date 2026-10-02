#!/usr/bin/env python3
"""Single raw-frame-pair end-to-end latency, separate from cached-head timing."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'encoders'))
import cache
import training


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--encoder',required=True,choices=['mae','dino','ijepa','mae_letterbox'])
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    cache.configure_determinism(); torch.manual_seed(42)
    device=torch.device(args.device)
    model=cache.p2.FrozenEncoderSharedForceSlip(cache.canonical_encoder(args.encoder),'decoupled')
    encoder=model.encoder.eval().requires_grad_(False).to(device)
    head=training.SlipBranch(model.decoder)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    config=saved['config']
    assert config['fold']=='htt_leave_p1' and config['seed']==20260914
    assert config['smoke'] is False and config['init']=='fresh'
    if args.encoder=='mae':
        assert config['source_checkpoint_sha256']=='850e4a74e6d9bd60e8ed22efc8b70d343c5d21bb5f60b3053b0a800674dffcb4'
        assert cache.tensor_state_sha256(encoder.state_dict())=='60ab1af933e4c9a6791e8090554ded0f2229f09c5303a2a2a97ba66922c1eea8'
    else:
        assert config['encoder']==args.encoder
    head.load_state_dict(saved['branch_state'],strict=True)
    head=head.eval().requires_grad_(False).to(device)
    del model
    root=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow')
    manifest=json.loads((root/'round1/splits.json').read_text())
    eligible=set(manifest['splits']['htt_leave_p1']['calibration'])
    row=sorted([r for r in manifest['episodes'] if r['id'] in eligible and r.get('task')=='slip'],key=lambda r:r['id'])[0]
    episode=cache.adapters.load_htt(row['path'])
    t=min(10,len(episode.images)-1)
    condition=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv'],text=True)
    def prepare_image():
        if args.encoder=='mae_letterbox':return cache.encoder_input(episode,t,args.encoder)
        # Deployment performs image preparation only, not adapter target bookkeeping.
        return torch.cat([cache.adapters.preprocess(episode.images[i],episode.reference)
                          for i in (t,max(t-5,0))])
    assert torch.equal(prepare_image(),cache.encoder_input(episode,t,args.encoder))
    def infer():
        image=prepare_image().unsqueeze(0).to(device)
        return head(encoder(image))
    with torch.inference_mode():
        for _ in range(20): infer()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        samples=[]
        for _ in range(100):
            torch.cuda.synchronize(device); start=time.perf_counter()
            output=infer(); torch.cuda.synchronize(device)
            samples.append((time.perf_counter()-start)*1000)
        assert torch.isfinite(output).all()
        memory=torch.cuda.max_memory_allocated(device)
    result=dict(encoder=args.encoder,checkpoint=str(args.checkpoint),checkpoint_sha256=cache.sha256_file(args.checkpoint),
        encoder_state_sha256=cache.tensor_state_sha256(encoder.state_dict()),
        episode_id=row['id'],frame=t,device=str(device),batch_size=1,warmups=20,repeats=100,
        milliseconds=samples,median_ms=float(np.median(samples)),p90_ms=float(np.quantile(samples,.9)),
        min_ms=min(samples),max_ms=max(samples),peak_allocated_bytes=memory,
        encoder_parameters=sum(p.numel() for p in encoder.parameters()),
        slip_parameters=sum(p.numel() for p in head.parameters()),
        image_only_preprocessing_exactly_matches_cache=True,
        gpu_condition=condition,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        process_local_device_index=device.index,
        gpu_condition_after=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv'],text=True),
        torch=torch.__version__,source_sha256=cache.sha256_file(Path(__file__)),
        scope='raw in-memory images/reference -> preprocessing two frames -> host-to-device -> frozen encoder -> slip logits; synchronized FP32 math SDPA',
        exclusions='disk/video decoding, camera capture and actuator I/O; no physical real-time guarantee on shared GPU')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result,indent=2)); os.replace(temporary,args.output)
    print(json.dumps({k:v for k,v in result.items() if k!='milliseconds'}))


if __name__=='__main__':main()
