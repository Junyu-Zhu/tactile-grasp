#!/usr/bin/env python3
"""Root-owned representative raw-image cold/streaming inference measurement."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('XFORMERS_DISABLED', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / 'scripts'))
import phase2_b_multitask as p2

spec = importlib.util.spec_from_file_location('r8_train_benchmark', HERE / 'training/train.py')
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--accepted-manifest',type=Path,required=True)
    parser.add_argument('--checkpoint-index',type=Path,required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--source-checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(42)
    payload = torch.load(args.data, map_location='cpu', weights_only=False)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = ckpt['run_config']
    accepted=json.loads(args.accepted_manifest.read_text())
    proof=accepted.get('training_audit',{})
    if accepted.get('status')!='complete' or train.sha256(Path(proof['path']))!=proof['sha256'] or json.loads(Path(proof['path']).read_text()).get('status')!='pass':
        raise ValueError('Accepted training audit required')
    if Path(accepted['checkpoint_index']['path']).resolve()!=args.checkpoint_index.resolve() or accepted['checkpoint_index']['sha256']!=train.sha256(args.checkpoint_index):
        raise ValueError('Checkpoint index identity mismatch')
    import csv
    index=list(csv.DictReader(args.checkpoint_index.open()))
    matches=[r for r in index if r['kind']=='best' and r['group']==config['group'] and int(r['seed'])==config['seed'] and Path(r['path']).resolve()==args.checkpoint.resolve()]
    if len(matches)!=1 or matches[0]['sha256']!=train.sha256(args.checkpoint):
        raise ValueError('Checkpoint not indexed as accepted best')
    if any(train.sha256(Path(p))!=h for p,h in config['source_hashes'].items()):
        raise ValueError('Training source identity changed')
    gpu_before=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv'],text=True)
    if config['mode'] != 'formal' or config['prepared_sha'] != train.sha256(args.data):
        raise ValueError('Invalid formal checkpoint/prepared identity')
    expected_sources={str(Path(item['checkpoint']).resolve()) for item in payload['provenance']['signed_force']['parents']}
    if expected_sources != {str(args.source_checkpoint.resolve())}:
        raise ValueError('Source checkpoint is not the prepared signed-force parent')
    upstream_proof=json.loads((HERE/'UPSTREAM_IDENTITY.json').read_text())
    if upstream_proof['actual_sha256'].get(str(args.source_checkpoint.resolve()))!=train.sha256(args.source_checkpoint):
        raise ValueError('Source checkpoint SHA differs from accepted R7 parent')
    group = config['group']
    head = train.make_model(group).eval().requires_grad_(False).to(args.device)
    head.load_state_dict(ckpt['model_state'], strict=True)
    head_before={k:train.tensor_sha(v) for k,v in head.state_dict().items()}
    upstream, _ = p2.load_b_checkpoint(args.source_checkpoint, torch.device(args.device))
    upstream.eval().requires_grad_(False)
    before = {k: train.tensor_sha(v) for k, v in upstream.state_dict().items()}
    row = payload['timelines']['outer']
    idx = min(range(len(row['t'])), key=lambda i: (row['episode_id'][i], int(row['t'][i])))
    episode, endpoint = row['episode_id'][idx], int(row['t'][idx])
    _, dataset_name, trajectory = episode.split('/', 2)
    dataset = p2.make_dataset(dataset_name, slip_horizon=0, encoder='mae')
    traj = next(k for k in dataset.trajectories if str(k) == trajectory)
    scale = torch.as_tensor(dataset.max_abs_forceXYZ, device=args.device)
    norm = payload['normalization']
    base_mean, base_std = [norm['base'][k].to(args.device) for k in ('mean', 'std')]
    delta_mean, delta_std = [norm['force_delta'][k].to(args.device) for k in ('mean', 'std')]

    def base(s):
        # Exact existing raw preprocessing (including background and temporal image pair).
        image = dataset._get_tactile_images(traj, s).unsqueeze(0).to(args.device)
        tokens = upstream.encoder(image)
        output = upstream.decoder(tokens)
        # Historic pooled visual cache was stored fp16; preserve that input convention.
        z = tokens.mean(1).half().float()
        probability = torch.softmax(output['slip'], 1)[:, 1:2]
        force = output['force'] * scale
        return torch.cat((z, probability, force), 1)

    def assemble(bases):
        raw = torch.stack(bases, 1)
        normalized = (raw - base_mean) / base_std
        delta = torch.zeros((1, 9, 3), device=args.device)
        if group != 'B_xyz':
            delta[:, 5:] = (raw[:, 5:, 769:772] - raw[:, :4, 769:772] - delta_mean) / delta_std
        valid = torch.zeros((1, 9, 1), device=args.device)
        valid[:, 5:] = 1
        return torch.cat((normalized, delta, valid), 2)

    def sync():
        if args.device.startswith('cuda'):
            torch.cuda.synchronize(args.device)

    cfg = json.loads((HERE / 'NUMERIC_PROTOCOL.json').read_text())['e2e']
    results = {}
    analysis_path=HERE/'ANALYSIS_PROTOCOL.json'
    parity_limits=json.loads(analysis_path.read_text())['e2e_parity_limits']
    with torch.inference_mode():
        reference_bases = [base(s) for s in range(endpoint - 8, endpoint + 1)]
        fresh = assemble(reference_bases)
        cache_x = train.assemble_inputs(payload, {k: (v[idx:idx+1] if torch.is_tensor(v) else v) for k, v in row.items()}, group).to(args.device)
        identity = {'fresh_vs_cache_max_abs': float((fresh-cache_x).abs().max()),
                    'fresh_vs_cache_mean_abs': float((fresh-cache_x).abs().mean()),
                    'note': 'FP32 device and historical batch/cache quantization differences reported explicitly'}
        identity['blocks_max_abs']={name:float((fresh[:,:,start:end]-cache_x[:,:,start:end]).abs().max()) for name,start,end in [('visual',0,768),('p_slip',768,769),('force_xyz',769,772),('delta_xyz',772,775),('valid',775,776)]}
        identity['prediction_max_abs'] = float((train.model_probabilities(head(fresh), group, [1,3,5])[0] - train.model_probabilities(head(cache_x), group, [1,3,5])[0]).abs().max())
        identity['acceptance_limits']=parity_limits
        identity['pass']=bool(torch.allclose(fresh,cache_x,atol=parity_limits['input_atol'],rtol=parity_limits['input_rtol'])) and identity['prediction_max_abs']<=parity_limits['risk_max_abs']
        if not identity['pass']:
            train.atomic_json(args.out, {'status':'failed_cache_parity','diagnostics':identity,'source_checkpoint_sha256':train.sha256(args.source_checkpoint)})
            raise ValueError('Fresh upstream/cache mismatch; diagnose before measuring deployment')
        for mode in ('cold_history', 'streaming'):
            blocks = []
            for repeat in range(cfg['repeats']):
                if args.device.startswith('cuda'):
                    torch.cuda.reset_peak_memory_stats(args.device)
                times = []
                for step in range(cfg['warmup'] + cfg['steps']):
                    sync()
                    start = time.perf_counter()
                    bases = [base(s) for s in range(endpoint-8, endpoint+1)] if mode == 'cold_history' else reference_bases[:-1] + [base(endpoint)]
                    x = assemble(bases)
                    probability = train.model_probabilities(head(x), group, [1,3,5])[0]
                    sync()
                    elapsed = (time.perf_counter() - start) * 1000
                    if not torch.isfinite(probability).all():
                        raise ValueError('Nonfinite end-to-end output')
                    if step >= cfg['warmup']:
                        times.append(elapsed)
                blocks.append({'repeat': repeat, 'samples_ms': times, 'median_ms': float(np.median(times)),
                               'p10_ms': float(np.quantile(times, .1)), 'p90_ms': float(np.quantile(times, .9)),
                               'peak_allocated_bytes': torch.cuda.max_memory_allocated(args.device) if args.device.startswith('cuda') else 0})
            results[mode] = blocks
    frozen = all(train.tensor_sha(v) == before[k] for k, v in upstream.state_dict().items())
    head_frozen=all(train.tensor_sha(v)==head_before[k] for k,v in head.state_dict().items())
    if not frozen or not head_frozen:
        raise ValueError('Upstream mutated during inference')
    result = {'status': 'complete', 'group': group, 'seed': config['seed'], 'episode': episode, 't': endpoint,
              'source_checkpoint': str(args.source_checkpoint), 'source_sha256': train.sha256(args.source_checkpoint),
              'checkpoint': str(args.checkpoint), 'checkpoint_sha256': train.sha256(args.checkpoint),
              'prepared_sha256': train.sha256(args.data), 'code_sha256': train.sha256(Path(__file__)),
              'analysis_protocol_sha256':train.sha256(analysis_path),'accepted_manifest_sha256':train.sha256(args.accepted_manifest),'checkpoint_index_sha256':train.sha256(args.checkpoint_index),
              'device':args.device,'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'torch_version':torch.__version__,'cuda_version':torch.version.cuda,
              'target_gpu':torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else None,
              'gpu_conditions_before':gpu_before,'head_frozen':head_frozen,
              'dataset_name':dataset_name,'trajectory':trajectory,'raw_dependency_range':[endpoint-13,endpoint],
              'dataset_config':p2.OmegaConf.to_container(p2.dataset_cfg(0,'mae'),resolve=True),'reference':'existing dataset preprocessing reference, before trajectory',
              'upstream_frozen': frozen, 'cache_identity_diagnostics': identity,
              'head_parameters': sum(p.numel() for p in head.parameters()),
              'encoder_parameters': sum(p.numel() for p in upstream.encoder.parameters()),
              'upstream_parameters': sum(p.numel() for p in upstream.parameters()),
              'measurement': results, 'config': cfg,
              'semantics': 'resident compressed raw frames, image decode/background/resize + encoder + force/current-slip + future; cold nine base calls, streaming one with eight historical outputs resident; excludes dataset IO, capture and weight load',
              'gpu_conditions_after': subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv'], text=True)}
    train.atomic_json(args.out, result)
    print(json.dumps({'status': 'complete', 'group': group, 'identity': identity}))


if __name__ == '__main__':
    main()
