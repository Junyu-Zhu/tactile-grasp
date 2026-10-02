#!/usr/bin/env python3
"""R20 true-change predictors with exact epoch recovery."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

GROUPS = ('K-current', 'D-history', 'T-visual', 'T-force')
SEEDS = (20260914, 20260915, 20260916)
HORIZONS = (1, 5, 10)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_save(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name+'.', suffix='.tmp')
    os.close(fd)
    try:
        torch.save(obj, tmp); os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_json(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name+'.', suffix='.tmp')
    os.close(fd)
    try:
        Path(tmp).write_text(json.dumps(obj, indent=2, sort_keys=True)+'\n'); os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class ChangeModel(nn.Module):
    def __init__(self, group):
        super().__init__()
        if group not in GROUPS: raise ValueError(group)
        self.group = group
        direct = group in GROUPS[:2]
        width = 48 if direct else 45
        input_width = 192 if group == 'T-visual' else 195
        self.input = nn.Sequential(nn.Linear(input_width, width), nn.GELU())
        self.gru = nn.GRU(width, width, batch_first=True)
        if direct:
            self.readout = nn.Linear(width, 9)
        else:
            self.transition = nn.Linear(width, width)
            self.readout = nn.Linear(width, 3)

    def forward(self, x, return_trajectory=False):
        if x.ndim != 3 or x.shape[1:] != (9, 195):
            raise ValueError(f'expected [batch,9,195], got {tuple(x.shape)}')
        # Two updates of the *same current* base state activate the GRU's
        # recurrent weights without introducing earlier observations.
        if self.group == 'K-current': x = x[:, -1:, :].repeat(1, 2, 1)
        if self.group == 'T-visual': x = x[..., :192]
        z = self.gru(self.input(x))[0][:, -1]
        if self.group in GROUPS[:2]:
            y = self.readout(z).reshape(-1, 3, 3)
            return (y, None) if return_trajectory else y
        sequence = []
        for _ in range(10):
            z = z + torch.tanh(self.transition(z))
            sequence.append(self.readout(z))
        trajectory = torch.stack(sequence, dim=1)
        y = trajectory[:, [0, 4, 9]]
        return (y, trajectory) if return_trajectory else y


def init_model(group, seed):
    torch.manual_seed(seed)
    m = ChangeModel(group)
    for name, p in m.named_parameters():
        if name.startswith('readout.') or p.ndim == 1: nn.init.zeros_(p)
        else: nn.init.xavier_uniform_(p)
    return m


def validate_data(data):
    if data['schema'] != 'round14_future_cache_v1' or tuple(data['horizons']) != HORIZONS:
        raise ValueError('incompatible accepted future cache')
    if set(data['roles']) != {'fit', 'selection', 'calibration', 'validation'}:
        raise ValueError('missing/unexpected role')
    seen = set(); keys = None
    for role, r in data['roles'].items():
        if r['x'].shape[1:] != (9,195) or r['y'].shape[1:] != (3,3) or r['y_current'].shape[1:] != (3,):
            raise ValueError(f'bad shape {role}')
        if not all(torch.isfinite(r[k]).all() for k in ('x','y','y_current')): raise ValueError('nonfinite cache')
        groups = set(r['leakage_group'])
        if seen & groups: raise ValueError('role leakage')
        seen |= groups
        if any(int(t)<13 for t in r['t']): raise ValueError('causal history incomplete')
        role_keys = {(e,int(t)) for e,t in zip(r['episode_id'],r['t'])}
        if len(role_keys) != len(r['t']): raise ValueError('duplicate endpoints')
        keys = role_keys if keys is None else keys | role_keys
    for k, ref in data['provenance']['immutable_upstream'].items():
        if sha(ref['path']) != ref['sha256']: raise ValueError(f'upstream hash mismatch {k}')
    for k in ('prepared','support'):
        if sha(data['provenance'][k]) != data['provenance'][k+'_sha256']:
            raise ValueError(f'provenance hash mismatch {k}')


def true_change(role):
    return role['y'] - role['y_current'][:,None,:]


def normalizer(fit):
    x = fit['x']; delta = true_change(fit)
    return {'x_mean': x.mean((0,1)), 'x_std': x.std((0,1),unbiased=False).clamp_min(1e-6),
            'delta_scale': delta.std((0,1),unbiased=False).clamp_min(1e-6)}


def norm_x(x, n): return (x-n['x_mean'])/n['x_std']
def native_change(y, n): return y*n['delta_scale']


def predict_batches(model, x, n, device, trajectory=False):
    model.eval(); preds = []; traj = []
    with torch.inference_mode():
        for xb in x.split(2048):
            output = model(xb.to(device), return_trajectory=trajectory)
            if trajectory:
                y, sequence = output
                traj.append(sequence.cpu())
            else: y = output
            preds.append(native_change(y.cpu(), n))
    if trajectory: return torch.cat(preds), torch.cat(traj)
    return torch.cat(preds)


def train(data, output, group, fold, seed, device='cpu', max_epochs=60, patience=10,
          interrupt_after=None):
    if group not in GROUPS or fold != 1 or seed not in SEEDS: raise ValueError('unregistered run')
    validate_data(data)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    fit = data['roles']['fit']; sel = data['roles']['selection']; n = normalizer(fit)
    xfit = norm_x(fit['x'], n); yfit = true_change(fit)/n['delta_scale']
    xsel = norm_x(sel['x'], n)
    model = init_model(group,seed).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    sampler = torch.Generator().manual_seed(seed+2000)
    norm_hash = hashlib.sha256(b''.join(n[k].numpy().tobytes() for k in ('x_mean','x_std','delta_scale'))).hexdigest()
    identity = {'group':group,'fold':fold,'seed':seed,'source_sha256':sha(__file__),
                'protocol_sha256':sha(Path(__file__).with_name('PROTOCOL.md')),
                'cache_sha256':data['round14_data_sha256'],'norm_sha256':norm_hash,
                'max_epochs':max_epochs,'patience':patience,'batch':256,'lr':.001,'weight_decay':.0001,
                'target':'true_clipped_delta_scale_only','selection':'true_delta_native_mae'}
    latest = out/'latest.pth'; best = out/'best.pth'
    start=0; best_metric=float('inf'); best_epoch=-1; wait=0; history=[]; best_state=None
    if latest.exists():
        ck = torch.load(latest,map_location='cpu',weights_only=False)
        if ck['identity'] != identity: raise ValueError('resume identity mismatch')
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
        for state in opt.state.values():
            for k,v in state.items():
                if torch.is_tensor(v): state[k]=v.to(device)
        sampler.set_state(ck['sampler_rng']); random.setstate(ck['python_rng']); np.random.set_state(ck['numpy_rng'])
        torch.set_rng_state(ck['torch_rng'])
        if torch.cuda.is_available() and ck['cuda_rng'] is not None: torch.cuda.set_rng_state_all(ck['cuda_rng'])
        start=ck['epoch']+1; best_metric=ck['best_metric']; best_epoch=ck['best_epoch']; wait=ck['wait']; history=ck['history']
        best_state=ck['best_state']
        # latest is authoritative; a crash between two atomic writes is repaired here.
        atomic_save({'identity':identity,'model':best_state,'normalizer':n,'best_epoch':best_epoch,
                     'best_metric':best_metric},best)
    for epoch in range(start,max_epochs):
        if wait >= patience: break
        model.train(); losses=[]
        for ix in torch.randperm(len(xfit),generator=sampler).split(256):
            opt.zero_grad(set_to_none=True)
            pred=model(xfit[ix].to(device)); loss=nn.functional.smooth_l1_loss(pred,yfit[ix].to(device),beta=1)
            if not torch.isfinite(loss): raise ValueError('nonfinite loss')
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('nonfinite gradient')
            opt.step(); losses.append(float(loss.detach()))
        delta = predict_batches(model,xsel,n,device)
        metric=float((delta-true_change(sel)).abs().mean())
        improved=metric<best_metric
        if improved:
            best_metric=metric;best_epoch=epoch;wait=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: wait+=1
        history.append({'epoch':epoch,'fit_loss':float(np.mean(losses)),'selection_true_delta_mae_n':metric})
        ck={'identity':identity,'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
            'optimizer':opt.state_dict(),'sampler_rng':sampler.get_state(),'python_rng':random.getstate(),
            'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'normalizer':n,'epoch':epoch,'best_metric':best_metric,'best_epoch':best_epoch,
            'wait':wait,'history':history,'best_state':best_state}
        atomic_save(ck,latest)
        if os.environ.get('R20_SMOKE_KILL_AFTER_LATEST_EPOCH') == str(epoch):
            os._exit(91)
        if improved:
            atomic_save({'identity':identity,'model':best_state,'normalizer':n,
                         'best_epoch':best_epoch,'best_metric':best_metric},best)
        if interrupt_after is not None and epoch>=interrupt_after:
            result={'status':'interrupted','epochs':len(history),'identity':identity}
            atomic_json(result,out/'summary.json');return result
    if best_epoch<0: raise ValueError('no valid checkpoint')
    result={'status':'complete','best_epoch':best_epoch,'best_metric_true_delta_n_mae':best_metric,
            'epochs':len(history),'parameters':sum(p.numel() for p in model.parameters()),
            'state_sha256':state_hash(torch.load(latest,map_location='cpu',weights_only=False)['model']),
            'identity':identity}
    atomic_json(result,out/'summary.json');return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--group',choices=GROUPS,required=True);p.add_argument('--fold',type=int,required=True)
    p.add_argument('--seed',type=int,choices=SEEDS,required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--max-epochs',type=int,default=60);p.add_argument('--patience',type=int,default=10)
    p.add_argument('--interrupt-after',type=int)
    a=p.parse_args();data=torch.load(a.data,map_location='cpu',weights_only=False)
    data['round14_data_sha256']=sha(a.data)
    print(json.dumps(train(data,a.output,a.group,a.fold,a.seed,a.device,a.max_epochs,a.patience,a.interrupt_after)))

if __name__=='__main__': main()
