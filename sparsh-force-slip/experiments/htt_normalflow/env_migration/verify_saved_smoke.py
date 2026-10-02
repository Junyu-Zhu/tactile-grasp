"""Reload a real Phase-2 smoke checkpoint and optimizer on CPU."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ['XFORMERS_DISABLED'] = '1'
sys.path.insert(0, '/home/zjy/document/tactile-grasp/sparsh-force-slip/scripts')
import torch
import phase2_b_multitask as p2


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    model, payload = p2.load_b_checkpoint(a.checkpoint, torch.device('cpu'))
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=1e-4)
    optimizer.load_state_dict(payload['optimizer_state'])
    loader = p2.make_loader([p2.VAL_DATASETS[0]], 0, 2, 0, False, False, encoder='mae')
    batch = next(iter(loader))
    with torch.inference_mode():
        output = model(batch['image'])
    if not all(torch.isfinite(value).all() for value in output.values()):
        raise ValueError('Nonfinite reloaded predictions')
    if not optimizer.state:
        raise ValueError('Optimizer state was not restored')
    report = {'status': 'pass', 'checkpoint': str(a.checkpoint),
              'epoch': payload['epoch'], 'model_strict_load': True,
              'optimizer_states': len(optimizer.state), 'prediction_finite': True,
              'device': 'cpu', 'dataset': p2.VAL_DATASETS[0],
              'shapes': {k: list(v.shape) for k, v in output.items()}}
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
