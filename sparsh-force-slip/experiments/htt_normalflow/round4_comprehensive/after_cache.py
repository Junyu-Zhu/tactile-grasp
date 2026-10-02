#!/usr/bin/env python3
"""Root-managed dependency gate: complete cache -> verified smoke -> 12 runs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser()
p.add_argument('--encoder',required=True)
p.add_argument('--cache-pid',required=True,type=int)
a=p.parse_args()
here=Path(__file__).resolve().parent
root=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round4_comprehensive/encoders')
cache=root/'cache'/a.encoder
deadline=time.monotonic()+7200
while not (cache/'cache_manifest.json').exists():
    if not Path(f'/proc/{a.cache_pid}').exists():
        raise RuntimeError('Cache process ended without complete manifest; inspect log, do not train')
    if time.monotonic()>deadline: raise TimeoutError('Cache gate exceeded two hours')
    time.sleep(10)
# run_all performs content verification before its smoke and formal jobs.
subprocess.run([sys.executable,str(here/'encoders/run_all.py'),'--encoder',a.encoder,
    '--cache-dir',str(cache),'--output-root',str(root),'--device','cuda:0',
    '--workers','2','--batch-size','128'],check=True)
