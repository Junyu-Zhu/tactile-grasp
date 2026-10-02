"""Snapshot environment versions and pin all non-GPU migration dependencies."""
import argparse
import importlib.metadata as metadata
import json
import platform
import sys
from pathlib import Path

ALLOWED = {'torch', 'torchvision', 'torchaudio', 'xformers', 'triton', 'sympy'}


def permitted(name):
    return name in ALLOWED or name.startswith('nvidia-')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--constraints', type=Path)
    p.add_argument('--reference', type=Path)
    a = p.parse_args()
    packages = {d.metadata['Name'].lower().replace('_', '-'): d.version
                for d in metadata.distributions() if d.metadata['Name']}
    result = {'python': platform.python_version(), 'prefix': sys.prefix,
              'packages': dict(sorted(packages.items()))}
    if a.reference:
        old = json.loads(a.reference.read_text())
        differences = {name: [old['packages'].get(name), packages.get(name)]
                       for name in sorted(old['packages'].keys() | packages.keys())
                       if old['packages'].get(name) != packages.get(name)}
        unexpected = {k: v for k, v in differences.items() if not permitted(k)}
        result.update(differences=differences, unexpected_differences=unexpected,
                      python_unchanged=old['python'] == result['python'])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    if a.constraints:
        a.constraints.write_text(''.join(f'{k}=={v}\n' for k, v in sorted(packages.items())
                                         if not permitted(k)))
    print(json.dumps({k: v for k, v in result.items() if k != 'packages'}))
    if a.reference and (unexpected or not result['python_unchanged']):
        raise SystemExit('Unexpected non-GPU dependency or Python change')


if __name__ == '__main__':
    main()
