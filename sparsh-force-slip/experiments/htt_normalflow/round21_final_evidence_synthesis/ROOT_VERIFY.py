#!/usr/bin/env python3
"""Final read-only R21 manifest verifier; no model/data loading or mutations."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

manifest = json.loads((ROOT / 'ROOT_MANIFEST.json').read_text())
errors = []
for record in manifest['files']:
    path = ROOT / record['relative_path']
    if not path.is_file():
        errors.append({'file': record['relative_path'], 'error': 'missing'})
    elif path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
        errors.append({'file': record['relative_path'], 'error': 'identity_mismatch'})
for record in manifest['history_acceptance_files']:
    path = ROOT.parent / record['experiment_relative_path']
    if not path.is_file() or sha(path) != record['sha256']:
        errors.append({'history': record['experiment_relative_path'], 'error': 'missing_or_changed'})
status = json.loads((ROOT / 'FINAL_STATUS.json').read_text())
if status['status'] != 'complete' or status['new_training'] != 0 or status['test_consumed']:
    errors.append({'error': 'final_status_or_scope_invalid'})
print(json.dumps({'status': 'pass' if not errors else 'fail',
                  'files_checked': len(manifest['files']),
                  'historical_acceptances_checked': len(manifest['history_acceptance_files']),
                  'manifest_sha256': sha(ROOT / 'ROOT_MANIFEST.json'),
                  'errors': errors}, ensure_ascii=False))
sys.exit(bool(errors))
