"""Bind the verified Round-2 delivery and its analysis source files."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
OUT = Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round2')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    required = ['EXECUTION_STATE.json', 'cache/index.json', 'cache_verification.json',
                'future_reconstruction_verification.json', 'metrics/COMPLETE.json',
                'report/CONCLUSIONS_ZH.md', 'report/REPORT.md', 'report/REPRODUCE.md',
                'report/detection_comparison.csv', 'report/future_comparison.csv',
                'report/failure_cases.csv', 'report/future_failure_cases.csv',
                'reports/COMPLETION_AUDIT.md', 'reports/METRICS_REVIEW.md',
                'environment/ENVIRONMENT_REPORT.md', 'environment/gpu_snapshot_delivery.txt',
                'events/episode_event_support.csv', 'events/group_event_support.csv']
    for name in required:
        if not (OUT / name).is_file():
            raise ValueError(f'Missing deliverable: {name}')
    for name in ['cache_runner.exit', 'cache_verification.exit', 'postprocess.exit']:
        if (OUT / name).read_text().strip() != '0':
            raise ValueError(f'Unsuccessful execution: {name}')
    index = json.loads((OUT / 'cache/index.json').read_text())
    verification = json.loads((OUT / 'cache_verification.json').read_text())
    complete = json.loads((OUT / 'metrics/COMPLETE.json').read_text())
    if (index['status'] != 'complete' or len(index['episodes']) != 272
            or verification['status'] != 'passed'
            or digest(OUT / 'cache/index.json') != complete['cache_index_sha256']
            or verification['index_sha256'] != complete['cache_index_sha256']):
        raise ValueError('Delivery inputs are not verified complete')
    for path, expected in {**index['code_hashes'], **index['sparsh_source_hashes']}.items():
        if digest(path) != expected:
            raise ValueError(f'Inference source changed: {path}')
    files = {OUT / name for name in required}
    for folder in ['metrics', 'report', 'reports', 'environment', 'events']:
        files.update(p for p in (OUT / folder).rglob('*') if p.is_file())
    files.update(p for p in OUT.glob('*.log') if p.is_file())
    files.update(p for p in OUT.glob('*.exit') if p.is_file())
    files.add(OUT.parent / 'PROJECT_STATE.md')
    sources = [p for p in CODE.rglob('*') if p.is_file()
               and p.suffix in {'.py', '.md', '.sh', '.json'}
               and 'results' not in p.relative_to(CODE).parts
               and '__pycache__' not in p.parts]
    manifest = {
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'verified_delivery', 'episodes': 272, 'frames': 56077,
        'gpu_runtime_validation': 'not_run_resource_unavailable_CPU_fallback',
        'cache_index_sha256': complete['cache_index_sha256'],
        'split_manifest_sha256': complete['split_manifest_sha256'],
        'stage1_checkpoint_sha256': index['checkpoint_sha256'],
        'future_checkpoint_sha256': index['future_checkpoint_sha256'],
        'artifact_sha256': {str(p.relative_to(OUT)): digest(p) for p in sorted(files) if p.is_relative_to(OUT)},
        'project_state_sha256': digest(OUT.parent / 'PROJECT_STATE.md'),
        'source_sha256': {str(p.relative_to(CODE)): digest(p) for p in sorted(sources)},
        'cache_note': 'Per-NPZ and source-data hashes are in cache/index.json; full verification passed. Local mirror retains index and reports, not all NPZ arrays.'
    }
    (OUT / 'DELIVERY.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'artifacts': len(manifest['artifact_sha256']),
                      'sources': len(manifest['source_sha256']),
                      'delivery_sha256': digest(OUT / 'DELIVERY.json')}))


if __name__ == '__main__':
    main()
