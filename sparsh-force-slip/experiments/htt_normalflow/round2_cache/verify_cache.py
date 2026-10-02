"""Validate full coverage, source/cache identity and temporal array alignment."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prepare_splits import file_hash, verify


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--index',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    manifest=json.loads(Path(args.manifest).read_text())
    verify(manifest)
    index=json.loads(Path(args.index).read_text())
    if index['status']!='complete' or index['split_manifest_sha256']!=file_hash(args.manifest):
        raise ValueError('Incomplete cache or wrong split manifest')
    allowed=set()
    for parts in manifest['splits'].values():
        for part in ('train','validation','calibration'):
            allowed.update(parts.get(part,[]))
    ids=[r['id'] for r in index['episodes']]
    if len(ids)!=len(set(ids)) or set(ids)!=allowed:
        raise ValueError('Development coverage mismatch')
    frames=0
    for row in index['episodes']:
        if file_hash(row['cache_path'])!=row['cache_sha256']:
            raise ValueError('Corrupt cache')
        for source,sha in row['source_files'].items():
            if file_hash(source)!=sha:
                raise ValueError('Source mutation')
        for role in row['development_roles']:
            if role['partition']=='test' or row['id'] not in manifest['splits'][role['fold']][role['partition']]:
                raise ValueError('Invalid development role')
        with np.load(row['cache_path'],allow_pickle=False) as d:
            n=row['frames']; frames+=n
            for key in ('z','force_pred','p_slip','p_future','labels','pose','time','image_delta_l1','frame_index'):
                if len(d[key])!=n:
                    raise ValueError(f'Alignment {row["id"]}/{key}')
            if not np.array_equal(d['frame_index'],np.arange(n)):
                raise ValueError('Frame index mismatch')
            for key in ('z','force_pred','p_slip','p_future'):
                if not np.isfinite(d[key]).all():
                    raise ValueError('Nonfinite predictions')
            if d['z'].shape!=(n,768) or d['force_pred'].shape!=(n,3) or d['p_future'].shape!=(n,3):
                raise ValueError('Feature shape mismatch')
            for key in ('p_slip','p_future'):
                if not ((d[key]>=0)&(d[key]<=1)).all():
                    raise ValueError('Invalid probabilities')
            if row['domain']=='normalflow':
                if row['group'] in ('seed','ball') or not (d['labels']==-1).all():
                    raise ValueError('NF test data or fabricated labels')
            elif row['task']=='force' and not (d['labels']==-1).all():
                raise ValueError('Fabricated HTT force labels')
    report={'status':'passed','episodes':len(ids),'frames':frames,'split_sha256':file_hash(args.manifest),
            'index_sha256':file_hash(args.index),'test_partition_roles_consumed':0,
            'note':'HTT per-fold development union includes all probes; no global blind-set claim'}
    Path(args.output).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
