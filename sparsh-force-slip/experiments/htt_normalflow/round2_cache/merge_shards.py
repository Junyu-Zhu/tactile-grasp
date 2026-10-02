"""Merge completed, identically-versioned disjoint inference shards."""
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--shards',type=int,default=4);a=p.parse_args()
    root=Path(a.root)
    indices=[json.loads((root/f'index_shard{i}.json').read_text()) for i in range(a.shards)]
    merged=dict(indices[0]); rows=[]
    for i,index in enumerate(indices):
        if index['status']!='complete' or index['shard']!=i or index['shards']!=a.shards:
            raise ValueError('Incomplete/misidentified shard')
        if index['preprocessing_fingerprint']!=merged['preprocessing_fingerprint']:
            raise ValueError('Different inference contracts')
        if len(index['episodes'])!=index['expected_shard_episodes']:
            raise ValueError('Incomplete shard coverage')
        rows.extend(index['episodes'])
    if len(rows)!=merged['expected_episodes'] or len({r['id'] for r in rows})!=len(rows):
        raise ValueError('Global coverage mismatch')
    merged.update(episodes=sorted(rows,key=lambda r:r['id']),shard=None,status='complete',
                  elapsed_seconds=max(i['elapsed_seconds'] for i in indices))
    tmp=root/'index.tmp';tmp.write_text(json.dumps(merged,indent=2));tmp.replace(root/'index.json')
    print(json.dumps({'status':'complete','episodes':len(rows)}))

if __name__=='__main__': main()
