#!/usr/bin/env python3
"""Verify every declared delivery file on either side; no large artifacts included."""
import argparse,hashlib,json
from pathlib import Path
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--side',choices=('local','remote'),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    m=json.loads(a.manifest.read_text());bad=[]
    for f in m['files']:
        path=Path(f[a.side+'_path'])
        if not path.is_file() or sha(path)!=f['sha256']:bad.append(str(path))
    result={'status':'pass' if not bad else 'fail','side':a.side,'verified_files':len(m['files']),'manifest_sha256':sha(a.manifest),'inventory_sha256':m['inventory_sha256'],'mismatches':bad}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    if bad:raise SystemExit(1)
if __name__=='__main__':main()
