"""Bind completed run artifacts, code and server-only checkpoint references."""
import hashlib
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
OUT=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round3_mae_slip_adaptation')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    audit=json.loads((OUT/'final_training_audit.json').read_text())
    assert audit['status']=='pass' and audit['passed_runs']==24
    assert json.loads((OUT/'postprocess_complete.json').read_text())['status']=='pass'
    snapshot=OUT/'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for p in HERE.iterdir():
        if p.is_file() and p.suffix in ('.py','.md','.sh'):
            shutil.copy2(p,snapshot/p.name)
    shutil.copy2(OUT/'cache/cache_manifest.json',OUT/'cache_index.json')
    checkpoints=[]
    for p in sorted((OUT/'runs').glob('*/*/*/training_summary.json')):
        s=json.loads(p.read_text())
        checkpoints.append({k:s[k] for k in ('fold','seed','init','best_checkpoint','best_checkpoint_sha256','latest_checkpoint','latest_checkpoint_sha256')})
    files=[]
    for p in sorted(OUT.rglob('*')):
        if not p.is_file() or 'cache' in p.relative_to(OUT).parts or p.suffix=='.pth' or '.tmp' in p.name or p.name=='DELIVERY.json':continue
        files.append({'path':str(p.relative_to(OUT)),'bytes':p.stat().st_size,'sha256':digest(p)})
    result={'status':'complete','scope':'HTT frozen MAE slip-only development adaptation; no outer test, future or NormalFlow training',
            'trained_runs':24,'baseline_folds':4,'artifact_root':str(OUT),'server_only_checkpoints':checkpoints,
            'artifacts':files,'cache_manifest_sha256':digest(OUT/'cache/cache_manifest.json'),
            'cache_location':str(OUT/'cache'),'cache_frames':25149,'cache_episodes':101}
    (OUT/'DELIVERY.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'complete','trained_runs':24,'artifacts':len(files)}))


if __name__=='__main__':main()
