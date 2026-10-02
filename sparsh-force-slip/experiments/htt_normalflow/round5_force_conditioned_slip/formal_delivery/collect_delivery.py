#!/usr/bin/env python3
"""Snapshot small formal artifacts and reproducible source after final acceptance."""
import argparse,hashlib,json,shutil
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--code-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--independent-review',type=Path,required=True);a=p.parse_args()
    state=json.loads((a.runs_root/'formal_scheduler_state.json').read_text())
    inventory_path=a.code_root/'RUN_INVENTORY.json'
    if sha(inventory_path)!='a46ed29b505df98e92d622fc57c3c0c38d9ecd63b9f89c8a40bdede8c9a1f7d4':raise RuntimeError('Frozen inventory identity mismatch')
    inv=json.loads(inventory_path.read_text());expected={j['id'] for j in inv['jobs']}
    if len(expected)!=140 or set(state['jobs'])!=expected or state['status']!='complete' or any(v!='complete' for v in state['jobs'].values()):raise RuntimeError('Formal queue identity/count/completion mismatch')
    def read(rel):return json.loads((a.runs_root/rel).read_text())
    training=read('formal_delivery/TRAINING_AUDIT.json')
    if training['status']!='pass' or training['verified_runs']!=66 or training['failures'] or training['review_required']:raise RuntimeError('Training audit not accepted')
    encoder=read('formal_delivery/ENCODER_RAW_INFERENCE_AUDIT.json')
    if encoder['status']!='pass' or not all(encoder['checks'].values()):raise RuntimeError('Raw inference encoder identity mismatch')
    current=read('analysis_summary/summary.json')
    if current['status']!='complete' or current['evaluations']!=48 or not current['all_expected_identities_present'] or not current['all_seeds_reported']:raise RuntimeError('Current comparison incomplete')
    force=read('analysis_force/summary/summary.json')
    if force['status']!='complete' or force['verified_jobs']!=82:raise RuntimeError('Force analysis incomplete')
    for domain,count in [('source',63),('htt',105)]:
        future=read(f'analysis_future/{domain}/evaluation.json')
        if future['status']!='complete' or future['formal'] is not True or len(future['results'])!=count:raise RuntimeError(f'{domain} future incomplete')
    review=json.loads(a.independent_review.read_text())
    if review.get('status')!='pass' or review.get('scope')!='round5_formal_final_pre_sync' or review.get('unresolved_findings')!=[]:raise RuntimeError('Independent final review not accepted')
    reviewed=review.get('reviewed_artifact_sha256',{})
    required_reviewed={str(a.runs_root/rel) for rel in ['formal_delivery/SUMMARY_ZH.md','formal_delivery/TRAINING_AUDIT.json','formal_delivery/ENCODER_RAW_INFERENCE_AUDIT.json','analysis_summary/summary.json','analysis_force/summary/summary.json','analysis_future/source/evaluation.json','analysis_future/htt/evaluation.json']}
    if not required_reviewed.issubset(reviewed):raise RuntimeError('Independent review lacks artifact binding')
    for raw,digest in reviewed.items():
        if not Path(raw).is_file() or sha(Path(raw))!=digest:raise RuntimeError(f'Reviewed artifact changed: {raw}')
    for rel in ['formal_delivery/SUMMARY_ZH.md','formal_delivery/FORMAL_REPRODUCE.md','analysis_summary/SUMMARY_ZH.md','analysis_force/summary/SUMMARY_ZH.md','analysis_future/source/REPORT_ZH.md','analysis_future/htt/REPORT_ZH.md']:
        if not (a.runs_root/rel).is_file() or not (a.runs_root/rel).stat().st_size:raise RuntimeError(f'Missing required report: {rel}')
    for rel in ['analysis_summary/figures','analysis_future/figures']:
        if not list((a.runs_root/rel).glob('*.png')):raise RuntimeError(f'Missing figures: {rel}')
    if a.output.exists():raise RuntimeError('Choose a new snapshot directory; historical deliveries are immutable')
    a.output.mkdir(parents=True)
    allowed={'.json','.md','.yaml','.yml','.py','.txt','.csv','.log','.png','.svg','.sha256'}
    source_folders=['formal','logs','analysis_force','analysis_future','analysis_summary','formal_delivery']
    selected=[]
    for folder in source_folders:
        root=a.runs_root/folder
        if not root.exists():continue
        for f in root.rglob('*'):
            if f.is_file() and f.suffix in allowed and a.output not in f.parents and '__pycache__' not in f.parts:
                selected.append((f,Path('results')/f.relative_to(a.runs_root)))
    for f in a.runs_root.glob('*.json'):selected.append((f,Path('results')/f.name))
    for f in a.code_root.rglob('*'):
        if f.is_file() and f.suffix in allowed and '__pycache__' not in f.parts and 'formal_delivery_bundle' not in f.parts:
            selected.append((f,Path('source')/f.relative_to(a.code_root)))
    records=[]
    for src,rel in sorted(selected,key=lambda x:str(x[1])):
        if src.stat().st_size>100*1024**2:raise RuntimeError(f'Unexpected large delivery file: {src}')
        dest=a.output/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
        digest=sha(dest)
        if digest!=sha(src):raise RuntimeError(f'Source changed during snapshot: {src}')
        records.append({'path':str(rel),'source':str(src),'bytes':dest.stat().st_size,'sha256':digest})
    (a.output/'SHA256_MANIFEST.json').write_text(json.dumps({'schema':1,'status':'snapshot_verified','files':records},indent=2)+'\n')
    print(json.dumps({'files':len(records),'bytes':sum(r['bytes'] for r in records),'output':str(a.output)}))
if __name__=='__main__':main()
