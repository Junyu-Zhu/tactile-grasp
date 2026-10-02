#!/usr/bin/env python3
"""Read-only original artifacts; copy/symlink mutation fixtures in a fresh tempdir."""
import argparse,copy,json,pathlib,subprocess,sys,tempfile
p=argparse.ArgumentParser();p.add_argument('--inventory',type=pathlib.Path,required=True);p.add_argument('--auditor',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();original=json.loads(a.inventory.read_text());results=[]
with tempfile.TemporaryDirectory(prefix='r9_audit_mutation_') as tmp:
 root=pathlib.Path(tmp)
 for mutation in ('missing_seed','smoke_summary','prediction_hash_replacement','missing_prediction_hash'):
  inv=copy.deepcopy(original);d=root/mutation;d.mkdir()
  if mutation=='missing_seed':inv['jobs'].pop()
  else:
   source=pathlib.Path(inv['jobs'][0]['output']);target=d/'copied_run';target.mkdir()
   for src in source.iterdir():
    if src.is_file():(target/src.name).symlink_to(src)
   inv['jobs'][0]['output']=str(target)
   summary=json.loads((source/'summary.json').read_text())
   if mutation=='smoke_summary':summary['smoke']=True
   if mutation=='missing_prediction_hash':summary['output_hashes'].pop('predictions_validation.csv')
   if mutation in ('smoke_summary','missing_prediction_hash'):
    (target/'summary.json').unlink();(target/'summary.json').write_text(json.dumps(summary))
   if mutation=='prediction_hash_replacement':
    q=target/'predictions_validation.csv';q.unlink();q.write_text('invalid replacement\n')
  ip=d/'inventory.json';ip.write_text(json.dumps(inv));r=subprocess.run([sys.executable,str(a.auditor),'--inventory',str(ip),'--output',str(d/'output')],capture_output=True,text=True)
  expected={'missing_seed':'ValueError: formal grid','smoke_summary':'ValueError: unaccepted/smoke run','prediction_hash_replacement':'ValueError: output hash','missing_prediction_hash':'ValueError: incomplete output hash grid'}[mutation]
  results.append(dict(mutation=mutation,rejected=r.returncode!=0 and expected in r.stderr,expected_rejection=expected,returncode=r.returncode,stderr_tail=r.stderr[-1500:]))
 if not all(r['rejected'] for r in results):raise ValueError(results)
a.output.write_text(json.dumps(dict(status='pass',original_artifacts_modified=False,mutations=results),indent=2));print(json.dumps({'status':'pass','mutations':len(results)}))
