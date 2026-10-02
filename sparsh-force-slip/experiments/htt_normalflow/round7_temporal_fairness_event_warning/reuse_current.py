#!/usr/bin/env python3
"""Identity-bound R6 current-slip reference, no inference or training."""
import argparse,hashlib,json,shutil
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--r6',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.r6;c=r/'current';a.output.mkdir(parents=True,exist_ok=True)
 review=json.loads((r/'INDEPENDENT_FINAL_REVIEW.json').read_text());assert sha(c/'summary.json')==review['reviewed_artifact_sha256']['current/summary.json']
 names=['SUMMARY_ZH.md','summary.json','operational_points.csv','validation_descriptive_envelope.csv','paired_fold_bootstrap.csv','sequence_metrics.csv','r5_operating_points.csv','representative_failures.csv'];items=[]
 for n in names:
  p=c/n;assert p.is_file();q=a.output/n;shutil.copy2(p,q);assert sha(p)==sha(q);items.append({'source':str(p),'file':n,'sha256':sha(q)})
 d={'status':'complete','mode':'exact R6 reference reuse; no current model retraining or retuning','files':items,'claim':'Future gains do not establish current force-fusion gains. Existing current calibration migration and recall tradeoffs retained.'};(a.output/'REUSE_AUDIT.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({'status':'complete','files':len(items)}))
if __name__=='__main__':main()
