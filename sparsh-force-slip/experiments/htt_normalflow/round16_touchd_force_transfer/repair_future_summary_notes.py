#!/usr/bin/env python3
"""Correct inherited wording only; metric and prediction files stay unchanged."""
import argparse,json
from pathlib import Path
from touchd_common import atomic_json,sha256
OLD='Change error uses common R10 predicted-current anchor; ideal GT-current persistence is non-deployable.'
NEW='Change error and persistence use this route\'s own predicted-current anchor; ideal GT-current persistence is non-deployable.'
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();records=[]
 for path in sorted((a.root/'evaluation/future').glob('*/SUMMARY.json')):
  before=sha256(path);d=json.loads(path.read_text());assert d['status']=='complete' and d['note'] in (OLD,NEW)
  d['note']=NEW;atomic_json(path,d);records.append({'path':str(path),'before_sha256':before,'after_sha256':sha256(path),'metrics_sha256':d['metrics_sha256'],'prediction_hashes':d['prediction_hashes']})
 assert len(records)==24
 result={'schema':'round16_future_summary_note_repair_v1','status':'pass','files':24,'change':'wording only: common R10 -> route-own predicted-current anchor','metrics_or_predictions_changed':False,'records':records};atomic_json(a.root/'audit/FUTURE_SUMMARY_NOTE_REPAIR.json',result);print(json.dumps({'status':'pass','files':24}))
if __name__=='__main__':main()
