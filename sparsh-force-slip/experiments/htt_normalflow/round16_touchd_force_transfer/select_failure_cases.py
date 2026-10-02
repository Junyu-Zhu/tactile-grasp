#!/usr/bin/env python3
"""Apply the locked complete-candidate failure-case and lexical tie rules."""
import argparse,csv,json,statistics as st
from collections import defaultdict
from pathlib import Path
SEEDS=(20260914,20260915,20260916)
def read(path):
 with Path(path).open(newline='') as f:return list(csv.DictReader(f))
def write(path,rows):
 with Path(path).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);out=[]
 for route in ('H','T_H'):
  for fold in range(1,5):
   for seed in SEEDS:
    rows=[r for r in read(a.evaluation/f'force/{route}_p{fold}_s{seed}/trial_axis_metrics.csv') if r['role']=='validation'];by=defaultdict(list)
    for r in rows:by[r['episode_id']].append(float(r['mae']))
    score={k:st.fmean(v) for k,v in by.items()};episode=min(score,key=lambda x:(-score[x],x));out.append({'task':'force','group':route,'fold':fold,'seed':seed,'episode_id':episode,'primary_score':score[episode],'secondary_score':''})
 slip=read(a.evaluation/'slip/trials.csv')
 for group in ('V','H','T_H'):
  for fold in range(1,5):
   for seed in SEEDS:
    rows=[r for r in slip if r['group']==group and r['fold']==f'htt_leave_p{fold}' and int(r['seed'])==seed and r['role']=='validation' and r['family']=='historical' and r.get('point')=='fixed0.5' and r.get('rule')=='raw']
    episode=min(rows,key=lambda r:(-int(r['static_fp']),-(int(r['gross_frames'])-int(r['gross_tp'])),r['episode']))
    out.append({'task':'slip','group':group,'fold':fold,'seed':seed,'episode_id':episode['episode'],'primary_score':episode['static_fp'],'secondary_score':int(episode['gross_frames'])-int(episode['gross_tp'])})
 future=[]
 for route in ('H','T_H'):
  for fold in range(1,5):
   for seed in SEEDS:
    rows=[r for r in read(a.evaluation/f'future_diagnostics/{route}_p{fold}_s{seed}/trial_diagnostics.csv') if r['role']=='validation' and r['method']=='neural' and r['stratum']=='all'];by=defaultdict(list)
    for r in rows:by[r['episode_id']].append(float(r['future_mae']))
    score={k:st.fmean(v) for k,v in by.items()};episode=min(score,key=lambda x:(-score[x],x));out.append({'task':'future','group':route,'fold':fold,'seed':seed,'episode_id':episode,'primary_score':score[episode],'secondary_score':''})
 write(a.output/'SELECTED_CASES.csv',out);fixed=[r for r in slip if r['fold']=='htt_leave_p4' and r['role']=='validation' and r['episode']=='htt/p3_sliding/0_press_13'];write(a.output/'FIXED_SLIP_CASE_ALL_RULES.csv',fixed)
 result={'schema':'round16_failure_cases_v1','status':'complete','selected_rows':len(out),'fixed_slip_case':'fold4 htt/p3_sliding/0_press_13','performance_changes_case_rules':False,'test_consumed':False};(a.output/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
