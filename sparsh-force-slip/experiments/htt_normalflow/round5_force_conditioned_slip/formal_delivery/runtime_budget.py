#!/usr/bin/env python3
"""Estimate remaining budget from completed run log birth/last-write times."""
import argparse,datetime,json,statistics,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_formal import accepted
def main():
    p=argparse.ArgumentParser();p.add_argument('--inventory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    inv=json.loads(a.inventory.read_text());rows=[]
    for j in inv['jobs']:
        if j['kind']!='neural':continue
        if not accepted(j):continue
        summary=Path(j['acceptance_path'])
        if not summary.is_file():continue
        s=json.loads(summary.read_text())
        if s.get('status')!='complete':continue
        log=Path(j['log_path']);birth,end=map(int,subprocess.check_output(['stat','-c','%W %Y',str(log)],text=True).split())
        if birth<=0 or end<birth:continue
        epochs=s.get('epochs_completed',len(s.get('history',[])))
        rows.append({'id':j['id'],'wall_seconds_log_lifetime':end-birth,'epochs':epochs,'seconds_per_epoch_including_initialization':(end-birth)/max(1,epochs)})
    rem=66-len(rows);sec=max((r['seconds_per_epoch_including_initialization'] for r in rows),default=60.)
    done={r['id'] for r in rows}
    remaining_epochs=sum(100 if 'train-htt' in j['argv'] else 40 if 'train-source' in j['argv'] else 30 for j in inv['jobs'] if j['kind']=='neural' and j['id'] not in done)
    # No three-GPU speed-up credit; retain each preregistered maximum epoch budget.
    worst=remaining_epochs*max(sec,60)*2
    now=datetime.datetime.now(datetime.timezone.utc);cutoff=datetime.datetime.fromisoformat(inv['deadline'])-datetime.timedelta(days=2)
    result={'recorded_at':now.isoformat(),'completed_neural':len(rows),'remaining_neural':rem,'measurement':'per-run log birth to last write, includes process setup; resumed logs include interruption time','measured_runs':rows,'conservative_remaining_seconds':worst,'conservative_assumption':'remaining fixed max epochs, max(observed slowest avg,60s)/epoch, x2 contention; no parallelism credit','available_until_new_training_cutoff_seconds':max(0,(cutoff-now).total_seconds()),'within_conservative_budget':worst<(cutoff-now).total_seconds()}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='measured_runs'}))
if __name__=='__main__':main()
