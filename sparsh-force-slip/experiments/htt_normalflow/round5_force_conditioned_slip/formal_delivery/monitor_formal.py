#!/usr/bin/env python3
"""Read-only scheduler/epoch snapshot, with optional durable report."""
import argparse,collections,datetime,json,time
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--inventory',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args()
    inv=json.loads(a.inventory.read_text());root=Path(inv['output_root'])
    state=json.loads((root/'formal_scheduler_state.json').read_text());rows=[]
    for j in inv['jobs']:
        status=state['jobs'][j['id']]
        if status=='pending':continue
        row={'id':j['id'],'kind':j['kind'],'status':status}
        log=Path(j['log_path'])
        if status in ('running','failed','blocked') and log.exists():
            lines=log.read_text(errors='replace').splitlines();epochs=[]
            for line in lines:
                if line.startswith('{"epoch":'):
                    try:epochs.append(json.loads(line))
                    except ValueError:pass
            row['epochs_logged']=len(epochs)
            if epochs:row['last_epoch']=epochs[-1]
            if status=='failed':row['error_tail']=lines[-12:]
        summary=Path(j['acceptance_path'])
        if status=='complete' and summary.exists():
            s=json.loads(summary.read_text());row['epochs_completed']=s.get('epochs_completed',len(s.get('history',[])))
        rows.append(row)
    result={'recorded_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scheduler_status':state['status'],'all_counts':dict(collections.Counter(state['jobs'].values())),'neural_counts':dict(collections.Counter(state['jobs'][j['id']] for j in inv['jobs'] if j['kind']=='neural')),'oom_retries':state.get('oom_serial_retries',[]),'records':rows}
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
