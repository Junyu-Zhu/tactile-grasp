#!/usr/bin/env python3
"""Event-level prior-alarm and delay-zero supplement for all validation rules."""
import argparse,csv,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np

def read(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r));p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)

def events(ep,alarm):
    gross=ep.stage==2;breaks=np.r_[True,np.diff(ep.t)!=1];out=[];event_i=0
    for i in range(len(ep.t)):
        if not gross[i] or (i and not breaks[i] and gross[i-1]):continue
        j=i+1
        while j<len(ep.t) and not breaks[j] and gross[j]:j+=1
        hit=np.flatnonzero(alarm[i:j]);left=bool(breaks[i]);prior=bool(i>0 and not breaks[i] and alarm[i-1])
        out.append({'event_index':event_i,'gross_start_t':int(ep.t[i]),'gross_end_t':int(ep.t[j-1]),
                    'left_censored':int(left),'right_boundary':int(j==len(ep.t) or (j<len(ep.t) and breaks[j])),
                    'preexisting_alarm':int(prior),'hit':int(bool(len(hit))),
                    'delay':int(hit[0]) if len(hit) and not left else '',
                    'preexisting_alarm_delay0':int(prior and bool(len(hit)) and int(hit[0])==0 and not left)})
        event_i+=1
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--round-dir',type=Path,required=True);ap.add_argument('--results',type=Path,required=True);a=ap.parse_args()
    sys.path.insert(0,str(a.round_dir/'evaluation'));from r13_evaluate import load_episodes,state_alarm
    idx=json.loads((a.round_dir/'ARTIFACT_INDEX.json').read_text())
    hist=read(a.results/'historical/metrics_48.csv');new=read(a.results/'new_policies/thresholds.csv');c4=read(a.results/'new_policies/confirm4_metrics.csv')
    details=[]
    for run in idx['selected_runs']:
        g,f,s=run['group'],run['fold'],str(run['seed']);eps=load_episodes(run,'validation')
        policies=[]
        for r in hist:
            if r['group']==g and r['fold']==f and r['seed']==s and r['role']=='validation':
                policies.append((f"historical|{r['point']}|{r['rule']}",float(r['threshold']),1 if r['rule']=='raw' else 2,'historical'))
        for r in new:
            if r['group']==g and r['fold']==f and r['seed']==s:
                policies.append((r['policy'],float(r['threshold']),int(r['k']),'new'))
        for r in c4:
            if r['group']==g and r['fold']==f and r['seed']==s and r['role']=='validation':
                policies.append((r['policy'],float(r['threshold']),4,'confirm4_reference'))
        for policy,threshold,k,kind in policies:
            for ep in eps:
                alarm,_=state_alarm(ep,threshold,k)
                for event in events(ep,alarm):
                    details.append({'group':g,'fold':f,'seed':s,'role':'validation','kind':kind,'policy':policy,
                                    'threshold':threshold,'k':k,'episode':ep.episode,'leakage_group':ep.group,**event})
    write(a.results/'events/validation_event_details.csv',details)
    grouped=defaultdict(list)
    for r in details:grouped[(r['group'],r['fold'],r['seed'],r['kind'],r['policy'])].append(r)
    summary=[]
    for key,rows in sorted(grouped.items()):
        observed=[r for r in rows if int(r['left_censored'])==0];hits=[r for r in observed if int(r['hit'])==1]
        pre=[r for r in observed if int(r['preexisting_alarm'])==1];zero=[r for r in observed if int(r['preexisting_alarm_delay0'])==1]
        summary.append({'group':key[0],'fold':key[1],'seed':key[2],'kind':key[3],'policy':key[4],
                        'gross_segments':len(rows),'left_censored':len(rows)-len(observed),'events':len(observed),
                        'hits':len(hits),'preexisting_alarm_events':len(pre),
                        'preexisting_alarm_delay0_events':len(zero),
                        'preexisting_alarm_rate':len(pre)/len(observed) if observed else '',
                        'preexisting_alarm_delay0_rate':len(zero)/len(observed) if observed else ''})
    write(a.results/'events/validation_event_summary.csv',summary)
    print(json.dumps({'status':'pass','event_rows':len(details),'summary_rows':len(summary),
                      'historical_summary_rows':sum(r['kind']=='historical' for r in summary)}))
if __name__=='__main__':main()
