#!/usr/bin/env python3
"""Generate the prespecified fixed-case alpha=.05 all-seed alarm timeline."""
import argparse,csv,json,sys
from pathlib import Path

def read(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r));p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--round-dir',type=Path,required=True);ap.add_argument('--results',type=Path,required=True);a=ap.parse_args()
    sys.path.insert(0,str(a.round_dir/'evaluation'))
    from r13_evaluate import load_episodes,state_alarm
    idx=json.loads((a.round_dir/'ARTIFACT_INDEX.json').read_text());runs=[r for r in idx['selected_runs'] if r['fold']=='htt_leave_p4']
    new=read(a.results/'new_policies/thresholds.csv');hist=read(a.results/'historical/metrics_48.csv')
    case='htt/p3_sliding/0_press_13';out=[]
    for run in runs:
        g,fold,seed=run['group'],run['fold'],str(run['seed'])
        episodes=load_episodes(run,'validation');ep=next(e for e in episodes if e.episode==case)
        hr=next(r for r in hist if r['group']==g and r['fold']==fold and r['seed']==seed and r['role']=='calibration' and r['point']=='FPR0.05' and r['rule']=='raw')
        policies=[('history_raw',float(hr['threshold']),1),('history_confirm2',float(hr['threshold']),2)]
        nr=[r for r in new if r['group']==g and r['fold']==fold and r['seed']==seed and float(r['alpha'])==.05]
        for r in nr:policies.append((f"{r['family']}|k{r['k']}",float(r['threshold']),int(r['k'])))
        for policy,threshold,k in policies:
            alarm,starts=state_alarm(ep,threshold,k)
            for i,t in enumerate(ep.t):
                out.append({'group':g,'seed':seed,'episode':case,'policy':policy,'k':k,'threshold':threshold,
                            't':int(t),'stage':int(ep.stage[i]),'score':float(ep.score[i]),
                            'alarm':int(alarm[i]),'alarm_start':int(starts[i])})
    write(a.results/'fixed_case/alpha05_timeline.csv',out)
    # Twelve panels (four groups x three seeds), eight policy rows each. Background encodes stage; blue encodes alarm.
    groups=('V_original','F_history_original','V_class_trial_balanced','F_class_trial_balanced');seeds=('20260914','20260915','20260916')
    policies=['history_raw','history_confirm2','trial_macro_static_FPR|k1','trial_macro_static_FPR|k2','trial_macro_static_FPR|k4','trial_any_static_alarm_rate|k1','trial_any_static_alarm_rate|k2','trial_any_static_alarm_rate|k4']
    short={'history_raw':'raw','history_confirm2':'confirm2','trial_macro_static_FPR|k1':'macro k1','trial_macro_static_FPR|k2':'macro k2','trial_macro_static_FPR|k4':'macro k4','trial_any_static_alarm_rate|k1':'any k1','trial_any_static_alarm_rate|k2':'any k2','trial_any_static_alarm_rate|k4':'any k4'}
    W=1900; panel_w=540; panel_h=188; left=175; top=65; gapx=25; gapy=40; H=top+4*(panel_h+gapy)+60
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><rect width="100%" height="100%" fill="white"/><text x="{W/2}" y="28" text-anchor="middle" font-size="22">固定case α=.05 告警时间线（蓝=alarm；底色：static灰 / incipient黄 / gross红）</text>']
    for gi,g in enumerate(groups):
      for si,seed in enumerate(seeds):
        x0=left+si*(panel_w+gapx);y0=top+gi*(panel_h+gapy)
        panel=[r for r in out if r['group']==g and r['seed']==seed]; ts=sorted({int(r['t']) for r in panel}); tmin,tmax=min(ts),max(ts); scale=(panel_w-10)/(tmax-tmin+1)
        parts.append(f'<text x="{x0}" y="{y0-8}" font-size="15">{g} / {seed}</text>')
        stage_by_t={int(r['t']):int(r['stage']) for r in panel if r['policy']=='history_raw'}
        for pi,p in enumerate(policies):
            y=y0+pi*21;parts.append(f'<text x="{x0-8}" y="{y+14}" text-anchor="end" font-size="11">{short[p]}</text>')
            for t in ts:
                bg={0:'#e6e6e6',1:'#fee08b',2:'#f4a3a3'}[stage_by_t[t]]
                parts.append(f'<rect x="{x0+(t-tmin)*scale:.2f}" y="{y}" width="{max(scale,1):.2f}" height="16" fill="{bg}"/>')
            for r in (q for q in panel if q['policy']==p and int(q['alarm'])==1):
                t=int(r['t']);parts.append(f'<rect x="{x0+(t-tmin)*scale:.2f}" y="{y+3}" width="{max(scale,1):.2f}" height="10" fill="#2166ac"/>')
        parts.append(f'<text x="{x0}" y="{y0+panel_h}" font-size="10">t={tmin}</text><text x="{x0+panel_w}" y="{y0+panel_h}" text-anchor="end" font-size="10">t={tmax}</text>')
    parts.append('</svg>');(a.results/'reporting/fixed_case_alpha05_timeline.svg').write_text(''.join(parts))
    print(json.dumps({'status':'pass','rows':len(out),'panels':12,'policies_per_panel':8}))
if __name__=='__main__':main()
