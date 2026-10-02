#!/usr/bin/env python3
"""Render the registered fold4 fixed-case source timeline for V/H/T_H."""
import argparse,csv,json
from pathlib import Path
CASE='htt/p3_sliding/0_press_13';GROUPS=('V','H','T_H');SEEDS=(20260914,20260915,20260916)
def read(path):
 with Path(path).open(newline='') as f:return list(csv.DictReader(f))
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);m=json.loads(a.manifest.read_text());thresholds=read(a.evaluation/'slip/thresholds.csv');source=[];panels=[]
 for gi,g in enumerate(GROUPS):
  for si,seed in enumerate(SEEDS):
   run=next(r for r in m['runs'] if r['group']==g and r['fold']=='htt_leave_p4' and int(r['seed'])==seed);ep={(r['episode_id'],int(r['t'])):r for r in read(run['endpoints']['validation']['path']) if r['episode_id']==CASE};pr={(r['episode_id'],int(r['t'])):r for r in read(run['predictions']['validation']['path']) if r['episode_id']==CASE};keys=sorted(set(ep)&set(pr),key=lambda x:x[1]);rows=[]
   for key in keys:
    row={'group':g,'seed':seed,'episode_id':CASE,'t':key[1],'stage':int(ep[key]['stage']),'score':float(pr[key]['p_slip'])};source.append(row);rows.append(row)
   def th(family,k):return float(next(r['threshold'] for r in thresholds if r['group']==g and r['fold']=='htt_leave_p4' and int(r['seed'])==seed and r['family']==family and r.get('alpha')=='0.05' and int(r['k'])==k))
   panels.append((gi,si,g,seed,rows,[('fixed',.5,'#555'),('macro k1',th('trial_macro_static_FPR',1),'#1b9e77'),('any k1',th('trial_any_static_alarm_rate',1),'#d95f02')]))
 with (a.output/'FIXED_CASE_SOURCE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(source[0]));w.writeheader();w.writerows(source)
 W,H=1200,900;pw,ph=360,245;parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><rect width="100%" height="100%" fill="white"/><text x="600" y="28" text-anchor="middle" font-size="18">R16 fixed case: fold4 {CASE}</text>']
 colors={0:'#d9d9d9',1:'#fee08b',2:'#f46d43'}
 for gi,si,g,seed,rows,lines in panels:
  x0=35+si*390;y0=55+gi*275;ts=[r['t'] for r in rows];lo,hi=min(ts),max(ts);sx=lambda t:x0+(t-lo)/max(1,hi-lo)*pw;sy=lambda q:y0+ph-(q*ph)
  for r in rows:parts.append(f'<rect x="{sx(r["t"]):.2f}" y="{y0}" width="{pw/max(1,len(rows)) + 1:.2f}" height="{ph}" fill="{colors[r["stage"]]}" opacity=".22"/>')
  points=' '.join(f'{sx(r["t"]):.2f},{sy(r["score"]):.2f}' for r in rows);parts.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="none" stroke="#222"/><polyline points="{points}" fill="none" stroke="#2166ac" stroke-width="1.5"/>')
  for name,value,color in lines:parts.append(f'<line x1="{x0}" x2="{x0+pw}" y1="{sy(value):.2f}" y2="{sy(value):.2f}" stroke="{color}" stroke-dasharray="5,3"/><text x="{x0+pw-3}" y="{sy(value)-3:.2f}" text-anchor="end" fill="{color}" font-size="10">{name} {value:.3f}</text>')
  parts.append(f'<text x="{x0+pw/2}" y="{y0-8}" text-anchor="middle" font-size="13">{g} seed {seed}</text><text x="{x0-8}" y="{y0+8}" text-anchor="end" font-size="10">1</text><text x="{x0-8}" y="{y0+ph}" text-anchor="end" font-size="10">0</text>')
 parts.append('</svg>');(a.output/'FIXED_CASE_TIMELINE.svg').write_text(''.join(parts));summary={'schema':'round16_fixed_case_figure_v1','status':'complete','case':CASE,'panels':9,'source_rows':len(source),'rules_shown':['fixed0.5','trial_macro_static_FPR alpha=.05 k=1','trial_any_static_alarm_rate alpha=.05 k=1'],'all_rules_table':str(a.evaluation/'failure_cases/FIXED_SLIP_CASE_ALL_RULES.csv')};(a.output/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()
