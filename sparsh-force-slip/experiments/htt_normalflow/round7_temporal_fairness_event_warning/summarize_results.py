#!/usr/bin/env python3
"""Descriptive seed summaries; never pool frames or refit thresholds."""
import argparse,csv,hashlib,json,math,statistics
from collections import defaultdict
from pathlib import Path
FIELDS=['average_precision','prevalence','balanced_accuracy','macro_f1','brier','frame_fpr','frame_recall','trial_false_alarm_rate','event_recall','active_overlap_recall','mean_lead_frames','lead_ge_3_recall','lead_ge_5_recall','false_alarm_starts','false_alarm_duration','misses','late_alarms','never_alarm_trials']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
 src=r/'evaluation/formal_analysis/metrics.csv';rows=list(csv.DictReader(src.open()));groups=defaultdict(list)
 for x in rows:
  if x['population']!='primary':continue
  if x['method_type']=='neural' and x['rule']!='raw' and x['rule_selected']!='True':continue
  mode='selected' if x['rule_selected']=='True' else 'raw_unselected'
  # Keep raw and selected distinct even when raw is selected, to avoid pooling rules.
  modes=['raw'] if x['rule']=='raw' else []
  if x['rule_selected']=='True':modes.append('selected')
  for mode in modes:groups[(x['method_type'],x['group'],int(x['horizon']),mode,x['operating_point'])].append(x)
 result=[]
 for key,rr in sorted(groups.items()):
  seeds=[x['seed'] for x in rr]
  if len(seeds)!=len(set(seeds)):raise ValueError(('duplicate_seed',key,seeds))
  d=dict(zip(['method_type','group','horizon','rule_mode','operating_point'],key));d['seeds']=seeds;d['rules']=[x['rule'] for x in rr];d['runs']=len(rr)
  d['metrics']={}
  for field in FIELDS:
   vals=[]
   for x in rr:
    try:v=float(x[field])
    except (ValueError,TypeError):continue
    if math.isfinite(v):vals.append(v)
   d['metrics'][field]={'mean':statistics.mean(vals) if vals else None,'sd_across_seeds':statistics.stdev(vals) if len(vals)>1 else None,'min':min(vals) if vals else None,'max':max(vals) if vals else None,'defined_runs':len(vals)}
  result.append(d)
 out={'status':'complete','source_sha256':sha(src),'source':str(src),'code_sha256':sha(Path(__file__)),'scope':'descriptive per-run means and seed sample SD, no independent-seed significance claim; undefined values counted explicitly','rows':result}
 (r/'SEED_SUMMARIES.json').write_text(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
 print(json.dumps({'summary_rows':len(result)}))
if __name__=='__main__':main()
