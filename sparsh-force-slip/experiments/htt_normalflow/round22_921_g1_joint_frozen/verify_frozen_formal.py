#!/usr/bin/env python3
import hashlib,json,statistics
from collections import Counter,defaultdict
from datetime import datetime
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 inv=json.loads((HERE/'FROZEN_RUN_INVENTORY.json').read_text()); auth_sha=sha(HERE/'G1_FORMAL_AUTHORIZATION.json'); proto_sha=sha(HERE/'FROZEN_PROTOCOL_LOCK.json'); rows=[];errors=[];starts=[];ends=[]
 for r in inv['runs']:
  out=Path(r['output'])
  try:
   s=json.loads((out/'summary.json').read_text());cm=json.loads((out/'COMMIT.json').read_text()); latest=out/cm['latest']['path'];best=out/cm['best']['path'];ck=torch.load(latest,map_location='cpu',weights_only=False);bck=torch.load(best,map_location='cpu',weights_only=False);ident=s['identity'];hist=ck['history']; key='selection_pAUC_0_0.1' if r['package'] in ('E1','E2') else 'selection_native_delta_mae'; vals=[x[key] for x in hist]; target=max(vals) if r['package'] in ('E1','E2') else min(vals); first=vals.index(target)
   checks={'summary_complete':s['status']=='complete','formal':ident['formal'] is True,'grid':(ident['group'],ident['fold'],ident['seed'])==(r['group'],r['fold'],r['seed']),'epochs':s['epochs']==len(hist) and 1<=s['epochs']<=60,'optimizer_steps':s['optimizer_steps']>0,'authorization':ident['authorization_sha256']==auth_sha,'protocol':ident['protocol_sha256']==proto_sha==r['protocol_sha256'],'source':ident['source_sha256']==r['trainer_sha256'],'data':ident['data_sha256']==r['data_sha256'],'latest_hash':sha(latest)==cm['latest']['sha256'],'best_hash':sha(best)==cm['best']['sha256'],'latest_identity':ck['identity']==ident,'best_identity':bck['identity']==ident,'latest_epoch':ck['epoch']==s['epochs']-1==cm['latest']['epoch'],'selector_best':abs(s.get('best_metric',s.get('best_metric_native_delta_mae'))-target)<1e-12 and s['best_epoch']==first==cm['best']['epoch']}
   bad=[k for k,v in checks.items() if not v]
   if bad:errors.append({'run':r['run'],'checks':bad})
   rows.append({'run':r['run'],'package':r['package'],'group':r['group'],'fold':r['fold'],'seed':r['seed'],'epochs':s['epochs'],'best_epoch':s['best_epoch'],'best_metric':s.get('best_metric',s.get('best_metric_native_delta_mae')),'optimizer_steps':s['optimizer_steps'],'latest_sha256':sha(latest),'best_sha256':sha(best)})
   starts.append((out/'config.json').stat().st_mtime);ends.append((out/'summary.json').stat().st_mtime)
  except Exception as e:errors.append({'run':r['run'],'exception':repr(e)})
 by=defaultdict(list)
 for x in rows:by[x['group']].append(x['epochs'])
 summary={'schema':'round22_frozen_formal_acceptance_v1','status':'pass' if len(rows)==84 and not errors else 'fail','accepted_runs':len(rows) if not errors else len(rows)-sum(1 for e in errors if any(x.get('run')==e.get('run') for x in rows)),'inventory_runs':len(inv['runs']),'errors':errors,'epochs':{'min':min(x['epochs'] for x in rows),'median':statistics.median(x['epochs'] for x in rows),'max':max(x['epochs'] for x in rows),'by_group':{g:{'min':min(v),'median':statistics.median(v),'max':max(v)} for g,v in sorted(by.items())}},'optimizer_steps_total':sum(x['optimizer_steps'] for x in rows),'observed_wall':{'first_config':datetime.fromtimestamp(min(starts)).astimezone().isoformat(),'last_summary':datetime.fromtimestamp(max(ends)).astimezone().isoformat(),'seconds':max(ends)-min(starts),'three_gpu_elapsed_gpu_hours':(max(ends)-min(starts))*3/3600},'formal':True,'max_epochs':60,'patience':10,'test_consumed':False}
 (HERE/'FROZEN_FORMAL_ACCEPTANCE.json').write_text(json.dumps({'summary':summary,'runs':rows},indent=2,sort_keys=True)+'\n');print(json.dumps(summary));assert summary['status']=='pass'
if __name__=='__main__':main()
