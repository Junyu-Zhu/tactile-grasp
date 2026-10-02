#!/usr/bin/env python3
"""Fixed historical case, all models/seeds; audited neighborhood evidence reused."""
import argparse,json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'evaluation'))
import evaluate as e
HERE=Path(__file__).resolve().parent

def class_rates(t):
 static=t['tn']+t['fp'];gross=t['tp']+t['fn']
 return {'static_fpr':t['fp']/static if static else None,'gross_miss':t['fn']/gross if gross else None,'static_applicable':bool(static),'gross_applicable':bool(gross)}

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--r11-root',type=Path,required=True);p.add_argument('--r11-sync-proof',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);protocol=json.loads((HERE/'CASE_PROTOCOL.json').read_text());m=json.loads(a.manifest.read_text());e.validate_manifest(m)
 sources={str(a.manifest):e.sha(a.manifest),str(HERE/'CASE_PROTOCOL.json'):e.sha(HERE/'CASE_PROTOCOL.json'),str(Path(__file__)):e.sha(__file__)};rows=[];curves=[];canonical={};fig,axes=plt.subplots(1,3,figsize=(16,4),squeeze=False)
 for j,seed in enumerate(protocol['seeds']):
  for group in protocol['groups']:
   run=next(r for r in m['runs'] if (r['group'],r['fold'],r['seed'])==(group,protocol['fold'],seed));rr=e.load_rows(run['predictions']['validation'],run['endpoints']['validation'],'validation',fold=protocol['fold']);cal=e.load_rows(run['predictions']['calibration'],run['endpoints']['calibration'],'calibration',fold=protocol['fold']);rr=[r for r in rr if r['episode']==protocol['case']];assert rr;identity=[(r['episode'],r['t'],r['stage'],r['group']) for r in rr]
   if seed in canonical:assert canonical[seed]==identity
   canonical[seed]=identity;threshold=e.choose(cal)['FPR0.05']
   for role in ('calibration','validation'):
    for k in ('predictions','endpoints'):sources[run[k][role]['path']]=run[k][role]['sha256']
   for rule,k in e.RULES.items():
    t=e.trial_metrics(rr,threshold,k)[0];rows.append(dict(group=group,fold=protocol['fold'],seed=seed,episode_id=protocol['case'],rule=rule,threshold=threshold,never_alarm=threshold>1,**{k:t[k] for k in ('tn','fp','fn','tp','false_starts','static_alarming_frames','events','hits','left_censored')},**class_rates(t)))
   curves.extend(dict(group=group,fold=protocol['fold'],seed=seed,episode_id=r['episode'],t=r['t'],stage=r['stage'],p_slip=r['score'],calibration_threshold=threshold) for r in rr);line=axes[0,j].plot([r['t'] for r in rr],[r['score'] for r in rr],label=group)[0];axes[0,j].axhline(threshold,color=line.get_color(),ls=':',alpha=.4)
  axes[0,j].step([r['t'] for r in rr],[r['stage']/2 for r in rr],color='grey',alpha=.3,label='stage/2');axes[0,j].set_title(str(seed));axes[0,j].set_ylim(-.05,1.05);axes[0,j].set_xlabel('Frame');axes[0,j].legend(fontsize=6);axes[0,j].grid(alpha=.2)
 fig.suptitle('Fixed R10/R11 failure: '+protocol['case']);fig.tight_layout();fig.savefig(a.output/'fixed_case_all_seeds.png',dpi=140);plt.close(fig);e.csvout(a.output/'fixed_case_metrics.csv',rows);e.csvout(a.output/'fixed_case_probabilities.csv',curves)
 cov=a.r11_root/'coverage_neighbors';proof=json.loads(a.r11_sync_proof.read_text());assert proof['status']=='pass';origins={r['origin']:r['sha256'] for r in proof['origin_files']};ca=cov/'AUDIT.json';assert origins[str(ca)]==e.sha(ca);audit=json.loads(ca.read_text());assert audit['status']=='pass' and audit['post_training'] and audit['case_selected_from_R10'];sources[str(a.r11_sync_proof)]=e.sha(a.r11_sync_proof);sources[str(ca)]=e.sha(ca)
 for path,h in audit['source_hashes'].items():assert e.sha(path)==h;sources[path]=h
 for name,h in audit['output_hashes'].items():assert e.sha(cov/name)==h;sources[str(cov/name)]=h
 prepared=[(p,h) for p,h in audit['source_hashes'].items() if p.endswith('/prepared.pt')];assert len(prepared)==1
 for group in protocol['groups'][-2:]:
  run=next(r for r in m['runs'] if (r['group'],r['fold'],r['seed'])==(group,protocol['fold'],20260914));su=json.loads(e.verify(run['training_summary']).read_text());cp=Path(run['training_summary']['path']).with_name('config.json');assert su['output_hashes']['config.json']==e.sha(cp);cfg=json.loads(cp.read_text());assert cfg['prepared']==dict(path=prepared[0][0],sha256=prepared[0][1]);sources[str(cp)]=e.sha(cp)
 summary=cov/'SUMMARY_ZH.md';assert origins[str(summary)]==e.sha(summary);sources[str(summary)]=e.sha(summary)
 (a.output/'COVERAGE_REUSE_ZH.md').write_text('复用第十一轮已审视觉邻域证据，核验原始来源与输出SHA及本轮新A/B fold4 seed20260914 prepared输入逐值身份。没有重新计算邻居。原证据是训练后固定历史案例的成像/冻结特征邻域代理，不是物理状态覆盖证明；该邻域仅一个seed，不能扩充为三seed覆盖结论。\n\n'+summary.read_text())
 e.js(a.output/'AUDIT.json',dict(status='pass',models=18,metrics=len(rows),case_preselected=True,no_new_coverage_fit=True,coverage_input_identity_verified=True,source_hashes=sources,outputs={f.name:e.sha(f) for f in a.output.iterdir() if f.is_file() and f.name!='AUDIT.json'}))
if __name__=='__main__':main()
