#!/usr/bin/env python3
"""R9 fixed-threshold future robustness; never fits or selects on outer."""
import argparse,csv,hashlib,importlib.util,json,sys,math
from pathlib import Path
from collections import defaultdict
import numpy as np
HERE=Path(__file__).resolve().parent
R8=HERE.parent.parent/'round8_force_dynamics_event_time'
sys.path.insert(0,str(R8/'evaluation'))
import evaluate_formal as ev
from schema import load_timeline_csv,validate_cross_run_identity

def sha(p): return ev.sha256(Path(p))
def read(p): return list(csv.DictReader(Path(p).open()))
def truth(v): return str(v).lower()=='true'
def dist(r):
    d=None if r['onset'] is None else r['onset']-r['t']
    return 'no_observed_onset' if d is None else ('1' if d==1 else '2..3' if d<=3 else '4..5' if d<=5 else '6..10' if d<=10 else '11+')
def prefix(r):
    n=r['onset']
    return 'no_observed_onset' if n is None else ('0..18' if n<=18 else '19..32' if n<=32 else '33..64' if n<=64 else '65+')
def trial_rate(recs,key,den):
    cohort=[r for r in recs if r[den]]
    return np.mean([float(r[key]) for r in cohort]) if cohort else np.nan

def stratum_frame(r7,rows,key,threshold):
    cohort=[r for r in rows if r['metric_mask']]
    if {r['target'] for r in cohort}=={0,1}:
        return r7.frame_confusion(rows,key,threshold,'metric_mask')
    y=np.asarray([r['target'] for r in cohort],dtype=bool)
    score=np.asarray([r[key] for r in cohort]);alarm=score>=threshold
    tp=int(np.sum(y&alarm));fn=int(np.sum(y&~alarm));fp=int(np.sum(~y&alarm));tn=int(np.sum(~y&~alarm))
    return dict(n=len(y),positive=int(y.sum()),prevalence=float(y.mean()) if len(y) else None,tp=tp,fn=fn,fp=fp,tn=tn,
                frame_recall=tp/(tp+fn) if tp+fn else None,frame_fpr=fp/(fp+tn) if fp+tn else None,
                balanced_accuracy=None,macro_f1=None,average_precision=None,brier=float(np.mean((score-y)**2)) if len(y) else None,
                metric_status='single_class; BA/F1/AP withheld',never_alarm=not bool(alarm.any()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--r8-output',type=Path,required=True);p.add_argument('--predictions',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((HERE/'PROTOCOL.json').read_text())
    pm=json.loads(a.predictions.read_text())
    if pm['status']!='complete' or pm['protocol_sha256']!=sha(HERE/'PROTOCOL.json'):raise ValueError('prediction identity')
    accepted=a.r8_output/'formal_delivery/EVALUATION_MANIFEST.json'
    manifest,runs,sources=ev.load_runs(accepted)
    pred_manifest_path=a.r8_output/'formal_delivery/PREDICTION_MANIFEST.json'
    pred_manifest=json.loads(pred_manifest_path.read_text())
    if pm['accepted_manifest_sha256']!=sha(pred_manifest_path) or pm['prepared_sha256']!=pred_manifest['prepared']['sha256'] or pm['checkpoint_index_sha256']!=pred_manifest['checkpoint_index']['sha256']:
        raise ValueError('R8 accepted prediction/prepared/index chain mismatch')
    if pm['inventory_sha256']!=sha(a.r8_output/'FORMAL_INVENTORY.json') or pm['trainer_sha256']!=sha(R8/'training/train.py') or pm['source_sha256']!=sha(HERE/'infer.py'):
        raise ValueError('inference source/inventory mismatch')
    if len(pm['unperturbed_parity'])!=9 or any(r['max_abs']>1e-6 for r in pm['unperturbed_parity']) or pm['head_parameters_unchanged'] is not True:raise ValueError('inference acceptance evidence')

    es=json.loads((a.r8_output/'formal_evaluation/summary.json').read_text())
    metricpath=a.r8_output/'formal_evaluation/metrics.csv'
    # Require the accepted evaluation output hash, not only a matching filename.
    hashes=es['output_hashes']; matching=[v for k,v in hashes.items() if Path(k).name=='metrics.csv']
    if len(matching)!=1 or matching[0]!=sha(metricpath):raise ValueError('metrics drift')
    selected=[r for r in read(metricpath) if r['group'] in protocol['groups'] and r['method_type']=='neural_operational' and r['population']=='primary' and truth(r['rule_selected'])]
    choices={(r['group'],int(r['seed']),int(r['horizon']),r['operating_point']):r for r in selected}
    if len(choices)!=9*3*11:raise ValueError('fixed working point grid')
    r7=ev._load_r7();metrics=[];trials=[];strata=[];records={};parity=[]
    artifacts=[dict(x,intervention='unperturbed') for x in manifest['artifacts'] if x['role']=='outer' and x['group'] in protocol['groups']]
    new=pm['artifacts']; expected={(g,s,k) for g in protocol['groups'] for s in protocol['seeds'] for k in protocol['new_perturbations']}
    if {(x['group'],x['seed'],x['intervention']) for x in new}!=expected or len(new)!=27:raise ValueError('new grid')
    artifacts+=new
    for artifact in artifacts:
        path=Path(artifact['path'])
        if sha(path)!=artifact['sha256']:raise ValueError('prediction hash')
        g,s,k=artifact['group'],int(artifact['seed']),artifact['intervention']
        rows=load_timeline_csv(path,protocol['horizons'],head_type=artifact['head_type'])
        validate_cross_run_identity(runs[(g,s)]['roles']['outer'],rows,protocol['horizons'])
        for h in protocol['horizons']:
            r7.add_rules(rows,f'p_H{h}',f'H{h}_');hr=r7.role_rows(rows,h,'primary')
            for op,_,_ in r7.OPS:
                c=choices[g,s,h,op];base=dict(group=g,seed=s,horizon=h,intervention=k,operating_point=op,rule=c['rule'],threshold=c['threshold'],threshold_status=c['threshold_status'],threshold_source='round8_unperturbed_calibration')
                if c['threshold']=='':metrics.append(base);continue
                th=float(c['threshold']);key=f"H{h}_{c['rule']}"
                fm=r7.frame_confusion(hr,key,th,'metric_mask');em,rec=r7.event_metrics(hr,key,th,'metric_mask')
                metrics.append(dict(base,**fm,**em))
                if k=='unperturbed':
                    errs={n:abs(float(c[n])-float(v)) for n,v in {**fm,**em}.items() if n in c and c[n] and n!='never_alarm' and not isinstance(v,bool) and math.isfinite(float(v))}
                    if max(errs.values(),default=0)>1e-10:raise ValueError('unperturbed metric parity')
                    parity.append(dict(group=g,seed=s,horizon=h,op=op,max_abs=max(errs.values(),default=0)))
                if op!='trial_FA_0.10':continue
                records[g,s,h,k]=rec
                trials.extend(dict(base,**r) for r in rec)
                for name,func in [('distance_to_onset',dist),('stable_prefix_length',prefix),('batch_proxy',lambda r:'/'.join(r['episode_id'].split('/')[:-1]))]:
                    levels=sorted({func(r) for r in hr if r['metric_mask']})
                    for level in levels:
                        # Retain full timelines and processed scores; only metric mask changes.
                        subset=[dict(r,metric_mask=bool(r['metric_mask'] and func(r)==level)) for r in hr]
                        f=stratum_frame(r7,subset,key,th)
                        if name!='distance_to_onset':
                            subset=[r for r in subset if func(r)==level]
                            e=r7.event_metrics(subset,key,th,'metric_mask')[0]
                        else:e={}
                        strata.append(dict(base,stratum=name,level=level,**f,**e))
    cis=[];rng=np.random.default_rng(20260916)
    for g in protocol['groups']:
      for h in protocol['horizons']:
       for kind in protocol['new_perturbations']:
        paired={s:({r['episode_id']:r for r in records[g,s,h,'unperturbed']},{r['episode_id']:r for r in records[g,s,h,kind]}) for s in protocol['seeds']}
        first=paired[protocol['seeds'][0]][0];clusters=sorted({r['leakage_group'] for r in first.values()})
        cluster_ids={c:[i for i,r in first.items() if r['leakage_group']==c] for c in clusters}
        for key,den in [('event_detected','uncensored_event'),('trial_false_alarm','negative_frames')]:
         values=[]
         for _ in range(200):
          draw=rng.choice(clusters,len(clusters),replace=True);deltas=[]
          for seed,(left,right) in paired.items():
           l=[];rr=[]
           for cluster in draw:
            ids=cluster_ids[cluster];l.extend(left[i] for i in ids);rr.extend(right[i] for i in ids)
           deltas.append(trial_rate(rr,key,den)-trial_rate(l,key,den))
          values.append(float(np.mean(deltas)))
         finite=np.asarray(values)[np.isfinite(values)]
         point=float(np.mean([trial_rate(list(right.values()),key,den)-trial_rate(list(left.values()),key,den) for left,right in paired.values()]))
         cis.append(dict(group=g,horizon=h,intervention=kind,metric=key,delta='perturbed minus unperturbed',point_estimate=point if math.isfinite(point) else None,ci_low=float(np.percentile(finite,2.5)) if len(finite) else None,ci_high=float(np.percentile(finite,97.5)) if len(finite) else None,unit='leakage_group',seeds=3,requested_repetitions=200,valid_repetitions=len(finite),status='available' if len(finite)==200 else 'partial_undefined_draws' if len(finite) else 'unavailable_no_valid_draws'))
    for name,value in [('metrics.csv',metrics),('trial_metrics.csv',trials),('strata.csv',strata),('paired_ci.csv',cis)]:ev.atomic_csv(a.output/name,value)
    # Reuse original diagnostic rows without recomputation/recalibration.
    for name in ['intervention_metrics.csv','fixed_multiplicative_gate_diagnostic.csv']:
        src=a.r8_output/'formal_evaluation'/name
        matching=[v for k,v in hashes.items() if Path(k).name==name]
        if len(matching)!=1 or matching[0]!=sha(src):raise ValueError('reused diagnostic hash drift')
        rows=[r for r in read(src) if r['group'] in protocol['groups']]
        ev.atomic_csv(a.output/('reused_'+name),rows)
    ev.atomic_json(a.output/'AUDIT.json',dict(status='pass',protocol_sha256=sha(HERE/'PROTOCOL.json'),predictions_sha256=sha(a.predictions),r8_metrics_sha256=sha(metricpath),source_sha256=sha(Path(__file__)),unperturbed_parity=parity,output_hashes={p.name:sha(p) for p in a.output.glob('*.csv')},limits=protocol['limitations'],object_identity='batch proxy only; no verified object mapping',stable_prefix_definition='first onset raw frame index; proxy for prefix length, not independently verified contact duration',reference_replacement='not triggered: no audited distinct valid earlier reference supplied'))
    print(json.dumps(dict(status='pass',metric_rows=len(metrics),strata_rows=len(strata))))
if __name__=='__main__':main()
