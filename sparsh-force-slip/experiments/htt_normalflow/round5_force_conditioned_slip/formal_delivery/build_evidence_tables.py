#!/usr/bin/env python3
"""Descriptive joins of accepted analyses; no new fitting or selection."""
import argparse,csv,json
from pathlib import Path

def write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--results-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    force=json.loads((a.results_root/'analysis_force/summary/summary.json').read_text())
    current=json.loads((a.results_root/'analysis_summary/summary.json').read_text())
    if force['status']!='complete' or current['status']!='complete':raise ValueError('Analyses are not complete')
    rows=list(csv.DictReader((a.results_root/'analysis_summary/per_run_metrics.csv').open()))
    confusions=[]
    for metric_path in sorted({r['metrics_path'] for r in rows}):
        d=json.loads(Path(metric_path).read_text())
        for op,modes in d['operating_points'].items():
            for mode,roles in modes.items():
                if mode not in ['raw','sequential']:continue
                for role in ['calibration','validation']:
                    r=roles[role];inc=r['incipient_score']
                    confusions.append({'model':d['model_id'],'fold':d['fold'],'seed':d['seed'],'operating_point':op,'mode':mode,'role':role,**{k:r[k] for k in ['tn','fp','fn','tp','positive_prevalence']},'incipient_count':inc['count'],'incipient_mean_score':inc.get('mean'),'incipient_q10':inc.get('q10'),'incipient_q90':inc.get('q90')})
    if len(confusions)!=768:raise ValueError('Expected48x8x2 confusion summaries')
    write(a.output/'current_confusions_and_incipient.csv',confusions)
    current_by_key={(r['model'],r['fold'],int(r['seed'])):r for r in rows if r['operating_point']=='fixed_0.5' and r['mode']=='raw'}
    joined=[]
    for r in force['force_runs']:
        if r['role']!='validation':continue
        f,s=r['fold'],r['seed'];v=current_by_key['V',f,s];adapt=current_by_key['F-adapt',f,s]
        joined.append({'fold':f,'seed':s,'force_mean_axis_rmse_n':sum(r['axis_rmse_n'])/3,'force_fn_rmse_n':r['Fn_rmse_n'],'force_ft_rmse_n':r['Ft_rmse_n'],'force_target_any_saturation_fraction':r['target_any_saturation_fraction'],'V_pauc':float(v['partial_tpr_auc_0_0p1']),'F_adapt_pauc':float(adapt['partial_tpr_auc_0_0p1']),'F_adapt_minus_V_pauc':float(adapt['partial_tpr_auc_0_0p1'])-float(v['partial_tpr_auc_0_0p1']),'interpretation':'different force and slip trial sets within matched fold/seed; descriptive, not trial-level causality'})
    if len(joined)!=12:raise ValueError('Expected all12 matched fold/seed rows')
    write(a.output/'force_slip_association.csv',joined)
    sensitivity=[]
    manifest=json.loads((Path(__file__).resolve().parents[1]/'analysis_force/ANALYSIS_MANIFEST.json').read_text())
    for job in manifest['jobs']:
        if not job['id'].startswith('sensitivity_'):continue
        d=json.loads(Path(job['acceptance_path']).read_text())
        for intervention,m in d['metrics'].items():
            sensitivity.append({'variant':d['variant'],'fold':d['fold'],'seed':d['seed'],'role':d['role'],'intervention':intervention,**m,'delta_pauc_vs_original':m['partial_tpr_auc_0_0p1']-d['metrics']['original']['partial_tpr_auc_0_0p1']})
    if len(sensitivity)!=144:raise ValueError(f'Expected48x3 sensitivity metric rows, got {len(sensitivity)}')
    write(a.output/'sensitivity_metric_changes.csv',sensitivity)
    timing=[]
    for model,r in force['e2e'].items():
        for mode,m in r['measurement'].items():timing.append({'model':model,'mode':mode,'total_parameters':r['total_unique_parameters'],'median_ms':m['median_ms'],'p10_ms':m['p10_ms'],'p90_ms':m['p90_ms'],'peak_allocated_mib':m['peak_allocated_bytes']/1024**2,'warmup':r['warmup'],'measured_steps':r['steps']})
    if len(timing)!=10:raise ValueError('Expected5 deployment paths x2 history modes')
    write(a.output/'e2e_cost.csv',timing)
    print(json.dumps({'status':'complete','force_slip_rows':len(joined),'sensitivity_rows':len(sensitivity),'timing_rows':len(timing)}))
if __name__=='__main__':main()
