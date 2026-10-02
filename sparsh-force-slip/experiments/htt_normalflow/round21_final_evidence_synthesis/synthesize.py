#!/usr/bin/env python3
"""Bounded synthesis of accepted outputs only; performs no model loading/inference."""
from __future__ import annotations

import csv, hashlib, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
LOCAL_REPO = HERE.parents[2]
LOCAL_CANONICAL_REPO = Path('/home/zjy/Documents/grasp/tactile_grasp/sparsh-force-slip')
SERVER_REPO = Path('/home/zjy/document/tactile-grasp/sparsh-force-slip')

SOURCES = {
    "r10_force": EXP/"round10_htt_force_supervision_adaptation/results/force_aggregate/per_run_summary.csv",
    "r10_status": EXP/"round10_htt_force_supervision_adaptation/FINAL_STATUS.json",
    "r10_regression_scope": EXP/"round10_htt_force_supervision_adaptation/reviews/REGRESSION_SCOPE.md",
    "r14_future": EXP/"round14_htt_future_force_dual/results/final_report/horizon_summary.csv",
    "r15_future": EXP/"round15_htt_future_force_residual_history/results/final_report/horizon_summary.csv",
    "r16_summary": EXP/"round16_touchd_force_transfer/results/reporting/SUMMARY_ZH.md",
    "r16_old_force_audit": EXP/"round16_touchd_force_transfer/results/old_force_regression/AUDIT.json",
    "r17_headline": EXP/"round17_htt_shared_temporal_multitask/results/final_report/HEADLINE_SUMMARY.csv",
    "r17_slip_runs_snapshot": HERE/"source_snapshots/R17_SLIP_RUN_RESULTS.csv",
    "r17_slip_ci": EXP/"round17_htt_shared_temporal_multitask/results/evaluation/bootstrap/PAIRED_CI.csv",
    "r17_all_workpoints": EXP/"round17_htt_shared_temporal_multitask/results/evaluation/slip/metrics.csv",
    "r17_descriptive_curves": EXP/"round17_htt_shared_temporal_multitask/results/evaluation/slip/descriptive_matched_curves.csv",
    "r17_summary": EXP/"round17_htt_shared_temporal_multitask/results/final_report/SUMMARY_ZH.md",
    "r18_headline": EXP/"round18_htt_force_conditioned_film/results/final_report/HEADLINE_SUMMARY.csv",
    "r18_headline_runs_snapshot": HERE/"source_snapshots/R18_HEADLINE_RUNS.csv",
    "r18_slip_ci": EXP/"round18_htt_force_conditioned_film/results/final_report/PAIRED_CI.csv",
    "r18_status": EXP/"round18_htt_force_conditioned_film/FINAL_STATUS.json",
    "r18_identity": EXP/"round18_htt_force_conditioned_film/IDENTITY_LOCK.json",
    "r18_film": EXP/"round18_htt_force_conditioned_film/results/final_report/FILM_DISTRIBUTION_RESULTS.csv",
    "r18_q4": EXP/"round18_htt_force_conditioned_film/DEFORMABLE_OBJECTS_AUDIT.json",
    "r18_remote_large_index": EXP/"round18_htt_force_conditioned_film/REMOTE_LARGE_ARTIFACTS.json",
    "r19_summary": HERE/"source_snapshots/R19_REPORT_SUMMARY.json",
    "r19_workpoints": HERE/"source_snapshots/R19_WORKPOINT_METRICS.csv",
    "r19_film": HERE/"source_snapshots/R19_FILM_SUMMARY.csv",
    "r19_slip_ci": HERE/"source_snapshots/R19_PAIRED_CI.csv",
    "r19_training": EXP/"round19_htt_partial_encoder_finetuning/TRAINING_AUDIT.json",
    "r20_summary": EXP/"round20_htt_contact_state_transition/results/report/SUMMARY.json",
    "r20_numeric": EXP/"round20_htt_contact_state_transition/results/evaluation/metrics/NUMERIC_AXIS_WINDOW.csv",
    "r20_strata_ci": EXP/"round20_htt_contact_state_transition/results/report/STRATA_PAIRED_CI.csv",
    "r20_future_ci": EXP/"round20_htt_contact_state_transition/results/report/PAIRED_CI.csv",
    "r20_q2": EXP/"round20_htt_contact_state_transition/Q2_SUPPORT_CONCLUSION.md",
    "r20_status": EXP/"round20_htt_contact_state_transition/FINAL_STATUS.json",
    "r14_future_ci": EXP/"round14_htt_future_force_dual/results/reporting/paired_group_ci.csv",
    "r15_future_ci": EXP/"round15_htt_future_force_residual_history/results/reporting/paired_group_ci.csv",
}
for p in sorted((EXP/'round19_htt_partial_encoder_finetuning/logs').glob('[VM]2_p*_s*.log')):
    SOURCES['r19_log_'+p.stem]=p

def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def rows(p):
    with p.open(newline='', encoding='utf-8') as f: return list(csv.DictReader(f))

def write_csv(name, fieldnames, data):
    p=HERE/name
    with p.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(data)
    return p

def write_text(name, s):
    p=HERE/name; p.write_text(s.rstrip()+"\n", encoding='utf-8'); return p

def identity_table():
    common="HTT train→fit/selection; calibration thresholds only; validation development-only; test reserved"
    data=[
      dict(component="R10-current-force",round="10",checkpoint_pointer="round10_htt_force_supervision_adaptation/results/formal_delivery/ALL_CHECKPOINT_INDEX.csv sha256=5288b58ea27534d06f9ba7fbad3f1009b29e5d2f7d57d770b0dae3ca12911c1f",upstream_hash_pointer="round10_htt_force_supervision_adaptation/results/formal_delivery/FORCE_TRAINING_AUDIT.json sha256=d5ece04bb2b7f7ce421c40ab9b905bb2713624e56c7fadf2ae5dd4515d1dbac9",trainable="force head/adaptation modules per R10 protocol",dataset_roles=common,history_endpoints="current target; diagnostics t>=13",force_target="clip((6d_force-ref_force)[:3],-20,20) N",selector="selection mean native-N axis MAE",calibration="none for force; slip thresholds separate",metrics="axis MAE/RMSE/bias/std/lag5/clip",source="R10 force report/index"),
      dict(component="R18-V0",round="18",checkpoint_pointer="round18_htt_force_conditioned_film/CHECKPOINT_INDEX.json sha256=8aef81fcf1d38f11297e4d1d937a9972bbf77c395915f5d635448fb3126612ae group=V0",upstream_hash_pointer="round18_htt_force_conditioned_film/IDENTITY_LOCK.json sha256=e457ff5fd2f9cd2f18511a08b5ccd5a53d04aaba37d9b18e7046e89c82e93a27",trainable="LN+GRU+risk head; frozen visual upstream",dataset_roles=common,history_endpoints="R17 future-complete common endpoints; t-13..t raw union",force_target="no force input",selector="implemented R18 selection pAUC with boundary numeric bias; chosen checkpoints retained after R19 audit",calibration="model-specific; fixed.5/maxBA/FPR1/5/10 + R13 rules",metrics="pAUC/AP/FPR/recall/alarms/event delay",source="R18 accepted package; SELECTOR_NUMERIC_AUDIT.json"),
      dict(component="R18-C0",round="18",checkpoint_pointer="round18_htt_force_conditioned_film/CHECKPOINT_INDEX.json sha256=8aef81fcf1d38f11297e4d1d937a9972bbf77c395915f5d635448fb3126612ae group=C0",upstream_hash_pointer="round18_htt_force_conditioned_film/IDENTITY_LOCK.json sha256=e457ff5fd2f9cd2f18511a08b5ccd5a53d04aaba37d9b18e7046e89c82e93a27",trainable="force concatenation+GRU+risk; frozen upstream",dataset_roles=common,history_endpoints="same R17 common endpoints",force_target="R10 predicted current xyz as input; no GT",selector="implemented R18 selection pAUC; historical checkpoint unchanged after numeric audit",calibration="model-specific",metrics="same slip suite",source="R18 accepted package"),
      dict(component="R18-M0-FiLM",round="18",checkpoint_pointer="round18_htt_force_conditioned_film/CHECKPOINT_INDEX.json sha256=8aef81fcf1d38f11297e4d1d937a9972bbf77c395915f5d635448fb3126612ae group=M0",upstream_hash_pointer="round18_htt_force_conditioned_film/IDENTITY_LOCK.json sha256=e457ff5fd2f9cd2f18511a08b5ccd5a53d04aaba37d9b18e7046e89c82e93a27",trainable="zero-init FiLM+LN+GRU+risk; frozen upstream",dataset_roles=common,history_endpoints="same R17 common endpoints",force_target="fit-normalized detached R10 predicted current xyz",selector="implemented R18 selection pAUC; counterfactual corrected patience path not replayed",calibration="model-specific",metrics="same slip suite + FiLM/perturbation diagnostics",source="R18 accepted package"),
      dict(component="R19-V2",round="19",checkpoint_pointer="round19_htt_partial_encoder_finetuning/CHECKPOINT_INDEX.json sha256=46176b9a735d8cf9cbfcca8eacdbb34ddbc4431901f6bd9bc513dc37a7f59f51 group=V2",upstream_hash_pointer="round19_htt_partial_encoder_finetuning/TRAINING_AUDIT.json sha256=bfd9dfd89d5e90fcd2195f3d24af97c3e87760487ee020a7f8664c5f6625a574; remote PREFIX_INDEX hash recorded per checkpoint",trainable="MAE blocks10-11 + new V head; original norm/pooler/trunk frozen",dataset_roles=common,history_endpoints="exact R17 common cache; t-13..t",force_target="no force input",selector="corrected pAUC[0,.1] on selection; earliest strict maximum",calibration="own calibration thresholds",metrics="same slip suite",source="R19 accepted package"),
      dict(component="R19-M2",round="19",checkpoint_pointer="round19_htt_partial_encoder_finetuning/CHECKPOINT_INDEX.json sha256=46176b9a735d8cf9cbfcca8eacdbb34ddbc4431901f6bd9bc513dc37a7f59f51 group=M2",upstream_hash_pointer="round19_htt_partial_encoder_finetuning/TRAINING_AUDIT.json sha256=bfd9dfd89d5e90fcd2195f3d24af97c3e87760487ee020a7f8664c5f6625a574; remote PREFIX_INDEX hash recorded per checkpoint",trainable="MAE blocks10-11 + zero-init FiLM/new head; force path frozen",dataset_roles=common,history_endpoints="exact R17 common cache; t-13..t",force_target="detached fit-normalized R10 predicted xyz",selector="corrected pAUC[0,.1] on selection; earliest strict maximum",calibration="own calibration thresholds",metrics="same slip suite + post-hoc force-error strata",source="R19 accepted package"),
      dict(component="R14-future",round="14",checkpoint_pointer="round14_htt_future_force_dual/ARTIFACT_INDEX.json sha256=9e698abdd9c6c9cdb0b7b132438a7cfe2b6205164fda49512a5e30a74de89995 (formal remote pointers)",upstream_hash_pointer="round14_htt_future_force_dual/PREPARE_AUDIT.json sha256=b2257f424248eb1559c439a7922bf0ef989668a474c4f7c1e2b8d828369bf620",trainable="V/F_concat/F_dual GRU future heads",dataset_roles=common,history_endpoints="9 base steps; t-13..t; complete h1/5/10",force_target="absolute reference-relative clipped xyz future N",selector="selection absolute future MAE",calibration="evaluation-only",metrics="absolute future and derived deployed-change MAE/RMSE",source="R14 accepted report"),
      dict(component="R15-future-residual",round="15",checkpoint_pointer="round15_htt_future_force_residual_history/CHECKPOINT_INDEX.csv sha256=d6d206e1cb59dfb09a610d0327a5affa1439861890c6123a80e40cc4521a594a",upstream_hash_pointer="round15_htt_future_force_residual_history/ARTIFACT_INDEX.json sha256=a347071ee9398d926f18a23509916799f2759e5eaecacb1a5a3bed5008ea37cc",trainable="matched C_current/H_history residual GRUs",dataset_roles=common,history_endpoints="current base vs 9-step history; same complete h1/5/10",force_target="residual to predicted-current anchor; targets clipped xyz",selector="selection reconstructed future MAE",calibration="evaluation-only",metrics="absolute future/change/anchor decomposition",source="R15 accepted report"),
      dict(component="R16-pretransfer",round="16",checkpoint_pointer="round16_touchd_force_transfer/CHECKPOINT_INDEX.json sha256=f1d08529143aaab27f0c084a527d3d4ed8bed62f7d65f052e3ecd47228a2c793",upstream_hash_pointer="round16_touchd_force_transfer/RUN_INDEX_AUDIT.json sha256=226e24813a52e67c486359ba4daf4bbff856244975a4167ef83d8a478647a14f",trainable="ToucHD pretransfer route then HTT-private adaptation (H vs T_H)",dataset_roles="R16 fixed roles over all released ToucHD objects; official probe 7/3 lists explicitly not assumed complete; HTT development folds; test not consumed",history_endpoints="paired compact ToucHD i/i-3; HTT t>=13",force_target="domain-private axes; HTT accepted clipped xyz N",selector="domain-specific internal selection",calibration="slip evaluation reuses accepted rules",metrics="HTT axis MAE/RMSE/change/bias; slip compatibility",source="R16 accepted report/audit; historical background only"),
      dict(component="R17-shared",round="17",checkpoint_pointer="round17_htt_shared_temporal_multitask/RUN_INDEX.json sha256=81af8f30f18aefda9982b564d1e0e334c91101b02ab25286456111fb6e26656d groups=S/F/J",upstream_hash_pointer="round17_htt_shared_temporal_multitask/ARTIFACT_INDEX.json sha256=9dd5905b998c1e00e9304822d96547e7529cc80d04ef7ddc4786a32d41e668bc",trainable="shared GRU with slip/future heads according to S/F/J",dataset_roles=common,history_endpoints="9-step t-13..t, complete h1/5/10",force_target="absolute clipped future xyz for F/J",selector="S/J slip pAUC; F future MAE; J one slip-selected checkpoint",calibration="slip thresholds only",metrics="slip suite + future/anchor/cross-term",source="R17 accepted negative result"),
      dict(component="R20-future-transition",round="20",checkpoint_pointer="round20_htt_contact_state_transition/CHECKPOINT_INDEX.json sha256=c8692474525fda463ac64d037ef88f9fdd5cea5a88783efba4c09768a9e31a80",upstream_hash_pointer="round20_htt_contact_state_transition/PREPARE.json sha256=3d665d05575112b2562ea855d8e3ac18b364c72c6f0182208c4b5dc5424e0572",trainable="K-current/D-history/T-visual/T-force matched small heads",dataset_roles="HTT p1 only: fit/selection; calibration/validation evaluation; test reserved",history_endpoints="same R14 cache; h1/5/10; K current vs histories/transitions",force_target="true clipped-force delta; shared R10 predicted-current anchor",selector="selection true-change MAE",calibration="evaluation-only",metrics="absolute future + true-change MAE/RMSE, anchor/cross-term, strata",source="R20 accepted single-fold package"),
      dict(component="Q2-audit",round="20",checkpoint_pointer="none; no training",upstream_hash_pointer="round20_htt_contact_state_transition/Q2_SUPPORT_AUDIT.json sha256=b4ae94fbc533d9dfdb6b74b989eb6022853328a231be3c4180e787078d9e2429",trainable="none",dataset_roles="four-fold role support audit",history_endpoints="R14 complete-window endpoints",force_target="not applicable",selector="gate requires >=3/class outside fit and >=10/class fit",calibration="p4 incipient only 1 trial",metrics="independent trial/leakage-group support",source="R20 Q2 audit"),
      dict(component="Q4-audit",round="18",checkpoint_pointer="none; no inference",upstream_hash_pointer="round18_htt_force_conditioned_film/DEFORMABLE_OBJECTS_AUDIT.json sha256=c9cc9532ef33f914c73d3fce909a9f6036570d976f1fb1d14e73d82e51827d7d",trainable="none",dataset_roles="development metadata only; testing unread",history_endpoints="sequence/window metadata; no onset times",force_target="arrays exist; HTT three-axis compatibility unverified",selector="none",calibration="no independent calibration-object role",metrics="content/split/applicability audit",source="R18 Q4 development audit"),
    ]
    return write_csv("COMPONENT_IDENTITY_MATRIX.csv", list(data[0]), data)

def force_table():
    rr=[r for r in rows(SOURCES['r10_force']) if r['task']=='slip_force' and r['role']=='validation' and r['population']=='all' and r['aggregation']=='complete_trial_macro']
    out=[]
    for variant in ('old','new'):
      for axis in ('shear_x','shear_y','normal'):
        x=[r for r in rr if r['variant']==variant and r['axis']==axis]
        def avg(k): return sum(float(r[k]) for r in x)/len(x)
        out.append(dict(domain="HTT slip-domain",task="R10 adapted current force",variant=variant,axis=axis,runs=len(x),mae_n=avg('mae'),rmse_n=avg('rmse'),bias_n=avg('bias'),prediction_std_n=avg('prediction_std'),target_std_n=avg('target_std'),amplitude_ratio=avg('prediction_std')/avg('target_std'),change_mae_lag5_n=avg('delta5_mae'),target_clip_fraction="unavailable_from_clipped_cache",prediction_outside_clip_fraction=avg('prediction_outside_clip_fraction'),aggregation="mean of fold×seed complete-trial macro validation outputs"))
    # accepted R16 exact report values, distinct domain/provenance
    vals={'H':[(.8537,1.3053,.5146,.0954),(.7869,1.2338,.4301,-.0334),(1.0390,1.5105,.6250,.0538)],'T_H':[(.8270,1.2469,.5061,.0444),(.7634,1.1801,.4235,-.0668),(1.0508,1.5441,.6365,.1441)]}
    for v, vs in vals.items():
      for axis,z in zip(('shear_x','shear_y','normal'),vs): out.append(dict(domain="HTT validation (R16 historical)",task="R16 pretransfer comparison",variant=v,axis=axis,runs=12,mae_n=z[0],rmse_n=z[1],bias_n=z[3],prediction_std_n="",target_std_n="",amplitude_ratio="",change_mae_lag5_n=z[2],target_clip_fraction="",prediction_outside_clip_fraction="",aggregation="accepted R16 descriptive mean"))
    return write_csv("FORCE_EVIDENCE.csv", list(out[0]), out)

def legacy_regression_table():
    rr=[r for r in rows(SOURCES['r10_force']) if r['task']=='old_force_regression' and r['role']=='validation' and r['population']=='all' and r['aggregation']=='complete_trial_macro']
    out=[]
    for variant in ('old','new'):
      for axis in ('shear_x','shear_y','normal'):
        x=[r for r in rr if r['variant']==variant and r['axis']==axis]
        def avg(k): return sum(float(r[k]) for r in x)/len(x)
        out.append(dict(domain="HTT dedicated historical force-task validation",explicitly_not="not Sparsh force regression; not ToucHD",variant=variant,axis=axis,runs=len(x),mae_n=avg('mae'),rmse_n=avg('rmse'),bias_n=avg('bias'),prediction_std_n=avg('prediction_std'),target_std_n=avg('target_std'),change_mae_lag5_n=avg('delta5_mae'),source="R10 force_aggregate/per_run_summary.csv task=old_force_regression role=validation; R10 reviews/REGRESSION_SCOPE.md"))
    return write_csv("LEGACY_FORCE_TASK_REGRESSION.csv", list(out[0]), out)

def slip_table():
    out=[]
    for r in rows(SOURCES['r18_headline']): out.append(dict(round=18,model=r['group'],support="4 folds × 3 seeds; future-complete HTT endpoints",pAUC=r['pAUC'],AP=r['AP'],frame_FPR=r['frame_static_FPR'],gross_recall=r['gross_recall'],false_starts_per_trial=r['false_starts_per_trial'],event_recall=r['event_recall'],CI_pointer="experiments/htt_normalflow/round18_htt_force_conditioned_film/results/final_report/PAIRED_CI.csv [SOURCE_INDEX:r18_slip_ci]",workpoint="own calibration FPR5; fixed executable"))
    j=json.loads(SOURCES['r19_summary'].read_text())
    for m,v in j['headline'].items():
      if m in ('V2','M2'): out.append(dict(round=19,model=m,support="4 folds × 3 seeds; same endpoints",pAUC=v['pAUC'],AP=v['AP'],frame_FPR=v['frame_static_FPR'],gross_recall=v['gross_recall'],false_starts_per_trial=v['false_starts_per_trial'],event_recall=v['event_recall'],CI_pointer="experiments/htt_normalflow/round19_htt_partial_encoder_finetuning/results/evaluation/bootstrap/PAIRED_CI.csv [SOURCE_INDEX:r19_slip_ci]",workpoint="own calibration FPR5; fixed executable"))
    for r in rows(SOURCES['r17_headline']):
      if r['task']=='slip': out.append(dict(round=17,model=r['model'],support="4 folds × 3 seeds; common endpoints",pAUC=r['pAUC'],AP=r['AP'],frame_FPR=r['frame_static_FPR'],gross_recall=r['gross_recall'],false_starts_per_trial=r['false_starts_per_trial'],event_recall=r['event_recall'],CI_pointer="experiments/htt_normalflow/round17_htt_shared_temporal_multitask/results/evaluation/bootstrap/PAIRED_CI.csv [SOURCE_INDEX:r17_slip_ci]",workpoint="calibration FPR5; fixed executable"))
    return write_csv("SLIP_EVIDENCE.csv", list(out[0]), out)

def slip_fold_seed_table():
    fields=['round','group','fold','seed','workpoint_kind','frame_static_FPR','gross_recall','pAUC','AP','false_starts_per_trial','event_recall','mean_detected_event_delay','CI_pointer','descriptive_curve_pointer']
    out=[]
    for rd,p,groups,ci,curve in [(17,SOURCES['r17_slip_runs_snapshot'],{'S','J'},'experiments/htt_normalflow/round17_htt_shared_temporal_multitask/results/evaluation/bootstrap/PAIRED_CI.csv [SOURCE_INDEX:r17_slip_ci]','experiments/htt_normalflow/round17_htt_shared_temporal_multitask/results/evaluation/slip/descriptive_matched_curves.csv [SOURCE_INDEX:r17_descriptive_curves]'),(18,SOURCES['r18_headline_runs_snapshot'],{'V0','C0','M0'},'experiments/htt_normalflow/round18_htt_force_conditioned_film/results/final_report/PAIRED_CI.csv [SOURCE_INDEX:r18_slip_ci]','round18 REMOTE_LARGE_ARTIFACTS.json reviewer supplement; descriptive only'),(19,SOURCES['r19_workpoints'],{'V2','M2'},'experiments/htt_normalflow/round19_htt_partial_encoder_finetuning/results/evaluation/bootstrap/PAIRED_CI.csv [SOURCE_INDEX:r19_slip_ci]','round19 DESCRIPTIVE_CURVES.csv; SHA recorded in SLIP_ALL_WORKPOINT_SOURCE_INDEX.md')]:
      for r in rows(p):
        if r.get('group') not in groups: continue
        if rd in (18,19) and not (r.get('role')=='validation' and r.get('policy')=='FPR5|raw'): continue
        out.append(dict(round=rd,group=r['group'],fold=r['fold'],seed=r['seed'],workpoint_kind='fixed executable calibration-FPR5 raw',frame_static_FPR=r['frame_static_FPR'],gross_recall=r['gross_recall'],pAUC=r['pAUC'],AP=r['AP'],false_starts_per_trial=r['false_starts_per_trial'],event_recall=r['event_recall'],mean_detected_event_delay=r['mean_detected_event_delay'],CI_pointer=ci,descriptive_curve_pointer=curve))
    return write_csv('SLIP_FOLD_SEED_FIXED_WORKPOINTS.csv',fields,out)

def future_table():
    out=[]
    for key,rd,target,anchor,scope in [('r14_future',14,'absolute clipped future force','direct output; predicted-current persistence comparator','4 folds×3 seeds'),('r15_future',15,'residual-reconstructed future force','R10 predicted-current anchor','4 folds×3 seeds')]:
      ci=('experiments/htt_normalflow/round14_htt_future_force_dual/results/reporting/paired_group_ci.csv [SOURCE_INDEX:r14_future_ci]' if rd==14 else 'experiments/htt_normalflow/round15_htt_future_force_residual_history/results/reporting/paired_group_ci.csv [SOURCE_INDEX:r15_future_ci]')
      for r in rows(SOURCES[key]): out.append(dict(round=rd,group=r['group'],method=r['method'],horizon=r['horizon'],future_mae_n=r['future_mae_mean'],true_change_mae_n="see round-specific detailed table",target=target,anchor=anchor,cross_term="reported in round diagnostics",segments="R14 fixed stable/transition/changing",scope=scope,CI_pointer=ci))
    for r in rows(SOURCES['r17_headline']):
      if r['task']=='future': out.append(dict(round=17,group=r['model'],method='neural',horizon='all headline',future_mae_n=r['future_mae'],true_change_mae_n=r['deployed_change_mae'],target='absolute clipped future force',anchor='R10 predicted current for deployed-change metric',cross_term='round17 FUTURE_ERROR_DECOMPOSITION.csv',segments='fit-quantile strata',scope='4 folds×3 seeds',CI_pointer='experiments/htt_normalflow/round17_htt_shared_temporal_multitask/results/evaluation/bootstrap/PAIRED_CI.csv [SOURCE_INDEX:r17_slip_ci]'))
    j=json.loads(SOURCES['r20_summary'].read_text())
    for g,z in j['summary_validation_mae_n'].items():
      for h in ('1','5','10'): out.append(dict(round=20,group=g,method='neural' if g in ('K-current','D-history','T-visual','T-force') else 'baseline',horizon=h,future_mae_n=z['future'][h],true_change_mae_n=z['true_change'][h],target='true clipped-force delta plus common predicted anchor',anchor='same R10 predicted current (ideal_GT_hold excepted)',cross_term='round20 error decomposition; common anchor',segments='R14 fixed physical thresholds; stable/changing separate',scope='p1 only ×3 seeds; exploratory',CI_pointer='experiments/htt_normalflow/round20_htt_contact_state_transition/results/report/PAIRED_CI.csv [SOURCE_INDEX:r20_future_ci]'))
    return write_csv("FUTURE_FORCE_EVIDENCE.csv", list(out[0]), out)

def diagnostic_tables():
    film=[]
    for rd,key,group in [(18,'r18_film','M0'),(19,'r19_film','M2')]:
      x=[r for r in rows(SOURCES[key]) if r['role']=='validation']
      def avg(k): return sum(float(r[k]) for r in x)/len(x)
      film.append(dict(round=rd,group=group,runs=len(x),anomalous_fraction_mean=avg('anomalous_fraction'),anomalous_fraction_min=min(float(r['anomalous_fraction']) for r in x),anomalous_fraction_max=max(float(r['anomalous_fraction']) for r in x),norm_ratio_q95_mean=avg('norm_ratio_q95'),norm_ratio_max_observed=max(float(r['norm_ratio_max']) for r in x),large_force_error_anomaly_fraction_mean=avg('anomaly_fraction_large_force_error'),other_anomaly_fraction_mean=avg('anomaly_fraction_other'),interpretation='descriptive association/sensitivity; not causal'))
    p1=write_csv('FILM_DIAGNOSTICS.csv',list(film[0]),film)
    hist=[]
    for k,p in SOURCES.items():
      if not k.startswith('r19_log_'): continue
      epochs=[]
      for line in p.read_text(errors='replace').splitlines():
        if not line.startswith('{"epoch"'): continue
        try: z=json.loads(line)
        except json.JSONDecodeError: continue
        if {'epoch','loss','pauc'}<=z.keys(): epochs.append(z)
      # recovery may duplicate prefix; last record for an epoch is authoritative
      by={int(z['epoch']):z for z in epochs}; epochs=[by[e] for e in sorted(by)]
      best=max(epochs,key=lambda z:(z['pauc'],-z['epoch'])); minloss=min(epochs,key=lambda z:(z['loss'],z['epoch']))
      hist.append(dict(run=p.stem,epochs=len(epochs),first_fit_loss=epochs[0]['loss'],final_fit_loss=epochs[-1]['loss'],minimum_fit_loss=minloss['loss'],minimum_fit_loss_epoch=minloss['epoch'],best_selection_pAUC=best['pauc'],best_selection_epoch=best['epoch'],same_epoch_minloss_and_best_pAUC=(minloss['epoch']==best['epoch']),interpretation='fit loss and selection pAUC differ; no validation-prediction trajectory, so overfit not confirmed'))
    p2=write_csv('R19_FIT_SELECTION_HISTORY.csv',list(hist[0]),hist)
    numeric=[r for r in rows(SOURCES['r20_numeric']) if r['role']=='validation']
    p3=write_csv('R20_AXIS_WINDOW_STABILITY.csv',list(numeric[0]),numeric)
    return [p1,p2,p3]

def evidence_svg(csv_name, metric, label_keys, panel_key, title, outname):
    rr=[]
    for r in rows(HERE/csv_name):
      try: rr.append((str(r[panel_key]),' / '.join(str(r.get(k,'')) for k in label_keys),float(r[metric])))
      except (ValueError,KeyError): pass
    panels=[]
    for x in rr:
      if x[0] not in panels: panels.append(x[0])
    y=42; body=[]
    for panel in panels:
      vals=[x for x in rr if x[0]==panel]; mx=max(v for _,_,v in vals) or 1
      body.append(f'<text x="5" y="{y+14}" font-size="14" font-weight="bold">{panel}</text>'); y+=22
      for _,lab,v in vals:
        w=420*v/mx; body.append(f'<text x="8" y="{y+13}" font-size="10">{lab}</text><rect x="250" y="{y}" width="{w:.1f}" height="14" fill="#4878a8"/><text x="{255+w:.1f}" y="{y+12}" font-size="10">{v:.4f}</text>'); y+=19
      y+=10
    return write_text(outname,f'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="{y+20}"><text x="5" y="20" font-size="16">{title}</text>'+''.join(body)+'</svg>')

def main():
    missing=[str(p) for p in SOURCES.values() if not p.exists()]
    if missing: raise SystemExit('missing inputs: '+json.dumps(missing))
    generated=[identity_table(),force_table(),legacy_regression_table(),slip_table(),slip_fold_seed_table(),future_table()]+diagnostic_tables()
    generated += [evidence_svg('FORCE_EVIDENCE.csv','mae_n',['variant','axis'],'task','Force MAE, paired within each stated task only','FORCE_EVIDENCE.svg'),evidence_svg('SLIP_EVIDENCE.csv','pAUC',['round','model'],'support','Slip pAUC on stated common support; all models retained','SLIP_EVIDENCE.svg'),evidence_svg('FUTURE_FORCE_EVIDENCE.csv','future_mae_n',['group','method','horizon'],'round','Future-force MAE faceted by round; targets/anchors differ across panels','FUTURE_FORCE_EVIDENCE.svg')]
    runlog=write_text('SYNTHESIS_RUN_LOG.json',json.dumps({'schema':'round21_synthesis_run_log_v1','status':'pass','mode':'existing accepted outputs only','inputs':len(SOURCES),'generated_numeric_and_figure_artifacts':len(generated),'training':0,'inference':0,'model_loads':0,'test_access':0,'external_dataset_access':0},indent=2))
    generated.append(runlog)
    source_index={}
    for k,p in SOURCES.items():
        rel=p.resolve().relative_to(LOCAL_REPO)
        source_index[k]={'repo_relative_path':rel.as_posix(),'local_path':str(LOCAL_CANONICAL_REPO/rel),'server_path':str(SERVER_REPO/rel),'sha256':sha(p),'bytes':p.stat().st_size}
    write_text('SOURCE_INDEX.json',json.dumps({'schema':'round21_source_index_v1','scope':'accepted existing outputs only','inputs':source_index},indent=2,ensure_ascii=False))
    out={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in generated}
    write_text('SYNTHESIS_OUTPUTS.json',json.dumps({'schema':'round21_generated_outputs_v1','outputs':out},indent=2,ensure_ascii=False))
    print(json.dumps({'status':'pass','inputs':len(SOURCES),'generated':len(generated)},ensure_ascii=False))

if __name__=='__main__': main()
