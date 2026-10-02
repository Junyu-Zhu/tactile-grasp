#!/usr/bin/env python3
"""Generate deterministic R13 tables, SVG figures, and Chinese conclusions."""
import csv, json, math, statistics as st
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'results'; O=R/'reporting'; O.mkdir(parents=True,exist_ok=True)
GROUPS=('V_original','F_history_original','V_class_trial_balanced','F_class_trial_balanced')
LABEL={'V_original':'R9视觉','F_history_original':'R10力','V_class_trial_balanced':'R12视觉','F_class_trial_balanced':'R12力'}

def read(p):
    with (R/p).open(newline='') as f:return list(csv.DictReader(f))
def num(x):
    try:return float(x)
    except:return float('nan')
def mean(rows,key):
    x=[num(r.get(key)) for r in rows];x=[v for v in x if math.isfinite(v)];return sum(x)/len(x) if x else float('nan')
def pct(x):return 'NA' if not math.isfinite(x) else f'{100*x:.2f}%'
def f2(x):return 'NA' if not math.isfinite(x) else f'{x:.2f}'
def write_csv(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)

m=read(Path('new_policies/metrics.csv'));h=read(Path('historical/metrics_48.csv'));ci=read(Path('paired/paired_ci.csv'))
loo=read(Path('diagnostics/raw_frame_fpr_leave_one_group_out.csv'));bs=read(Path('new_policies/calibration_bootstrap_summary.csv'))
fixed=read(Path('fixed_case/all_rules.csv'));support=read(Path('train_support/support_inventory.csv'))

main=[]
for g in GROUPS:
  for family in ('trial_macro_static_FPR','trial_any_static_alarm_rate'):
    for k in (1,2,4):
      rows=[r for r in m if r['role']=='validation' and r['group']==g and r['family']==family and num(r['alpha'])==.05 and int(r['k'])==k]
      main.append({'group':g,'label':LABEL[g],'family':family,'alpha':.05,'k':k,
                   **{x:mean(rows,x) for x in ('frame_static_FPR','trial_macro_static_FPR','trial_any_static_alarm_rate','gross_recall','event_recall','mean_detected_event_delay','balanced_accuracy','macro_f1')}})
write_csv(O/'MAIN_ALPHA05.csv',main)

historical=[]
for g in GROUPS:
  for rule in ('raw','confirm2'):
    rows=[r for r in h if r['role']=='validation' and r['group']==g and r['point']=='FPR0.05' and r['rule']==rule]
    historical.append({'group':g,'label':LABEL[g],'rule':rule,
                       'frame_static_FPR':mean(rows,'static_fpr'),'gross_recall':mean(rows,'gross_recall'),
                       'event_recall':mean(rows,'event_recall'),'mean_detected_event_delay':mean(rows,'mean_delay'),
                       'balanced_accuracy':mean(rows,'balanced_accuracy')})
write_csv(O/'HISTORICAL_FPR05.csv',historical)

migration=[]
for family in ('trial_macro_static_FPR','trial_any_static_alarm_rate'):
  for alpha in (.01,.05,.10):
    cal=[r for r in m if r['role']=='calibration' and r['family']==family and num(r['alpha'])==alpha]
    val=[r for r in m if r['role']=='validation' and r['family']==family and num(r['alpha'])==alpha]
    migration.append({'family':family,'alpha':alpha,'calibration_max':max(num(r[family]) for r in cal),
                      'validation_mean':mean(val,family),'validation_min':min(num(r[family]) for r in val),
                      'validation_max':max(num(r[family]) for r in val)})
write_csv(O/'TARGET_MIGRATION.csv',migration)

loo_abs=[abs(num(r['threshold_difference'])) for r in loo if math.isfinite(num(r['threshold_difference']))]
boot_width=[num(r['threshold_q975'])-num(r['threshold_q025']) for r in bs if math.isfinite(num(r['threshold_q975']))]
rank_max={'AP':0.0,'pAUC':0.0}
hist_index={(r['group'],r['fold'],r['seed'],r['role']):r for r in h if r['point']=='FPR0.05' and r['rule']=='raw'}
for r in m:
    old=hist_index[(r['group'],r['fold'],r['seed'],r['role'])]
    for key in rank_max:rank_max[key]=max(rank_max[key],abs(num(r[key])-num(old[key])))

def svg_bar(path,title,rows,value_key,color):
    W,H=980,520; left,bottom,top=95,85,70; pw=W-left-30; ph=H-bottom-top
    vals=[r[value_key] for r in rows]; ymax=max(vals)*1.15 if max(vals)>0 else 1
    bars=[]; labels=[]; bw=pw/len(rows)*.65
    for i,(r,v) in enumerate(zip(rows,vals)):
        x=left+(i+.5)*pw/len(rows)-bw/2;y=top+ph*(1-v/ymax);hh=ph*v/ymax
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{hh:.1f}" fill="{color}"/><text x="{x+bw/2:.1f}" y="{y-6:.1f}" text-anchor="middle" font-size="12">{100*v:.1f}%</text>')
        labels.append(f'<text transform="translate({x+bw/2:.1f},{H-bottom+18}) rotate(35)" text-anchor="start" font-size="11">{r["label"]} {r["family_short"]} k={r["k"]}</text>')
    grid=[]
    for j in range(6):
        v=ymax*j/5;y=top+ph*(1-j/5);grid.append(f'<line x1="{left}" y1="{y:.1f}" x2="{W-30}" y2="{y:.1f}" stroke="#ddd"/><text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-size="11">{100*v:.0f}%</text>')
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><rect width="100%" height="100%" fill="white"/><text x="{W/2}" y="32" text-anchor="middle" font-size="21" font-family="sans-serif">{title}</text>{"".join(grid)}{"".join(bars)}{"".join(labels)}<line x1="{left}" y1="{top+ph}" x2="{W-30}" y2="{top+ph}" stroke="#222"/></svg>')

plotrows=[]
for r in main:
    if r['k'] in (1,4):plotrows.append({**r,'family_short':'宏FPR' if r['family'].startswith('trial_macro') else '任意告警'})
svg_bar(O/'alpha05_trial_any_alarm.svg','α=0.05 新规则：validation 试次任意静态告警率',plotrows,'trial_any_static_alarm_rate','#d95f02')
svg_bar(O/'alpha05_gross_recall.svg','α=0.05 新规则：validation gross帧召回',plotrows,'gross_recall','#1b9e77')

# Fixed case table, normalizing historical and new field names.
fixed_summary=[]
for g in GROUPS:
    z=[r for r in fixed if r['group']==g]
    for seed in ('20260914','20260915','20260916'):
        selected=[]
        for r in z:
            if r['seed']!=seed:continue
            if r.get('point')=='FPR0.05' and r.get('rule') in ('raw','confirm2'): selected.append(r)
            if r.get('family') and num(r.get('alpha'))==.05: selected.append(r)
        for r in selected:
            static_fp=num(r.get('static_fp') or r.get('fp'))
            gross_frames=num(r.get('gross_frames'))
            if not math.isfinite(gross_frames):gross_frames=num(r.get('tp'))+num(r.get('fn'))
            gross_tp=num(r.get('gross_tp'))
            if not math.isfinite(gross_tp):gross_tp=num(r.get('tp'))
            policy=(r.get('point','')+'|'+r.get('rule','')).strip('|') if r.get('point') else r.get('policy')
            fixed_summary.append({'group':g,'seed':seed,'policy':policy,'static_frames':49,
                                  'static_fp':int(static_fp),'gross_recall':gross_tp/gross_frames})
write_csv(O/'FIXED_CASE_ALPHA05.csv',fixed_summary)
fixed_new=[r['static_fp'] for r in fixed_summary if not r['policy'].startswith('FPR0.05')]
fixed_new_min,fixed_new_max=min(fixed_new),max(fixed_new)

lines=['# 第十三轮总结：试次级低误报告警校准与稳定状态覆盖验证','',
'## 执行结论','',
'本轮完成48个冻结模型的历史复用与新规则评价，新增神经训练为0，未读取test。正式产物包含每角色480个历史工作点、864个新校准工作点、144个confirm4参考点，以及172,800条完整泄漏组bootstrap阈值重拟合记录。阈值拟合接口只接收calibration；validation只用于固定阈值评价。','',
'核心结论是否定且清晰：试次级校准确实能在calibration上满足经验约束，但不能把低误报告警目标稳定迁移到validation。更严格的“任意静态告警试次率”规则进一步降低误报，却以明显gross召回和延迟代价换取；它不是无需代价的部署改进。','',
'## α=0.05主结果（四折×三seed描述性均值）','',
'| 模型 | 规则 | k | 帧FPR | 试次宏FPR | 任意静态告警试次率 | gross召回 | 事件召回 | 已检出事件平均延迟/帧 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in main:
    lines.append(f'| {r["label"]} | {"宏FPR" if r["family"].startswith("trial_macro") else "任意告警"} | {r["k"]} | {pct(r["frame_static_FPR"])} | {pct(r["trial_macro_static_FPR"])} | {pct(r["trial_any_static_alarm_rate"])} | {pct(r["gross_recall"])} | {pct(r["event_recall"])} | {f2(r["mean_detected_event_delay"])} |')
lines += ['', '历史raw-cal FPR5在validation上的四组raw均值为18.79%–19.97%帧FPR、78.94%–84.07% gross召回；confirm2只小幅降低FPR，并增加延迟。新宏FPR族保持较高召回（α=.05、k=1时79.54%–83.24%），但试次任意告警率仍为35.83%–38.26%。新任意告警族把该比例降到19.58%–22.71%（k=1），gross召回则降至55.67%–59.70%；k=4通常再降低少量误报，同时继续损失召回或增加延迟。','',
'## 目标迁移与跨折稳定性','',
f'- 宏FPR族在calibration严格满足α；α=.05时validation宏FPR总体均值为{pct(next(r["validation_mean"] for r in migration if r["family"].startswith("trial_macro") and r["alpha"]==.05))}，单运行范围0–46.69%。',
f'- 任意告警族因calibration静态试次数分辨率有限，α=.01与.05的完整拟合均退化为零校准试次误告警；validation任意告警率总体均值仍为{pct(next(r["validation_mean"] for r in migration if r["family"].startswith("trial_any") and r["alpha"]==.05))}，范围0–50%。这不是总体保证。',
f'- 564个逐组留一raw阈值中，α=.05有{sum(abs(num(r["threshold_difference"]))>0 for r in loo if num(r["alpha"])==.05)}/{sum(num(r["alpha"])==.05 for r in loo)}发生变化；绝对变化最大{max(abs(num(r["threshold_difference"])) for r in loo if num(r["alpha"])==.05):.4f}。bootstrap阈值区间宽度中位数为{st.median(boot_width):.4f}、最大{max(boot_width):.4f}，确认校准组成敏感。',
'- 分折配对区间方向不一致且多重比较很多；不据此选择validation赢家。R12力相对R12视觉在任意告警族α=.05、k=1的描述性均值仅少1.53个百分点任意告警，但gross召回少0.47个百分点、事件召回少7.78个百分点；宏FPR族则误报更高且召回更低。现有证据不支持稳定的力增量部署收益。','',
'## 固定失败案例','',
f'fold4的`htt/p3_sliding/0_press_13`仍是稳定反例。历史raw在四组全三seed均为49/49静态帧误报，confirm2仅降到48/49，证明主要是确认冷启动延迟。新规则并未统一解决它：α=.05下不同模型/seed/k仍有{fixed_new_min}–{fixed_new_max}个静态误报帧；部分规则保持gross召回，是因为告警已在gross之前持续，不能解释为可靠起点检出。完整逐seed表见`FIXED_CASE_ALPHA05.csv`，预注册时序主图见`fixed_case_alpha05_timeline.svg`。历史和新规则的逐事件已有告警/延迟0子集见`events/validation_event_details.csv`与`validation_event_summary.csv`。','',
'## 排序、力与train支持','',
f'- 新规则的AP/pAUC与R12原始score复算最大差分别为{rank_max["AP"]:.3g}和{rank_max["pAUC"]:.3g}；确认/校准改变的是工作点和时序状态，不是原始排序。',
'- train各fold有51/51/51/53个episode和同数泄漏组，static帧917–1138、gross帧10907–11274；每折18–20个episode无static。static连续段最短1–2帧、最长88帧，稳定状态支持不均。',
'- “力变化但不滑”缺少预先定义的物理门槛和独立真值，不能从标签相关的HTT阶段规则中自行确证困难负例。R11邻域与R12贡献证据只能按其单case/单seed适用范围复用。','',
'## 下一阶段建议','',
'1. 优先做数据与校准设计：增加独立的稳定static试次和泄漏组，预先定义物理困难负例，并留出从未参与模型/规则开发的盲测。','2. 将试次任意告警作为显式成本，与事件漏报和延迟共同预注册；不要只用帧FPR或只对已检出事件的平均延迟排名。','3. 在新盲测建立前，不建议因本轮个别fold优势更换部署规则，也不建议直接扩大结构。若数据补足后排序仍不足，再比较小型时序/状态模型与现有确认规则。','4. 力信号可继续作为受控候选特征，但须用独立物理标签验证增量，避免把力规则参与生成的标签当作因果证据。','',
'## 限制','',
'四折存在跨折样本重叠，三seed不是独立物理重复；validation参与过历史模型开发选择；标签部分依赖力规则；没有独立test、实物成功率或世界模型证据。所有跨折均值仅为描述性统计。']
(O/'SUMMARY_ZH.md').write_text('\n'.join(lines)+'\n')

key={'schema':'round13_key_results_v1','status':'complete','main_alpha05':main,'historical_fpr05':historical,
     'target_migration':migration,'rank_reproduction_max_abs_difference':rank_max,
     'leave_one_group_out':{'records':len(loo),'nonzero_alpha05':sum(abs(num(r['threshold_difference']))>0 for r in loo if num(r['alpha'])==.05),
                            'max_abs_alpha05':max(abs(num(r['threshold_difference'])) for r in loo if num(r['alpha'])==.05)},
     'bootstrap':{'records':172800,'all_valid':all(int(r['valid'])==200 for r in bs),'threshold_ci_width_median':st.median(boot_width),'threshold_ci_width_max':max(boot_width)},
     'fixed_case':'htt/p3_sliding/0_press_13','new_neural_trainings':0,'test_consumed':False}
(O/'KEY_RESULTS.json').write_text(json.dumps(key,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'status':'pass','files':sorted(p.name for p in O.iterdir()),'main_rows':len(main)},ensure_ascii=False))
