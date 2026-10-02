#!/usr/bin/env python3
"""Read-only Round 3 error diagnosis; no model selection or training."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP))
import adapters

ROOT = Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow')
OUT = ROOT/'round4_comprehensive/diagnostics'


def longest(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return int(max(ends-starts, default=0))


def write_csv(path, rows):
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT/'round1/splits.json').read_text())
    episodes = {x['id']: x for x in manifest['episodes']}
    rows, positions, sequences = [], [], {}
    for path in sorted((ROOT/'round3_mae_slip_adaptation/runs').glob('*/*/*/predictions/validation.csv')):
        variant, fold, seed = path.parts[-5:-2]
        groups = {}
        for r in csv.DictReader(path.open()):
            groups.setdefault(r['episode_id'], []).append(r)
        for eid, records in groups.items():
            records.sort(key=lambda r: int(r['t']))
            p = np.array([float(r['p_slip']) for r in records])
            y = np.array([int(r['stage']) for r in records])
            t = np.array([int(r['t']) for r in records])
            assert np.array_equal(t, np.arange(len(t)))
            fp, fn = (y == 0)&(p >= .5), (y == 2)&(p < .5)
            row = dict(variant=variant, fold=fold, seed=seed, episode_id=eid,
                       probe=episodes[eid]['group'], static=int(sum(y == 0)), gross=int(sum(y == 2)),
                       fp=int(sum(fp)), fn=int(sum(fn)), high_conf_fp=int(sum((y == 0)&(p >= .9))),
                       longest_fn=longest(fn), longest_fp=longest(fp),
                       static_alarm_starts=int(sum(fp & ~np.r_[False, p[:-1] >= .5])),
                       static_fpr=float(np.mean(p[y == 0] >= .5)) if sum(y == 0) else None,
                       gross_miss=float(np.mean(p[y == 2] < .5)) if sum(y == 2) else None)
            rows.append(row); sequences[(variant,fold,seed,eid)] = (t,y,p)
            for q in range(4):
                mask=(t/len(t) >= q/4)&(t/len(t) < (q+1)/4)
                positions.append(dict(variant=variant, fold=fold, seed=seed, episode_id=eid,
                    quarter=q, static=int(sum(mask&(y == 0))), gross=int(sum(mask&(y == 2))),
                    fp=int(sum(mask&fp)), fn=int(sum(mask&fn))))
    write_csv(OUT/'per_trial.csv', rows); write_csv(OUT/'sequence_quarters.csv', positions)
    # Select descriptive extremes separately per initialization, not new training decisions.
    examples=[]
    for variant in ['B','C']:
        subset=[r for r in rows if r['variant']==variant]
        for criterion in ['high_conf_fp','longest_fn']:
            row=max(subset,key=lambda r:(r[criterion],r['episode_id']))
            eid=row['episode_id']; ep=adapters.load_htt(episodes[eid]['path'])
            t,y,p=sequences[(variant,row['fold'],row['seed'],eid)]
            err=(y==0)&(p>=.9) if criterion=='high_conf_fp' else (y==2)&(p<.5)
            candidates=np.flatnonzero(err)
            ts=[int(candidates[0]),int(candidates[len(candidates)//2]),int(candidates[-1])]
            fig,axes=plt.subplots(2,3,figsize=(12,7))
            ax=axes[0,0]; ax.plot(t,p); ax.axhline(.5,color='k',ls='--')
            ax.fill_between(t,0,1,where=y==2,alpha=.12,color='r',label='gross')
            ax.fill_between(t,0,1,where=y==1,alpha=.15,color='orange',label='incipient')
            ax.set(xlabel='Frame',ylabel='P(gross)',ylim=(-.02,1.02)); ax.legend()
            axes[0,1].imshow(ep.reference.astype(np.uint8)); axes[0,1].set_title('Provided reference')
            axes[0,2].hist(p[y==0],bins=20,alpha=.6,label='static')
            axes[0,2].hist(p[y==2],bins=20,alpha=.6,label='gross'); axes[0,2].legend()
            for ax,ti in zip(axes[1],ts):
                ax.imshow(ep.images[ti]); ax.set_title(f't={ti}; stage={y[ti]}; p={p[ti]:.4f}'); ax.axis('off')
            fig.suptitle(f'{variant} {row["fold"]} {row["seed"]}: {eid}\n{criterion}')
            fig.tight_layout(); name=f'{variant}_{criterion}.png'; fig.savefig(OUT/name,dpi=150); plt.close(fig)
            examples.append({**row,'criterion':criterion,'frames':ts,'figure':name})
    # Limited deterministic image audit: endpoints and midpoint per trial, no full re-audit.
    stats=[]
    for eid in sorted({r['episode_id'] for r in rows}):
        ep=adapters.load_htt(episodes[eid]['path'])
        indices=sorted(set([0,len(ep.images)//2,len(ep.images)-1]))
        sample=ep.images[indices].astype(float)
        diff=sample-ep.reference
        h,w=sample.shape[1:3]
        stats.append(dict(episode_id=eid, height=h,width=w,
            sampled_indices=indices, intensity=float(sample.mean()),
            reference_intensity=float(ep.reference.mean()),
            mean_abs_reference_difference=float(np.abs(diff).mean()),
            difference_clipped_fraction=float(np.mean(np.abs(diff)>127.5)),
            legacy_crop_height=int(h/(4/3)) if h/w != 4/3 else h,
            static_frames=int(sum(ep.labels==0))))
    (OUT/'image_audit.json').write_text(json.dumps(stats,indent=2))
    assoc=[]
    for variant in ['B','C']:
        for key in ['intensity','reference_intensity','mean_abs_reference_difference','difference_clipped_fraction','static_frames']:
            xs=[]; ys=[]
            for s in stats:
                values=[r['static_fpr'] for r in rows if r['variant']==variant and r['episode_id']==s['episode_id'] and r['static_fpr'] is not None]
                if values: xs.append(s[key]); ys.append(np.mean(values))
            corr=float(np.corrcoef(xs,ys)[0,1]) if np.std(xs)>0 and np.std(ys)>0 else None
            assoc.append(dict(variant=variant,covariate=key,pearson=corr,trials=len(xs)))
    write_csv(OUT/'descriptive_associations.csv',assoc)
    (OUT/'examples.json').write_text(json.dumps(examples,indent=2))
    geometry=dict(triggered=True, reason='Existing square input center crop then anisotropic resize',
        audited_trials=len(stats), shapes=sorted({(s['height'],s['width']) for s in stats}),
        correction='aspect-preserving fit and centered neutral127/255 pad; one geometry operation',
        caveat='Geometry/FOV hypothesis, not causal attribution; no validation-based tuning')
    (OUT/'geometry_trigger.json').write_text(json.dumps(geometry,indent=2))
    text=['# 第三轮残余错误诊断','',
          '所有统计来自既有 B/C validation 预测，0.5 阈值；同一试次跨种子/折重复，不作为独立样本。',
          '接触阶段没有独立真值；序列位置、图像差异只是可观察代理，不证明因果。',
          '高置信误报与连续漏报分别选出各组的极端案例，供诊断，不代表总体发生率。',
          '逐试次统计区分告警启动次数与处于告警的 static 帧数；大量连续误报可能属于一个持续告警。','',
          '|组|类型|试次|FP|FN|最长连续FN|','|---|---|---|---:|---:|---:|']
    for r in examples: text.append(f'|{r["variant"]}|{r["criterion"]}|{r["episode_id"]}|{r["fp"]}|{r["fn"]}|{r["longest_fn"]}|')
    text += ['', '图像审计每个试次只抽取首/中/尾三帧；相关系数为未控制 probe/试次混杂的描述性统计，不能据此确定根因。',
        '参考帧亮度、背景差异、稳定负例帧数的关联见 descriptive_associations.csv；完整序列位置计数见 sequence_quarters.csv。',
        '上游几何实现对方形图先裁去垂直视野，再拉伸成 320×240；触发预先固定的单一几何消融。保留原链用于主编码器对照，消融单独报告。',
        '适配后的改善不能证明原训练数据是唯一根因，尚需在统一几何下检验，并最终进行无标记 GSmini 独立实物验证。']
    (OUT/'REPORT_ZH.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({'trials':len(stats),'trial_run_rows':len(rows),'examples':len(examples),'output':str(OUT)}))


if __name__ == '__main__': main()
