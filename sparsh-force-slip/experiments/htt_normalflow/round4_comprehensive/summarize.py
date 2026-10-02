#!/usr/bin/env python3
"""Generate the Chinese synthesis only after independent computation acceptance."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
OUT=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round4_comprehensive')


def main():
    audit=json.loads((OUT/'integration/independent_audit.json').read_text())
    assert audit['status']=='pass', 'Cannot synthesize accepted results before independent audit'
    runs=[json.loads(p.read_text()) for p in sorted((OUT/'alarms/runs').glob('*/*/*/metrics.json'))]
    assert len(runs)==48
    models=['mae','dino','ijepa','mae_letterbox']
    def values(model,op,mode,key):
        return [r['operating_points'][op][mode]['validation'][key] for r in runs if r['model']==model]
    def mean(model,op,mode,key):return float(np.mean(values(model,op,mode,key)))
    nf=json.loads((OUT/'normalflow/summary.json').read_text())
    lines=['# 第四轮综合实验结论','',
        '本轮完成 48 次新增神经网络训练：编码器适配 24 次、NormalFlow 6 次、future 6 次、单一几何消融 12 次；复用第三轮 MAE 新头 12 次。条件任务均有明确触发证据。',
        '这是冻结编码器上的开发实验。四折存在重叠，validation 用于早停与最终开发评估；下列均值为描述性汇总，不是独立盲测或实物成功率。',
        '参数冻结、恢复和指标独立重算已经验收；最终本地文件一致性另见 LOCAL_SYNC_VERIFICATION.json。','',
        '## 检测头与编码器对比','',
        '每种模型都先计算四折×三个种子的指标，再取描述性平均。前三行保持原输入链，最后一行为独立几何消融，不能把其收益归因于更换编码器。','',
        '|模型|BA@0.5|macro-F1|static误报率|gross召回|AP|正类自然占比|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for model in models:
        keys=['balanced_accuracy','macro_f1','static_fpr','gross_recall','average_precision','positive_prevalence']
        lines.append('|'+model+'|'+'|'.join(f'{mean(model,"fixed_0.5","raw",k):.4f}' for k in keys)+'|')
    scores={m:mean(m,'fixed_0.5','raw','balanced_accuracy') for m in models[:3]}
    best=max(scores,key=scores.get)
    base={(r['fold'],r['seed']):r for r in runs if r['model']=='mae'}
    lines += ['',f'在固定 0.5 工作点下，前三种编码器中描述性平均 BA 最高的是 {best}；这只反映本开发协议，不能据此宣称实物泛化更好。',
        'DINO/I-JEPA 对照只验证新 slip 分支。原 force 权重没有更新，但不能据此认为把原 force 头接到新编码器上后输出仍然有效；本轮 future 继续固定原 MAE/force 路径，没有进行这种替换。','',
        '|对照MAE|配对BA平均差|12组中BA提高数|配对static FPR平均差|','|---|---:|---:|---:|']
    for model in models[1:]:
        delta=[];fpr=[]
        for r in runs:
            if r['model']!=model:continue
            b=base[(r['fold'],r['seed'])]['operating_points']['fixed_0.5']['raw']['validation']
            c=r['operating_points']['fixed_0.5']['raw']['validation']
            delta.append(c['balanced_accuracy']-b['balanced_accuracy']);fpr.append(c['static_fpr']-b['static_fpr'])
        lines.append(f'|{model}|{np.mean(delta):+.4f}|{sum(x>0 for x in delta)}/12|{np.mean(fpr):+.4f}|')
    lines += ['', '逐折/逐种子完整指标、样本级预测及配对差异见 alarms/run_metrics.csv、alarms/summary.json 和各运行 metrics.json；未挑选最好种子。置信区间按完整试次/泄漏组重采样，不能把重叠折或相邻帧当作独立样本。',
        '', '## 低误报告警','',
        '阈值和连续确认/滞回参数只在 calibration 选择。下面是验证集的实际表现，5% 仅是校准目标，不是验证集保证。','',
        '|模型|规则|实际static FPR|gross召回|事件召回|每试次误告警启动次数|验证集无主任务告警运行数|',
        '|---|---|---:|---:|---:|---:|---:|']
    for model in models:
        for mode in ['raw','sequential']:
            vals=[mean(model,'fpr_0.05',mode,k) for k in ['static_fpr','gross_recall','gross_event_recall','false_alarm_starts_per_trial']]
            none=sum(values(model,'fpr_0.05',mode,'observed_no_primary_alarm'))
            lines.append(f'|{model}|{mode}|'+'|'.join(f'{x:.4f}' for x in vals)+f'|{none}/12|')
    lines += ['', '固定 0.5、最大 BA、1%/5%/10% 所有工作点均已交付，不能只展示最有利工作点。连续误报可能只产生一次启动但持续很多帧，因此逐试次同时报告启动次数和告警持续帧数。检测延迟仅对命中事件统计，并与漏报事件数一起解读。FPR 约束工作点的平均检测延迟约 11–23 帧；事件召回 100% 只表示较长 gross 段内曾经告警，不表示及时检测或提前预警。无告警不能算作检测改善。',
        '', '## NormalFlow 状态预测','',
        '|方法|H=1 MSE|H=3 MSE|H=5 MSE|','|---|---:|---:|---:|']
    for name in ['persistence','linear']:
        key=name if name in nf['baseline'] else next(k for k in nf['baseline'] if 'linear' in k)
        metrics=nf['baseline'][key]['metrics']['overall']
        lines.append('|'+name+'|'+'|'.join(f'{metrics[str(h)]:.7f}' for h in [1,3,5])+'|')
    for name in ['C','D']:
        lines.append('|'+name+'|'+'|'.join(f'{nf["mean_by_variant"][name][str(h)]:.7f}' for h in [1,3,5])+'|')
    lines += ['', 'C 为轻量 GRU，D 为相同结构加入相对运动辅助监督。C 在三个预测窗口均优于保持不变，但只在 H=3 略优于线性预测；D 在三个窗口的平均 MSE 均高于 C，尽管仍低于保持不变。预测动态变化幅度偏保守，不能把保留当前状态造成的低 MSE 解释为学到了充分的抓取动力学。',
        '各方法使用共同有效的四步历史和多窗口样本范围，不能直接与前轮按单个窗口取全部可用帧的 MSE 作升级归因。线性基线仅 9,216 个拟合参数，C/D 为 642,048/644,370；模块计时及 CPU/GPU 条件见 normalflow/BASELINE_EFFICIENCY_ZH.md，不能直接跨设备作速度排名。',
        '验证集仅有两个物体，物体级区间非常有限；运动只是辅助监督，不是 slip 标签，也不是预测时的已知未来输入。此轮直接联合预测多个窗口，没有将直接多步预测冒充递归 rollout 验证。',
        '', '## Future 与预警','',
        '在预先固定的 leave-p1 开发折，完整历史窗口的 train-only 支持审计选择了 8 帧 first-gross 目标：84 个正例帧/31 个事件试次，332 个负例帧/24 个试次。validation 只有 77 个合格帧、16 个正例、6 个正事件试次；calibration 只有 56 个负帧，FPR 分辨率约 1.79%。',
        'MLP 与 GRU 都使用同样可获得的四步特征、预测力及其历史差分，重新初始化训练；并非直接延续历史 future 权重，也不能单独量化“只修正输入”带来的收益。底层 Sparsh 两帧输入沿用既定启动填充，四步特征完整不等于最早阶段的全部原图历史均未填充。',
        'MLP/GRU 的平均 AP 分别约 0.4652/0.5832，当前 slip 概率基线约 0.1860，支持风险排序改善。但最大 BA 阈值下平均 FPR 仍约 78.7%/73.2%；严格低误报工作点多为零告警或极低召回。高事件召回与约 7–8 帧的命中提前量伴随高误报，不能单独称为可靠预警。',
        '补充启动段审计发现：validation 一半正例来自底层历史仍有起点填充的 t=3–7 帧；在 calibration 最大 BA 阈值下，这一段六个运行的误报率均为 100%。排除该段后，train-only H8 仍满足最低事件支持，但验证表现存在明显分层差异；padding、序列位置与试次构成目前无法分离。这进一步限制了 AP 提升的解释，不能称为稳健预警。该审计没有重调阈值或重新选择任务。',
        'HTT 阶段标签包含回溯生成的成分；这是标签定义下的短时风险任务，没有独立证据证明物理上提前预见滑移。完整校准、Brier、逐试次漏报/晚报和所有种子见 future/evaluation/REPORT_ZH.md。',
        '', '## 误差诊断与几何消融的边界','',
        '第三轮 B/C 仍有高置信 static 误报与连续 gross 漏报。诊断交付具体试次、错误帧、参考图像和概率曲线，以及按 probe/序列位置/稳定负例覆盖的统计。图像强度和背景差异的相关性属于描述性观察，没有证明原训练数据是唯一原因。',
        '旧图像链会对方形输入先裁剪再非等比缩放，满足预先定义的几何触发条件。原图本身存在黑边，裁剪并不必然错误；新方案保留宽高比及原视野，改变了几何/FOV这一项处理。其实际收益或损失应以上面的配对结果为准，不把它称为已证明的根因修复。',
        '', '## 运行代价与世界模型表述','',
        '|模型|端到端中位延迟ms|p90 ms|编码器参数|slip参数|峰值分配MiB|',
        '|---|---:|---:|---:|---:|---:|']
    for model in models:
        b=json.loads((OUT/f'benchmark/{model}.json').read_text())
        lines.append(f'|{model}|{b["median_ms"]:.2f}|{b["p90_ms"]:.2f}|{b["encoder_parameters"]:,}|{b["slip_parameters"]:,}|{b["peak_allocated_bytes"]/1024**2:.1f}|')
    lines += ['', '计时包含内存中的原图预处理、编码器和检测头，排除相机采集、磁盘解码与机器人控制；记录了共享 GPU 条件和设备映射。冻结并不会减少编码器推理量，约 8600 万参数的编码器不能因为下游模块较小就被整体称为轻量模型。',
        '目前较稳妥的表述是“冻结触觉表征上的轻量状态预测与风险预警模块”。本轮没有证明动作条件化动力学、反事实动作预测、递归展开可靠性或闭环控制收益，不能仅凭低 MSE 或 AP 提升宣称完整世界模型有效。',
        '', '## 下一阶段','',
        '优先用独立无标记 GSmini 实物试次补足稳定接触困难负例和滑移事件，并通过外部相对运动测量建立同步真值。先固定检测和阈值校准，再检验真正稳定阶段的提前预警。是否更换编码器应同时考虑跨折一致性、低误报召回和部署代价，而不是单个最好结果。',
        'NormalFlow 的结果不支持立即扩大当前 GRU 或增加同类辅助损失；先保留线性基线，核查动态监督与真实任务的对应关系。若要推进世界模型概念，下一阶段需要动作记录、干预对照和控制收益验证。没有自动启动新一轮训练。',
        '具体采集与测试方案见 REAL_GSMINI_NEXT_STAGE.md；复现入口见 REPRODUCE.md，所有 checkpoint 精确位置见 RUN_INVENTORY.json。大型缓存/权重保留服务器，最终同步清单及字节一致性记录与报告一并交付。']
    text='\n'.join(lines)+'\n'
    (OUT/'SUMMARY_ZH.md').write_text(text);(HERE/'SUMMARY_ZH.md').write_text(text)
    fig,axes=plt.subplots(2,2,figsize=(11,8),sharey=True)
    for i,ax in enumerate(axes.flat,1):
        for j,model in enumerate(models):
            vals=[r['operating_points']['fixed_0.5']['raw']['validation']['balanced_accuracy'] for r in runs if r['model']==model and r['fold']==f'htt_leave_p{i}']
            ax.scatter(np.arange(len(vals))*.08+j-.08,vals,s=30)
            ax.plot([j-.18,j+.18],[np.mean(vals)]*2,color='black')
        ax.set_xticks(range(4),['MAE','DINO','I-JEPA','MAE geometry'])
        ax.set_title(f'Development fold {i}');ax.set_ylabel('Balanced accuracy @ 0.5');ax.grid(alpha=.2)
    fig.suptitle('All three seeds per fold; horizontal bars are means (not confidence intervals)')
    fig.tight_layout();fig.savefig(OUT/'encoder_fold_seeds.png',dpi=160);plt.close(fig)
    print(json.dumps({'status':'synthesis_written','detector_runs':len(runs)}))


if __name__=='__main__':main()
