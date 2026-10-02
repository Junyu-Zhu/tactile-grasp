#!/usr/bin/env python3
"""Render clarified Chinese reports from immutable Round-5 future evaluation JSON."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def render_report(payload):
    lines=[f"# Round-5 future {payload['domain']} 评价","",f"- formal: `{payload['formal']}`",f"- evaluation protocol SHA256: `{payload['evaluation_protocol_sha256']}`","","所有数值均来自固定协议；共享数据上的 seed 标准差只描述初始化波动。"]
    if payload["domain"]=="source":lines += ["","Source 没有独立 calibration 角色，因此只报告固定 0.5 与阈值无关指标，不提供调阈值后的部署结论。future-any-slip 帧标签不足以验证独立事件 onset，因此不报告 source 事件提前量。"]
    else:
        event_sample=payload["results"][0]["validation_events"]
        fixed_gates=[r for r in payload["results"] if r.get("seed") is not None and r["method"]=="raw_mul_pslip_fixed" and r["operating_point"]=="fixed_0.5"]
        zero_alarm=sum(r["validation_observed_no_alarm"] for r in fixed_gates)
        lines += ["",f"HTT 任务为当前 static 端点预测 `(t,t+{payload['horizon']}]` 内首次 gross。晚报只是在因果 `t>=13` 全轨迹上的诊断；不属于主 static 资格帧指标。没有完整前窗的 onset 记作 censored。",f"事件覆盖只有 {event_sample['events_eligible']}/{event_sample['events_total']} 个 onset；其余 {event_sample['events_censored']} 个因缺完整前窗而 censored。命中时 lead={payload['horizon']} 恰好位于 H={payload['horizon']} 标签窗口边界，不能解释成独立证明了稳定的 {payload['horizon']} 帧提前能力。",f"固定 0.5 的乘法门控在 {zero_alarm}/{len(fixed_gates)} 个 learned run 上 validation 零告警；这是单类决策，不是检测改善。"]
    selected=[r for r in payload["results"] if r["operating_point"] in (("fixed_0.5",) if payload["domain"]=="source" else ("max_ba",))]
    lines += ["",f"下表工作点：`{'fixed_0.5' if payload['domain']=='source' else 'max_ba（仅由 calibration 选择）'}`。","","| variant | seed | method | operating point | H | BA | FPR | recall | AP | Brier |","|---|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in selected:
        v=r["validation"];lines.append(f"| {r['variant']} | {r.get('seed') or '-'} | {r['method']} | {r['operating_point']} | {r.get('horizon',payload.get('horizon','-'))} | {v['balanced_accuracy']:.4f} | {v['fpr']:.4f} | {v['recall']:.4f} | {v['average_precision']:.4f} | {v['brier']:.4f} |")
    if payload["domain"]=="htt":
        low=[a for a in payload["seed_aggregates"] if a["method"]=="raw" and a["operating_point"] in ("fpr_0.01","fpr_0.05","fpr_0.10")]
        lines += ["","以下阈值只在 calibration 满足目标 FPR；表中是应用到 validation 后三个 seed 的描述性均值。","","| variant | calibration operating point | validation BA | validation FPR | validation recall |","|---|---|---:|---:|---:|"]
        for a in sorted(low,key=lambda x:(x["variant"],x["operating_point"])):
            v=a["validation"];lines.append(f"| {a['variant']} | {a['operating_point']} | {v['balanced_accuracy']['mean']:.4f} | {v['fpr']['mean']:.4f} | {v['recall']['mean']:.4f} |")
    lines += ["","每个 operating point 的 JSON 同时记录 calibration/validation 是否实际零告警，以及概率标准差、唯一值数和决策类别数。单类决策是需要明确报告的模型结果，不能静默计作成功。","","旧 GT force-delta future 实验使用了不同输入条件，本轮未将其数值并入部署一致主表；仅可在总报告中作为有明确输入差异的历史参照。","","证据限制：validation 参与 checkpoint 选择，属于开发证据；calibration 的 FPR 约束不保证 validation FPR；乘法门控是固定分数组合，不能证明 raw future 概率已校准。",""]
    return "\n".join(lines)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--evaluation",type=Path,required=True);parser.add_argument("--output",type=Path);args=parser.parse_args()
    payload=json.loads(args.evaluation.resolve().read_text())
    if payload.get("status")!="complete":raise SystemExit("evaluation is not complete")
    output=(args.output or args.evaluation.with_name("REPORT_ZH.md")).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_name(output.name+f".tmp.{os.getpid()}");temporary.write_text(render_report(payload));os.replace(temporary,output)


if __name__=="__main__":main()
