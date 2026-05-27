#!/usr/bin/env python3
"""Phase lambda decoupled sweep finalization and best-lambda selection."""
from __future__ import annotations

import argparse, json, math, sys, subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import phase2_b_multitask as p2  # noqa: E402

REPO = Path('/home/zjy/document/tactile-grasp')
WORKSPACE = REPO / 'sparsh-force-slip'
REPORT_ROOT = WORKSPACE / 'reports/phase_lambda_decoupled'
PHASE2_ROOT = Path('/vla1/zjy/sparsh_runs/force_slip_phase2')
LAMBDA_SPECS = [('lam010',0.10),('lam025',0.25),('lam050',0.50),('lam075',0.75)]
EXISTING_LAM1_RUN_ID = 'phase3_1_decoupled_gsmini_20260516_154730'
WANDB_BASE = 'https://wandb.ai/junyuzhuzjy-zhejiang-university/sparsh-finetune-tactile-grasp/runs'


def wandb_name_for(run_id: str, lam: float | None = None) -> str:
    if lam is None or lam == 1.0:
        return run_id
    tag = f'lam{int(round(float(lam) * 100)):03d}'
    return f'{run_id}_lambda_{tag}'

def run_script_text(stamp: str, tag: str) -> str | None:
    script_dir = WORKSPACE / 'runbooks' / f'phase_lambda_decoupled_{stamp}'
    candidates = list(script_dir.glob(f'run_mae_decoupled_{tag}_gpu*.sh'))
    return candidates[0].read_text(encoding='utf-8') if candidates else None


def read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding='utf-8'))

def write_json(p: Path, data: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, default=p2.json_default), encoding='utf-8')

def fmt(v: Any, n: int=4) -> str:
    if v is None: return 'n/a'
    try:
        f=float(v)
        if math.isnan(f) or math.isinf(f): return 'n/a'
        return f'{f:.{n}f}'
    except Exception:
        return str(v)

def pct(v: Any) -> str:
    if v is None: return 'n/a'
    return f'{float(v):+.2f}%'

def run_dir(run_id: str) -> Path:
    return PHASE2_ROOT / run_id / 'mae_decoupled_multitask'

def train_config(run_id: str) -> dict[str, Any]:
    return read_json(run_dir(run_id) / 'train_config.json')

def complete(run_id: str) -> bool:
    return (run_dir(run_id) / 'training_summary.json').exists()

def select_checkpoint(run_id: str, a_eval: dict[str, Any], report_dir: Path) -> dict[str, Any]:
    hist_path = run_dir(run_id) / 'history.json'
    if not hist_path.exists():
        raise FileNotFoundError(hist_path)
    hist = read_json(hist_path)
    candidates=[]
    for rec in hist:
        val=rec.get('val')
        if not val: continue
        epoch=int(rec['epoch'])
        ckpt=run_dir(run_id)/'checkpoints'/f'epoch-{epoch:04d}.pth'
        if not ckpt.exists(): continue
        gate=p2.gate_status(a_eval, {'aggregate': val})
        candidates.append({
            'epoch': epoch,
            'global_step': rec.get('global_step'),
            'checkpoint': str(ckpt),
            'force_rmse_mean_N': val.get('force_rmse_mean_N'),
            'slip_f1': val.get('slip_f1'),
            'slip_accuracy': val.get('slip_accuracy'),
            'gate': gate,
        })
    if not candidates:
        raise RuntimeError(f'No validation candidates for {run_id}')
    feasible=[c for c in candidates if c['gate']['status'] != 'hard_fail']
    pool=feasible or candidates
    selected=sorted(pool, key=lambda c: (
        float(c.get('force_rmse_mean_N') or 1e9),
        -float(c.get('slip_f1') or 0.0),
        -float(c.get('slip_accuracy') or 0.0),
        int(c.get('epoch') or 0),
    ))[0]
    out={'policy':'exclude_hard_fail_then_min_force_rmse_then_slip_metrics','run_id':run_id,'run_dir':str(run_dir(run_id)),'checkpoint':selected['checkpoint'],'epoch':selected['epoch'],'candidate_count':len(candidates),'feasible_count':len(feasible),'gate_from_history':selected['gate'],'candidates':candidates}
    write_json(report_dir / f'checkpoint_selection_{run_id}.json', out)
    return out

def eval_run(run_id: str, lam: float, a_eval: dict[str, Any], reference: dict[str, Any], report_dir: Path, batch_size: int, num_workers: int, refresh: bool) -> dict[str, Any]:
    cache = report_dir / 'eval_cache' / f'{run_id}_val.json'
    if cache.exists() and not refresh:
        ev=read_json(cache)
    else:
        sel=select_checkpoint(run_id, a_eval, report_dir)
        ev=p2.evaluate_b_checkpoint_for_report(run_id,'mae',reference,batch_size,num_workers,decoder_variant='decoupled',checkpoint_path=sel['checkpoint'])
        ev['checkpoint_selection']=sel
        write_json(cache, ev)
    agg=ev['aggregate']
    gate=p2.gate_status(a_eval, ev)
    a=a_eval['aggregate']
    row={
        'lambda_slip': lam,
        'run_id': run_id,
        'checkpoint': ev.get('checkpoint'),
        'checkpoint_epoch': ev.get('checkpoint_epoch'),
        'wandb_name': wandb_name_for(run_id, lam),
        'wandb_url': f'{WANDB_BASE}/{wandb_name_for(run_id, lam)}',
        'force_rmse_mean_N': agg.get('force_rmse_mean_N'),
        'deltaF_pct': gate.get('force_rmse_increase_pct'),
        'slip_f1': agg.get('slip_f1'),
        'deltaSF1_pct_drop': ((a.get('slip_f1') - agg.get('slip_f1')) / a.get('slip_f1') * 100.0) if a.get('slip_f1') else None,
        'slip_accuracy': agg.get('slip_accuracy'),
        'deltaSA_pct_drop': ((a.get('slip_accuracy') - agg.get('slip_accuracy')) / a.get('slip_accuracy') * 100.0) if a.get('slip_accuracy') else None,
        'gate': gate,
        'eval': ev,
    }
    return row

def choose_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sweep=[r for r in rows if r.get('lambda_slip') != 1.0]
    feasible=[r for r in sweep if r['gate']['status'] != 'hard_fail']
    pool=feasible or sweep
    best=sorted(pool, key=lambda r: (float(r.get('force_rmse_mean_N') or 1e9), -float(r.get('slip_f1') or 0.0), -float(r.get('slip_accuracy') or 0.0), float(r['lambda_slip'])))[0]
    lam1=next((r for r in rows if r.get('lambda_slip') == 1.0), None)
    better_than_lam1=None
    if lam1:
        better_than_lam1 = bool(best['gate']['status'] != 'hard_fail' and (best['force_rmse_mean_N'] < lam1['force_rmse_mean_N'] or (abs(best['force_rmse_mean_N']-lam1['force_rmse_mean_N']) < 1e-6 and best['slip_f1'] >= lam1['slip_f1'])))
    return {'selected': best, 'feasible_count': len(feasible), 'policy': 'non-hard-fail -> min Force RMSE -> max SF1/SA', 'lambda1_reference': lam1, 'best_beats_lambda1_reference': better_than_lam1}

def render(payload: dict[str, Any]) -> str:
    lines=['# Phase Lambda Decoupled Sweep Report','',f"- generated_at: `{payload['generated_at']}`",f"- stamp: `{payload['stamp']}`",'- encoder: `mae`','- decoder_variant: `decoupled`','- raw_data_modified: `False`','- selection policy: non-hard-fail -> minimum Force RMSE -> higher Slip F1/accuracy.','', '## Current force-slip validation results','', '| lambda | run id | F RMSE | ΔF | SF1 | ΔSF1 drop | SA | ΔSA drop | gate | epoch |','|---:|---|---:|---:|---:|---:|---:|---:|---|---:|']
    for r in payload['rows']:
        lines.append(f"| {r['lambda_slip']:.2f} | `{r['run_id']}` | {fmt(r['force_rmse_mean_N'])} | {pct(r['deltaF_pct'])} | {fmt(r['slip_f1'])} | {pct(r['deltaSF1_pct_drop'])} | {fmt(r['slip_accuracy'])} | {pct(r['deltaSA_pct_drop'])} | {r['gate']['status']} | {r.get('checkpoint_epoch')} |")
    b=payload['best_lambda']['selected']
    lines += ['', '## Best lambda selection', '', f"Selected λ: `{b['lambda_slip']:.2f}`", f"- checkpoint: `{b['checkpoint']}`", f"- gate: `{b['gate']['status']}`", f"- Force RMSE: `{fmt(b['force_rmse_mean_N'])}`; Slip F1: `{fmt(b['slip_f1'])}`; Slip accuracy: `{fmt(b['slip_accuracy'])}`", f"- beats existing λ=1.0 reference: `{payload['best_lambda'].get('best_beats_lambda1_reference')}`", '', '## W&B links', '']
    for r in payload['rows']:
        if r.get('lambda_slip') != 1.0:
            lines.append(f"- λ={r['lambda_slip']:.2f}: [{r.get('wandb_name')}]({r.get('wandb_url')})")
    lines += ['', '## Training commands', '']
    for tag, cmd in payload.get('training_commands', {}).items():
        lines.append(f"### {tag}\n\n```bash\n{cmd.strip()}\n```")
    lines += ['', '## Interpretation', '', payload['interpretation']]
    return '\n'.join(lines)+'\n'

def command_phase_a(args: argparse.Namespace) -> None:
    report_dir=REPORT_ROOT / args.stamp
    report_dir.mkdir(parents=True, exist_ok=True)
    incomplete=[]
    run_ids=[]
    for tag, lam in LAMBDA_SPECS:
        rid=f'phase_lambda_decoupled_mae_{tag}_{args.stamp}'
        run_ids.append(rid)
        if not complete(rid): incomplete.append(rid)
    if incomplete and not args.allow_incomplete:
        raise SystemExit('Incomplete training runs: ' + ', '.join(incomplete))
    reference_path=report_dir/'force_ratio_reference.json'
    if reference_path.exists() and not args.refresh:
        reference=read_json(reference_path)
    else:
        reference=p2.collect_force_ratio_reference(slip_horizon=0); write_json(reference_path, reference)
    a_cache=report_dir/'eval_cache/a_mae_allsource_val.json'
    if a_cache.exists() and not args.refresh:
        a_eval=read_json(a_cache)
    else:
        a_eval=p2.evaluate_a_models('mae', reference, args.batch_size, args.num_workers); write_json(a_cache, a_eval)
    rows=[]
    for tag, lam in LAMBDA_SPECS:
        rid=f'phase_lambda_decoupled_mae_{tag}_{args.stamp}'
        if complete(rid): rows.append(eval_run(rid, lam, a_eval, reference, report_dir, args.batch_size, args.num_workers, args.refresh))
    if args.include_lambda1:
        rows.append(eval_run(EXISTING_LAM1_RUN_ID, 1.0, a_eval, reference, report_dir, args.batch_size, args.num_workers, args.refresh))
    best=choose_best(rows)
    selected=best['selected']
    interp=(f"Among the completed small-lambda sweep runs, λ={selected['lambda_slip']:.2f} is selected by the predefined policy. "
            f"It has Force RMSE {fmt(selected['force_rmse_mean_N'])}, Slip F1 {fmt(selected['slip_f1'])}, and gate {selected['gate']['status']}. "
            "This checkpoint should be used as the Stage-I interface for the next future causal-input ablation unless a later audit finds a stronger constraint violation.")
    launch_manifest_path = report_dir / 'launch_manifest.json'
    launch_manifest = read_json(launch_manifest_path) if launch_manifest_path.exists() else {}
    training_commands = {}
    for tag, _lam in LAMBDA_SPECS:
        txt = run_script_text(args.stamp, tag)
        if txt:
            training_commands[tag] = txt
    payload={'generated_at':datetime.now().isoformat(timespec='seconds'),'stamp':args.stamp,'phase':'phase_lambda_decoupled','repo':str(REPO),'branch':subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip(),'head':subprocess.check_output(['git','rev-parse','--short','HEAD'],cwd=REPO,text=True).strip(),'launch_manifest':launch_manifest,'training_commands':training_commands,'a_eval':a_eval,'rows':rows,'best_lambda':best,'interpretation':interp,'incomplete_runs':incomplete,'json_path':str(report_dir/'phase_lambda_decoupled_report.json'),'md_path':str(report_dir/'phase_lambda_decoupled_report.md')}
    write_json(report_dir/'phase_lambda_decoupled_report.json', payload)
    (report_dir/'phase_lambda_decoupled_report.md').write_text(render(payload),encoding='utf-8')
    current=REPORT_ROOT/'current_phase_lambda_decoupled.md'
    current.write_text(f"# Current Phase Lambda Decoupled\n\n- stamp: `{args.stamp}`\n- report: `{report_dir/'phase_lambda_decoupled_report.md'}`\n- best_lambda: `{selected['lambda_slip']:.2f}`\n- checkpoint: `{selected['checkpoint']}`\n",encoding='utf-8')
    print(json.dumps({'report':str(report_dir/'phase_lambda_decoupled_report.md'),'best_lambda':selected['lambda_slip'],'checkpoint':selected['checkpoint']},indent=2), flush=True)

def build_parser() -> argparse.ArgumentParser:
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('phase-a')
    p.add_argument('--stamp',required=True)
    p.add_argument('--batch-size',type=int,default=100)
    p.add_argument('--num-workers',type=int,default=2)
    p.add_argument('--refresh',action='store_true')
    p.add_argument('--allow-incomplete',action='store_true')
    p.add_argument('--include-lambda1',action='store_true',default=True)
    p.set_defaults(func=command_phase_a)
    return ap

def main() -> None:
    args=build_parser().parse_args(); args.func(args)
if __name__=='__main__': main()
