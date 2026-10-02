#!/usr/bin/env python3
"""Rebuild a factual run manifest without starting or modifying training."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow')
OUT=ROOT/'round4_comprehensive'
SEEDS=[20260914,20260915,20260916]
rows=[]
for enc in ['mae_reused','dino','ijepa','mae_letterbox']:
    for fold in range(1,5):
        for seed in SEEDS:
            path=(ROOT/f'round3_mae_slip_adaptation/runs/B/fold_p{fold}/seed_{seed}' if enc=='mae_reused'
                  else OUT/f'encoders/runs/{enc}/htt_leave_p{fold}/{seed}')
            summary=path/'training_summary.json'
            data=json.loads(summary.read_text()) if summary.exists() else {}
            rows.append(dict(package='B' if enc!='mae_letterbox' else 'F',model=enc,fold=fold,seed=seed,
                reused=enc=='mae_reused',status=data.get('status','pending'),directory=str(path),
                checkpoint=str(path/'best.pth'),summary=str(summary)))
for variant in ['C','D']:
    for seed in SEEDS:
        # Resolve current runner artifacts via aggregate, retain exact recorded path.
        summary=OUT/'normalflow/summary.json'
        data=json.loads(summary.read_text()) if summary.exists() else {}
        run=next((r for r in data.get('runs',[]) if r['variant']==variant and r['seed']==seed),{})
        rows.append(dict(package='D',model=variant,seed=seed,reused=False,status=run.get('status','pending'),
            directory=str(OUT/f'normalflow/runs/{variant}/{seed}'),
            checkpoint=str(OUT/f'normalflow/runs/{variant}/{seed}/best.pt'),summary=str(summary)))
for model in ['mlp','gru']:
    for seed in SEEDS:
        path=OUT/f'future/runs/{model}/seed_{seed}'
        candidates=[path/'training_summary.json',path/'summary.json']
        summary=next((p for p in candidates if p.exists()),candidates[0])
        data=json.loads(summary.read_text()) if summary.exists() else {}
        rows.append(dict(package='E',model=model,seed=seed,reused=False,status=data.get('status','pending'),
            directory=str(path),checkpoint=str(path/'best.pth'),summary=str(summary)))
result=dict(goal_status='active_until_independent_review_and_verified_delivery',
    new_neural_planned=48,reused_mae_runs=12,conditions={'E':'train-only H8 support; complete-history gate required','F':'geometry audit triggered'},
    runs=rows)
(OUT/'RUN_INVENTORY.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'rows':len(rows),'statuses':{s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})}}))
