#!/usr/bin/env python3
import json, shlex
from datetime import datetime
from pathlib import Path

repo = Path('/home/zjy/document/tactile-grasp')
ctx_path = Path('/vla1/zjy/sparsh_runs/force_slip_phase1') / (repo/'sparsh-force-slip/phase1_run_id.txt').read_text().strip() / 'phase1_context.json'
ctx = json.load(open(ctx_path))
run_id = ctx['run_id']
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
derived = ctx['derived_root']
force_train = ctx['force_train_datasets']
force_val = ctx['force_val_datasets']
slip_train = ctx['slip_train_datasets']
slip_val = ctx['slip_val_datasets']
runbook_dir = repo/'sparsh-force-slip/runbooks'
logs_dir = repo/'sparsh-force-slip/logs'/run_id
reports_dir = repo/'sparsh-force-slip/reports/phase1'
runbook_dir.mkdir(parents=True, exist_ok=True)
logs_dir.mkdir(parents=True, exist_ok=True)
reports_dir.mkdir(parents=True, exist_ok=True)

def hydra_list(items):
    return '[' + ','.join(json.dumps(x) for x in items) + ']'

def shell_quote(s):
    return shlex.quote(str(s))

def make_command(gpu, encoder, task, cfg, ckpt, train_list, val_list, extra_overrides=None):
    extra_overrides = extra_overrides or []
    run_name = f"{run_id}_{encoder}_{task}_gsmini_{timestamp}"
    parts = [
        'cd /home/zjy/document/sparsh',
        'source /home/zjy/miniconda3/etc/profile.d/conda.sh',
        'conda activate sparsh',
        f'export CUDA_VISIBLE_DEVICES={gpu}',
        'export WANDB_MODE=online',
        f'export WANDB_NAME={shell_quote(run_name)}',
        'export PYTHONPATH=.',
        'python train_task.py',
        f'--config-name={cfg}',
        'paths=zjy_4090',
        'wandb=tactile_grasp',
        '+trainer.devices=1',
        'ssl_model_size=base',
        f'ssl_name={encoder}',
        f'experiment_name={shell_quote(run_name)}',
        f'task.checkpoint_encoder={ckpt}',
        f'data.dataset.config.path_dataset={shell_quote(derived)}',
        shell_quote('data.dataset.config.list_datasets=' + hydra_list(train_list)),
        shell_quote('data.dataset.config.list_datasets_test=' + hydra_list(val_list)),
        shell_quote('test.data.dataset_name=' + hydra_list(val_list)),
    ] + extra_overrides
    return run_name, ' && '.join(parts[:7]) + ' && ' + ' '.join(parts[7:])

run_specs = {
    'dinov2_force': {
        'gpu': 0,
        'encoder': 'dinov2',
        'task': 'force',
        'config': 'experiment/downstream_task/force/gelsight_dinov2',
        'checkpoint': '/vla1/zjy/sparsh_models/sparsh-dinov2-base/dinov2_vitbase.ckpt',
        'train_list': force_train,
        'val_list': force_val,
        'extra_overrides': [],
    },
    'dinov2_slip': {
        'gpu': 1,
        'encoder': 'dinov2',
        'task': 'slip',
        'config': 'experiment/downstream_task/slip/gelsight_dinov2',
        'checkpoint': '/vla1/zjy/sparsh_models/sparsh-dinov2-base/dinov2_vitbase.ckpt',
        'train_list': slip_train,
        'val_list': slip_val,
        'extra_overrides': [shell_quote('data.dataset.config.max_delta_forceXYZ=[0.80,0.80,0.40]')],
    },
    'mae_force': {
        'gpu': 2,
        'encoder': 'mae',
        'task': 'force',
        'config': 'experiment/downstream_task/force/gelsight_mae',
        'checkpoint': '/vla1/zjy/sparsh_models/sparsh-mae-base/mae_vitbase.ckpt',
        'train_list': force_train,
        'val_list': force_val,
        'extra_overrides': [],
    },
    'mae_slip': {
        'gpu': 3,
        'encoder': 'mae',
        'task': 'slip',
        'config': 'experiment/downstream_task/slip/gelsight_mae',
        'checkpoint': '/vla1/zjy/sparsh_models/sparsh-mae-base/mae_vitbase.ckpt',
        'train_list': slip_train,
        'val_list': slip_val,
        'extra_overrides': [shell_quote('data.dataset.config.max_delta_forceXYZ=[0.80,0.80,0.40]')],
    },
}

runs = {
    name: make_command(
        spec['gpu'],
        spec['encoder'],
        spec['task'],
        spec['config'],
        spec['checkpoint'],
        spec['train_list'],
        spec['val_list'],
        spec['extra_overrides'],
    )
    for name, spec in run_specs.items()
}

for window, (run_name, cmd) in runs.items():
    script = runbook_dir / f'run_{window}.sh'
    script.write_text(f'''#!/usr/bin/env bash
set -euo pipefail
# Generated for {run_id}
# W&B run: {run_name}
mkdir -p {shell_quote(logs_dir)}
{{
  echo "START $(date -Is) {window} {run_name}"
  cat <<'COMMAND_EOF'
{cmd}
COMMAND_EOF
  {cmd}
  echo "END $(date -Is) {window} {run_name}"
}} 2>&1 | tee {shell_quote(logs_dir / (window + '.log'))}
''', encoding='utf-8')
    script.chmod(0o755)

launcher = runbook_dir / 'launch_phase1_tmux.sh'
launcher.write_text(f'''#!/usr/bin/env bash
set -euo pipefail
SESSION="force-slip-phase1"
cd /home/zjy/document/tactile-grasp
git status --short --branch
if ! git branch --show-current | grep -qx 'sparsh-force-slip'; then
  echo "ERROR: expected tactile-grasp branch sparsh-force-slip" >&2
  exit 1
fi
source /home/zjy/miniconda3/etc/profile.d/conda.sh
conda activate sparsh
wandb login --verify
for ckpt in /vla1/zjy/sparsh_models/sparsh-dinov2-base/dinov2_vitbase.ckpt /vla1/zjy/sparsh_models/sparsh-mae-base/mae_vitbase.ckpt; do
  test -f "$ckpt" || {{ echo "Missing checkpoint $ckpt" >&2; exit 1; }}
done
test -d {shell_quote(derived)} || {{ echo "Missing derived dataset {derived}" >&2; exit 1; }}
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "ERROR: tmux session $SESSION already exists; attach with: tmux attach -t $SESSION" >&2
  exit 1
fi
tmux new-session -d -s "$SESSION" -n dinov2_force 'bash {runbook_dir / 'run_dinov2_force.sh'}'
tmux new-window -t "$SESSION" -n dinov2_slip 'bash {runbook_dir / 'run_dinov2_slip.sh'}'
tmux new-window -t "$SESSION" -n mae_force 'bash {runbook_dir / 'run_mae_force.sh'}'
tmux new-window -t "$SESSION" -n mae_slip 'bash {runbook_dir / 'run_mae_slip.sh'}'
tmux list-windows -t "$SESSION"
''', encoding='utf-8')
launcher.chmod(0o755)

record = {
    'run_id': run_id,
    'generated_at': datetime.now().isoformat(),
    'derived_root': derived,
    'session': 'force-slip-phase1',
    'logs_dir': str(logs_dir),
    'runs': {
        k: {
            'gpu': run_specs[k]['gpu'],
            'encoder': run_specs[k]['encoder'],
            'task': run_specs[k]['task'],
            'config': run_specs[k]['config'],
            'checkpoint': run_specs[k]['checkpoint'],
            'wandb_name': v[0],
            'command': v[1],
            'script': str(runbook_dir / f'run_{k}.sh'),
        }
        for k, v in runs.items()
    },
    'launcher': str(launcher),
    'force_train_datasets': force_train,
    'force_val_datasets': force_val,
    'slip_train_datasets': slip_train,
    'slip_val_datasets': slip_val,
}
(repo/'sparsh-force-slip/reports/phase1/training_commands.json').write_text(json.dumps(record, indent=2, ensure_ascii=False)+"\n", encoding='utf-8')
md = ['# Phase 1 Downstream Training Commands', '', f'- run_id: `{run_id}`', f'- derived_root: `{derived}`', '- tmux session: `force-slip-phase1`', f'- logs: `{logs_dir}`', '', '## Launch', '', f'```bash\n{launcher}\n```', '']
for k, v in record['runs'].items():
    md += [
        f'## {k}',
        '',
        f'- gpu: `{v["gpu"]}`',
        f'- encoder/task: `{v["encoder"]}` / `{v["task"]}`',
        f'- wandb: `{v["wandb_name"]}`',
        f'- script: `{v["script"]}`',
        '',
        '```bash',
        v['command'],
        '```',
        '',
    ]
(repo/'sparsh-force-slip/reports/phase1/training_commands.md').write_text('\n'.join(md), encoding='utf-8')
print(json.dumps(record, indent=2, ensure_ascii=False))
