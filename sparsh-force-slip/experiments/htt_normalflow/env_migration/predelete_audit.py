"""Read-only gate for retiring the two explicitly authorized old environments."""
import json
import os
import subprocess
from pathlib import Path

ROOT = Path('/home/zjy/miniconda3/envs')
OLD = [ROOT / 'sparsh', ROOT / 'force-slip-r2']
NEW = ROOT / 'sparsh-rtx6000d'
OUT = Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/environment_migration/sparsh-rtx6000d')


def main():
    active = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            exe = os.readlink(proc / 'exe')
            command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            mappings = (proc / 'maps').read_text()
        except (PermissionError, FileNotFoundError, ProcessLookupError, OSError):
            continue
        for prefix in OLD:
            needle = str(prefix) + '/'
            if needle in exe or needle in command or needle in mappings:
                active.append({'pid': int(proc.name), 'environment': str(prefix), 'executable': exe})
    dangling_dependencies = []
    for path in NEW.rglob('*'):
        if path.is_symlink():
            target = str(path.resolve())
            if any(target.startswith(str(old) + '/') for old in OLD):
                dangling_dependencies.append(str(path))
    backups = ['sparsh_conda_explicit.txt', 'sparsh_pip_freeze.txt',
               'force_slip_r2_conda_explicit.txt', 'force_slip_r2_pip_freeze.txt']
    missing = [name for name in backups if not (OUT / name).is_file() or not (OUT / name).stat().st_size]
    check = subprocess.run([str(NEW / 'bin/python'), '-m', 'pip', 'check'], capture_output=True, text=True)
    result = {'active_old_environment_processes': active,
              'new_environment_symlinks_to_old_environments': dangling_dependencies,
              'missing_backups': missing, 'pip_check_exit': check.returncode,
              'pip_check': check.stdout + check.stderr,
              'status': 'pass' if not (active or dangling_dependencies or missing or check.returncode) else 'fail',
              'scope': 'Deletion still separately requires successful final GPU verification.'}
    (OUT / 'predelete_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    if result['status'] != 'pass':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
