#!/usr/bin/env python3
"""One-shot, read-only G2 status; run on zjy-4090. No polling or job changes."""
import collections
import datetime
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def process(pid):
    try:
        proc = Path('/proc') / str(int(pid))
        cmd = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        stat = (proc / 'stat').read_text().split(') ', 1)[1].split()
        return {'alive': bool(cmd) and stat[0] != 'Z', 'ppid': int(stat[1]), 'command': cmd}
    except (OSError, ValueError, TypeError):
        return {'alive': False}


def main():
    state = json.loads((HERE / 'G2_QUEUE_STATE.json').read_text())
    pid_file = HERE / 'logs/g2_queue.pid'
    pid = int(pid_file.read_text()) if pid_file.exists() else None
    running = []
    for row in state['runs']:
        if row['status'] != 'running':
            continue
        info = {k: row.get(k) for k in ('run', 'pid', 'device', 'attempts')}
        info['process'] = process(row.get('pid'))
        log = HERE / 'logs' / f"formal_{row['group']}_p{row['fold']}_s{row['seed']}.log"
        events = []
        if log.exists():
            with log.open('rb') as handle:
                handle.seek(max(0, log.stat().st_size - 32768))
                for line in handle.read().decode(errors='replace').splitlines():
                    try:
                        event = json.loads(line)
                        if event.get('event') in ('first_optimizer_step', 'epoch_complete'):
                            events.append(event)
                    except (ValueError, AttributeError):
                        pass
        info['latest_progress'] = events[-1] if events else None
        running.append(info)
    print(json.dumps({'checked_at': datetime.datetime.now().astimezone().isoformat(),
                     'queue_pid': pid, 'queue_process': process(pid),
                     'counts': dict(collections.Counter(r['status'] for r in state['runs'])),
                     'running': running, 'quarantined_devices': state.get('quarantined_devices', []),
                     'stop_dispatch': state.get('stop_dispatch'),
                     'state_updated_at': state.get('updated_at')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
