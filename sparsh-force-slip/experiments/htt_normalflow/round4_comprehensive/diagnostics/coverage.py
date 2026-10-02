#!/usr/bin/env python3
import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

root=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow')
out=root/'round4_comprehensive/diagnostics'
cache=json.loads((root/'round3_mae_slip_adaptation/cache/cache_manifest.json').read_text())
groups=defaultdict(list)
for entry in cache['entries']:
    labels=None
    for fold,role in entry['roles_by_fold'].items():
        if role not in ('train','calibration','validation'): continue
        if labels is None: labels=np.load(entry['label_path'],allow_pickle=False)
        probe=entry['episode_id'].split('/')[1].split('_')[0]
        groups[(fold,role,probe)].append((entry['episode_id'],labels))
rows=[]
for (fold,role,probe),items in sorted(groups.items()):
    rows.append(dict(fold=fold,role=role,probe=probe,trials=len(items),
        trials_with_static=sum(np.any(y==0) for _,y in items),
        trials_with_no_static=sum(not np.any(y==0) for _,y in items),
        static_frames=sum(int(sum(y==0)) for _,y in items),
        incipient_frames=sum(int(sum(y==1)) for _,y in items),
        gross_frames=sum(int(sum(y==2)) for _,y in items),
        median_static_frames=float(np.median([sum(y==0) for _,y in items]))))
with (out/'role_probe_negative_coverage.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with (out/'REPORT_ZH.md').open('a') as f:
    f.write('\n稳定负例覆盖按每折 train/calibration/validation 与 probe 统计，见 role_probe_negative_coverage.csv；统计排除每折 test 角色，复用四折开发角色联合缓存，不把这些折称为全局盲测。这里的 static 仅指官方滑移阶段，并非独立确认的稳定抓取或无接触负例。\n')
print(json.dumps({'coverage_rows':len(rows)}))
