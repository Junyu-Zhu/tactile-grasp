import json,csv,hashlib
from pathlib import Path
from collections import defaultdict
import torch
C=Path(__file__).resolve().parents[1];O=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round10_htt_force_supervision_adaptation')
rows=[];folds=[];sources={}
for f in range(1,5):
 p=O/f'prepare/p{f}_s20260914/prepared.pt';sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();d=torch.load(p,map_location='cpu',weights_only=False);r=d['roles']['train'];by=defaultdict(lambda:defaultdict(list))
 for i,e in enumerate(r['episode_id']):
  c=int(r['stage'][i])
  if c in [0,2]:by[e][c].append(i)
 N=sum(len(ix) for v in by.values() for ix in v.values());J=len(by);nc={c:sum(len(v.get(c,[])) for v in by.values()) for c in [0,2]};jc={c:sum(c in v for v in by.values()) for c in [0,2]};sums=defaultdict(float)
 for eid,v in sorted(by.items()):
  for c,ix in sorted(v.items()):
   n=len(ix);ws={'original':N/(2*nc[c]),'round11':N/(J*len(v)*n),'round12':N/(2*jc[c]*n)}
   for method,w in ws.items():
    sums[(method,c)]+=w*n
    rows.append(dict(fold=f,method=method,episode_id=eid,probe=r['probe'][ix[0]],stage=c,n=n,class_trials=jc[c],class_frames=nc[c],present_trial_classes=len(v),frame_weight=w,total_weight=w*n,fraction_total=w*n/N))
 for c in [0,2]:assert abs(sums[('round12',c)]/N-.5)<1e-12
 folds.append(dict(fold=f,trials=J,static_trials=jc[0],gross_trials=jc[2],static_frames=nc[0],gross_frames=nc[2],primary_frames=N,original_static_fraction=sums[('original',0)]/N,round11_static_fraction=sums[('round11',0)]/N,round12_static_fraction=sums[('round12',0)]/N))
out=C/'audit'
with (out/'TRAIN_CONTRIBUTIONS.csv').open('w',newline='') as fp:w=csv.DictWriter(fp,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
result={'status':'pass','role':'train only','sources':sources,'folds':folds,'formulas':{'original':'N/(2N_c)','round11':'N/(J K_j n_jc)','round12':'N/(2 J_c n_jc)'},'no_evaluation_selection':True,'all_endpoints_preserved':True,'limitations':['Class-first preserves 50% total per class but changes total contribution pertrial; cannot preserve both constraints for missing-class trials without another design','No new stable states, labels or physical coverage evidence created']}
(out/'TRAIN_CONTRIBUTIONS.json').write_text(json.dumps(result,indent=2))
t='# R12训练前贡献审计\n\n只使用train，三方案保持同一端点与标签；表格比较每试次/类别的权重分配，不检查或选择模型成绩。\n\n|fold|static/gross试次|static/gross帧|原static份额|R11份额|R12份额|\n|---|---|---|---:|---:|---:|\n'
for x in folds:t+=f"|{x['fold']}|{x['static_trials']}/{x['gross_trials']}|{x['static_frames']}/{x['gross_frames']}|{x['original_static_fraction']:.2%}|{x['round11_static_fraction']:.2%}|{x['round12_static_fraction']:.2%}|\n"
t+='\nR12 w=N/(2J_c n_jc)：全局两类各50%，类内每个有该类的试次贡献N/(2J_c)。若一个试次有两类，其总贡献是两项之和；只有gross者只获得gross项，因此R12不再要求每个试次总贡献相等。原方案、R11、R12均不创造缺失的稳定外观或修正未知标签。\n'
(out/'TRAIN_CONTRIBUTIONS.md').write_text(t);print(json.dumps(folds))
