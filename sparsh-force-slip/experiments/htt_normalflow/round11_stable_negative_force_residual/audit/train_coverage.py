import json,csv,hashlib
from pathlib import Path
from collections import defaultdict
import torch
C=Path(__file__).resolve().parents[1];O=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round10_htt_force_supervision_adaptation')
rows=[];bins=[];sources={};folds=[]
for f in range(1,5):
 p=O/f'prepare/p{f}_s20260914/prepared.pt';sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();d=torch.load(p,map_location='cpu',weights_only=False);r=d['roles']['train'];s=r['stage'];primary=(s==0)|(s==2);N=int(primary.sum());counts={c:int((s==c).sum()) for c in [0,2]};ixs=defaultdict(list)
 for i,e in enumerate(r['episode_id']):
  if int(s[i]) in [0,2]:ixs[e].append(i)
 J=len(ixs);statics=0
 for eid,ix in sorted(ixs.items()):
  present=set(int(s[i]) for i in ix);K=len(present);statics+=0 in present;probe=r['probe'][ix[0]];positions=[int(r['t'][i]) for i in ix];last=max(positions)
  for c in sorted(present):
   ci=[i for i in ix if int(s[i])==c];n=len(ci);old=N/(2*counts[c]);new=N/(J*K*n)
   rows.append({'fold':f,'episode_id':eid,'probe':probe,'stage':c,'frames':n,'present_classes':K,'old_frame_weight':old,'new_frame_weight':new,'old_total_weight':n*old,'new_total_weight':n*new,'old_fraction_total':n*old/N,'new_fraction_total':n*new/N,'t_min':min(int(r['t'][i]) for i in ci),'t_max':max(int(r['t'][i]) for i in ci)})
   hist=[0]*4
   for i in ci:hist[min(3,int(4*(int(r['t'][i])-13)/max(last-13+1,1)))]+=1
   for b,nq in enumerate(hist):bins.append({'fold':f,'episode_id':eid,'probe':probe,'stage':c,'relative_observed_primary_position_quartile':b+1,'frames':nq,'old_weight':nq*old,'new_weight':nq*new})
 folds.append({'fold':f,'primary_frames':N,'trials':J,'trials_with_static':statics,'trials_without_static':J-statics,'static_frames':counts[0],'gross_frames':counts[2]})
out=C/'audit';out.mkdir(exist_ok=True)
for name,data in [('TRAIN_COVERAGE.csv',rows),('TRAIN_POSITION_COVERAGE.csv',bins)]:
 with (out/name).open('w',newline='') as fp:w=csv.DictWriter(fp,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
report={'status':'pass','data_role':'train only','seed':'20260914 representative input labels/endpoints common across three seeds, verified in upstream audits','sources':sources,'folds':folds,'fixed_old_weight':'N/(2*N_class)','fixed_new_weight':'N/(J*K_trial*n_trial_class)','no_strategy_selection_from_new_model_results':True,'position_semantics':'quartile of observed primary endpoint span from13 to final primary endpoint; not physical contact stage or time','limits':['Weights redistribute existing supervision, do not introduce new stable examples','No visual similarity or ground-truth label correction inferred from these counts']}
(out/'TRAIN_COVERAGE.json').write_text(json.dumps(report,indent=2))
text='# 训练前稳定负例覆盖诊断\n\n仅统计四折train，未读取本轮模型成绩。所有可靠static/gross端点保留，incipient不参与监督。旧权重N/(2N_class)给予全局两类各半总权重，新权重N/(J K_j n_jc)先使每试次总贡献相同，再平衡试次内可用类；并不保证全局两类各半。\n\n|fold|试次|有static试次|无static试次|static帧|gross帧|\n|---|---:|---:|---:|---:|---:|\n'
for q in folds:text+=f"|{q['fold']}|{q['trials']}|{q['trials_with_static']}|{q['trials_without_static']}|{q['static_frames']}|{q['gross_frames']}|\n"
text+='\n逐probe、试次、类别的帧数及权重前后见CSV。序列四分位只表示t>=13主要端点在观察区间的位置，不是接触阶段或可信时间；短static段即使重权仍不提供新的稳定外观。不能由这一统计判断未出现状态已覆盖，更不能将加权视为纠正标签。\n'
(out/'TRAIN_COVERAGE.md').write_text(text);print(json.dumps(report['folds']))
