#!/usr/bin/env python3
import argparse,json,hashlib
from pathlib import Path
import numpy as np,torch
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train_f1 as T

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--inventory',type=Path,required=True);ap.add_argument('--core',type=Path,required=True);ap.add_argument('--support-audit',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();inv=json.loads(a.inventory.read_text())['runs'];support=json.loads(a.support_audit.read_text())['targets'];selected=json.loads((a.core/'SELECTED_CASES.json').read_text())['selected'];a.output.mkdir(parents=True,exist_ok=False);rows=[]
 for i,item in enumerate(selected):
  rec=item['record'];fold=int(rec['fold']);seed=int(rec['seed']);leak=rec['leakage_group'];rr=next(r for r in inv if r['package']=='F1' and r['group']=='K-VF' and r['fold']==fold and r['seed']==seed);out=Path(rr['output']);cm=json.loads((out/'COMMIT.json').read_text());ck=torch.load(out/cm['best']['path'],map_location='cpu',weights_only=False);d=torch.load(rr['data'],map_location='cpu',weights_only=False);r=d['roles']['validation'];mask=np.asarray(r['leakage_group'],object)==leak;eid=sorted(set(np.asarray(r['episode_id'],object)[mask]))[0];mask=np.asarray(r['episode_id'],object)==eid;n=ck['normalizer'];m=T.init_model('K-VF',seed);m.load_state_dict(ck['model']);pred=T.predict(m,T.normalize_x(r['x'],n,'K-VF'),n,'cpu').numpy()[mask,2];truth=T.delta(r).numpy()[mask,2];half=.5*pred;t=r['t'].numpy()[mask];stable=np.max(np.abs(truth),axis=1)<=.25;changing=np.max(np.abs(truth),axis=1)>=1;src=Path(support[eid]['source_path']);z=np.load(src,allow_pickle=False);imgs=z['tactile_img'];err=np.abs(pred-truth).mean(1);ids=t[np.argsort(-err)[:3]];fig=plt.figure(figsize=(12,7));gs=fig.add_gridspec(2,3)
  for ai,axis in enumerate('xyz'):
   ax=fig.add_subplot(gs[0,ai]);ax.plot(t,truth[:,ai],label='GT delta');ax.plot(t,pred[:,ai],label='K-VF delta');ax.plot(t,half[:,ai],label='F2 half');ax.plot(t,np.zeros(len(t)),label='hold');ax.scatter(t[stable],np.full(stable.sum(),ax.get_ylim()[0]),s=6,c='#66bd63',label='stable' if ai==0 else None);ax.scatter(t[changing],np.full(changing.sum(),ax.get_ylim()[0]),s=6,c='#d73027',label='changing' if ai==0 else None);ax.set_title(f'h10 {axis}');ax.set_xlabel('native frame t');ax.set_ylabel('delta N');ax.legend(fontsize=7)
  for j,k in enumerate(ids):ax=fig.add_subplot(gs[1,j]);ax.imshow(imgs[int(k)]);ax.set_title(f'largest K error t={int(k)}');ax.axis('off')
  stable_diff=float(rec['stable_delta_mae_difference']);display_label=item['rule']
  if item['rule']=='largest_F2_stable_worsening' and stable_diff<0:display_label='least stable improvement (no worsening in selected maximum)'
  fig.suptitle(f'{display_label}: {eid}; F2-K-VF stable MAE={stable_diff:+.7f} N');fig.tight_layout();name=f'force_model_failure_{i+1}.png';fig.savefig(a.output/name,dpi=180);plt.close(fig);rows.append({'rule':item['rule'],'display_label':display_label,'stable_delta_mae_difference_n':stable_diff,'interpretation':'negative means F2 improves over K-VF','episode_id':eid,'leakage_group':leak,'fold':fold,'seed':seed,'source_path':str(src),'source_sha256':sha(src),'figure':name,'model':'K-VF','horizon':10,'f2':0.5})
 (a.output/'FAILURE_FIGURES.json').write_text(json.dumps(rows,indent=2)+'\n');summary={'schema':'round22_force_failure_figures_v1','status':'complete','figures':len(rows),'models':['K-VF','F2-half','hold'],'horizon':10,'axes':['x','y','z'],'strata':['stable','changing'],'selection':'pre-registered core SELECTED_CASES unchanged','test_consumed':False};(a.output/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()
