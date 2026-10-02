#!/usr/bin/env python3
"""Raw tactile frames and GT/prediction curves for all registered worst cases."""
import argparse,csv,json
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw
from touchd_common import atomic_json,sha256
AXES=('fx','fy','fz');HORIZONS=(1,5,10);FIXED='htt/p3_sliding/0_press_13'
def read(path):
 with path.open(newline='') as f:return list(csv.DictReader(f))
def write(path,rows):
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def svg(path,title,panels):
 cols=3;pw,ph=340,190;rows=(len(panels)+cols-1)//cols;W,H=cols*380,rows*255+72;parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"><rect width="100%" height="100%" fill="white"/><text x="{W/2}" y="24" text-anchor="middle" font-size="15">{title}</text>',f'<line x1="{W/2-95}" x2="{W/2-70}" y1="49" y2="49" stroke="#111" stroke-width="2"/><text x="{W/2-64}" y="53" font-size="11">GT</text><line x1="{W/2+5}" x2="{W/2+30}" y1="49" y2="49" stroke="#d73027" stroke-width="2"/><text x="{W/2+36}" y="53" font-size="11">prediction</text>']
 for k,(label,t,y,pred) in enumerate(panels):
  x0=34+(k%cols)*380;y0=70+(k//cols)*255;lo=float(min(np.min(y),np.min(pred)));hi=float(max(np.max(y),np.max(pred)));pad=max((hi-lo)*.05,.05);lo-=pad;hi+=pad;sx=lambda q:x0+(q-t[0])/max(1,t[-1]-t[0])*pw;sy=lambda q:y0+ph-(q-lo)/(hi-lo)*ph
  py=' '.join(f'{sx(tt):.2f},{sy(v):.2f}' for tt,v in zip(t,y));pp=' '.join(f'{sx(tt):.2f},{sy(v):.2f}' for tt,v in zip(t,pred));parts += [f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="none" stroke="#333"/>',f'<polyline points="{py}" fill="none" stroke="#111" stroke-width="1.5"/>',f'<polyline points="{pp}" fill="none" stroke="#d73027" stroke-width="1.2"/>',f'<text x="{x0+pw/2}" y="{y0-7}" text-anchor="middle" font-size="11">{label}</text>']
  for q in (float(t[0]),float(t[-1])):parts += [f'<line x1="{sx(q):.2f}" x2="{sx(q):.2f}" y1="{y0+ph}" y2="{y0+ph+4}" stroke="#333"/>',f'<text x="{sx(q):.2f}" y="{y0+ph+15}" text-anchor="middle" font-size="9">{q:g}</text>']
  for q in (lo,hi):parts += [f'<line x1="{x0-4}" x2="{x0}" y1="{sy(q):.2f}" y2="{sy(q):.2f}" stroke="#333"/>',f'<text x="{x0-6}" y="{sy(q)+3:.2f}" text-anchor="end" font-size="9">{q:.2g}</text>']
  parts += [f'<text x="{x0+pw/2}" y="{y0+ph+29}" text-anchor="middle" font-size="10">frame t</text>',f'<text transform="translate({x0-28} {y0+ph/2}) rotate(-90)" text-anchor="middle" font-size="10">force (N)</text>']
 parts.append('</svg>');path.write_text(''.join(parts))
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);selected=read(a.root/'evaluation/failure_cases/SELECTED_CASES.csv');r10=a.root.parent/'round10_htt_force_supervision_adaptation';support1=json.loads((r10/'force_support/fold_p1.json').read_text());contract_path=Path(support1['provenance']['contract']);assert sha256(contract_path)==support1['provenance']['contract_sha256'];contract=json.loads(contract_path.read_text());split_path=Path(contract['split_manifest']);split=json.loads(split_path.read_text());meta={e['id']:e for e in split['episodes']};inputs={str(split_path):sha256(split_path)};source=[];plots=[]
 # Unmodified original tactile frames for the preregistered fixed slip case.
 episode=meta[FIXED];npz_path=Path(episode['path']);assert sha256(npz_path)==episode['source_files'][str(npz_path)];z=np.load(npz_path);images=z['tactile_img'];inputs[str(npz_path)]=sha256(npz_path)
 manifest=json.loads((a.root/'evaluation/EVALUATION_MANIFEST.json').read_text());run=next(x for x in manifest['runs'] if x['group']=='V' and x['fold']=='htt_leave_p4' and int(x['seed'])==20260914);ep=[x for x in read(Path(run['endpoints']['validation']['path'])) if x['episode_id']==FIXED];by_stage={int(x['stage']):int(x['t']) for x in ep};times=sorted(set([13,len(images)-1,*by_stage.values()]));raw=[]
 for t in times:
  path=a.output/f'TACTILE_RAW_t{t:03d}.png';Image.fromarray(images[t]).save(path);raw.append({'t':t,'stage':next((int(x['stage']) for x in ep if int(x['t'])==t),None),'path':str(path),'sha256':sha256(path),'source_npz':str(npz_path),'source_npz_sha256':sha256(npz_path),'array_key':'tactile_img','source_index':t})
 thumb=[]
 for r in raw:
  im=Image.open(r['path']).copy();canvas=Image.new('RGB',(224,250),'white');canvas.paste(im,(0,26));ImageDraw.Draw(canvas).text((5,5),f"t={r['t']} stage={r['stage']}",fill='black');thumb.append(canvas)
 sheet=Image.new('RGB',(224*len(thumb),250),'white');[sheet.paste(im,(i*224,0)) for i,im in enumerate(thumb)];sheet.save(a.output/'TACTILE_RAW_CONTACT_SHEET.png');write(a.output/'TACTILE_RAW_INDEX.csv',raw)
 # Force worst-case curves.
 for s in [x for x in selected if x['task']=='force']:
  route=s['group'];fold=int(s['fold']);seed=int(s['seed']);key=f'{route}_p{fold}_s{seed}';support_path=r10/f'force_support/fold_p{fold}.json';support=json.loads(support_path.read_text());sm={e['episode_id']:e for e in support['entries']};pm=json.loads((a.root/f'formal/predictions/{key}/prediction_manifest.json').read_text());pred_entry=next(e for e in pm['entries'] if e['episode_id']==s['episode_id']);target=np.load(sm[s['episode_id']]['force_native_n_path']);pred=np.load(pred_entry['prediction_path']);t=np.arange(5,len(target));panels=[]
  for ai,axis in enumerate(AXES):
   panels.append((axis,t,target[t,ai],pred[t,ai]));source.extend({'task':'force','route':route,'fold':fold,'seed':seed,'episode_id':s['episode_id'],'selection_rule':'max validation episode mean three-axis MAE; lexical tie','selection_score':s['primary_score'],'t':int(tt),'axis':axis,'horizon':'','method':'neural','target':float(target[tt,ai]),'prediction':float(pred[tt,ai])} for tt in t)
  path=a.output/f'FORCE_{key}.svg';svg(path,f'force worst case {key} {s["episode_id"]}',panels);plots.append({'task':'force','key':key,'episode_id':s['episode_id'],'path':str(path),'sha256':sha256(path)})
 # Future worst-case curves.
 for s in [x for x in selected if x['task']=='future']:
  route=s['group'];fold=int(s['fold']);seed=int(s['seed']);key=f'{route}_p{fold}_s{seed}';path=a.root/f'evaluation/future/{key}/predictions_validation.pt';d=torch.load(path,map_location='cpu',weights_only=False);episodes=np.asarray(d['episode_id'],str);mask=episodes==s['episode_id'];t=d['t'].numpy()[mask];target=d['y'].numpy()[mask];pred=d['predictions']['neural'].numpy()[mask];panels=[]
  for hi,h in enumerate(HORIZONS):
   for ai,axis in enumerate(AXES):
    panels.append((f'h{h} {axis}',t,target[:,hi,ai],pred[:,hi,ai]));source.extend({'task':'future','route':route,'fold':fold,'seed':seed,'episode_id':s['episode_id'],'selection_rule':'max validation neural future MAE over axes/horizons; lexical tie','selection_score':s['primary_score'],'t':int(tt),'axis':axis,'horizon':h,'method':'neural','target':float(yy),'prediction':float(pp)} for tt,yy,pp in zip(t,target[:,hi,ai],pred[:,hi,ai]))
  out=a.output/f'FUTURE_{key}.svg';svg(out,f'future worst case {key} {s["episode_id"]}',panels);plots.append({'task':'future','key':key,'episode_id':s['episode_id'],'path':str(out),'sha256':sha256(out)})
 write(a.output/'CASE_CURVE_SOURCE.csv',source);write(a.output/'CASE_PLOT_INDEX.csv',plots)
 result={'schema':'round16_registered_case_evidence_v1','status':'complete','fixed_slip_case':FIXED,'raw_tactile_frames':len(raw),'raw_frames_are_direct_unmodified_tactile_img_entries':True,'curve_plots':len(plots),'force_plots':sum(x['task']=='force' for x in plots),'future_plots':sum(x['task']=='future' for x in plots),'curve_source_rows':len(source),'selection_changes_training_threshold_or_method':False,'test_consumed':False,'inputs':inputs,'outputs':{n:sha256(a.output/n) for n in ('TACTILE_RAW_INDEX.csv','TACTILE_RAW_CONTACT_SHEET.png','CASE_CURVE_SOURCE.csv','CASE_PLOT_INDEX.csv')}};atomic_json(a.output/'AUDIT.json',result);print(json.dumps({'status':'complete','raw_frames':len(raw),'plots':len(plots),'source_rows':len(source)}))
if __name__=='__main__':main()
