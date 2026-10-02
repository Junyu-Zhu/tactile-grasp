import json,csv,hashlib
from pathlib import Path
import torch,numpy as np
torch.set_num_threads(4)
C=Path('/home/zjy/document/tactile-grasp/sparsh-force-slip/experiments/htt_normalflow/round11_stable_negative_force_residual');O=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round11_stable_negative_force_residual/coverage_neighbors');audit=json.loads((O/'AUDIT.json').read_text());assert audit['status']=='pass' and audit['post_training'] and audit['case_selected_from_R10'];hashes={}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
for p,h in audit['source_hashes'].items():assert sha(p)==h;hashes[p]=h
for f,h in audit['output_hashes'].items():assert sha(O/f)==h;hashes[str(O/f)]=h
rows=list(csv.DictReader((O/'nearest_endpoints.csv').open()));dist=list(csv.DictReader((O/'distance_distribution.csv').open()));trials=list(csv.DictReader((O/'per_trial_distribution.csv').open()));assert len(rows)==984
prep=next(Path(p) for p in audit['source_hashes'] if p.endswith('prepared.pt'));ck=next(Path(p) for p in audit['source_hashes'] if p.endswith('best.pth'));data=torch.load(prep,map_location='cpu',weights_only=False);norm=torch.load(ck,map_location='cpu',weights_only=False)['normalizer'];tr=data['roles']['train'];va=data['roles']['validation'];assert set(tr['leakage_group']).isdisjoint(va['leakage_group']);assert [(tr['stage']==s).sum().item() for s in (0,2)]==[1138,11274];mask=(tr['stage']==0)|(tr['stage']==2);flat=tr['x'][mask,:,:192].double().reshape(-1,192);assert torch.equal(flat.mean(0).float(),norm['mean'][:192]);assert torch.equal(flat.std(0,unbiased=False).clamp_min(1e-6).float(),norm['std'][:192]);tx=((tr['x'][:,:,:192]-norm['mean'][:192])/norm['std'][:192]).double();vx=((va['x'][:,:,:192]-norm['mean'][:192])/norm['std'][:192]).double();ti={(ep,int(t)):i for i,(ep,t) in enumerate(zip(tr['episode_id'],tr['t']))};vi={(ep,int(t)):i for i,(ep,t) in enumerate(zip(va['episode_id'],va['t']))};samples=[]
for mode in ('last_base','history'):
 for stage in (0,2):
  subset=[r for r in rows if r['mode']==mode and int(r['candidate_stage'])==stage];chosen=[subset[0],subset[len(subset)//2],subset[-1],next(r for r in subset if r['query_partition']=='historical_case')]
  candidates=torch.nonzero(tr['stage']==stage).flatten();features=tx[candidates,-1] if mode=='last_base' else tx[candidates].reshape(len(candidates),-1)
  for r in chosen:
   i=vi[(r['query_episode'],int(r['query_t']))];q=vx[i,-1] if mode=='last_base' else vx[i].reshape(-1);d=((features-q)**2).mean(1).sqrt();value,index=d.min(0);k=int(candidates[index]);assert abs(float(value)-float(r['distance_rms']))<1e-6;assert tr['episode_id'][k]==r['nearest_train_episode'] and int(tr['t'][k])==int(r['nearest_train_t']);samples.append((mode,stage,r['query_episode'],r['query_t']))
for r in dist:
 rr=[z for z in rows if z['mode']==r['mode'] and z['candidate_stage']==r['candidate_stage'] and ((z['case_high_confidence']=='True') if r['query_partition']=='case_high_confidence' else z['query_partition']==r['query_partition'])];assert len(rr)==int(r['n_endpoints']);v=[float(z['distance_rms']) for z in rr]
 for q in (0,.25,.5,.75,.9,1):assert abs(np.quantile(v,q)-float(r[f'q{int(q*100):02d}']))<1e-10
for r in trials:
 rr=[z for z in rows if z['mode']==r['mode'] and z['candidate_stage']==r['candidate_stage'] and z['query_episode']==r['query_episode']];assert len(rr)==int(r['n_endpoints']);assert abs(np.median([float(z['distance_rms']) for z in rr])-float(r['median_distance']))<1e-10
case=[r for r in rows if r['mode']=='last_base' and r['candidate_stage']=='0' and r['query_partition']=='historical_case'];other=[r for r in rows if r['mode']=='last_base' and r['candidate_stage']=='0' and r['query_partition']=='other_validation_static'];assert len(case)==49 and all(float(r['query_r10_probability'])>=.95 for r in case);assert len(other)==197 and len({r['query_episode'] for r in other})==7
for mode in ('last_base','history'):
 for s in ('0','2'):
  vals=[float(r['median_distance']) for r in trials if r['mode']==mode and r['candidate_stage']==s and r['query_partition']=='other_validation_static'];print(mode,s,'other trial medmedian',np.median(vals))
result={'status':'pass','reviewer':'r11_eval, not coverage implementation author','scope':'bounded post-training historical-case visual proxy supplement','source_output_hashes_verified':hashes,'independent_direct_difference_NN_checks':len(samples),'checks':['984 endpoint rows exact','train static1138 gross11274; train/validation leakage disjoint','normalizer recomputed exactly from train primary endpoints across history','16 sampled exact minima independently recomputed using direct squared differences, not dot-product identity','all reported distance quantiles and pertrial medians independently recomputed','historical case49 static all R10 p>=.95; comparator197 endpoints7trials','candidate density imbalance, correlated endpoints, correlated features, posthoc/single-case limitations disclosed','no new fit/test/calibration/training strategy changes'],'scientific_boundary':'Supports visual-neighborhood difference under fixed representation. Does not prove physical state coverage, label correctness, unique error cause or population generalization.'};(C/'reviews/INDEPENDENT_COVERAGE_SUPPLEMENT_REVIEW.json').write_text(json.dumps(result,indent=2));print('PASS')
