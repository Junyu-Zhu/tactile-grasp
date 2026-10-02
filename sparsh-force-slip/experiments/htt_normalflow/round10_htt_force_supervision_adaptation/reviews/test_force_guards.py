import ast,copy,json,pathlib,tempfile,types,importlib.util,random,hashlib
import numpy as np
import torch
C=pathlib.Path('/home/zjy/document/tactile-grasp/sparsh-force-slip/experiments/htt_normalflow/round10_htt_force_supervision_adaptation')
R=pathlib.Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round10_htt_force_supervision_adaptation')
p=C/'force/train.py';sp=importlib.util.spec_from_file_location('r10_force_review_train',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
base=json.loads((R/'force_support/fold_p1.json').read_text());r9=json.loads(pathlib.Path(base['provenance']['r9_prepared']).with_name('audit.json').read_text());checks={}
with tempfile.TemporaryDirectory(prefix='r10_review_') as tmp:
 tmp=pathlib.Path(tmp)
 for name in ('validation_to_fit','missing_episode','duplicate_episode','forged_group','wrong_token_path'):
  data=copy.deepcopy(base)
  if name=='validation_to_fit':next(e for e in data['entries'] if e['role']=='validation')['role']='fit'
  if name=='missing_episode':data['entries'].pop()
  if name=='duplicate_episode':data['entries'].append(copy.deepcopy(data['entries'][0]))
  if name=='forged_group':data['entries'][0]['leakage_group']='forged'
  if name=='wrong_token_path':data['entries'][0]['token_path']='/tmp/other.tokens.npy'
  path=tmp/(name+'.json');path.write_text(json.dumps(data))
  a=types.SimpleNamespace(manifest=path,fold='htt_leave_p1',seed=20260914,r5_checkpoint=pathlib.Path(r9['provenance']['force_checkpoint']['path']),source_checkpoint=pathlib.Path(r9['provenance']['source_checkpoint']['path']),smoke=True,device='cpu',output=tmp/name,workers=0,interrupt_after_epoch=None)
  try:m.run(a)
  except AssertionError:checks[name]='rejected_before_model_or_training'
  else:raise AssertionError(name+' was not rejected')
 # Execute exact export reuse guard AST, avoiding any GPU model instantiation.
 ep=C/'force/export.py';tree=ast.parse(ep.read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');branch=next(n for n in main.body if isinstance(n,ast.If) and ast.unparse(n.test)=='done.exists()')
 fn=ast.FunctionDef(name='reuse_guard',args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),body=[branch],decorator_list=[]);code=compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(ep),'exec')
 for name in ('regression_mode','wrong_entries','wrong_manifest'):
  done=tmp/'prediction_manifest.json';ck=tmp/'best.pth';manifest=tmp/'manifest.json';ck.write_text('ck');manifest.write_text('manifest')
  sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();a=types.SimpleNamespace(regression=False,manifest=manifest);conf={'fold':'htt_leave_p1','seed':20260914,'smoke':False};entries=[{'episode_id':'one'}]
  old={'force_checkpoint':{'sha256':sha(ck)},'source_sha256':sha(ep),'regression':False,**conf,'manifest_sha256':sha(manifest),'entries':[{'episode_id':'one'}]}
  if name=='regression_mode':a.regression=True
  if name=='wrong_entries':old['entries'][0]['episode_id']='other'
  if name=='wrong_manifest':old['manifest_sha256']='wrong'
  done.write_text(json.dumps(old));env={'done':done,'json':json,'sha':sha,'ck':ck,'__file__':str(ep),'a':a,'conf':conf,'entries':entries};exec(code,env)
  try:env['reuse_guard']()
  except AssertionError:checks[name]='rejected'
  else:raise AssertionError(name+' not rejected')
 gen=torch.Generator().manual_seed(11);m.seed_all(11);state=m.capture_rng(gen)
 def draws():return (random.random(),float(np.random.rand()),torch.rand(4),torch.randperm(7,generator=gen))
 left=draws();m.restore_rng(state,gen);right=draws();assert left[:2]==right[:2] and all(torch.equal(a,b) for a,b in zip(left[2:],right[2:]));checks['python_numpy_torch_sampler_rng_restore']='exact'
print(json.dumps({'status':'pass','checks':checks,'scope':'CPU-only guard/RNG tests; actual interrupted GPU training proof separately required','source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest(),str(ep):hashlib.sha256(ep.read_bytes()).hexdigest()}}))
