import copy,json,tempfile,unittest
from pathlib import Path
import evaluate as e
class IdentityTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.p=Path(self.tmp.name);self.audits={};runs=[]
  for source,groups in [('round9',('V_temporal','F_history','F_delta')),('round10',('F_history_new',))]:
   accepted=[]
   for group in groups:
    for fold in [f'htt_leave_p{i}' for i in range(1,5)]:
     for seed in e.SEEDS:
      sg='F_history' if group=='F_history_new' else group
      q=self.p/f'{source}_{group}_{fold}_{seed}.json';q.write_text(json.dumps(dict(schema='round10_htt_fusion_summary_v1' if source=='round10' else 'round9_htt_training_summary_v1',status='complete',smoke=False,group=sg,fold=fold,seed=seed)))
      rec={'path':str(q),'sha256':e.sha(q)};item=dict(group=group,fold=fold,seed=seed,training_summary=rec,checkpoint=rec,predictions={'calibration':rec,'validation':rec});accepted.append(item)
      if group!='F_delta':runs.append({**item,'group':'F_history_old' if source=='round9' and group=='F_history' else group,'source_group':sg,'source_round':source})
   path=self.p/f'{source}_audit.json';e.js(path,dict(schema='round9_training_audit_v1' if source=='round9' else 'round10_fusion_training_audit_v1',status='pass',expected_count=len(accepted),test_role_consumed=False,accepted_runs=accepted));self.audits[source]={'path':str(path),'sha256':e.sha(path)}
  self.m=dict(schema='round10_evaluation_manifest_v1',synthetic=False,runs=runs,training_audits=self.audits,evaluation_source_sha256=e.sha(e.__file__),evaluation_protocol_sha256=e.sha(Path(e.__file__).with_name('PROTOCOL.md')))
 def test_valid(self):self.assertEqual(len(e.validate_manifest(self.m)[1]),36)
 def test_bad_grid(self):
  p=Path(self.audits['round9']['path']);a=json.loads(p.read_text());a['accepted_runs'][-1]=a['accepted_runs'][0];e.js(p,a);self.audits['round9']['sha256']=e.sha(p)
  with self.assertRaisesRegex(ValueError,'audit grid'):e.validate_manifest(self.m)
 def mutate_summary(self,key,value):
  run=self.m['runs'][-1];p=Path(run['training_summary']['path']);x=json.loads(p.read_text());x[key]=value;e.js(p,x);rec={'path':str(p),'sha256':e.sha(p)}
  for k in ('training_summary','checkpoint'):run[k]=rec
  run['predictions']={k:rec for k in ('calibration','validation')}
  ap=Path(self.audits['round10']['path']);a=json.loads(ap.read_text());item=a['accepted_runs'][-1]
  for k in ('training_summary','checkpoint','predictions'):item[k]=run[k]
  e.js(ap,a);self.audits['round10']['sha256']=e.sha(ap)
 def test_summary_group(self):
  self.mutate_summary('group','V_temporal')
  with self.assertRaisesRegex(ValueError,'source identity'):e.validate_manifest(self.m)
 def test_summary_seed(self):
  self.mutate_summary('seed',20260913)
  with self.assertRaisesRegex(ValueError,'source identity'):e.validate_manifest(self.m)
 def test_swap_predictions(self):
  old=next(r for r in self.m['runs'] if r['group']=='F_history_old');new=next(r for r in self.m['runs'] if r['group']=='F_history_new');old['predictions'],new['predictions']=new['predictions'],old['predictions']
  with self.assertRaisesRegex(ValueError,'prediction not accepted'):e.validate_manifest(self.m)
if __name__=='__main__':unittest.main()
